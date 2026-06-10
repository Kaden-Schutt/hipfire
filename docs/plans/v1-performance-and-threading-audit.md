# Performance and Threading Audit: hipfire v1.0

This audit evaluates the execution latencies, synchronization bottlenecks, and CPU/GPU overlap limitations of both the current prototype architecture and the proposed v1.0 (Clap + Axum + Clustering) architecture. It maps out exactly where OS threads and specialized scheduling should be implemented in Rust.

---

## 1. Performance Limitations of the Current Architecture

The v0.2.0-era codebase prioritizes raw kernel speed (using custom HIP GEMVs and DFlash speculative pipelines), but suffers from several architectural host-side latency bottlenecks.

### A. The Single-Threaded TypeScript Orchestration Bottleneck
- **Location:** `cli/index.ts`
- **Issue:** The TypeScript engine runs a single-threaded event loop. When orchestrating multi-turn chat sessions, loading JSON function specifications, and running prompt pre-tokenization, Bun's main thread blocks. 
- **Impact:** Serializing and deserializing heavy payloads (such as large tool-calling schemas or batch prefill arrays) stalls HTTP message loop processing, adding **3–12ms** of host-side latency before a single token can be dispatched to the Rust daemon.

### B. Synchronous Host/Device Sampler execution (No Overlap)
- **Location:** `crates/hipfire-runtime/src/llama.rs`, `qwen35.rs`, and `examples/daemon.rs`
- **Issue:** During the autoregressive decode loop, the host thread blocks sequentially on:
  1. Launching the decode forward pass kernels on the default stream.
  2. Synchronously copying the final layer's output logits back to host memory via `hipMemcpyDtoH`.
  3. Running the token sampler (temperature, top-P, top-K) on the CPU.
  4. Feeding the sampled token ID into the BPE tokenizer for string decoding.
- **Impact:** The GPU sits completely idle during stages 2, 3, and 4. At 150 tok/s, each decode cycle is ~6.6ms. If logits copying and CPU sampling consume 1.5ms, **over 22% of the GPU's performance capacity is wasted on CPU-stalls**.

### C. Synchronous Weight Paging and MoE Expert Thrashing
- **Location:** `crates/hipfire-runtime/src/weight_pager.rs`
- **Issue:** In Mixture-of-Experts (MoE) architectures like DeepSeek-V4 or Qwen-35-MoE, expert weights are loaded dynamically via LRU. When `WeightPager::ensure_expert_module_resident` triggers, the active execution thread blocks to read weight bytes from disk and run synchronous memory copies (`hipMemcpyHtoD`) to stage them to VRAM.
- **Impact:** Stalling the main model forward pass while waiting for disk I/O and synchronous PCIe DMA transfers blocks token generation completely, leading to catastrophic latency spikes during expert transitions.

### D. Multi-GPU Serialized Kernels
- **Location:** `crates/hipfire-runtime/src/multi_gpu.rs`
- **Issue:** When running multi-GPU configurations (pipeline or tensor parallel), HIP kernel launches for GPU 0 and GPU 1 are serialized sequentially from a single host CPU thread.
- **Impact:** Launch overhead for HIP kernels is ~15-30 microseconds. Serializing dozens of layers across multiple GPUs accumulates severe launch latency overhead that stalls device-level pipelines.

---

## 2. Performance Limitations & Risks of the Proposed (v1.0) Architecture

While migrating to Clap, Axum, and XDNA NPUs improves modularity, it introduces new performance concerns.

### A. Tokio Cooperative Scheduling Starvation
- **Risk:** Axum runs on a cooperative multi-threaded Tokio runtime. If heavy HTTP deserialization, JSON generation, metrics collection, and clustering heartbeats saturate the Tokio thread pool, Tokio's task-stealing scheduler may delay yielding execution time to the tasks performing GPU kernel launches.
- **Impact:** Microsecond-level delays in task execution can prevent prompt prefill batches from launching exactly when a stream is ready, increasing tail latency ($P_{99}$).

### B. XDNA NPU Sync Latency
- **Risk:** XDNA IPU (NPU) processing blocks operate via deep instruction queues over the XRT (Xilinx Runtime) API. Synchronizing host memory buffers with the NPU's local scratch memory blocks requires explicit DMA transfers. If done on the primary thread, this blocks concurrent GPU calculations.

### C. Clustering Network and Proxy Overhead
- **Risk:** Routing requests between nodes using a basic HTTP reverse proxy introduces severe connection and parsing overhead on every token streaming chunk.

---

## 3. Dedicated Threading & Concurrency Plan for v1.0

To resolve these limitations, we will introduce a multi-threaded, pipelined architecture in Rust using dedicated OS-level threads, lock-free queues, and asynchronous workers.

```
                    [Axum Thread Pool (Tokio)]
                                │  (Submit request payload)
                                ▼
         ┌─────────────────────────────────────────────┐
         │ Lock-Free Request Ring Buffer (Crossbeam)   │
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
              ┌───────────────────────────────────┐
              │ Dedicated GPU Dispatch Thread     │
              │ - Pinned to dedicated CPU core   │
              │ - Monopolizes HIP driver context  │
              └─────────────────┬─────────────────┘
                                │  (Overlapped Pipeline)
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
   [GPU Execution Stream]              [Background CPU Sampler Thread]
   - Forward pass layer N              - Token Sampling & BPE Decode
   - Speculative prefill verification   - Async MoE Weight Pager prefetch
```

### A. Dedicated GPU Dispatch Thread (Non-blocking)
- **Implementation:** 
  - Do **not** run GPU forward loops directly inside cooperative Tokio tasks.
  - Spawn a dedicated OS thread (`std::thread::spawn`) pinned to a physical CPU core using the `core_affinity` crate.
  - This thread will run a continuous loop, reading generation requests from a lock-free, bounded ring buffer (`crossbeam_channel::bounded`).
  - It maintains exclusive ownership of the HIP context, keeping the GPU instruction queue saturated without cooperative task-switching overhead.

### B. Double-Buffered CPU Sampler & Tokenizer Thread
- **Implementation:**
  - Create a pipelined execution scheme that overlaps the GPU execution of step $T$ with the CPU sampling of step $T-1$.
  - Use double-buffered HIP host pinned buffers for logits.
  - **The Pipeline Cycle:**
    1. GPU executes forward pass layers for step $T$.
    2. At the final layer, GPU writes logits to host pinned buffer `H_Logits_A` asynchronously via `hipMemcpyDtoHAsync` on Stream 0.
    3. GPU immediately signals a completion event (`hipEventRecord`) and begins execution of step $T+1$ (using `H_Logits_B` and Stream 1 for its final layer).
    4. Simultaneously, a background CPU Sampler Thread waits on the Stream 0 event, samples the token from `H_Logits_A`, decodes the token string, and pushes the text to the Axum response stream.

### C. Asynchronous Multi-Threaded Weight Paging (for MoE)
- **Implementation:**
  - Refactor `WeightPager` to perform I/O asynchronously via a thread pool (`rayon` or a custom thread pool utilizing `tokio-uring` / direct I/O).
  - While the GPU is executing attention layers $1..N$, background worker threads prefetch the required expert weights for layer $N+1$ from disk into pinned host staging buffers (`AlignedHostBuffer`).
  - The weights are then transferred to VRAM via `hipMemcpyAsync` on a dedicated memory stream, fully overlapping disk/PCIe transfer latency with active attention calculations.

### D. Dedicated Thread-per-GPU in Multi-GPU Crate
- **Implementation:**
  - In `crates/hipfire-runtime/src/multi_gpu.rs`, spawn a dedicated controller thread for **each** active GPU in the system.
  - These threads coordinate using lightweight lock-free barrier synchronization primitives (`parking_lot` or `crossbeam-utils` barriers), allowing each GPU to queue and execute kernels on its local HIP stream without host-side serial launch bottlenecking.

### E. Lightweight Cluster Gossip Thread
- **Implementation:**
  - Run the mDNS cluster autodiscovery, node heartbeat, and state replication loops on a dedicated low-priority Tokio background thread inside the Axum server.
  - This keeps cluster topology updates completely isolated from the hot generation and inference pathways.
