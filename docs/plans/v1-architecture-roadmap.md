# Architectural Roadmap: hipfire v1.0

This document outlines the strategic plan to transition `hipfire` from a high-performance research prototype into a production-ready, distributed inference engine. It incorporates upcoming changes including unified Rust CLI/APIs, XDNA NPU support, embedded WebUI, API metrics/accounting, cluster autodiscovery, crate consolidations, and external dependency definitions.

---

## 1. Modularization Strategy (The "Stabilize First" Mandate)

According to `docs/plans/stabilize-before-extraction.md`, large runtime files (like `crates/hipfire-arch-qwen35/src/qwen35.rs` and `crates/rdna-compute/src/dispatch.rs`) **must not** be split until the no-GPU and hardware gates (particularly around MQ3, MQ6, and MTP admission) are stable.

### Staged Extraction Strategy
1. **Stabilization:** First land the current invariants for MQ/MTP. Ensure all coherence and speed gates pass. Do not mix behavior-preserving refactoring with new kernel or format features.
2. **Behavior-Preserving Tests:** Before splitting, ensure unit test coverage exists for the boundaries we are about to create.
3. **Decouple and Extract:** Once the codebase is stable, extract the following into dedicated submodules or crates:
   - **MoE Routing:** Move batched prefill admission and dispatch planning out of `qwen35.rs`.
   - **Speculative Verification:** Isolate the interactions between `dflash.rs` and `mtp_mirror.rs`.
   - **Memory Management:** Formalize prefill batch scratch allocation and shape invariants.
   - **Kernel Dispatch:** Split `dispatch.rs` by kernel family (e.g., `gemv.rs`, `wmma.rs`, `norm.rs`, `moe.rs`).

---

## 2. Replacing Bun with Rust (Clap + Axum)

Currently, the CLI is written in Bun (`cli/index.ts`) and interacts with `examples/daemon.rs` over HTTP. This introduces IPC overhead, packaging complexity, and architectural fragmentation. We will unify the stack entirely into Rust.

### CLI Layer (`clap.rs`)
Create a new binary crate `crates/hipfire-cli` powered by `clap`.
- **Commands:** Support `pull`, `run`, `serve`, `list`, and `sidecar-gen` subcommands.
- **Auto-Discovery:** Port the draft model auto-discovery matching rules (pairing targets with `*-dflash-*` or `*.triattn.hfq` sidecars) directly into Rust.
- **Process Lifecycle:** Run the server in-process for `hipfire serve` or as an embedded Tokio task for `hipfire run`.

### Daemon & HTTP Layer (`axum.rs`)
Replace `examples/daemon.rs` and `cli/index.ts` API logic with `crates/hipfire-server` powered by `axum`.
- **Routing:** Expose clean OpenAI-compatible endpoints `/v1/chat/completions`, `/v1/models`, and cluster-specific internal coordination routes.
- **State Management:** Avoid global `OnceLock` and `thread_local!` states. Inject context (`Tokenizer`, `Gpu`, and active `Model` engines) cleanly using Axum's `State` extractor.
- **Error Handling:** Replace synchronous `.unwrap()` calls with proper async/await error propagation using custom `Result<T, AppError>` types.

---

## 3. Workspace Crate Migration & Consolidation

To maintain a clean compile tree and minimize dependency bloat, we will consolidate standard architectures and promote runtime modules out of temporary directories.

### A. Arch Crate Consolidation
Consolidate standard autoregressive Transformer crates (`hipfire-arch-llama`, `hipfire-arch-qwen2`, `hipfire-arch-toy`, and `hipfire-arch-dots-ocr`) into a single unified crate:
- **`crates/hipfire-arch-transformers`**

Keep highly specialized/hybrid architectures isolated:
- `hipfire-arch-qwen35` (DeltaNet linear-attention & custom DFlash speculative loops)
- `hipfire-arch-deepseek4` (MLA and complex Mixture-of-Experts layout)
- `hipfire-arch-lfm2moe` (Liquid Foundation Model MoE structure)

### B. Workspace Additions
- **`crates/hipfire-server`**: An official workspace-member crate carrying Axum routing, metrics, and static asset mapping, fully migrating the server code out of `examples/daemon.rs`.
- **`crates/hipfire-cli`**: Houses Clap argument parsing, peer network request coordination, and console output progress tracking.
- **`crates/xdna-compute`**: Holds FFI bindings to the Xilinx Runtime (`libxrt`) to manage XDNA NPU-specific compilation and kernel execution.

---

## 4. Hardware Abstraction: XDNA NPU Support

To support AMD XDNA NPUs alongside the existing AMD RDNA/CDNA HIP GPU paths, we must abstract the compute layer.

- **Compute Trait:** Introduce a backend-agnostic `ComputeBackend` trait in `hipfire-runtime`.
- **New Crate:** Create `crates/xdna-compute` to interface with the XRT (Xilinx Runtime) API.
- **Dynamic Routing:** During initialization, `hipfire` will probe for both RDNA (via `hip-bridge`) and XDNA (via XRT), routing dispatch operations through the active backend.
- **Unified Quantization:** Quantized weights (MQ3/MQ4/HFP4) will be mapped appropriately to the NPU's tensor processing blocks.

---

## 5. WebUI, Metrics, and Accounting

### Embedded WebUI
- **Minimal Footprint:** Build a simple, lightweight chat UI (similar to llama.cpp's WebUI).
- **Embedded Assets:** Bundle the pre-compiled WebUI assets directly into the Rust binary using `rust-embed` or `include_dir!`.
- **Axum Router:** Serve the static files under `/ui` via Axum's `ServeDir` service.

### Metrics & Accounting API
- **Instrumentation:** Use the `metrics` and `tracing` crates to track execution data in real-time.
- **Tracks:**
  - Prefill and Decode Speed (tokens/second)
  - Speculative Accept Rate ($\tau$)
  - VRAM Consumption & KV Cache Eviction Statistics
  - Client API tokens utilized (for accounting/billing)
- **API Endpoint:** Expose a `/metrics` Prometheus-compatible endpoint.
- **Metrics Dashboard:** Embed a basic charts view within the `/ui` dashboard.

---

## 6. Distributed Inference: Autodiscovery & Clustering

To support scaling across multiple local or networked systems:

- **Autodiscovery:** Use mDNS (Multicast DNS) or custom UDP broadcast packages via `tokio::net::UdpSocket` to let active `hipfire` nodes discover each other on the same network subnet.
- **Distributed State:** Maintain a registry of cluster nodes, their available model parameters, and current VRAM/NPU memory margins.
- **Request Proxying:** When a node receives a `/v1/chat/completions` request for a model it cannot fit or doesn't have loaded, proxy the execution seamlessly to a peer instance.

---

## 7. External Dependencies (Cargo.io Integration)

The following external dependencies are selected to implement these modules securely and asynchronously:

### A. Serving Layer (`crates/hipfire-server`)
- `axum = { version = "0.7", features = ["macros"] }` (HTTP server)
- `tokio = { version = "1", features = ["full"] }` (Async runtime)
- `tower-http = { version = "0.5", features = ["cors", "fs", "compression-gzip"] }` (WebUI serving & Gzip)
- `rust-embed = { version = "8" }` (Compile-time WebUI embedding)

### B. CLI and Networking (`crates/hipfire-cli`)
- `clap = { version = "4", features = ["derive", "cargo"] }` (CLI compiler)
- `reqwest = { version = "0.12", features = ["stream", "json"] }` (Downloads & proxies)
- `indicatif = { version = "0.17" }` (CLI download bars)

### C. Metrics, Monitoring, & Discovery
- `metrics = { version = "0.22" }` & `metrics-exporter-prometheus = { version = "0.15" }`
- `tracing = { version = "0.1" }` & `tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }`
- `mdns-sd = { version = "0.11" }` (Zero-config autodiscovery)

---

## 8. Crate Dependency Graph (Post-Migration)

```
                      [crates/hipfire-cli] (Clap, Reqwest)
                               │
                               ▼
                    [crates/hipfire-server] (Axum, metrics, rust-embed)
                               │
                               ▼
                   [crates/hipfire-runtime] (Engine loop, state caches)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [crates/rdna-compute] (HIP GPUs)     [crates/xdna-compute] (XDNA NPUs)
            │                                     │
            ▼                                     ▼
     [hip-bridge]                          [libxrt (XRT)]
```

---

## 9. Execution Roadmap

```
Phase 1: CLI Unification
├── Implement crates/hipfire-cli (Clap CLI parsing)
└── Implement crates/hipfire-server (Axum server & OpenAI routes)
    └── Port draft auto-discovery rules into Rust

Phase 2: UI & Metrics
├── Build and pre-compile static WebUI (OpenAI chat compatible)
├── Embed assets with `rust-embed` and serve on `/ui` via Axum
└── Implement `metrics` tracking and Prometheus telemetry

Phase 3: Stabilization & Modularization
├── Verify and lock MQ3/MTP invariants
└── Break apart dispatch.rs & qwen35.rs along documented boundaries

Phase 4: Hardware & Distributed Clusters
├── Abstract ComputeBackend trait
├── Implement crates/xdna-compute (XDNA / XRT API support)
└── Implement UDP/mDNS discovery and cluster proxying
```
