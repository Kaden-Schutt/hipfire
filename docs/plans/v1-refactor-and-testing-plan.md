# Refactor, Stabilization, and Testing Plan

This document details the step-by-step refactoring strategy to eliminate the technical debt logged in `BUGS.md`, encapsulate global state, split the monolithic files, and establish a robust multi-tiered testing harness. 

---

## Part 1: Refactor & Stabilization Plan

We prioritize low-risk correctness and safety refactoring first, followed by structural state encapsulation, and finally monolithic splitting—strictly adhering to the "Stabilize First" rule from `docs/plans/stabilize-before-extraction.md`.

```
STABILIZE FIRST (Phase 1-2)  ──>  MODERNIZE & ENCAPSULATE (Phase 3)  ──>  DECOUPLE & SPLIT (Phase 4)
- Panic elimination             - Context encapsulation              - File division
- Safe memory aliasing          - Deprecate OnceLock / globals       - Crate-level extraction
```

### Phase 1: Panic Elimination & Safe Memory Aliasing (Immediate Correctness)
- **Goal:** Eradicate all unchecked `.unwrap()` calls in critical paths and replace raw unsafe memory aliases with secure wrappers.
- **Action Items:**
  1. Define a core error enum `RuntimeError` inside `crates/hipfire-runtime/src/lib.rs` representing model loading, parsing, and GPU allocation errors.
  2. Implement `From<serde_json::Error>`, `From<std::io::Error>`, and `From<hip_bridge::HipError>` for `RuntimeError`.
  3. Systematically refactor `hfq.rs`, `llama.rs`, and the arch-specific forward paths to return `Result<T, RuntimeError>`.
  4. Encapsulate unsafe aliased buffers (e.g., `gpu.mq_x_rot.as_ref().unwrap().buf.alias()`) in a safe abstraction:
     ```rust
     pub struct GpuScratchRotator {
         buf: Option<DeviceBuffer>,
     }
     impl GpuScratchRotator {
         pub fn get_alias(&self) -> Result<DeviceBuffer, RuntimeError> {
             self.buf.as_ref()
                 .map(|b| unsafe { b.alias() })
                 .ok_or(RuntimeError::UninitializedBuffer("mq_x_rot"))
         }
     }
     ```

### Phase 2: Context Encapsulation (Deprecating Global State)
- **Goal:** Eliminate global `OnceLock` and thread-local state to allow concurrent model execution and simplified testing.
- **Action Items:**
  1. Define a `RuntimeContext` struct containing:
     - `FeatureFlags`
     - `RuntimeConfig`
     - Loaded `KernelCompiler`
     - Warm `Gpu` session instance
  2. Thread `&RuntimeContext` or `Arc<RuntimeContext>` through all dispatch and forward routines, eliminating global `OnceLock` caches in `dispatch.rs`.
  3. Relocate global atomic stats (like `MMQ_CURRENT_LAYER`) to active session states or engine-level state managers.

### Phase 3: CLI & Server Modernization (Unifying under Rust)
- **Goal:** Migrate `cli/index.ts` to `clap` and `examples/daemon.rs` to an `axum` service, unifying process ownership.
- **Action Items:**
  1. Establish `crates/hipfire-cli` and `crates/hipfire-server` within the Cargo workspace.
  2. Implement an Axum server that shares an `Arc<EngineContext>` across handlers.
  3. Implement the OpenAI completions payload schemas natively in Rust (`serde` serializable structures) to eliminate the Bun-to-daemon parsing layer.
  4. Port draft auto-discovery filename regex matches to native Rust `regex` or glob matching logic.

### Phase 4: Monolithic File Extraction (Breaking the God-Files)
- **Goal:** Extract and split the codebase without mutating behavior, leveraging the newly encapsulated boundaries.
- **Action Items:**
  1. **Split `dispatch.rs`:** Group dispatches by families into `crates/rdna-compute/src/dispatch/`:
     - `gemv.rs` (Uniform & Lloyd-Max)
     - `wmma.rs` (Fused and GEMM layers)
     - `norm.rs` (RMSNorm / LayerNorm dispatches)
     - `moe.rs` (Deepseek / SwiGLU expert dispatches)
  2. **Split `qwen35.rs`:** Extract layers:
     - `moe_routing.rs` (MoE batched prefill admission)
     - `deltanet.rs` (DeltaNet linear attention updates and 1D causal convolutions)
  3. **Split `daemon.rs`:** Move handlers into `crates/hipfire-server/src/routes/` (`chat.rs`, `models.rs`, `metrics.rs`).

---

## Part 2: Outline of Missing Unit and Integration Tests

Test coverage is uneven: some runtime, scheduler, and weight-pager paths already have focused tests, but dispatch routing and mock-device coverage are still thin. We will introduce a multi-tiered testing framework: Unit (no-GPU), Integration (GPU emulator / Mock), and Hardware (live GPU).

```
                 ▲   [Hardware Gates] (coherence-gate-dflash.sh)
                 │   - Live model evaluations, τ validation
                 │
                 │   [Integration Tests] (cargo test --features gpu-mock)
                 │   - Thread/Mutex lock guards, state-cache consistency
                 │
                 │   [Unit Tests] (cargo test --lib)
                 └── - Tokenizer, config parser, layout stride calculations
```

### 1. `rdna-compute` / `dispatch.rs` Tests
- **Stride & Memory Bounds Validation:**
  - Verify layout bytes for uniform formats and Lloyd-Max structures.
  - Assert that `LLOYD_MQ3_GROUP_BYTES` (112 B) and `LLOYD_MQ4_GROUP_BYTES` (160 B) stride assumptions match model dimensions.
- **Feature Flag & Routing Rules:**
  - Mock various architecture targets (e.g., `gfx1100`, `gfx1201`) and assert that the dispatch selector correctly prioritizes paths (e.g., FP8-dot4 decode GEMV path vs fallback).
  - Verify that `FP8_WMMA_MIN_BATCH` limits are enforced.

### 2. `hipfire-runtime` Core Tests
- **Tokenizer Bounds:**
  - Add tests for edge-case prompts (empty prompts, extremely long prompts, special and control tokens like `<|endoftext|>` and `<|im_end|>`).
  - Unit-test prompt-shape normalization rules (e.g., verifying `\n{3,}` collapsing behavior).
- **Weight Pager & Cache Consistency:**
  - Unit test `WeightPager::evict_lru_until` behavior to ensure VRAM bounds are respected and that eviction triggers free resources sequentially without leaks.
  - Assert cache-invalidation state matches expected lifecycles on rapid model load/unload swaps.

### 3. `hipfire-server` API & Clustering Tests
- **Request State Cache (Prefix Caching):**
  - Verify state checkpointing compatible and incompatible cases.
  - Assert prefix-checkpoint manifest correctness during eviction or storage spilling under load.
- **Model Routing & Scheduler Tests:**
  - Unit-test scheduler priority parsing (`PriorityDecodeScheduler` / `PriorityPrefillScheduler`).
  - Test batch prefill queuing metrics under simulated concurrent loads.
- **Cluster Peer Autodiscovery:**
  - Mock UDP multicast sockets to ensure node addition, heartbeat loss, and remote request proxy routing are correctly executed.

---

## Part 3: Validation Protocol & CI/CD Pipeline

To ensure zero-regression refactoring, we will integrate tests into the CI workflow.

1. **Gate 1: Static Analysis & Local Checkouts (`no-gpu-ci.sh`)**
   - Runs workspace Rust checks, targeted no-GPU Rust units, eval-harness smoke, Python CPU tests, env/docs drift checks, shell syntax checks, and Bun tests/typecheck when Bun is installed.
   - Keep `cargo fmt --check` and `cargo clippy --all-targets --all-features` as stronger local/release gates unless they are added to `scripts/no-gpu-ci.sh`.
2. **Gate 2: Emulated GPU Pipeline**
   - Run unit/integration tests with dummy device memory structures to test state management.
3. **Gate 3: Hardware Verification Gate (`coherence-gate-dflash.sh`)**
   - Execute baseline 4-stage coherence runs after any layer, kernel, or compilation change.
   - Assert $\tau$ (accept rate) bounds and inspect output for attractor loops.
