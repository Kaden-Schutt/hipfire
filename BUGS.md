# Bugs To Investigate

This is a lightweight reminder list. Add a short description, or record
revision + file + line number with a one-line explanation. Do not turn entries
into full investigations here.

- Qwen3 no-output-gate FullAttention faults in fused Q/K/V MQ4 projection;
  split projection should be used until the fused kernel is shape-audited.
- Rust/Axum `hipfire serve` still lacks Bun-equivalent request cancellation:
  Bun sends daemon `{type:"abort", id}` on stream/non-stream client disconnect
  and `{type:"force_answer", id}` after the thinking watchdog, but the Rust
  daemon adapter currently owns stdin/stdout behind one mutable engine during
  generation and the daemon main loop is synchronous while generating. The
  shared protocol now has typed `abort`/`force_answer` messages, and Axum
  streaming drops the daemon when it detects a closed SSE channel after a
  daemon event. Effective mid-prefill cancellation and force-answer still need
  split write/read transport ownership plus generation-loop checkpoints.
- Qwen3.5-397B-A17B HFQM v2 paged-expert forced serial prefill can panic when
  `HIPFIRE_QWEN35_EXPERT_CACHE_MB` is too small for the per-layer routed set;
  observed with 64 MB as `patch_expert_module_ptr_table: layer=0 expert=9 not
  resident` after same-layer LRU eviction. `auto` now routes paged K_TOP=10
  MQ6 models through the grouped-MoE bucket backend instead of this forced
  serial diagnostic path.
- Qwen3.5-397B-A17B HFQM v2 paged grouped-MoE B=4 suffix replay previously
  OOMed while paging an expert module with `HIPFIRE_QWEN35_EXPERT_CACHE_MB=16`
  or 64 when scratch over-reserved rows; keep this covered by live-row scratch
  sizing tests.
- Qwen3.5-397B-A17B HFQM v2 paged grouped-MoE prefill was reserving the default
  256-row `PrefillBatchScratch` envelope for much smaller live batches; this
  reduces expert-paging headroom and should stay live-row-sized unless
  `HIPFIRE_PREFILL_MAX_BATCH` is explicitly set.
- Qwen3.5-397B-A17B HFQM v2 paged grouped-MoE B=2 suffix replay with 16 tokens
  per session previously OOMed at `HIPFIRE_QWEN35_EXPERT_CACHE_MB=16` when
  scratch over-reserved rows; keep this covered by live-row scratch sizing
  tests.
- Qwen3.5-397B-A17B HFQM v2 post-prefill AR decode previously used the old
  per-token paged expert path and could panic or hit the
  `paged Qwen35-MoE decode requires GPU top-k indexed dispatch` gate; cache128
  now reaches the indexed MQ6 decode path, but decode batching/orchestration is
  still not complete.
- Qwen3.5-397B-A17B HFQM v2 paged AR decode requires enough expert-cache
  budget to hold the K_TOP routed expert module set for a token; cache16 is too
  small for K_TOP=10 and should reject before streaming, while cache128 can run
  the indexed MQ6 decode path.
- The daemon example can leave a stale `~/.hipfire/daemon.pid` after a
  subprocess kill in smoke harnesses; the next daemon start then reports
  `FATAL: hipfire daemon already running` even though the PID is gone.
- Qwen3.5-397B-A17B HFQM v2 paged grouped-MoE smoke matrix exposed a daemon
  exit when issuing a second independent `generate_batch_prefill` request in
  the same daemon process after the first batch completed; fresh-process
  per-case smokes are still needed until repeated prefill lifecycle is audited.
- Qwen3.5-397B-A17B HFQM v2 paged grouped-MoE fresh-process prefill still fails
  for B=8 with 8 suffix tokens/session: cache16/cache64 report `hipMalloc: out
  of memory` while paging an expert module, and cache128 was SIGKILLed during
  the run. B=4 with 16 suffix tokens/session passes, so session fanout pressure
  needs a separate audit from total live-row scratch sizing.
- **[CRASH — FIXED] Qwen3.5-397B-A17B load on RDNA3.5 APU caused kernel
  deadlock requiring hard reboot** (2026-06-11). Root cause: `HfqFile::open`
  called `mmap.advise(Sequential)` + `posix_fadvise(SEQUENTIAL)`, triggering
  291 GiB of kernel readahead. The slab loader then opened a second O_DIRECT fd
  to the same inode; the readahead kworker (`kworker/6:0`) held an inode lock
  that the O_DIRECT I/Os also needed → deadlock. For 35B (26 GiB) the readahead
  completes in ~6 s before O_DIRECT starts; for 397B (291 GiB) it ran for ~70 s
  concurrently. Fixed in `crates/hipfire-runtime/src/hfq.rs`: Sequential advice
  is skipped for files > 64 GiB, and `drop_mmap()` now calls `FADV_DONTNEED` to
  cancel any in-flight readahead before the O_DIRECT path starts.

## [High] crates/rdna-compute/src/dispatch.rs is excessively large
- Category: Maintainability
- Location: crates/rdna-compute/src/dispatch.rs
- Summary: The file is ~1.67MB, acting as a massive god-file for kernel dispatching.
- Suggested fix: Split dispatch logic by architecture or kernel family into smaller files.
- Scope: Architectural
- Confidence: High

## [High] crates/hipfire-runtime/examples/daemon.rs is a massive monolith
- Category: Maintainability
- Location: crates/hipfire-runtime/examples/daemon.rs
- Summary: The file is ~16.5K lines, indicating poor module boundaries for the HTTP server and orchestration layer.
- Suggested fix: Extract routing, state management, and request lifecycle logic into separate modules under `src/`.
- Scope: Architectural
- Confidence: High

## [High] Excessive use of .unwrap() leading to potential panics
- Category: Reliability / Maintainability
- Location: Project-wide (e.g., crates/hipfire-quantize/src/main.rs, crates/hipfire-arch-deepseek4/src/forward.rs)
- Summary: The codebase heavily relies on `.unwrap()` on Results and Options, which can cause the daemon or CLI to crash abruptly on unexpected inputs.
- Suggested fix: Replace `.unwrap()` with proper error handling using `Result` and `?`, or provide descriptive `expect()` messages.
- Scope: Cross-cutting
- Confidence: High

## [Medium] Excessive global state via OnceLock and thread_local!
- Category: Architecture / Maintainability
- Location: Project-wide (e.g., crates/hipfire-arch-qwen35/src/qwen35.rs, crates/rdna-compute/src/dispatch.rs)
- Summary: Global variables and thread-locals are used extensively for caching and environment configuration, making testing difficult and hiding dependencies.
- Suggested fix: Inject configuration and state through structs/context objects instead of relying on global statics.
- Scope: Architectural
- Confidence: High

## [High] Missing unit tests for critical path logic in dispatch.rs
- Category: Testing
- Location: crates/rdna-compute/src/dispatch.rs
- Summary: A 46,000-line file that manages critical GPU dispatch logic contains only a single test (`mq_signs_128_deterministic`).
- Suggested fix: Add unit tests for routing logic, fallback choices, and error handling.
- Scope: Local (but high impact)
- Confidence: High

## [High] Unsafe block memory mapping and unchecked aliasing in llama.rs
- Category: Reliability / Security
- Location: crates/hipfire-runtime/src/llama.rs
- Summary: Usage of `unsafe` with `gpu.mq_x_rot.as_ref().unwrap().buf.alias()` combines panics and unsafe pointer aliasing.
- Suggested fix: Validate buffer initialization before attempting unsafe aliasing and provide safe abstractions for GPU memory management.
- Scope: Architectural
- Confidence: High

## Collated Findings from Gemini/Docs Review

- [Critical] Global state coupling is spreading across runtime and architecture crates:
  - `OnceLock` / `thread_local!` are used for environment-derived behavior in hot and shared code paths (`crates/hipfire-arch-deepseek4/src/forward.rs`, `crates/rdna-compute/src/dispatch.rs`, `crates/hipfire-arch-qwen35/src/qwen35.rs`, `crates/hip-bridge/src/ffi.rs`).
  - This hides explicit configuration inputs and increases hidden coupling.
  - Suggested triage: list all env-backed globals and move them behind explicit config contexts when touching module boundaries.

- [High] Potential panic hazard in `crates/hipfire-runtime/src/triattn.rs`:
  - Uses `TAP_STATE.lock().unwrap()` in hot sections.
  - A poisoned mutex now can take the process down during unrelated thread panics.
  - Suggested triage: convert to recoverable lock error handling and include lock-loss telemetry.

- [High] Unchecked `unwrap()`/`as_ref().unwrap()` patterns are still concentrated in project-critical paths:
  - `crates/hipfire-runtime/src/weight_pager.rs` (`as_ref`, `PreadH2DTransport::open`, `pop_front`)
  - `crates/hipfire-runtime/src/llama.rs` around unsafe blocks.
  - Recommended: replace with explicit `Option`/`Result` handling and actionable error messages before crash.

- [High] Architectural correctness bug candidates remain explicitly referenced in comments:
  - `crates/hipfire-arch-deepseek4/src/spec_decode.rs` and `crates/hipfire-arch-deepseek4/src/forward.rs`: chunk/ring overwrite edge-case comments.
  - `crates/hipfire-arch-dots-ocr/src/dots_ocr.rs`: lane write-size mismatch comment (16-element target vs larger writes) and decoded prompt divergence.
  - `crates/hipfire-arch-qwen35-vl/tests/channel_order.rs`: `(C,T,h,w)` vs `(T,C,h,w)` transpose path in `extract_patches`.
  - Suggested triage: validate each as still-reproducible and either close with explicit evidence or move to fixed list if already mitigated.

- [Medium] Documentation-driven bug in archive guidance:
  - `docs/CLI.md` in the legacy archive still indicates `.triattn.bin` extension as canonical.
  - `AGENTS.md` and CLI code now require/allow canonical `.triattn.hfq` while continuing to parse legacy `.triattn.bin` for compatibility.
