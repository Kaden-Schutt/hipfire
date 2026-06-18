# Native Tier-1 calibration collector — status & roadmap

Status as of the 2026-06-18 build session. The native single-load calibration
artifact collector is built and verified; the remaining work is daemon/scheduler
integration + cross-model capture, flagged below.

## Done + verified (committed)

- **Reduction kernels** (`kernels/src/calib_reduce.hip`): `calib_sumsq_reduce_f32`
  (imatrix Σx²) + `calib_hessian_outer_f32` (Hessian Σxxᵀ, tiled). CPU-verified
  (`rdna-compute/examples/test_calib_reduce`).
- **Capture wiring** (`rdna-compute` `ActivationCapture`): `Gpu.active_capture`
  (Arc) + `capture_names` (weight-ptr→name); fired from the BF16/F16 chokepoints
  the lowered super-op path actually uses (`gemm_bf16_x_bf16_wmma_labeled`,
  `gemm_f16_batched_lmhead`). Gated on `is_none()` ⇒ non-calibration forwards are
  byte-identical. `n`/`k` passed by the gemm (the input is a shared scratch buffer
  whose shape ≠ the linear's input width). Verified: `test_capture_hook`.
- **Lib-ified collector** (`hipfire_runtime::calibration::CalibCollector`): generic
  Hessian + imatrix accumulator + `drain()` → HFQ tensors + consistency. Reusable
  by the CLI and the (future) daemon op without an arch-crate cycle.
- **Single-load CLI** (`hipfire-runtime/examples/collect_artifacts`): loads a bf16
  `.hfq` once, arms the collector, forwards over the corpus, writes a unified
  `<model>.calib.hfq`.
- **Artifacts in the `.calib.hfq`** (the unify-on-HFQ decision):
  - `<name>.hessian` [K,K] + `<name>.imatrix` [K] — verified vs the Python Hessian
    on 0.8B: 186 tensors, all K match, byte-identical size, `diag(Σxxᵀ)==Σx²`.
  - `moe_router_histogram` metadata (top1/topk per expert, per-layer, top-64
    co-occurrence = the scheduler-affinity signal) — verified on
    `qwen3.5-35b-a3b-mq4` (256 experts, top-8, all experts hit).
  - `lm_head.kldref_{idx,logit,logz}` (`--kldref`) — verified on 0.8B.
  - AWQ: derived at quant time from the captured imatrix + weights; not stored
    separately (avoids a stale-prone artifact).
- **`hfq` tool** (`hipfire-runtime` bin): `list` / `extract` / `meta-set` /
  `meta-get` — split a Hessian out, embed a jinja2 template, query provenance.
  Bundle-vs-separate is a runtime choice.

## Remaining (needs design/review — paused for the user)

1. **Daemon `Collect` op** — host `CalibCollector` on the resident model (calibrate
   without reload). Additive `DaemonRequest` variant + handler that arms the
   collector, forwards the corpus, writes the `.calib.hfq`, returns the path. The
   data plane stays daemon-internal (only a control message + file path cross the
   JSONL boundary). NOTE: touches the daemon↔eval interface flagged as possibly
   unstable — do with review. Expose ALSO as a CLI subcommand so the capability
   doesn't depend on the eval seam.
2. **eval `calibrate` battery** — additive `BatteryId` that spawns
   `collect_artifacts` via the existing examples (subprocess) executor + checks
   the consistency/tensor-count result. Low-risk but moderate surgery in the
   3.6k-line `hipfire-eval/src/lib.rs`.
3. **Runtime per-session MoE histogram → microbatch scheduler** — the daemon
   already calls `reset`/`take_moe_router_histogram` (`hipfire-daemon/src/main.rs`).
   Wire the per-session histogram (esp. the co-occurrence pairs) into the
   scheduler's expert-affinity grouping so the paged-expert (`WeightPager`) path
   sees fewer page-ins. Scheduler hot-path — do with review.
4. **MoE-expert capture for A3B Hessians** — CAPTURE SIDE DONE (loop session 3,
   see Update below): `build_capture_names` maps MoE dense projections (full
   Hessian) + resident routed experts (imatrix-only). BLOCKED on E2E by the bf16
   A3B MoE forward gap (also below). Paged-mode experts still uncovered (buffers
   owned by the WeightPager, ptrs patched per-token — needs pager-side capture).
5. **#11 cross-model** — once (4) lands, generate full `.calib.hfq` (Hessian +
   histogram) for the two `qwen3.5/3.6-35b-a3b` models and re-run the
   importance/KLD sweep to confirm generality.

## Update (2026-06-18, loop session 2)

- **Driver lib-ified** into `qwen35::collect_calibration_artifacts` (+ `CalibOpts`/
  `CalibArtifacts`); `collect_artifacts` is now a thin CLI. The daemon op + a CLI
  subcommand reuse this driver (no duplication).
- **#11 cross-model VALIDATED on `qwen3.5-9b-bf16`** (dense, same hybrid arch):
  248 hessian tensors, `diag(Σxxᵀ)==Σx²` CONSISTENT, `[4096,4096]` Hessians. The
  collector generalizes beyond 0.8B. (Run used 16 tokens — mechanism check, not a
  full-quality Hessian.)
- **Scaling finding (important):** full **fp32** Hessians for a 9B = **~16 GB**
  held in RAM (`art.tensors` is all in-memory) AND written at once — it filled the
  63 GB RAM-backed `/tmp`. For ≥9B this does NOT scale. Mitigations, in order of
  preference:
  1. **Streaming writer** — write tensors to disk incrementally instead of holding
     all of `art.tensors` in RAM (the current `write_hfqm_package_mem` takes the
     full `&[HfqMemTensor]`). Best fix; no precision loss.
  2. **imatrix-default, Hessian opt-in** — the imatrix (K-vector) is tiny;
     store Hessians only for GPTQ-target tensors or on `--hessian`.
  3. **fp16 Hessian** — halves size, but the Hessian is Cholesky-factored at quant
     time and fp16's range/precision may hurt GPTQ quality; NOT a safe default.
  Write big-model artifacts to a real disk path, not the RAM-backed `/tmp`.

- **Daemon `Collect` op — design (review-gated, NOT done autonomously):** the daemon
  (`hipfire-daemon/src/main.rs`, ~9k lines) dispatches via a custom JSON
  message-parser loop (`parse_*_request` at ~8865+), not a clean `match
  DaemonRequest`; the `DaemonRequest` enum is the *client/adapter* send-side. So a
  `Collect` op spans: (a) a `DaemonRequest::Collect(CollectRequest)` variant
  (`hipfire-daemon-protocol`), (b) an adapter send method
  (`hipfire-daemon-adapter`), (c) a server-side message parser + handler in the
  daemon loop that calls `qwen35::collect_calibration_artifacts` on the resident
  `LoadedModel.q35_weights` (+ `q35_config`, the daemon's main `Gpu`, tokenizer),
  writes the `.calib.hfq`, returns the path. Data plane stays daemon-internal
  (only request + path cross JSONL). Touches the flagged-unstable daemon
  interface ⇒ do with review. The `collect_artifacts` CLI already provides the
  standalone (in-process, daemon-free) path.

## Perf note
Per-token AR forward + per-token K×K outer-product is slow (~35 s / 256 tok on
gfx1151). A full 262k-token calibration wants **batched-prefill capture** (process
many tokens per forward, batch the outer-product) — the throughput follow-up.
Always state the box for perf numbers (gfx1151/Strix Halo here).

## Update (2026-06-18, loop session 3)

- **Buffer-and-flush capture landed (perf).** `CalibCollector` now stages
  activation rows into a `[FLUSH_BATCH=256, K]` buffer and runs a SINGLE batched
  `calib_hessian_outer_f32` / `calib_sumsq_reduce_f32` per 256 rows (the tiled
  GEMM is built for N≥16, so per-token N=1 wasted ~256×). Verified on
  `qwen3.5-0.8b-bf16` (gfx1151): 186 hessians, `diag(Σxxᵀ)==Σx²` CONSISTENT,
  **4.5–7.8 s** vs the prior ~35 s/256-tok. Commit 6084ade2.

- **MoE-expert capture (A3B) — capture side built + verified-by-construction.**
  `build_capture_names` now maps the MoE-layer dense projections (attention
  q/k/v/o or qkv/z/a/b/o, the router `mlp.gate`, and the shared expert
  gate/up/down) → **full Hessian** (same gemm chokepoint as the dense layers),
  and the resident routed experts (`mlp.experts.{x}.{gate_up,down}_proj`) →
  **imatrix-only**. The collector gained `CalibCollector::with_imatrix_only(substr)`:
  tensors whose name contains a substring (here `".experts."`) accumulate only
  Σx² (no [K,K] Hessian alloc / outer-product). Rationale: a full per-expert
  Hessian for A3B is **256 experts × 40 layers × [K,K] ≈ 196 GB** — does not fit
  on-GPU; the imatrix is a K-vector (~100 MB total) and is the importance signal
  AWQ-style quant needs. Dense path re-verified unchanged (0.8B: 186 hessians,
  CONSISTENT). Routed-expert names are emitted only when experts are **resident**
  (`HIPFIRE_QWEN35_PAGED_EXPERTS` unset = the default; paged mode owns buffers in
  the WeightPager and patches ptrs per-token, so capture-by-buf-ptr can't key them).

- **BLOCKER — bf16 A3B forward is unsupported (orthogonal to capture).** E2E
  verification on `qwen3.6-35b-a3b-bf16` (the calib SOURCE) loads fully (40
  layers, 64.56 GiB on gfx1151) but the FIRST forward token fails:
  `HIP error: unsupported gemv.unknown for /`. Root: `KernelKey::for_gemv(dtype,
  Plain)` (`hipfire-dispatch/src/types.rs:561`) has **no `(BF16, Plain)` entry**,
  so it returns `UnsupportedVariant{family:"gemv", variant:"unknown", arch:"",
  quant:""}` (→ "gemv.unknown for /"). `weight_gemv` short-circuits BF16 to
  `gemm_bf16_x_bf16_wmma` (`llama.rs:780`), but a MoE-path `gemv_family.run_auto`
  call reaches `for_gemv` with a bf16 weight WITHOUT that short-circuit. The dense
  hybrid `qwen3.5-0.8b-bf16` (same DeltaNet+FullAttn arch) calibrates fine, so the
  gap is isolated to the **MoE FFN bf16 dispatch** (never exercised before — bf16
  MoE source models are normally quantized, not inferred). **Next step (cheap, but
  do with a coherence gate on a WORKING mq4 MoE model, not blind):** get a
  debug-build backtrace / add an eprintln at the `for_gemv` `_ =>` arm to name the
  exact call site, then either short-circuit BF16 → `gemm_bf16_x_bf16_wmma` at that
  site or add a `(BF16, Plain)` gemv entry. Until then, A3B calibration must run
  against a model whose MoE FFN forward is supported (e.g. an mq* variant) — but
  capture only fires at the BF16/F16 chokepoints, so a faithful A3B Hessian needs
  the bf16 MoE forward fixed first.
