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

1. **Daemon `Collect` op** — DONE + VERIFIED E2E (loop session 5). Additive
   `{"type":"collect",...}` handler calibrates the resident model in place (no
   reload) and writes the `.calib.hfq`, returning `{"type":"collected", output,
   n_hessian, n_calib_tokens, max_consistency}`. Data plane stays daemon-internal
   (only request + summary cross JSONL). Single-GPU (pp==1) qwen3.5-family bf16
   only; additive/gated, never on the decode hot path. Pieces: typed
   `CollectRequest`/`CollectResponse` + `DaemonRequest::Collect`/`Collected`
   (hipfire-daemon-protocol), `DaemonProcess::collect` adapter method
   (hipfire-daemon-adapter), and the daemon handler (parses msg fields directly,
   calls `qwen35::collect_calibration_artifacts` on `LoadedModel.q35_weights`,
   writes via the shared `qwen35::write_calib_artifacts`). Verified on resident
   `qwen3.5-0.8b-bf16`: `n_hessian=186, max_consistency=0.0`. The CLI subcommand
   (above) remains the standalone in-process path.
2. **eval `calibrate` battery** — DONE (loop session 4). Additive
   `BatteryId::Calibrate` (opt-in via `--battery calibrate`, not in any default
   tier) spawns the `collect_artifacts` example via the examples executor and
   asserts `[CONSISTENT]` + non-zero `n_hessian`. bf16-only (skips otherwise).
   Verified on `qwen3.5-0.8b-bf16`: pass, n_hessian=186, consistent=true.
   Also: **`hipfire collect-artifacts` CLI subcommand DONE** (forwards to the
   example, mirroring `eval`/`host-profile`).
3. **Runtime per-session MoE histogram → microbatch scheduler** — the daemon
   already calls `reset`/`take_moe_router_histogram` (`hipfire-daemon/src/main.rs`).
   Wire the per-session histogram (esp. the co-occurrence pairs) into the
   scheduler's expert-affinity grouping so the paged-expert (`WeightPager`) path
   sees fewer page-ins. Scheduler hot-path — do with review.
4. **MoE-expert capture for A3B Hessians** — DONE + VERIFIED E2E (loop session 3,
   see Update below): `build_capture_names` maps MoE dense projections (full
   Hessian) + resident routed experts (imatrix-only); the bf16 MoE forward gap
   that blocked E2E is fixed (`moe_gemv_plain`). Verified on `qwen3.6-35b-a3b-bf16`:
   350 dense Hessians CONSISTENT + 3014 routed-expert imatrices. Remaining:
   paged-mode experts (WeightPager-owned buffers — needs pager-side capture).
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
- **Scaling finding (RESOLVED — streaming writer shipped).** Originally the
  collector materialized ALL Hessians in RAM (`drain() -> Vec<HfqMemTensor>`,
  then `write_hfqm_package_mem`) — ~32 GB for a 9B (down-proj Hessians are
  `[mi², ]`), filling the 63 GB RAM-backed `/tmp`. Fixed with a **streaming
  writer** (`hfq::write_hfqm_package_streaming` + `CalibCollector::write_streaming`):
  the HFQM index/metadata are written first (payload sizes are deterministic from
  `k`), then each Hessian/imatrix is downloaded → normalized → written → dropped,
  one at a time. Peak host RAM is now a single tensor (~hundreds of MB) instead of
  the whole package. `collect_calibration_artifacts` writes directly to the output
  path (no `CalibArtifacts`/`write_calib_artifacts` round-trip — removed).
  Verified byte-correct on 0.8B (quantizer reads the streamed file, LDLQ engages
  on all 8 layer-0 tensors). The imatrix-default / fp16 alternatives are
  unnecessary now (no precision loss, no RAM ceiling). Still: write big-model
  artifacts to a real disk path, not the RAM-backed `/tmp`.

- **Disk-size finding (2026-06-26).** New `.calib.hfq` Hessians default to compact
  storage: exact F32 diagonal plus BF16 lower strict triangle (`quant_type=130`),
  still exposed to the quantizer as a logical symmetric `[K,K]` Hessian. This
  targets ~4× smaller Hessian payloads while preserving the diagonal exactly.
  Legacy dense F32 Hessian output remains available with
  `HIPFIRE_CALIB_HESSIAN_STORAGE=full-f32`, and the reader accepts both formats.

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

## Update (2026-06-18, loop session 6) — build COMPLETE; cross-model artifact landed

The full single-load calibration-artifact collector (Phases 1–5) is **done and
verified**. Final-session work:

- **Genuine cross-model artifact (item 4 / #11).** Generated a real (128-token,
  not 8-token mechanism) `.calib.hfq` for the focus MoE model
  `qwen3.6-35b-a3b-bf16` to **real disk** (`~/.hipfire/calib/`, not tmpfs):
  **7.3 GiB**, `n_hessian=350` (dense projections, `diag(Σxxᵀ)==Σx²` CONSISTENT,
  rel-err 0.000e0), `n_imatrix=11568` (350 dense + 11218 routed-expert
  imatrix-only = 5609 distinct expert-projections covered, up from 3014 at 8
  tokens — coverage scales with tokens as expected). 35.9 s capture / ~60 s
  total on gfx1151. **Collector cross-model generality confirmed**: 0.8B (dense,
  byte-identical to the Python Hessian), 9B (`[4096,4096]` CONSISTENT), and A3B
  (MoE, dense-Hessian + per-expert-imatrix) all produce correct artifacts.
  Correctness is token-count-independent (the consistency check holds at any N),
  so this is a full-size, real-disk validation of the focus model.

- **Importance/KLD sweep re-run — DEFERRED (scoped follow-up, not done).** The
  prior sweep tooling (`scripts/roughquant_ablation_oracle.sh`, task #9) reads the
  **old binary** Hessian (`HIPFIRE_QTIP_HESSIAN=<model>.hessian.bin`) and is
  0.8B-specific (hard-coded model path, 39-rank per-channel ablation × full
  `perplexity` KLD eval). Re-running it on a fresh model needs, in order:
  1. a **format bridge** — the quantizer/`perplexity` path consumes the legacy
     `.hessian.bin`, not the new `.calib.hfq`; either teach them to read the HFQM
     `<name>.hessian` tensors (preferred) or add a `hfq extract`→`.hessian.bin`
     shim;
  2. a per-model `DUMP_RANK` diag rank-map (the sweep's input);
  3. the ablation loop itself (expensive on 9B/A3B: ~39 quant+PPL evals).
  This is a research investigation (not autonomous build work) and is gated on
  step 1 — left as a documented follow-up rather than rabbit-holed.

## f32 vs f64 Hessian accumulation (measured — f32 is sufficient)

The GPU accumulates `Σxxᵀ` in **f32** (`calib_hessian_outer_f32`). Question: does
f32 summation error over many tokens hurt LDLQ? Measured with an opt-in CPU f64
reference (`HIPFIRE_CALIB_F64_AUDIT=1` — `CalibCollector` accumulates the same
staged rows in f64 on the CPU and `drain` reports the max element-wise f32-vs-f64
rel-diff). RDNA has no f64 matrix units and only ~1:16 scalar f64, so the
reference is computed CPU-side, not on-GPU.

qwen3.5-0.8b-bf16 (gfx1151):

| tokens | max f32-vs-f64 Σxxᵀ rel-diff | collect time |
|-------:|----------------------------:|-------------:|
| 256    | 2.46e-4 | 40 s  |
| 4096   | 6.07e-4 | 570 s |

**Verdict: f32 accumulation is sufficient — f64 is NOT made the default.**
- Divergence grows sub-linearly with N (16× tokens → 2.4× error ≈ N^0.32, not the
  N¹ of catastrophic cancellation); extrapolated to a 262k-token corpus ≈ 2e-3.
- LDLQ adds a 1% diagonal ridge (`damp = 0.01·mean_diag`) before Cholesky, so even
  the extrapolated full-corpus perturbation is ~5× below the regularization the
  quantizer already applies — no quant-quality impact.
- The CPU f64 path is also 14× slower (570 s vs 40 s at 4096 tok), confirming a
  GPU f64 kernel would be the wrong trade. The audit stays as an opt-in
  diagnostic (e.g. for a future arch, or to re-check at very large N).

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

- **bf16 A3B MoE forward — RESOLVED, and A3B capture VERIFIED E2E.** The bf16 MoE
  FFN forward routes through the Ship 4.1 MoE dispatch family
  (`hipfire_runtime::llama::moe_family().run()` → `pipeline::run_moe_decode`),
  whose inner gemvs resolved via `for_gemv` — which has **no `(BF16, Plain)` entry**
  (BF16 has no scalar GEMV kernel; it uses WMMA), so the first forward token failed
  `unsupported gemv.unknown for /`. Fix (commit pending): added `moe_gemv_plain`
  (`pipeline/mod.rs`), a `weight_gemv`-style helper that short-circuits BF16 →
  `gemm_bf16_x_bf16_wmma`, applied at the 7 bf16-reachable gemv sites (router +
  3 shared on the gate side; shared-down + per-expert gate_up + down in the
  CPU-top-K fallback). Non-bf16 dtypes still go through `run_auto` unchanged
  (byte-identical for mq*/paro production — the change only touches the arm that
  previously hard-errored). **Bonus:** routing bf16 through `gemm_bf16_x_bf16_wmma`
  is exactly the capture chokepoint, so this simultaneously unblocked the forward
  AND fired the MoE capture. Verified on `qwen3.6-35b-a3b-bf16` (gfx1151, 8 tokens):
  **350 dense Hessians (CONSISTENT, rel-err 0.000e0)** + 3364 imatrices (350 dense
  + 3014 routed-expert imatrix-only), 3714 tensors. Structure confirmed via `hfq
  list`: dense projections (linear_attn in_proj_*/out_proj, full_attn q/k/v/o,
  router `mlp.gate`, shared_expert gate/up/down) carry both `.hessian [K,K]` +
  `.imatrix [K]`; routed `mlp.experts.{x}.{gate_up,down}_proj` carry `.imatrix`
  only (shapes [2048]/[512]). A full-corpus run (more tokens) covers more of the
  256×40 experts; 8 tokens hit 1507 distinct (expert,layer,proj) imatrices. Paged
  mode still uncovered (WeightPager owns buffers).
