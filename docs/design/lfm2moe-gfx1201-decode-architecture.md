# LFM2.5 350M MQ4 decode optimization on gfx1201

## Status

Approved 2026-07-18. This design supersedes the decode exclusions in the prefill campaign only for the exact target below. The prefill implementation and its admissions remain unchanged.

## Objective

Increase autoregressive decode throughput for the exact LFM2.5 350M MQ4 artifact on gfx1201 by eliminating redundant activation preparation and projection launches. Preserve numerical behavior, recurrent state, KV state, other models, other quant formats, and every other GPU architecture.

The first implementation uses LFM-local orchestration around existing validated kernels. It does not modify shared HIP sources or global dispatch.

## Exact admission

The optimized runtime route requires every condition:

- GPU reports exact `gfx1201`; generic `is_rdna4()` is insufficient.
- Architecture id is 11.
- Hidden size 1024, vocabulary 65536, 16 query heads, 8 KV heads, head dimension 64.
- Intermediate size 4608, 16 dense layers, no experts.
- RoPE theta 1,000,000, RMS epsilon 1e-5, convolution kernel size 3.
- Mixer sequence is `CCACCACCACACACAC`.
- Dense projections are MQ4G256 with 136 bytes per 256-weight group.
- Embedding and tied lm_head are Q8; retained admission requires a Q8 KV cache with `max_seq == physical_cap == 2048`.
- Every conv in-projection, attention Q/K/V projection, and dense gate/up projection has `awq_scale.is_none()`.
- Both LFM loaders recorded source-level absence of every canonical `<weight-stem>.awq_scale.weight` tensor used by those projections. A malformed or unused sidecar still rejects fusion.
- `HIPFIRE_LFM2_GFX1201_DECODE_FUSION=1`; unset, `0`, or any other value selects ordinary decode.
- The lowered path is active and graph capture is off.

The frozen runtime artifact identity is the 229,474,032-byte base HFQ with md5 `cb5284b8ad5c6f9e4ca859c0aff0bcd0`. On Linux, the trusted HFQ loader rejects overlays, copies exactly that length from its already-open file into an anonymous memfd while digesting the same stream, applies and verifies immutable seals, and reparses/rebinds every later HFQ read to the sealed snapshot before combining the opaque identity token with structural validation. Short source/tensor reads fail and non-Linux targets mint no retained provenance. A length or digest mismatch, a directory source, or any REAP overlay selects ordinary decode. MD5 identifies this frozen fixture; it is not treated as a cryptographic authorization primitive.

Sidecar provenance is LFM-local metadata populated by both the HFQ and generic `ModelSource` loaders using the quantizer's canonical naming rule. This does not load or apply dense AWQ scales and does not change ordinary decode. If any participating sidecar exists, fusion is refused and the existing decode path handles the model unchanged.

The admission predicate lives in `hipfire-arch-lfm2moe`; every miss takes ordinary decode without partial state or scratch effects. The campaign does not flip this path to default-on.

## Authoritative baseline

Source branch: `lfm-redline` at `e20a3bab2`.

Discovery identity:

- Daemon md5: `9ee43d2673866775786d8075fb5b6e76`.
- Model md5: `cb5284b8ad5c6f9e4ca859c0aff0bcd0`.
- `hipfire profile` one-shot wall observations: tg128@ctx128 577.6 tok/s; tg512@ctx128 586.6 tok/s.

These wall observations are discovery evidence, not the final reportable baseline. A reportable comparison requires fresh-process ABBA runs after rebuilding both baseline and candidate.

Rocprof is authoritative for kernel duration, launch count, and Amdahl ranking. `hipfire profile` supplies end-to-end wall throughput and compiled-kernel resource metadata. Internal timers are supplementary because LFM decode and `conv1d_gated_decode_f32` are not fully instrumented.

The authoritative trace recorded 800 complete decode-step equivalents with exact per-token launch ratios:

| Kernel | Calls | GPU time |
|---|---:|---:|
| `gemv_hfq4g256_multirow_r2` | 60/token | 26.46% |
| `rmsnorm_f32` | 45/token | 16.21% |
| `mq_rotate_x` | 92/token | 12.63% |
| `gemv_hfq4g256_residual` | 32/token | 12.06% |
| `attention_q8_0_kv` | 6/token | 11.74% |
| `gemv_q8_0_wide` | 1/token | 9.33% |
| `rope_f32` | 6/token | 4.18% |
| `conv1d_gated_decode_f32` | 10/token | 2.62% |
| `silu_mul_f32` | 16/token | 2.48% |
| `kv_cache_write_q8_0` | 12/token | 1.63% |
| `embedding_q8` | 1/token | 0.40% |

Total: 281 compute kernels per token, including `embedding_q8` and excluding rocCLR transfer/setup rows. The same trace also contains `__amd_rocclr_copyBuffer` (907 calls, 0.2508%) and `__amd_rocclr_fillBufferAligned` (12 calls, 0.0200%); they are retained in the raw CSV but are not normalized into the compute-kernel census. Rocprof reduced absolute throughput to 331.9 tok/s and its parent process hung during teardown after writing the CSV. That instrumented throughput is not used as a baseline; the completed trace statistics are used only for attribution.

## Current data flow

The default-on lowered path is:

`generate_lfm2moe` → `decode_step` → `decode_step_inner` → `decode_step_layers_and_head_lowered` → LFM-local layer handlers.

`HIPFIRE_FORWARD_LOWERED=0` retains the diagnostic hand loop. `HIPFIRE_LFM2_GRAPH` stays off; growing attention geometry is not safe to capture.

Each MQ4 linear currently rotates its input independently. Q/K/V therefore perform three identical rotations. Dense gate/up perform two identical rotations. RMSNorm is also a separate launch before each layer half.

## Optimization sequence

Each stage is independently correctness-tested, profiled, measured, reviewed, and committed. A later stage starts from the last banked stage.

### Stage A: activation-preparation fusion

Use the existing validated `fused_rmsnorm_rotate_for_mq` helper inside exact-admitted LFM handlers:

- Conv operator norm: fused RMSNorm plus rotation, then prerotated in-projection.
- Attention operator norm: one fused RMSNorm plus rotation shared by Q, K, and V.
- Dense FFN norm: one fused RMSNorm plus rotation shared by gate and up.

Shared rotation is legal only after the admission predicate has proved both runtime `awq_scale.is_none()` and source-level sidecar absence for every participating projection. A future AWQ-enabled path must either prove the sidecars byte-identical or rotate separately per projection.

Use `weight_gemv_prerotated`; never call `run_auto` on a rotated activation. A second rotation would silently undo FWHT.

Reuse the existing caller-owned `state.ffn_x_rot` tensor for operator and FFN activation preparation. Their lifetimes are sequential and do not overlap: every Q/K/V or gate/up consumer completes before the tensor is reused. Never use the global `gpu.scratch.mq_x_rot` for a shared live activation; residual projections retain that independent scratch. This adds no allocation to any LFM cohort.

Expected structural reduction: 60 launches/token. This is a launch-count target, not a throughput prediction.

### Stage B: multi-projection fusion

After Stage A is banked, use existing kernels from LFM-local callsites:

- `fused_qkv_hfq4g256` for the six attention layers.
- `fused_gate_up_hfq4g256` for all sixteen dense FFNs.

No shared kernel source or global dispatch changes. Expected additional reduction: 28 launches/token.

### Stage C: SwiGLU/down-input fusion

Use existing `weight_gemv_swiglu_residual` from the LFM dense-down handler. It combines SiLU/multiply with MQ rotation before the residual down projection. Expected additional reduction: 16 launches/token.

Stages A-C reduce the structural target from 281 to 177 kernels/token. They do not imply a 37% throughput gain because weight bandwidth remains.

### Stage D: re-profile and choose the next kernel

Run rocprof again. Only the new top Amdahl term may justify a new kernel:

- Residual GEMV: create an LFM-prefixed gfx1201 kernel and wrapper if compiler/ISA evidence supports it.
- Attention/KV: create LFM-local paired/fused routing if context-scaled attention dominates.
- lm_head: quantify GPU GEMV versus logits D2H before considering GPU sampling.

Conv is not eligible from the discovery profile because it contributes only 2.62% of GPU time. PM4/graph work remains separate until growing-sequence attention geometry has a correctness proof.

## Isolation contract

The campaign must not:

- Modify shared Qwen/global kernel dispatch.
- Modify shared HIP source bodies during Stages A-C.
- Route other LFM sizes or quant formats through the exact admission.
- Widen gfx1201 checks to gfx1200 or generic RDNA4.
- Change prefill routing, prompt caching, speculative decode, sampling, or chat framing.
- Enable LFM graph capture.

Existing shared GPU methods are reusable because only LFM-local callsites select them. If a new source becomes necessary after Stage D, it receives an LFM-prefixed symbol and gfx1201-specific source identity.

## Correctness gates

Before performance measurement for each stage:

1. A dedicated oracle runs the exact production lowered route in two fresh child processes: reference with `HIPFIRE_LFM2_DECODE_FUSION=0`, candidate with `=1`; both force graph off and lowered on. It uses the committed token vector `[1,17,42,256,1024,4096,8191,7,511,2048,63,30000]` from reset state.
2. Candidate stderr must contain a one-time machine-readable active-route marker emitted only after full admission. The reference must not contain it. This prevents the existing `decode_step_capture` hand-loop bypass from validating untouched code.
3. At every step, logits are finite, argmax is exact, cosine is at least 0.999999, maximum absolute error is at most 0.05, and KL is at most 5e-4. Mean KL across steps is at most 1e-4. These constants are committed in the failing oracle before implementation and are not loosened after candidate output is observed.
4. `state.n_tokens` equals the consumed-token count after every step.
5. After every step, each convolution tail has cosine at least 0.999999 and maximum absolute error at most 0.01. Written Q8 K/V positions, compared after dequantization, have cosine at least 0.99999 and maximum absolute error at most 0.05. Unwritten positions remain unchanged.
6. Route-planner tests cover the positive conjunction and misses for flag unset/0/other, lowered off, graph on, non-gfx1201, wrong shape/topology/dtype/group bytes, non-Q8 head/embed/KV, runtime AWQ metadata, and source-level sidecar provenance.
7. The existing 350M MQ4 prefill parity matrix still passes.
8. A committed five-prompt JSON fixture drives reference and candidate production serve chains at temperature 0 and seed 4242. The fixture md5 and both daemon md5s are recorded; candidate output must exactly match reference output on 5/5 turns and pass the harness's empty, runaway, and attractor predicates.
9. An independent reviewer finds no admission leak, double rotation, scratch alias, loader-provenance, or reset/state regression.

No per-layer hidden-state threshold is claimed: the current `decode_step_capture` forces the hand loop and cannot witness Stage A. The oracle instead covers production-route logits plus both recurrent state classes at every step.

## Performance gates

For baseline and each candidate:

- Rebuild the release daemon and record git hash, daemon md5, model md5, GPU, and ROCm version.
- Set `HIPFIRE_DAEMON_BIN` to the just-built daemon.
- Set `HIPFIRE_DPM_WARMUP_SECS=10`, Q8 KV, graph off, and exact candidate flag.
- Run fresh-process ABBA at ctx128 for tg128 and tg512.
- Synthetic `bench_decode` has a code-defined token stream and therefore no prompt md5. Real serve evidence records the committed prompt md5.
- Use rocprof to prove the intended symbol path and launch-count reduction.
- Accept only an end-to-end improvement outside the measured noise band that transfers to both decode lengths.
- A microkernel win without end-to-end improvement is rejected.
- Null results are reverted or recorded explicitly; speculative changes do not accumulate in the production branch.

## Artifact discipline

Implementation occurs in an isolated integration worktree. Each stage follows:

1. Add the failing parity/route oracle.
2. Implement one lever.
3. Build and run correctness gates.
4. Commit the candidate.
5. Run fresh-process baseline/candidate measurement and rocprof.
6. Review and either promote to `lfm-redline` or revert/document the null result.

The parent owns integration, GPU serialization, authoritative measurement, and final promotion.