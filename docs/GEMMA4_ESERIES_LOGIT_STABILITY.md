# Gemma 4 E-Series Prefill Logit Stability

This note records the logit-level investigation of the Gemma 4 E-series prefill routes on gfx1100 and gfx1201. It separates full-vocabulary numerical drift from top-k ordering changes, greedy-token divergence, and eventual text divergence. All comparisons use temperature-zero autoregressive decoding with EAGLE and hipGraph disabled. Early r1 traces predate the explicit `sampled_token` field, so they support top-1 comparison but report the sampled-token metric as unavailable; newer traces record both fields.

## Route Isolation

The first isolation run used Gemma 4 E2B Q8 on gfx1201 and LongBench case 01 for 256 decode steps.

| Route enabled over baseline | Full top-k values first differ | Top-k token order first differs | Top-k membership first differs | Top-1 first differs |
|---|---:|---:|---:|---:|
| Batched embedding only | never | never | never | never |
| Fused PLE activation only | never | never | never | never |
| Batched PLE branch only | 0 | 0 | 4 | 88 |
| All three routes | 0 | 0 | 4 | 88 |

The batched embedding and fused PLE activation routes therefore reproduced the baseline trace exactly in this isolation window. The PLE branch route is the component that introduces numerical drift and later changes greedy decoding.

## Source-Level Arithmetic Difference

The row-wise route calls `weight_gemv`, which selects `gemv_q8_0_wide` for the 256-wide E-series PLE input. That kernel reads FP32 activations, accumulates four FP32 streams, combines them as `(acc0 + acc1) + (acc2 + acc3)`, and then performs the wave reduction. The batched route calls `gemm_q8_0_batched_chunked`, which dispatches to `gemm_q8_0_wmma` on both tested wave32 architectures. Its launcher converts the FP32 activation batch to FP16, and the kernel accumulates FP16 WMMA products into an architecture-specific FP32 accumulator layout. Both routes are memory-safe and produce close FP32 outputs, but they do not implement the same arithmetic contract.

## Full-Vocabulary Comparison

The following snapshots compare the row-wise baseline with the original batched PLE branch WMMA route on gfx1201 LongBench case 01. Each snapshot contains all 262,144 FP32 logits.

| Step | Exact fraction | Max abs | Mean abs | RMS | Cosine | Baseline top-1 | Candidate top-1 | Top-8 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00001144 | 0.155341 | 0.029861 | 0.035310 | 0.999999161 | 100 | 100 | 8/8 |
| 1 | 0.00021362 | 0.139654 | 0.003485 | 0.005447 | 0.999999985 | 45518 | 45518 | 8/8 |
| 32 | 0.00006485 | 0.171317 | 0.012277 | 0.016113 | 0.999999772 | 20433 | 20433 | 7/8 |
| 64 | 0.00001526 | 0.109517 | 0.017271 | 0.021708 | 0.999996687 | 506 | 506 | 8/8 |
| 87 | 0.00001526 | 0.192977 | 0.033615 | 0.041401 | 0.999995051 | 562 | 562 | 8/8 |
| 88 | 0.00002670 | 0.170994 | 0.026648 | 0.033904 | 0.999996549 | 187395 | 3927 | 8/8 |

At step 88 the two leading candidates reverse order. The baseline margin is 0.027674 and the candidate margin is 0.046499; token 187395 ranks second in the candidate and token 3927 ranks second in the baseline. The full top-8 set remains identical. This is consistent with reduction-order drift crossing a low-margin decision boundary, not with an address or layout error.

## Repeated Greedy Divergence

The same PLE branch route changes top-1 on both tested architectures and on multiple prompts.

| Architecture / case | First top-1 step | Baseline token | Candidate token | Baseline margin | Candidate margin | Relationship |
|---|---:|---:|---:|---:|---:|---|
| gfx1201 / LongBench 01 | 88 | 187395 | 3927 | 0.027674 | 0.046499 | candidates reverse top-1/top-2 |
| gfx1201 / LongBench 07 | 92 | 16476 | 6742 | 0.253986 | 0.230612 | candidates reverse top-1/top-2 |
| gfx1201 / LongBench 23 | 70 | 41392 | 7714 | 0.023525 | 0.019808 | candidates reverse top-1/top-2 |
| gfx1100 / LongBench 22 | 154 | 9949 | 529 | 0.041433 | 0.045122 | candidates reverse top-1/top-2 |

Once a different token is committed, subsequent hidden states no longer represent the same sequence and later logit distances cannot be interpreted as a kernel-level error metric. The meaningful boundary is the first committed-token divergence.

## Scalar-Batched Replacement

A scalar-batched Q8 experiment was evaluated as a possible numerically safer replacement. It was rejected.

| Architecture / case | Top-1 result | Prefill baseline | Prefill scalar-batched | Speed change |
|---|---|---:|---:|---:|
| gfx1100 / LongBench 22 | no divergence through 512 steps | 416.56 tok/s | 428.42 tok/s | +2.85% |
| gfx1201 / LongBench 01 | first divergence at step 75 | 452.18 tok/s | 456.04 tok/s | +0.85% |

On gfx1201 the scalar-batched route was numerically farther from the baseline: step-0 mean absolute error was 0.076518 with max absolute error 0.182679; at step 88 mean absolute error was 0.070736 with max absolute error 0.356346. Its accumulator structure also differs from the row-wise wide GEMV reduction, so it does not preserve the reference arithmetic contract.

## Exact Batched Resolution

The replacement kernel `gemm_q8_0_batched_wide_exact` shares each Q8_0 weight load across at most eight input rows without changing the reference arithmetic. Each row retains four independent FP32 accumulators, the same `(acc0 + acc1) + (acc2 + acc3)` combine, and the same wave32 shuffle reduction as `gemv_q8_0_wide`. It does not convert activations to FP16.

The dispatcher applies this kernel independently to each PLE projection only when its input width is within the existing wide-GEMV contract (`Q8_0` and `K <= 1536`). A wider projection continues through the original row-wise `weight_gemv` path. This distinction is required: E4B's wider input-gate projection uses the single-accumulator GEMV contract, while its smaller projection uses the four-accumulator wide contract. Applying one batched reduction shape to both matrices changed E4B's greedy trajectory.

| Architecture / model / case | Prefill baseline | Prefill exact-batched | Change | Logit trace result |
|---|---:|---:|---:|---|
| gfx1100 / E2B / LongBench 22 | 420.85 tok/s | 450.31 tok/s | +7.00% | 192/192 steps exact |
| gfx1201 / E2B / LongBench 01 | 454.32 tok/s | 487.34 tok/s | +7.27% | 128/128 steps exact |
| gfx1100 / E4B / LongBench 01 | 354.17 tok/s | 370.55 tok/s | +4.63% | 128/128 steps exact |
| gfx1201 / E4B / LongBench 01 | 398.32 tok/s | 412.48 tok/s | +3.55% | 128/128 steps exact |

“Exact” here means no divergence in recorded top-k FP32 values, token order, membership, top-1, or sampled token. An E2B step-0 full-vocabulary comparison additionally matched all 262,144 FP32 logits exactly (`max_abs=0`, `exact_fraction=1.0`). The compiled kernel uses 56 VGPRs and 46 SGPRs with no reported register spill or private segment on both gfx1100 and gfx1201.

Two wider-input batching alternatives were rejected. The existing 64-row scalar-batched kernel restored E4B correctness but was performance-neutral and compiled with 107 SGPRs and 98 SGPR spills. A purpose-built eight-row single-accumulator kernel removed the spills (20 VGPRs, 46 SGPRs) but reduced E4B prefill by 1.69%, so it was removed. Keeping the wider matrix row-wise is both the exact and faster policy in the measured case.

## Performance Context

The original PLE branch WMMA route showed meaningful single-case prefill gains: 451.78 to 510.28 tok/s on gfx1201 LongBench 01 (+12.95%) and 419.06 to 463.95 tok/s on gfx1100 LongBench 22 (+10.71%). Those gains are not suitable for a default route because the same implementation repeatedly changes deterministic greedy trajectories.

The gfx1201 route-isolation run measured 454.03 tok/s for baseline, 453.29 tok/s for embedding only, 468.17 tok/s for PLE activation only, 509.12 tok/s for the unsafe PLE branch, and 536.37 tok/s for all routes. These are single diagnostic cases rather than median performance claims; the final hard30 run is the production-policy gate.

## Production Decision

The exact replacement passed the full cross-model hard30 gate and is therefore enabled by auto policy on gfx1100 and gfx1201. It remains independently disableable through `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL=0`; other architectures remain off. Across E2B and E4B on both validated architectures, all 120 candidate predictions were byte-identical to their paired all-routes-off baselines, with no correctness regressions or gains.

| Architecture | Model | Accuracy off -> exact | Prefill off -> exact | Prefill delta | TTFT off -> exact | TTFT delta | Identical predictions |
|---|---|---|---:|---:|---:|---:|---:|
| gfx1100 W7900 | E2B Q8 | 0.4000 -> 0.4000 | 451.875 -> 529.285 tok/s | +17.13% | 56.081 -> 47.879 s | -14.63% | 30/30 |
| gfx1100 W7900 | E4B Q8 | 0.4000 -> 0.4000 | 382.485 -> 426.435 tok/s | +11.49% | 66.255 -> 59.426 s | -10.31% | 30/30 |
| gfx1201 | E2B Q8 | 0.4000 -> 0.4000 | 493.955 -> 568.280 tok/s | +15.05% | 51.303 -> 44.593 s | -13.08% | 30/30 |
| gfx1201 | E4B Q8 | 0.4667 -> 0.4667 | 429.620 -> 474.880 tok/s | +10.53% | 58.986 -> 53.364 s | -9.53% | 30/30 |

The exact-policy artifacts are under `target/validation/gemma4-longbench-prefill/gfx1100/exact-ple-hard30-w7900-r1-20260816` and, on the gfx1201 validation host, `target/validation/gemma4-longbench-prefill/gfx1201/exact-ple-hard30-gfx1201-r1-20260816`.

The final hard30 gate ran 30 paired prompts for both E2B and E4B on each architecture with an 8,192-token output cap. All 120 safe-policy predictions were byte-identical to their corresponding all-routes-off baselines, with zero correctness regressions or gains.

| Architecture | Model | Accuracy off -> safe | Prefill off -> safe | Prefill delta | TTFT off -> safe | TTFT delta | Identical predictions |
|---|---|---|---:|---:|---:|---:|---:|
| gfx1100 W7900 | E2B Q8 | 0.4000 -> 0.4000 | 451.980 -> 479.780 tok/s | +6.15% | 56.067 -> 52.819 s | -5.79% | 30/30 |
| gfx1100 W7900 | E4B Q8 | 0.4000 -> 0.4000 | 383.175 -> 405.300 tok/s | +5.77% | 66.136 -> 62.524 s | -5.46% | 30/30 |
| gfx1201 | E2B Q8 | 0.4000 -> 0.4000 | 494.255 -> 519.645 tok/s | +5.14% | 51.272 -> 48.767 s | -4.89% | 30/30 |
| gfx1201 | E4B Q8 | 0.4667 -> 0.4667 | 428.385 -> 451.550 tok/s | +5.41% | 59.156 -> 56.121 s | -5.13% | 30/30 |

The paired artifacts are under `target/validation/gemma4-longbench-prefill/gfx1100/beta-safe-default-w7900-r7-20260816` and `target/validation/gemma4-longbench-prefill/gfx1201/beta-safe-default-gfx1201-r7-20260816`. Each root contains `comparison.json`, per-mode summaries, raw JSONL records, and per-case outputs.

The diagnostic implementation consists of `HIPFIRE_GEMMA4_LOGIT_TRACE_DIR` tracing in the daemon, `scripts/diag-gemma4-logit-routes.sh`, and `scripts/compare-gemma4-logits.py`. The comparator reports value, token-order, membership, top-1, sampled-token, trace-length, and numerical-failure divergence separately; selected steps can additionally be compared from atomic full-vocabulary FP32 dumps. Versioned traces are schema-checked, while legacy unversioned traces remain readable as v0 with unavailable metrics marked explicitly.

A gfx1100 schema-v1 smoke (`logit-schema-v1-smoke-gfx1100-20260816`) compared baseline with the safe PLE-activation route for two decode steps. Both traces contained the same two steps, reported finite logits, and had no divergence in top-k values, token order, membership, top-1, or the recorded sampled token.
