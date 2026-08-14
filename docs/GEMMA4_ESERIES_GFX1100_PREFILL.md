# Gemma 4 E-Series gfx1100 Prefill Batch

## Result

On one W7900/gfx1100, increasing `HIPFIRE_GEMMA4_PREFILL_BATCH` from 8 to 64 substantially improves Q8 prefill while leaving decode throughput unchanged. The same policy was subsequently validated and promoted on gfx1201; see `GEMMA4_ESERIES_GFX1201_VALIDATION.md`.

| Model | Batch 8 | Batch 16 | Batch 32 | Batch 64 | B64/B8 |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 418.265 tok/s | 689.175 tok/s | 994.240 tok/s | 1,171.270 tok/s | 2.80x |
| Gemma 4 E4B Q8 | 289.890 tok/s | 483.580 tok/s | 685.805 tok/s | 861.195 tok/s | 2.97x |

The sweep used ten fixed GSM8K 8-shot prompts per configuration, greedy decoding, a 16-token output cap, and a fresh daemon for every batch size. Output text hashes matched across all four batch sizes for all ten prompts on both models.

A second gate ran the first 30 GSM8K prompts with the normal 4,096-token output cap at batch 64. Relative to the existing batch-8 full run, both models had 30/30 byte-identical outputs and unchanged correctness decisions:

| Model | Median prefill, B64 | Median decode, B64 | Accuracy | Runtime errors |
|---|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 1,143.610 tok/s | 100.670 tok/s | 80.0% | 0 |
| Gemma 4 E4B Q8 | 856.420 tok/s | 68.140 tok/s | 90.0% | 0 |

## Dispatch Boundary

The Q8 projection path already selects the wave32 `gemm_q8_0_wmma` kernel on gfx1100. The main loss at batch 8 is under-filled 16-row WMMA tiles plus repeated per-chunk fixed overhead. This change therefore selects batch 64 by default only on exact `gfx1100`. `HIPFIRE_GEMMA4_PREFILL_BATCH` remains authoritative, and all unvalidated architectures retain the existing batch-1 default.

## gfx1100 production policy

The three cross-model routes now use an architecture-aware `auto` policy. The PLE branch and activation routes passed direct logits/KV validation, short-output hash gates, and the 30-question GSM8K gate; batched embedding passed direct logits/KV validation and 120 paired short-output hashes. Batched embedding lookup, batched PLE branch projections, and fused PLE activation are enabled by default on validated gfx1100 and gfx1201 paths; gfx1101, gfx1102, gfx1151, gfx1200, and other unvalidated architectures retain their previous behavior. Each route remains independently disableable with `HIPFIRE_GEMMA4_BATCHED_EMBEDDING_PREFILL=0`, `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL=0`, or `HIPFIRE_GEMMA4_PLE_ACTIVATION_FUSED_PREFILL=0`.

The smaller PLE model-projection batching probe and Q8 projection fusion remain opt-in. The former did not compose safely with the higher-value branch route; the latter improved E2B but was neutral on E4B, so it is not a family-wide default.

The final policy was remeasured with one release binary, ten fixed GSM8K prompts, three repeats, batch 64, and 16 generated tokens. `off` explicitly disabled the three promoted routes, `auto` removed all three overrides, and `on` explicitly enabled them:

| Model | Policy | Prefill median | TTFT median | Decode median |
|---|---|---:|---:|---:|
| Gemma 4 E2B Q8 | off | 1,177.520 tok/s | 646.157 ms | 104.920 tok/s |
| Gemma 4 E2B Q8 | auto | 2,264.140 tok/s | 334.035 ms | 102.750 tok/s |
| Gemma 4 E2B Q8 | on | 2,254.410 tok/s | 335.309 ms | 103.230 tok/s |
| Gemma 4 E4B Q8 | off | 870.885 tok/s | 877.704 ms | 68.970 tok/s |
| Gemma 4 E4B Q8 | auto | 1,481.880 tok/s | 513.841 ms | 68.670 tok/s |
| Gemma 4 E4B Q8 | on | 1,472.545 tok/s | 517.351 ms | 68.670 tok/s |

Relative to explicit off, auto improves prefill by 92.28% on E2B and 70.16% on E4B while reducing TTFT by 48.30% and 41.46%, respectively. Auto and explicit on differ by at most 0.68% across their prefill/TTFT medians, confirming that the architecture default selects the intended routes. Every corresponding off/auto/on short output was byte-identical for both models. Raw artifacts are under `target/validation/gemma4-gfx11-production-policy/{off-r3,auto-r3,on-r3}`.

## gfx1201 pre-promotion baseline

Before gfx1201 promotion, the rebased policy was checked on one R9700/gfx1201 with the same E2B/E4B Q8 artifacts. The daemon used an explicit batch 64 only to exercise the shared batched-forward path, while all three promoted feature policies remained disabled. These numbers preserve the sequential-route baseline and are superseded for current policy by `GEMMA4_ESERIES_GFX1201_VALIDATION.md`.

| Model | Policy | Prefill tok/s | TTFT ms | Decode tok/s | Wall s |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | off | 1,360.675 | 575.537 | 96.97 | 0.7741 |
| Gemma 4 E2B Q8 | auto | 1,364.205 | 574.209 | 96.97 | 0.7710 |
| Gemma 4 E4B Q8 | off | 989.960 | 790.085 | 65.04 | 1.0679 |
| Gemma 4 E4B Q8 | auto | 991.860 | 787.551 | 65.04 | 1.0654 |

The pre-promotion auto policy differed from explicit off by only +0.26% prefill on E2B and +0.19% on E4B; decode medians were identical. All 30 repeated outputs per model matched between auto and off. These artifacts remain useful as the gfx1201 sequential baseline under `target/validation/gemma4-gfx11-prefill-batch/{x570-auto-20260814,x570-off-20260814}` on the validation host.

Reproduce with `scripts/bench-gemma4-gfx11-prefill-batch.sh`.

## Q8 Projection Fusion Probe

The existing gfx1100 fused Q8 WMMA kernels can stage one shared activation for QKV or gate/up projections. The Gemma path admits them only behind `HIPFIRE_GEMMA4_Q8_FUSED_PREFILL=1`; shared-KV layers remain Q-only, and the flag is ignored outside exact gfx1100.

Three repeats of the same ten-prompt batch-64 workload produced the following medians:

| Model | Fusion | Prefill tok/s | TTFT ms | Decode tok/s |
|---|---|---:|---:|---:|
| Gemma 4 E2B Q8 | off | 1,173.600 | 650.165 | 103.900 |
| Gemma 4 E2B Q8 | on | 1,182.945 | 645.754 | 103.230 |
| Gemma 4 E4B Q8 | off | 864.170 | 883.111 | 68.970 |
| Gemma 4 E4B Q8 | on | 869.055 | 875.057 | 68.235 |

Prefill improved by 0.80% on E2B and 0.57% on E4B; all 60 paired outputs were byte-identical. The gain is positive but too small to promote the route to a default. The flag remains an opt-in characterization path while the larger 64x64 four-wave WMMA dispatcher is evaluated.

Reproduce the paired runs with:

```bash
FUSED_Q8_PREFILL=0 BATCHES=64 LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
FUSED_Q8_PREFILL=1 BATCHES=64 LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
```

## Larger Q8 WMMA tile probes

The existing 64x64 four-wave Q8 WMMA kernel was tested first and rejected for Gemma 4: E2B prefill fell from 1,178.06 to 1,078.88 tok/s (-8.42%), while E4B fell from 869.87 to 805.65 tok/s (-7.38%). All 60 paired outputs were byte-identical, so this is a shape-specific performance boundary rather than a correctness failure.

The single-wave 16x64 kernel was also rejected. E2B fell from 1,178.06 to 1,154.79 tok/s (-1.98%), while E4B moved from 869.87 to 870.38 tok/s (+0.06%, effectively neutral). Its 60 paired outputs were also byte-identical. Neither route is retained in production dispatch; the existing single-wave 16x16 Q8 WMMA remains the gfx1100 default for these Gemma projection shapes.

Raw paired artifacts are preserved under `target/validation/gemma4-gfx11-prefill-4w/{off-r3,on-r3}` and `target/validation/gemma4-gfx11-prefill-x64/on-r3`.

## Batched PLE projection probe

E-series prefill originally projected each row into the packed per-layer-input buffer with a separate GEMV, re-streaming the same Q8 model-projection matrix once per row. `HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL=1` replaces those row-wise calls with one batched projection on exact gfx1100 while preserving the embedding, normalization, and fallback paths.

Paired three-repeat runs show a consistent gain that grows with the prefill batch size:

| Model | Batch | Flag off | Flag on | Prefill delta | TTFT delta |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 8 | 414.510 tok/s | 416.195 tok/s | +0.41% | -0.44% |
| Gemma 4 E4B Q8 | 8 | 283.055 tok/s | 284.610 tok/s | +0.55% | -0.53% |
| Gemma 4 E2B Q8 | 64 | 1,178.060 tok/s | 1,202.930 tok/s | +2.11% | -2.14% |
| Gemma 4 E4B Q8 | 64 | 869.870 tok/s | 885.415 tok/s | +1.79% | -1.68% |

All 120 paired generated outputs were byte-identical. The direct sequential-versus-batched correctness harness also passed at `B=1,2,8,64` for both models: every last-token and post-batch KV-check argmax matched, and the minimum observed logit cosine was `0.9999020`. Batch 1 deliberately retains the original GEMV route. The optimization remains opt-in because the batch-8 gain is small and the WMMA route is not bit-identical to the F32 GEMV reference.

Reproduce the timing runs with:

```bash
PLE_BATCHED_PREFILL=0 BATCHES="8 64" LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
PLE_BATCHED_PREFILL=1 BATCHES="8 64" LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
```

Reproduce the direct logit/KV gate with either E-series Q8 artifact:

```bash
HIP_VISIBLE_DEVICES=1 HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL=1 \
  target/release/examples/verify_batch_gemma4 \
  --model /path/to/gemma4-e2b-or-e4b-q8.hfq --bs 1,2,8,64
```

Raw timing artifacts are preserved under `target/validation/gemma4-gfx11-ple-batched/{b8-off-r3,b8-on-r3,on-r3}`; the matching batch-64 baseline is `target/validation/gemma4-gfx11-prefill-4w/off-r3`.

## Batched PLE branch projections

The larger E-series-specific gap was inside every layer's PLE branch. Prefill issued one `input_gate` GEMV and one output `projection` GEMV per row, so batch 64 re-read both Q8 matrices and launched both kernels 64 times per layer. The auto policy routes both projections through the existing batched Q8 dispatcher on exact gfx1100 when `B > 1`; setting `HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL=0` restores the row-wise path. All other architectures, dtypes, and batch 1 keep their original route.

Three-repeat paired measurements show that eliminating these per-row launches is the dominant gfx1100 E-series prefill optimization:

| Model | Batch | Flag off | Flag on | Prefill delta | TTFT delta |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 8 | 414.510 tok/s | 443.485 tok/s | +6.99% | -6.56% |
| Gemma 4 E4B Q8 | 8 | 283.055 tok/s | 296.890 tok/s | +4.89% | -4.63% |
| Gemma 4 E2B Q8 | 64 | 1,172.740 tok/s | 1,725.170 tok/s | +47.11% | -32.46% |
| Gemma 4 E4B Q8 | 64 | 864.560 tok/s | 1,212.715 tok/s | +40.27% | -28.99% |

The 16-token paired gate produced 60/60 byte-identical outputs at both batch sizes. Direct sequential-versus-batched validation passed at `B=1,2,8,64` for E2B and E4B: every last-token and post-batch KV-check argmax matched, with minimum observed logit cosine `0.9998514`.

A separate 30-question GSM8K gate used a 4,096-token output cap. The generated wording diverged on 5/30 prompts for each model, as expected from a non-bit-identical WMMA path over long greedy trajectories, but every extracted prediction and correctness decision matched the row-wise baseline. Accuracy remained 80% for E2B and 90% for E4B. Median prefill improved from 1,143.610 to 1,702.915 tok/s (+48.91%) on E2B and from 856.420 to 1,194.965 tok/s (+39.53%) on E4B; decode throughput was unchanged within normal run variance.

The smaller model-projection probe and this branch probe are intentionally not composed. Enabling both produced one short-output trajectory divergence in the characterization set; when both flags are present, the higher-value branch route takes precedence and the model projection retains its row-wise GEMV.

A scalar-order composition was also tested and rejected. The model projection was routed through the existing `gemm_q8_0_batched` kernel, whose per-batch accumulator is designed to follow the row-wise Q8 GEMV reduction order, while the branch and activation optimizations remained enabled. Direct E2B/E4B validation passed at `B=1,2,8,64`, including the post-batch KV check, but 3/30 paired E2B short outputs diverged (E4B: 0/30), so the route is not serving-level bit-exact. Batch-64 timing also regressed on both models:

| Model | Exact projection off | Exact projection on | Prefill delta | TTFT delta |
|---|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 2,279.275 tok/s | 2,242.960 tok/s | -1.59% | +1.47% |
| Gemma 4 E4B Q8 | 1,500.000 tok/s | 1,428.425 tok/s | -4.77% | +5.17% |

The scalar batched kernel keeps up to 64 independent accumulators per wave and did not repay that register/loop cost for these projection shapes. No scalar-composition dispatch change is retained. Raw artifacts are under `target/validation/gemma4-gfx11-ple-exact-compose/{off-r3,on-r3}`.

Reproduce the paired branch measurements with:

```bash
PLE_BRANCH_BATCHED_PREFILL=0 BATCHES="8 64" LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
PLE_BRANCH_BATCHED_PREFILL=1 BATCHES="8 64" LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
```

Raw artifacts are under `target/validation/gemma4-gfx11-ple-branch-batched/{branch-off-r3,branch-on-r3,b8-branch-on-r3,branch-on-full30}`.

## Fused PLE activation and strided multiply

After batching the two branch projections, every E-series layer still issued one full-batch GELU launch followed by one multiply launch per row. The gfx1100 auto policy replaces that sequence with one graph-capture-safe kernel that reads the selected layer slice directly from the packed PLE buffer. At batch 64 this removes 64 launches per layer, or 2,240 launches per E2B prefill chunk and 2,688 per E4B chunk. Setting `HIPFIRE_GEMMA4_PLE_ACTIVATION_FUSED_PREFILL=0` restores the original kernels.

Three-repeat paired batch-64 measurements used the same binary with only the activation flag changed:

| Model | Fusion off | Fusion on | Prefill delta | TTFT off | TTFT on | TTFT delta |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 1,763.610 tok/s | 2,159.330 tok/s | +22.44% | 430.979 ms | 350.028 ms | -18.78% |
| Gemma 4 E4B Q8 | 1,234.250 tok/s | 1,435.570 tok/s | +16.31% | 621.220 ms | 529.851 ms | -14.71% |

At batch 8, the same route improved E2B from 443.485 to 469.360 tok/s (+5.84%) and E4B from 296.890 to 309.505 tok/s (+4.25%). All short-run output hashes matched their branch-only baselines. The direct sequential-versus-batched logits/KV harness passed at `B=1,2,8,64` for both models, with every current and next-token argmax matching.

The 30-question, 4,096-token-cap GSM8K gate also matched the branch-only baseline exactly: both models had 30/30 identical full-output SHA256 values, extracted predictions, and correctness decisions. Accuracy remained 80% for E2B and 90% for E4B.

The gfx1100 JIT audit reports wave32, 9 VGPRs, 18 SGPRs, zero VGPR/SGPR spills, and no private segment. Its three pointer arguments are classified as read-only, read-only, and write-only. The explicit arguments occupy 40 bytes; `launch_maybe_blob` pads the recorded buffer to 48 bytes, which is the size registered in the replay resource contract.

Reproduce the paired batch-64 measurements with:

```bash
PLE_BRANCH_BATCHED_PREFILL=1 PLE_ACTIVATION_FUSED_PREFILL=0 \
  BATCHES=64 LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
PLE_BRANCH_BATCHED_PREFILL=1 PLE_ACTIVATION_FUSED_PREFILL=1 \
  BATCHES=64 LIMIT=10 REPEATS=3 \
  scripts/bench-gemma4-gfx11-prefill-batch.sh
```

Raw artifacts are under `target/validation/gemma4-gfx11-ple-activation-fused/{activation-off-r3,activation-on-r3,b8-activation-on-r3,activation-on-full30}`.

## Rejected main-FFN activation fusion

A follow-up probe fused the main FFN `gelu_tanh(gate)` and `multiply(up)` sequence into one graph-capture-safe gfx1100 kernel. The kernel compiled to wave32 with 8 VGPRs, 18 SGPRs, no spills, and no private segment, but the paired batch-64 workload regressed on both E-series models:

| Model | Fusion off | Fusion on | Prefill delta | TTFT delta |
|---|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 2,210.810 tok/s | 2,180.370 tok/s | -1.38% | +1.50% |
| Gemma 4 E4B Q8 | 1,462.455 tok/s | 1,444.625 tok/s | -1.22% | +1.55% |

All 60 paired outputs were byte-identical. Removing one launch and one intermediate read/write did not offset the fused kernel's execution cost for these shapes, so the implementation and flag were removed. Raw probe artifacts remain under `target/validation/gemma4-gfx11-ffn-activation-fused/{off-r3,on-r3}`.

## Batched main and PLE embedding lookup

The main and E-series PLE embedding paths originally issued one lookup per row and copied every main embedding row into the batch buffer separately. The gfx1100 auto policy reuses the existing HFQ4/Q8 batched embedding kernels for both tables and shares one token-ID buffer between them. Setting `HIPFIRE_GEMMA4_BATCHED_EMBEDDING_PREFILL=0` restores the row-wise route; unsupported embedding formats and all other architectures retain the original path without allocating the token-ID buffer.

Three-repeat paired measurements kept the PLE branch and activation optimizations enabled and changed only the embedding flag:

| Model | Batch | Flag off | Flag on | Prefill delta | TTFT delta |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 8 | 468.305 tok/s | 473.505 tok/s | +1.11% | -1.31% |
| Gemma 4 E4B Q8 | 8 | 309.255 tok/s | 310.195 tok/s | +0.30% | -0.31% |
| Gemma 4 E2B Q8 | 64 | 2,214.280 tok/s | 2,247.860 tok/s | +1.52% | -1.74% |
| Gemma 4 E4B Q8 | 64 | 1,461.840 tok/s | 1,469.415 tok/s | +0.52% | -0.60% |

All 120 paired short outputs were byte-identical. Direct sequential-versus-batched validation also passed at `B=1,2,8,64` for both models, including the post-batch KV check and argmax comparison. The improvement is small but consistent across both E-series models and both tested batch sizes; combined with the larger PLE wins, this is part of the gfx1100 auto policy.

Raw artifacts are under `target/validation/gemma4-gfx11-batched-embedding/{off-r3,on-r3,off-b8-r3,on-b8-r3}`.

## Rejected batched Q/K norm and RoPE fusion

The decode path's fused weighted Q/K RMSNorm, Q scaling, and RoPE kernel was extended diagnostically to a `[heads, batch]` grid and tested as a whole-block prefill replacement. Direct E2B/E4B validation passed at `B=1,2,8,64`, including current logits and the post-batch KV check, but same-binary batch-64 timing regressed on both models:

| Model | Fusion off | Fusion on | Prefill delta | TTFT delta |
|---|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 2,281.745 tok/s | 2,272.220 tok/s | -0.42% | +0.29% |
| Gemma 4 E4B Q8 | 1,506.075 tok/s | 1,481.940 tok/s | -1.60% | +1.99% |

Removing three launches per layer did not offset the fused batched kernel's execution cost, so the batched entry point, runtime wrapper, and feature flag were removed. The existing single-position decode fusion remains unchanged. Raw artifacts are under `target/validation/gemma4-gfx11-batched-qk-rope/{off-r3,on-r3}`.

## Remaining kernel profile

`verify_batch_gemma4 --profile-bs 64` profiles only the selected batched forward and aggregates the runtime's existing HIP event timers. With the PLE branch and activation optimizations enabled, the largest counted category is the existing Q8 WMMA projection path:

| Model | Q8 WMMA GEMM | RMSNorm | Residual add | Row-wise PLE projection |
|---|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 13.937 ms / 275 calls | 3.192 ms / 241 calls | 2.272 ms / 106 calls | 2.070 ms / 65 calls |
| Gemma 4 E4B Q8 | 25.083 ms / 342 calls | 4.229 ms / 301 calls | 1.388 ms / 127 calls | 2.937 ms / 65 calls |

These are sums of instrumented kernel events, not a decomposition of end-to-end wall time. They nevertheless rule out paged attention as the next batch-64 target in this workload: the counted attention kernels total about 0.23 ms for either model.

Reproduce with `--profile-bs 64 --bs 64`. The profiler deliberately rejects a profile batch that is absent from `--bs`.

## Q8 fusion after PLE optimization

The original Q8 projection-fusion probe was repeated after the PLE branch and activation changes. On the same binary, E2B prefill improved from 2,210.810 to 2,278.550 tok/s (+3.06%) and TTFT fell from 342.884 to 333.390 ms (-2.77%). E4B was effectively neutral at 1,462.455 versus 1,462.585 tok/s. All 60 paired outputs were byte-identical. This supports an E2B-specific gfx1100 policy, not a model-family-wide default. Raw artifacts are under `target/validation/gemma4-gfx11-cumulative-q8-fusion/on-r3`; the paired baseline is `target/validation/gemma4-gfx11-ffn-activation-fused/off-r3`.

## Rejected batched post-norm fusion

The decode path's existing `rmsnorm_residual_add_f32` was also evaluated as a whole-block replacement for batched prefill. The initial reuse exposed an ABI mismatch because prefill scratch tensors are flat `[B*dim]` buffers while the decode helper infers rows from tensor shape; an explicit-shape diagnostic variant fixed the illegal access and passed direct logits/KV checks at `B=1,2,8,64` for both models.

The corrected path improved E2B prefill from 2,173.290 to 2,215.560 tok/s (+1.95%), but E4B regressed from 1,436.120 to 1,430.500 tok/s (-0.39%). More importantly, 3/30 short E2B outputs diverged from the unfused path. The production change and explicit-shape helper were therefore removed. Raw artifacts remain under `target/validation/gemma4-gfx11-batched-postnorm/{off-r3-v2,on-r3-v2}`.

## Rejected F16 RMSNorm projection staging

The Q8 WMMA wrappers convert each F32 activation matrix to F16 before dispatch. A profiler-only timer now exposes this cost separately as `convert_f32_to_f16[_uncached]`. At batch 64, the cumulative uncached conversion cost was 3.413 ms over 210 calls for E2B and 2.749 ms over 252 calls for E4B.

A gfx1100-only probe changed the input and pre-FFN normalization stages feeding fused Q8 projections to write F16 directly. It eliminated 50 standalone conversion launches on E2B and 66 on E4B while leaving the existing Q8 WMMA kernels unchanged. Direct validation passed at `B=1,2,8,64` for both models, including current-token argmax and post-batch KV checks. All 60 paired short outputs were byte-identical.

The serving-level timing did not justify a production route:

| Model | F16 RMSNorm off | F16 RMSNorm on | Prefill delta | TTFT off | TTFT on | TTFT delta |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 2,377.910 tok/s | 2,398.060 tok/s | +0.85% | 318.198 ms | 315.424 ms | -0.87% |
| Gemma 4 E4B Q8 | 1,507.805 tok/s | 1,489.980 tok/s | -1.18% | 503.674 ms | 511.065 ms | +1.47% |

The E2B signal was below the 1% retention threshold and did not generalize to E4B, so the F16 RMSNorm kernel, feature flag, and Gemma dispatch changes were removed. The profiler instrumentation remains because it separates activation staging from WMMA execution without changing dispatch. Raw timing artifacts are under `target/validation/gemma4-gfx11-rmsnorm-f16/{off,on}`.

## Rejected direct residual output

The batched attention, FFN, and PLE blocks reset `x` from the saved residual and then add the normalized branch in place. A gfx1100-only probe replaced each reset-copy plus in-place-add pair with the existing graph-safe out-of-place `add_f32` kernel, writing the residual sum directly to `x`. This preserved the same F32 addition and passed direct E2B/E4B validation at `B=1,2,8,64`, including current-token and post-batch KV argmax checks.

Same-binary, three-repeat batch-64 measurements did not show a serving benefit:

| Model | Original path | Direct output | Prefill delta | TTFT off | TTFT on | TTFT delta |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q8 | 2,373.005 tok/s | 2,371.010 tok/s | -0.08% | 319.012 ms | 319.438 ms | +0.13% |
| Gemma 4 E4B Q8 | 1,503.965 tok/s | 1,487.665 tok/s | -1.08% | 506.806 ms | 513.671 ms | +1.36% |

All 60 paired output hashes were identical. Avoiding the explicit device-to-device copy did not compensate for the out-of-place add path, so no dispatch or feature flag is retained. Raw artifacts are under `target/validation/gemma4-gfx11-direct-residual-add/{off,on}`.

## Rejected direct-int8 Q8 projection backend

The existing gfx1151 dense Q8_0 x Q8_1 i8-WMMA backend was compiled and run unchanged on gfx1100 as a candidate replacement for the F16-WMMA projection path. The kernel executed correctly on gfx1100 and its Q8_1 activation approximation stayed at `rms_rel=0.0050` on the tested `K=4096` shapes. Including activation quantization, however, it was slower than the current single-wave F16-WMMA kernel throughout Gemma 4's maximum batch of 64:

| M | Batch | F16 WMMA | i8 MMQ | i8 delta |
|---:|---:|---:|---:|---:|
| 256 | 64 | 24.9 us | 67.1 us | -62.9% |
| 1,024 | 64 | 31.6 us | 73.9 us | -57.2% |
| 4,096 | 64 | 80.0 us | 98.1 us | -18.5% |

The i8 backend became faster at batches 256 and 1,024, but `forward_batch` currently admits at most 64 tokens. The gfx1100 adaptation was therefore removed rather than adding an approximate path outside its profitable range. The existing gfx1151 backend and production dispatch remain unchanged.

## Rejected FP16-shadow rocBLAS projection backend

The remaining row-wise PLE model projection was also evaluated against the repository's mature FP16-shadow plus rocBLAS path before implementing any Q8 shadow support. The comparison used the exact E2B (`M=8960,K=1536`) and E4B (`M=10752,K=2560`) model-projection shapes with F16 weights and activations. At batch 64, rocBLAS was only 1.05x faster than the hand-written F16-WMMA kernel for E2B (`289.8` versus `303.0 us`) and 4.73x slower for E4B (`582.7` versus `123.2 us`). Q8 weight expansion, activation staging, and the model-lifetime shadow allocation would add costs not included in those timings. Consequently no Q8 shadow or rocBLAS Gemma dispatch was implemented.
