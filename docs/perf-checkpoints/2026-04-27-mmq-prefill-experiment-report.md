# 2026-04-27 MMQ Prefill Experiment Report

Scope: Qwen3.5 9B, HFQ4/MQ4 target, Q8 KV cache, Strix Halo / gfx1151,
ROCm 7.2 toolbox `llama-rocm-7.2`.

This report summarizes the prefill investigation that moved hipfire from the
initial Q8 prefill baseline to a llama.cpp-style MMQ path and then past the
1000 tok/s target for pp2048.

## Baseline and Target

The reference comparison was llama.cpp on the same Strix Halo machine:

```bash
llama-bench \
  -m /home/kotdath/omp/personal/amd-strix-halo-toolboxes/models/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 -p 2048 -n 1 -r 3 -ctk q8_0 -ctv q8_0 -fa 1
```

Measured llama.cpp prompt-processing reference:

| Prompt tokens | Prefill tok/s |
| ---: | ---: |
| 2048 | 1076.08 +/- 3.87 |
| 4096 | 1043.52 +/- 7.44 |

Hipfire started far below that when using Q8 KV:

| Stage | pp2048 prefill tok/s |
| --- | ---: |
| Original gfx1151 path | ~233 |
| gfx1151 dot2/k2x32 routing | ~354 |
| Experimental MMQ baseline | ~813-814 |
| MMQ full-tile fast path | ~1029-1065 |
| Full-tile loader/writeback specialization | ~1053-1125 |

## What Was Changed

### 1. gfx1151 prefill routing

The first safe improvement was to avoid the old WMMA prefill GEMM path for
gfx1150/gfx1151 and route QKV/QKVZA/gate-up to the dot2 path instead. The
residual GEMM selected the existing `k2x32` WMMA variant only on gfx1150/gfx1151.

Why: direct A/B tests showed the existing WMMA path was slower on Strix Halo,
while comments in the source indicated `k2x32` was not generally better on
gfx1100. This kept the change scoped to gfx1150/gfx1151.

Result: pp2048 improved from roughly `233 tok/s` to roughly `354 tok/s`.

### 2. Reusable prefill scratch

`Qwen35Scratch` gained optional reusable `PrefillBatchScratch`, enabled by:

```bash
HIPFIRE_PREFILL_REUSE_PBS=1
HIPFIRE_PREFILL_MAX_BATCH=2048
```

Why: the MMQ experiment needs stable, reusable large-batch buffers so pp2048 can
run as one batch instead of being split into smaller chunks.

Result: this did not solve the main gap by itself; it made the later MMQ path
measurable and usable from the existing prefill entry point.

### 3. Experimental HFQ4 MMQ path

An opt-in `HIPFIRE_MMQ=1` path was added for gfx1100/gfx1101/gfx1102/gfx1103
and gfx1150/gfx1151. It follows the llama.cpp MMQ structure:

- quantize activations into a Q8_1/MMQ layout;
- use 128 batch columns x 128 output rows per workgroup;
- stage activations and HFQ4 weights in LDS;
- use RDNA3/RDNA3.5 i8 WMMA for the dot product;
- apply HFQ4 scale/zero correction during accumulation.

Why: profiling showed hipfire was GEMM-bound, while llama.cpp uses MMQ for
quantized prompt processing. Experiments with Q8 activations outside MMQ were
slow, so the useful part was not "Q8 activations" alone; it was the row/batch
tiling algorithm.

Result, fresh-process pp2048:

| Run | Prefill tok/s |
| ---: | ---: |
| 1 | 813.9 |
| 2 | 814.1 |
| 3 | 812.5 |

### 4. Full-tile MMQ fast path

The current best commit is:

```text
81c588e Add MMQ full-tile prefill fast path
```

It adds a separate `gemm_hfq4g256_residual_mmq_full` kernel and dispatches to it
only when both dimensions are exact 128-wide MMQ tiles:

```text
m % 128 == 0 && batch_size % 128 == 0
```

All tail shapes remain on the checked generic MMQ kernel.

Why: a temporary global no-check kernel passed the standalone 128x128 QA case
but caused a GPU memory fault in the real Qwen3.5 forward. That proved tail
forms exist in production. The final version keeps the optimization only where
it is mathematically safe.

Result, fresh-process pp2048:

| Run | Prefill ms | Prefill tok/s |
| ---: | ---: | ---: |
| 1 | 1989.8 | 1029.2 |
| 2 | 1974.7 | 1037.1 |
| 3 | 1923.8 | 1064.6 |

### 5. Full-tile loader and writeback specialization

The follow-up experiment kept the same safety split between full-tile and
generic-tail kernels, then specialized two remaining pieces inside the full
path:

- `load_hfq4_tile<true>` removes the row clamp for full tiles;
- `gemm_hfq4g256_residual_mmq_full_{set,add}` removes the runtime set/add
  branch in full-tile writeback.

The generic kernel remains checked and is still used for non-128 batch or row
shapes.

Result, fresh-process pp2048:

| Run | Prefill tok/s |
| ---: | ---: |
| 1 | 1073.6 |
| 2 | 1095.2 |
| 3 | 1116.9 |
| 4 | 1125.0 |
| 5 | 1125.4 |
| 6 | 1077.0 |
| 7 | 1053.2 |

Additional shape checks:

| Prompt tokens | Prefill tok/s |
| ---: | ---: |
| 4096 | 997.9, 1002.9 |
| 1937 | 753.5 |

Follow-up `HIPFIRE_PREFILL_MAX_BATCH=4096` checks:

| Prompt tokens | Prefill tok/s |
| ---: | ---: |
| 2048 | 1166.3, 1143.6 |
| 4096 | 1040.9, 1016.8 |
| 8192 | 727.3 |

One larger-batch check was also run:

| Prompt tokens | `HIPFIRE_PREFILL_MAX_BATCH` | Prefill tok/s |
| ---: | ---: | ---: |
| 8192 | 8192 | 715.8 |

The pp2048 result is now at or above the measured llama.cpp reference in most
fresh-process runs. The pp4096 result reaches the 1000 tok/s target, and the
larger 4096-token prefill chunk improves the long-prompt result. This remains
an opt-in setting for now because the larger reusable prefill scratch is a
poor default for smaller VRAM AMD cards. The pp8192 check suggests 4096 is a
better current cap than 8192 on this machine.

Profile delta, serialized kernel timers:

| Kernel group | Before full-tile | After full-tile |
| --- | ---: | ---: |
| `gemm_hfq4g256_mmq_set` | 1409.1 ms | 936.4 ms |
| `gemm_hfq4g256_residual_mmq` | 722.6 ms | 445.0 ms |
| Total serialized | 2629.1 ms | 1898.0 ms |

## Negative Experiments

These were tested and not kept:

- MMQ `set`/`add` split entrypoints: regressed to about `764 tok/s`.
- `MMQ_X=64`: passed QA but regressed to about `688 tok/s`.
- `MMQ_Y=64`: broke QA with the current writeback layout.
- Full HFQ4 prepack into MMQ-ready shadow: correct but slower, about
  `654-668 tok/s`.
- Single-thread-per-row metadata fill: correct but slightly slower,
  about `804 tok/s`.
- `__launch_bounds__(..., 1)`: regressed to about `755 tok/s`.
- `HIPFIRE_ROCBLAS_ALL_ARCHS=1` FP16-shadow path on RDNA: about `239 tok/s`.
- Global no-check MMQ kernel: unsafe; caused a GPU memory fault in real forward.
- Removing the second-128-block guard from the full kernel regressed pp2048 to
  about `1072 tok/s`, so that simplification was rejected.

## Correctness and Safety Checks

Checks performed after the full-tile path:

```text
HFQ4G256 QA PASS: gpu_cpu_err=0.000366 quant_ref_err=0.000000 mmq_err=0.040617
```

After the loader/writeback specialization:

```text
HFQ4G256 QA PASS: gpu_cpu_err=0.000366 quant_ref_err=0.000000 mmq_err=0.040617
```

Prompt-processing correctness was then added to the inference QA harness as:

```bash
HIPFIRE_VERIFY_GRAPH=0 HIPFIRE_MMQ=1 \
  target/release/examples/test_inferenceQA \
  --model ~/.hipfire/models/qwen3.5-9b.mq4 \
  --qa-case prefill_batch_matches_sequential
```

The case compares `forward_prefill_batch` against token-by-token
`forward_scratch` at lengths 2, 7, 17, and 33. It checks:

- final prefill greedy token matches;
- final prefill selected-logit drift is below tolerance;
- one additional decode step from the post-prefill KV/DeltaNet state also
  matches.

Result with MMQ enabled:

```text
QA PASS prefill_batch_matches_sequential:
n=2  prefill(max=0.5607,mean=0.08337,sel=0.1198) next(max=0.2369,mean=0.03658,sel=0.0159)
n=7  prefill(max=0.2964,mean=0.04197,sel=0.0080) next(max=0.2075,mean=0.03194,sel=0.0160)
n=17 prefill(max=0.5449,mean=0.08397,sel=0.0264) next(max=0.4400,mean=0.05932,sel=0.0410)
n=33 prefill(max=0.8963,mean=0.09999,sel=0.0601) next(max=0.8364,mean=0.08171,sel=0.0146)
```

The same case also passed with:

```text
HIPFIRE_PREFILL_REUSE_PBS=1 HIPFIRE_PREFILL_MAX_BATCH=2048
```

Short generation smokes after the same change:

- AR baseline Fibonacci prompt: coherent text, `47.26 tok/s`.
- DFlash Fibonacci prompt: coherent text, `decode_tau=1.6000`,
  `decode_tok_s=13.58`.

Tail-shape smoke:

```text
--prefill 1937: completed without GPU fault, 733.0 tok/s
```

Short generation smokes:

- AR baseline Fibonacci prompt: coherent text, `43.59 tok/s`.
- DFlash Fibonacci prompt: coherent text, `decode_tau=1.4615`,
  `decode_tok_s=12.90`.

Canonical DFlash coherence gate:

```text
coherence-gate-dflash: 27B models not present, skipping (no hard error)
report: /tmp/coherence-dflash-20260427-211326.md
```

This means the mandatory gate was invoked, but it did not run its 27B cases
because the expected 27B target/draft files were not present locally.

## MR Validation Addendum

After rebasing the final diff onto upstream `603c267`, the MR branch was
validated again on Strix Halo / gfx1151 in the ROCm 7.2 toolbox.

Build and binary identity:

```text
cargo build --release --features deltanet --example test_hfq4g256QA --example test_inferenceQA --example bench_qwen35_mq4 -p engine

md5sum target/release/examples/bench_qwen35_mq4
7505b5121044d70847b826f3d084898d
```

HFQ4/MMQ channel QA:

```text
HFQ4G256 QA PASS: gpu_cpu_err=0.000366 quant_ref_err=0.000000 mmq_err=0.040617
```

Prefill and KV-cache correctness QA:

- command was run with `HIPFIRE_PREFILL_REUSE_PBS=1` and
  `HIPFIRE_PREFILL_MAX_BATCH=2048`;
- one run used default dispatch (`HIPFIRE_MMQ` unset);
- one run used the new path (`HIPFIRE_MMQ=1`);
- each run covered `q8`, `asym4`, `asym3`, and `asym2`;
- each mode compared batched prefill against sequential forward for
  `n=2,7,17,33`;
- for each comparison it checked:
  - the final prefill top token matches;
  - selected-logit drift stays below the threshold;
  - the next decode step after prefill also matches top token and selected-logit
    tolerance.

Both default and MMQ runs passed. This is a behavioral KV-cache validation: if
batched prefill wrote K/V slots incorrectly, the immediately following decode
step would diverge from the sequential reference.

Before/after prefill matrix, same branch with `HIPFIRE_MMQ` unset vs enabled,
`--prefill-runs 3`, median tok/s:

| KV mode | pp | MMQ off | MMQ on | Speedup |
| --- | ---: | ---: | ---: | ---: |
| q8 | 256 | 363.1 | 1127.6 | 3.11x |
| q8 | 512 | 352.0 | 1179.8 | 3.35x |
| q8 | 1024 | 328.9 | 1222.7 | 3.72x |
| q8 | 2048 | 318.2 | 1168.5 | 3.67x |
| asym4 | 256 | 368.6 | 1108.8 | 3.01x |
| asym4 | 512 | 360.7 | 1173.3 | 3.25x |
| asym4 | 1024 | 333.9 | 1223.0 | 3.66x |
| asym4 | 2048 | 312.3 | 1151.7 | 3.69x |
| asym3 | 256 | 361.4 | 1124.5 | 3.11x |
| asym3 | 512 | 359.8 | 1187.3 | 3.30x |
| asym3 | 1024 | 329.9 | 1259.1 | 3.82x |
| asym3 | 2048 | 314.1 | 1216.5 | 3.87x |
| asym2 | 256 | 374.0 | 1116.2 | 2.98x |
| asym2 | 512 | 356.6 | 1173.2 | 3.29x |
| asym2 | 1024 | 340.1 | 1208.5 | 3.55x |
| asym2 | 2048 | 311.4 | 1142.9 | 3.67x |

Kernel compile/hash checks:

- `./scripts/write-kernel-hashes.sh` completed.
- An isolated compile check for the new
  `gemm_hfq4g256_residual_mmq.hip` source passed for
  `gfx1010`, `gfx1030`, `gfx1100`, `gfx1200`, and `gfx1201`.
- The full `./scripts/compile-kernels.sh gfx1010 gfx1030 gfx1100 gfx1200
  gfx1201` matrix was invoked, but it still fails on pre-existing unrelated
  kernels in the current upstream matrix (`57 failed`). The new MMQ source no
  longer blocks non-RDNA3 targets; unsupported targets compile stub entry
  points and dispatch never selects the MMQ path there.

## How Scientific Is This?

The core causal claim is reasonably supported:

- The comparison used the same machine, same ROCm toolbox, same Q8 KV mode, and
  the same pp2048 prompt-processing shape.
- The main result was reproduced in three fresh processes, which matters because
  same-process measurements showed DPM/thermal drift.
- Profiling before and after shows the improvement lands exactly in the MMQ GEMM
  kernels that were changed.
- Negative controls ruled out several plausible but wrong causes: final logits,
  KV mode, rocBLAS, plain Q8 activations, graph capture, prepacking, and simple
  launch-bound tuning.
- The unsafe global no-check variant was rejected after a real GPU fault, and
  the kept version is guarded by exact full-tile predicates.

The experiment is not fully publication-grade:

- It uses one Strix Halo system and one main 9B model.
- The hipfire target is HFQ4/MQ4 while llama.cpp uses Q4_K_M GGUF, so this is a
  practical systems comparison, not a bit-identical format comparison.
- The canonical 27B DFlash coherence gate could not run because the 27B files
  were absent.
- The prompt-processing command uses synthetic token IDs for hipfire's focused
  bench, while llama.cpp's `llama-bench` uses its own benchmark harness.
- The new prompt-processing correctness QA uses a short real tokenizer prompt
  and validates batched-vs-sequential equivalence, but it is still not a full
  long-context semantic equivalence proof.
- No power/clock pinning was enforced; fresh-process medians reduce this risk
  but do not eliminate it.
- The latest pp2048 runs range from `1053` to `1125 tok/s`; that is an
  engineering win, but the spread means it should be reported as a range rather
  than a single exact number.

Practical conclusion: the result is strong enough for engineering decisions and
future optimization work, but should be described as an experimentally
reproduced local Strix Halo result, not as a universal AMD performance claim.

## Current Conclusion

The low hipfire prefill speed was primarily a GEMM algorithm issue. Moving the
hot HFQ4 prefill projections to a llama.cpp-style MMQ tiled algorithm closed
most of the gap, and the full-tile fast path brought pp2048 above 1000 tok/s on
gfx1151. The follow-up full-tile specializations bring pp2048 into the same
range as the local llama.cpp reference. Using a 4096-token prefill chunk also
brings pp4096 above 1000 tok/s on this Strix Halo system, while pp8192 still
drops to roughly 720 tok/s.

Remaining work should focus on:

- reducing MMQ overhead for tail shapes;
- checking pp4096 and longer contexts after the full-tile path;
- comparing exact model/format effects more carefully;
- running the 27B coherence gate once the required files are available;
- keeping the path opt-in until more AMD targets have been checked.
