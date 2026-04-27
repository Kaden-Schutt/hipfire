# 2026-04-27 Strix Halo Q8 Prefill

Hardware: Radeon 8060S / gfx1151, ROCm 7.2, toolbox `llama-rocm-7.2`.

Models:

- llama.cpp: `/home/kotdath/omp/personal/amd-strix-halo-toolboxes/models/Qwen3.5-9B-Q4_K_M.gguf`
- hipfire target: `~/.hipfire/models/qwen3.5-9b.mq4`
- hipfire draft: `~/.hipfire/models/qwen35-9b-dflash-mq4.hfq`
- hipfire target converted from the Unsloth GGUF:
  `/home/kotdath/omp/personal/amd-strix-halo-toolboxes/models/qwen3.5-9b-unsloth.hf4`

Binaries:

- `bench_qwen35_mq4`: `7cd9a6b059dc4d05b78ffb504470c680`
- `dflash_spec_demo`: `8df7bd8dc9d8f716a79fab8f7c95f825`
- `lru_cache_pep8_strict.txt`: `df5dedc8040ce70ba55080c4548e6024`

## llama.cpp Q8_0 KV

Command pattern:

```bash
llama-bench -m /home/kotdath/omp/personal/amd-strix-halo-toolboxes/models/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 -p <N> -n 1 -r 3 -ctk q8_0 -ctv q8_0 -fa 1
```

`-fa 1` is required for `-ctv q8_0`; without it llama.cpp fails context creation.

| Prompt tokens | Prefill tok/s | Gen tok/s |
| ---: | ---: | ---: |
| 2048 | 1076.08 +/- 3.87 | 33.70 +/- 0.42 |
| 4096 | 1043.52 +/- 7.44 | 33.66 +/- 0.42 |

## hipfire Q8 KV

Command pattern:

```bash
HIPFIRE_KV_MODE=q8 ./target/release/examples/bench_qwen35_mq4 \
  ~/.hipfire/models/qwen3.5-9b.mq4 --prefill <N> --prefill-runs 3 --warmup 0 --gen 1
```

Before the gfx1151 routing change, q8 pp2048 measured `233.0 tok/s` median.

After routing gfx1150/gfx1151 away from WMMA prefill GEMM and onto dot2:

| Prompt tokens | Run tok/s | Median tok/s | Gen tok/s |
| ---: | --- | ---: | ---: |
| 2048 | 322.1, 320.5, 323.5 | 322.1 | 42.4 |
| 4096 | 307.1, 306.3, 302.5 | 306.3 | 38.0 |

After additionally selecting the existing residual `k2x32` WMMA variant on
gfx1150/gfx1151 only:

| Prompt tokens | Variant | Run tok/s | Median tok/s | Gen tok/s |
| ---: | --- | --- | ---: | ---: |
| 2048 | `k2` override | 340.6, 343.0, 337.2, 340.5, 336.7 | 340.5 | 38.0 |
| 2048 | auto `k2x32` | 360.1, 357.4, 357.4, 356.4, 355.7 | 357.4 | 39.1 |

This is a small but repeatable +5% over the same-binary `k2` baseline.

Re-check after the negative experiments were reverted:

| Prompt tokens | Run tok/s | Median tok/s | Gen tok/s |
| ---: | --- | ---: | ---: |
| 2048 | 355.8, 354.2, 354.0 | 354.2 | 42.8 |

Changing `HIPFIRE_PREFILL_MAX_BATCH` did not materially move pp2048:

| Max batch | Prefill tok/s |
| ---: | ---: |
| 64 | 361.5 |
| 128 | 356.3 |
| 192 | 358.9 |
| 256 | 357.2 |
| 384 | 359.0 |
| 512 | 351.6 |
| 1024 | 304.5 |
| 2048 | 306.8 |

The larger 1024/2048-token chunks regress instead of converging toward
llama.cpp, so the gap is not caused by hipfire's default chunk size being too
small.

## Model Artifact Check

The Unsloth GGUF was converted into hipfire `hf4` format and benchmarked with
the same Q8 KV prefill command. The result was `357.8 tok/s` median and `43.5`
gen tok/s, matching the native hipfire `.mq4` within noise. This rules out the
final Unsloth/GGUF model artifact as the cause of the 3x prefill gap; the gap
is in the hipfire execution format/kernels.

## Negative Experiments

These were tested and not kept:

- `HIPFIRE_ROCBLAS_ALL_ARCHS=1 HIPFIRE_ROCBLAS_MIN_BATCH=1`: `106.3 tok/s`
  prefill, much slower than the custom kernels on gfx1151.
- `BATCH_TILE=16` for dot2 qkv/qkvza/gate_up: regressed to about `223 tok/s`.
- `PREFILL_MAX_BATCH=512`: about `352 tok/s`, no useful gain over 256 and
  higher scratch footprint.
- `HIPFIRE_GRAPH=1`: about `348 tok/s`, so launch overhead is not the main
  limiter.
- `HIPFIRE_WMMA=1`: about `241 tok/s`; the old WMMA prefill path remains worse
  on gfx1151.
- 32x2 thread-block grouping for `gemm_gate_up_hfq4g256_dot2`: compiled and ran
  after fixing launch bounds, but measured about `354 tok/s`, i.e. no useful
  improvement over the simpler dot2 kernel.
- `HIPFIRE_WMMA=1` re-check after the residual routing change: about `225 tok/s`;
  gate/up WMMA alone was roughly 18 ms/call vs about 11 ms/call for dot2.
- Gate/up WMMA x32, modeled after the residual `k2x32` variant: about
  `235.5 tok/s`; still much slower than dot2.
- Gate/up dot2 with `BATCH_TILE=16` only: about `239 tok/s`; more batch rows per
  workgroup increased register pressure enough to lose badly.
- Gate/up dot2 multi-wave/LDS weight-sharing experiment: about `264.7 tok/s`;
  sharing did not compensate for occupancy and scheduling costs.
- `HIPFIRE_ROCBLAS_ALL_ARCHS=1 HIPFIRE_ROCBLAS_MIN_BATCH=1 ROCBLAS_USE_HIPBLASLT=1`:
  about `294 tok/s`; RDNA rocBLAS/hipBLASLt was still slower than the custom
  kernels for this format.
- Gate/up Q8-activation integer-dot experiment: compiled with `sudot4` on
  gfx1151 but measured about `309 tok/s`. Simple per-call Q8 activation
  quantization without MMQ-style row/batch tiling is a regression.
- Gate/up dot2 `BATCH_TILE=4`: the first apparent `~378 tok/s` result was an
  invalid host/kernel tile mismatch. The corrected implementation measured
  about `294 tok/s`, so it was reverted.
- Gate/up dot2 `__launch_bounds__(32, 16)`: measured `360.8 tok/s` median vs
  `359.7 tok/s` for the default `__launch_bounds__(32, 8)` in the same A/B
  session. This is measurement noise, not a useful tuning point.
- `HIPFIRE_MW16=1` residual path with per-call HFQ4->FP16 dequant: regressed to
  `~72-108 tok/s`. Dequantizing every prefill call is too expensive.
- Cached FP16-shadow residual MW16 prototype: reusing dequantized residual
  weights still measured only `~178 tok/s`, slower than the existing residual
  `k2x32` WMMA path on gfx1151.
- Skipping final prefill logits (temporary `HIPFIRE_PREFILL_SKIP_LOGITS=1`) only
  moved pp2048 to `~363 tok/s`. This matters for apples-to-apples methodology
  because llama.cpp's `llama-bench` calls `llama_batch_get_one(...)` with
  `logits=nullptr`, but it does not explain the gap.
- Gate/up Q8 activation prototype (`HIPFIRE_GATE_UP_Q8X=1`) quantized the
  activation matrix once per gate/up call and used gfx1151 integer dot
  instructions, but stayed in hipfire's row-wise work decomposition. It
  measured `~189.5 tok/s`, much slower than the dot2 baseline. This confirms
  that the llama.cpp-like win is not "Q8 activations" alone; it requires the
  MMQ row/batch tiling layout.
- Extending the existing `HIPFIRE_ROCBLAS_ALL_ARCHS=1` experiment so gate/up
  also used the FP16-shadow rocBLAS path measured `~348.2 tok/s` with
  `ROCBLAS_USE_HIPBLASLT=1`, still below the current dot2/k2x32 path. rocBLAS
  FP16 shadows are not a useful RDNA/Strix Halo bypass for this model.
- Gate/up via two calls to the existing residual `k2x32` WMMA kernel after
  zero-filling the output buffers measured `~312.4 tok/s`. The wider residual
  row tile is not a drop-in replacement for fused gate/up; the extra launches,
  memset, and separate output traffic outweigh the tiling benefit.
- Gate/up Q8 activation with a 4-row workgroup tile (`HIPFIRE_GATE_UP_Q8R4=1`)
  measured `~297.7 tok/s` and was also only approximate math. Reusing a small
  X tile across four rows was not enough; the quantization overhead and
  row-wise HFQ4 format still lose to the current dot2 path.
- Existing decode-style Q4_K kernels are not the missing piece. The
  `bench_hfq4g128` microbench on gfx1151 measured HFQ4 faster than Q4K for
  single-vector GEMV shapes, e.g. `4096x4096`: HFQ4 `15.4 us` vs Q4K
  `32.6 us`, and `12288x4096`: HFQ4 `50.3 us` vs Q4K `97.9 us`. The llama.cpp
  advantage is the batched MMQ algorithm, not the raw Q4_K format alone.

## DFlash Smoke

Prompt: `benchmarks/prompts/lru_cache_pep8_strict.txt` (`231` tokens), `--ctx 2048 --kv-mode q8 --no-adaptive-b --no-chatml`.

| Mode | Prefill tok/s | Decode tok/s | Notes |
| --- | ---: | ---: | --- |
| AR baseline | 358.9 | 47.84 | target-only greedy |
| DFlash | 359.0 | 92.65 | tau 8.75, accept rate 0.583 |

## Experimental MMQ Port

Commit under test: `c5a28ca` plus local MMQ changes.

Binaries:

- `bench_qwen35_mq4`: `5fcafd6c8719275046caa10e629140c4`
- `dflash_spec_demo`: `fef1598678742ee1153b12ffd9c7a76a`
- `test_hfq4g256QA`: `15139ebc102703ebd5ffeb871655d79a`

New opt-in flags:

- `HIPFIRE_MMQ=1`: routes gfx1100/gfx1101/gfx1102/gfx1103/gfx1150/gfx1151
  HFQ4 prefill GEMMs through an experimental llama.cpp-style MMQ path.
- `HIPFIRE_PREFILL_REUSE_PBS=1`: allocates a reusable `PrefillBatchScratch`
  in `Qwen35Scratch`.
- `HIPFIRE_PREFILL_MAX_BATCH=2048`: lets the regular `forward_prefill_batch`
  process pp2048 as one chunk instead of eight 256-token chunks.

Command:

```bash
HIPFIRE_VERIFY_GRAPH=0 HIPFIRE_MMQ=1 \
HIPFIRE_PREFILL_REUSE_PBS=1 HIPFIRE_PREFILL_MAX_BATCH=2048 \
target/release/examples/bench_qwen35_mq4 \
  ~/.hipfire/models/qwen3.5-9b.mq4 \
  --prefill 2048 --prefill-runs 1 --gen 0 --warmup 0
```

Fresh-process pp2048 results:

| Run | Prefill ms | Prefill tok/s |
| ---: | ---: | ---: |
| 1 | 2516.2 | 813.9 |
| 2 | 2515.6 | 814.1 |
| 3 | 2520.6 | 812.5 |

Same-process 5-run result with the same env showed DPM/thermal drift:
`813.5, 815.0, 797.5, 793.3, 792.4 tok/s`. Per the project playbook,
fresh-process runs are the fairer number for this bench class.

Kernel QA:

```text
HFQ4G256 QA PASS: gpu_cpu_err=0.000366 quant_ref_err=0.000000 mmq_err=0.040617
```

Short 9B DFlash sanity with `HIPFIRE_MMQ=1` produced coherent Fibonacci
text, `decode_tau=3.6522`, `decode_tok_s=24.98`. The canonical
`coherence-gate-dflash.sh` was invoked with the same MMQ env, but skipped
because the expected 27B files (`qwen3.5-27b.mq4` and
`qwen35-27b-dflash.mq4`) were not present in `~/.hipfire/models`.

## MMQ Full-Tile Fast Path

Commit under test: local changes after `d77f1ec`.

Binaries:

- `bench_qwen35_mq4`: `e466040940b12e46c73bf00504294b41`
- `dflash_spec_demo`: `7f87da798bc93cff8bd3cbb8ddff6351`
- `test_hfq4g256QA`: `c07383316121d12bc43824db8fea9bee`

The next bottleneck was the generic MMQ kernel's tail handling. A temporary
no-check kernel proved unsafe globally: it passed the standalone 128x128 QA
case, but the real Qwen3.5 forward hit smaller/tail MMQ shapes and faulted.
The kept version therefore adds a separate `gemm_hfq4g256_residual_mmq_full`
entrypoint and dispatches to it only when both `m` and `batch_size` are exact
128-wide tiles. All other shapes stay on the checked generic kernel.

Fresh-process pp2048 results with the same command/env as above:

| Run | Prefill ms | Prefill tok/s |
| ---: | ---: | ---: |
| 1 | 1989.8 | 1029.2 |
| 2 | 1974.7 | 1037.1 |
| 3 | 1923.8 | 1064.6 |

Tail-shape smoke (`--prefill 1937`) completed without a GPU fault and measured
`733.0 tok/s`, confirming that non-128 batch sizes route through the safe
generic path.

Profile delta, pp2048, serialized kernel timers:

| Kernel group | Before full-tile | After full-tile |
| --- | ---: | ---: |
| `gemm_hfq4g256_mmq_set` | 1409.1 ms | 936.4 ms |
| `gemm_hfq4g256_residual_mmq` | 722.6 ms | 445.0 ms |
| Total serialized | 2629.1 ms | 1898.0 ms |

Correctness/smoke:

- `HFQ4G256 QA PASS: gpu_cpu_err=0.000366 quant_ref_err=0.000000 mmq_err=0.040617`
- AR baseline short Fibonacci prompt produced coherent text at `43.59 tok/s`.
- DFlash short Fibonacci prompt produced coherent text, `decode_tau=1.4615`,
  `decode_tok_s=12.90`.

Additional negative experiments after the MMQ baseline:

- MMQ `set`/`add` split entrypoints regressed to about `764 tok/s`; the uniform
  runtime branch was not the limiter.
- `MMQ_X=64` with adjusted row mapping passed QA but regressed to about
  `688 tok/s`.
- `MMQ_Y=64` broke QA with the current writeback layout.
- Full HFQ4 prepack into an MMQ-ready shadow was correct but slower
  (`~654-668 tok/s`), likely due the larger global-memory footprint and less
  favorable cache behavior.
- Single-thread-per-row metadata fill was correct but slightly slower
  (`~804 tok/s`), so duplicated metadata loads were not material.
- `__launch_bounds__(..., 1)` regressed to about `755 tok/s`; keeping two
  resident blocks is better on gfx1151.
- `HIPFIRE_ROCBLAS_ALL_ARCHS=1` with FP16 shadows measured only `~239 tok/s`
  on the second run, so rocBLAS is still not a useful RDNA bypass here.

## MMQ Full-Tile Specialization Follow-Up

Commit under test: local changes after `947eef8`.

Binaries:

- `bench_qwen35_mq4`: `d926b1e1a6cc25cd498eda390f2fbe65`
- `dflash_spec_demo`: `3840178c5e166c0ec42a2a7ce5e7a744`
- `test_hfq4g256QA`: `2fec248d246697539f0831ee0854f32d`

Two full-tile-only refinements were added:

- `load_hfq4_tile<true>` removes the row clamp inside the full-tile kernel;
  the generic kernel still uses `load_hfq4_tile<false>`.
- `gemm_hfq4g256_residual_mmq_full_{set,add}` specialize the full-tile
  writeback for set vs residual-add. The checked generic tail kernel remains
  unchanged.

Fresh-process results:

| Prompt tokens | Run tok/s |
| ---: | --- |
| 2048 | 1073.6, 1095.2, 1116.9, 1125.0, 1125.4, 1077.0, 1053.2 |
| 4096 | 997.9, 1002.9 |
| 1937 | 753.5 |

The pp2048 runs vary with the same DPM/thermal sensitivity seen earlier, but
the post-specialization range is consistently around or above the llama.cpp
pp2048 reference measured earlier in the same investigation (`1076.08 +/- 3.87`
tok/s). The pp4096 result also reaches the 1000 tok/s target once, but is still
too close to the threshold to claim a stable win over llama.cpp's
`1043.52 +/- 7.44` pp4096 reference.

Correctness:

```text
HFQ4G256 QA PASS: gpu_cpu_err=0.000366 quant_ref_err=0.000000 mmq_err=0.040617
```

Short generation smokes after the same change:

- AR baseline Fibonacci prompt: coherent text, `47.26 tok/s`.
- DFlash Fibonacci prompt: coherent text, `decode_tau=1.6000`,
  `decode_tok_s=13.58`.

Additional negative experiment:

- Removing the second-128-block `if` from the full kernel regressed pp2048 to
  about `1072 tok/s`, so the compiler prefers the guarded version despite the
  condition being effectively always true for the tested HFQ4 shapes.

## Interpretation

Q8 KV does not explain the prefill gap: llama.cpp stays around 1.0k tok/s with
Q8_0 KV, while hipfire stays GEMM-bound. Profiling after the gfx1151 routing
fix shows the remaining prefill time is dominated by 4-bit GEMM:

- `gemm_gate_up_hfq4g256_dot2`: ~49%
- `gemm_hfq4g256_residual_wmma_k2x32`: ~27%
- `gemm_qkvza_hfq4g256_dot2`: ~13%
- `gemm_qkv_hfq4g256_dot2`: ~3.5%

The gap also does not appear to be DPM/power-state driven. During the same
session, sysfs reported `power_dpm_force_performance_level=auto` and
`sclk=775Mhz`, but llama.cpp still measured `1069.93 tok/s` for pp2048. That
keeps the comparison valid: llama.cpp is fast in the same observed state where
hipfire is about `354-357 tok/s`.

The structural difference is the GEMM algorithm. llama.cpp's HIP/CUDA backend
routes quantized prompt processing through MMQ: it quantizes the activation
matrix into a `q8_1` layout and tiles both the batch dimension and the weight
rows. The experimental hipfire MMQ port confirms this diagnosis: moving the
hot HFQ4 prefill GEMMs from row-wise dot2/WMMA kernels to 128x128 Q8_1/HFQ4
MMQ tiling lifts pp2048 from ~354 tok/s to ~813 tok/s on gfx1151.

Two details from llama.cpp matter for a faithful port:

- `tools/llama-bench/llama-bench.cpp::test_prompt` uses
  `llama_batch_get_one(tokens.data(), n_tokens)`.
- `src/llama-batch.cpp::llama_batch_get_one` sets `logits = nullptr`, so the
  prompt-processing number is not dominated by final logits. Hipfire's
  temporary no-logits test confirmed this is not the main gap.

The MMQ path then routes through `ggml_cuda_mul_mat_q`, first quantizing the
activation matrix with `quantize_mmq_q8_1_cuda`, then launching
`mul_mat_q_case<GGML_TYPE_Q4_K>`. For AMD, llama.cpp uses `mmq_y=128`,
`MMQ_ITER_K=256`, and Q4_K/Q8_1 vector-dot helpers with
`VDR_Q4_K_Q8_1_MMQ=8`. That design reuses the activation tile across many
weight rows; the experimental hipfire MMQ path ports that same structural
idea to HFQ4-G256 storage.

The measured safe improvements before the MMQ port were:

1. Avoid the current WMMA prefill qkv/qkvza/gate_up path on gfx1150/gfx1151 and
   route those kernels to dot2.
2. Use residual `k2x32` only on gfx1150/gfx1151, where it is faster than `k2`;
   keep gfx1100 and other AMD targets on the previous auto path because the
   source comment records `k2x32` as slower on gfx1100.

The remaining gap to llama.cpp is now much smaller but still visible
(~813 tok/s vs ~1076 tok/s pp2048). The next work should focus on reducing
MMQ kernel overhead and closing format-specific gaps versus llama.cpp's Q4_K
implementation, not KV-cache tuning.
