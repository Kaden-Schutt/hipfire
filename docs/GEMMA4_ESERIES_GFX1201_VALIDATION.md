# Gemma 4 E-Series gfx1201 Validation

## Environment

- Host GPU: AMD Radeon AI PRO R9700, `gfx1201`, 32 GiB VRAM
- ROCm/HIP: `7.14.60850`
- Build: `cargo build --release --features deltanet --example daemon -p hipfire-runtime`
- Build result: success in 54.40 seconds
- KV cache: Q8
- Sampling: greedy, temperature 0
- Runtime graph and EAGLE: disabled

## Model Artifacts

| Model | SHA256 |
|---|---|
| Gemma 4 E2B Q8 | `7dc98107c0e802036c30b1ee906f74a4214cbba75f81c7983e5bf184ee357a2d` |
| Gemma 4 E4B Q8 | `0224a0e3240030c0b5d14dcbf4f3ace1c96bea3ae166e8372068c843f589e80a` |

## GSM8K Smoke

The first 30 fixed GSM8K CoT prompts were run on each model. Both models completed every request without a runtime error.

| Metric | E2B gfx1201 | E4B gfx1201 |
|---|---:|---:|
| Completed | 30/30 | 30/30 |
| Runtime errors | 0 | 0 |
| Flexible numeric accuracy | 80.0% | 90.0% |
| Median prefill | 513.685 tok/s | 346.525 tok/s |
| Median decode | 96.575 tok/s | 64.435 tok/s |
| Median TTFT | 1,490.46 ms | 2,208.07 ms |
| Median wall time | 3.738 s | 4.005 s |

## Cross-Architecture Comparison

The same 30 prompts and model artifacts were compared against the prior W7900/gfx1100 run.

| Model | Metric | W7900/gfx1100 | R9700/gfx1201 | Delta |
|---|---|---:|---:|---:|
| E2B | Prefill | 414.685 tok/s | 513.685 tok/s | +23.87% |
| E2B | Decode | 102.490 tok/s | 96.575 tok/s | -5.77% |
| E2B | TTFT | 1,844.80 ms | 1,490.46 ms | -19.21% |
| E2B | Wall time | 3.968 s | 3.738 s | -5.80% |
| E4B | Prefill | 281.685 tok/s | 346.525 tok/s | +23.02% |
| E4B | Decode | 68.225 tok/s | 64.435 tok/s | -5.55% |
| E4B | TTFT | 2,713.77 ms | 2,208.07 ms | -18.63% |
| E4B | Wall time | 4.415 s | 4.005 s | -9.28% |

All 30 normalized predictions and correctness decisions matched between gfx1100 and gfx1201 for both models. Generated text was byte-identical for 23/30 E2B prompts and 24/30 E4B prompts. The remaining responses diverged during greedy generation but retained the same normalized final answer and correctness result. This supports functional parity for the tested set, not full bitwise parity across GPU architectures.

The current E-series PLE and shared-KV implementation therefore runs on gfx1201 without an additional architecture-specific path. This is a functional validation, not a claim that gfx1201 prefill or decode is fully optimized.
