# RDNA3 QKVZA Split-Tail A/B Summary

This records the local W7900 evidence for the opt-in
`HIPFIRE_QKVZA_SPLIT_TAIL=1` route. The benchmark toggles only this
environment variable and runs the production `bench_qwen35_mq4` prefill path.

Common setup:

- GPU: gfx1100 / W7900, ROCm 7.2
- Benchmark script: `scripts/bench_qwen36_qkvza_split_tail_ab.sh`
- Prompt prefill tokens: `4096`
- Prefill runs per mode: `3`
- Generation tokens: `1` (smoke only; the result below is a prefill result)
- KV mode: `q8`
- DPM warmup: `2s`

| Model | Result dir | off median tok/s | on median tok/s | Delta |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B MQ4 | `benchmarks/results/qkvza_split_tail_rdna3_qwen35_08b_20260706_162111` | 7974.7 | 8403.4 | +5.38% |
| Qwen3.5-4B MQ4 | `benchmarks/results/qkvza_split_tail_rdna3_qwen35_4b_20260706_1635` | 2685.0 | 2826.9 | +5.28% |
| Qwen3.5-27B MQ4 | `benchmarks/results/qkvza_split_tail_rdna3_qwen35_27b_20260706_1638` | 598.3 | 620.6 | +3.73% |
| Qwen3.6-27B MQ4 | `benchmarks/results/qkvza_split_tail_rdna3_20260706_131840` | 595.0 | 613.8 | +3.16% |

Interpretation:

- The switch is consistently positive across 0.8B, 4B, and 27B Qwen-family
  MQ4 checkpoints on gfx1100.
- The largest relative gains appear on smaller models where the QKVZA route is
  a larger share of total prefill time.
- This remains an opt-in RDNA3 dGPU path and is not a decode-throughput claim.
