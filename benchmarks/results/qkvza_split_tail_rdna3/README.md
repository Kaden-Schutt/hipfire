# RDNA3 QKVZA Split-Tail A/B

This directory contains the consolidated local W7900 evidence for the opt-in
`HIPFIRE_QKVZA_SPLIT_TAIL=1` route.

The benchmark toggles only `HIPFIRE_QKVZA_SPLIT_TAIL` and runs the production
`bench_qwen35_mq4` prefill path through
`scripts/bench_qwen36_qkvza_split_tail_ab.sh`.

Common setup:

- GPU: ROCm device 0, `gfx1100` / AMD Radeon Pro W7900, ROCm 7.2
- Prompt prefill tokens: `4096`
- Prefill runs per mode: `3`
- Generation tokens: `1` (smoke only; results below are prefill results)
- KV mode: `q8`
- DPM warmup: `2s`

Hardware evidence is recorded in `system_info.txt` from `rocm-smi` and
`rocminfo`. The test host has two `gfx1100` W7900-class GPUs; these benchmark
runs used `GPU_ID=0` / `HIP_VISIBLE_DEVICES=0`, mapped to the AMD Radeon Pro
W7900 device.

## Median Summary

| Model | off median tok/s | on median tok/s | Delta |
|---|---:|---:|---:|
| Qwen3.5-0.8B MQ4 | 7974.7 | 8403.4 | +5.38% |
| Qwen3.5-4B MQ4 | 2685.0 | 2826.9 | +5.28% |
| Qwen3.5-27B MQ4 | 598.3 | 620.6 | +3.73% |
| Qwen3.6-27B MQ4 | 595.0 | 613.8 | +3.16% |

Interpretation:

- The opt-in split-tail route is consistently positive across 0.8B, 4B, and
  27B Qwen-family MQ4 checkpoints on gfx1100.
- The largest relative gains appear on smaller models where the QKVZA route is
  a larger share of total prefill time.
- This is a prefill-path result, not a decode-throughput claim.

Files:

- `summary.tsv`: median off/on throughput and delta by model.
- `raw_prefill.tsv`: per-run raw prefill timings used to compute the medians.
- `system_info.txt`: ROCm device inventory for the benchmark host.
