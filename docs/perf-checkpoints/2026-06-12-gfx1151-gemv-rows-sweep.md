# gfx1151 GEMV Rows Sweep

Date: 2026-06-12
Host GPU: gfx1151 (Strix Halo)
Branch: `chaingun`

## Question

gfx115x APUs had been defaulting HFQ4/MQ4 decode GEMV to the RDNA3 multi-row
path with `R=2`. The open question was whether larger row groups (`R=4` or
`R=8`) should replace that default for Qwen3.5 dense decode, or whether the
multi-row path should stay opt-in.

## Commands

```bash
HIPFIRE_GEMV_ROWS=$rows HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
  cargo run --release --features deltanet -p hipfire-runtime \
  --example bench_qwen35_mq4 -- \
  ~/.hipfire/models/qwen3.5-4b.mq4.hfq \
  --prefill 32 --prefill-runs 1 --warmup 5 --gen 50

HIPFIRE_GEMV_ROWS=$rows HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
  cargo run --release --features deltanet -p hipfire-runtime \
  --example bench_qwen35_mq4 -- \
  ~/.hipfire/models/qwen3.5-9b.mq4.hfq \
  --prefill 32 --prefill-runs 1 --warmup 5 --gen 50
```

Logs:

- `/tmp/gfx1151-gemv-rows-sweep-20260612-094318.log`
- `/tmp/gfx1151-gemv-rows-sweep-9b-20260612-094335.log`

## Results

| Model | Rows | gen tok/s | avg ms/tok | prefill tok/s |
|---|---:|---:|---:|---:|
| Qwen3.5-4B MQ4 | 1 | 66.8 | 14.81 | 583.3 |
| Qwen3.5-4B MQ4 | 2 | 66.2 | 14.94 | 584.6 |
| Qwen3.5-4B MQ4 | 4 | 66.1 | 14.99 | 582.7 |
| Qwen3.5-4B MQ4 | 8 | 65.8 | 15.09 | 592.3 |
| Qwen3.5-9B MQ4 | 1 | 44.2 | 22.48 | 282.2 |
| Qwen3.5-9B MQ4 | 2 | 44.0 | 22.58 | 286.8 |
| Qwen3.5-9B MQ4 | 4 | 42.6 | 23.34 | 281.7 |
| Qwen3.5-9B MQ4 | 8 | 42.9 | 23.14 | 285.2 |

## Decision

Keep the RDNA3 multi-row kernels available, but do not default gfx115x to them.
For the two dense Qwen3.5 decode shapes measured here, the single-row path wins
or ties within noise and the larger row groups regress. gfx115x now defaults to
`R=1`; `HIPFIRE_GEMV_ROWS=2/4/8` remains the opt-in tuning hook for future
larger-shape or server-batched decode experiments.
