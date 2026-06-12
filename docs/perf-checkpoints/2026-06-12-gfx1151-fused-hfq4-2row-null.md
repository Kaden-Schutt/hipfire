# gfx1151 Fused HFQ4 Two-Row Decode Probe

Date: 2026-06-12
Branch: `chaingun`
GPU: `gfx1151` (Strix Halo / Radeon 8060S)

## Question

The refreshed Qwen3.5/Qwen3.6 gfx1151 profiles put decode time back on the
HFQ4 GEMV family. The fused QKV/QKVZA kernels are still one 32-lane row worker
per block on RDNA, while the existing CDNA wave64 sibling packs two 32-lane row
workers per block. Since that sibling uses no CDNA-only intrinsic, this probe
tests whether gfx1151 benefits from the same two-row CTA shape.

## Implementation

`HIPFIRE_FUSED_HFQ4_2ROW_GFX1151=1` routes gfx1151 single-token
`fused_qkv_hfq4g256` and `fused_qkvza_hfq4g256` through the existing
two-row `*_wave64` kernels:

- `fused_qkv_hfq4g256_wave64`
- `fused_qkvza_hfq4g256_wave64`

The route is default-off because the A/B below is flat. It is kept as an
explicit probe so future larger-shape or server-batched decode work can retest
without re-plumbing dispatch.

## Commands

```bash
HIPFIRE_FUSED_HFQ4_2ROW_GFX1151=$mode HIPFIRE_PROFILE_DECODE=1 \
HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
  target/release/examples/bench_qwen35_speed \
  ~/.hipfire/models/qwen3.6-35b-a3b-mq4.hfq \
  --prefill 32 --prefill-runs 1 --warmup 3 --gen 12

HIPFIRE_FUSED_HFQ4_2ROW_GFX1151=$mode HIPFIRE_PROFILE_DECODE=1 \
HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
  target/release/examples/bench_qwen35_speed \
  ~/.hipfire/models/qwen3.5-9b.mq4.hfq \
  --prefill 32 --prefill-runs 1 --warmup 3 --gen 12
```

Raw log: `/tmp/gfx1151-fused-hfq4-2row-ab-20260612-101415.log`

## Results

| Model | Mode | gen tok/s | decode wall | fused QKVZA | fused QKV |
|---|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B MQ4 | `0` | 52.4 | 229.1 ms | 24.8 ms | 6.5 ms |
| Qwen3.6-35B-A3B MQ4 | `1` | 52.7 | 227.9 ms | 24.9 ms | 6.5 ms |
| Qwen3.5-9B MQ4 | `0` | 38.6 | 310.6 ms | 37.7 ms | 10.6 ms |
| Qwen3.5-9B MQ4 | `1` | 38.7 | 310.3 ms | 37.6 ms | 10.6 ms |

## Decision

Do not default the two-row fused route. The end-to-end deltas are inside the
short-run noise band and the targeted fused rows do not materially improve.
The remaining gfx1151 decode work should move to a lower-level GEMV lever
instead of block-packing the already fused projection kernels.
