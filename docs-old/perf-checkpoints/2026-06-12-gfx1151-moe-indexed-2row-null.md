# gfx1151 Indexed MoE HFQ4 Two-Row Null Result

## Target

Qwen3.6-35B-A3B MQ4 decode on gfx1151. The profile hotspot was the indexed
MoE HFQ4 decode pair:

- `gemv_hfq4g256_moe_gate_up_k8_indexed`
- `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`

## Candidate

`HIPFIRE_MOE_INDEXED_2ROW_GFX1151=1`:

- routes indexed gate/up through the existing two-row CTA kernels used by the
  CDNA wave64 path;
- routes expanded down through
  `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_2row_gfx1151`.

Default remains off.

## A/B

Command shape:

```bash
HIPFIRE_MOE_INDEXED_2ROW_GFX1151={0,1} \
HIPFIRE_PROFILE_DECODE=1 \
HIPFIRE_KV_MODE=asym3 \
HIPFIRE_GRAPH=1 \
target/release/examples/bench_qwen35_speed \
  ~/.hipfire/models/qwen3.6-35b-a3b-mq4.hfq \
  --prefill 32 --prefill-runs 1 --warmup 3 --gen 12
```

| Mode | Decode tok/s | Decode wall | Gate/up | Expanded down |
|---|---:|---:|---:|---:|
| off | 52.5 | 228.8 ms | 23.2 ms | 16.7 ms |
| on | 52.6 | 228.3 ms | 23.4 ms | 16.8 ms |

## Conclusion

This is a null result. The two-row route halves grid.x for the indexed MoE
GEMVs, but on gfx1151 the measured hot rows are unchanged within noise and
slightly worse per-kernel. Keep the route as a default-off probe so future
work can test it against different A3B shapes without rebuilding the kernel
plumbing, but do not auto-dispatch it.
