# DeepSeek V4 gfx1201 TP3 MQ2 decode screen

Date: 2026-08-07  
Branch: `ds4-gfx1201-opt`  
Candidate checkpoint: `0144263bc`  
Campaign line: `54.903757` tok/s with the selected T1024 HC control candidate

## Fused local down plus combine

The exact-gfx1201 TP3 experiment replaced deterministic expanded MQ2 down plus
`moe_down_combine_k8_batched` with one wave per output row. The wave iterated
all six routing slots in the incumbent order, computed only the two rank-owned
dots, included positive-zero contributions for non-owned slots, and performed
the same in-place residual add.

The channel passed raw-bit identity but was decisively sub-threshold:

| Arm | Time/layer | 43-layer projection |
|---|---:|---:|
| expanded EP down + fixed-order combine | 0.025114 ms | — |
| fused local down + combine | 0.024778 ms | +0.014450 ms/token |

That is a 1.014x channel speedup and only +0.079% projected product throughput,
far below the 2% continuation gate. All 4,096 combined outputs matched at raw
bits. The candidate uses 95 VGPR, 64 SGPR, wave32, and no spills or private
segment; the incumbent expanded down uses 93 VGPR and 38 SGPR.

The conclusion is useful: expanded output traffic and the combine launch are
not the routed-down bottleneck. The owned MQ2 dot and its weight stream dominate,
so further work must improve that body rather than remove its bookkeeping.

Evidence:
`hiptrx:/home/kaden/ds4-hiptrx-evidence/2026-08-07-gfx1201-mq2-down-combine/`.

## Next gate

Screen an owned-dot kernel structure that materially raises bandwidth across
both production shapes: gate/up `(M=4096,K=4096)` and down
`(M=4096,K=2048)`. The current exact kernels are direct ports with 74 and 93
VGPR respectively. Any candidate must preserve the MQ2 codebook and FP32
reduction order, beat the occurrence-weighted 2% product projection, and stay
behind the exact gfx1201 device type.
