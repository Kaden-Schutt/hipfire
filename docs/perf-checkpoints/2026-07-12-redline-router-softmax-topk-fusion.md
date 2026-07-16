# Exact router softmax+top-k fusion: 40 launches removed, throughput neutral

## Verdict

Reverted under the product bar. The exact fused router is bit-exact and reduces
the retained tape from 833 to 793 launches, but its order-balanced tg128 result
is +0.07%, below reproducibility. It is useful evidence that dispatch count by
itself is no longer the limiting cost in the retained-PM4 champion.

## Candidate

The dormant `moe_softmax_topk_renorm_k8` kernel previously differed from the
established `softmax_f32` + `moe_topk_renorm_k8` sequence by one ULP. The
experiment rebuilt it around the exact control arithmetic:

- emulate `softmax_f32`'s `min(256,n_exp)` reduction width inside the 256-thread
  fused workgroup;
- preserve `-1e30`, `expf`, reduction-tree, and direct-division order;
- run the live optimized top-k comparison, invalidation, tie, and k=0..7
  renormalization order unchanged;
- keep the router-probability overwrite contract intact.

The gfx1201 kernel compiled at 31 VGPR, 1,088 B LDS, zero scratch, and zero
spills. Its retained-PM4 pointer effects and exact 32-byte kernarg ABI were
described explicitly.

## Gates

- `hiptrx`, gfx1201, automatic clocks.
- Qwen 3.6 35B A3B MQ4R, Q8 KV, MTP off.
- Fifteen consecutive PM4 positions: exact logits/KV/recurrent/blob parity.
- Valid public-HSA loader contract for all 25 captured kernels.
- Stable 793-launch, 25-kernel tape.
- Context 128, 100 measured tokens, ten warmups, 30 rows in both A/B orders.

## Product A/B

| Order | Control | Fused | Delta |
|---|---:|---:|---:|
| control then fused | 194.239 tok/s | 193.653 tok/s | -0.30% |
| fused then control | 193.476 tok/s | 194.334 tok/s | +0.44% |
| order-balanced mean | 193.858 tok/s | 193.994 tok/s | **+0.07%** |

The sign follows execution order. No eight-turn run was warranted because the
tg128 gate did not establish a reproducible win.

## Interpretation

Both control nodes are already one-workgroup kernels. Fusion removes 40 CP
dispatches but does not remove a material model-sized intermediate: router
probabilities are only `n_exp` floats per layer. The longer fused workgroup's
barrier lifetime offsets the launch saving. This separates tape size from token
latency: fewer retained packets remain desirable substrate, but not at the cost
of a larger kernel unless model traffic also disappears.

Artifacts on `hiptrx`:

```text
/home/kaden/.redline-work/hipfire-pm4-lean/.redline-work/
  router-fused-exact-shadow15-v2.json
  router-fused-control-30.json
  router-fused-candidate-30.json
  router-fused-reverse-candidate-30.json
  router-fused-reverse-control-30.json
```
