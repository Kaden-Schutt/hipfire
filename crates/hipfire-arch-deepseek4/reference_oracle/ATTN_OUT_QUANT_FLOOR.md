# L0 attn_out quantization floor (ref vs ref)

SPDX-License-Identifier: Apache-2.0

## Question

Is parent-vs-ref L0 `attn_out` cosine 0.9993 a real defect, or the
fp8/act_quant noise floor of the attention path?

## Method (self-calibrating, port not involved)

Same L0 Attention, same `attn_in` (post hc_pre+attn_norm), same weights,
run twice through the reference harness only:

1. **FP8 mode** — default `kernel_shim` (`act_quant` + `fp8_gemm` via `linear()`)
2. **Exact mode** — `act_quant(..., inplace=True)` no-op; `linear()` dequants
   FP8/FP4 weights to f32 and matmuls without activation quant

L0 `compress_ratio = 0` (pure SWA, no compressor/indexer).

## Floor results

| comparison | cosine | rel_l2 | norm_ratio |
|------------|-------:|-------:|-----------:|
| ref-fp8 vs ref-f32 **attn_out** ALL | 0.99954490 | 3.021538e-02 | 0.997819 |
| ref-fp8 vs ref-f32 **wq_a Linear** ALL | 0.99985224 | 1.720097e-02 | 0.999241 |
| parent vs ref **attn_out** ALL (prior) | 0.99930200 | 3.742400e-02 | — |

Per-position ref-fp8 vs ref-f32 attn_out:

| pos | cosine | rel_l2 | nr |
|----:|-------:|-------:|---:|
| 0 | 0.99959496 | 2.847987e-02 | 0.998506 |
| 1 | 0.99945509 | 3.306890e-02 | 1.001463 |
| 64 | 0.99950733 | 3.192129e-02 | 0.993688 |
| 400 | 0.99950924 | 3.143412e-02 | 0.996896 |
| 448 | 0.99951758 | 3.107494e-02 | 0.998493 |
| 512 | 0.99955137 | 3.002742e-02 | 0.997409 |
| 800 | 0.99965450 | 2.646862e-02 | 0.996538 |
| 1023 | 0.99960775 | 2.800799e-02 | 0.999924 |

## Cross-check against known Linear floor

wq_a alone: rel_l2=1.7201e-02. Prior harness fp8-Linear floor was ~1.55e-2.
Match confirms exact-mode bypass and fp8 sim are both live.

## Verdict

**AT_FLOOR**

- floor cos=0.999545, parent-vs-ref cos=0.999302
- rel ratio parent/floor = 1.24× (not ≫1; not ~100×)
- quadrature excess rel ≈ 2.2081e-02 after removing floor in RSS

**STOP — do not bisect L0 attn internals; defect is elsewhere**

Main's gate: if ref-fp8 vs ref-f32 ≈ 0.9993 then parent 0.9993 is floor.
Measured floor is 0.999545 — same ballpark as 0.9993, **not** 0.99999.
The 22× jump attn_norm→attn_out in parent-vs-ref is the multi-GEMM quant
envelope (several fp8 Linears + kv act_quant + sinkhorn-free SWA), not a
porting bug signature.

## Implications

1. **Divergence still starts at L0** (ratio-0 pure SWA) in residual CONTENT,
   but the L0 `attn_out` stage delta is **not** actionable as a defect — it sits
   on the stage's own quant floor.
2. Compressed-KV / joint-softmax remain killed for the *onset* (L0 is ratio-0),
   yet we must **not** spend the next cut inside L0 attention internals.
3. Deeper residual cosine fall (L10–L42 mean cos 0.997→0.992) still needs an
   explanation. Candidates now:
   - per-layer floor noise accumulating coherently in residual direction
   - a later-stage real defect (MoE, HC, head) that *is* above its floor
   - domain mismatch that only shows once errors compound
4. Next measurement should establish floors for **L2 moe_out** and **layer-exit
   residual** the same way (ref-fp8 vs ref-f32), not bisect L0 Q/K/V/RoPE/scores.

## Assets

- `attn_out_quant_floor.py`
- `artifacts/attn_out_quant_floor/attn_out_quant_floor.json`
