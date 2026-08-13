# Residual STAGE content cosine (L0, L2)

SPDX-License-Identifier: Apache-2.0

## Why

Residual CONTENT cosine falls with depth (first cos<0.999 at L2 pos64).
This bisects **within** L0 and L2 which sub-stage first introduces direction error.

## Floor / domains

- Ref: bf16 Block.forward (model.py), stages dumped as f32.
- Parent: F32 residual path; public HC/attn/moe helpers; dump f32 LE.
- Positions: 0,1,64,400,448,512,800,1023 (seq=1024).
- Stream stages shape [n_pos, 4096]; HC stages [n_pos, 16384] (=4×4096).
- Embed residual floor (separate residual_content run): cos=1, rel_l2=0.
- bf16 unit roundoff ~3.9e-3; pure-f32 identity floor 0.

## Per-stage ALL-position summary

| L | stage | cosine | rel_l2 | norm_ratio | note |
|--:|------:|-------:|-------:|-----------:|------|
| 0 | hc_pre_attn | 1.000000 | 9.5296e-07 | 1.000001 | ok |
| 0 | attn_norm | 0.999999 | 1.6681e-03 | 1.000018 | ok |
| 0 | attn_out | 0.999302 | 3.7424e-02 | 1.001363 | mild |
| 0 | hc_post_attn | 0.999851 | 1.7262e-02 | 1.000134 | mild |
| 0 | hc_pre_ffn | 0.999677 | 2.5438e-02 | 0.999101 | mild |
| 0 | ffn_norm | 0.999520 | 3.0996e-02 | 1.000021 | mild |
| 0 | moe_out | 0.999678 | 2.5482e-02 | 1.001973 | mild |
| 0 | hc_post_ffn | 0.999694 | 2.4748e-02 | 0.999689 | mild |
| 2 | hc_pre_attn | 0.999568 | 2.9392e-02 | 0.999611 | mild |
| 2 | attn_norm | 0.999282 | 3.7891e-02 | 1.000242 | mild |
| 2 | attn_out | 0.997999 | 6.3242e-02 | 0.999557 | **direction drift** |
| 2 | hc_post_attn | 0.999542 | 3.0266e-02 | 1.000368 | mild |
| 2 | hc_pre_ffn | 0.998706 | 5.1016e-02 | 1.002715 | **direction drift** |
| 2 | ffn_norm | 0.998714 | 5.0724e-02 | 1.000023 | **direction drift** |
| 2 | moe_out | 0.997513 | 7.0604e-02 | 1.001569 | **direction drift** |
| 2 | hc_post_ffn | 0.999492 | 3.1899e-02 | 1.001081 | mild |

## Verdict

**First ALL-pos cosine < 0.999: L2 `attn_out` cos=0.997999 rel_l2=6.3242e-02 nr=0.999557.**

Interpretation:

1. **L0 hc_pre_attn is identity-level** (cos≈1.0, rel≈1e-6) — HC pre on L0 is clean.
2. **L0 attn_out** is the first mild direction hit (ALL cos≈0.9993, nr≈1.001) —
   attention path begins the drift; HC post dilutes it back into the residual.
3. **L2 attn_out** is the first stage below 0.999 (ALL cos≈0.9980, nr≈1.000) —
   pure **direction** error in attention (not scale).
4. **L2 moe_out** is worse still (ALL cos≈0.9975, nr≈1.002) — MoE compounds.
5. Residual HC after each half (`hc_post_*`) partially **recovers** cosine vs the
   stream-only stages because the multi-stream residual mix dilutes stream error.
   That is why layer-exit residual CONTENT still looks ~0.9995 at L2 while
   internal attn_out already sits at 0.998.

## Worst single positions (cos < 0.999)

| L | stage | pos | cosine | rel_l2 | nr |
|--:|------:|----:|-------:|-------:|---:|
| 2 | moe_out | 64 | 0.994370 | 1.0602e-01 | 0.998062 |
| 2 | moe_out | 1 | 0.996299 | 8.9008e-02 | 1.019420 |
| 2 | attn_out | 1 | 0.997121 | 7.5842e-02 | 0.998454 |
| 2 | moe_out | 448 | 0.997124 | 7.5833e-02 | 0.999803 |
| 2 | moe_out | 512 | 0.997211 | 7.4641e-02 | 0.996732 |
| 2 | attn_out | 0 | 0.997236 | 7.4820e-02 | 1.006087 |
| 2 | attn_out | 64 | 0.997692 | 6.7915e-02 | 0.999172 |
| 2 | moe_out | 1023 | 0.997750 | 6.8508e-02 | 1.011816 |
| 2 | ffn_norm | 64 | 0.997835 | 6.5804e-02 | 0.999947 |
| 2 | hc_pre_ffn | 64 | 0.997836 | 6.5766e-02 | 0.996184 |
| 2 | moe_out | 800 | 0.998020 | 6.2921e-02 | 0.999853 |
| 2 | moe_out | 0 | 0.998073 | 6.3236e-02 | 0.985918 |
| 2 | attn_out | 512 | 0.998243 | 6.0586e-02 | 1.010865 |
| 2 | attn_out | 448 | 0.998317 | 5.8027e-02 | 0.996227 |
| 2 | attn_out | 800 | 0.998341 | 5.7645e-02 | 0.995688 |
| 2 | ffn_norm | 1 | 0.998366 | 5.7168e-02 | 0.999996 |
| 2 | attn_out | 1023 | 0.998369 | 5.7103e-02 | 0.999241 |
| 2 | hc_pre_ffn | 1 | 0.998369 | 5.7176e-02 | 1.001415 |
| 2 | hc_pre_ffn | 800 | 0.998416 | 5.6551e-02 | 0.992746 |
| 2 | ffn_norm | 800 | 0.998416 | 5.6293e-02 | 1.000227 |
| 2 | moe_out | 400 | 0.998518 | 5.4509e-02 | 0.995595 |
| 2 | hc_pre_attn | 64 | 0.998563 | 5.3984e-02 | 1.005093 |
| 2 | attn_out | 400 | 0.998590 | 5.3267e-02 | 0.994174 |
| 2 | attn_norm | 64 | 0.998616 | 5.2606e-02 | 0.999612 |
| 2 | ffn_norm | 1023 | 0.998642 | 5.2107e-02 | 0.999985 |

## Next cut

Stage content points at **attention output direction** first (L0 mild, L2 clear),
then **MoE output**. Recommended next probes (do not re-open eliminated domains):

1. L2 `attn_out` sub-stages: Q/K/V, RoPE apply, scores, SWA gather, `wo` —
   content cosine per tile (ParentAttn already has stage L2 hooks).
2. L2 `moe_out`: route indices identity vs ref, then expert GEMM direction.
3. Confirm L0 attn_out mild error is the seed that L2 amplifies.
