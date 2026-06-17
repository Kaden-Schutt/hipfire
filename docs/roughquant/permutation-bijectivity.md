# RoughQuant — permutation bijectivity (the 5 main permutations)

Verifier: `scripts/roughquant_permute_verify.py` (no GPU; synthetic per-block
forwards on the real Qwen3.5-0.8B arch dims, fp64, ~machine-zero tolerance).
A permutation is "free" iff its sub-computation is invariant under apply+propagate.

## Result

| # | permutation | free / function-preserving? | propagation required |
|---|---|---|---|
| 1 | hidden-dim (general inter-linear) | ✅ | producer rows + consumer cols |
| 2 | MLP neurons (SwiGLU) | ✅ | `gate`/`up` ROWS + `down` COLS (activation elementwise) |
| 3 | attention heads (GQA) | ✅ | Q/K/V head-blocks + `o_proj` input-blocks, **within GQA groups** (a KV head and its query-head group move together) |
| 4 | per-head dims | ⚠️ only **without RoPE** | Q/K identical perm (dot-product invariant), V + `o_proj` cols consistent — BUT see RoPE constraint |
| 5 | residual stream (global) | ✅ | embed rows + EVERY reader input-cols + EVERY writer output-rows + RMSNorm γ + lm_head + tied embed |

## The #4 / RoPE constraint (the key finding)

Per-head-dim permutation is bijective for the bare Q·K dot product, but **NOT under
RoPE**: RoPE assigns frequency θ_i to dim-pair `(i, i+hd/2)`, so permuting head
dims changes which dim carries which frequency and breaks the *relative-position*
dot product (`max|Δ|` ≈ 7.7, not 0). It is free ONLY if the permutation preserves
RoPE structure — i.e. permute the `hd/2` frequency-pairs as units (keeping each
`(i, i+hd/2)` together), not arbitrary dims. (QK-norm is per-dim and permutes
consistently, so it's not the blocker; RoPE is.) Verifier note: the effect is
invisible at equal q/k positions — must test `pos_q ≠ pos_k`.

## Arch specifics (Qwen3.5-0.8B hybrid)

- d_model 1024; MLP intermediate 3584; full-attn = 16 query / 2 KV heads (GQA
  8:1), head_dim 128 (value path), QK-norm + RoPE.
- #3/#4 apply only to the 6 full-attention layers; the 18 linear-attn (DeltaNet)
  layers have their own `in_proj_{qkv,z,a,b}`/`out_proj` (residual readers/writers
  for #5, but no head structure for #3/#4).
- #2 applies to all 24 layers; #1/#5 are global.

## Status

- **Verified** (this doc): which permutations are free + their propagation spec.
- **Production machinery** (next): Rust appliers that permute a loaded model's
  weights per type, enforcing the GQA grouping (#3) and the RoPE-pair constraint
  (#4), with the #5 global propagation across embed/norms/all-projections/lm_head.
  Primary use case: **gather-free RoughQuant correction**. The real packed format's
  protected-channel correction otherwise needs a runtime gather `x[S]` (readers) +
  scatter `y[S]` (writers); a #5 residual permutation that makes S a contiguous
  block turns those into pointer-offset slices — eliminating the
  `rq_gather_f32`/`rq_scatter_add_f32` launches + index buffers. #5 is runtime-free
  (baked offline), so the only residual cost is the correction GEMV's ~5% MACs.
  (NB: framing is "gather-free", NOT "foldable rotation" — channel protection needs
  no fold; see docs/roughquant/phase3-real-format-scope.md.)
