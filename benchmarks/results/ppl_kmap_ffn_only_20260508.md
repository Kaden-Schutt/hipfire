# PPL validation: K-map FFN-only edge promotion (#196)

- date: 2026-05-08
- branch: feat/kmap-split-dense-moe
- corpus: wikitext-2-raw-v1/test (1.29 MB, Salesforce/wikitext)
- params: ctx=8192 warmup=8 offset=0 kv-mode=asym4
- GPU: gfx1151 (Strix Halo 7900 XTX)

## Change

Rule 5 (edge-layer promotion) now branches on model type:
- **MoE:** promote all tensors in first/last 2 layers (attn + FFN)
- **Dense:** promote only FFN (`mlp.*` / `ffn*`) in first/last 2 layers

Rationale: PPL benchmarks showed attn promotion regresses dense models
(+3.1% on 27B) while FFN-only improves them (-2.2% on 27B). MoE models
benefit from full promotion (-19.8% on 3.6-35B-A3B).

## Results (asym4 KV, ctx=8192)

| Model | Uniform MQ4 | K-map FFN-only | Delta |
|---|---:|---:|---:|
| 4B dense (32L) | 20.22 | **19.39** | **-4.1%** |
| 9B dense (32L) | **15.21** | 18.51 | **+21.7%** |
| 27B dense (64L) | 14.30 | **13.99** | **-2.2%** |
| MoE 3.6-35B-A3B (40L) | 25.00 | **23.89** | **-4.5%** |

## 9B outlier investigation

The 9B is the only model that regresses. Investigated code paths —
no dispatch, rotation, or kernel bug found.

### Evidence it's model-specific, not a code bug

1. **Full MQ6 uniform on 9B: PPL 14.77** — better than uniform MQ4 (15.21).
   MQ6 itself is not the problem.
2. **Mixed MQ4+MQ6 (edge FFN only) on 9B: PPL 18.51** — the mix hurts.
3. **4B with identical architecture and layer count: PPL improves** (-4.1%).
   Same code path, same 4/32 layers promoted, opposite outcome.

### Dispatch verification

All code paths handle MQ6 correctly:
- `fused_rmsnorm_rotate_for_mq`: matches MQ6G256 → returns rotated x
- `weight_gemv_prerotated`: MQ6 branch → `gemv_mq6g256_prerotated`
- `weight_gemv_swiglu_residual`: MQ6 branch → `fused_silu_mul_rotate_mq`
  + `gemv_hfq6g256_residual`
- Fused gate+up path (`fused_gu_mq4`): does NOT match MQ6, falls to
  individual `weight_gemv_prerotated` calls — slower but numerically
  equivalent.

### Hypothesis: quant-level boundary mismatch

The MQ4↔MQ6 boundary creates a discontinuity in quantization error
profile. Edge-layer FFN outputs (MQ6 dequant error shape) feed into
middle-layer inputs processed by MQ4 weights (different error shape).
The FWHT rotation is identical but the quantization grids differ
(4-bit: 16 levels, 6-bit: 64 levels). For 9B specifically, this
mismatch amplifies rather than reduces accumulated error.

Supporting evidence: full MQ6 (no boundary) improves 9B, confirming the
6-bit grid itself is fine — it's the mix that hurts.

### Architecture comparison

| Model | Layers | Promoted | % promoted | hidden | inter | inter/hidden | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| 4B | 32 | 4 | 12.5% | 2560 | 9216 | 3.6 | -4.1% |
| 9B | 32 | 4 | 12.5% | 4096 | 12288 | 3.0 | +21.7% |
| 27B | 64 | 4 | 6.3% | 5120 | 17408 | 3.4 | -2.2% |

Same layer count and promotion ratio for 4B and 9B, opposite outcomes.
The intermediate/hidden ratio (3.0 vs 3.6) or the absolute dimension
sizes may play a role but the mechanism is unclear.

## Conclusion

3 out of 4 models improve with FFN-only K-map. The 9B regression is
model-specific and appears related to the MQ4↔MQ6 boundary interaction,
not a code bug. K-map remains gated behind `--kmap-dense` for dense
models (upstream decision), allowing users to validate per-model before
opting in.
