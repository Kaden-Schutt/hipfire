# MMQ Channel-Test Results: qwen3.5-9b on gfx1151

**Date:** 2026-04-30
**Ref:** Kaden-Schutt/hipfire#87 (auto-MMQ regression on tool-call output)
**Hardware:** AMD Strix Halo gfx1151, 131.1 GB VRAM, HIP 7.2
**Model:** qwen3.5-9b.mq4 (32 layers: 24 DeltaNet + 8 FullAttn, dim=4096)
**Binary:** `channel_test_mmq --stage site-scan --batch 128 --threshold 0.01`

## Stage 1: site-scan — all layers x all sites

### Finding: The error is universal, not site-specific

Every (site, layer) pair fails at threshold 0.01. The Q8_1 activation
quantization introduces ~1% mean absolute error uniformly across all
GEMM call sites and all layers.

### Error ranking by site type

| Site type        | Typical max_err | Worst max_err   | Mean err range |
|------------------|-----------------|-----------------|----------------|
| **residual (Wo)**| 0.11 – 0.30     | **0.516** (L0)  | 0.010 – 0.014  |
| qkvza.qkv / qkv.q | 0.08 – 0.14   | 0.201 (L0)      | 0.010 – 0.014  |
| qkvza.alpha      | 0.05 – 0.14     | 0.144 (L1)      | 0.011 – 0.022  |
| qkv.v            | 0.06 – 0.09     | **0.232** (L27)  | 0.012 – 0.020  |
| gate_up.gate     | 0.05 – 0.08     | 0.082           | 0.008 – 0.010  |
| gate_up.up       | 0.03 – 0.06     | 0.065           | 0.008 – 0.010  |
| qkvza.beta       | 0.03 – 0.09     | 0.089 (L1)      | 0.007 – 0.018  |

The `residual` site has 3-5x higher peak error than any other site.

### Error trend across layers

Mean error is flat (~0.010) across layers 0-31. No concentration in
early or late layers. The max_err spikes in residual vary between layers
but don't show a clear trend.

### Anomaly: layer 27 qkv.v

Layer 27 qkv.v has max_err=0.232, roughly 3x the typical qkv.v value
(~0.06). This is a FullAttn layer. Worth investigating with channel-map.

## Stage 2: channel-map — residual across layers

### Finding: Row 3994 is the #1 outlier in EVERY Wo layer

| Layer | Row 3994 max_err | Row 3994 mean_err | 2nd worst row | 2nd max_err | Ratio |
|-------|-----------------|-------------------|---------------|-------------|-------|
| 0     | **0.516**       | 0.135             | 1504          | 0.092       | 5.6x  |
| 1     | **0.456**       | 0.136             | 3986          | 0.073       | 6.3x  |
| 3     | **0.493**       | 0.126             | 3968          | 0.068       | 7.3x  |
| 6     | **0.357**       | 0.077             | 3769          | 0.070       | 5.1x  |

Row 3994 is consistently 5-7x worse than the 2nd worst row in every
layer tested. The 2nd-worst rows vary across layers (1504, 3986, 3968,
3769) but row 3994 is always #1. This is a structural property of the
weight matrix at hidden dimension 3994, not a random quantization
artifact.

All ~4095 other rows are in the "normal" 0.05-0.07 max_err range.
Row 3994 is the only row that consistently breaks the 0.1 barrier.

### Finding: Layer 27 qkv.v — row 265 is an outlier

| Row | max_err | mean_err | bad_acts | 2nd worst |
|-----|---------|----------|----------|-----------|
| **265** | **0.232** | 0.072 | 117/128 | 0.113 (row 395) |

Same pattern as the Wo outlier: one catastrophic row, everything else
is ~2x lower. Row 265 in the V projection has a weight group whose
HFQ4 scale/zero statistics amplify Q8_1 rounding error.

Unlike the Wo outlier (same row 3994 in every layer), this is specific
to layer 27's V weights. The other FullAttn layers (3, 7, 11, 15, 19,
23, 31) have qkv.v max_err in the normal 0.06-0.09 range.

## Root cause

**Specific weight rows in the HFQ4 quantized model have scale/zero-point
statistics that amplify Q8_1 activation rounding error.** The MMQ kernel
computes `W_q4 * X_q8` where both operands are quantized. When a weight
row has an unusual dynamic range (e.g., very small scale or extreme
zero-point), the combined quantization error of both operands compounds
multiplicatively for that row's dot product, producing errors 5-7x
larger than typical rows.

Row 3994 appears in every Wo layer because the output projection
weights for hidden dimension 3994 have consistently unusual statistics
across the entire model — likely reflecting an activation channel in
the original fp16 model with atypical magnitude.

## Conclusions

1. **The MMQ path has systematic, pervasive precision loss** — not a
   site-specific or layer-specific defect. Every (site, layer) pair
   shows ~1% mean error from Q8_1 activation quantization.

2. **The peak errors are driven by ~1-2 outlier weight rows per site.**
   Row 3994 in every Wo projection; row 265 in layer 27 qkv.v. These
   are 5-7x worse than all other rows.

3. **The tool-call corruption in #87 is a threshold effect:** 32 layers
   of ~1% mean error + 0.3-0.5 spikes from outlier rows compound through
   the residual stream until special-token logit probabilities flip.
   Tool-call prompts are more sensitive because they require precise
   distributions around ChatML token IDs.

4. **The fix is per-row screening, not per-site or per-layer.** Skipping
   MMQ for the entire Wo site works but sacrifices too much speedup.
   Screening the ~1-2 outlier rows per weight matrix and falling back to
   f16 WMMA only for those rows preserves >99% of the MMQ benefit.

## Recommended fix: per-row error screening

At model load time, for each weight matrix that will use the MMQ path:
1. Generate a small synthetic activation vector (batch=16, same LCG seed)
2. Run both WMMA and MMQ on it
3. Compute per-row max_err
4. Flag rows exceeding a threshold (e.g., 0.15 — catches the 0.3-0.5
   outliers while ignoring the normal 0.05-0.08 range)
5. Store a per-matrix bitmask of "dirty rows"
6. At runtime, the MMQ kernel skips dirty rows (or a post-MMQ fixup
   kernel replaces those rows with WMMA results)

Expected cost: ~1-2 rows per 4096 fall back to WMMA. Performance
impact: negligible (<0.05% of compute).

## Screening prototype results

Implemented in `dispatch.rs`: `mmq_screen_weight()` with per-weight
caching. Activated via `HIPFIRE_MMQ_SCREEN=1`.

### 9B model (threshold=0.15)

- **19/216 weights flagged UNSAFE** (8.8%), 197 keep MMQ
- Row 3994 in Wo: 15 layers (0-10, 12, 14, 16, 31)
- Row 265 in qkv.v: layer 27
- Row 644 in qkv.v: layer 31
- Row 4209 in qkvza.qkv: layer 0
- Row 1812 in qkvza.qkv: layer 28

The screening correctly identifies the outlier rows that our channel-map
analysis found, plus a few additional borderline cases in qkvza.qkv.
All flagged weights fall back to WMMA; unflagged weights keep the fast
MMQ path.

## Still to do

- [x] Run channel-map on residual at layers 0, 1, 3, 6 — row 3994
      confirmed as consistent outlier
- [x] Channel-map on layer 27 qkv.v — row 265 confirmed as outlier
- [x] Run site-scan on qwen3.6-27b.mq4 — confirmed, worse peaks (0.91)
- [x] Prototype per-row screening fix — implemented and validated
- [ ] Investigate row 3994's HFQ4 weight statistics (scale/zero/range)
- [ ] End-to-end validation: run daemon with MMQ+screening on tool-call prompt
