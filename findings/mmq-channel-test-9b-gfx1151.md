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

## Stage 2: channel-map — residual, layer 0

### Finding: Row 3994 is a catastrophic outlier

| Row  | max_err   | mean_err  | bad_acts (of 128) |
|------|-----------|-----------|-------------------|
| **3994** | **0.516** | **0.135** | **123/128**    |
| 1504 | 0.092     | 0.022     | 92/128            |
| 3092 | 0.085     | 0.019     | 85/128            |
| 3986 | 0.084     | 0.020     | 85/128            |

Row 3994 is 5.6x worse than the next worst row. All 4096 rows exceed
the 0.01 threshold, but 4095 of them are in the "normal" ~0.02 mean
error range. Row 3994 is the only structural outlier:

- 0.516 max error (vs ~0.08 for 2nd worst)
- 0.135 mean error (vs ~0.02 for 2nd worst)
- 123/128 batch elements exceed threshold (vs ~85 for 2nd worst)

### Interpretation

Row 3994 in the Wo projection of layer 0 maps to hidden dimension 3994
in the residual stream. A 0.5 absolute error here at layer 0 propagates
through all 31 subsequent layers, compounding through attention and FFN.

The Q8_1 quantization of the activation vector (the `ensure_q8_1_mmq_x`
step) loses precision specifically for the dot product involving this
weight row. This is likely caused by the HFQ4 weight statistics for row
3994 having a scale/zero-point combination that amplifies the Q8_1
rounding error beyond what other rows experience.

## Conclusions

1. **The MMQ path has systematic, pervasive precision loss** — not a
   site-specific or layer-specific defect. Every (site, layer) pair
   shows ~1% mean error from Q8_1 activation quantization.

2. **The residual (Wo) site has the highest peak errors** (up to 0.5)
   because of isolated outlier rows in the weight matrix. These spikes
   compound through the residual stream across layers.

3. **The tool-call corruption in #87 is likely a threshold effect:**
   the cumulative error from 32 layers of ~1% mean error plus occasional
   0.5 spikes eventually flips special-token logits. Tool-call prompts
   are more sensitive because they require precise probability
   distributions around ChatML token IDs.

## Recommended fix approaches

1. **Skip MMQ for residual (Wo) site only** — use f16 WMMA for Wo,
   keep MMQ for QKV and gate/up. Wo is the single-matrix call
   (not fused), so falling back to WMMA is cheap. This eliminates the
   worst spikes while preserving most of the prefill speedup.

2. **Per-row error screening** — at model load time, run a quick
   synthetic comparison for each Wo row and flag rows with max_err
   above a threshold. Use mixed-precision: MMQ for clean rows, WMMA
   for dirty rows. More complex but preserves more of the speedup.

3. **Raise batch-size threshold** — if MMQ only activates at batch >= 256
   instead of >= 128, fewer prefill chunks hit the path and the
   corruption probability drops. Least invasive but gives up the most
   performance.

## Still to do

- [ ] Run channel-map on residual at other high-max_err layers (1, 3, 6)
      to check if the same rows are consistently bad across layers
- [ ] Run site-scan on qwen3.6-27b.mq4 (the canonical #87 reproducer)
- [ ] Channel-map on layer 27 qkv.v anomaly
- [ ] Investigate row 3994's weight statistics (HFQ4 scale/zero/range)
