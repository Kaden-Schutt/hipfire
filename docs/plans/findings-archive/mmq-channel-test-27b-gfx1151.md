# MMQ Channel-Test Results: qwen3.6-27b on gfx1151

**Date:** 2026-04-30
**Ref:** Kaden-Schutt/hipfire#87 (auto-MMQ regression on tool-call output)
**Hardware:** AMD Strix Halo gfx1151, 131.1 GB VRAM, HIP 7.2
**Model:** qwen3.6-27b.mq4 (64 layers: 48 DeltaNet + 16 FullAttn, dim=5120)
**Binary:** `channel_test_mmq --batch 128 --threshold 0.01`

## Stage 1: site-scan — residual site across all 64 layers

All 432 (site, layer) pairs fail at threshold 0.01 — same universal
error pattern as 9B.

### Residual (Wo) max_err by layer (all 64 layers)

Top 10 worst:

| Layer | max_err   | mean_err |
|-------|-----------|----------|
| 3     | **0.910** | 0.013    |
| 0     | **0.887** | 0.012    |
| 1     | 0.635     | 0.013    |
| 8     | 0.541     | 0.013    |
| 7     | 0.540     | 0.012    |
| 5     | 0.451     | 0.013    |
| 2     | 0.432     | 0.013    |
| 11    | 0.420     | 0.012    |
| 9     | 0.415     | 0.013    |
| 4     | 0.406     | 0.013    |

Error decreases with depth — early layers (0-11) have max_err 0.4-0.9,
later layers (50-63) drop to 0.09-0.19. Mean error is flat (~0.013).

### Comparison with 9B

| Metric          | 9B (32 layers) | 27B (64 layers) |
|-----------------|----------------|-----------------|
| Worst max_err   | 0.516 (L0)     | **0.910** (L3)  |
| Mean err range  | 0.010-0.014    | 0.012-0.016     |
| Outlier row     | 3994           | **3994**         |

The 27B has ~1.7x higher peak error than 9B, consistent with its larger
hidden dim (5120 vs 4096) producing longer dot products with more
accumulated Q8_1 rounding.

## Stage 2: channel-map — residual, layer 0

### Row 3994 confirmed as outlier on 27B

| Row  | max_err   | mean_err  | bad_acts | Ratio vs 2nd |
|------|-----------|-----------|----------|--------------|
| **3994** | **0.887** | **0.234** | **125/128** | **8.7x** |
| 1846 | 0.102     | 0.022     | 97/128   | —            |
| 1332 | 0.098     | 0.023     | 96/128   | —            |
| 3986 | 0.095     | 0.025     | 101/128  | —            |

Row 3994 is 8.7x worse than 2nd place (vs 5.6x on 9B). Same hidden
dimension, same structural defect, worse magnitude on the larger model.

The 2nd-worst rows differ between 9B and 27B (1846 vs 1504) because the
weight matrices are different models. But row 3994 is consistent across
both — this is a property of the Qwen model family's weight distribution
at hidden dimension 3994.

## Conclusion

The 27B canonical reproducer confirms the 9B findings with stronger
signal. Row 3994 in the Wo projection is the primary corruption source,
and the per-row screening fix is the correct approach.
