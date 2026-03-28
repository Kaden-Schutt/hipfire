# Mathematical Synthesis: Novel RDNA1 Optimizations

## The Path to 200+ tok/s on Qwen3-8B

### Key Insight: EAGLE-3 alone gets you to 200 tok/s

No kernel changes needed. The existing batched GEMM (BATCH_TILE=8) handles
verification. Because weight reads dominate (4.29 GB/token), verifying
batch=7 draft tokens costs nearly the same as batch=1.

```
EAGLE-3 with acceptance=3.2 (conservative):
  Weight reads: 4.29 GB (same for batch=1 and batch=7)
  Cycle time: 4.29/0.282 + 0.7ms draft/sync = 15.9ms
  Effective: 3.2 / 0.0159 = 201 tok/s
```

### Three Novel Cross-Domain Insights

**1. HFQ2 has identical VGPR pressure to HFQ4 on RDNA1**

The v_cvt_f32_ubyte0/1/2/3 instructions convert byte→float in one cycle.
For 2-bit: load dword, shift+mask to isolate dibits in byte positions,
then v_cvt_f32_ubyte extracts each. Result: ~18 VGPRs (same as HFQ4).

This means 2-bit weights are "free" on RDNA1 — no occupancy loss.
Model drops from 4.4GB to 2.2GB → ~120 tok/s without speculative decode.

**2. Full x-vector fits in LDS (dim=4096 = 16KB)**

The entire input activation vector for Qwen3-8B fits in 64KB LDS.
Pre-loading x into LDS saves 19/20 redundant global reads per CU
(20 waves share one 16KB copy). Estimated: ~12% of forward time saved.

**3. Turbo2 + Flash Decoding = super-linear synergy**

turbo2 KV is 3.5x smaller per position. Flash Decoding tile of 256
turbo2 positions = 9.2KB (fits L1). FP32 tile of 64 = 64KB (spills L1).
4x larger tiles = 4x fewer reduction passes. Neither agent alone saw this.

## Performance Projections

| Configuration | tok/s | vs baseline |
|--------------|-------|-------------|
| Current (HFQ4 + Q8 KV) | 59.8 | — |
| + x-vector LDS staging | ~67 | +12% |
| HFQ2 FFN + Q4 attn | ~96 | +60% |
| EAGLE-3 (current format) | ~200 | +234% |
| HFQ2 FFN + EAGLE-3 | ~240 | +301% |
| Full stack (2-bit + EAGLE + Flash + fused) | ~255 | +326% |
| **Theoretical ceiling** | **311** | +420% |

## Implementation Priority (impact/effort)

1. **EAGLE-3 speculative decode** → 200 tok/s (2-3 weeks)
   Batched GEMM exists. Need: draft head weights, verification loop.

2. **x-vector LDS preload** → +12% (2 days)
   Load dim=4096 into LDS once per CU, all waves share.
   Zero VGPR cost, 16KB of 64KB LDS budget.

3. **HFQ2 with v_cvt_ubyte** → ~120 tok/s standalone (2 weeks)
   New quantization format + GEMV kernel. Same 18 VGPRs as HFQ4.
   With FWHT incoherence (W*H precomputed offline): quality parity.

4. **Flash Decoding** → 4-5x attention at 2K ctx (1 week)
   Tile=64 for Q8, tile=256 for turbo2. 31 VGPRs.

5. **Persistent Flash Decoding** → +5-8% at long ctx (3 days)
   Single-pass with intra-CU LDS handshake. Eliminates reduction kernel.

## Theoretical Ceiling Analysis

Minimum reads per token (2-bit FFN + 4-bit attention):
  Weights: 2.73 GB
  KV (turbo2, 2K): 0.05 GB
  Activations: 0.01 GB
  Total: 2.79 GB

At 282 GB/s (63%): 2.79/0.282 = 9.89ms → 101 tok/s (no speculation)
At 336 GB/s (75%, with kernel fusion): 8.30ms → 120 tok/s
With EAGLE-3 (acceptance=2.8): 2.8 / 0.009 = **311 tok/s**
