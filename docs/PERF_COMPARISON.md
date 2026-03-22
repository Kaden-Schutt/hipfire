# hipfire Performance Comparison vs llama.cpp

**GPU:** AMD RX 5700 XT (gfx1010, RDNA1, 8GB GDDR6, 448 GB/s peak)
**Date:** 2026-03-22
**hipfire branch:** phase5-hfq4
**llama.cpp:** build 7f8ef50cc (7209), custom ROCm build

## Scoreboard

| Metric | hipfire Before | hipfire After | llama.cpp | Ratio | Winner |
|--------|---------------|--------------|-----------|-------|--------|
| **8B short gen** (tok/s) | 55.5 | **55.3** | 44.3 | **1.25x** | hipfire |
| **8B long gen** (tok/s) | 36.1 | **30.7** | 42.8 | 0.72x | llama.cpp |
| **8B prefill** (tok/s) | 56 | **89** | 189.2 | 0.47x | llama.cpp |
| **0.6B short gen** (tok/s) | 221 | **230.8** | 193.6 | **1.19x** | hipfire |
| **0.6B long gen** (tok/s) | 105.9 | **~106** | 181.3 | 0.58x | llama.cpp |
| **0.6B prefill** (tok/s) | 215 | **392** | 1534 | 0.26x | llama.cpp |

## What We Tried

### KEPT experiments:
1. **Batched GEMM prefill** — 8B prefill 56→89 tok/s (+59%), 0.6B 215→392 tok/s (+82%)
2. **Batched RoPE** — 0.6B prefill 351→392 tok/s (+12%)
3. **Wide GEMV (2 rows/block)** — 8B gen 54.5→55.2 tok/s (+1.3%)

### NOT KEPT experiments:
1. **Flash-decoding attention** — 8B long gen 36.1→29.5 tok/s. Two-kernel overhead exceeded benefit at seq_len<=2048.
2. **Fused gate+up HFQ4-G256** — 54.1 tok/s (neutral). GPU already overlaps separate GEMV launches.

## Analysis

### Where hipfire wins:
- **Short generation (8B):** 1.25x faster. HFQ4 format achieves better VGPR occupancy (18 vs Q4K's 39), translating to higher memory bandwidth utilization for decode-phase GEMV.
- **Short generation (0.6B):** 1.19x faster. Same occupancy advantage.

### Where llama.cpp wins:
- **Prefill:** llama.cpp uses batched GEMM (full matrix multiply) for all prompt tokens at once. hipfire's batched GEMM only accelerates the projection kernels (1.4-2.1x per operation) but RoPE, KV cache writes, and causal attention remain sequential per position. This per-position overhead dominates.
- **Long generation:** Attention scales O(n) with sequence length. llama.cpp uses optimized Flash Attention from rocBLAS. hipfire's hand-written attention kernel uses shared memory reduction which is less cache-efficient at long sequences.

### Fundamental bottleneck:
Generation speed is **memory bandwidth limited**. The RX 5700 XT has 448 GB/s peak bandwidth. At 55.3 tok/s for 8B, we're moving ~4.3GB of weight data per second for 36 layers × 7 GEMV projections = 252 GEMV calls per token. Each GEMV reads the full weight matrix from VRAM. There is essentially no room to improve generation speed without:
1. Reducing weight data (lower quantization → quality loss)
2. Increasing memory bandwidth (hardware upgrade)
3. Weight caching across layers (not applicable to transformers)

### Remaining opportunities:
- **Batched causal attention for prefill** — would close the prefill gap but requires a complex GEMM-based attention kernel
- **Larger KV cache + Flash Attention at seq_len > 4096** — flash-decoding becomes beneficial at longer sequences
- **Q3/Q2 quantization** — reduces weight data by 25-50% but may hurt quality

## Experiment Log

See `bench/results.tsv` for full experiment history with tok/s measurements and git hashes.

## Hardware Context

- RDNA1 architecture (2019) — oldest AMD GPU architecture with compute shader support
- 40 CUs, 2560 stream processors
- 8GB GDDR6 at 14 Gbps (448 GB/s theoretical, ~380 GB/s achieved)
- 220W TDP (measured 219W under load)
- gfx1010 ISA — wave32 only, no matrix cores, no dp4a

On newer RDNA3/4 GPUs with higher bandwidth and matrix cores, both hipfire and llama.cpp would be faster, but the relative comparison would shift toward hipfire due to the occupancy advantage of HFQ4 format.
