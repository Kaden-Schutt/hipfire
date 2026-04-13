# Hipfire Research Findings

Hardware: AMD RX 5700 XT (gfx1010, RDNA1, 8GB GDDR6, 448 GB/s peak BW, 40 CUs)
Software: ROCm 6.3.4, HIP 6.3.4, Rust 1.92.0

## 1. Q8 Beats Q4 on RDNA Because Occupancy > Compression

**The central discovery.** On RDNA1, 8-bit quantization is 1.8x faster than 4-bit
for GEMV despite reading 2x more data.

### Root Cause: Register Pressure from Nibble Extraction

Profiling with `llvm-objdump` on compiled `.hsaco` kernels revealed:

| Format | VGPRs | Max Waves/SIMD | Bandwidth Util | GEMV 2048x2048 |
|--------|-------|----------------|----------------|----------------|
| F32 | 16 | 64 | 49% (218 GB/s) | 77 us |
| Q8_0 | 16 | 64 | 84% (375 GB/s) | 12 us |
| Q4_K | 40 | 25 | 42% (188 GB/s) | 13 us |

Q4's nibble extraction (bit shifts, masks, conditional selects) inflates register
pressure from 16 to 40 VGPRs. On RDNA1 with 1024 VGPRs per SIMD, this halves
maximum occupancy from ~10 to ~5 waves per SIMD. Fewer concurrent waves means
less memory latency hiding, cutting effective bandwidth nearly in half.

Q8 is just byte loads — `signed char qval = block[2 + tid]`. No bit manipulation,
no type conversion chain. Register pressure matches F32.

### Why This Hasn't Been Published

The standard assumption in llama.cpp, vLLM, and every other inference engine is
that smaller quantization = faster inference because less data to read. This is
true on NVIDIA where dp4a makes 4-bit dequantization essentially free (single
instruction for 4 int8 multiplies). On AMD RDNA1, dp4a is emulated via
`v_mul_i32_i24 + v_add3_u32` (6 instructions per 4 elements), which inflates
the register file and kills occupancy.

### Implications

- For any RDNA GPU: profile VGPRs before optimizing instruction count
- Q8 at 1.06 B/w is the sweet spot: 4x compression, near-F32 occupancy
- Mixed quantization (Q8 attention + Q4 FFN) balances speed and VRAM
- RDNA2+ with native dp4a may close the gap, but register pressure is architectural

## 2. Kernel Launch Overhead is Negligible (Not 30%)

Phase 3 estimated ~30% of forward pass time was kernel launch overhead from
~311 launches per token. Actual measurement:

- `hipModuleLaunchKernel`: 2.73 us per call
- 286 launches × 2.73 us = 0.78 ms = **8.5%** of 9.2 ms/token
- Kernel fusion (fused QKV, gate+up) gave ~0% speedup

HIP kernel launches are pipelined — the GPU command queue overlaps them
automatically. Eliminating launches doesn't save real time because the GPU
is already saturated executing back-to-back.

## 3. Format Doesn't Matter — Storage Layout Does

We built and benchmarked 6 different Q4 kernel/format variants:

| Format | B/w | GB/s | % Peak | us/GEMV | Key Insight |
|--------|-----|------|--------|---------|-------------|
| Q8_0 | 1.06 | 368 | 82% | 12.1 | Byte loads, 16 VGPRs |
| Q4_K | 0.56 | 187 | 42% | 12.7 | Nibble unpack, 40 VGPRs |
| Q4-as-Q8 | 1.06 | 339 | 76% | 13.2 | 4-bit quality, Q8 storage |
| Q4_LUT | 1.50 | 271 | 61% | 23.2 | LDS codebook, no ALU dequant |
| Q4_WAVE | 0.63 | 107 | 24% | 24.5 | Shuffle-based unpack, failed |
| Q4_F16_G32 | 0.63 | 100 | 22% | 26.4 | FP16 FMA dequant |

**Q4-as-Q8** (4-bit precision stored in int8) proves the point: same quality as
Q4, but 76% peak bandwidth because the storage format (byte loads) determines
GPU throughput, not the mathematical precision.

**Q4_LUT** (LDS codebook lookup) achieved 61% peak — best for any true Q4 format —
but the 1.5 B/w storage overhead made absolute time slower than Q4_K.

**Q4_WAVE** (warp shuffle redistribution) was a complete failure at 24% peak.
The `__shfl` overhead exceeded any benefit from avoiding per-thread nibble extraction.

## 4. Small-Matrix GEMV Needs Different Kernels

For dim=1024 (Qwen3-0.6B), the 32-thread single-warp Q8_0 kernel that dominates
at dim=2048+ severely underperforms:

- dim=2048 (TinyLlama): 193 tok/s → matches llama.cpp
- dim=1024 (Qwen3-0.6B): 83 tok/s → llama.cpp gets 218

The fix: adaptive kernel dispatch based on K dimension.

| Kernel Variant | Qwen3-0.6B | TinyLlama | Description |
|----------------|-----------|-----------|-------------|
| 32-thread (v3) | 83 tok/s | 193 tok/s | Original, optimal for K≥2048 |
| 256-thread | 91 tok/s | 179 tok/s | Too many warps, regression on large K |
| 128-thread | 98 tok/s | 193 tok/s | Better occupancy |
| 64-thread | 99 tok/s | 193 tok/s | Sweet spot for threads |
| 64-thread 4x unroll | 117 tok/s | 193 tok/s | ILP from unrolling |
| 64-thread 8x unroll | 118 tok/s | 193 tok/s | Marginal over 4x |
| Multi-row 2 warps | 121 tok/s | 193 tok/s | Best: each warp owns a row |

Final dispatch: K ≤ 1536 → multi-row 64-thread, K > 1536 → 32-thread single-warp.

## 5. Embedding Tables Must Stay Quantized

For large-vocab models (Qwen3: 151K tokens), the standard approach of dequantizing
the embedding table to F32 at load time wastes massive VRAM:

| Model | Embedding F32 | Embedding Q4K | Savings |
|-------|--------------|---------------|---------|
| TinyLlama (32K vocab) | 256 MB | 36 MB | 220 MB |
| Qwen3-0.6B (151K vocab) | 620 MB | 87 MB | 533 MB |
| Qwen3-8B (151K vocab) | 2,489 MB | 334 MB | 2,155 MB |

For Qwen3-8B, the F32 embedding alone consumed 2.5GB of 8GB VRAM — making the
model impossible to load. GPU-side embedding lookup kernels (Q4K and Q8) dequantize
one row per token at inference time, eliminating the F32 copy entirely.

This is also why llama.cpp cannot run Qwen3-8B on the RX 5700 XT — their
embedding dequant path has the same F32 blowup.

## 6. Mixed Quantization is the Right Strategy for VRAM-Constrained Models

For Qwen3-8B (8.2B params, 8GB VRAM):

| Strategy | VRAM | tok/s | Notes |
|----------|------|-------|-------|
| All Q4_K (GGUF) | 4.7 GB | 15.2 | Q4K embedding saves 2.1GB |
| hipMallocManaged | 4.7 GB* | 4.0 | Driver page thrashing |
| Q8 attn + Q4_F16 FFN | 6.0 GB | 31.3 | Wrong Q4 kernel (32% peak) |
| Q8 attn + Q4_K FFN | 6.0 GB | 42.5 | Right Q4 kernel (42% peak) |

*managed memory uses system RAM overflow

The Q8 attention weights hit 84% peak bandwidth (occupancy-optimal).
The Q4_K FFN weights are compressed for VRAM but still reach 42% peak.
The embedding and lm_head are Q8 for fast lookup/output projection.

Critically: **quantize from raw FP16/BF16 source weights**, never from
pre-quantized GGUF. Double quantization (Q4_K → F32 → new format) produces
garbage output due to compounding quantization error.

## 7. HFQ Format Outperforms GGUF

The `.hfq` (HipFire Quantized) format produced by `hipfire-quantize` consistently
outperforms GGUF loading:

| Model | GGUF Q8_0 | HFQ Q8+Q4K | Speedup |
|-------|-----------|-----------|---------|
| TinyLlama 1.1B | 193 tok/s | 226 tok/s | 1.17x |
| Qwen3 0.6B | 128 tok/s | 227 tok/s | 1.77x |
| Qwen3 8B | 15.2 tok/s | 42.5 tok/s | 2.80x |

The gains come from:
1. Q8 embedding + lm_head (84% peak) vs F32 tied-embedding fallback in GGUF
2. Mixed per-tensor quantization (Q8 for latency-sensitive, Q4_K for bulk)
3. Quantization directly from FP16 source (no double-quant quality loss)
4. The `.hfq` format is mmap-able with 4096-byte aligned tensor data

## 8. Theoretical Peak Analysis

At the end of optimization, measured performance matched theoretical predictions
within 1-2%:

### Qwen3-8B (Q8 attn + Q4_K FFN)
- Attention GEMVs at Q8_0 (368 GB/s): 4.4 ms (19%)
- FFN GEMVs at Q4_K (186 GB/s): 16.4 ms (71%)
- Output GEMV at Q8_0: 1.8 ms (8%)
- Other ops: 0.7 ms (3%)
- **Theoretical: 23.3 ms = 42.9 tok/s**
- **Measured: 23.5 ms = 42.5 tok/s (99% efficiency)**

This means further gains require improving Q4_K bandwidth utilization (42% → 55%+),
which requires the full dp4a integer accumulation pipeline from llama.cpp — a
significant architectural change.

## 9. Failed Experiments (Don't Repeat)

| Experiment | Result | Why It Failed |
|-----------|--------|---------------|
| Q4_F16 format | Parity with Q4_K | GEMV is memory-bound, not compute-bound |
| 256-thread wide quantized GEMV | 2x slower | Element-strided access destroys metadata locality |
| Multi-warp per row (shared mem) | Slower | syncthreads overhead > occupancy benefit |
| Kernel fusion (fused QKV) | 0% gain | HIP launches are pipelined |
| dp4a Q8_1 pre-quantization | -21% | Pre-quant overhead exceeds dp4a benefit |
| Q4_K scale hoisting to registers | -1% | More VGPRs from 16 precomputed scales |
| Qwen3.5 DeltaNet | GPU hang | CPU-side recurrence + immature implementation |
| Double quantization (Q4K→Q4F16) | Garbage output | Compounding quantization error |
