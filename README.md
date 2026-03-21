# Hipfire: RDNA-Native ML Inference Engine

Rust-native LLM inference engine for AMD RDNA GPUs. No Python in the hot path, no ROCm link-time dependency — just `dlopen`, raw HIP kernels, and 198 tok/s on a $200 GPU.

## The Finding: Q8 Beats Q4 on RDNA

**On RDNA1, 8-bit quantization is 1.8x faster than 4-bit despite reading 2x more data.**

Everyone assumes smaller quantization = faster inference because you're reading less data. That's true on NVIDIA where dp4a makes 4-bit dequantization essentially free. On AMD RDNA, the story is different:

| Format | Bytes/weight | Bandwidth util | VGPRs | Waves/SIMD | TinyLlama tok/s |
|--------|-------------|---------------|-------|------------|----------------|
| F32 | 4.00 | 49% (218 GB/s) | 16 | ~10 | — |
| **Q8_0** | **1.06** | **84% (375 GB/s)** | **16** | **~10** | **198** |
| Q4_K | 0.56 | 42% (188 GB/s) | 40 | ~5 | 109 |

Q4's nibble extraction (bit shifts, masks, conditional selects) inflates register pressure from 16 to 40 VGPRs, halving occupancy from ~10 to ~5 waves per SIMD. Fewer concurrent waves means less memory latency hiding, which cuts effective bandwidth nearly in half — erasing the advantage of reading less data.

Q8 is just byte loads. No nibble extraction, no bit manipulation. The dequant path is `load byte -> convert -> FMA`. Register pressure matches F32 (16 VGPRs), occupancy matches F32 (~10 waves), and bandwidth utilization hits 84% of the 448 GB/s theoretical peak.

**This means the optimal quantization for RDNA isn't the smallest — it's the one that keeps occupancy high.** Q8 at 1 byte/weight is the sweet spot: 4x compression over F32 with near-F32 memory efficiency.

### Implications

- **RDNA1 (RX 5700 XT):** Q8 is strictly faster than Q4 for any model that fits in VRAM
- **RDNA2/3/4:** Likely similar, since the occupancy/register tradeoff is architectural. dp4a on RDNA2+ might close the gap but won't eliminate the register pressure from nibble unpacking
- **Mixed quantization:** For VRAM-constrained models, Q8 for attention weights (latency-sensitive) + Q4 for FFN weights (bulk storage) could be optimal
- **Nobody has published this.** The standard assumption in llama.cpp, vLLM, and every other inference engine is that Q4 < Q8 in speed because less data. On RDNA, the opposite is true.

## Performance

All measurements on AMD RX 5700 XT (gfx1010, RDNA1, 8GB GDDR6, 448 GB/s peak).

```
TinyLlama 1.1B:
  Phase 0 baseline:     7.2 tok/s
  Phase 3 (Q4_K opt):   106.1 tok/s   (14.7x)
  Phase 4 (Q8 + occ):   198.2 tok/s   (27.5x from baseline, 1.87x from Q4_K)

Qwen3 0.6B:
  GGUF Q8_0:             83.1 tok/s
  HFQ Q8_FP16:          111.4 tok/s   (1.34x — Q8 lm_head vs F32 tied-embed fallback)
```

## Architecture

Three-crate Rust workspace, following [rustane](https://github.com/ncdrone/rustane)'s pattern:

```
hipfire/
├── crates/
│   ├── hip-bridge/          # Safe FFI to libamdhip64.so via dlopen
│   ├── rdna-compute/        # HIP kernel compilation, dispatch, tensor ops
│   ├── engine/              # GGUF/HFQ model loading, LLaMA/Qwen3 forward pass
│   └── hipfire-quantize/    # Standalone quantizer: safetensors -> .hfq
├── docs/
│   └── Q4_F16_SPEC.md      # Q4_F16 format spec (explored, not faster on RDNA1)
└── findings/                # Phase research documentation
```

**Key design decisions:**
- **dlopen, not link-time:** Loads `libamdhip64.so` at runtime. Works across ROCm versions without recompilation.
- **Runtime kernel compilation:** HIP C++ kernels embedded as string constants, compiled via `hipcc --genco` on first use, `.hsaco` cached to disk.
- **No HSA_OVERRIDE_GFX_VERSION:** Native gfx1010 support. No lying about hardware identity.
- **Zero Python:** Pure Rust from tokenizer to token output.

## Supported Models

| Model | Format | VRAM | tok/s |
|-------|--------|------|-------|
| TinyLlama 1.1B | Q8_0 (GGUF) | ~1.2 GB | 193 |
| TinyLlama 1.1B | Q8_FP16 (HFQ) | ~1.2 GB | 198 |
| TinyLlama 1.1B | Q4_K_M (GGUF) | ~0.6 GB | 109 |
| Qwen3 0.6B | Q8_FP16 (HFQ) | ~0.9 GB | 111 |
| Qwen3 0.6B | Q8_0 (GGUF) | ~0.6 GB | 83 |

Architectures: LLaMA, Qwen3 (dense). Qwen3.5 (DeltaNet hybrid) is in progress.

## Quick Start

```bash
# Build
cd hipfire
cargo build --release

# Run with a GGUF model
cargo run --release --example infer -- /path/to/model.gguf "Hello, world"

# Quantize from HuggingFace safetensors to Q8_FP16
cargo run --release -p hipfire-quantize -- \
  --input /path/to/model-dir \
  --output model.hfq \
  --format q8f16

# Run with HFQ model
cargo run --release --example infer_hfq -- model.hfq "Hello, world"
```

### Requirements

- AMD GPU with ROCm (tested on gfx1010/RDNA1, should work on RDNA2+)
- `hipcc` in PATH (from ROCm installation)
- Rust 1.75+

## How It Works

### GEMV Kernel (the hot loop)

For decode-phase inference, the bottleneck is matrix-vector multiplication (GEMV): one row of the weight matrix times the activation vector per output element. The kernel that does this determines throughput.

**Q8_0 GEMV v3** (the fast one):
- 32 threads per block (single RDNA warp), warp shuffle reduction
- Processes 8 consecutive Q8_0 blocks (256 elements) per loop iteration
- `#pragma unroll` over the 8 blocks for instruction-level parallelism
- Each iteration: load f16 scale, load i8 value, `scale * (float)qval * x[k]`
- 16 VGPRs allocated -> max occupancy -> 84% peak bandwidth

**Q4_K GEMV** (the slower one despite less data):
- Same 32-thread warp structure
- But: 6-bit packed scale decoding, nibble extraction via shifts/masks, type conversions
- 40 VGPRs allocated -> half occupancy -> 42% peak bandwidth

### Standalone Quantizer

`hipfire-quantize` reads FP16/BF16 safetensors directly from HuggingFace model directories and produces `.hfq` (HipFire Quantized) files. Quantization is done once, offline.

Q8_FP16 symmetric quantization per group of 32:
```
scale = max(|weights|) / 127
quantized[i] = round(weight[i] / scale)  // int8, [-128, 127]
```

Block format: `f16 scale (2B) + int8 values[32] (32B) = 34 bytes per 32 weights`.

### .hfq File Format

Binary format with mmap-able tensor data:
- Header: magic, version, architecture, tensor count, offsets
- Metadata: JSON blob with model config + tokenizer reference
- Tensor index: name, quant type, shape, data offset
- Tensor data: 4096-byte aligned, directly mmap-able

## Research Log

This project follows [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) methodology: strategy document -> agent modifies code -> fixed eval -> keep/discard -> repeat.

### Phase 0-2: Foundation
- dlopen FFI to libamdhip64.so (no link-time ROCm dependency)
- GGUF parser, BPE tokenizer, LLaMA forward pass
- Basic GEMV kernel: 7.2 tok/s baseline

### Phase 3: Kernel Optimization
- Single-warp Q4_K GEMV with `__launch_bounds__(32, 20)`: 178.8 GB/s
- GPU-side attention, RoPE, embedding lookup (eliminate CPU round-trips)
- Fused silu_mul, in-place residual add
- Result: 106.1 tok/s (14.7x from baseline)

### Phase 4: The Q8 Discovery
- Built Q4_F16 format, quantizer, .hfq file format (infrastructure)
- Q4_F16 GEMV at parity with Q4_K (both ~42% peak) — format doesn't matter
- Profiled with llvm-objdump: Q4_K uses 40 VGPRs, Q8_0 uses 16
- Hypothesis: occupancy (waves/SIMD) explains the 42% vs 49% bandwidth gap
- **Unrolled Q8_0 kernel hits 84% peak (375 GB/s)** — 2x Q4_K's bandwidth
- End-to-end: 198 tok/s TinyLlama, 111 tok/s Qwen3 (27.5x total speedup)

### Key Failed Experiments
- **Q4_F16 format:** Simpler dequant doesn't help because GEMV is memory-bound, not compute-bound. The bottleneck is occupancy, not instruction count.
- **256-thread wide quantized GEMV:** Element-strided access destroys metadata locality. Quantized blocks require block-sequential access for cache efficiency.
- **Kernel fusion (fused QKV, gate+up):** Negligible gain (~0%) because HIP kernel launches are pipelined at 2.73us each. The GPU command queue overlaps them automatically.
- **Multi-warp per row:** Shared memory reduction overhead > occupancy benefit for quantized GEMV.

## License

MIT
