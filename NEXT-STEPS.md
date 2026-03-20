# Next Steps for hipfire

## Current State (2026-03-20)

Working end-to-end Rust-native inference engine on AMD RX 5700 XT (gfx1010):
- **hip-bridge**: dlopen FFI to libamdhip64.so — device mgmt, memory, streams, modules, kernel launch
- **rdna-compute**: HIP kernel compiler with caching, GEMV/RMSNorm/add/mul/silu/softmax GPU ops
- **engine**: GGUF parser, Q4_K/Q6_K/Q8_0/F32/F16 dequantizers, LLaMA + Qwen3 forward pass

Verified: TinyLlama 1.1B generates coherent text at 7.3 tok/s.

## Priority 1: Quantized GPU Kernels

The biggest limitation is dequantizing all weights to F32 before upload. This means:
- TinyLlama 1.1B (4.4GB F32) barely fits in 8GB VRAM
- Qwen3-8B (32GB F32) doesn't fit at all

**Fix**: Write GPU kernels that operate directly on Q4_K/Q8_0 data:
- `gemv_q4k`: read Q4_K blocks, dequantize in-flight, accumulate in F32
- `gemv_q8_0`: same for Q8_0
- This would let Qwen3-8B Q4_K_M (4.7GB) fit in VRAM while running 2-3x faster

## Priority 2: GPU-Side Attention

Currently attention runs on CPU. Moving it to GPU would:
- Eliminate the Q/K/V download-upload round-trips per layer
- Enable Flash Attention-style memory-efficient attention
- Remove the main performance bottleneck (currently ~60% of forward time)

## Priority 3: Qwen3 Output Quality

Qwen3-0.6B loads and runs but produces repeated tokens. Debug:
- Validate Q8_0 dequantization against llama.cpp reference values
- Verify QK normalization produces correct intermediate values
- Test with proper chat template tokenization

## Priority 4: Tokenizer

Add BPE tokenizer (Qwen3 uses GPT-2 style BPE):
- Decode generated token IDs to text
- Encode user input text to token IDs
- Parse GGUF tokenizer metadata (tokens, merges, special tokens)
- Support chat templates (im_start/im_end)

## Priority 5: Performance Optimization

Current: ~7 tok/s (TinyLlama 1.1B, naive kernels, CPU attention)
Target: ~30+ tok/s

- Tiled GEMV with shared memory (current kernel is ~5% of peak)
- Fused RMSNorm + GEMV
- Batch multiple heads in attention
- Pipelining: overlap compute with memory transfers
- Persistent kernel cache (avoid recompilation)

## Priority 6: Portability

Test on other RDNA GPUs:
- RDNA2 (RX 6000 series): change --offload-arch=gfx1030
- RDNA3 (RX 7000 series): gfx1100
- RDNA4 (RX 9000 series): gfx1200
- Should work with minimal changes — the dlopen approach is architecture-agnostic

## Architecture Notes

The crate structure follows rustane's pattern:
```
hip-bridge    → Safe FFI layer (like ane-bridge)
rdna-compute  → Kernel dispatch (like metal-decode)
engine        → Inference orchestrator
```

Key design decisions:
- dlopen (no link-time ROCm dependency) → works across ROCm versions
- hipcc --genco for kernel compilation → native gfx1010 code objects
- Shared memory reduction in GEMV → correct for RDNA wavefront size (32)
- CPU-side attention as starting point → move to GPU after correctness
