# hipfire

LLM inference engine for AMD RDNA GPUs. Written from scratch in Rust + HIP. **9x faster than llama.cpp** on Qwen3.5 DeltaNet models.

## Quickstart

```bash
# Install (Linux, requires AMD GPU)
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash

# Windows (PowerShell, requires AMD GPU)
irm https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.ps1 | iex

# Pull a model and run
hipfire pull qwen3.5:9b
hipfire run qwen3.5:9b "What is the capital of France?"

# Or just run — auto-pulls if needed
hipfire run qwen3.5:4b "Hello"
```

## Performance

**RX 5700 XT (8GB, gfx1010):**

| Model | Quant | tok/s | Notes |
|-------|-------|-------|-------|
| Qwen3.5-0.8B | HFQ4 | **222** | DeltaNet, tiled LDS GDN |
| Qwen3.5-0.8B | HFQ6 | **210** | Higher quality, ~600MB |
| Qwen3.5-2B | HFQ4 | **141** | |
| Qwen3.5-4B | HFQ4 | **63** | Best balance of speed + quality |
| Qwen3.5-4B | HFQ6 | **53** | |
| Qwen3.5-9B | HFQ4 | **45** | Best quality, fits 8GB |
| Qwen3.5-9B | HFQ6 | **37** | Near-FP16 quality |
| Qwen3-8B | HFQ4 | **59.9** | Standard attention |
| ollama Qwen3.5-9B | — | 4.93 | llama.cpp + ROCm (same GPU) |

**RX 7900 XTX (24GB, gfx1100):** ([benchmarks by DomKo](https://github.com/Kaden-Schutt/hipfire/issues/2))

| Model | Quant | tok/s | Notes |
|-------|-------|-------|-------|
| Qwen3.5-9B | HFQ4 | **62** | |
| Qwen3.5-27B | HFQ4 | **25-27** | Good for simple tasks, degrades on coding |
| Qwen3.5-27B | HFQ6 | **16-20** | Use this for complex/coding tasks |

Recommended picks:
- **Speed**: 0.8B HFQ4 (222 tok/s) — fast drafting, coding assistants
- **Balance**: 4B HFQ4 (63 tok/s) — best quality-per-token for 8GB
- **Quality (8GB)**: 9B HFQ4 (45 tok/s) — strongest reasoning on 8GB cards
- **Quality (16GB)**: 27B HFQ4 (`hipfire pull qwen3.5:27b`) — good for simple tasks, degrades on coding/complex output
- **Quality (24GB)**: 27B HFQ6 (`hipfire pull qwen3.5:27b-hfq6`) — best quality, needs 7900 XTX or similar

Full benchmarks: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

## TurboQuant KV Cache

Compress the KV cache with FWHT + quantization for longer context in less VRAM:

| KV Mode | Compression | Speed (4B) | Quality |
|---------|-------------|------------|---------|
| Q8 (default) | 3.8x | 63 tok/s | baseline |
| Turbo4 | **7.8x** | 61 tok/s | minimal loss |
| Turbo2 | **15.5x** | 59 tok/s | good on ≤4B |

```bash
# Use turbo4 for 2x longer context
./target/release/examples/infer models/qwen3.5-4b.q4.hfq --turbo4 "Your prompt"
```

## Features

- **Qwen3.5 DeltaNet**: Gated linear attention with tiled LDS kernel (32 VGPRs, 20 waves)
- **HFQ4 + HFQ6**: Two weight quantization levels, both with GPU GEMV kernels
- **Vision-Language (VL)**: GPU vision encoder, `hipfire run model --image img.png "Describe this"`
- **TurboQuant KV**: Symmetric turbo4/turbo2 with 256-dim FWHT, up to 15.5x compression
- **Thinking mode**: `<think>` reasoning with n-gram loop prevention
- **Pre-compiled kernels**: Ship .hsaco blobs, no ROCm SDK needed at runtime
- **Kernel integrity**: Source-hash sidecar files detect stale blobs when hipcc is available (see [#2](https://github.com/Kaden-Schutt/hipfire/issues/2))
- **4 GPU arches**: gfx1010 (5700 XT), gfx1030 (6800 XT), gfx1100 (7900 XTX), gfx1200 (9070)
- **Zero VRAM leak**: Explicit GPU free + pool drain for model eviction
- **OpenAI-compatible API**: `hipfire serve` → `/v1/chat/completions` with SSE streaming

## Supported Models

| Family | Sizes | Arch | Quants |
|--------|-------|------|--------|
| Qwen3.5 | 0.8B, 2B, 4B, 9B, 27B | DeltaNet hybrid | HFQ4, HFQ6 |
| Qwen3.5-VL | 0.8B, 4B, 9B | DeltaNet + ViT | HFQ4 + F16 vision |
| Qwen3 | 0.6B, 8B | LLaMA attention | HFQ4 |

## CLI

```bash
hipfire pull qwen3.5:9b                      # Download model
hipfire run qwen3.5:9b [prompt]              # Generate (auto-pulls if needed)
hipfire run qwen3.5:9b --image img.png [prompt]  # Vision-language
hipfire serve [port]                         # OpenAI-compatible HTTP server
hipfire list -r                              # Show local + available models
hipfire rm qwen3.5:9b                        # Delete model
```

## API (Server Mode)

`hipfire serve` starts an OpenAI-compatible HTTP server. Works with Open WebUI, SillyTavern, and any OpenAI-compatible frontend.

> **Status:** Server mode works but is less tested than `hipfire run`. Multi-arch stability is the current priority. If you hit issues, try `hipfire run` first to isolate whether the problem is the server or the model.

```bash
hipfire serve                          # Default port 11435
hipfire serve 8080                     # Custom port

curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## Architecture

```
Bun CLI (hipfire serve/run)
  └→ Rust daemon (JSON lines IPC)
       └→ GPU kernels (pre-compiled .hsaco, 100+ kernels per arch)
            ├→ HFQ4/HFQ6 GEMV (18 VGPRs, max occupancy)
            ├→ Tiled LDS GDN (32 VGPRs, warp shuffle)
            ├→ TurboQuant KV (turbo4/turbo2, 128+256-dim FWHT)
            └→ Vision encoder (GEMM, LayerNorm, ViT attention)
```

## Contributing

hipfire is in alpha (v0.1.1) — benchmarks, bug reports, and PRs are welcome.

Technical deep-dive: [docs/DELTANET.md](docs/DELTANET.md) — how DeltaNet/Qwen3.5 works on AMD, kernel math, bug history, porting guide.

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to run and submit benchmarks
- Quantizing new models
- Setting up a dev environment
- What we need help with

**Benchmarks wanted**: if you have a 6800 XT, 7900 XTX, or 9070, we need your numbers!

## License

MIT
