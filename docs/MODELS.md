# hipfire Model Reference

## Supported Model Families

### Qwen3.5 (DeltaNet hybrid attention)
The primary model family for hipfire alpha. Uses DeltaNet hybrid architecture with mixed linear + full attention layers. head_dim=256.

| Tag | File | Size | VRAM | Notes |
|-----|------|------|------|-------|
| `qwen3.5:0.8b` | qwen3.5-0.8b.q4.hfq | 0.5GB | 1GB | Fastest, 222 tok/s on 5700 XT |
| `qwen3.5:0.8b-hfq6` | qwen3.5-0.8b.hfq6.hfq | 0.6GB | 1GB | 210 tok/s |
| `qwen3.5:2b` | qwen3.5-2b.q4.hfq | 1.2GB | 2GB | 141 tok/s |
| `qwen3.5:2b-hfq6` | qwen3.5-2b.hfq6.hfq | 1.6GB | 3GB | 127 tok/s |
| `qwen3.5:4b` | qwen3.5-4b.q4.hfq | 2.1GB | 4GB | Best balance, 63 tok/s |
| `qwen3.5:4b-hfq6` | qwen3.5-4b.hfq6.hfq | 3.3GB | 5GB | 53 tok/s |
| `qwen3.5:9b` | qwen3.5-9b.q4.hfq | 4.5GB | 6GB | Best quality on 8GB, 45 tok/s |
| `qwen3.5:9b-hfq6` | qwen3.5-9b.hfq6.hfq | 6.8GB | 8GB | 37 tok/s, near-FP16 quality |
| `qwen3.5:27b` | qwen3.5-27b.q4.hfq | 14.3GB | 16GB | 25-27 tok/s. Good for simple tasks, degrades on coding |
| `qwen3.5:27b-hfq6` | qwen3.5-27b.hfq6.hfq | 21.4GB | 24GB | 16-20 tok/s. Use this for complex/coding tasks |

### Qwen3 (standard LLaMA attention)
Earlier architecture with standard multi-head attention. head_dim=128. Supported but not the primary focus.

| Tag | File | Size | VRAM | Notes |
|-----|------|------|------|-------|
| `qwen3:0.6b` | qwen3-0.6b-hfq4.hfq | 0.4GB | 1GB | Standard attention |
| `qwen3:8b` | qwen3-8b.q4.hfq | 4.1GB | 6GB | 59.9 tok/s on 5700 XT |

## Quantization Formats

### HFQ4 (`.q4.hfq`)
4-bit quantization with 256-weight groups. 0.53 bytes per weight.
- Block format: `[f32 scale][f32 zero][128B packed 4-bit]` = 136 bytes per 256 weights
- RDNA-optimized GEMV kernel: 18 VGPRs, max occupancy
- Best speed. Good quality for inference.

### HFQ6 (`.hfq6.hfq`)
6-bit quantization with 256-weight groups. 0.78 bytes per weight.
- Block format: `[f32 scale][f32 zero][192B packed 6-bit]` = 200 bytes per 256 weights
- ~15% slower than HFQ4, noticeably better quality
- Recommended when VRAM allows

### Experimental variants (Qwen3 only)
These exist in the Qwen3 0.6B and 8B repos from earlier development:

| File | Format | Notes |
|------|--------|-------|
| `qwen3-0.6b-hfq4-v2.hfq` | HFQ4 | Alternate quantization pass |
| `qwen3-0.6b-hfq4g256.hfq` | HFQ4-G256 | Explicit group size suffix (same as default) |
| `qwen3-8b-q4k-all.hfq` | Q4_K | GGML-compatible Q4_K format, all layers |

These are kept for historical reference but are not the recommended quants.

## Naming Convention

**Current state (alpha):** Filenames are inconsistent — they reflect the order things were built:
- Qwen3.5 uses `.q4.hfq` for HFQ4 and `.hfq6.hfq` for HFQ6
- Qwen3 uses `-hfq4.hfq` with dashes
- The `.hfq` extension is sometimes doubled (`.hfq6.hfq`)

**Planned:** Unified naming convention tracked in [GitHub issue #1](https://github.com/Kaden-Schutt/hipfire/issues). The CLI tags (`qwen3.5:9b`, `qwen3.5:9b-hfq6`) are the stable interface — filenames may change.

## HuggingFace Repos

Each model size has its own repo:

| Repo | Models |
|------|--------|
| [schuttdev/hipfire-qwen3.5-0.8b](https://huggingface.co/schuttdev/hipfire-qwen3.5-0.8b) | 0.8B HFQ4 + HFQ6 |
| [schuttdev/hipfire-qwen3.5-2b](https://huggingface.co/schuttdev/hipfire-qwen3.5-2b) | 2B HFQ4 + HFQ6 |
| [schuttdev/hipfire-qwen3.5-4b](https://huggingface.co/schuttdev/hipfire-qwen3.5-4b) | 4B HFQ4 + HFQ6 |
| [schuttdev/hipfire-qwen3.5-9b](https://huggingface.co/schuttdev/hipfire-qwen3.5-9b) | 9B HFQ4 + HFQ6 |
| [schuttdev/hipfire-qwen3.5-27b](https://huggingface.co/schuttdev/hipfire-qwen3.5-27b) | 27B HFQ4 + HFQ6 |
| [schuttdev/hipfire-qwen3-0.6b](https://huggingface.co/schuttdev/hipfire-qwen3-0.6b) | 0.6B HFQ4 + variants |
| [schuttdev/hipfire-qwen3-8b](https://huggingface.co/schuttdev/hipfire-qwen3-8b) | 8B HFQ4 + Q4_K |

## Quantizing New Models

```bash
cargo build --release -p hipfire-quantize

# From HuggingFace (auto-downloads)
hipfire-quantize --input Qwen/Qwen3.5-9B --output qwen3.5-9b.q4.hfq --format hfq4
hipfire-quantize --input Qwen/Qwen3.5-9B --output qwen3.5-9b.hfq6.hfq --format hfq6

# Available formats: hfq4, hfq6, q8hfq, q4k
```
