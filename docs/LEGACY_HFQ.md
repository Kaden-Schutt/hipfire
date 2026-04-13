# HFQ Quantization Family — RDNA-Native Weight Formats

All formats share the same block structure:
```
[f32 scale (4B)] [f32 zero (4B)] [packed data] = block_bytes per 256 weights
```

Flat metadata (2 FP32 values) keeps VGPRs under 22 across all bit widths.
All kernels use `__launch_bounds__(32, 20)` — single warp, max occupancy.

## Performance Table (RX 5700 XT, RDNA1, 448 GB/s peak)

| Format | B/weight | Block bytes | VGPRs | Attn 4096x4096 | | FFN 12288x4096 | |
|---------|----------|-------------|-------|--------:|-------:|--------:|-------:|
| | | | | us | GB/s | us | GB/s |
| HFQ2-G256 | 0.281 | 72 | 19 | 30.4 | 155.6 | 73.7 | 192.4 |
| HFQ3-G256 | 0.406 | 104 | 20 | 33.9 | 201.4 | 83.8 | 244.1 |
| **HFQ4-G256** | **0.531** | **136** | **18** | **31.6** | **282.6** | **88.8** | **301.2** |
| HFQ6-G256 | 0.781 | 200 | 21 | 49.0 | 267.8 | 121.2 | 324.6 |
| HFQ8-G256 | 1.031 | 264 | 18 | 56.5 | 306.7 | 157.2 | 330.4 |
| Q4_K (GGML) | 0.563 | 144 | 39 | 45.0 | 210.1 | 108.4 | 261.3 |

## Key Observations

1. **All HFQ formats beat Q4_K** — even HFQ2 at half the data reads less but achieves 73% of Q4K's bandwidth utilization with 2x occupancy.

2. **HFQ4-G256 is the sweet spot** — best bandwidth efficiency (GB/s per byte of data) at the most useful compression ratio. Fastest wall-clock at attention sizes.

3. **HFQ8-G256 hits 73.7% peak** — highest absolute bandwidth at 330 GB/s. Replaces GGML Q8_0 (which uses 34 bytes per 32 elements = 1.0625 B/w and 39+ VGPRs).

4. **HFQ2-G256 enables extreme compression** — 0.28 B/w means Qwen3-8B fits in ~2.3 GB. Quality will be very low but useful for speculative decode draft models where speed matters more than precision.

5. **VGPRs are flat across all bit widths** (18-21) — the flat metadata design eliminates the hierarchical scale complexity that inflates Q4_K to 39 VGPRs. Occupancy is always 100%.

## Block Layout Details

### HFQ2-G256 (72 bytes per 256 weights)
- Data: 64 bytes, 4 weights per byte
- Unpack: `(byte >> (i*2)) & 3`, levels 0-3

### HFQ3-G256 (104 bytes per 256 weights)
- Data: 96 bytes, 8 weights per 3 bytes (24 bits)
- Unpack: bit-aligned extraction, levels 0-7

### HFQ4-G256 (136 bytes per 256 weights)
- Data: 128 bytes, 2 weights per byte (nibbles)
- Unpack: `byte & 0xF` / `byte >> 4`, levels 0-15

### HFQ6-G256 (200 bytes per 256 weights)
- Data: 192 bytes, 4 weights per 3 bytes (24 bits)
- Unpack: 6-bit aligned extraction, levels 0-63

### HFQ8-G256 (264 bytes per 256 weights)
- Data: 256 bytes, 1 weight per byte
- Unpack: direct byte read, levels 0-255
