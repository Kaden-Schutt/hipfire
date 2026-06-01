# Vision pipeline cleanup — issue #326

Branch: `vision-pipeline-cleanup-326`

## Status assessment (2026-06-01)

| Item | Description | Status | Action |
|------|-------------|--------|--------|
| **A1** | `rfind` UTF-8 think-pair tracker — O(N²) | ✅ **Fixed** | Replaced with token-ID tracking (`think_depth`/`think_count`) |
| **A2** | `generate_vl` is 401 lines, needs refactor | Deferred | Extract helpers (larger refactor, separate PR) |
| **A3** | Multiple early-return `write_error` in `generate_vl` | ✅ **Fixed** | Added invariant comment documenting the GPU-safety property |
| **A4** | `assistant_prefix` hardcoded | Already correct | `AssistantPrefix::Plain` with explanatory comment — correct for VL |
| **B1** | `https://` image URLs silently dropped | ✅ **Fixed** | Added `unsupportedImage = true` for non-data: URLs |
| **B2** | Multi-turn VL rejection relies on daemon reset | ✅ **Fixed** | Added defensive reset at VL dispatch when `seq_pos > 0` |
| **B3** | Large base64 memory spike | Deferred | No use case yet |
| **C1** | 189 GPU alloc/free per image in `vision_forward` | Deferred | Larger change, separate PR |
| **C2** | `pos_embed` uploaded per image | Deferred | Single-image only currently |
| **C3** | `vit_attention_opt` dead code | **NOT APPLICABLE** | Live code used by dots-ocr (has rotary in dots_ocr path) |
| **C4** | No VL output coherence gate | Deferred | Needs GPU hardware to test, separate PR |
| **C5** | channel_order tests are pure-color only | ✅ **Fixed** | Added `quadrant_pixels_keep_rgb_spatial_order` test with spatial+channel verification |
| **D2** | `smart_resize` usize overflow on 32-bit | ✅ **Fixed** | Added u64 arithmetic in both qwen35-vl and dots-ocr `smart_resize` |
| **D3** | `--no-verify` policy undocumented | Already documented | CONTRIBUTING.md line 91; CLAUDE.md § coherence-gate |
