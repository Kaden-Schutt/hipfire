# dots.ocr real-model validation, 2026-05-26

Model artifact:

- Source: `rednote-hilab/dots.ocr`
- Quantized artifact: `/mnt/nas/kaden/models/dots-ocr.q8.hfq`
- Quant command used `--format q8 --arch-id 8 --include-vision`
- HFQ inspector reported Qwen2 config: hidden 1536, 28 text layers, 12 heads, 2 KV heads, head_dim 128, vocab 151936.

Runs:

| File prefix | Host/GPU | Result |
|---|---|---|
| `gfx1100_ocr_e2e` | local `gfx1100` RX 7900 XTX | Full OCR run passed. Vision 51.3s, prefill 3.9s, generated 4633 tokens in 71.3s. Grade: 13/13 regions, F1 1.000. |
| `gfx1100_ocr_e2e_gated_short` | local `gfx1100` RX 7900 XTX | Post-gate smoke. Logged `vision kernels: rdna3-wmma`, confirming RDNA3 path remains active after gating. |
| `hipx_gfx1030_ocr_e2e_gated2_timeout` | hipx `gfx1030` RX 6950 XT via `ROCR_VISIBLE_DEVICES=2` | Post-gate RDNA2 smoke. Logged `vision kernels: scalar-fallback` and cleared all 42 vision blocks. Timed out at 240s after vision block completion; no RDNA3 WMMA compile failure. |
| `hipx_gfx1151_ocr_e2e_gated_short` | hipx `gfx1151` Radeon 8060S via `ROCR_VISIBLE_DEVICES=1` | Post-gate RDNA3.5 smoke. Logged `vision kernels: rdna3-wmma`, completed vision/text/prefill with `--max-tokens 0`. |

The pre-gate gfx1030 run failed while compiling `gemm_f16_wmma` for `gfx1030`:
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32 needs target feature gfx11-insts,wavefrontsize32`.
The gate fix routes dots.ocr vision WMMA kernels and Qwen2 WMMA causal prefill
through `ArchCaps::has_wmma_w32()`.
