# hiptrx gfx12 Causal WMMA Port Validation

Host: `hiptrx`
GPU: 4x AMD Radeon AI PRO R9700, `gfx1201`
ROCm: HIP 7.2.53211 / ROCm 7.2.2
Test GPU: `ROCR_VISIBLE_DEVICES=0`

Code under test:

- `7acad6c1` added the gfx12 causal DFlash WMMA sibling.
- `d33de23b` enabled the Qwen2 batch-prefill gate on gfx12.
- `a570ea63` added timing output to the parity harness.

## Result Summary

| Area | Command / log | Result | Notes |
|---|---|---|---|
| Build | `cargo build --release -p rdna-compute --example parity_causal_wmma -p hipfire-arch-qwen2 --example infer_qwen2 -p hipfire-arch-dots-ocr --example ocr_e2e` | PASS | Built on `hiptrx` before model smokes. Existing repo warnings only. |
| Direct causal WMMA parity + timing | `raw/causal-parity-bench-a570ea63.log` | PASS | gfx1201 parity max-abs-diff `1.439e-4` at all tested sizes. WMMA beat scalar at B=128, 512, 1024, and 2048. |
| Qwen2 real model smoke | `raw/qwen2-smoke-a570ea63.log` | PASS | `/mnt/nas/kaden/models/qwen2-1.5b.arch7.q8.hfq`; tokenizer parity exact; 16/16 generated token top-1 matches reference. |
| dots OCR bounded smoke | `raw/dots-ocr-max0-a570ea63.stderr` | PASS | `/mnt/nas/kaden/models/dots-ocr.q8.hfq`; full vision tower completed on scalar fallback; Qwen2 text path loaded; 5095-token batch prefill completed with `--max-tokens 0`. |
| Repro hashes | `raw/md5-a570ea63.log` | RECORDED | Binary md5s plus prompt/reference/image md5s captured from `hiptrx`. |

## Causal WMMA Timing

Measured with `target/release/examples/parity_causal_wmma <B> <iters>` on GPU 0:

| B | Iters | Scalar us/call | WMMA us/call | Speedup |
|---:|---:|---:|---:|---:|
| 128 | 50 | 79.8 | 53.8 | 1.48x |
| 512 | 20 | 1438.7 | 225.3 | 6.39x |
| 1024 | 10 | 7310.4 | 593.5 | 12.32x |
| 2048 | 5 | 33427.8 | 1955.0 | 17.10x |

## Routing Conclusion

The current wiring is now validated in both layers:

- Existing PR319 wiring works on hiptrx/gfx1201 without loading gfx11-only kernels.
- The new gfx12 causal WMMA sibling is correct against the scalar path on real gfx1201 hardware.
- Qwen2 can keep the gfx12 batch-prefill WMMA gate enabled: the direct harness shows a speedup at the production-shaped causal attention call, and the real Qwen2/dots OCR smokes both complete.

The dots OCR vision tower is still `scalar-fallback` on gfx1201. That is separate follow-up port work; this validation only covers the Qwen2 text causal prefill path.
