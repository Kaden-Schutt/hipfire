# hiptrx PR319 Forward-Train Validation

Host: `hiptrx`
GPU: 4x AMD Radeon AI PRO R9700, `gfx1201`
ROCm: HIP 7.2.53211 / ROCm 7.2.2 clang, driver `7.0.0-15-generic`
Test GPU: `ROCR_VISIBLE_DEVICES=0`

Code under test:

- `a1109791` for the initial broad environment/build/kernel-smoke sweep.
- `0bcc6bb8` for the post-fix targeted rerun and model smokes.

## Result Summary

| Area | Command / log | Result | Notes |
|---|---|---|---|
| Arch caps | `raw/hiptrx-arch-caps-20260526.log` | PASS | `cargo test -p rdna-compute arch_caps --lib`, 17/17 passed. |
| Targeted build | `raw/hiptrx-targeted-build-0bcc6bb8-20260526.log` | PASS | Built rdna direct probes plus Qwen2 and dots OCR examples at `0bcc6bb8`. |
| gfx12 WMMA/Lloyd kernel smokes | `raw/hiptrx-gfx1201-kernel-smokes-20260526.log` | MIXED, superseded | Initial broad run passed gfx12 QKV/QKVZA/gate/residual/HFQ6/MQ3/MQ4 fused probes, but direct gfx11-only probes failed on gfx1201. |
| gfx11-only direct probe rerun | `raw/hiptrx-gfx1201-wiring-rerun-20260526.log` | PASS / SKIP | MQ4-Lloyd residual base gfx12 path passed all shapes. `_mb2`/`_mb4` now skip as gfx11-only. Causal WMMA parity now skips on gfx1201 instead of compiling gfx11 WMMA. |
| Qwen2 real model smoke | `raw/hiptrx-gfx1201-qwen2-smoke-20260526.stderr` | PASS | `/mnt/nas/kaden/models/qwen2-1.5b.arch7.q8.hfq`; tokenizer parity exact; 16/16 generated token top-1 matches reference. |
| dots OCR bounded smoke | `raw/hiptrx-gfx1201-dots-ocr-max0-20260526.stderr` | PASS | `/mnt/nas/kaden/models/dots-ocr.q8.hfq`; full vision tower loaded; `vision kernels: scalar-fallback`; all 42 vision blocks completed; text loaded; 5095-token batch prefill completed. |

## Routing Conclusion

The current PR wiring works on hiptrx/gfx1201 as a safe fallback:

- Qwen2 runs end-to-end against the reference smoke artifact.
- dots OCR does not try to load the gfx11-only vision WMMA route on gfx1201; it explicitly takes `scalar-fallback`.
- The direct causal WMMA parity harness is now skipped on gfx1201 because that kernel has no `_w32_gfx12` sibling yet.
- MQ4-Lloyd residual gfx12 base path is valid; only the `_mb2`/`_mb4` fanout experiments are gfx11-only and now skip.

## Follow-up Port Work

The remaining performance lift is real gfx12 port work, not current-wiring correctness:

- Add a gfx12 causal WMMA sibling for `attention_dflash_wmma_m64_n128_f16kv_v3_causal_f32`.
- Consider gfx12 dots OCR vision WMMA siblings after the causal path is settled.
- Keep production gates conservative until each gfx12 sibling has hardware parity data on hiptrx.
