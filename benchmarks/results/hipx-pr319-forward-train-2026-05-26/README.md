# hipx PR319 Forward-Train Validation — 2026-05-26

Validation artifacts for local integration branch
`integration/pr319-forward-train` through this WMMA-gate fix-forward commit.

The branch contains sequential merges of PRs 319, 337, 338, 330, 335,
336, 333, and 331, plus fix-forward commits:

- `441924f6` — restore HFQ3 MMQ y-variant dispatch after PR319
- `46368fdc` — keep HFQ4 MMQ off unsupported sdot4 archs
- `19fe9bff` — use q8 for DFlash perf gates
- `028d1138` — update Qwen2 infer smoke tokenizer error path
- this commit — gate dots.ocr / Qwen2 WMMA paths to RDNA3

Host: `hipx`

Detected GPUs:

- `ROCR_VISIBLE_DEVICES=1`: `gfx1151` Radeon 8060S / Strix Halo
- `ROCR_VISIBLE_DEVICES=2`: `gfx1030` RX 6950 XT
- `ROCR_VISIBLE_DEVICES=0`: `gfx1010` RX 5700 XT

## Summary

| Surface | GPU | Result | Raw artifact |
|---|---:|---|---|
| Release build + static checks | host | PASS: `cargo check`, `bash -n`, release examples | see session notes |
| Arch capability gating | host | PASS: `cargo test -p rdna-compute arch_caps --lib`, 17/17 | see session notes |
| DFlash coherence, q8 KV | gfx1151 | PASS: no hard errors, sane output | `raw/hipx-pr319-forward-gfx1151-coherence-20260526-213421.md` |
| DFlash perf, 27B, q8/max256 | gfx1151 | PASS: `decode_tok_s=70.46`, `tau=8.519` | `raw/hipx-pr319-forward-gfx1151-q8-max256-20260526-213611.log` |
| DFlash smoke, 9B, q8/max256 | gfx1030 | PASS: `decode_tok_s=40.46`, `tau=6.7576` | `raw/hipx-pr319-forward-gfx1030-q8-max256-20260526-213831.log` |
| DFlash smoke, 9B, q8/max256 | gfx1010 | PASS: `decode_tok_s=32.17`, `tau=6.5294` | `raw/hipx-pr319-forward-gfx1010-q8-max256-20260526-213903.log` |
| MQ4-Lloyd WMMA parity | gfx1151 | PASS: all shapes | `raw/hipx-pr319-gfx1151_mq4_lloyd_wmma-20260526-214031.log` |
| MQ3-Lloyd WMMA parity | gfx1151 | PASS: all shapes, max abs `5.829e-5` | `raw/hipx-pr319-gfx1151_mq3_lloyd_wmma-20260526-214102.log` |
| HFQ3 MMQ sweep | gfx1030 | PASS: sweep completed | `raw/hipx-pr319-gfx1030_hfq3_mmq_sweep-20260526-214104.log` |
| FWHT-128 GPU vs CPU | gfx1151 | PASS: max abs `0e0` | `raw/hipx-pr319-gfx1151_fwht128-20260526-214331.log` |
| Paro4G128 GEMV variants | gfx1151 | PASS: all variants | `raw/hipx-pr319-gfx1151_paro4g128-20260526-214335.log` |
| Causal WMMA attention parity | gfx1151 | PASS: max abs `1.439e-4` | `raw/hipx-pr319-gfx1151_causal_wmma_parity-20260526-214336.log` |
| Decode attention microbench | gfx1151 | PASS: kernels launch and compare | `raw/hipx-pr319-gfx1151_decode_attention_short-20260526-214339.log` |
| MTP head smoke, 9B | gfx1151 | PASS: finite logits, KV readback signal | `raw/hipx-pr319-gfx1151_mtp_head_smoke_9b-20260526-214341.log` |
| MTP-only decode, 9B, q8 KV | gfx1151 | PASS: `tok_s=25.53`, `tau=2.0323` | `raw/hipx-pr319-gfx1151_mtp_only_9b-20260526-214351.log` |
| CLI unit tests after Bun install | hipx CPU | PASS: 121 tests | `raw/hipx-pr319-cli-bun-tests-20260526-214651.log` |
| dots.ocr real model, Q8 HFQ | local gfx1100 | PASS: full OCR output graded 13/13, F1 1.000 | `../dots-ocr-real-2026-05-26/` |
| dots.ocr WMMA gate smoke | gfx1030 | PASS: selected `scalar-fallback`, cleared all 42 vision blocks; timed out after vision stack, no gfx11 WMMA compile failure | `../dots-ocr-real-2026-05-26/hipx_gfx1030_ocr_e2e_gated2_timeout_stderr.log` |
| dots.ocr WMMA gate smoke | gfx1151 | PASS: selected `rdna3-wmma`, completed vision/text/prefill with `--max-tokens 0` | `../dots-ocr-real-2026-05-26/hipx_gfx1151_ocr_e2e_gated_short_stderr.log` |

An earlier hipx CLI attempt before installing Bun is kept as
`raw/hipx-pr319-cli-bun-tests-20260526-214504.log` and records the
environment gap (`bun not found`).

## PR Surface Mapping

| PR / commit | Main surface | Coverage in this artifact set |
|---|---|---|
| PR319 | Paro4G128, FWHT-128, arch caps, quant/runtime dispatch | FWHT-128, Paro4G128, DFlash coherence/perf |
| `441924f6` | HFQ3 MMQ y-variant dispatch fix-forward | gfx1030 HFQ3 MMQ sweep |
| PR337 | FeatureFlags / ArchCaps refactor, env/config routing | `cargo check`, arch cap tests, cross-arch DFlash smokes |
| PR338 | MTP head/spec paths, DFlash/MTP demos, docs/scripts | MTP head smoke and MTP-only q8 decode |
| PR330 | gfx1151 MQ4-Lloyd K4/mb4 WMMA kernels | MQ4-Lloyd WMMA parity on gfx1151 |
| PR335 | RDNA2/RDNA1 HFQ4/HFQ3 MMQ routing | gfx1030 and gfx1010 q8 DFlash smokes, arch cap tests |
| `46368fdc` | Prevent HFQ4 MMQ on unsupported sdot4 archs | arch cap test plus gfx1010 smoke |
| PR336 | causal/flash attention kernels and decode attention work | causal WMMA parity and decode-attention microbench |
| PR333 | KLD baseline data/docs | data-only; covered by repository inclusion and build sanity |
| PR331 | CLI/serve config, pflash logging, cask error surfaces | Bun CLI tests; daemon build only for serve compile surface |
| `19fe9bff` | q8 DFlash perf/gate script update | all DFlash perf/gate runs use q8; no asym KV perf runs |
| `028d1138` | Qwen2 standalone smoke error-path cleanup | Qwen2 real dots.ocr artifact parsed and loaded in OCR E2E validation |
| this commit | dots.ocr / Qwen2 WMMA arch gate fix-forward | gfx1030 scalar-fallback smoke plus gfx1151 RDNA3.5 WMMA smoke |

## KV Policy

For DFlash perf and coherence gates in this validation, `asym*` KV modes
were not used. Runs used `--kv-mode q8`. FWHT KV modes remain valid
fallbacks for future validation when appropriate.

## dots.ocr / RDNA2 Note

The dots.ocr real-model pass uses `/mnt/nas/kaden/models/dots-ocr.q8.hfq`,
quantized from `rednote-hilab/dots.ocr` with `--format q8 --arch-id 8
--include-vision`. The vision tower stores dense F16 GPU weights after
load/dequant, so the RDNA2 HFQ3/HFQ4 sdot4 MMQ family does not accelerate
that vision path yet. gfx1030 still has `has_hfq4_mmq()` / `has_hfq3_mmq()`
coverage for text/quantized projection routes, but it does not have WMMA.

The pre-fix gfx1030 OCR attempt failed compiling `gemm_f16_wmma` for
`gfx1030` because the gfx11 `_w32` WMMA builtin requires
`gfx11-insts,wavefrontsize32`. This commit gates dots.ocr vision
GEMM/attention WMMA and Qwen2 batched causal WMMA prefill on
`ArchCaps::has_wmma_w32()`: gfx1030 now routes to scalar fallback, while
gfx1151 keeps the RDNA3 WMMA path.
