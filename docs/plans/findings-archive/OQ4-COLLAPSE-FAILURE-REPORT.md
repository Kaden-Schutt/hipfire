# OQ4+ Dual-Layout Collapse — Failure Report

**Date:** 2026-06-24
**Branch:** `chaingun`
**Outcome:** Attempted, **reverted** (unresolved integration bug). Working tree restored to `b7a1f8a0`.
**Author:** Claude Opus 4.8 (session `session_0146eACngDJtYnxHUX3iFgjZ`)

---

## 1. Goal

OQ4+ decode was brought to parity with mq4+ (59.4 tok/s) by introducing an
**interleaved** weight layout for the decode GEMVs, appended *after* the existing
**split** layout that prefill (MMQ / f16-WMMA) reads. This dual-layout costs
**~+0.25 GB VRAM** (and, once materialized on disk by `hipfire repack`, ~+0.25 GB
file size) because the same nibbles + scales are stored twice in two arrangements.

The collapse task: port the **prefill** kernels to read the interleaved layout too,
then drop the split region — yielding a single interleaved layout shared by both
phases, reclaiming the VRAM and disk. This mirrors mq4, which is single-layout.

## 2. Why it is sound in principle

mq4 (`hfq4g256`) was confirmed to use **one** layout for both phases:

- **Prefill GEMM** — `kernels/src/gemm_hfq4g256.hip`: `row_ptr + g*136`, scale at +0,
  nibbles at +8 (136 B record = `[f32 scale][f32 zp][128 nibbles]`).
- **Prefill WMMA** — `gemm_hfq4g256_residual_wmma_gfx12_bt.hip`: `rb + g*136`, same.
- **Decode GEMV** — `gemv_hfq4g256.hip`: same 136 B records.

So a single interleaved layout serving both phases is a proven design. OQ4+ is
symmetric (no zero-point), so its record is **132 B** = `[f32 scale][128 nibbles]`,
scale at +0, nibbles at +4.

## 3. What was changed (all reverted)

- **Loader** (`crates/hipfire-arch-qwen35/src/qwen35.rs`): `oq4_pack_arch_combined`
  rewritten to emit interleaved-only (`m*ng*132`) at offset 0; `oq4_arch_combined_len`
  updated; qt=34 and qt=37 arms consume it.
- **Decode dispatch** (`crates/hipfire-dispatch/src/pipeline/steps.rs`): all
  `sub_offset(m*(k/2)+m*ng*4, …)` interleaved views changed to `sub_offset(0, …)`;
  the `GemvResidual{Prerotated}` Oq4 arm switched from the split
  `gemv_oq4_grouped_residual` to `gemv_oq4_interleaved_residual`.
- **Prefill kernels**:
  - `kernels/src/gemm_oq4_residual_mmq.hip` — `load_oq4_tile` ported to read the
    132 B records (`(row*gpr+kb)*132`, scale@+0, nibbles@+4) instead of the split
    nibble-plane + distant `Ws` scales plane. This `load_oq4_tile` is shared by the
    bounds-checked and `full_set`/`full_add` (the live fast path) variants.
  - `kernels/src/gemm_oq4_grouped_f16_wmma.hip` — ported the same way.

The fused W4A4 WMMA kernels (`fused_qkvza_oq4_wmma`, `fused_gate_up_oq4_wmma`) and
`gemm_oq4_grouped_wmma` were confirmed **dead** (no live callers) and not touched.

## 4. The symptom

After the collapse, **all OQ4 models produce NaN / garbage** (`<think>!!!!!!…`):

- `perplexity … --kld-ref` → `Scored: 0`, `non-finite NLL at pos=…` for every position.
- `infer_qwen35` → `<think>` (template-forced) then a single-token `!!!` attractor.
- Reproduces on `oq4.hfq`, `oq4awq.hfq`, **and** `oq4+.hfq` (so **not** AWQ-related).

Reverting to `b7a1f8a0` restores coherence (`KLD/tok: 0.046337`, fluent output).

## 5. Every component was verified individually correct

This is the crux of why it is unresolved: **each piece checks out in isolation,
but the whole fails.**

| Component | Verification | Result |
|---|---|---|
| Loader output | `HIPFIRE_OQ4_PACK_DEBUG=1` dump | 186 tensors, **exact** `m*ng*132` sizes, sane scales |
| Interleaved bytes | Byte-for-byte diff vs the validated dual-layout interleaved region | **Identical** record content (scale + nibbles) |
| `gemm_oq4_grouped_f16_wmma` (W4A16 prefill) | New CPU-reference parity harness, interleaved input | **PASS** (rel ≤ 0.0004) across all real dims |
| `gemm_oq4_residual_mmq` (int8 MMQ prefill) | Same harness | **PASS** (rel ≤ 0.0041) across all real dims |
| Real dims tested | 6144×1024, 1024×3584 (ng=14), 1024×2048, 3584×1024, 256×1024 | all PASS |
| Decode offset | `sub_offset(0,…)` → pointer at buffer start (offset 0, dtype-size-independent) | Correct |
| Weight pointers in `forward_prefill` | `&layer.wq.buf` / `&layer.wo.buf` etc. passed at offset 0, no stale sub_offset | Correct |
| Build freshness | Forced `arch-qwen35` + `rdna-compute` recompile, cleared `~/.hipfire/kernels/gfx1103` | Confirmed fresh |

The CPU-reference parity harness (a rewrite of
`crates/rdna-compute/examples/parity_gemm_oq4_mmq.rs`) builds the interleaved buffer
exactly as the loader does and compares **both** prefill kernels to a CPU dot-product
reference. Both passed at every real model dimension — proving the kernel ports read
the interleaved layout correctly.

## 6. Localization

`perplexity` KLD is **prefill-only** (it prefills the context and scores each
position's logit; no autoregressive decode). Its output is NaN. Therefore the bug
is in the **prefill integration**, not in decode and not in the kernels themselves
(which the parity harness proves correct). The `<think>` token in `infer` is
template-forced, so it does not indicate a working prefill.

This is a genuine paradox: loader correct + kernels correct (parity) + offsets
correct + pointers correct, yet the end-to-end prefill forward produces NaN.

## 7. What is NOT the cause (ruled out)

- **Stale kernel cache** — `~/.hipfire/kernels/gfx1103` validates by
  `hash(source+arch)` and recompiles via hipcc on mismatch; cleared and reproduced.
- **Stale binary** — forced recompile of `hipfire-arch-qwen35` (loader) and
  `rdna-compute` (`include_str!` kernels); reproduced.
- **AWQ / rotation** — plain `oq4.hfq` (no AWQ sidecar) reproduces identically.
- **Kernel layout/sign bug** — CPU-reference parity PASS across all dims.
- **Loader byte layout** — instrumented; exact sizes and sane scales.
- **An unported live kernel** — full live-call inventory taken
  (`gemm_oq4_{qkv,qkvza,gate_up}_mmq`, `gemm_oq4_grouped{,_residual}_act_batched`,
  `gemm_oq4_grouped_f16_wmma` all resolve to the two ported kernels; fused W4A4 WMMA
  + `gemm_oq4_grouped_wmma` are dead).

## 8. Gotchas discovered (useful regardless)

1. **`include_str!` kernels need an explicit rebuild trigger.** Editing a `.hip`
   does not always force the embedding crate (`rdna-compute`) to recompile; the
   binary can ship stale kernel source. Fix: `touch crates/rdna-compute/src/kernels.rs`
   and confirm `Compiling rdna-compute` / `Compiling hipfire-arch-qwen35` in the
   build output before testing.
2. **`HIPFIRE_OQ4_BATCHED_PREFILL=0` does NOT fully disable batched prefill** — it
   cannot be used to force a pure decode-only path for isolation.
3. **`GpuTensor::sub_offset(offset_elems, len_elems)` is element-scaled**
   (`offset_elems * dtype.size()`), but for the Raw-dtype weight buffer
   `dtype.size() == 1`, so the existing byte-count arguments work; offset 0 is
   pointer-at-start regardless.

## 9. Recommended next step

Do **not** re-attempt by re-reading kernels — every component is already proven.
Instead, instrument the **real forward** at GPU-I/O granularity:

1. In both the dual-layout (`b7a1f8a0`) and a collapse build, dump layer-0 prefill
   **inputs and outputs** — the rotated/quantized activation that reaches the first
   `gemm_oq4_*` call, and that call's output tensor — to host and diff them.
2. If the activation differs → the integration feeds the kernel different data than
   the parity harness does (e.g. a quantization/rotation/stream-ordering interaction
   that only manifests in the full pipeline).
3. If the activation matches but the output differs → the kernel *as dispatched*
   (grid/shared-mem/stream/launch-blob args) differs from the parity invocation;
   compare the actual launch parameters, not just the kernel body.

The CPU-reference parity harness (Section 5) is the right tool to keep; it should be
committed as a permanent regression guard once the layout question is settled.

## 10. Status of the shipped wins (intact)

The decode work that motivated this — and which is **not** affected by the revert —
remains committed and pushed:

- **`01ed7360`** — interleaved decode layout: 56.0 → 58.1 tok/s.
- **`67633cd7`** — 4-group unroll on interleaved decode GEMVs: 58.1 → 59.4 tok/s
  (decode **parity** with mq4+ 59.3–59.5; KLD bit-identical 0.046337).
- **`83475458`** — `oq4_repack` engine + qt=37 arch-packed layout.
- **`b7a1f8a0`** — `hipfire repack` CLI (arch-tagged `<model>.<arch>.hfq`).

OQ4+ remains net-superior to mq4+ on every axis (decode parity, prefill 1610 > 1578,
KLD 0.046 < 0.078, half VRAM). The dual-layout's **~+0.25 GB** is the documented,
accepted cost until the collapse is landed.
