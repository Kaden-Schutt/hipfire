# QTIP G128 — trellis quant for ÷128-not-÷256 hidden shapes

Status: planned (not started)
Owner: unassigned
Depends on: qtip4 G256 serving (commit `8e1413f37`) + the shared
`hipfire-quant-codecs` crate (`e26d899e1…f6a82b4bc`).

## Problem

The QTIP codec (and its decode GEMV) is hardwired to **256-element groups**
(`QTIP3_GROUP = 256`, `gen_fwht_signs(.., 256)`, `cpu_fwht_256`). A weight
tensor only packs to `Qtip{3,4}G256` when `k = shape[1]` is divisible by 256;
otherwise the offline driver (`pack_qtip_real_tensors`) leaves it **BF16**.

Small models with a hidden size that is a multiple of 128 but **not** 256 pay
for this: e.g. **Qwen2.5-0.5B** (`hidden = 896 = 7·128`). Its `q/k/v/o/gate/up`
linears all have `k = 896` → BF16 fallback; only `down_proj` (`k = 4864 = 19·256`)
becomes QTIP. So most of the model is not compressed and the qtip4 decode path is
barely exercised.

This is a solved problem for the other quant families — the DType enum already
carries `HFQ4G128`, `HFQ3G128`, `HFQ2G128`, `MQ4G128` **specifically** to cover
÷128 shapes with a 128-wide FWHT (`RotationPlan::FwhtG128`). QTIP simply never
got a G128 variant. `Qtip3G128` does not exist either, so this is a QTIP-family
enhancement, not qtip4-specific.

## Goal

Add `Qtip3G128` and `Qtip4G128` so trellis quant serves ÷128 shapes, reusing the
existing `FwhtG128` rotation and the exact wiring pattern established by the G256
serving work.

## Byte layouts (128-element group)

Trellis codebook (computed 1MAD, 12-bit state) and the sliding-window decode are
**group-size-agnostic** — the state resets to 0 at each group boundary, so a
128-wide group just resets twice as often (still ≫ the 3–4 symbol window, so no
meaningful quality loss expected; validate).

- **Qtip3G128**: `[f32 scale][48 B packed 3-bit]` = **52 B/group** → 0.406 B/w
  (vs G256 0.391; the +0.4 bit is the extra per-group scale).
- **Qtip4G128**: `[f32 scale][64 B nibble-packed 4-bit]` = **68 B/group** →
  0.531 B/w (vs G256 0.516; = MQ4 territory).

## Work items (mirrors the G256 effort, ~one session)

1. **Codec** (`hipfire-quant-codecs`): parametrize group size off the hardcoded
   256. Options: add `pack_qtip4_group_g128` / thread a `group: usize` through
   `qtip_quantize_dequant` + the pack/unpack fns. The 3-bit `pack8_3bit`
   bitstream is already chunk-based (32 chunks of 8 → 96 B for 256); at 128 it's
   16 chunks → 48 B. The 4-bit nibble pack is `k/2` bytes — trivially 64 B at 128.
   FWHT: use `gen_fwht_signs(.., 128)` + a `cpu_fwht_128` (confirm one exists /
   the FwhtG128 path already has it; MQ4G128 uses it).
2. **On-disk driver** (`hipfire-quantize` `pack_qtip_real_tensors`): today it
   packs `k % 256 == 0`. Add a G128 pass for `k % 128 == 0 && k % 256 != 0` (or
   make the whole driver group-parametric and prefer G256 when both divide).
   Tag `Qtip3G128` / `Qtip4G128`. `--format qtip3`/`qtip4` should auto-pick G128
   for the odd linears (like MQ4 auto-splits G256/G128).
3. **QuantType bytes** (`hipfire-quant-format`): reserve `Qtip3G128`,
   `Qtip4G128` codes (next free after 42) + `from_code` + stability test.
4. **DType + rotation** (`rdna-compute` / `hipfire-dispatch`): `DType::Qtip3G128`,
   `DType::Qtip4G128`; `size()` byte-level arm; `dtype_arch_predicate` → Always;
   `dtype_rotation_plan` → **`FwhtG128`** (already exists); `dtype_post_rotation_variant`
   → Prerotated.
5. **Kernels**: `gemv_qtip3g128` + `gemv_qtip4g128` (+ `_residual`). Copy the
   G256 kernels; change `groups_per_row = K / 128`, the group byte stride
   (52 / 68), and the per-thread tiling (128 weights/group: e.g. 16 threads × 8,
   or 32 threads × 4). Trellis recurrence + `qtip_decode` codebook unchanged.
6. **Dispatch** (`hipfire-dispatch`): `KernelKey` variants +
   `for_gemv_{prerotated,residual,swiglu_residual}` maps + the three launch arms.
7. **Loader** (`hipfire-runtime/hfq.rs`): `WeightTensor` arms for the new
   quant_type bytes (mirror the `42 =>` qtip4 arm).
8. **Catalog** (`docs/model-support.toml`) + `dtype_for_quant_type` +
   `weights.rs` preflight/route bridge; regenerate model-support.
9. **Parity + gate**: `parity_gemv_qtip4g128` (GPU vs CPU trellis oracle at
   group=128), coherence-gate-dflash.

## Method reminder (from the G256 work)

Add the DType variant, `cargo build` to surface **exhaustive** matches (`size`,
`dtype_arch_predicate`); the **functional** dispatch/loader sites use wildcards,
so `grep` every `Qtip4G256` reference to find them.

## Open questions

- Is a `cpu_fwht_128` already present (FwhtG128 path), or does it need adding?
- Trellis quality at group=128 vs 256 — measure recon/KLD; the shorter groups
  reset the trellis more often. Expected negligible; confirm on a real 896 model.
- Scope: ship both `qtip3g128` + `qtip4g128`, or just `qtip4g128` first (qtip4
  is the newer, and 4-bit decode is the more likely deployment)?
- Whether to make the driver fully group-parametric (auto G256-or-G128 per
  tensor, like MQ4) vs two explicit passes.

## Not worth it if

QTIP deployment only ever targets ÷256-hidden models (most ≥1B models: Llama-3.2
2048, Qwen3-0.6B 1024, etc.). G128 matters specifically for the sub-1B / 896-class
tier. Prioritize accordingly.
