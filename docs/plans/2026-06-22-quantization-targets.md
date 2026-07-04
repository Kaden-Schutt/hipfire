# hipfire quantization targets — the committed matrix

Status: **decision** — 2026-06-22. Owner: chaingun. Supersedes the ad-hoc format
sprawl (mq3/mq4/mq6/oq4/awq/lloyd/qtip/kvarn/asym2-4) with a small, hardware-aligned
target set.

## The organizing principle

Quantization has three independent axes; conflating them is what produced the sprawl.

1. **Activation precision = the quality bottleneck.** The oq4 (W4A4) work established
   this directly: W4A4 is coherent-but-fragile *because of the int4 activations*, not
   the weights (SmoothQuant/AWQ helped, LDLQ on weights added nothing). So the
   **activation tier is the primary dial**, mapped to the integer WMMA path it can use.
2. **Weight bits = a memory dial, quality-cheap to 4-bit.** W4A16 (mq4, FWHT-rotated
   affine int4) is the quality-gated default and is excellent. Shaving weight bits
   below 4 (mq3/mq2) hits a quality cliff for little memory — only worth it as the
   *bulk* of a mixed-precision scheme (salient channels/layers kept higher).
3. **KV precision = orthogonal**, covered by the deferred-hierarchical ladder
   (f32 / fp16 / q8 / KVarN-4 / 2-bit cold; see
   [hierarchical-kv-followups](2026-06-22-hierarchical-kv-followups.md)).

## Hardware: integer WMMA is the lever

Fleet: RDNA3 (gfx1100/1103), RDNA3.5 (gfx1151), RDNA4 (gfx1201), all wave32. Every
arch has `v_wmma_i32_16x16x16_iu8` and `iu4`; **only gfx1201 has fp8.** Decision:
**iu8 is the canonical A8 path** — it is the lowest common denominator across the
fleet, and software-fp8 emulation on machines with native integer matrix units is
pure overhead. fp8 stays an optional RDNA4-only fast-path, not a format.

Throughput vs f16 WMMA (verify exact per-arch rates with the
`hipfire-amd-matrix-calculator` skill): iu8 ≈ 2×, iu4 ≈ 4×.

## The affine-vs-codebook partition (mechanical roster split)

Integer WMMA multiplies int×int→int32, with the per-group weight scale and the
activation scale applied to the **int32 accumulator AFTER** the matmul (scales never
enter the WMMA). That only works for **uniform/affine** quants, where the code *is*
the integer level. This cleanly partitions every format:

- **Affine quants** (uniform mq4, oq4-style int4, int8) → ride integer WMMA (A8/A4).
  These are the production formats.
- **Codebook / non-uniform** (mq4-Lloyd, QTIP) → the code indexes arbitrary fp16
  levels → cannot feed an integer matmul → inherently **A16** (dequant→f16, f16 WMMA).
  Keep only where their quality edge pays for the lost integer speedup.

## The committed matrix

| Format | Weights | Act / WMMA | Throughput | Lane |
|---|---|---|---|---|
| **W4A16** (uniform mq4) | affine int4 | f16 WMMA | 1× | quality default, any arch |
| **W4A8** | affine int4 → **expand to int8** | iu8 | ~2× | **production sweet spot (build)** |
| **W8A8** | int8 | iu8 | ~2× | max-fidelity integer |
| **W4A4** (oq4) | affine int4 | iu4 | ~4× | opt-in 4×, rotation+SmoothQuant-gated |
| Lloyd / QTIP | codebook | f16 (A16 only) | 1× | niche — keep only on proven quality edge |
| W2/W3, A2 | — | — | — | **mixed-precision only** (demoted) |

KV axis (orthogonal): f32 / fp16 / **q8 (serve default)** / KVarN-4 / 2-bit cold.

### W4A8 and W8A8 are ONE kernel

The key structural win: both ride the **same iu8 WMMA core**, differing only in the
weight-load prologue:
- **W8A8**: load int8 weights directly → iu8 WMMA.
- **W4A8**: load 4-bit codes, **expand nibble→int8 in registers/LDS on load** → same
  iu8 WMMA. Storage and global bandwidth stay 4-bit; only the in-flight tile is int8.

W4A8 trades oq4's iu4 4× for iu8 2×, but keeps W4 memory *and* gets A8 quality — it
**dominates oq4 for production** (same weight footprint, far better quality, real
speedup over mq4's f16 path). oq4/W4A4 remains the opt-in "I need the 4× and earned
it with rotation+SmoothQuant" tier.

## What exists vs the build

**Exists:**
- iu8 WMMA core: `kernels/src/gemm_iu8_i32_wmma.hip` + `gemv_iu8_i32.hip`, dispatched
  in hipfire-rdna (signed int8, zero-LDS / gfx1103-safe). **Library primitive — not
  yet wired into the qwen35 forward.**
- iu4 WMMA core + oq4 (W4A4) end-to-end (affine int4 emit, int8 activation quant,
  SmoothQuant sidecar) — reusable front-end for W4A8's A8 side.
- W4A8 offline quality exploration: `crates/hipfire-quantize/examples/{quant_opus_mqplus,
  quant_wxax_explore,quant_w4a4_improve}.rs`, `docs/quant-formats/opus-mqplus-eval-plan.md`.

**Build — W4A8 end-to-end (incremental; reuses oq4 machinery + the iu8 core):**
1. **GEMM parity** (first step): affine-int4 weights → expand→int8 → `gemm_iu8_i32_wmma`
   → dequant by (w_group_scale × a_scale), vs an f32 reference. Validate nibble-expand
   + scale-after-accumulate numerics on the existing iu8 core. Add `parity_w4a8`.
2. **On-GPU nibble-expand prologue**: a load path that reads 4-bit codes and expands
   to int8 in-tile (storage/bandwidth stay 4-bit) feeding the iu8 core — vs the host
   expand in step 1.
3. **Quantizer emit**: a W4A8 QuantType (affine int4 weights + per-group scale; reuse
   oq4's emit + SmoothQuant; drop the A4 rotation pressure — A8 needs far less).
4. **Loader + forward dispatch**: wire the iu8 GEMM (+ activation int8 quant per
   token/row) into `kv_cache_attention_dispatch`-adjacent GEMMs (qkv, gate/up, o,
   down), behind a format flag; parity-gate against W4A16.
5. **Validate**: KLD vs BF16 (expect near-W4A16 quality, ~2× the f16 GEMM), coherence
   gate, and a fresh-process perf A/B per the perf-benchmarking rules.

## Consolidation / migration

- **Keep first-class:** W4A16 (mq4 default), W8A16, **W4A8 (build)**, W8A8, W4A4 (oq4).
- **Demote to mixed-precision-only:** W2/W3 weights (mq2/mq3 — lost cause standalone,
  the coherence gate shows mq3 wobbling), A2 activations (no native iu2 WMMA; viable
  only as outlier-preserving mixed precision).
- **Codebook (Lloyd/QTIP):** A16-only; retain a format only with a demonstrated
  quality edge over uniform-W4 at equal bits, else retire.
- **fp8:** optional gfx1201-only fast-path under the A8 umbrella, not a distinct format.

## Notes for implementers

- gfx1103 LDS hazard: the iu8/iu4 cores are deliberately zero-LDS (register-tiled).
  Keep the nibble-expand prologue LDS-free too.
- Scales NEVER enter the WMMA — int32 accumulate, then dequant by w_scale·a_scale.
  Asymmetric (zero-point) weights need the zp correction term folded into the dequant.
- iu8 core requires K % 16 == 0.
