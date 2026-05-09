# HFP4 — Hipfire FP4 Quantization (RDNA-optimal E2M1 family)

**Status:** v1 draft. First-kernel scope: HFP4G32 GEMV correctness anchor on gfx1100. WMMA-FP8 hero kernel and rotation variants are v2/v3 (deferred — see `docs/QUANTIZATION.md` for the production format index).

**Related**: [`docs/QUANTIZATION.md`](../QUANTIZATION.md) (production format index) · [`docs/QUANTIZE.md`](../QUANTIZE.md) (CLI usage) · `crates/hipfire-quantize/src/main.rs` (quantizer) · `kernels/src/gemv_hfp4g32*.hip` (kernels).

---

## Mission

HFP4 is hipfire's RDNA-optimal answer to MXFP4 (OCP) and NVFP4 (Blackwell). Both reference formats are designed for non-AMD silicon and don't exploit AMD-specific RDNA ISA features (`v_ldexp_f32` for free UE8M0 dequant, native FP8 WMMA on gfx1201, V_PERMLANE16 cross-lane scale broadcast, VOPD dual-issue on RDNA3+).

HFP4 keeps the OCP E2M1 element wire-compatible — so MXFP4 / NVFP4 checkpoints can be re-quantized to HFP4 by transforming only the scale layer, not the codes — while specifying scale in the way that's cheapest to dequant on RDNA. The format is documented here so other AMD-side projects can adopt it.

## Format taxonomy

```
HFP4G16   — E2M1 + UE8M0 g16 + FP16 row scale       (NV-aligned, FP16-WMMA-K aligned)
HFP4G32   — E2M1 + UE8M0 g32 + FP16 row scale       canonical (FP8-WMMA-K aligned)
HFP4G64   — E2M1 + UE8M0 g64 + FP16 row scale       (best amortization, RDNA1/2 sweet spot)
MFP4G32   — HFP4G32 + offline FWHT rotation         (drop-in MQ4 replacement; v1.5)
MFP4G32R  — HFP4G32 + online block-diag-128         (AMD recipe; v3)
HFP4G32MX — HFP4G32 with no row scale               (strict OCP MXFP4 interop alias; v2)
HFP4G16NV — HFP4G16 + E4M3 scale + FP32 tensor      (strict NVFP4 interop alias; v2)
```

## Element format (locked to OCP E2M1)

4-bit signed FP. Sixteen codes; eight magnitudes:

| nibble | sign | exp | mant | value | nibble | sign | exp | mant | value |
|:------:|:----:|:---:|:----:|:-----:|:------:|:----:|:---:|:----:|:-----:|
| 0000   | +    | 0   | 0    | +0.0  | 1000   | −    | 0   | 0    | −0.0  |
| 0001   | +    | 0   | 1    | +0.5  | 1001   | −    | 0   | 1    | −0.5  |
| 0010   | +    | 1   | 0    | +1.0  | 1010   | −    | 1   | 0    | −1.0  |
| 0011   | +    | 1   | 1    | +1.5  | 1011   | −    | 1   | 1    | −1.5  |
| 0100   | +    | 2   | 0    | +2.0  | 1100   | −    | 2   | 0    | −2.0  |
| 0101   | +    | 2   | 1    | +3.0  | 1101   | −    | 2   | 1    | −3.0  |
| 0110   | +    | 3   | 0    | +4.0  | 1110   | −    | 3   | 0    | −4.0  |
| 0111   | +    | 3   | 1    | +6.0  | 1111   | −    | 3   | 1    | −6.0  |

Locked to spec: changing the magnitudes breaks RDNA4 hardware path (`v_cvt_pk_fp8_e2m1`), the LUT-decode lever (gfx12 dominant strategy), and MX/NVFP4 interop.

## Block scale — UE8M0

Every block of `g` elements (g ∈ {16, 32, 64}) carries **one UE8M0 byte** = unsigned 8-bit exponent-only scale. The encoded value `e ∈ [0, 254]` represents `2^(e − 127)`. The reserved code `0xFF` encodes block-NaN (every element in the block is NaN).

Why UE8M0 (RDNA-specific justification): on every RDNA tier `v_ldexp_f32(acc, e − 127)` is one VALU op (no multiply, no constant load). NVFP4's E4M3 scale costs one FP8→FP16 conversion + one FP16 multiply per block; on Hopper this is amortized by the FMA pipeline, but on RDNA the `v_ldexp` path is strictly faster.

UE8M0 alone is too coarse — the OCP MX paper documents 1–2% perplexity loss vs FP16 — so HFP4 layers a FP16 row-scale on top.

## Per-row second-level scale (FP16, the AMD-specific lever)

Each weight row carries a 16-byte aligned header containing two FP16 row scales (`row_scale_a`, `row_scale_b`) plus format flags. Effective dequant per element:

```
value = row_scale_a * 2^(block_e − 127) * E2M1_LUT[nibble]
```

The `row_scale_a` multiply hoists outside the K loop (applied once per row at GEMV finalize, or folded into the WMMA output stage on gfx11/gfx12). `row_scale_b` exists for fused dual-output kernels where the same input row produces two output columns with different per-row scales (e.g., gate+up FFN, qkv fused projection). When `row_scale_b == 0` the kernel skips the second scale.

Why this beats single-level FP16 row-scale (HFQ-style): finer block granularity (g=32 vs 256) catches outlier groups within a row that a row-shared scale alone would clip or under-quantize. Why this beats single-level UE8M0 (MX-strict): the FP16 row-scale gives full FP16 dynamic range without paying for it in the dequant inner loop.

## Byte layout — HFP4G32 (canonical)

For a row of K elements (K must be a multiple of 32):

```
Per row (16 B, aligned):
  +0  : f16  row_scale_a      // primary FP16 second-level scale
  +2  : f16  row_scale_b      // secondary scale for fused-dual outputs (else 0)
  +4  : u8   block_count_lo   // K / 32, low 8 bits
  +5  : u8   block_count_hi   // K / 32, high 8 bits  (supports K up to 16M)
  +6  : u8   format_flags     // bit 0: rotation present
                              // bit 1: row_scale_b used
                              // bits 2-3: rotation_kind
                              //   00 = off
                              //   01 = offline FWHT (existing MQ pattern)
                              //   10 = online block-diag-128 (v3)
                              //   11 = online HadaCore-16 (v3)
                              // bits 4-7: reserved
  +7  : u8   reserved
  +8  : u32  reserved         // future: D_diag pointer offset (joint-D smoothing)
  +12 : u32  reserved

Per block × (K / 32):
  +0  : u8   block_e          // UE8M0 power-of-2 exponent
  +1  : u8[16] nibbles        // 32 E2M1 codes, low nibble = even index, high nibble = odd
                              // ordering bit-identical to MX/NVFP4 wire format
```

**Total per row**: `16 + 17 × (K / 32)` bytes.

For a 5120-dim Qwen3.5-9B `q_proj` row: `16 + 17 × 160 = 2736 B` (vs MQ4G256's 2720 B; +0.6% with finer scale granularity and FP16 row-scale).

**Effective bpw**: `(17 × 8) / 32 = 4.25 bpw` from the per-block payload, plus `128 / K` extra bits per weight from the per-row header (negligible for large K — for K=5120, header overhead is 0.025 bpw).

### Worked example

A row with K=64 (two blocks):

```
Offset  Bytes                                       Meaning
------  ------------------------------------------  ----------------------------------
0x00    BC 3F  00 00  02 00  00 00  00 00 00 00     row_scale_a = 0.99 (FP16: 0x3FBC)
        00 00 00 00                                 row_scale_b = 0
                                                    block_count = 2
                                                    format_flags = 0x00 (no rotation)
                                                    reserved
0x10    7F  31 22 13 04 F5 E6 D7 C8 31 22 13 04 ... block 0: block_e=127 (=2^0=1.0)
        ...                                         + 16 bytes of packed nibbles
0x21    7E  ...                                     block 1: block_e=126 (=2^-1=0.5)
        ...                                         + 16 bytes of packed nibbles
0x32    EOF
```

The second block's `block_e=126` = `2^-1`, meaning every element in that block is half the magnitude of an equivalent block 0 nibble. The `row_scale_a` then scales both blocks' final values uniformly.

### Nibble packing convention

Within each 16-byte payload, byte `b` at position `b ∈ [0, 16)` encodes elements at indices `2b` (low nibble = `byte & 0x0F`) and `2b + 1` (high nibble = `byte >> 4`). This matches the existing HFQ4 packing (kernels can reuse the bit-extract pattern verbatim) and is bit-identical to MX/NVFP4 wire format.

## Quantization recipe

For each row of FP16 weights `W[K]`:

1. Compute the FP16 row scale: `row_scale_a = max_abs(W) / 6.0` (E2M1 max code is ±6.0).
2. Normalize: `W_n[i] = W[i] / row_scale_a`. Now `|W_n[i]| ≤ 6.0`.
3. For each 32-element block:
   a. Compute the UE8M0 block exponent. `block_max = max_abs(W_n[block])`. Pick `e` such that `2^e` is the largest power-of-2 ≤ `block_max / 6.0`. Encode as UE8M0 with bias 127: `block_e = clamp(round(log2(block_max / 6.0)) + 127, 0, 254)`.
   b. Apply block scale: `W_b[i] = W_n[block][i] / 2^(block_e − 127)`.
   c. Round-to-nearest in the E2M1 lattice: `nibble[i] = argmin_k |E2M1_LUT[k] − W_b[i]|`.
   d. Pack nibbles into 16 bytes (low nibble = even index).

Round-trip per-block max-abs error is bounded by `row_scale_a · 2^(block_e − 127) · 0.5` (half the smallest E2M1 step at the scale's working magnitude).

## Rotation modes (configurable, default `off` in v1)

| Mode | Storage | Rotation kind bits | When used |
|------|---------|---------------------|-----------|
| `off` | unrotated weights, plain `x` | 00 | v1 default; calibration baseline; well-conditioned models |
| `offline-fwht` | pre-rotated weights; `x_rot` computed once per layer via existing `mq_rotate_x` | 01 | drop-in MQ4 replacement; ships v1.5 as `MFP4G32` |
| `online-bd128` | unrotated weights; per-block 128-elt block-diag Hadamard fused with dequant | 10 | v3 — matches AMD ROCm blog recipe; requires Stiefel-manifold calibration |
| `hadacore-16` | unrotated weights; 16×16 Hadamard via WMMA fragments | 11 | v3 research path; gfx1201 only |

Online modes (`10`, `11`) block fused QKV/QKVZA/gate_up kernels until their fused-rotation siblings exist. Offline mode (`01`) reuses the existing MQ infrastructure (`mq_rotate_x`, `fused_rmsnorm_mq_rotate`, `fused_silu_mul_mq_rotate`, signs1/signs2 buffers) without modification.

## Quant-type IDs (`.hfq` file format)

| ID | Variant | Status |
|:--:|---------|--------|
| 21 | `HFP4G32` | v1 — first kernel target |
| 22 | `HFP4G16` | v1.5 ablation |
| 23 | `HFP4G64` | v1.5 ablation |
| 24 | `MFP4G32` | v1.5 (HFP4G32 + offline FWHT) |
| 25 | `HFP4G32MX` | v2 (strict OCP MXFP4 interop alias) |
| 26 | `HFP4G16NV` | v2 (strict NVFP4 interop alias) |
| 27 | `HFP8E4M3G32` | v2 HFP8 family |
| 28 | `HFP8E5M2G32` | v2 HFP8 family |
| 29 | `MFP4G32R` | v3 (online block-diag rotation) |

IDs 21–29 are reserved at v1 ship time. Future PRs MUST NOT squat these IDs even if the corresponding variant has not yet shipped.

## Configurability

### Compile-time (kernel macros, set per-kernel-source)

| Macro | Values | Default |
|-------|--------|---------|
| `HFP4_BLOCK_SIZE` | 16 / 32 / 64 / 128 | 32 |
| `HFP4_SCALE_FORMAT` | `UE8M0` / `E4M3` / `FP16` | `UE8M0` |
| `HFP4_SECOND_LEVEL` | 0 / 1 | 1 |
| `HFP4_ROTATION_KIND` | 0–3 (matches `format_flags` bits 2–3) | 0 |
| `HFP4_R` | 1 / 2 / 4 multirow | 1 |

### Runtime (env vars, matching existing `HIPFIRE_*` taxonomy)

| Variable | Values | Effect |
|----------|--------|--------|
| `HIPFIRE_HFP_USE_FP8_WMMA` | 0 / 1 | gfx12 only; 0 forces FP16-WMMA path for ablation |
| `HIPFIRE_HFP_ROTATION` | `off` / `offline` / `online-bd128` | overrides per-tensor metadata for ablation |
| `HIPFIRE_HFP_VARIANT` | kernel-variant name | pin specific variant for debugging |

### Quantize CLI (`hipfire-quantize`)

| Flag | Effect |
|------|--------|
| `--format hfp4` (or `hfp4g32`, `hf4p`, `fp4`) | HFP4G32 quant — v1 default |
| `--format hfp4g16` | HFP4G16 ablation (v1.5) |
| `--format hfp4g64` | HFP4G64 ablation (v1.5) |
| `--format mfp4` (or `mfp4g32`) | HFP4G32 + offline FWHT rotation (v1.5) |
| `--block-size {16,32,64}` | override block size for HFP4 family |
| `--rotation {off,offline-fwht}` | override rotation mode |

## Per-arch dequant ISA targets

| Arch | Path | Lever | Expected |
|------|------|-------|----------|
| gfx1201/1200 (RDNA4) | LDS-LUT → FP8 → `v_wmma_f32_16x16x32_fp8_fp8` → `ldexpf` per block → `v_pk_mul_f16` per row | native FP8 WMMA | 55–65 TFLOPS (vs rdna4-guide's 40.8) |
| gfx1100/1151 (RDNA3 / 3.5) | SGPR-LUT → FP16 → `v_wmma_f32_16x16x16_f16` + VOPD-paired dequant + V_PERMLANE16 scale broadcast | WMMA-FP16 + VOPD | 70–75% WMMA-FP16 theoretical |
| gfx1030 (RDNA2) | SGPR-LUT → FP16 → `v_dot2_f32_f16` + V_PERMLANE16 + R=2 multirow | V_DOT2 | parity ±5% vs MQ4G256 |
| gfx1010 (RDNA1) | LDS-LUT → packed FP16 → `v_pk_fma_f16` only + `__shfl_sync` scale broadcast | BW-bound (no dot, no WMMA) | 85–90% peak BW |
| gfx906 (CDNA1) | wave64 + `v_pk_fma_f16` + LDS scale broadcast | BW-bound | parity with gfx1010 path |

The first kernel (`gemv_hfp4g32.gfx1100.hip`) is the **correctness anchor** — no WMMA, no FP8, no rotation — so byte-exact gating works against the existing infrastructure. WMMA-FP8 hero kernel is v2.

## v1 first-kernel inner loop

Mirrors the HFQ4G256 4-accumulator + tail-by-`g%4` invariant from `kernels/src/gemv_hfq4g256.hip:33–124`. Diff from HFQ4 is one substitution + drop the zero-point load:

```c
// In kernel preamble:
__shared__ _Float16 lut[16];
if (threadIdx.x < 16) {
    static const _Float16 E2M1[16] = {
         (_Float16)+0.0f, (_Float16)+0.5f, (_Float16)+1.0f, (_Float16)+1.5f,
         (_Float16)+2.0f, (_Float16)+3.0f, (_Float16)+4.0f, (_Float16)+6.0f,
         (_Float16)-0.0f, (_Float16)-0.5f, (_Float16)-1.0f, (_Float16)-1.5f,
         (_Float16)-2.0f, (_Float16)-3.0f, (_Float16)-4.0f, (_Float16)-6.0f,
    };
    lut[threadIdx.x] = E2M1[threadIdx.x];
}
__syncthreads();

// Per-block in inner loop:  sc = row_scale_a * ldexpf(1.0f, block_e - 127);
// Per-element:               value = sc * (float)lut[nibble];
```

Compared to HFQ4: `(sc * (float)((pk) & 0xFu) + zp)` becomes `sc * (float)lut[(pk) & 0xFu]`. The `zp` argument and `+ zp` term drop because E2M1 is signed.

## Validation

The first-kernel ships are gated by:

1. CPU round-trip: per-tensor max-abs error ≤ `row_scale_a · 2^(block_e − 127) · 0.5` per group.
2. CPU vs kernel: per-element error ≤ `1e-3 · max(|y_ref|)` on (M=512, K=2048) random tensors.
3. NRMSE vs FP16: per-tensor NRMSE < 5e-3 across all weight tensors of Qwen3.5-0.8B and Qwen3.5-9B.
4. Quality regression vs MQ4G256: HFP4G32 NRMSE ≤ MQ4G256 NRMSE on geomean across 50 prompts.
5. Coherence gate (`scripts/coherence-gate.sh`): fluent output on 9B + 27B HFP4 conversions.
6. Speed gate (`scripts/probe_commits.sh master HEAD`): decode tok/s within ±5% of MQ4G256 baseline.
7. Zero spills + register budget per `docs/skills/gfx-kernel-metadata`.

## Comparison to neighbors

| Property | HFQ4G256 | MQ4G256 | MXFP4 (strict) | NVFP4 | **HFP4G32** |
|----------|:---:|:---:|:---:|:---:|:---:|
| Element format | INT4 uniform | INT4 uniform (FWHT-rotated) | E2M1 | E2M1 | E2M1 |
| Block size | 256 | 256 | 32 | 16 | 32 |
| Block scale | FP16 (in FP32 slot) | FP32 | UE8M0 | E4M3 | UE8M0 |
| Block-scale dequant cost | 1 multiply | 1 multiply | 1 `ldexpf` (free) | 1 mul (1 FP8→FP16 + 1 FP16 mul) | 1 `ldexpf` (free) |
| Secondary scale | none | none | none | FP32 per-tensor | FP16 per-row |
| Per-group bytes (excl. row hdr) | 136 | 136 | 17 | ~10 | 17 |
| Effective bpw | 4.25 | 4.25 | 4.25 | 4.5 | 4.25 + ~0.025 (row hdr) |
| Rotation | none | offline FWHT (mandatory) | none | none | configurable |
| MX import path | re-quant | re-quant | byte-identical | n/a | re-scale only (codes preserve) |
| FP8-WMMA on gfx12 | n/a | n/a | external demo | n/a | first-class (v2) |
| Adoptable spec | HFQ-house | MQ-house | OCP standard | NV-only | HFP-spec (this doc) |

## References

- OCP Microscaling Formats (MX) Specification v1.0 — <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
- AMD ROCm Blog — High-Accuracy MXFP4, MXFP6 — <https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html>
- AMD ROCm Blog — Advanced MXFP4 with Online Rotation — <https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-online-rotation/README.html>
- NVIDIA — Introducing NVFP4 — <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- rdna4-wmma-guide (40.8 TFLOPS MXFP4 on R9700) — <https://github.com/JohnTDI-cpu/rdna4-wmma-guide>
- HadaCore — <https://pytorch.org/blog/hadacore/>
- SpinQuant (arXiv 2405.16406) — <https://arxiv.org/abs/2405.16406>
- AMD RDNA4 ISA Reference — <https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture>
