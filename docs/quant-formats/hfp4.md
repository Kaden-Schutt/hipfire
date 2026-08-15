# HFP4 — Hipfire FP4 (RDNA-oriented E2M1 family)

**Status (member metadata — INDEX truth labels):**

| Variant | Wire qt | State |
|---|---|---|
| `HFP4G32` | 21 | **shipped / ref-pinned** — encoder, GEMV, WMMA prefill on RDNA3/4 |
| `MFP4G32` | 24 | **shipped / ref-pinned** — HFP4G32 + offline FWHT (MQ4-class drop-in) |
| `MFP4G32Lloyd` / `MFP4G32P` / `MFP4G32E8` (+ SoA) / `MFP3G32E8` / `MFP2G32E8` | 32–37 | **shipped / ref-pinned** encoders + selective kernels; experimental opt-in qualifier — not CLI defaults |
| HFP4G16 / G64 / MX / NV aliases, online rotation (`MFP4G32R`), HFP8 | see ID table | **planned** (unbuilt) or **reassigned** wire slots — do not claim shipped |

Broader production index: [`docs/QUANTIZATION.md`](../QUANTIZATION.md).
CLI: [`docs/QUANTIZE.md`](../QUANTIZE.md) · quantizer `crates/hipfire-quantize` ·
kernels `kernels/src/gemv_hfp4g32*.hip`, `gemm_*_hfp4g32_wmma*.hip`.

---

## Mission

HFP4 is hipfire’s RDNA-oriented answer to OCP **MXFP4** and NVIDIA **NVFP4**.
Those reference formats target non-AMD silicon and do not exploit RDNA-specific
ISA levers (`v_ldexp_f32` for UE8M0 dequant, native FP8 WMMA on gfx12,
`V_PERMLANE16` scale broadcast, VOPD dual-issue on RDNA3+).

HFP4 keeps the **OCP E2M1** nibble lattice (same sixteen codes). There is
**no** implemented MXFP4 or NVFP4 checkpoint importer in the quantizer today —
foreign checkpoints require full **re-quantization** into HFP4/MFP4. NVFP4’s
G16 / E4M3 scaling cannot generally become HFP4 G32 / UE8M0 by transforming
scales while preserving codes. Documented so other AMD-side projects can adopt
the layout.

## Format taxonomy

```
HFP4G32   — E2M1 + UE8M0 g32 + FP16 row scale       canonical (qt 21)
MFP4G32   — HFP4G32 + offline FWHT-256              drop-in MQ4-class (qt 24)
MFP4G32P  — MFP4 with E4M3 (non-PoT) block scale    experimental (qt 33)
MFP4G32E8 — E8 lattice codewords in MFP4+P frame    experimental (qt 34+)
MFP4G32Lloyd — per-tensor 16-entry Lloyd codebook   experimental (qt 32)

# Reserved / not product defaults
HFP4G16   — was qt 22 reservation; **ID 22 reassigned to TidI32** (DeepSeek tables)
HFP4G64   — **reserved** qt 23 (ablation; not built as product)
HFP4G32MX / HFP4G16NV / HFP8* — former 25–27 reservations; still **reserved**, unbuilt
MFP4G32R (online-R) — former qt 29; **reassigned to PARO4G128T** — online-R has no live reserved ID
```

## Element format (OCP E2M1)

4-bit signed FP. Sixteen codes; eight magnitudes:

| nibble | value | nibble | value |
|:------:|:-----:|:------:|:-----:|
| 0000 | +0.0 | 1000 | −0.0 |
| 0001 | +0.5 | 1001 | −0.5 |
| 0010 | +1.0 | 1010 | −1.0 |
| 0011 | +1.5 | 1011 | −1.5 |
| 0100 | +2.0 | 1100 | −2.0 |
| 0101 | +3.0 | 1101 | −3.0 |
| 0110 | +4.0 | 1110 | −4.0 |
| 0111 | +6.0 | 1111 | −6.0 |

Locked to spec: changing magnitudes breaks RDNA4 hardware paths
(`v_cvt_pk_fp8_e2m1` where used), LUT-decode strategies, and MX/NV interop.

## Block scale — UE8M0

Every block of `g` elements (canonical **g = 32**) carries one **UE8M0** byte:
unsigned exponent-only. Encoded `e ∈ [0, 254]` means `2^(e − 127)`. `0xFF` is
block-NaN.

On RDNA, `v_ldexp_f32(acc, e − 127)` is one VALU op (no multiply). UE8M0 alone
is coarse (OCP MX notes ~1–2% PPL loss vs FP16 in literature), so HFP4 adds a
FP16 **row** scale.

## Per-row second-level scale (FP16)

Each weight row has a 16-byte aligned header with `row_scale_a`, a reserved
`row_scale_b` slot, block count, and `format_flags`. Effective dequant (v1):

```
value = row_scale_a * 2^(block_e − 127) * E2M1_LUT[nibble]
```

`row_scale_a` hoists outside the K loop (GEMV finalize or WMMA output stage).
**`row_scale_b` is reserved:** the encoder always writes zero, and current
kernels read only `row_scale_a` — it does **not** control or skip a second
scale today. Format flag bit 1 remains defined for a future dual-output use.

Finer blocks (g=32 vs HFQ/MQ G256) catch within-row outliers a single row scale
would clip; the FP16 row scale restores dynamic range without paying a multiply
in the inner dequant loop beyond `ldexp`.

## Byte layout — HFP4G32 (canonical)

For a row of K elements (**format** is K%32-aligned; **v1 GEMV/WMMA path
requires K%256==0** — quantizer and loaders enforce the kernel constraint):

```
Per row (16 B, aligned):
  +0  : f16  row_scale_a
  +2  : f16  row_scale_b      // reserved; encoder writes 0; kernels ignore
  +4  : u8   block_count_lo   // K/32 low
  +5  : u8   block_count_hi
  +6  : u8   format_flags
              bit 0: rotation present
              bit 1: row_scale_b used
              bits 2-3: rotation_kind
                00 = off
                01 = offline FWHT (MFP4)
                10 = online block-diag-128 (planned)
                11 = online HadaCore-16 (planned)
              bits 4-7: reserved
  +7  : u8   reserved
  +8  : u32  reserved
  +12 : u32  reserved

Per block × (K/32):
  +0  : u8   block_e          // UE8M0
  +1  : u8[16] nibbles        // 32 E2M1 codes; low nibble = even index
```

**Total per row:** `16 + 17 × (K / 32)` bytes.
**Effective bpw:** 4.25 from per-block payload + `128/K` from the row header.

MFP4 stamps `format_flags = 0x05` (rotation present + offline FWHT) after the
same row pack; runtime uses the shared MQ FWHT signs (seeds 42 / 1042).

### Nibble packing

Byte `b` encodes elements `2b` (low nibble) and `2b+1` (high nibble). Matches
HFQ4 extract patterns and MX/NVFP4 wire ordering.

## Quantization recipe (HFP4G32)

For each row `W[K]`:

1. `row_scale_a = max_abs(W) / 6.0` (E2M1 max ±6).
2. Normalize: `W_n = W / row_scale_a`.
3. Per 32-element block: pick UE8M0 `e` so `2^e` covers `block_max/6`; divide;
   round-to-nearest on the E2M1 lattice; pack nibbles.

Round-trip per-block max-abs error is bounded by the **local half-gap** of the
E2M1 lattice under nearest rounding. Gaps are not uniform: adjacent magnitudes
include steps of 0.5, 1.0, and **2.0** (between 4 and 6), so the largest
half-gap is **1.0** scaled unit. A valid **global** bound is therefore
`row_scale_a · 2^(block_e − 127) · 1.0` (not `· 0.5`). Example: a normalized
value of 5 can round to 4 or 6 with unit error 1.0 before scales.

MFP4 applies offline FWHT-256 on the row (same signs as MQ4) before the HFP pack.

## Rotation modes

| Mode | Storage | flags bits 2–3 | Status |
|------|---------|----------------|--------|
| `off` | unrotated | 00 | HFP4 default |
| `offline-fwht` | pre-rotated weights; `mq_rotate_x` on x | 01 | **MFP4 shipped** |
| `online-bd128` | unrotated + fused online Hadamard | 10 | planned |
| `hadacore-16` | WMMA-fragment Hadamard | 11 | planned / research |

Online modes block fused QKV/gate_up until fused-rotation siblings exist.
Offline mode reuses MQ infrastructure (`mq_rotate_x`, fused rmsnorm/silu+rotate,
signs1/signs2) unchanged.

## Quant-type IDs

| ID | Variant | Status |
|:--:|---------|--------|
| 21 | `HFP4G32` | **shipped / ref-pinned** |
| 22 | **`TidI32`** (DeepSeek tid2eid) | **reassigned** from former HFP4G16 reservation — do not emit HFP4G16 as 22 |
| 23 | HFP4G64 | **reserved** ablation (not product) |
| 24 | `MFP4G32` | **shipped / ref-pinned** |
| 25–27 | MX/NV/HFP8 reservations | **reserved** / unbuilt (**planned** product intent only) |
| 28 | PARO4G128 (unrelated family) | **shipped / ref-pinned** load path — listed so IDs are not squatted |
| 29 | PARO4G128T (unrelated) | **reassigned** from former `MFP4G32R` online-R slot |
| 30 | MQ4G256Lloyd (unrelated) | **shipped / ref-pinned** research MQ opt-in — renumbered off 21 |
| 32 | `MFP4G32Lloyd` | **shipped / ref-pinned** impl; experimental opt-in |
| 33 | `MFP4G32P` | **shipped / ref-pinned** impl; experimental opt-in |
| 34 | `MFP4G32E8` | **shipped / ref-pinned** impl; experimental opt-in |
| 35 | `MFP4G32E8SOA` | **shipped / ref-pinned** impl; experimental opt-in |
| 36 | `MFP3G32E8` | **shipped / ref-pinned** impl; experimental cold-tier opt-in |
| 37 | `MFP2G32E8` | **shipped / ref-pinned** impl; experimental cold-tier opt-in |

**Current reserved wire IDs:** 23 and 25–27. IDs 22, 29, and 30 were reassigned.
Planned online-R / HFP8 variants have no reserved ID beyond this table.
Future PRs must not reuse reserved HFP slots for unrelated formats except the
documented reassignments above.

## Configurability

### Runtime env (live)

| Variable | Effect |
|----------|--------|
| `HIPFIRE_FP8_WMMA=1` | **Live.** Default **off**. On gfx12 with WMMA w32, enables the FP8-WMMA HFP path when `batch_size >= FP8_WMMA_MIN_BATCH` (**1024**); otherwise FP16-WMMA is used. Sources: `feature_flags.rs`, `dispatch.rs`, `gemm.rs`. |

There are **no** current source definitions for `HFP4_BLOCK_SIZE`,
`HFP4_SCALE_FORMAT`, `HFP4_SECOND_LEVEL`, `HFP4_ROTATION_KIND`, `HFP4_R`,
`HIPFIRE_HFP_USE_FP8_WMMA`, `HIPFIRE_HFP_ROTATION`, or `HIPFIRE_HFP_VARIANT`.
Treat any of those names as **planned** only — do not document them as live
controls.

### Quantize CLI / binary

| Flag | Effect |
|------|--------|
| `--format hfp4` (`hfp4g32`, `hf4p`, `fp4`) | HFP4G32 |
| `--format mfp4` (`mfp4g32`, `mf4p`) | MFP4G32 + offline FWHT |
| `--format mfp4p` / `mfp4e8` / `mfp4l` / … | experimental siblings (binary) |

Thin `hipfire quantize` help emphasizes mq/hf defaults; HFP/MFP are available on
the full `hipfire-quantize` format alias set.

## Runtime support boundaries

| Concern | Behavior |
|---------|----------|
| Batched prefill | `HFP4G32` / `MFP4G32` are runtime batch-eligible under `is_batchable_la` on WMMA arches (gfx11/115x/12) only — runtime batch eligibility, **not** registry admission ([`admissions.yml`](../admissions.yml)) |
| Decode GEMV | `gemv_hfp4g32` family (scalar / arch variants); MFP uses prerotated path |
| Fused QKV / gate_up prefill | WMMA-only HFP keys on RDNA3/4 — no scalar fused fallback |
| K alignment | Loaders refuse K%256≠0 for v1 kernels |
| CDNA / pre-WMMA | Fall back or reject batch path; do not claim parity |

## v1 correctness-anchor inner loop

Mirrors HFQ4G256 multi-accumulator structure with E2M1 LUT + `ldexpf` instead of
INT4×scale+zp:

```c
// preamble: shared E2M1 lut[16]
// per block: sc = row_scale_a * ldexpf(1.0f, block_e - 127);
// per nibble: value = sc * (float)lut[nibble];
```

No zero-point term (E2M1 is signed).

## Validation (how to claim quality/perf)

Do **not** treat the historical checklist below as a universal gate (see
[VALIDATION.md](../VALIDATION.md)). For HFP work, prefer:

1. CPU round-trip max-abs ≤ local half-gap (global ≤ `row_scale · 2^(e−127) · 1.0`).
2. CPU vs kernel element error on fixed (M,K) tensors.
3. NRMSE / KLD vs FP16 or MQ4 baselines on **named** models — report as
   **measured** with fixture identity.
4. Prefill path proof: rocprof / internal profile shows HFP WMMA symbols when
   claiming WMMA prefill.
5. No coherence-gate battery as product admission.

Aspirational speed numbers (e.g. TFLOPS vs third-party MXFP4 demos) are
**not** route-certified here.

## Comparison to neighbors

| Property | HFQ4G256 | MQ4G256 | MXFP4 (strict) | NVFP4 | **HFP4G32** |
|----------|:---:|:---:|:---:|:---:|:---:|
| Element | INT4 | INT4 + FWHT | E2M1 | E2M1 | E2M1 |
| Block size | 256 | 256 | 32 | 16 | 32 |
| Block scale | f32 affine scale + f32 min | f32 affine scale + f32 min | UE8M0 | E4M3 | UE8M0 |
| Secondary scale | none | none | none | FP32 tensor | FP16 row (`row_scale_a`; `row_scale_b` reserved) |
| ~bpw | 4.25 | 4.25 | 4.25 | ~4.5 | ~4.25 |
| Rotation | none | offline FWHT (`R=D2·H·D1/16`) | none | none | offline FWHT on MFP4; online **planned** |
| MX/NV import | re-quant | re-quant | **no importer** (re-quant) | **no importer** (re-quant; G16/E4M3 ≠ G32/UE8M0 scale-only) | re-quant only |

## Provenance / references

- OCP Microscaling Formats (MX) v1.0 — <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
- AMD ROCm — MXFP4/MXFP6 quantization — <https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html>
- AMD ROCm — MXFP4 online rotation — <https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-online-rotation/README.html>
- NVIDIA NVFP4 — <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- rdna4-wmma-guide (third-party MXFP4 demo) — <https://github.com/JohnTDI-cpu/rdna4-wmma-guide>
- HadaCore — <https://pytorch.org/blog/hadacore/>
- SpinQuant (arXiv 2405.16406) — <https://arxiv.org/abs/2405.16406>
- AMD RDNA4 ISA — <https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture>

HFP4 is an original hipfire layout specification; E2M1 magnitudes follow OCP MX.
MQ/FWHT attribution: MagnumQuant rotation (seeds 42/1042) shared with the MQ
weight family in QUANTIZATION.md.
