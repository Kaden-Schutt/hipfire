# WMMA / MFMA arch matrix

Operand shapes, builtin names, and lane layouts for every matrix
intrinsic hipfire dispatches. Source: ROCm 7's
`/opt/rocm/include/ck/utility/amd_wmma.hpp` and
`/opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_*`.
Verify against the live ROCm install before extending.

## fp16×fp16→fp32 16×16×16

The most common matmul shape in hipfire (HFQ4-G256 dequant → fp16
in LDS → WMMA into fp32 accumulator).

| Arch family | Wave | A-vec | B-vec | C-vec | Builtin | LLVM intrinsic |
|---|---|---|---|---|---|---|
| **gfx11** (RDNA3 — 7900 XTX, V620) | 32 | `<16 x fp16>` | `<16 x fp16>` | `<8 x f32>` | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | `llvm.amdgcn.wmma.f32.16x16x16.f16` |
| **gfx11** wave64 | 64 | `<16 x fp16>` | `<16 x fp16>` | `<4 x f32>` | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w64` | (CDNA-style ABI) |
| **gfx12** (RDNA4 — 9070 XT, R9700) | 32 | `<8 x fp16>` | `<8 x fp16>` | `<8 x f32>` | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` | `llvm.amdgcn.wmma.f32.16x16x16.f16.gfx12` |
| **gfx94x** (CDNA3 — MI300X) | 64 | n/a (MFMA) | n/a (MFMA) | `<4 x f32>` | `__builtin_amdgcn_mfma_f32_16x16x16_f16` | `llvm.amdgcn.mfma.f32.16x16x16.f16` |

**Key load-bearing differences gfx11 → gfx12:**

1. **K-packing per lane halves**: gfx11 packs 16 fp16s in each lane's
   A/B operand; gfx12 packs only 8. The K-tile striding in your LDS
   loads must change accordingly.
2. **kRepeat changes**: gfx11 has `kRepeat=2` (the 16-K is split
   across 2 lane-groups), gfx12 has `kRepeat=1` (the 8-K fits in one
   lane group).
3. **Builtin name suffix**: `_w32` → `_w32_gfx12`. Same arity (3
   operands), but the LLVM intrinsic is a different node — gfx11
   codegen patterns DO NOT match the gfx12 intrinsic.
4. **C-mapping (output → row/col)**: gfx11's mapping is
   `acc[j] = C[2*j + (tid>>4)][tid & 15]` (validated 2026-04-12 in
   `project_wmma_correctness_fix.md` after a 6-week silent
   corruption bug). **gfx12's mapping is unverified by hipfire** —
   needs hardware validation, do NOT assume identical. Working
   hypothesis baked into the canonical scaffold
   (`gemm_qkv_hfq4g256_wmma.gfx12.hip`, commit 6924f2a) is
   `acc[j] = C[8*(tid>>4) + j][tid & 15]` — derived from the CK
   trait swap of `kCM0PerLane: 8 → 1` and `kCM1PerLane: 1 → 8`.
   Lane group 0 holds output rows 0..7; group 1 holds rows 8..15.
   This must be channel-tested on R9700 / 9070 XT before any
   kernel using it is wired into `dispatch.rs`.

## bf16×bf16→fp32 16×16×16

| Arch | Builtin |
|---|---|
| gfx11 | `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` |
| gfx12 | `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` |

Same operand-shape divergence as fp16. Not currently used by
hipfire (no bf16 GEMM kernels yet) but worth knowing.

## fp16×fp16→fp16 16×16×16

| Arch | Builtin |
|---|---|
| gfx11 | `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` |
| gfx12 | (verify in ROCm 7 headers — `_gfx12` variant if present) |

Used for fast inference paths that don't need fp32 accumulator
range. Hipfire doesn't dispatch this today.

## i8×i8→i32 16×16×16

| Arch | Builtin |
|---|---|
| gfx11 | `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` |
| gfx12 | `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` |

Used in CK quantization paths. Hipfire's quant kernels currently
operate on packed 4-bit + scale, so this isn't directly relevant
yet, but `gfx12` adds new fp8/bf8 mixed-precision variants that may
become useful for HFQ4 future work:

- `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
- `__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12`
- `__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12`
- `__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12`

These are gfx12-only — no gfx11 equivalent.

## Preprocessor arch macros

The HIP compiler defines (at `--offload-arch=...`):

| Compile target | `__gfxNNN__` defined | `__gfx11__` family | `__gfx12__` family |
|---|---|---|---|
| `gfx1100` | `__gfx1100__` | yes | no |
| `gfx1101` | `__gfx1101__` | yes | no |
| `gfx1102` | `__gfx1102__` | yes | no |
| `gfx1150` | `__gfx1150__` | (see ROCm release notes) | no |
| `gfx1200` | `__gfx1200__` | no | yes |
| `gfx1201` | `__gfx1201__` | no | yes |
| `gfx942` | `__gfx942__` | no (MFMA) | no (MFMA) |

The family macros (`__gfx11__`, `__gfx12__`) are defined when
compiling for ANY chip in that family — use them for shared
codepaths. Use the per-chip macro (`__gfx1201__`) when one chip in
a family needs different tuning.

## How to verify any of this against your install

```bash
rg --no-heading -n 'wmma_f32_16x16x16_f16' /opt/rocm/include/ | head -10
```

If you're on an older ROCm (pre-7.0), the gfx12 builtins may not
yet be available — `_gfx12` suffix was added in LLVM 19+ AMD-LLVM.
Check `clang -E -dM -x hip /dev/null --offload-arch=gfx1201` for
the active feature macros.

## Source / verification trail

- `/opt/rocm/include/ck/utility/amd_wmma.hpp` — gfx11 + gfx12 + wave64
  + fp8 variants, all builtin names.
- `/opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp` —
  operand vector lengths per arch (the `<16 x fp16>` vs `<8 x fp16>`
  difference).
- `/opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp` —
  the `#ifdef __gfx11__` / `#ifdef __gfx12__` dispatch pattern AMD
  themselves use in CK.
- Commit `b7ac66a` ("wmma correctness fix + MQ6 family + cross-arch
  prefill + gate framework") — fixes the gfx11 C-mapping. Assume
  any new arch's C-mapping is wrong until proven on hardware.

## Last verified

Date: 2026-04-26.
ROCm: 7.2.
LLVM (AMD): 22.0.0git.
By: hipfire arch-port skill author, on a 7900 XTX (gfx1100) host.
Re-verify before each new arch port — ROCm's intrinsic table moves.
