<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# WMMA / MFMA matrix (hipfire contracts)

Operand shapes, builtins, and lane layouts for matrix paths hipfire
dispatches. **Runtime routing** is `ArchCaps` + `gemm.rs` / `attention.rs` /
`kernels.rs`. **ISA names** match ROCm CK headers when present; always
re-verify on the build host before extending.

| Field | Value |
|---|---|
| Caps source | `crates/rdna-compute/src/arch_caps.rs` |
| Kernel siblings | `kernels/src/*.gfx12.hip`, base `*.hip` |
| Include registry | `crates/rdna-compute/src/kernels.rs` |
| CK reference (if installed) | `/opt/rocm/include/ck/utility/amd_wmma.hpp`, `/opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_*traits.hpp` |

## ArchCaps (what hipfire actually gates on)

Computed once at `Gpu::init` from the HIP arch string. **Live atom membership
and molecule composition are owned by**
`crates/rdna-compute/src/arch_caps.rs` — read that file; do not treat the
names below as a frozen allowlist.

Capability *names* hipfire routing commonly uses (examples):

- Family / molecule helpers: `is_rdna3`, `is_rdna3_dgpu`, `is_rdna3p5`,
  `is_rdna4`, `is_cdna3`, …
- WMMA capability split: `has_wmma`, `has_wmma_w32` (gfx11-family builtins),
  `has_wmma_w32_gfx12` (gfx12 builtins)
- Wave helpers: `is_wave32`, `is_wave64_native`
- Chip atoms (orchestration only when a plan names one chip): `is_gfx1201()`,
  `is_gfx1100()`, … — prefer molecule/capability caps for shared GEMM
  routing unless the plan is single-chip

**Rule:** gfx11 kernels do **not** lower on gfx12 (`Cannot select intrinsic
%llvm.amdgcn.wmma…`). Sister sources are selected with
`has_wmma_w32_gfx12()` vs `has_wmma_w32()`, not a macro swap inside one file.
Confirm which atoms set each helper in current `arch_caps.rs`.

## fp16×fp16→fp32 16×16×16 (primary GEMM shape)

HFQ/MQ dequant → fp16 LDS/reg → WMMA into fp32 accumulator.

| Family | Wave | A/B vec | C vec | Builtin | Caps gate |
|---|---|---|---|---|---|
| **gfx11** (RDNA3 / 3.5) | 32 | 16×fp16 (`half16_t`) | 8×f32 | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | `has_wmma_w32` |
| **gfx12** (RDNA4) | 32 | 8×fp16 (`half8_t`) | 8×f32 | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` | `has_wmma_w32_gfx12` |
| **gfx11 wave64 ABI** (CK) | 64 | 16×fp16 | 4×f32 | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w64` | Not hipfire’s primary RDNA path |
| **CDNA3** | 64 | 4×fp16 (`half4_t`) | 4×f32 | `__builtin_amdgcn_mfma_f32_16x16x16f16` | `is_cdna3` / MFMA paths |

CK `WmmaTraitsBase` (ROCm 7.x on a typical install):

| | gfx11_t | gfx12_t |
|---|---|---|
| A/B elements per lane | 16 | 8 |
| `kABKLane` | 1 | 2 |
| `kABK1PerLane` | 16 | 8 |
| `kCM0PerLane` / `kCM1PerLane` | 8 / 1 | 1 / 8 |
| `kRepeat` | 2 | 1 |

### Load-bearing gfx11 → gfx12 differences
1. **K packing halves** — each lane holds 8 fp16 of A/B, not 16. LDS/global
   K-tile loads must match.
2. **K split across lane groups on gfx12** — `k_grp = tid >> 4` (0 → K[0..7],
   1 → K[8..15]) with `kABKLane=2`. gfx11 replicates K with `kABKLane=1`.
3. **`kRepeat` drops from 2 → 1** — gfx11 repeats the WMMA op twice per K
   tile step; gfx12 uses a single op (`kRepeat=1`). Tile loops and LDS staging
   must match the target trait, not a shared count.
4. **Builtin suffix** — `_w32` vs `_w32_gfx12`; distinct LLVM nodes.
5. **C-output mapping** (row = M, col = batch N = `tid & 15`):

| Arch | Mapping used in hipfire kernels | Status |
|---|---|---|
| gfx11 | `acc[j] = C[2*j + (tid>>4)][tid & 15]` | Validated (gfx11 correctness fix era; channel tests) |
| gfx12 | `acc[j] = C[8*(tid>>4) + j][tid & 15]` | Contiguous rows 0..7 / 8..15 per lane group; **silicon-validated on R9700/gfx1201** for residual/QKV/Q8 paths via `test_wmma_*_gfx12` / PR #56-class channel tests — still **re-check** any new sibling before wiring |

Canonical gfx12 pattern files (among others):
`kernels/src/gemm_qkv_hfq4g256_wmma.gfx12.hip`,
`kernels/src/gemm_gate_up_hfq4g256_wmma.gfx12.hip`,
`kernels/src/gemm_hfq4g256_residual_wmma.gfx12.hip`,
`kernels/src/gemm_q8_0_wmma.gfx12.hip` / residual / qkv sisters.

## Other builtins (presence ≠ hipfire dispatch)

| Shape | gfx11 | gfx12 | Hipfire note |
|---|---|---|---|
| bf16→f32 16×16×16 | `…_bf16_w32` | `…_bf16_w32_gfx12` | CK; not a primary hipfire GEMM family today |
| fp16→fp16 acc | `…_f16_…_w32` | verify `_gfx12` in headers | Not primary |
| i8→i32 | `…_iu8_w32` | `…_iu8_w32_gfx12` | Used on some MoE grouped paths (e.g. `gemm_hfq4g256_moe_grouped_wmma_k2` / `HIPFIRE_MOE_GROUPED_I8`). **Default-on vs opt-in is per-arch and mutable** — read current `FeatureFlags` and call sites; do not assert defaults from this skill |
| fp8/bf8 mixed (gfx12-only) | — | `…_fp8_fp8_…_gfx12` and bf8 mixes | Gated (e.g. `HIPFIRE_FP8_WMMA`); batch floors apply (`FP8_WMMA_MIN_BATCH` in `dispatch.rs`) — re-check flags before claiming |

Do not claim an fp8/i8 path is default-on without reading the current
`FeatureFlags` / call site.

## Compile macros

HIP `--offload-arch=` defines **per-chip** atoms (`__gfx1100__`, `__gfx1201__`,
…). It does **not** define family macros `__gfx11__` / `__gfx12__` by itself.
CK headers synthesize those family macros from atom/generic macros. Standalone
hipfire kernels must not rely on `__gfx11__` / `__gfx12__` unless they include
that synthesis or define the macros themselves. Prefer family ISA via shared
sibling sources + host `ArchCaps` routing over kernel `#ifdef` alone.

Family ↔ chip mapping is owned by `arch_caps.rs` + the offload list you
pass to HIP. Rough orientation only: gfx11 WMMA → `has_wmma_w32`; gfx12
WMMA → `has_wmma_w32_gfx12`; CDNA3 MFMA → `is_cdna3`. Re-read caps before
claiming a chip is in or out of a family.

## Dispatch checklist for a new port

1. Add or reuse a **sibling** `.gfx12.hip` (or MFMA) source — do not only rename
   the builtin in the gfx11 file.
2. `include_str!` + public const in `kernels.rs` with a comment on mapping.
3. Branch in the Gpu method: `has_wmma_w32_gfx12()` before `has_wmma_w32()`.
4. Channel-test on **real** target silicon (`test_kernels` and/or
   `test_wmma_*_gfx12` examples). Wrong C-map is silent corruption.
5. Env-gated experimental families stay gated until parity is earned
   (e.g. `HIPFIRE_LLOYD_GFX12` for Lloyd WMMA on gfx12).
6. Validation routes: `validation.md` → `docs/VALIDATION.md`.

## Verify headers on the build machine

```bash
rg --no-heading -n 'wmma_f32_16x16x16_f16' /opt/rocm/include/ | head
rg -n 'kCM0PerLane|kABKLane|ext_vector' \
  /opt/rocm/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp
```

Pre-ROCm-7 / pre-LLVM-19 AMD toolchains may lack `_gfx12` builtins.

## Related history (provenance, not live gates)

- gfx11 C-map corruption class and fix era: commit `b7ac66a` narrative in skill
  entry / playbook.
- gfx12 pattern bring-up: QKV scaffold and residual/QKV channel tests on R9700;
  inspect current `kernels.rs` comments for which symbols are wired vs
  opt-in/unvalidated.
- Issue #54 class: missing gfx12 intrinsic selection → codegen fail; fix is
  sibling kernel + caps branch, not silence.

Re-derive this table from `ArchCaps` + CK headers when ROCm or hipfire routing
moves — do not treat this skill file as a frozen ISA dump.
