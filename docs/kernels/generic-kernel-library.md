# Generic Kernel Library — Plan & Manifest

Status: **in progress** (foundation). This document is the authoritative plan
for a library of tested, dtype-generic GEMM/GEMV kernels that new model
arch crates can reuse instead of round-tripping every op through f32 or
reaching for a quant-format-specific kernel.

## Motivation

hipfire's production path is its own block-quant containers (`q8`, `hfq*`,
`mq*`). As a result the *generic* IEEE/native-dtype kernel set is sparse:
the f32 reference path is complete, but fp16/bf16 exist mostly as the GEMM
weight tier, and generic int8/int4 kernels are essentially absent (their
math lives fused inside the quant kernels). As we add more models with
varied quirks, a tested generic kernel library massively speeds bring-up.

## Scope (this phase)

- **Families:** GEMM and GEMV.
- **Dtypes (input → output):** `iu4→i32`, `iu8→i32`, `bf16→bf16`,
  `bf16→f32`, `f16→f16`, `f16→f32`.
- **Arch targets we can test now:** RDNA3 dGPU (`gfx1100`, k9lin) and
  RDNA3.5 UMA (`gfx1151`, hipx). Local dev box is `gfx1103` (Phoenix UMA),
  same RDNA3 WMMA ISA — usable for numeric validation with the **no-LDS
  bodies only** (see hazard below).

All six dtype combos map to a native WMMA instruction on RDNA3/RDNA4
(verified via `third_party/amd_matrix_instruction_calculator`):

| Combo      | WMMA instruction                      |
|------------|---------------------------------------|
| `f16→f32`  | `v_wmma_f32_16x16x16_f16`             |
| `f16→f16`  | `v_wmma_f16_16x16x16_f16`             |
| `bf16→f32` | `v_wmma_f32_16x16x16_bf16`            |
| `bf16→bf16`| `v_wmma_bf16_16x16x16_bf16`           |
| `iu8→i32`  | `v_wmma_i32_16x16x16_iu8`             |
| `iu4→i32`  | `v_wmma_i32_16x16x16_iu4`             |

## Current inventory (audit 2026-06-20)

GEMM (existing kernels accumulate-and-store **F32**):

| Combo      | Status   | Existing kernel / note |
|------------|----------|------------------------|
| `f16→f32`  | ✅ exists | `gemm_f16_wmma` (Wf16×Xf32), `gemm_f16_x_f16_wmma` (Wf16×Xf16→F32) |
| `bf16→f32` | ✅ exists | `gemm_bf16_x_bf16_wmma` (named bf16×bf16 but **stores F32**) + `_gfx1151_m128` LDS large-M variant |
| `f16→f16`  | ❌ missing| need `wmma_f16_16x16x16_f16` + u16 store epilogue |
| `bf16→bf16`| ❌ missing| need `wmma_bf16_16x16x16_bf16` |
| `iu8→i32`  | ❌ missing| only synthetic probe `bench_iu8_wmma_gfx1151` |
| `iu4→i32`  | ⚠️ partial| `gemm_s4s4_wmma_tile_gfx1151` (signed-only, gfx1151-tagged) — generalize + test |

GEMV:

| Combo      | Status   | Existing kernel / note |
|------------|----------|------------------------|
| `f16→f32`  | ⚠️ partial| `gemv_f16_xf32` (Wf16×Xf32) — only this one |
| all others | ❌ missing| no bf16 / iu8 / iu4 / →f16 GEMV |

## Arch-class strategy

The win/loss profile differs by memory architecture, not just by gfx id.
Findings from `third_party/fsr4-rdna3-optimization` (gfx1151) plus our own
gfx1103 hazard note drive a **register-tiled (no-LDS) body on UMA** vs an
**LDS-staged body on dGPU**:

- UMA iGPUs share **system DRAM** with the CPU. Naive LDS staging of small
  working sets *regressed* on gfx1151 (fsr4 O10–O13). LDS still wins when it
  amortizes reuse across warps for large M (cf. `gemm_bf16_x_bf16_wmma_gfx1151_m128`).
- **Scalar INT8 MAC beat the packed-dot intrinsic by ~32% on gfx1151**
  (much smaller on gfx1100). The packed/DP4A default is wrong on UMA.
- **HAZARD:** on `gfx1103` (Phoenix), LDS-heavy kernels can wedge the GPU
  (page-fault → MES hang → full reset, sticky 719) — firmware bug. Prefer
  register tiling; only use LDS where proven and never test the LDS variant
  blind on a gfx1103 box.

Selection helper to add (taxonomy gap): `arch_caps` groups `gfx1103` under
`is_rdna3` but in **neither** `is_rdna3_dgpu` (1100/1/2) **nor**
`is_rdna3p5` (1150/1/2). The kernel UMA class must be
`is_gfx1103() || is_rdna3p5()`. Add an `is_rdna3_uma()` cap rather than
open-coding this at every call site.

Each kernel source carries both bodies behind arch macros (the JIT compiles
per `self.arch` via `compiler.compile`), or ships a `.gfx1151.hip` sibling
selected in Rust — mirror whichever the neighbouring kernels already use.

## Dispatch / wiring pattern (per kernel)

1. `kernels/src/<name>.hip` (+ optional `kernels/src/gfx1151/<name>.gfx1151.hip`).
2. `pub const <NAME>_SRC: &str = include_str!(...)` in `crates/rdna-compute/src/kernels.rs`.
3. `pub fn <name>(&mut self, …)` in `dispatch.rs`: `bind_thread` →
   `ensure_kernel(name, SRC, entry)` → params → `launch_kernel`.
4. Numeric test vs an f32 CPU reference (small M/K/N), gated to run only on
   an RDNA3 GPU.

## Documented omissions (circle back)

- **gfx906 / MI50 / GCN5.1 (real target — cheap 32 GB HBM2, ~1 TB/s):**
  NO matrix cores; not modeled by the matrix calculator. Needs a separate
  V_DOT/scalar codegen path (wave64): `iu8→i32` via `v_dot4_i32_i8`,
  `f16→{f16,f32}` via `v_dot2_f32_f16`/`v_pk_fma_f16`. **No bf16 ALU** →
  bf16 must upconvert to f32 (correct, not fast). **No int4 dot** → unpack
  iu4→iu8 + `v_dot4`, or scalar. Deferred: untestable from current fleet
  position; build after RDNA3/3.5 set lands.
- **gfx1201 / RDNA4 (hiptrx):** strict WMMA superset. Deferred extras worth
  designing the API to accommodate: fp8/bf8 matmul
  (`v_wmma_f32_16x16x16_{fp8,bf8}_*`), wide int4 (`…16x16x32_iu4`), and
  SWMMAC 2:4 sparse (`v_swmmac_*`).
- **gfx1103 LDS-staged bodies:** compile but must not be perf/coherence-tested
  on a gfx1103 box until the firmware hang is mitigated; validate LDS bodies
  on gfx1100/gfx1151 only.
- **fp8 on RDNA3:** no native fp8 WMMA — not a gap on gfx1100/1103/1151.

## Build order

1. Foundation (this doc) + `is_rdna3_uma()` cap helper.
2. GEMM missing set: `bf16→bf16`, `f16→f16`, then `iu8→i32`, generalize
   `iu4→i32`. (Reuse existing `bf16→f32` / `f16→f32`.)
3. GEMV across all six (no-LDS register/dot bodies; UMA-first since GEMV is
   memory-bound and the MI50/UMA story centers on it).
4. Numeric tests for each on gfx1100 + gfx1151.
5. Later phases: gfx906 V_DOT path, gfx1201 fp8/sparse.
