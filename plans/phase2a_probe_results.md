# Phase 2a Probe Results — gfx906 MMQ Redesign

Date: 2026-05-04
Hardware: gfx906 (Vega 20 / MI50)
Target: `__launch_bounds__(256, 2)` (2 WGs/CU, 512 threads/CU)

## Methodology caveat

Phase 2a was a **resource probe**, not a **correctness probe**. Both
probe stubs (`kernels/src/gfx906_mmq_probe.hip`,
`gfx906_mmq_probe_option_b.hip`) use `volatile` reads on dummy data
and never compute a real dot product. The numbers below come from
synthetic indexing under the desired topology; they confirm the budget
is *feasible*, not that the real Option B kernel will hit those exact
counts.

The real numbers from the shipped Phase 2b body
(`gemm_hfq4g256_residual_mmq_gfx906_body.cuh`) are reported in §3 below
for comparison.

## 1. 256-K Resident Layout (v1 Plan, rejected)

Stub: `gfx906_mmq_probe.hip`, `X_STRIDE=68`.
*   **LDS usage:** 45,056 B (44 KiB) per WG.
*   **Occupancy Check:** `44 KiB × 2 = 88 KiB` (Exceeds 64 KiB cap).
*   **Compiler warning:** `desired occupancy was 2, final occupancy is 1`.
*   **Gate 3 status:** **FAILED.** 256-K-resident layout cannot achieve
    2 WGs/CU.

## 2. 32-K Streaming Layout (Option B, accepted)

Stub: `gfx906_mmq_probe_option_b.hip`, `X_STRIDE_STREAM=16`.
*   **LDS usage:** 18,432 B (18 KiB) per WG.
    (x_qs 8,192 + x_dm 1,024 + tile_y 9,216).
*   **Occupancy Check:** `18 KiB × 2 = 36 KiB` (Fits within 64 KiB cap).
*   **VGPR count:** 112 / thread (under 128 limit).
*   **VGPR spills:** 0.
*   **Gate 1 / 2 / 3 status:** **PASSED.**

## 3. Real Phase 2b kernel (post-build verification)

Built via `scripts/compile-kernels.sh gfx906`, ELF metadata extracted
with `clang-offload-bundler --unbundle` + `llvm-readelf --notes`.

Note: `group_segment_fixed_size = 0` for all entry symbols because
body.cuh uses `extern __shared__` (dynamic LDS). The runtime LDS budget
is enforced by dispatch.rs at launch (`debug_assert!(shared_mem ≤
32*1024)`) — not by the compiler.

| Variant | vgpr_count | vgpr_spill_count | sgpr_count | static LDS |
|---|---|---|---|---|
| `_x8`  / `_full_*_x8`   | 48 | 0 | 40 | 0 (dynamic) |
| `_x16` / `_full_*_x16`  | 66 | 0 | 40 | 0 (dynamic) |
| `_x32` / `_full_*_x32`  | 82 | 0 | 41–42 | 0 (dynamic) |
| `_x64` / `_full_*_x64`  | 89 | 0 | 40 | 0 (dynamic) |

All variants:
*   ≤ 128 VGPR ceiling (Gate 1 honored at compile time).
*   0 spills (Gate 2 honored).
*   `max_flat_workgroup_size = 256` confirms `__launch_bounds__(256, 2)`.

Real x_qs LDS (Phase 2b post-fix, X_STRIDE=8 not 16):
*   x_qs   = 128 × 8 × 4    = 4,096 B
*   x_dm   = 128 × 8        = 1,024 B
*   tile_y = mmq_x × 36 × 4 ≤ 64 × 144 = 9,216 B
*   **Total per-WG ≤ 14,336 B** at mmq_x=64 (well under 32 KiB).

## 4. Decision

Proceed with Option B (32-K streaming) for Phase 2b. Final design point:
*   `X_STRIDE = 8` ints/row (32-K data, no headroom — saves 4 KiB vs
    probe stub's X_STRIDE=16; ds_read_b128 not used in v1 per §Q8).
*   `mmq_y = 128`.
*   `nwarps = 4`, block dim (64,4,1).
*   `__launch_bounds__(256, 2)` → 2 WGs/CU.
*   8 mmq_x variants {8, 16, 24, 32, 40, 48, 56, 64} × 3 entry symbols =
    24 entry symbols total.

## 5. Status

Phase 2a probe → COMMITTED (836b522).
Phase 2b kernel + dispatch → IN PROGRESS (must-fix items addressed
2026-05-04, see plan §"Phase 2b validation findings").
Phase 3 test harnesses + Phase 4 NRMSE matrix → OUTSTANDING.
