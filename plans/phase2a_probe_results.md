# Phase 2a Probe Results — gfx906 MMQ Redesign

Date: 2026-05-04
Hardware: gfx906 (Vega 20 / MI50)
Target: `__launch_bounds__(256, 2)` (2 WGs/CU, 512 threads/CU)

## 1. 256-K Resident Layout (v1 Plan)
*   **LDS usage:** 45,056 B (44 KiB) per WG.
*   **Occupancy Check:** `44 KiB * 2 = 88 KiB` (Exceeds 64 KiB cap).
*   **Compiler Warning:** `desired occupancy was 2, final occupancy is 1`.
*   **Gate 3 Status:** **FAILED.** 256-K resident layout cannot achieve 2 WGs/CU.

## 2. 32-K Streaming Layout (Option B)
*   **LDS usage:** 17,408 B (17 KiB) per WG.
*   **Occupancy Check:** `17 KiB * 2 = 34 KiB` (Fits within 64 KiB cap).
*   **VGPR count:** 112 registers per thread (under 128 limit).
*   **VGPR spills:** 0.
*   **Gate 1/2/3 Status:** **PASSED.**

## 3. Decision
Based on the probe, we will **proceed with Option B (32-K Streaming)** for Phase 2b.
*   **x_qs layout:** `mmq_y * 16` ints (16 ints = 64 K-elements = 2 sub-blocks of 32). 
*   Wait, the probe used `X_STRIDE_STREAM 16` which covers 64 elements. 
*   If we want to match stock's 32-K streaming precisely, we could use `X_STRIDE_STREAM 8`.
*   The probe at `X_STRIDE_STREAM 16` used 17 KiB per WG, which fits comfortably.
*   **Final Choice:** Option B with `X_STRIDE_STREAM = 16` to provide 16-byte alignment for `ds_read_b128` while staying well within LDS and VGPR limits.

## 4. Next Steps
Move to Phase 2b (Full Rewrite) using the Option B topology.
