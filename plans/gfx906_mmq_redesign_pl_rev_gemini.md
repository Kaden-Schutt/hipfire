# Adversarial Review: gfx906 MMQ Redesign

## 1. The LDS / Occupancy Contradiction (CRITICAL)
The plan dictates `__launch_bounds__(256, 2)` (2 WGs per CU) but calculates an LDS requirement of 43.5 KiB for `mmq_x=64`.
*   **Hardware Limit:** gfx906 (Vega 20) has exactly **64 KiB of LDS per CU**.
*   **Conflict:** 43.5 KiB × 2 WGs = **87 KiB**. This explicitly exceeds the 64 KiB hardware limit.
*   **Consequence:** The hardware scheduler will be forced to launch only **1 WG per CU**, immediately halving your theoretical occupancy and destroying the performance gains you expect from matching stock. You must either reduce LDS usage to <32 KiB to keep 2 WGs/CU, or explicitly accept 1 WG/CU and change the launch bounds to `__launch_bounds__(256, 1)`.

## 2. The VGPR "Zero Margin" Cliff (CRITICAL)
The plan accepts `vgpr_count ≈ 128` while targeting 2 WGs/CU.
*   **Hardware Limit:** gfx906 provides 65,536 VGPRs per CU.
*   **Math:** 256 threads/WG × 2 WGs/CU × 128 VGPRs/thread = **65,536 VGPRs**.
*   **Conflict:** You are at exactly 100% VGPR capacity for 2 WGs. If the AMD compiler allocates even **129 VGPRs** due to your `#pragma unroll` strategy or the wider tile, occupancy hard-drops to 1 WG/CU.
*   **Mitigation:** Relying on the compiler to hit exactly ≤128 VGPRs while changing loop structures is exceptionally fragile. You must verify what stock actually achieves (does it run at 1 WG/CU?) and adjust expectations/bounds accordingly.

## 3. Dispatch Logic & N-Remainders
Phase 3 proposes: `mmq_x = next_pow2(batch_size).clamp(8, 64)`.
*   **Flaw:** What happens for batch sizes that aren't exact multiples of `mmq_x`? For example, batch size 65. The plan dictates entry symbols for `N%mmq_x==0`, but doesn't explain how the host dispatch loops over N or handles the remainder.
*   **Consequence:** If you clamp to 64 and the kernel expects `N%mmq_x==0`, batch size 65 will either fail the bounds check, use a wildly inefficient fallback, or silently compute wrong results. The host dispatch must chunk `N` into `mmq_x` sized blocks and issue a separate kernel launch for the remainder (e.g., one block of 64, one block of 1 using `mmq_x=8` padded).

## 4. `ds_read_b128` Alignment Assumption
Q8 proposes emitting explicit `int4` reads to map to `ds_read_b128`.
*   **Flaw:** `ds_read_b128` strictly requires 16-byte aligned pointers.
*   **Check:** Your `x_qs` layout is 128 rows × 65 ints. 65 ints = 260 bytes. 260 is **not** a multiple of 16 (260 % 16 = 4).
*   **Consequence:** If row pointers are not 16-byte aligned, `ds_read_b128` will silently fall back to slower reads (or trap, depending on generation). You must pad the `x_qs` stride to an even multiple of 16 bytes (e.g., 68 ints = 272 bytes) to guarantee the optimization works.

## 5. Memory Coalescing & Idle Threads
Q4 recommends leaving 128 threads idle during X load (1 thread per row × 128 rows).
*   **Flaw:** If thread 0 reads row 0, thread 1 reads row 1... they are accessing addresses `base + 0`, `base + 65*4`, `base + 130*4`. This is a heavily strided memory access pattern.
*   **Consequence:** This guarantees massive uncoalesced global memory reads and likely LDS bank conflicts on write. While the compute phase dominates, terrible memory efficiency here could introduce stall bubbles. A coalesced vector-load pattern is required.

## Conclusion
The current design mathematically cannot meet its own `__launch_bounds__(256, 2)` requirement due to the LDS and VGPR constraints. 
**Recommendation:** 
1. Accept 1 WG/CU (`__launch_bounds__(256, 1)`) if that's what stock does.
2. Pad `x_qs` to 68 ints for 16-byte alignment.
3. Update Phase 3 dispatch logic to properly chunk `N`.