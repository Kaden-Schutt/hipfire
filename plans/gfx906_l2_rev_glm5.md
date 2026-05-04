# Adversarial review: gfx906_mmq_l2.md

**Reviewer:** glm-5-turbo (opencode agent)
**Date:** 2026-05-04
**Scope:** `plans/gfx906_mmq_l2.md` v1 (L2 prefetch for MMQ_X=8 kernel at commit `39b1eb7`)

**Bottom line: the prefetch design as written has a low probability of reaching the 10% keep threshold and a material risk of regressing performance. The bottleneck diagnosis is likely wrong, the prefetch coverage is too small to matter, and the VGPR risk could drop occupancy to 1 — a configuration we already measured as zero-gain. I recommend pivoting to software pipelining (double-buffered X tile) instead, which attacks the same latency axis at full tile width.**

---

## 1. Critical issues

### 1.1 Bottleneck misdiagnosis: memory isn't the bottleneck

The plan states (line 14-16):

> the SIMDs idle ~67% of the time waiting on memory completions, even though MemUnitStalled is only 2.9%

This is contradicted by the plan's own data. `MemBusy = 24.4%` means the memory units are active only 24% of wall-clock time. If the SIMDs were stalled on memory, the memory units would be saturated (MemBusy near 100%) and stalled (MemStall high). Instead we see MemBusy=24% and MemStall=2.9% — the memory system has 75% spare capacity. It is not the bottleneck.

On AMD GPUs, VALUBusy and MemBusy are **not complementary**. The remaining ~67% of wall-clock is not "waiting on memory" — it is one or more of:

- **Instruction fetch / decode stalls.** The kernel binary is 73 KB (ELF at MMQ_X=8). The hot loop body is large: 4 k01 iterations × 4 j0 iterations × 2 i0 iterations × 8 vdr dp4a calls = 256 dp4a sites, each with ~10 surrounding instructions (load, scale, convert, accumulate). That's ~2,500 instructions in the loop body. gfx906 has 32 KiB L1 instruction cache. The loop body likely fits, but branch mispredicts, fetch bubbles, and the 3 barrier synchronizations per kg iteration add up.

- **`s_waitcnt lgkmcnt(0)` — LDS barrier stalls.** The `mmq_body` loop has **3 `__syncthreads()` per kg iteration** (lines 397, 399, 403). With `groups_per_row = 16` for K=4096, that's **48 barriers** per kernel invocation. Each barrier forces all 128 threads to converge. The warp 1 threads (lanes 64-127) are **idle during the entire X-load phase** (line 393) because `load_hfq4_tile_dp4a` uses 1 thread per row and all 128 threads participate, but the load is per-row serial. Actually — re-reading the code, all 128 threads load their own row in parallel, so the load phase is well-parallelized. The barrier waits are for the Y-load and compute phases to complete across warps.

- **Scratch load/store latency.** Still 0.58 scratch ops per dp4a (144 spills). Each scratch round-trip costs 10-100 cycles depending on L1/L2 hit. This is the same stall axis the spill-reduction work attacked, and it's not yet zero.

- **Register dependency chains in the dp4a loop.** Each `sumi = __builtin_amdgcn_sdot4(x_int, y_int, sumi, false)` has a 4-cycle latency on gfx906 and feeds directly into the next iteration. With 8 consecutive dp4a calls in the vdr loop (line 301-305), that's a 32-cycle dependency chain per (i, j, k01) triple. The compiler may not be able to schedule independent dp4a chains to hide this latency.

**The prefetch plan does not address any of these.** It only addresses HBM-to-L2 fetch latency for the X tile, which is a small fraction of the idle time given MemBusy=24%.

### 1.2 Prefetch coverage is too small to move the needle

The design issues 32 `global_load_dword` instructions (32 lanes × 4 B = 128 B), triggering 2 L2 cache line fills (64 B each).

The actual X tile per kg iteration is:
- 128 rows × 136 B/group = 17,408 B (full HFQ4 groups including headers)
- Or 128 rows × 128 B = 16,384 B (payload only)
- That's **256 cache lines**

Prefetching 2 cache lines = **0.78% coverage**.

The plan acknowledges this (line 98-100): "That's tiny compared to the 16 KB total X tile, but the goal is to trigger the L2 fetch ahead of time, not to load all of X." This reasoning is flawed. On gfx906 there is **no hardware prefetcher for global memory** (unlike CPUs where a single cache-line touch can trigger a spatial prefetch stream). Each `global_load_dword` brings exactly one 64 B cache line into L2. Period. Two prefetches bring 128 B. The next iteration's `load_hfq4_tile_dp4a` still issues ~4,096 uint loads that miss on 254 out of 256 cache lines.

The row spacing makes this worse. The plan targets rows 0, 4, 8, ..., 124 (32 rows out of 128, lines 116-120). Rows 1-3, 5-7, 9-11, etc. (96 rows) get zero prefetch coverage. Even if we optimistically assume each 64 B cache line fill covers 14 uint reads (chunks 0-13 of the payload starting at offset 8), the total coverage is 32 rows × 14 chunks = 448 out of 128 × 32 = 4,096 chunks = **10.9%**.

A 10.9% reduction in cache misses, on an axis that accounts for at most ~25% of wall-clock (MemBusy), yields at best a **2.7% wall-clock improvement**. That's well below the 10% keep threshold.

### 1.3 VGPR risk: occupancy drop to 1 is a known zero-gain configuration

The plan identifies the VGPR risk (lines 153-164): the kernel is at 128/128 VGPRs with `__launch_bounds__(128, 2)`. Adding 1 VGPR for the prefetch dummy could push to 129 and drop occupancy to 1.

The plan's mitigations are weak:
1. "Hope LLVM keeps it inside the existing VGPR budget" — not a strategy
2. "Lower to 16 lanes" — even less cache coverage
3. "Write to LDS scratch" — adds LDS traffic, defeats the purpose

But the real problem is that **we already measured occ=1 as zero-gain.** From the spill-reduction log (optimization attempt #2):

```
| (128, 2) → 46.7 tk/s, vgpr_spill_count=2121
| (128, 1) → 46.8 tk/s, vgpr_spill_count=1780 (−16%)
```

Dropping to occ=1 freed 16% of spills and gained **0.1 tk/s** — statistically zero. The occupancy loss from latency hiding exactly cancelled the spill reduction. If prefetch pushes us to occ=1, we get the occupancy penalty with no spill reduction benefit. This is a strict regression.

The plan does not quantify this risk in the expected-outcomes table (line 248-253). "Spill increase >50%, perf neutral" is listed as "low" likelihood, but the risk isn't spill increase — it's **occupancy drop without any spill improvement**, which is a guaranteed ~0% delta at best and a regression at worst.

### 1.4 The reference implementation's prefetch is dead code — the signal is negative, not neutral

The plan correctly identifies that llama.cpp-gfx906's prefetch helpers are never called (line 38-45). But it frames this as "the pattern is documented" and "the expected yield is not validated."

The stronger inference is: **the llama.cpp authors wrote five prefetch variants, tested them, and deliberately chose not to ship any of them.** Five iterations (v1, v2, v4, _second, _noop) suggests active experimentation. The `_noop` stub (lines 181-190) is particularly telling — it exists to suppress warnings when prefetch is compiled out, meaning they had a toggle and tested with it off.

The plan says "we adopt the technique but treat the yield as unknown." The empirical signal is not unknown — it is **negative**. The prior art tried this and chose not to use it.

---

## 2. Design concerns

### 2.1 Prefetch placement is suboptimal for the actual bottleneck

The plan places prefetch "right after `load_hfq4_tile_dp4a(kg)`, before the first `__syncthreads()`" (line 83). This means the prefetch's latency-hiding window is:

```
[prefetch issue] → [barrier 1] → [vec_dot_dp4a #1] → [barrier 2] → [load Y #2] → [barrier 3] → [vec_dot_dp4a #2] → [barrier 4] → [load X kg+1 starts]
```

The window from prefetch issue to the next X-load is roughly:
- 2 compute calls + 2 barrier waits + 1 Y-load
- At VALUBusy=8.85%, compute is fast — maybe ~10% of wall-clock
- Barriers force synchronization, adding idle time
- Net window: maybe 30-50% of one kg iteration's wall-clock

At 6.72 ms per kernel call and 16 kg iterations, each iteration is ~420 µs. A 30-50% window is ~125-210 µs. gfx906 HBM latency is ~200-400 cycles ≈ 115-230 ns at 1.7 GHz. So the window is 500-1800× longer than the HBM round-trip. **The latency is already well-hidden by the compute time.** The problem isn't latency — it's bandwidth (or rather, the lack of it, since MemBusy=24%).

Wait — if HBM latency is already hidden, what's causing the MemBusy=24%? It's the sheer volume of cache misses in `load_hfq4_tile_dp4a` (4,096 uint loads per kg, mostly L2 misses on first access). Prefetching 128 B doesn't reduce the miss count enough to matter.

### 2.2 Address calculation targets group header, not the hot payload path

The prefetch targets offset 0 of the HFQ4 group:
```cpp
const char* gp_next = A + ((long long)actual_row * groups_per_row + kg_next) * 136;
```

This warms the cache line containing scale (offset 0-3), zp (offset 4-7), and nibble payload bytes 8-63. The actual load path in `load_hfq4_tile_dp4a` reads scale/zp at offset 0-7 (line 164-165) then loops from offset 8+chunk*4 (line 172). So the prefetch covers chunks 0-13 (payload bytes 8-59) within the 64 B cache line — that's 14 out of 32 chunks per row.

This is actually fine for the header coverage. But it means the prefetch covers **less than half** of each row's payload. Chunks 14-31 (payload bytes 64-135) are in the second and third cache lines of the group and are NOT prefetched.

### 2.3 Row stride between prefetched rows doesn't match cache-line spatial locality

Rows are spaced 4 apart (rows 0, 4, 8, ..., 124). The row stride in the weight matrix is `groups_per_row * 136` bytes. For K=4096, that's `16 * 136 = 2,176` bytes between consecutive rows. The gap between prefetched row 0's group and prefetched row 4's group is `4 * 2176 = 8,704` bytes — no spatial locality at all. Each prefetch touches a completely different region of memory.

The plan acknowledges this (line 119: "L2 line fills tend to bring adjacent rows along") but this is incorrect for gfx906 global memory — there is no spatial prefetcher. Each cache line fill is exactly 64 bytes at the requested address.

### 2.4 rocprof validation expectations are wrong

Step 6 (line 202-206) expects:
> VALUBusy should rise (less idle time)
> MemStall should drop or stay flat

If prefetch works:
- Total iteration time decreases (fewer cache misses in the next X-load)
- Compute time stays the same (same dp4a count)
- VALUBusy = compute_time / total_time → **could stay flat or even drop** if the time savings come from a non-VALU stall phase
- MemBusy drops (fewer outstanding memory transactions)
- MemStall drops (fewer cache misses)

The correct expectation is: **wallclock time decreases, MemBusy decreases.** VALUBusy movement is ambiguous. The plan should measure wallclock directly (step 5 already does this) and not rely on counter heuristics.

### 2.5 `asm volatile` + `"v"` constraint on a 64-bit address needs two VGPRs, not one

On gfx906, `global_load_dword` is a VMEM instruction that operates on 64-bit flat addresses. The `"v"(gp_next)` constraint requires the compiler to place the 64-bit address in a VGPR pair (v[n] and v[n+1]). Plus the `"=v"(dummy)` output VGPR. That's **3 VGPRs** consumed by the prefetch, not 1 as the plan estimates (line 160-162).

This makes the occupancy risk worse than stated.

---

## 3. Minor issues

### 3.1 The `volatile` + `"memory"` clobber doesn't do what the plan claims

Line 134: "The volatile + : 'memory' clobber prevents the compiler from CSE'ing or removing the dummy load."

The `"memory"` clobber prevents reordering of memory operations across the asm, but `asm volatile` already prevents removal. The `"memory"` clobber is unnecessary for preventing dead-code elimination and adds unnecessary scheduling constraints. Drop it — just use `asm volatile`.

### 3.2 `v_mov_b32 %0, %0` may be eliminated by newer LLVM

Line 146: `asm volatile("v_mov_b32 %0, %0" : "+v"(dummy))`. On ROCm 6.x (which this kernel targets), LLVM may recognize this as a no-op and eliminate it despite `volatile`. A more robust pattern is to store the dummy to a volatile global or use `s_nop 0` as a scheduling barrier. But since the plan marks this as optional (line 149), it's a minor issue.

### 3.3 L2 cache line size not cited

Line 97 states "gfx906 L2 line = 64 B" without citation. The skyne98 wiki confirms this but the plan should reference it.

### 3.4 The "2 cache lines" claim on line 97 is wrong

32 lanes each issue one `global_load_dword` (4 bytes). If the 32 target addresses span different 64 B cache lines, the number of unique lines touched depends on the address spacing. With rows spaced 4 apart and row stride = `groups_per_row * 136`:
- For K=4096: stride = 2,176 B. Row 0 is at offset 0, row 4 at offset 8,704. These are in completely different cache lines (128 lines apart). **All 32 prefetches hit 32 distinct cache lines**, not 2.

The plan's "2 cache lines" claim assumes the 32 dwords are within a 128 B window. They're not — they're spread across 32 different rows, each in a different part of the weight matrix. So the plan actually triggers **32 cache line fills** (2,048 B total), not 2 (128 B). That's 12.4% of the 16.4 KB payload — better than 0.78%, but still well below what's needed.

This is a significant arithmetic error in the plan.

---

## 4. Alternative: software pipelining (double-buffered X tile)

Instead of prefetching 128-2048 B of the next X tile with spare lanes, restructure the loop to overlap the **entire** X tile load with compute:

```
Current (no overlap):
  for kg = 0..N-1:
    load_X(kg)          ← 16.4 KB from HBM, serial
    load_Y_half1(kg)    ← small
    barrier
    compute_half1()     ← dp4a
    barrier
    load_Y_half2(kg)    ← small
    barrier
    compute_half2()     ← dp4a
    barrier

Proposed (double-buffered X):
  load_X(0)                          ← prime the pump
  for kg = 0..N-2:
    load_Y_half1(kg)
    load_X(kg+1)  // async overlap   ← starts WHILE compute runs
    barrier
    compute_half1()
    barrier
    load_Y_half2(kg)
    barrier
    compute_half2()
    barrier
  // final kg: compute last half without prefetch
```

This overlaps 16.4 KB of X loads with compute, not 128 B. It requires:
- Double the X LDS buffer (33 KB × 2 = 66 KB) — **exceeds the 64 KiB LDS cap** on gfx906. This doesn't fit.

So full double-buffering won't work with the current LDS budget. But a **partial** approach might:

**Half-group double-buffer:** Load only the first 8 bytes (scale+zp) of each row in the next group while computing. That's 128 × 8 = 1,024 B — trivially fits in a small LDS staging area or even in VGPRs. The scale/zp load is the first thing `load_hfq4_tile_dp4a` does (lines 164-166), and it's the data that the correction term needs first. Warming just the headers doesn't save much though — the bulk of the load is the 128 B payload.

**Chunk-pipelined load:** Instead of loading all 32 chunks per row at once, interleave chunks with compute:

```
for kg = 0..N-1:
  load_X_chunks(kg, 0..15)     // first half of payload
  load_Y_half1(kg)
  barrier; compute_half1()
  load_X_chunks(kg, 16..31)    // second half, overlaps with first compute
  barrier; compute_half2()     // uses first-half X data already in LDS
  barrier
```

This requires restructuring `load_hfq4_tile_dp4a` into two phases and adjusting `vec_dot_dp4a` to use only the first half of X for the first compute call. More invasive, but overlaps 8 KB of payload loads with compute — **4× more** than the prefetch approach.

### Recommendation

Before spending time on L2 prefetch (estimated yield: 0-3%, with occupancy risk), investigate:

1. **Can we reduce LDS to fit double-buffered X?** At MMQ_X=8, tile_y is only 1,152 B. If we can overlap tile_y with the second X buffer (they don't need to be live simultaneously — load Y into the space freed by the first X half), we might fit. Total LDS needed: 33 KB (X buf A) + 33 KB (X buf B) + 1 KB (x_dm) = 67 KB. Still 3 KB over the 64 KiB cap. We'd need to shrink X_STRIDE from 65 to 64 (lose the bank-conflict padding — measured at 0% conflict anyway) to get 128 × 64 = 32,768 B × 2 + 1,024 = 66,560 B. Still over. What if we reduce MMQ_Y from 128 to 64? Then X is 64 × 64 = 4,096 B × 2 + 512 = 8,704 B. Fits easily, but halves arithmetic intensity.

2. **Is the real bottleneck instruction fetch?** Run rocprof with `FetchSize` and `FETCH_UNIT_BUSY`. The Phase 0 counter file (phase0_rocprof_counters.txt) already includes `FetchSize`. If the fetch unit is stalled, the fix is loop compression, not prefetch.

3. **Can we eliminate the 3 barriers per kg?** The Y-twice pattern requires barriers because both warps read tile_y. If we restructure so each warp computes on its own Y column subset, we might merge or eliminate some barriers. This directly attacks the ~67% idle time.

---

## 5. Summary table

| Issue | Severity | Likelihood of impact | Verdict |
|---|---|---|---|
| Bottleneck misdiagnosis (memory not the limit) | Critical | High | Prefetch targets wrong axis |
| 128-2048 B coverage vs 16.4 KB tile | Critical | High | Too small to reach 10% threshold |
| VGPR risk → occ=1 (known zero-gain) | Critical | Medium | Could regress, not improve |
| Reference code is dead (negative signal) | High | High | Prior art rejected this approach |
| 32 cache lines, not 2 (arithmetic error) | Medium | Certain | Changes coverage estimate 16× |
| rocprof validation expectations wrong | Medium | Medium | Could mislead go/no-go decision |
| 64-bit address needs 3 VGPRs, not 1 | Medium | High | Worsens the occupancy risk |
| `"memory"` clobber unnecessary | Low | Low | Minor scheduling constraint |

**Recommended action:** Do not implement L2 prefetch v1 as designed. Instead:
1. Profile with `FetchSize` to confirm the actual bottleneck (1 hour).
2. If fetch stalls are significant, compress the loop body (loop unroll reduction, merge barriers).
3. If fetch is clean, investigate chunk-pipelined X load or barrier elimination.
4. Revisit L2 prefetch only if (a) the bottleneck is confirmed to be HBM→L2 latency and (b) we can prefetch ≥25% of the tile (8+ cache lines per row across ≥64 rows) without exceeding 128 VGPRs.
