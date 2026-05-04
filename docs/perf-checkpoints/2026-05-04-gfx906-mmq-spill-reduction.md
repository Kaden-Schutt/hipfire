# 2026-05-04 gfx906 MMQ Spill Reduction

Hardware: AMD MI50 (gfx906 / Vega 20), ROCm 6.4.3.
Model: Qwen 3.5 9B prefill (pp128), HFQ4-G256 quant, asym3 KV cache.
Starting point: commit `8081822` (correctness fix landed). MMQ kernel
ran but was 3× slower than FP16 wave64 on full prefill (140.7 → 46.7
tk/s), making it unusable in production despite passing the coherence
gate.

This checkpoint walks the spill-reduction lever from initial profile
through saturation: a single config knob (`MMQ_X` tile width) brought
prefill from 46.7 → 120.9 tk/s (33% → 86% of FP16 baseline) by
eliminating 93% of VGPR spills.

## TL;DR

The kernel is correct (NRMSE 0.25%, coherence battery passes) but runs
**~3× slower than FP16 wave64 on full prefill** (140.7 → 46.7 tk/s).
Per-call it is **7.4× slower** (4.42 ms → 32.7 ms avg).

The bottleneck is **catastrophic VGPR spilling to scratch**:
- 2,121 spilled VGPRs/thread per kernel launch (ELF metadata)
- 7,048 B/thread of `private_segment_fixed_size` (scratch)
- 13,853 `buffer_load`/`buffer_store` instructions emitted in the kernel
  (vs **0** for the FP16 reference) — roughly **1 scratch op per dp4a**
- VALUBusy = **1.0%** while `MemUnitBusy` = ~12% — the SIMDs are idle
  >85% of the time waiting on scratch round-trips through L1/L2/HBM

Conversely, what is **NOT** the problem:
- LDS bank conflicts: 0.009% (negligible)
- Real LDS traffic volume: only 2× FP16's
- Global memory traffic: in line with theoretical (185 `global_load`
  instructions across the kernel; MMQ stages everything via LDS)

## Bench numbers

```
                        prefill  decode (ref)
FP16 wave64 (baseline)  140.7    50.5 tk/s
HIPFIRE_MMQ=1            46.7    50.6 tk/s   ← 33% of baseline
```

Decode is unaffected because batch=1 hits the GEMV path, not MMQ.

## Per-kernel comparison (rocprof v1, gfx906, Qwen 3.5 9B pp128)

Two GEMM shapes call this kernel per layer × 32 layers = 64 calls:
K=4096 (residual, attn output) and K=12288 (residual, MLP down).

### Group 0: VALUBusy + Memory pressure

| Kernel                       | DurNs (avg) | VALUBusy% | MemBusy% | MemStall% |
|------------------------------|-------------|-----------|----------|-----------|
| residual_fp16_wave64 (K=4096) |  2.12 M     | 61.6      | 73.0     | 1.30      |
| residual_fp16_wave64 (K=12288)|  6.66 M     | 54.7      | 64.8     | 0.40      |
| **mmq_gfx906** (K=4096)      | 16.4 M      | **1.0**   | **11.2** | 2.0       |
| **mmq_gfx906** (K=12288)     | 49.0 M      | **1.0**   | **11.2** | 2.0       |

VALUBusy collapsed from 55–62% (FP16) to **1%** (MMQ). The kernel is
not compute-bound and not main-memory-stall-bound — it's stalled on
scratch latency that doesn't show up in `MemUnitStalled` because it's
satisfied by L1/L2.

### Group 1: LDS counters

| Kernel                       | LDSBank% | SQ_INSTS_LDS | SQ_INSTS_SMEM | SQ_WAIT_INST_LDS |
|------------------------------|----------|--------------|---------------|------------------|
| residual_fp16_wave64 (K=4096) | 0       |  1,310,720  | 98,304        | ~35,000          |
| **mmq_gfx906** (K=4096)      | 0.009    |  2,603,008  | 512           | 0                |
| **mmq_gfx906** (K=12288)     | 0.009    |  7,809,024  | 512           | 0                |

LDS is well-behaved: zero conflicts, no LDS-issue waits. **LDS is not
the problem.** The wiki's `+1 vec4` padding + our `+1 int` choice held up.

### Group 2: VMEM / VALU instruction counts

| Kernel                       | VMEM_RD | VMEM_WR | FLAT     | VALU      |
|------------------------------|---------|---------|----------|-----------|
| residual_fp16_wave64 (K=4096) | 5,505 K | 262 K   | 5,767 K  | 132,612 K |
| residual_fp16_wave64 (K=12288)| 15,991 K| 262 K   | 16,253 K | 369,590 K |
| **mmq_gfx906** (K=4096)      | 4,988 K | **4,183 K** | 55 K | 17,394 K  |
| **mmq_gfx906** (K=12288)     | 14,933 K| **12,506 K** | 133 K | 51,956 K |

Two anomalies vs FP16:
1. **VMEM_WR is 16× higher** (4.18 M vs 0.26 M for FP16). FP16 only
   writes the output; MMQ writes 4.18 M times — every one of those is a
   spill store to scratch. There is no other source of write traffic
   in the kernel (the `Y[]` write-back path produces only ~262 K writes).
2. **VALU is 7.6× lower** (17.4 M vs 132 M). MMQ uses dp4a (one
   instruction = 4 MACs) where FP16 uses v_fma (one instruction = 2 MACs)
   *and* needs more VALUops for FP16 conversions. Lower VALU is the
   *intended* MMQ benefit; the wallclock loss is coming from elsewhere.

## ELF metadata (post-fix kernel)

Three entry symbols (`base`, `_full_add`, `_full_set`), all with
identical resource profile:

| Property                     | Value      |
|------------------------------|------------|
| `vgpr_count`                 | 128        |
| `vgpr_spill_count`           | **2,121–2,148** |
| `sgpr_count`                 | 28–32      |
| `sgpr_spill_count`           | 0          |
| `private_segment_fixed_size` | **7,048 B/thread** |
| `group_segment_fixed_size`   | 0 (LDS allocated dynamically; runtime requests 43,520 B) |
| `max_flat_workgroup_size`    | 128        |
| `wavefront_size`             | 64         |
| `__launch_bounds__`          | (128, 2)   |

7,048 × 128 threads = **902 KB of scratch per workgroup**.
2,121 spills × 64 lanes = ~136 K spill events per wave; with 2 waves/WG
and 64 WGs/dispatch, that's ~17 M spill-event opportunities — matches
the 4.99 M VMEM_RD + 4.18 M VMEM_WR observed (each spill is one read +
one write at minimum).

The 128-VGPR ceiling was forced by `__launch_bounds__(128, 2)`. Without
it the kernel would consume all 256 arch VGPRs at occupancy=1, which is
worse for latency hiding. With the bound, it spills hard to honour the
budget. **Both arms of this trade-off are bad.**

## ISA composition (single entry, ~17 K instructions)

```
buffer_load_dword  / buffer_store_dword  (scratch ops):  13,853
v_dot4_i32_i8                                            12,288
ds_read_*  (LDS load)                                     3,660
ds_write_* (LDS store)                                      177
global_load                                                 185
```

Ratio of scratch ops to dp4a calls = **1.13 : 1**. For a healthy MMQ
kernel this should be ≤ 0.05 : 1 (most dp4a inputs come from VGPRs
already). Each scratch op adds ~10 cycles (L1 hit) to ~hundreds of
cycles (L2/HBM miss) of latency.

## Where the spills come from (source-side hypotheses, untested)

The accumulator declaration on line 384 of
`kernels/src/gemm_hfq4g256_residual_mmq_gfx906.hip`:

```cpp
// MMQ_X=64, MMQ_NWARPS=2, MMQ_Y=128, WAVE_SIZE=64
float sum[(MMQ_X / MMQ_NWARPS) * (MMQ_Y / WAVE_SIZE)] = {0.0f};
// = float sum[32 * 2] = float sum[64]
```

64 floats × 4 B = 256 B = **64 VGPRs just for accumulators**. That alone
fits in the 128-VGPR budget, but combined with:

- The two unrolled compute loops in `vec_dot_dp4a` (lines 274-318):
  `k01 ∈ {0,8,16,24}` × `j0 ∈ [0,64) step 2` × `i0 ∈ [0,128) step 64` ×
  `v ∈ [0,8)`. With `#pragma unroll` on all four, that's
  `4 × 32 × 2 × 8 = 2,048` dp4a sites in the body.
- Per-site live values: `x_int`, `y_int`, `sumi`, plus loaded `dm_i.x`,
  `dm_i.y`, `ds_j` half2, `dsf.x`, `dsf.y`, `scale_w`, `zp_w`, `d_x`,
  `sum_x` — and an FMA-style `8.0f * scale_w` precompute. Each of those
  needs a VGPR. With 2,048 sites of intersecting liveness ranges, the
  scheduler can't fit even small subsets in registers.
- Float→half conversions and the `__half22float2(ds_j)` call further
  fragment liveness.

Compared to llama.cpp-gfx906's `mmq.cuh:3425`:
```cpp
float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};
```
With their `mmq_x=mmq_y=64, nwarps=2, warp_size=64`: `sum[32]`. That's
*half* the size of ours. Their full-width tile (`mmq_x=mmq_y=128`)
matches ours but they use `__launch_bounds__(128, 2)` and pay
similarly. **Important next-step question**: do their 128×128 tile
kernels also spill on gfx906? If so, llama.cpp's headline 235 tk/s
must be coming from a smaller `mmq_x` (likely 16/32 for prefill batch
sizes ≤ 64), not from 128×128. The dispatch table in their config
should resolve which `mmq_x` is selected for our N=128 prefill.

## Top contributors to slowdown (estimated)

Based on the per-call ratios:

| Source                               | Contribution |
|--------------------------------------|--------------|
| Scratch load/store latency           | ~70%         |
| Reduced VALU bandwidth (1% busy)     | already counted in stall |
| LDS issue latency (tile reload 2×)   | ~10%         |
| Setup/teardown / barrier sync        | ~10%         |
| Other (kernel launch, FP conversion) | ~10%         |

(Rough split — exact attribution would need cycle-level tracing.)

## Open observations / questions for next pass

1. **`mmq_x` shape sensitivity**: at our prefill N=128 the tile fully
   covers the batch. Reducing to `mmq_x=32` cuts the inner-j unroll
   from 32 to 16 sites — should halve spill pressure. llama.cpp does
   this dynamically; we hard-coded 64.

2. **Per-thread acc tile shape**: `(MMQ_X / MMQ_NWARPS) × (MMQ_Y /
   WAVE_SIZE) = 32 × 2`. Inverting to `2 × 32` (transpose the
   accumulator) might let the scheduler keep all 32 K-strided columns
   live across one i-block before moving on, reducing intersection.

3. **`#pragma unroll` aggression**: the kernel unrolls all four
   compute loops. Selectively un-unrolling the j-loop (32 iters) would
   let the scheduler pipeline a smaller window. Trade-off: lose dp4a
   ILP4 → ILP1.

4. **Eliminate the `8.0f * scale_w` recompute** at every site: it's
   loop-invariant per row and can be fused into `zp_w_eff = zp_w +
   8 * scale_w` once per row in `load_hfq4_tile_dp4a`, saved alongside
   `(scale_w, zp_w)` in `x_dm`. Saves one VGPR per active row.

5. **Y-twice pattern cost**: the X-once + Y-twice structure in
   `mmq_body` (lines 386-401) requires two `__syncthreads()` per kg
   plus two `vec_dot_dp4a` calls. Each call materializes the full 64-
   element acc; if we hold the acc in registers spanning both halves
   we're fine, but the spill suggests the compiler is checkpointing
   `sum[]` to scratch around the syncthreads. Worth checking with
   ISA inspection around the sync points.

6. **L2 prefetch deferred**: per Phase 0 doc §6.4, llama.cpp uses a
   manual `global_load_dword` prefetch to hide HBM→L2 latency between
   k-blocks. We deferred this to a follow-up. Even fixing the spill
   problem first leaves prefetch as ~10-15% additional headroom.

## Hardware ceiling check

gfx906 peak dp4a: ~43 TOPS (4 MACs × 64 lanes × 4 SIMDs × 60 CUs ×
1.7 GHz / 2). Our kernel's *intended* arithmetic load:
- per output cell: 32 × 4 = 128 ops per K-block × K/32 K-blocks
- N=128, M=4096, K=4096: 128 × 4096 × 4096 ops = 2.15 G dp4a ops total
- At 43 TOPS: 50 µs minimum
- Observed: 16.4 ms — **328× over the compute floor**

That's the gap to close. Even a 10× recovery (to 1.6 ms/call) would
take MMQ from 46.7 → ~250 tk/s on prefill, beating the 235 llama.cpp
reference. The compute headroom is enormous; we're just not using it.

## Raw artifacts

- `/tmp/rocprof_mmq/mmq_g0.csv` — VALU/Mem counters (MMQ)
- `/tmp/rocprof_mmq/mmq_g1.csv` — LDS counters (MMQ)
- `/tmp/rocprof_mmq/mmq_g2.csv` — VMEM/VALU counts (MMQ)
- `/tmp/rocprof_mmq/mmq_g{0,1,2}_fp16.csv` — same counters, FP16 reference
- `/tmp/rocprof_mmq/g{0..4}.txt` — rocprof input files
- `/tmp/mmq_gfx906.elf` — extracted ELF (offset 0x1000 from hsaco)
- `/tmp/mmq_gfx906.s` — full disassembly (62,563 lines)

## Optimization attempt #1 — `zp_eff = zp + 8·scale` fold (no-op)

Folded the `8 * scale_w` term into `x_dm.y` at load time so the
per-site FMA in `vec_dot_dp4a` simplifies from
`(zp_w + 8 * scale_w) * sum_x` to `zp_eff * sum_x`.

| Metric             | Before | After  |
|--------------------|--------|--------|
| `vgpr_spill_count` | 2,121  | 2,121  |
| Prefill tok/s      | 46.7   | 46.8   |

**Result: no perf change.** Compiler was already CSE'ing the constant.
Kept the edit; moved on.

## Optimization attempt #2 — `__launch_bounds__(128, 1)` (no-op)

Lifted occupancy ceiling from 2 → 1 to give the compiler 256 VGPRs
instead of 128.

| Metric             | (128, 2) | (128, 1) |
|--------------------|----------|----------|
| `vgpr_count`       | 128      | 256      |
| `vgpr_spill_count` | 2,121    | 1,780 (−16%) |
| `private_segment`  | 7,048 B  | 6,440 B  |
| Prefill tok/s      | 46.7     | 46.8     |

**Result: no wallclock change.** Spills dropped 16% but the latency-
hiding loss from occ=1 cancels the gain. **Reverted.**

## Optimization attempt #3 — `MMQ_X = 64 → 32 → 16 → 8 → 4` (saturation at 8)

Halved the column-tile width four times. Each step roughly halves
the j-loop unroll count, which roughly halves spill state. The
pattern saturated at MMQ_X=8 — the MMQ_X=4 step was a flat zero.

| MMQ_X | sum[] | j-loop | dp4a sites | spills | scratch B/thread | prefill tk/s | %FP16 | step gain |
|-------|-------|--------|------------|--------|------------------|--------------|-------|-----------|
| 64    | 64    | 32     | 2,048      | 2,121  | 7,048            | 46.7         | 33%   | —         |
| 32    | 32    | 16     | 1,024      | 914    | 3,280            | 66.4         | 47%   | +42%      |
| 16    | 16    | 8      | 512        | 372    | 1,464            | 92.2         | 66%   | +39%      |
| **8** | 8     | 4      | 256        | 144    | 564              | **120.9**    | **86%** | +31%    |
| 4     | 4     | 2      | 128        | 68     | 264              | 121.0        | 86%   | **+0%** ← saturated |

Side effects per step:
- Workgroup count doubles (64 → 128 → 256 → 512 WGs across 60 CUs)
- LDS budget shrinks (43 → 38 → 36 → 35 KiB)
- ELF size shrinks (420 → 206 → 115 → 73 KB)
- Synthetic correctness preserved: NRMSE 0.12% (K=4096) / 0.04% (K=12288)
- Decode unaffected (50.5 → 49.9, within noise)

`load_q8_1_tile` thread distribution shifted at each step:
- MMQ_X=64: 2 threads/col × 64 cols, all 128 active
- MMQ_X=32: 4 threads/col × 32 cols, all 128 active
- MMQ_X=16: 4 threads/col × 16 cols, threads 64..127 idle
- MMQ_X=8:  4 threads/col × 8 cols,  threads 32..127 idle

The wasted threads in the load phase are cheap relative to the dp4a
body — load phase is ~1 KiB of LDS writes, body is ~250 dp4a sites
× many cycles each.

### Bottleneck check at MMQ_X=16 (rocprof, before going to MMQ_X=8)

At MMQ_X=16, `VALUBusy = 3.75%` (was 1.0% at MMQ_X=64) — climbing,
but kernel still mostly idle. Per-WG scratch volume dropped 83%, but
**scratch ops per dp4a is still 0.76:1** (was 1.13). The pattern
moves but doesn't break: spills remain the dominant cost, just less
of them. Predicted MMQ_X=8 would give another ~30%; it gave +31%.

**Status: kept at MMQ_X=8.** MMQ_X=4 was tested and confirmed flat
(121.0 tk/s vs 120.9). Reverted to MMQ_X=8 since strictly better:
same perf, half the workgroups (256 vs 512), less dispatch overhead.

### Coherence check (production config: MMQ_X=8 + screening on)

The fix in 8081822 requires `HIPFIRE_MMQ_SCREEN=1` to be coherent on
real Qwen weights (some weight rows have distributions that the dp4a
path can't represent within tolerance — the screen routes them to
FP16 wave64 instead). Without screening, the gibberish reproduces
on commit 8081822 itself, so this is not a regression introduced by
the MMQ_X reduction.

Coherence battery with `HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=1`:

| Prompt | Status | Output |
|--------|--------|--------|
| 0.8B cap | OK | "Paris." |
| 4B code  | OK | one-line `def square(n): return n * n` ✓ |
| 9B reason | OK | "If all but 9 of the sheep die, then 9 are left" ✓ |
| 9B tool-call | OK | proper `<tool_call>` JSON emitted ✓ |
| 9B mq3 reason | OK | minor incoherence (pre-existing on 8081822, not a regression) |
| 27B mq3 cap | OK | "The capital of France is Paris." ✓ |

Real-data NRMSE check: **0.29%** (was 0.25% pre-change). Within tolerance.

**Final production prefill: 125.2 tk/s** (with screening on — slightly
better than 120.9 with screening off because some weights take the
FP16 wave64 path which is faster than dp4a on this kernel size).
That's **89% of FP16 baseline** (140.7 tk/s).

### Bottleneck check at MMQ_X=8 (rocprof)

| Counter      | MMQ_X=64 | MMQ_X=16 | **MMQ_X=8** |
|--------------|----------|----------|-------------|
| VALUBusy     | 1.0%     | 3.75%    | **8.85%**   |
| MemBusy      | 11.2%    | 18.3%    | 24.4%       |
| MemStall     | 2.0%     | 2.5%     | 2.9%        |
| LDS WAIT     | 0        | 0        | 0           |
| Scratch ops/dp4a | 1.13 | 0.76     | 0.58        |
| Avg call ms  | 32.7     | 11.7     | **6.72**    |

Scratch ops per dp4a halved over four steps. VALUBusy still only
8.85% — kernel mostly idle. Remaining time is `s_waitcnt` for VMEM
completions on global_load (HBM→reg) and remaining scratch loads.

**Bottleneck shifted from spill latency to operand fetch latency.**
Next levers:
- L2 prefetch (Phase 0 §6.4 lists this; verified llama.cpp-gfx906
  has the helpers but doesn't actually call them — value uncertain)
- Stream-K work partitioning (https://arxiv.org/abs/2301.03598) —
  flagged for later, mostly relevant when WG count is below
  CU saturation, not our current regime
- Operand reuse via VGPR caching across j-iterations — potential
  but competes with the spill problem we just solved

## Conclusion

The "what" and "where" are clear: **VGPR spilling to scratch is the
single dominant cost, accounting for ~70% of per-call time**. LDS is
clean. Memory bandwidth is barely tickled. The compute path is mostly
correct in shape (dp4a is being emitted with the right operand
density) but the scheduler can't pipeline it under the `(128, 2)`
launch bound with the current accumulator/loop structure.

Two paths forward (no code yet, just direction):

- **Path A — reduce live state**: shrink `mmq_x` to 32, fold
  `8 * scale_w` into `x_dm`, reconsider `#pragma unroll` density.
  Targets the VGPR pressure directly.
- **Path B — change launch bounds**: drop to occupancy=1 (256 VGPRs
  available), accept latency hiding via ILP and prefetch instead of
  TLP. llama.cpp's 1-occupancy variant (`__launch_bounds__(..., 1)`
  at `mmq.cuh:3481`) suggests they use this on some shapes.

Both worth measuring; pick the higher-yield one based on the data.
