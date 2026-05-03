# gfx906 MMQ — Phase 0 complete

Phase 0 of `gfx906_mmq_plan.md` was verification: are MMQ's underlying
assumptions valid on gfx906, and is the projected speedup realistic? This
doc rolls up the four Phase 0 investigations and the llama.cpp-gfx906
reference-architecture review into a single document.

**Bottom line: GO.** The MMQ port is feasible, with revised expectations.
Realistic ceiling on Qwen 3.5 9B prefill is ~225–235 tk/s (1.65–1.7× over
the current 137 tk/s baseline) if we replicate the llama.cpp-gfx906
prefetch optimization. Bypassing prefetch lands at ~190 tk/s (1.4×).

Captured 2026-05-03 on MI50 (gfx906), Qwen 3.5 9B prefill, ROCm 6.4.3.

---

## 1. Baseline measurement

Bench: `./target/release/examples/bench_qwen35_mq4 --prefill 128 --warmup 5 --gen 50`
Config: `HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 HIPFIRE_DPM_WARMUP_SECS=3`.

| Workload | Result |
|---|---|
| Qwen 3.5 9B pp128 prefill | **137 tk/s** |
| Qwen 3.5 9B decode | 50.6 tk/s |

Plan §1 said 74 tk/s prefill; that number was the pre-wave64 baseline.
Current FP16 wave64 hybrid (already merged on the parent branch
`fix/gfx906-prefill-wave64-dispatch`) delivers 137 tk/s. The 74 → 235
"3.2× gap" framing in the plan is now ~1.7×.

---

## 2. Root-cause profile (rocprof) — what is the current FP16 path doing?

rocprof v1, two perf-counter groups (split for HW limit), per-kernel
averages on the four FP16 wave64 GEMM kernels:

| Kernel | VALUBusy% | MemBusy% | MemStall% | LDSBank% | L2 hit% | VGPR | LDS B |
|---|---|---|---|---|---|---|---|
| gate_up | **65.1** | 76.1 | 1.8 | 0.0 | 90.1 | 64 | 0 |
| qkv | 63.7 | 74.8 | 1.9 | 0.0 | (n/a) | 64 | 0 |
| qkvza | 63.7 | 74.6 | 1.9 | 0.0 | 90.1 | 64 | 0 |
| residual | 58.3 | 69.1 | 0.9 | 0.0 | 90.5 | 64 | 0 |

All four: wgr=64, wave_size=64, lds=0 — single wave per workgroup, no LDS.

**Interpretation against plan §Phase 0 go/no-go (>70% VALU = MMQ helps,
<50% = no point):**

The kernels land in the **gray zone** (58–65%). The plan's binary rule
gives no clean answer. The breakdown:

1. **Memory busy but not stalled** (MemBusy 75%, MemStall <2%). HBM/L2 are
   keeping pace — kernel is rate-limited by ALU throughput including
   latency-hidden waitcnt slots, not by waiting on memory.
2. **L2 hit ~90%** absorbs most weight reuse. The plan's framing
   "no LDS staging → memory-bound" is wrong on this hardware. **MMQ's
   value here is arithmetic density (dp4a: 4 MACs/instruction vs FP16 v_fma:
   2 MACs/instruction), not bandwidth savings.**
3. **LDSBankConflict = 0%** because LDS is unused; bank-conflict tuning
   becomes a real concern only after we add LDS staging.
4. **VGPR=64** — already at the gfx906 occupancy=4 ceiling. Adding ~32
   VGPRs of i32 accumulators for a dp4a tile will push us to occupancy=2
   or 1.

VALUBusy at 65% (not 90%) implies ~30% idle ALU slots even before MMQ —
likely waitcnt time. dp4a inherits the same waitcnt structure, so the
arithmetic gain only partially translates to wall-clock.

Raw data: `plans/phase0_fp16_wave64_9b_pp128.csv` (group 0 + L2),
`plans/phase0_fp16_wave64_9b_pp128_g1.csv` (VALU/Mem/LDS).

---

## 3. VGPR pressure for dp4a (derived from rocprof + ISA estimation)

Current FP16 wave64 kernels use 64 arch VGPRs (rocprof column).
gfx906 occupancy bands at 256 threads/WG:

| VGPR/thread | Waves/SIMD | Notes |
|---|---|---|
| ≤32 | 8 | Tight; dp4a 4-acc tile only |
| ≤48 | 5 | dp4a 8-acc tile feasible if no spill |
| ≤64 | 4 | Current FP16 path; comfortable for dp4a |
| ≤84 | 3 | First occupancy drop |
| ≤128 | 2 | Safe ceiling for 128×128 tile |
| ≤256 | 1 | Spill territory |

dp4a 128×128 tile with 8 i32 accumulators (32 VGPRs) + 2 i32 operand temps
(8 VGPRs) + LDS pointers + scale/zp scratch likely lands at 56–80
VGPRs/thread. **Target ≤84 to hold occupancy=3, accept ≤128 (occ=2) as a
fallback.** The plan's "occupancy=1, rely on ILP" is the *worst* case.
llama.cpp-gfx906's `mmf` kernel uses `__launch_bounds__(256, 4)` → 4
WGs/CU — they keep occupancy high.

---

## 4. Wave64 indexing verification (Phase 0 §3.5) — GREEN

Built `benchmarks/bench_wave64_indexing_gfx906.hip` to verify three
patterns the MMQ kernel uses:

1. Block dim `(32, 8, 1)` → linearization `tid = ty*32 + tx`. ✅
2. `tile<16,8,...>::get_i(l)` and `::get_j(l)` formulas. ✅
3. `__shfl_xor(v, off, 32)` width-32 semantics on wave64. ✅

All three pass with **0 errors over 256 thread probes**. The launch shape
`(32, 8, 1)` keeps `threadIdx.x ∈ [0, 32)` regardless of wave size, so
per-warp formulas using `threadIdx.x % 16` and `threadIdx.x / 16` produce
identical (i, j) on wave32 and wave64. Width-32 shuffles correctly stay
within each 32-lane half-wave on wave64.

The plan §3.5's "indexing is likely correct" was right. My mid-investigation
worry that lanes 32..63 of a wave64 would fall out of bounds was a misread:
on this block layout, lanes 32..63 of wave 0 correspond to `(ty=1, tx=0..31)`,
not to `tx=32..63`.

**Caveat:** existing kernel uses `mma_i8` builtin which is RDNA3-only
(`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`); the gfx906 path produces empty
stubs. The wave64 indexing question only matters for the **new** dp4a
compute loop we'll write. The existing `tile<>` abstraction's
`ne = I*J/32` baked-in assumption is something to reconsider when we
write the dp4a loop, but per llama.cpp's pattern (next section) we
should not reuse that abstraction at all on gfx906.

Test artifact: `benchmarks/bench_wave64_indexing_gfx906.hip`.

---

## 5. LDS bank-conflict microbench — inconclusive, deferred

Built `benchmarks/bench_lds_wave64_gfx906.hip` to validate the wiki's
`+1 vec4` rule (1865 vs 3974 GB/s claim). Three problems made the result
unreliable:

1. The compiler aggressively rewrote the unrolled column-sweep into a
   row-rotated read pattern that doesn't match the source bank-conflict
   math.
2. Reported bandwidths (~16 TB/s) exceed the wiki's stated theoretical
   peak (~11 TB/s). Byte counter is ambiguous because the compiler emitted
   `ds_read_b128` (4 b32s per instruction) instead of `ds_read_b32`,
   producing a 2× discrepancy.
3. Stride 32 / 33 / 36 produce nearly identical numbers — contradicts
   wiki, possibly because the compiler restructured around the conflict.

**Decision:** skip standalone LDS bench. Tune padding empirically on the
real MMQ kernel ISA after Phase 1 implementation, using rocprof's
working `LDSBankConflict` counter (verified at 0% on FP16 kernels).

llama.cpp-gfx906 (next section) uses **`+4` element padding** in their
working FP16 GEMM kernel — a concrete data point that supersedes the
wiki's `+1 vec4` guidance.

Tracked as task #5 ("Phase 1+: Verify LDS bank conflicts on real MMQ
kernel via rocprof").

---

## 6. llama.cpp-gfx906 reference architecture

Source tree: `/home/kread/mygit/llama.cpp-gfx906/ggml/src/ggml-cuda/`,
gfx906 paths under `gfx906/{matmul,quantize,attention,fused}/`,
config in `gfx906/gfx906-config.h`.

**Treat as known-good floor, not provable ceiling.** llama.cpp-gfx906 hits
~235 tk/s on Qwen 3.5 9B prefill where our current code gets 137 tk/s, so
it's an unambiguously strong floor. But several choices read as reasonable
defaults (nwarps=2, fixed prefetch budget) rather than fully-explored
optima. Copy first, then look for places to do better.

### 6.1 Workgroup topology — `nwarps = 2` on gfx906

`gfx906-config.h:10` sets `GFX906_MMQ_NWARPS = 2`. Block dim becomes
`(64, 2, 1)` = 128 threads = 2 wave64s where each `threadIdx.y` is exactly
one wave — clean wave-native indexing.

`mmq.cuh:285-298` selects `nwarps` per-arch via a host helper:
```cpp
return amd_mfma_available(cc) ? 8 : 256/warp_size;
// Override on gfx906: GFX906_MMQ_NWARPS (=2)
```

Their tile load (`mmq.cuh:312-358`) uses `nrows = warp_size /
threads_per_row`, so the same source emits 8 rows/warp on wave64 vs 4 on
wave32 — wave size becomes a *feature* (more parallelism per wave), not a
porting hazard.

**Recommendation:** for our gfx906 dp4a port, use `nwarps = 2` and block dim
`(64, 2, 1)`. The 32×8 layout from the RDNA3 kernel is the wrong starting
shape on this hardware.

### 6.2 dp4a primitive — one builtin, not a loop

`common.cuh:666-691`:
```cpp
#if defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);  // clamp = false
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4(true, a, true, b, c, false);  // sudot4 only on RDNA3+
```

gfx906 uses the symmetric `sdot4`. The plan §4 Sign-extension analysis is
correct: HFQ4 nibbles 0..15 are within positive int8 range, safe for
`sdot4` without biasing.

### 6.3 dp4a compute loop — copy this verbatim

`mmq.cuh:1094-1123` for Q8_1 × Q8_1, structurally:

```cpp
for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += vdr) {  // vdr=8 for Q8_1
    const int k0 = k00 + k01;
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            sum[idx] += vec_dot_q8_1_q8_1_impl<vdr>(
                &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0],
                &y_qs[j*MMQ_TILE_Y_K + k01],
                x_dm[...], y_ds[...]);
        }
    }
}
```

And the per-element function (`vecdotq.cuh:254-278`):
```cpp
int sumi = 0;
for (int i = 0; i < vdr; ++i) sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
```

`vdr = 8` → 8 dp4a calls per (i, j, k01). The outer K loop runs
`MMQ_TILE_NE_K / vdr = 32/8 = 4` iterations. **32 dp4a calls per output
element per k-block of 32 K-elements.**

Notes:
- The `m8s8 / (QI8_1 / vdr)` divisor compensates for K-dim chunking
  across threads. This is the plan §3.6 "single bias correction" with the
  exact divisor — easy to miss; copy verbatim.
- Pattern is **direct LDS-reload**, no shuffle dance. The plan's
  "LDS-reload approach (recommended for PoC)" is exactly what they do
  in production. Skip the WMMA tile abstraction entirely on gfx906.

### 6.4 L2 prefetch — undocumented optimization the plan missed

`gfx906/matmul/mmq-prefetch.cuh:1-90`. They use spare lanes from warp 0 to
issue inline `global_load_dword` for the *next* k-block's Y tile. Two
prefetch passes (lanes 0–15 and 16–31), 1 KB each = 2 KB/k-block.

```cpp
asm volatile(
    "global_load_dword %0, %1, off\n"
    : "=v"(prefetch_data) : "v"(prefetch_addr) : "memory"
);
```

**This is the optimization that explains the gap between 137 tk/s and 235
tk/s.** Our current 90% L2 hit rate isn't accidental on llama.cpp's side —
it's *engineered* via prefetch. Without prefetch, the k-block boundary is
where stalls would accumulate.

The plan does not mention prefetch. Add it to Phase 1.

### 6.5 LDS padding — `+4`, not `+1 vec4`

`gfx906/matmul/mmf.cuh:31-32`:
```cpp
constexpr int lds_a_stride = tile_k + 4;
constexpr int lds_b_stride = tile_n + 4;
```

For their FP16 GEMM, padding is `+4` elements. For their MMQ tile,
`mmq.cuh:189-227` has type-specific constants like
`MMQ_MMA_TILE_X_K_Q8_0 = 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4`.
Pattern is consistent: **always `+4 ints`**, supersedes the wiki's
`+1 vec4`.

### 6.6 Tile sizes — they precompile multiple

| Variant | Tile (M × N × K) | Threads | Acc/thread | Occupancy |
|---|---|---|---|---|
| `mmf` (FP16) | 32 × 64 × 64 | 256 | 8 | 4 (`__launch_bounds__(256, 4)`) |
| MMQ dp4a | mmq_y × mmq_x × MMQ_TILE_NE_K | warp_size × nwarps | sum_per_thread | configurable |

llama.cpp uses `mmq_x ∈ {8, 16, 32, 64, 128}` depending on batch size.
Our MMQ kernel hard-codes `MMQ_X = MMQ_Y = 128`.

**Recommendation:** initial PoC at 128×128 is fine, but plan for the
dispatch table to pick from at least {64×64, 128×128} based on N. Plan
§3.2's argument for sticking with 128×128 still holds for *initial*
testing, but production should select.

### 6.7 Where they are likely **not** optimal

- **Wiki ILP4 sweet spot**: dot4 hits ~80% of theoretical peak at ILP=4.
  llama.cpp's `mmf` uses ILP=8 — possibly suboptimal per the wiki, worth
  measuring ILP4 vs ILP8 in our kernel.
- **No `v_dot8_u32_u4`**. Per wiki, dot8 doubles dot4 throughput at the
  same ILP4 (43→86 TOPS). HFQ4 is already nibble-packed → Path B
  (plan §5) directly consuming HFQ4 nibbles via `udot8` skips the
  per-use unpack. May exceed llama.cpp's dp4a-only ceiling if Path A
  shows compute-bound behavior.
- **Fixed prefetch budget** (16+16 lanes, 1 KB each). Adapting to the
  actual k-block dwell time may be more efficient.
- **mmq_x precompiled set**. An empirical sweep over our specific Qwen
  3.5 9B GEMM dimensions might find a better fit.

Phase 1 should target llama.cpp as an **interim correctness/perf
reference**, then Phase 2/3 chase the gaps above.

### 6.8 Files worth reading in full when implementing

| File | Purpose |
|---|---|
| `gfx906/gfx906-config.h` | NWARPS=2 and other gfx906 constants |
| `gfx906/matmul/mmq.cuh` | Tile-load macros (Q8_0 specifically) |
| `gfx906/matmul/mmq-prefetch.cuh` | L2 prefetch implementation |
| `gfx906/matmul/mmf.cuh` | FP16 GEMM with 8-acc ILP — pattern reference |
| `mmq.cuh:1094-1123` | dp4a Q8_1×Q8_1 compute loop |
| `vecdotq.cuh:254-278` | Per-element dp4a accumulator |
| `common.cuh:666-691` | `ggml_cuda_dp4a` builtin wrapper |

---

## 7. Revised throughput estimate

| Approach | Conservative | Optimistic | Days |
|---|---|---|---|
| Current FP16 wave64 (baseline) | 137 tk/s | 137 tk/s | — |
| Path A: dp4a MMQ, no prefetch | 175 (1.28×) | 195 (1.42×) | 8–12 |
| Path A: dp4a MMQ + L2 prefetch | 215 (1.57×) | 245 (1.79×) | +3–5 |
| Path B: dot8/HFQ4-direct (if Path A compute-bound) | 230 (1.68×) | 285 (2.08×) | +8–12 |
| llama.cpp gfx906 reference | — | 235 tk/s | — |

Plan §9 estimates were anchored to 74 tk/s and predicted 92–130 tk/s
"on residual-only." Updated against the 137 tk/s wave64 baseline and the
llama.cpp-gfx906 reference, the realistic ceiling for Path A with
prefetch is **205–235 tk/s on full prefill** (1.5–1.7×), reaching parity
with llama.cpp. Path B remains exploratory.

---

## 8. Phase 0 task closure

| Task | Status | Notes |
|---|---|---|
| Establish baseline prefill numbers | ✅ done | 137 tk/s captured; formal speed-gate baseline deferred until Phase 1 implementation |
| Profile FP16 wave64 to determine bottleneck | ✅ done | gray-zone (58-65% VALU); GO with adjusted expectations |
| VGPR pressure estimation for dp4a loop | ✅ done | derived from rocprof; target ≤84 VGPR for occupancy=3 |
| Wave64 indexing verification test | ✅ done | 0 errors; plan §3.5 was correct |
| Verify LDS bank conflicts on real MMQ kernel via rocprof | ⏳ deferred to Phase 1 | microbench inconclusive; tune empirically on real kernel |

---

## 9. Phase 1 entry criteria

- Use `nwarps = 2`, block dim `(64, 2, 1)` — wave-native.
- Write a fresh dp4a compute loop following llama.cpp pattern; do not
  port the existing RDNA3 `tile<>` abstraction or `mma_i8` paths.
- Include L2 prefetch from day one (don't bolt on later).
- Use `+4 int` LDS padding (not `+1 vec4`).
- Target gate_up first (45% of prefill time per plan §3.4).
- Gate behind `HIPFIRE_MMQ=1` env override; do not change default
  dispatch until Phase 2.
- Speed-gate baseline file (`tests/speed-baselines/gfx906.txt`) captured
  *before* the first dp4a commit so the floor is defended.

---

## 10. Raw artifacts

- `plans/phase0_fp16_wave64_9b_pp128.csv` — rocprof L2 + arch_vgpr
- `plans/phase0_fp16_wave64_9b_pp128_g1.csv` — VALU/Mem/LDS counters
- `plans/phase0_rocprof_counters.txt` — rocprof input file
- `plans/phase0_rocprof_group1.txt` — VALU group input file
- `benchmarks/bench_wave64_indexing_gfx906.hip` — wave64 indexing test
- `benchmarks/bench_lds_wave64_gfx906.hip` — LDS microbench (kept for ref)
- llama.cpp-gfx906 reference: `/home/kread/mygit/llama.cpp-gfx906/`
