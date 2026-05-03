# gfx906 MMQ Adaptation Plan

**Goal:** Close the prefill gap to llama.cpp-gfx906 (~235 tk/s on Qwen 3.5 9B) by adapting hipfire's tiled MMQ kernel to gfx906 (MI50/MI60).

**Status:** Phase 0 complete (2026-05-03). Feasibility confirmed against measured rocprof + llama.cpp-gfx906 reference. Phase 1 entry criteria below. See `plans/phase0_complete.md` for the full Phase 0 rollup (rocprof data, wave64 indexing test, llama.cpp synthesis, revised throughput estimates).

**Current baseline:** 137 tk/s on Qwen 3.5 9B pp128 prefill (FP16 wave64 hybrid already merged). The 74 tk/s number cited earlier in this plan was the pre-wave64 baseline — gap to llama.cpp is ~1.7×, not 3.2×.

**Source reviews:** `gfx906_vec_plan_rev_glm5.md` Appendix A, `gfx906_mmq_plan_gemini.md`, `gfx906_mmq_plan_claude.md`. **Reference implementation:** `/home/kread/mygit/llama.cpp-gfx906/ggml/src/ggml-cuda/gfx906/`.

---

## 1. Background

hipfire's gfx906 prefill uses row-parallel FP16 wave64 kernels (`*_fp16_wave64.hip`). The wave64 hybrid path already merged on the parent branch achieves **137 tk/s on Qwen 3.5 9B pp128 prefill** (rocprof measured 2026-05-03). llama.cpp-gfx906 achieves ~235 tk/s on the same model class.

Phase 0 rocprof shows the FP16 wave64 kernels are in a compute-leaning gray zone (VALUBusy 58–65%, MemBusy 75%, MemStall <2%, L2 hit ~90%). The current path is rate-limited by ALU throughput including waitcnt slots, not by HBM bandwidth — L2 already absorbs most weight reuse. **MMQ's value on this hardware is arithmetic density (dp4a 4 MACs/instruction vs FP16 v_fma 2 MACs/instruction), not bandwidth savings.**

The architectural differences:
- **hipfire:** One warp per output row, no LDS, no explicit data reuse. Each weight element loaded once but L2 catches reuse.
- **llama.cpp-gfx906:** Tiled 128×128 GEMM with LDS staging, pre-quantized activations, dp4a (`__builtin_amdgcn_sdot4`), **explicit L2 prefetch** for the next k-block (this is undocumented in any wiki and accounts for a meaningful fraction of their lead — see §3.7 below).

hipfire already has an MMQ kernel (`gemm_hfq4g256_residual_mmq.hip`) that implements the llama.cpp approach — but it's gated to RDNA3 only because it uses `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`, which doesn't exist on gfx906.

gfx906 has `v_dot4_i32_i8` (dp4a) and `v_dot8_i32_i4` (dot8) — both verified on real MI50 hardware in the skyne98 wiki. These are the path to integer dot-product acceleration on gfx906.

---

## 2. Architecture reference (gfx906)

From skyne98 wiki (2026-02-21, measured on real MI50):

| Property | Value |
|---|---|
| CUs | 60 (MI50) / 64 (MI60) |
| Clock | 1725 MHz |
| Wavefront size | 64 |
| LDS/CU | 64 KiB, 32 banks |
| L1 Instruction Cache/CU | 32 KiB |
| L2 | 4 MiB |
| VGPR file/CU | 256 KiB |
| HBM2 bandwidth | ~1 TB/s |

**Integer dot-product instructions (verified with `llvm-mc -mcpu=gfx906`):**

| Instruction | Intrinsic | Throughput (ILP4) | MACs/op |
|---|---|---|---|
| `v_dot4_i32_i8` | `__builtin_amdgcn_sdot4` | 43-44 TOPS | 4 |
| `v_dot8_i32_i4` | `__builtin_amdgcn_sdot8` | 85-86 TOPS | 8 |
| `v_dot8_u32_u4` | `__builtin_amdgcn_udot8` | 85-86 TOPS | 8 |
| `v_dot4c_i32_i8` | — | **NOT available** | — |
| `v_dot8c_i32_i4` | — | **NOT available** | — |
| `v_mfma*` | — | **NOT available** | — |

**QDQ amortization (wiki FP32 vs QDQ study):**

| Scenario | FP32 | dot4 (i8) | dot2 (f16) |
|---|---|---|---|
| Per-use QDQ in hot loop | 5.95 TOPS | **2.00 TOPS** | 4.19 TOPS |
| Amortized (convert once, reuse) | 13.0 TOPS | **21.7 TOPS** | 21.9 TOPS |

MMQ's tile reuse (128x128 = 16,384 dot products per tile load) satisfies the amortization condition. Pre-quantization overhead is paid once; the arithmetic advantage is real.

**LDS bandwidth (wiki latency-hiding study):**

| Instruction | Bandwidth |
|---|---|
| `ds_read/write_b32` | 1.9-3.9 TB/s |
| `ds_read/write_b64` | 4.3-8.8 TB/s |
| `ds_read/write_b128` | 9.5-11.2 TB/s |

Use `ds_read/write_b128` for all LDS staging. No `s_clause`/`s_waitcnt_depctr` on gfx906 — rely on ILP and `s_waitcnt` timing.

**LDS bank conflicts (wiki LDS layout study):**

- Column-style stride=32 vec4: **1865 GB/s** (56% collapse)
- Column-style stride=33 vec4: **3974 GB/s** (93% recovered)

Rule: pad column-consumed LDS strides to `K_vec + 1` in vec4 units.

**Wave64 LDS baseline:** gfx906 has 32 banks but 64-lane waves. Every `ds_read_b128` from a wave64 has an inherent 2-way bank conflict (lane 0 and lane 32 both hit bank 0). The wiki's 93% recovery number already accounts for this. Target is "2-way peak" (the native wave64 rate), not "conflict-free" (impossible for wave64 on 32 banks).

---

## 3. Blockers

### 3.1 WMMA does not exist on gfx906 — dp4a loop is non-trivial

The MMQ kernel's compute is `mma_i8` (`gemm_hfq4g256_residual_mmq.hip:170-182`):

```cpp
acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], true);
acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true);
```

This is a single ISA op on RDNA3. On gfx906 it must be replaced with dp4a or dot8 loops.

**Instruction budget per mma_i8 call:**

| Approach | Instructions per mma_i8 | MACs | Relative to WMMA |
|---|---|---|---|
| WMMA (RDNA3) | 2 | 4096 | 1x |
| dp4a | 1024 | 4096 | 512x more instructions |
| dot8 | 512 | 4096 | 256x more instructions |

**Implementation complexity (revised after Claude review):**

The WMMA layout (`DATA_LAYOUT_I_MAJOR_MIRRORED`) distributes the 16x8 tile across 32 threads in a pattern optimized for WMMA addressing. dp4a needs explicit element access, which means either:

1. **LDS-reload approach (recommended for PoC):** Skip the WMMA tile abstraction entirely. Write a raw dp4a loop that reads operands directly from the LDS arrays. Each thread computes its (row, col) output subset by reading weight and X data from LDS in the dp4a loop. Simpler code, more LDS reads, but avoids cross-thread shuffles. Start here.

2. **Shuffle-optimized approach (later optimization):** Use `__shfl_sync` to redistribute data from WMMA layout to dp4a-friendly layout. More complex, fewer LDS reads, but higher register pressure and implementation risk.

The LDS-reload approach increases LDS traffic but avoids register/shuffle complexity. Since LDS bandwidth is 9.5-11.2 TB/s (vs HBM at ~1 TB/s), the extra LDS reads are cheap compared to the simplicity gained.

**Revised Task A1 estimate:** 3-5 days (not 1 day), accounting for layout debugging and correctness verification.

**I-cache pressure (from Gemini review):** A heavily unrolled dp4a loop body may exceed the 32 KiB L1 icache. Prioritize loop compression (compact loops with `#pragma unroll` on small inner loops) over full unrolling. Verify with `rocprof` for `FETCH_UNIT_BUSY` stalls.

**Frontend pressure:** On gfx906, 512+ dp4a instructions must pass through the main VALU issue slot (RDNA3 has a specialized WMMA unit). Instruction cache misses are a real risk — the hot loop must fit in 32 KiB icache.

### 3.2 Tile size: start at 128x128, not 64x64 (revised)

Both adversarial reviews identified that reducing tile size from 128x128 to 64x64 has a hidden cost: arithmetic intensity drops 2x.

**Arithmetic intensity analysis:**

| Tile | LDS loaded per K-block | Compute per K-block | MACs/byte |
|---|---|---|---|
| 128x128 | 128x256 w + 128x256 x = 65,536 int8 | 128x128x256 = 4.2M MACs | **64** |
| 64x64 | 64x256 w + 64x256 x = 32,768 int8 | 64x64x256 = 1.0M MACs | **32** |
| 32x128 | 128x256 w + 32x256 x = 40,960 int8 | 128x32x256 = 1.0M MACs | **26** |

Smaller tiles mean each byte loaded from LDS produces fewer MACs. If the kernel is LDS-bandwidth-bound, smaller tiles are slower regardless of occupancy gains.

**Revised strategy (both reviews agree):**

1. **Start with 128x128 tile at occupancy=1.** Accept single workgroup/CU. Rely on ILP (8+ independent accumulators) to hide memory latency rather than occupancy.
2. **Measure VALU utilization** with `rocprof --perfcounter SQ_ACTIVE_INST_VALU`. If >70%, kernel is compute-bound — dp4a helps. If <50%, kernel is memory-bound — dp4a arithmetic advantage is irrelevant.
3. **Only reduce to 64x64 or 32x128** if profiling shows occupancy (not compute or LDS bandwidth) is the bottleneck.

### 3.3 LDS bank conflict padding needs gfx906-specific tuning — **revised (2026-05-03)**

Phase 0 attempted a standalone LDS microbench (`benchmarks/bench_lds_wave64_gfx906.hip`) but the compiler aggressively rewrote the access pattern, making the result inconclusive (see Phase 0 §5). Two concrete data points override the earlier wiki guidance:

- llama.cpp-gfx906 uses **`+4` int padding** in their working kernels (`gfx906/matmul/mmf.cuh:31-32`: `lds_a_stride = tile_k + 4`, `lds_b_stride = tile_n + 4`; `mmq.cuh:189-227`: type-specific tile stride constants all end in `+4`). This supersedes the wiki's `+1 vec4` recommendation.
- rocprof's `LDSBankConflict` counter is verified working (0% on existing FP16 kernels because LDS is unused).

**Action:** Use `+4 int` padding for MMQ tile staging from the start. After Phase 1 implementation, profile the live kernel with rocprof `LDSBankConflict`. If <5% conflict on the real kernel, no further tuning needed. If higher, inspect the emitted `ds_read*` ISA and adjust empirically.

### 3.7 L2 prefetch — **new (2026-05-03), missed in original plan**

llama.cpp-gfx906's most impactful gfx906-specific optimization is **explicit L2 prefetch of the next k-block's Y tile via inline `global_load_dword`** issued by spare lanes in warp 0 (`gfx906/matmul/mmq-prefetch.cuh`). Two prefetch passes (lanes 0–15 and 16–31), 1 KB each = 2 KB warmed into L2 per k-block.

```cpp
asm volatile(
    "global_load_dword %0, %1, off\n"
    : "=v"(prefetch_data) : "v"(prefetch_addr) : "memory"
);
```

The current FP16 wave64 kernel already shows 90% L2 hit rate (Phase 0 rocprof). For llama.cpp's MMQ to maintain comparable L2 hit while running 4× faster on dp4a, prefetch is doing real work — without it, the k-block boundary is where stalls would accumulate.

**Action:** Include L2 prefetch in Phase 1 from day one. Do not implement dp4a MMQ first and add prefetch later — it changes the dispatch shape (need spare lanes available in warp 0 during compute) and retrofitting is harder than building it in. Estimated size: ~50 LOC, matches the `gfx906_prefetch_y_tile_v4` and `_second` reference.

This is the single biggest gap between "dp4a MMQ that beats current FP16" and "dp4a MMQ that matches llama.cpp." Estimated impact: **~30% of llama.cpp's lead vs naive dp4a**.

### 3.4 Fused kernel ordering: gate_up first (revised)

The original plan started with residual MMQ (27% of prefill). Both reviews correctly identified this as suboptimal.

**Prefill kernel time breakdown:**

| Kernel | MMQ variant? | Prefill share |
|---|---|---|
| `gemm_gate_up_hfq4g256_fp16_wave64` | No | **~45%** |
| `gemm_hfq4g256_residual_fp16_wave64` | Yes | ~27% |
| `gemm_qkvza_hfq4g256_fp16_wave64` | No | ~17% |
| `gemm_qkv_hfq4g256_fp16_wave64` | No | ~5% |

**Why gate_up first:** Even a 2x improvement on residual alone gives only 1.14x overall (27% halved). A 2x improvement on gate_up gives 1.22x overall (45% halved). gate_up is the highest-impact single target.

**Implementation note:** gate_up MMQ requires 2 weight matrix pointers (A_gate, A_up) and 2 output pointers (Y_gate, Y_up). The dp4a compute core is shared — only the dispatch and output routing differs. Estimate +2 days over residual-only.

### 3.5 Wave64 vs wave32 indexing — **VERIFIED (2026-05-03)**

**Phase 0 result: GREEN.** `benchmarks/bench_wave64_indexing_gfx906.hip` ran three indexing checks (block-dim linearization, `tile<>::get_i/get_j`, `__shfl_xor` width=32 reduction) on real gfx906 hardware. **0 errors over 256 thread probes.**

Why it works: the existing kernel launches with block dim `(32, 8, 1)`. That keeps `threadIdx.x ∈ [0, 32)` *regardless of wave size*, so per-warp formulas using `threadIdx.x % 16` and `threadIdx.x / 16` produce identical results on wave32 and wave64. Width-32 shuffles correctly stay within each 32-lane half-wave on wave64 (lanes 0–31 reduce among themselves, lanes 32–63 reduce among themselves).

**However, this conclusion is for the existing kernel structure only.** Per §6.1 of the Phase 0 rollup, llama.cpp-gfx906 uses a different (better) topology: **`nwarps = 2`, block dim `(64, 2, 1)`** — wave-native, one wave per `threadIdx.y`. When we write the dp4a port we should adopt that shape (see §5 Path A revisions below), not preserve the RDNA3 32×8 inheritance. The wave64 indexing check above means: porting either way is safe, the existing shape *would* work, but we should use `(64, 2, 1)` for cleaner code and to follow the reference.

### 3.6 Correction tax for dot8 (Path B) — resolved

Gemini raised that HFQ4 is affine (`w = scale * nibble + zp`), requiring 4 correction terms when both operands use affine quantization:

```
sum(w*x) = scale_w * scale_x * sum(nib*qx) + zp_w * sum(x_orig) + scale_w * sum(nib) * zp_x + N * zp_w * zp_x
```

**Resolution:** For Path A (dp4a with Q8_1 activations), Q8_1 is **symmetric** (no zero_point). HFQ4 × Q8_1 requires only **2 correction terms**, not 4:

```cpp
sum += scale_w * scale_x * raw_dot(nib, qx);  // main term
sum += zp_w * sum(x_orig);                     // single bias correction
```

The existing MMQ kernel already implements this correctly at `gemm_hfq4g256_residual_mmq.hip:324-325`:
```cpp
sum[...] += dmA.x * dsB.x * (float)C_frag.x[l];  // scale_w * scale_x * raw_dot
sum[...] += dmA.y * dsB.y;                         // zp_w * sum_x
```

**For Path B (dot8):** If X is quantized to a Q4_1 format (symmetric, no zp), the same 2-term correction applies. If X uses an affine Q4 format (with zp), 4 terms are needed and the correction tax (~10-15 ALU ops per dot8 call) could wipe out dot8's 2x arithmetic advantage. **Use symmetric quantization for X in Path B** to avoid this.

---

## 4. Sign-extension analysis

### For dp4a (Path A)

HFQ4 nibbles are unsigned 0-15. The existing unpack in `load_hfq4_tile` (line 248-257) packs nibbles into the low nibble of each byte in an int32:

```cpp
x_qs[...] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
```

`sdot4` interprets each byte as signed int8. Values 0-15 are within positive int8 range (0-127), so **unsigned nibbles 0-15 are safe for sdot4**. No biasing needed.

Verified by existing dp4a usage in `gemv_hfq4g256.gfx1030.v4.hip:80-87` — same nibble packing pattern.

### For dot8 (Path B)

`sdot8` treats each 4-bit field as signed int4 (-8 to +7). Nibbles 8-15 would be misinterpreted as negative.

**Fix:** Use `__builtin_amdgcn_udot8` (`v_dot8_u32_u4`) which treats operands as unsigned int4 (0-15). Verified available on gfx906 in the wiki. Accumulator is still i32. No biasing needed.

---

## 5. Implementation paths

### Path A: dp4a MMQ (recommended) — **revised post-Phase 0**

Take llama.cpp-gfx906 as the known-good reference for the compute structure. Do **not** port the existing RDNA3 MMQ kernel (`gemm_hfq4g256_residual_mmq.hip`) verbatim — its 32×8 thread topology, `tile<>` abstraction, and `mma_i8` builtin are all RDNA3-shaped and not the right starting point on gfx906.

**Topology (per llama.cpp-gfx906 §6.1):**
- `nwarps = 2`, block dim `(64, 2, 1)` — wave-native, one wave per `threadIdx.y`.
- `__launch_bounds__(128, 2)` to start (2 WGs/CU, 256 threads/CU = 4 wave64s/CU). Aim for higher occupancy if VGPR budget allows; llama.cpp's `mmf` runs at 4 WGs/CU.

**Compute loop (per llama.cpp-gfx906 §6.3):** the simple direct-LDS pattern from `mmq.cuh:1094-1123`:
```cpp
for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += vdr) {       // vdr = 8 for Q8_1
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            sum[idx] += vec_dot_q8_1_q8_1_impl<vdr>(...);  // 8 dp4a calls + zp correction
        }
    }
}
```
Per-element: `for (i=0; i<vdr; ++i) sumi = __builtin_amdgcn_sdot4(v[i], u[i], sumi, false);`
followed by `return sumi*d8d8 + m8s8 / (QI8_1 / vdr);`. Copy this verbatim — the `/ (QI8_1/vdr)` divisor is easy to miss and required.

**HFQ4 specifics (different from llama.cpp's Q8_0):** llama.cpp's path is Q8_0 weights × Q8_1 activations (both signed). Our weights are HFQ4 (unsigned nibbles 0–15, asymmetric scale+zp per group). Sign-extension §4 still applies: nibbles 0–15 are within positive int8 range, safe for `sdot4` with no biasing. Zero-point is handled outside the dp4a loop via the `m8s8` correction term.

**LDS staging:** `+4 int` padding (per llama.cpp; see §3.3 revision above). Stage HFQ4 weights once per k-block, reuse across all `mmq_x` columns.

**L2 prefetch (per §3.7):** include from day one, not as an afterthought.

**Tile sizes:** start with `mmq_y × mmq_x = 128 × 128`, but structure the kernel as a template on `mmq_x` so we can dispatch multiple precompiled variants later (llama.cpp uses 8/16/32/64/128).

**Tasks:**

| # | Task | Days | Files |
|---|---|---|---|
| ~~A0~~ | ~~Wave64 indexing verification test~~ | ~~done in Phase 0~~ | `benchmarks/bench_wave64_indexing_gfx906.hip` |
| A1 | New gfx906 dp4a MMQ kernel: nwarps=2, direct-LDS dp4a, +4 padding, with L2 prefetch | 5–7 | new `gemm_hfq4g256_residual_mmq.gfx906.hip` |
| A2 | Port dp4a core to gate_up (2 weight ptrs, 2 output ptrs) | 2 | new `gemm_gate_up_hfq4g256_mmq.gfx906.hip` |
| A3 | Add gfx906 preprocessor guard + Q8_1 quantize variant if needed | 0.5 | same |
| A4 | Wire dispatch: add gfx906 to `has_mmq_i8_wmma()` (rename to `has_mmq_dp4a` or split) and `should_use_mmq` arch-specific batch threshold | 1 | `dispatch.rs` |
| A5 | rocprof LDS bank conflict + I-cache verification on real kernel (replaces failed Phase 0 microbench) | 0.5 | profiling, no kernel change |
| A6 | MMQ screening + correctness validation | 1–2 | existing screening infra |
| A7 | Performance tuning: ILP4 vs ILP8 sweep, prefetch budget, tile sizes | 2–3 | same |

**Total:** 12–16 days. Two days saved (Phase 0 wave64 test done) but added explicit prefetch from day one.

**Expected:** see §9 (revised throughput estimates).

### Path B: dot8 MMQ (deprioritized)

Use `v_dot8_u32_u4` with raw HFQ4 nibbles. No weight unpacking. New X quantization to symmetric int4.

**Deprioritization rationale (from Gemini review):** The "correction tax" of HFQ4's affine format adds ~10-15 ALU ops per dot8 call if X also uses affine quantization. Even with symmetric X quantization, the need for a new quantization kernel, new LDS layout, and new correction logic makes Path B high-risk for uncertain gain. Path A with aggressive ILP is more likely to reach 150-180 tk/s than Path B is to reach 235 tk/s.

**Kept as Phase 3** but only if Path A demonstrates that the dp4a MMQ approach is viable and the bottleneck is clearly arithmetic throughput.

### Fused MMQ variants (integrated into Path A)

| Kernel | Fused outputs | Est. effort |
|---|---|---|
| `gemm_gate_up_hfq4g256_mmq` | 2 (gate, up) | 2 days (included in A2) |
| `gemm_qkv_hfq4g256_mmq` | 3 (q, k, v) | 3-4 days |
| `gemm_qkvza_hfq4g256_mmq` | 4 (qkv, z, beta, alpha) | 4-5 days |

---

## 6. Phased rollout

### Phase 0: Verification — **DONE 2026-05-03**

See `plans/phase0_complete.md` for full rollup. Outcome: **GO**.

- ✅ Baseline: 137 tk/s on Qwen 3.5 9B pp128.
- ✅ Root-cause: VALUBusy 58–65% (gray zone), L2 hit 90%, MemStall <2%. Compute-leaning. dp4a's arithmetic density is the right lever.
- ✅ VGPR baseline: 64 (current FP16). dp4a target ≤84 for occ=3, ≤128 for occ=2.
- ✅ Wave64 indexing: 0 errors over 256 thread probes (`benchmarks/bench_wave64_indexing_gfx906.hip`).
- ⏭️ LDS bank-conflict microbench: inconclusive, deferred to Phase 1+ (use rocprof `LDSBankConflict` on real kernel).

Plan corrections from Phase 0:
- Switch to llama.cpp-gfx906 topology (`nwarps=2`, block `(64, 2, 1)`).
- Add §3.7 L2 prefetch as a new blocker.
- Use `+4 int` LDS padding (not `+1 vec4`).
- Don't port the existing RDNA3 MMQ kernel — write fresh against the llama.cpp-gfx906 pattern.

### Phase 1: dp4a PoC with prefetch (8–12 days)

- A1 + A2: new dp4a MMQ kernel (residual + gate_up), `nwarps=2` topology, direct-LDS pattern from `mmq.cuh:1094-1123`, `+4` padding, **L2 prefetch from day one**.
- Speed-gate baseline file `tests/speed-baselines/gfx906.txt` captured before the first kernel commit so the floor is defended.
- Gate behind `HIPFIRE_MMQ=1` env override; do not change default dispatch.
- Q8_1 activation quantize kernel: replace the dead `#else` stub at `gemm_hfq4g256_residual_mmq.hip:40-45` with a working gfx906 implementation (reference: llama.cpp's `quantize_mmq_q8_1` — verify against the existing wave64 indexing analysis in §3.5).

**Go criteria:** ≥1.3× full-prefill on Qwen 3.5 9B (≥178 tk/s). Includes gate_up. Decode regression <2%. Coherence gate passes.

### Phase 2: Tuning + dispatch integration (4–6 days)

- A5: rocprof `LDSBankConflict` and `FETCH_UNIT_BUSY` on the real kernel; tune padding and unroll if conflicts >5% or icache stalls measurable.
- A6: MMQ screening for gfx906 (`dispatch.rs`).
- A7: ILP4 vs ILP8 sweep, prefetch budget tuning, tile-size sweep (128×128 vs 64×64).
- A4: dispatch integration — add gfx906 to `has_mmq_dp4a()` (rename from `has_mmq_i8_wmma()` since the underlying primitive differs).

**Success criteria:** ≥1.5× full-prefill (≥205 tk/s). Decode within 1% of baseline. No coherence regression.

### Phase 3: dot8 / HFQ4-direct MMQ (8–12 days, conditional)

Only pursued if Phase 2 rocprof shows the dp4a kernel is **clearly compute-bound** (VALUBusy >75%) AND wall-clock is below 230 tk/s. Otherwise skip — Path A with prefetch should reach llama.cpp parity, and the new quant format / LDS layout / correction logic in Path B is a high-cost route to incremental wins above llama.cpp.

**Success criteria:** ≥1.7× full-prefill (≥235 tk/s — llama.cpp parity).

### Phase 4: Remaining fused variants (5–8 days, if Phase 2 passes)

- Port dp4a + prefetch to `gemm_qkv_hfq4g256` (5% prefill share) and `gemm_qkvza_hfq4g256` (17% prefill share).
- Full-prefill benchmark on real model.

**Success criteria:** ≥1.6× full-prefill (≥220 tk/s) sustained across the Qwen 3.5 27B and 9B prefill mix.

---

## 7. Risk register

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| dp4a throughput insufficient | High | Medium | Phase 0 VALU profiling reveals this before implementation |
| I-cache thrashing from large loop body | High | Medium | Loop compression; rocprof FETCH_UNIT_BUSY check; 128x128 tile (1 WG/CU avoids icache contention between WGs) |
| LDS bank conflicts on gfx906 | Medium | Medium | Wiki `+1 vec4` rule; microbenchmark in A5 |
| 1 workgroup/CU at 128x128 tile | Medium | Certain | Rely on ILP8+ to hide latency; benchmark smaller tiles only if profiling shows occupancy bottleneck |
| dot8 correction tax wipes arithmetic advantage | High | Medium | Use symmetric X quantization (2 terms, not 4); deprioritize Path B |
| VGPR spill under dp4a register pressure | Medium | Medium | rocprof VGPR profiling; LDS-reload approach (no shuffle) minimizes register use |
| Wave64 indexing bug | High | Low | Phase 0 verification test; analysis suggests indexing is correct |
| Scale/zp precision loss from int8 quantization | Medium | Low | MMQ screening catches this; existing Q8_1 correction validated on RDNA3 |
| No fused gate_up MMQ → limited overall impact | High | Low | gate_up is Phase 1 target (not residual) |
| Quantization noise changes model output | Medium | Low | Expected — use fluency/topic coherence, not byte-exact match |

---

## 8. Testing

### Correctness

- **Wave64 verification (Phase 0):** Minimal kernel that verifies thread→tile mapping on wave64. Compare quantize kernel output against CPU reference for small inputs (N=1, K=256).
- **MMQ screening** (`mmq_screen_weight`): per-weight comparison of MMQ vs FP16 reference. Already implemented in `dispatch.rs:1011-1083`. Extend to gfx906. This catches scale/zp precision loss.
- **Numerical tolerance test:** Run small GEMM (M=128, K=4096, N=8) through both FP16 kernel and MMQ kernel. Expect token-level diffs (quantization noise). Pass criteria: `max_abs_error` within reasonable tolerance, no systematic bias.
- **Coherence gate:** `./scripts/coherence-gate.sh` will show token diffs from quantization noise. This is expected. Pass criteria: output is fluent and on-topic, no attractor loops, reasonable perplexity vs FP16 baseline.

### Performance

- **Prefill benchmark:** `./scripts/speed-gate.sh` (measures `prefill_tok_s` at pp32 and pp128)
- **rocprof counters:** `--basic-block-profiler` for VGPR/LDS occupancy; `--perfcounter SQ_ACTIVE_INST_VALU` for VALU utilization; `--perfcounter FETCH_UNIT_BUSY` for I-cache pressure
- **ISA inspection:** `hipcc -S --offload-arch=gfx906 -O3` → verify dp4a instructions emitted, check loop body size

### Multi-architecture

- Test on gfx906 (primary target) and gfx1100 (regression check — MMQ must still work on RDNA3)

---

## 9. Revised throughput estimates — **post-Phase 0 (2026-05-03)**

Anchored to the measured 137 tk/s baseline (FP16 wave64 hybrid, Qwen 3.5 9B pp128) and the llama.cpp-gfx906 reference (~235 tk/s on the same model class). All numbers full-prefill, not residual-only.

| Approach | Conservative | Optimistic | Probability | Days |
|---|---|---|---|---|
| Current FP16 wave64 (baseline) | 137 tk/s | 137 tk/s | — | — |
| Path A: dp4a MMQ, **no prefetch** (gate_up + residual) | 175 (1.28×) | 195 (1.42×) | 80% | 8–12 |
| Path A: dp4a MMQ + **L2 prefetch** | **215 (1.57×)** | **245 (1.79×)** | 65% | 12–16 |
| Path B: dot8/HFQ4-direct (only if Path A compute-bound) | 230 (1.68×) | 285 (2.08×) | 30% | +8–12 |
| llama.cpp-gfx906 reference | — | 235 tk/s | — | — |

**Key insights from Phase 0:**

1. The original plan's 92–130 tk/s estimate was anchored to 74 tk/s — a pre-wave64 number. Against the actual 137 tk/s baseline the same multipliers (1.25–1.75×) give **170–240 tk/s**, which puts Path A's optimistic case at parity with llama.cpp.
2. **L2 prefetch is the difference between "dp4a MMQ that beats current FP16" and "dp4a MMQ that matches llama.cpp."** It is not optional for parity — the plan originally missed this entirely (§3.7 added post-Phase 0).
3. Path A with prefetch is a credible parity target. **Beating 235 tk/s** is open: wiki ILP4 sweet spot suggests room above llama.cpp's ILP8 in `mmf`; `v_dot8_u32_u4` directly on HFQ4 nibbles (Path B) skips the per-use unpack and could beat dp4a peak.
4. VALUBusy at 65% on FP16 implies ~30% idle ALU time in the current path. dp4a inherits the same waitcnt structure, so the *full* arithmetic gain doesn't translate to wall-clock — Amdahl on the non-compute fraction caps the realistic win at ~1.7–1.8× even with perfect dp4a.

---

## 10. References

- skyne98 wiki: https://skyne98.github.io/wiki-gfx906/
  - Architecture baseline: `studies/2026-02-21/mi50-mi60-architecture-baseline.html`
  - dot4/dot8 exploration: `studies/2026-02-21/gfx906-dot4-dot8-exploration.html`
  - Special ISA for quant/dequant: `studies/2026-02-21/gfx906-special-isa-quant-dequant.html`
  - FP32 vs QDQ dot: `studies/2026-02-21/fp32-vs-qdq-dot-gfx906.html`
  - LDS layout: `studies/2026-02-21/gfx906-lds-layout-standard-llm.html`
  - Latency hiding: `studies/2026-02-21/gfx906-latency-hiding-ops.html`
- LLVM gfx906 ISA: https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html
- LLVM gfx908 ISA (contrast — has v_mfma): https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX908.html
- Existing MMQ kernel (RDNA3, do not port verbatim): `kernels/src/gemm_hfq4g256_residual_mmq.hip`
- Existing dp4a pattern: `kernels/src/gemv_hfq4g256.gfx1030.v4.hip:80-99`
- Dispatch gating: `crates/rdna-compute/src/dispatch.rs:96-122`
- Reviews: `gfx906_vec_plan_rev_glm5.md`, `gfx906_mmq_plan_gemini.md`, `gfx906_mmq_plan_claude.md`
- **Phase 0 rollup (read first):** `plans/phase0_complete.md`
- **llama.cpp-gfx906 reference:** `/home/kread/mygit/llama.cpp-gfx906/ggml/src/ggml-cuda/`
  - Topology / nwarps=2: `gfx906/gfx906-config.h:10`, `mmq.cuh:285-298`
  - dp4a builtin wrapper: `common.cuh:666-691`
  - dp4a Q8_1×Q8_1 compute loop: `mmq.cuh:1094-1123`, `vecdotq.cuh:254-278`
  - L2 prefetch: `gfx906/matmul/mmq-prefetch.cuh`
  - LDS `+4` padding: `gfx906/matmul/mmf.cuh:31-32`, `mmq.cuh:189-227`
