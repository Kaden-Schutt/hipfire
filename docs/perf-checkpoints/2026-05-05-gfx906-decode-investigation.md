# gfx906 decode investigation — gap to llama.cpp

Date: 2026-05-05
Hardware: MI50 / gfx906 / HBM2 1024 GB/s peak.
Model: Qwen 3.5 9B (hipfire MQ4 / stock Q4_K_M).
Workload: AR decode at batch=1, 128 tokens, asym3 KV.

## Headline

| Implementation | tg128 tok/s | ms/token | Effective BW |
|---|---:|---:|---:|
| **hipfire (this branch, MQ4)** | **50.7** | 19.7 | ~250 GiB/s |
| Stock llama.cpp (Q4_K_M) | 61.55 | 16.2 | ~305 GiB/s |
| skyne98 fork (Q4_K_M) | 63.48 | 15.7 | ~315 GiB/s |
| Stock 27B Q4_K_M | (not measured) | — | — |
| **hipfire 27B MQ4** | **17.0** | 58.8 | ~238 GiB/s |

We're **3.5 ms/tok behind stock**, **4 ms/tok behind the unverbraucht/skyne98 fork**, **~25% of HBM2 peak**. Decode is HBM-bound but not saturated.

## Decode hot path (rocprof, hipfire 9B AR)

Per-token, 128-token gen, 32-layer 9B:

| Kernel | Calls | Avg | Share |
|---|---:|---:|---:|
| `fused_gate_up_hfq4g256_wave64` | 4096 | 173 µs | **25.5 %** |
| `gemv_hfq4g256_residual_wave64` | 8192 | 78 µs | **23.1 %** |
| `fused_qkvza_hfq4g256_wave64` | 3072 | 88 µs | 9.8 % |
| `fused_qkv_hfq4g256_wave64` | 1024 | 74 µs | 2.7 % |
| `gemv_hfq4g256_wide` (lm_head) | 129 | 1730 µs | 8.0 % |
| `fused_rmsnorm_mq_rotate` | 8256 | 17 µs | 5.2 % |
| `attention_flash_asym3_tile` | 1024 | 116 µs | 4.3 % |
| `sample_top_p` | 129 | 729 µs | 3.4 % |

**61 % of decode is GEMV**: `fused_gate_up + gemv_residual + fused_qkvza + fused_qkv = 61.1 %`.

## PMC — what's limiting the GEMVs

| Kernel | VALUBusy | MemUnitStalled | FetchSize/call | LDSBankConflict |
|---|---:|---:|---:|---:|
| `fused_gate_up_*_wave64` | **41.4 %** | **4.12** | 52 KB | 0.0 |
| `fused_qkv_*_wave64` | 37.2 % | 4.00 | 22 KB | 0.0 |
| `fused_qkvza_*_wave64` | 37.9 % | 4.23 | 26 KB | 0.0 |
| `gemv_residual_*_wave64` | 25.9 % | 0.95 | 18 KB | 0.0 |

VALUBusy 26-41 % means ALU is mostly idle. MemUnitStalled 0.95-4.23 says memory **is** the limiting factor on `fused_gate_up/qkv/qkvza` (>4 % stall is non-trivial), but **not** on `gemv_residual` (<1 % stall — that one is *compute-light* / per-row work too small to amortize launch overhead).

Computed effective BW per kernel:
- `fused_gate_up` 52 KB × 4096 calls / 0.71 s ≈ **300 MB/s effective per kernel**
- That's ~0.03 % of peak HBM2 — clearly memory parallelism, not raw bandwidth, is the bottleneck.

The gap is **memory-level parallelism (MLP)**: not enough in-flight HBM transactions per CU to hide latency. The kernels do already split each wave64 into 2× 32-lane half-waves on different output rows (`row = blockIdx.x * 2 + warp_id`) and interleave 4 quads per inner iter — so the obvious "double the row count per WG" lever is already applied. The remaining gap is something subtler than topology — see the lever audit below.

## What llama.cpp-gfx906 (iacopPBK fork, commit `eec153c`) does differently

Inspecting the fork's `ggml/src/ggml-cuda/gfx906/` directory reveals 29 gfx906-specific kernel files. Key decode-relevant differences:

### 1. `mmvq-q4_0.cuh` / `mmvq-q4_1.cuh` / `mmvq-q8_0.cuh` — warp-cooperative GEMV

```c
__launch_bounds__(64, 1)
__global__ void gfx906_mul_mat_vec_q4_0_warp_coop(...) {
    const int half_lane = lane_id % 32;
    const int row_offset = lane_id / 32;
    const int row = blockIdx.x * 2 + row_offset;
    ...
    for (int ib = half_lane; ib < blocks_per_row; ib += 32) {
        // 8 dp4a per iter, 4 nibble-decoded ints
    }
    sumf = warp_reduce_sum<32>(sumf);  // half-warp reduction
}
```

Two key tricks:
- **2 rows per WG** via half-wave split (`lane_id / 32`). One wave64 computes two output rows simultaneously, **doubling memory parallelism per CU**.
- **No LDS staging at all** — direct HBM → register → dp4a. 64 threads × 2 rows × 1 wave/CU × 60 CUs × launch_bounds(64,1) = lots of waves competing for HBM, fully saturating the bandwidth.
- Reduction via `warp_reduce_sum<32>` keeps the two row-results in different half-waves.

### 2. `mmq-prefetch.cuh` — explicit Y-tile prefetch (software pipelining)

```c
template<int mmq_x, int mmq_tile_y_k, int nwarps, int warp_size>
__device__ __forceinline__ int gfx906_prefetch_y_tile_v4(
    const int * y, const int ncols_y,
    const int kb0, const int kb0_stop, const int qk, const int blocks_per_iter) {
    ...
    // 16 lanes from warp 0 prefetch 16 cache lines (1KB) for next iteration
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(prefetch_data) : "v"(prefetch_addr) : "memory"
    );
    return prefetch_data;
}
```

Issues `global_load_dword` for the *next* iteration's data while current iteration computes. The compiler doesn't reorder these into the critical path because they're inline-asm'd. **L2 cache gets warmed for the next iter's HBM fetches.** This is the prefetch lever the original gfx906 plan considered (see PRD §3.3) — and *they* use it on the prefill MMQ path, not on the GEMV decode path.

### 3. `mmq.cuh` — load-defer via separate cache phase

```c
// LOAD phase: each iter's HBM data → per-thread register cache
GFX906_LOAD_TILES_Q8_0_ASYNC(cache_size, ..., qs0_cache, qs1_cache, ...)

// STORE phase (separate, later): register cache → LDS
GFX906_STORE_TILES_Q8_0_LDS_MMA(cache_size, x_qs, qs0_cache, qs1_cache, ...)
```

The HBM load and the LDS store occupy different instruction issue slots. The compiler pipelines `cache_size` HBM transactions in flight while the LDS unit is idle (during compute), then drains to LDS in a quick burst. This is **load-defer software pipelining** — orthogonal to (1) and (2).

### 4. `gfx906-common.cuh` — utility primitives

- `sgpr_broadcast_*` via `__builtin_amdgcn_readfirstlane` to free up VGPRs by hoisting wave-uniform values to scalar registers.
- `fast_exp_f32`, `fast_log2_f32`, `fast_rcp_f32`, `fast_tanh_f32` via single-instruction `v_exp_f32` / `v_log_f32` / `v_rcp_f32`.
- DPP-based `hip_add_xor*_f32` and `hip_max_xor*_f32` reductions.

These would help any kernel doing softmax, layer-norm reduction, or warp reductions. We have hand-rolled equivalents for some of these.

## Implications for hipfire

### Correction: P1 (2 rows per WG) is already applied

When attempting to prototype a 2-rows-per-WG variant, inspection of the existing kernels showed the half-wave-split topology is **already in place** in all four hot-path decode kernels:

- `gemv_hfq4g256_residual_wave64.hip:7` — *"block=[64,1,1] packs two rows per block (one per warp); grid halves from M to (M + 1) / 2"* — `row = blockIdx.x * 2 + warp_id`.
- `fused_gate_up_hfq4g256_wave64.hip` — same pattern, `gid = blockIdx.x * 2 + warp_id`.
- `fused_qkv_hfq4g256_wave64.hip` — same pattern.
- `fused_qkvza_hfq4g256_wave64.hip` — same pattern.

Furthermore, our GEMV inner loop already does a **4-quad interleave** (`acc0..acc3` over 4 consecutive HFQ4 groups, 4 packed-int loads from 4 different group rows per inner iter). That's strictly *more* MLP per WG than iacopPBK's `mmvq_q4_0_warp_coop` 8-dp4a-per-iter pattern.

So the headline lever the original analysis pointed at is moot. The MemUnitStalled 4 % / VALUBusy 26-41 % numbers were measured *with* this topology already in flight. The gap to llama.cpp must come from somewhere else.

### Remaining levers to investigate

### P1' — Y-tile prefetch on decode (was P2)

Issue `global_load_dword` for the next layer's input activation while the current GEMV computes. Decode is sequential per-token but per-layer there's spatial locality (next layer reads previous layer's output, which we just computed). **Estimated impact: 1.05-1.1× speedup** by warming L2. iacopPBK uses this on prefill MMQ; the same pattern is unused on our decode path.

### P2' — Per-WG occupancy and `__launch_bounds__`

iacopPBK uses `__launch_bounds__(64, 1)` to force one wave per CU at compile time, which is the gfx906 sweet spot for memory-bound GEMVs. Worth checking what `--save-temps`-emitted register pressure / occupancy actually is on our kernels and whether HIP is choosing a different occupancy than is optimal for HBM throughput.

### P3' — `v_dot4_i32_i8` for the mantissa multiply

Our HFQ4 GEMVs decode 4-bit nibbles into FP and run a scalar FMA chain (the `DOG` macro). iacopPBK's mmvq path uses dp4a (`v_dot4_i32_i8`) on packed int8 values: 4 multiply-adds per instruction in 1 issue slot. We sidestepped it because HFQ4 is dequant-then-FMA, not int8-MMA. But we could pre-quantize x to int8 per-block (mirroring Q8_1 activations from MMQ prefill) and run the *decode-time* GEMV through dp4a too. **Estimated impact: 1.3-1.6× on ALU throughput**, useful even on memory-bound kernels because lower issue rate → less time the load unit is contended.

Risk: changes the math — needs FP-equivalence validation against current GEMV.

### P4' — `sgpr_broadcast` / `readfirstlane` for wave-uniform values

iacopPBK's `gfx906-common.cuh` hoists wave-uniform values (group scale, zero-point) to scalar registers via `__builtin_amdgcn_readfirstlane`. Our GEMV reloads `sc0..sc3, zp0..zp3` per lane every quad iteration. If the compiler isn't already promoting these to SGPRs, manual hoisting would free up VGPRs (helping occupancy) and cut redundant register reads. Estimated impact: <5 % per kernel, but cheap to apply.

## Recommended next investigation

Two parallel probes, in this order:

1. **Audit register pressure & occupancy** of the 4 decode kernels via `--save-temps` or rocm-objdump on the cached binaries. We need to know if we're at 1 wave/CU or 2, and whether VGPR pressure is the constraint. This is data-cheap and reframes whether P1' or P2' is the right next lever.

2. **Profile L2 hit rate** on the activation-x reads with `L2CacheHit` PMC counter on a clean run. iacopPBK's prefetch only helps if our L2 hit rate on `x[]` is currently <90 %. If we're already at 95 %+, P1' is small.

If both come back uninformative, the dp4a port (P3') is the single biggest lever — but it's also the most invasive and needs careful FP-equivalence work.

## Probe results (2026-05-05)

### Phase 1: Occupancy audit (kernel-descriptor metadata)

Extracted via `clang-offload-bundler` + `llvm-readelf --notes` from the cached `.hsaco` binaries. See `docs/skills/gfx-kernel-metadata/SKILL.md` for the procedure.

| Kernel | VGPR | SGPR | LDS | Spills | Theoretical waves/SIMD |
|---|---:|---:|---:|---:|---:|
| `gemv_hfq4g256_residual_wave64` | 29 | 14 | 0 | 0 | **8** (cap 32) |
| `fused_gate_up_hfq4g256_wave64` | 29 | 20 | 0 | 0 | **8** |
| `fused_qkv_hfq4g256_wave64` | 46 | 20 | 0 | 0 | **5** (cap 51) |
| `fused_qkvza_hfq4g256_wave64` | 46 | 22 | 0 | 0 | **5** |

Zero spills, zero LDS, well under the VGPR ceiling. **P2' (`__launch_bounds__` tuning) is not worth pursuing** — we already have plenty of in-flight wave headroom; the limiter is HBM-side, not occupancy-side.

### Phase 2: L2 hit rate (PMC `L2CacheHit`)

Single-counter rocprof pass, AR decode of 9B MQ4, 128 generated tokens, 17000+ contexts collected for the 4 hot-path decode kernels:

| Kernel | Calls | L2 Hit % | HBM-miss % |
|---|---:|---:|---:|
| `gemv_hfq4g256_residual_wave64` | 8256 | **39.5** | 60.5 |
| `fused_gate_up_hfq4g256_wave64` | 4128 | **40.2** | 59.8 |
| `fused_qkvza_hfq4g256_wave64` | 3096 | **47.1** | 52.9 |
| `fused_qkv_hfq4g256_wave64` | 1032 | **44.3** | 55.7 |

For comparison: `attention_flash_asym3_tile` hits **78.8 %** (KV-cache reuse working), small per-token utility kernels 56–62 %.

### Interpretation

The hot-path GEMVs run at **~40-47 % L2 hit**. The activation `x[]` (≤18 KB at 9B hidden_dim) easily fits L2 and should be ~100 % hit, so the miss volume is the **weights**: 256-element-per-group HFQ4 streamed once per WG, no reuse across WGs within a token, no reuse across tokens (next token reads next-layer weights).

This validates iacopPBK's prefetch lever as **legitimately applicable**: shifting even half of the 60 % miss rate to L2-hit by issuing `global_load_dword` for the next quad's `pk0..pk3 / sc0..sc3 / zp0..zp3` while the current quad's compute runs would meaningfully reduce HBM-bus stalls. Estimated lift: 1.10-1.15× on the 4 hot-path kernels (memory-bound regime, ~10-15 % of HBM traffic shifted to L2 latency).

### Revised lever ranking

1. **P3' — dp4a port** (1.3-1.6× ALU throughput, also implicitly halves weight fetch volume by quantizing x to int8 — biggest theoretical lift; most invasive)
2. **P1' — across-quad weight prefetch** in the GEMV inner loop (1.10-1.15×; medium effort; data-supported by 40 % L2 hit)
3. **P4' — `readfirstlane` SGPR hoisting** (<5 %; cheap sweep; consider rolling in alongside any other change)
4. ~~P2' — `__launch_bounds__` tuning~~ — ruled out by occupancy audit

## Phase 3: prefetch result (2026-05-05)

Implemented across-quad weight prefetch (P1') as software-pipelined
prologue/steady/epilogue in `gemv_hfq4g256_residual_wave64_prefetch.hip`:

| Kernel | Decode share | Result |
|---|---:|---|
| `gemv_hfq4g256_residual_wave64` | 23.1 % | **+4.8 %** end-to-end (51.9 → 54.4 tok/s, 3-run median) |
| `fused_gate_up_hfq4g256_wave64` | 25.5 % | flat-to-noise (within ±0.3 tok/s, undistinguishable from residual-only) |
| `fused_qkv_hfq4g256_wave64` | 2.7 %  | flat (combined with qkvza, ~0 % additional lift) |
| `fused_qkvza_hfq4g256_wave64` | 9.8 %  | flat (combined with qkv, ~0 % additional lift) |

Direct measurement, holding all else equal:
- residual prefetch only:           54.4 tok/s median
- residual + gate_up prefetch:      54.3 tok/s median
- all four prefetch:                54.1 tok/s median

**Only `gemv_residual` benefits.** The other three kernels show no
additional lift despite identical inner-loop structure and matching
40-47 % L2 hit rates. Plausible reasons:

- `fused_qkv` and `fused_qkvza` cross a wave64 occupancy boundary on
  prefetch: 46 → 53 VGPR moves them from 5 → 4 waves/SIMD (gfx906
  granule table). The 25 % occupancy drop offsets the L2 win.
- `fused_gate_up` stays at 5 waves/SIMD (29 → 46 VGPR, same band as
  residual) but still doesn't measurably improve. Possibly the
  warp-id row routing (warp 0 = gate row k, warp 1 = up row k) gives
  the existing kernel implicit MLP that the prefetch can't add to.
- The PMC L2 hit rates are kernel-aggregate; different rows within
  one launch may have different patterns and the average masks the
  variance.

**Decision: ship only the residual variant** (commit 3ef127d). The
three rolled-out variants were reverted because they ship +17 VGPR of
dead weight without lift. The dp4a port (P3') is the next single
biggest lever — but see the design sketch below for why we deferred
it and pivoted to P4'.

## Phase 4: dp4a port — design sketch (deferred)

Sketched the port before starting it; the cost/value math came out
ambiguous enough to defer rather than plunge in. Captured here so a
future revisit doesn't redo the analysis.

### Math (proven; lifted from existing MMQ kernel)

The gfx906 MMQ body (`gemm_hfq4g256_residual_mmq_gfx906_body.cuh`) already
runs HFQ4 weights × Q8_1 activations with dp4a (`v_dot4_i32_i8`). Per
32-K sub-block:

```
sumi   = 0   // int32, dp4a accumulator
for v in 0..8:
    sumi = __builtin_amdgcn_sdot4(x_int8_packed[v], w_nibbles_as_int8[v], sumi, false)
sum   += scale_w * d_x * (float)sumi + zp_eff * sum_x
```

Where `scale_w` = HFQ4 group scale, `zp_eff = zp + 8*scale_w` (compensates
for nibbles stored as `(n - 8) & 0xFF` in the int8 lane), `d_x` = Q8_1
quantize scale per 32-block, `sum_x` = sum of the 32 ungquantized x
values in that sub-block (Q8_1 stores both as `half2`).

For batch=1 GEMV: drop the LDS staging that MMQ uses for the y-tile,
keep the streaming x-tile pattern. Each warp = 1 output row; 4-quad
interleave equivalent maps onto the dp4a path as 8 sub-blocks per
HFQ4-G256 group (256 K-elements / 32 sub-block = 8).

### Implementation cost

1. New kernel `gemv_hfq4g256_residual_dp4a_wave64.hip` (~150 LoC).
2. Rust dispatch:
   - Pre-call: invoke `quantize_q8_1_mmq_ds4` on the float `x_rot_alias`
     input with N=1 (single token), output to a per-process staging
     buffer of size `K/128 * 144` bytes. Quantize kernel already
     exists in `gemm_hfq4g256_residual_mmq.hip:46`.
   - GEMV call: pass quantized buffer as `block_q8_1_mmq*` instead of
     `float*`.
3. Correctness: build a reference test comparing dp4a output to
   FP path on random inputs. Q8_1 introduces ~1 % per-element error.
   Coherence gate must still pass.

### Why we deferred

The predicted +1.3-1.6× lift comes from iacopPBK's `mmvq` kernels,
which are ALU-bound. **Our kernels are not ALU-bound:**

- VALUBusy 26-41 %, sub-1 % MemUnitStalled on `gemv_residual` —
  the kernel sits in the small slice between memory-saturation and
  ALU-saturation where neither is the dominant limiter.
- The prefetch experiment confirmed there's *some* memory headroom
  to absorb (residual responded +4.8 %, gate_up didn't). Adding dp4a
  trades ALU instruction count for memory bandwidth (Q8_1 x fits in
  ~4.6 KB vs 18 KB float — 75 % reduction in x-traffic).
- BUT weights are ~90 % of fetched bytes per the L2 analysis, so the
  *total* HBM-side savings from cheaper x are ~10 %, not 75 %.

Net: ~0-5 % expected lift on our system, against ~1 work-session of
cost (kernel + dispatch wiring + reference correctness test +
quantizer overhead bench). The asymmetry favored pivoting to P4'
(SGPR hoisting, smaller probe cost) and revisiting dp4a only if a
future PMC pass shows a clearer ALU-bound regime.

### Conditions under which dp4a becomes attractive again

- If we ship something that pushes VALUBusy above ~70 %, then dp4a's
  ALU-throughput lift translates to wall clock and the port pays off.
  Candidate triggers: another round of weight-prefetch tuning that
  shifts the bottleneck back to ALU; or a future kernel-fusion pass.
- DFlash-decode at batch≥4 already crosses into ALU-busier territory
  per the MMQ data (we use MMQ at batch≥16 on gfx906). If we widen
  the spec-decode batch sweet spot, dp4a-GEMV at smaller batches
  becomes the natural bridge between batch=1 FP and batch=16 MMQ.

## Phase 5: P4' SGPR hoisting — ruled out (structural)

Tried P4' (readfirstlane / readlane SGPR hoisting of the wave-uniform
sc/zp values per HFQ4 group) as a cheap probe. Two variants:

1. `if (lane == 0) load; __shfl(val, 0, 32)` — width-32 intra-warp
   broadcast.
2. Plain global load + `__builtin_amdgcn_readlane(val, warp_id<<5)`
   — keep the divergent load, hint that the value collapses per-warp.

**Both variants showed 0 % net (within noise across 3 runs).**
Expected <5 %; got 0.

Disassembly diagnostic explained why:

| Variant | VGPR | SGPR | global_load | extra |
|---|---:|---:|---:|---|
| Prefetch baseline | 46 | 14 | 37 | — |
| Prefetch + P4 (`__shfl`) | 53 | 16 | 41 | 27× `ds_bpermute` (LDS round-trip per shuffle) |
| Prefetch + P4 (`readlane`) | 47 | 31 | 41 | 22× `v_readlane` |

The `__shfl` variant lowered to `ds_bpermute_b32`, an LDS-permute
that adds ~4-cycle latency per call and contends with LDS bandwidth.
That's *worse* than the original divergent load. The `readlane`
variant correctly promoted values to SGPRs (14 → 31) but global
loads still went *up* by 4 — the compiler couldn't fuse adjacent
sc/zp dwords into a single `dwordx2` because their post-readlane
destinations diverge.

**Structural reason:** the lever assumed wave = warp. On wave64
with 2-rows-per-WG (warp 0 = row k, warp 1 = row k+1), each warp
sees a *different* sc/zp address — so the wave-uniformity needed
for `s_load_dword` doesn't hold. The compiler's vector-to-scalar
pass (`SIWholeQuadMode` / `AMDGPUOptimizeUniformIntrinsics`) only
fires when the *whole wave* is uniform, not the per-warp half.

To make readfirstlane help here we'd need to drop back to wave32
or restructure to 1 row per WG (giving up the half-wave-split MLP
gain). Neither is appealing. Cleaner fix would be a backend pass
that recognizes per-warp uniformity, but that's an LLVM project,
not a hipfire one.

**Decision: P4' ruled out for our 2-rows-per-WG topology.** The
lever stays on the table for any future kernel that runs 1 row
per WG (e.g., the wide-LM-head GEMV) but is moot for the four
hot-path decode GEMVs.

## Phase 6: per-kernel PMC pass — what the prefetch win actually was

Three-counter rocprof (MemUnitStalled / VALUBusy / FetchSize) on
9B AR decode, baseline vs prefetch, aggregated per kernel:

### Baseline vs prefetch

| Kernel | Δ MemUnitStalled | Δ VALUBusy | Δ FetchSize |
|---|---:|---:|---:|
| `gemv_residual` | +0.90 pp (1.0 → 1.9) | **+7.7 pp** (25.6 → **33.3**) | ≈0 |
| `fused_gate_up`  | -0.11 (3.86 → 3.75) | flat | flat |
| `fused_qkv`      | -0.22 (3.84 → 3.62) | flat | flat |
| `fused_qkvza`    | -0.13 (4.67 → 4.54) | flat | flat |

(Last three didn't get prefetch in the shipped state — their
numbers are baseline-vs-baseline, just confirming run-to-run
stability.)

### The reframe

I expected prefetch to shift weight fetches from HBM to L2-hit
(reducing FetchSize) — that was the L2CacheHit-driven hypothesis.
**FetchSize is unchanged.** The kernel does the same amount of
HBM-side reading.

What actually changed: **VALUBusy on `gemv_residual` jumped from
25.6 % to 33.3 % — a 30 % relative increase in ALU utilization**.
That's where the +4.8 % wall-clock came from.

**The bottleneck in `gemv_residual` was instruction-issue
serialization, not L2 hits.** Both VALU (25.6 %) and MemUnit
(0.95 %) were under-utilized — the front end was waiting on
in-flight loads to retire before issuing the next FMA chain. By
issuing the next-quad's loads *before* the current quad's
compute, prefetch gave the scheduler more independent work to
keep both units busier per cycle.

Prefetch is misnamed in our codebase. It's really
**instruction-level parallelism injection**.

### Why the other three didn't benefit

The three sister kernels sit at 37-41 % VALUBusy and 3.8-4.7 %
MemUnitStalled — closer to the memory boundary. Issuing loads
earlier doesn't help when the memory unit is the bound: the
loads still have to wait their turn. Their bottleneck is the
load unit's queue depth / HBM round-trip latency, which prefetch
can't shorten — only L2-hit-rate or fetch-volume reduction can.

### Real lever ranking (post-PMC)

| Lever | Bottleneck it addresses | Estimated lift |
|---|---|---:|
| ILP injection (prefetch / sw pipeline) | inst-issue serialization | already shipped on residual |
| HBM volume reduction (dp4a, weight repacking) | mem-stall | for gate_up/qkv/qkvza only |
| Larger weight fetch granule (wider load chunks) | mem-issue rate | open |
| MoE indexed kernels (already wave64) | n/a — different shape | — |

The dp4a port might actually pay off **on the three kernels
that didn't benefit from prefetch** (where mem-stall is the
real limiter), but we ruled it out earlier on a generic-cost
basis. Reconsider: dp4a's HBM-side savings (75 % x-traffic
reduction even if x is 10 % of fetch volume = ~7-8 % FetchSize
reduction) lands directly on the bottleneck for those three
kernels. **Estimated lift specifically on gate_up: ~5-8 % per
call**; ~+1.5-2 % end-to-end given 25.5 % decode share.

Updated dp4a verdict: **worth doing on the three sister kernels
even though it's flat for `gemv_residual`.** That changes the
cost/value balance — the same quantize_q8_1_mmq_ds4 invocation
serves all three call sites.

## Reproducing

```sh
# AR decode bench
hipfire bench qwen3.5:9b --runs 3

# rocprof kernel-trace
rocprofv3 --kernel-trace --stats -d ./run -o trace --output-format csv -- \
    target/release/examples/daemon < decode_input.jl

# PMC (one counter per pass on gfx906)
for ctr in VALUBusy MemUnitStalled FetchSize VALUUtilization L2CacheHit; do
    printf 'pmc: %s\ngpu: 0\n' "$ctr" > pmc.txt
    rocprof -i pmc.txt -o "run_${ctr}.csv" \
        target/release/examples/daemon < decode_input.jl
done

# L2CacheHit pass on bench_qwen35_mq4 (validated working command)
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
rocprof -i pmc_l2.txt -o decode_l2.csv \
    target/release/examples/bench_qwen35_mq4 \
    $HOME/.hipfire/models/qwen3.5-9b.mq4 \
    --prefill 16 --warmup 1 --gen 128

# Occupancy / register-pressure audit — see
# docs/skills/gfx-kernel-metadata/SKILL.md for the full procedure.

# Stock llama.cpp comparison (constrain to MI50, the iGPU breaks rocprofv3)
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
    /tmp/llama-stock/build/bin/llama-bench \
    -m /data/models/.../Qwen3.5-9B-Q4_K_M.gguf \
    -ngl 99 -p 0 -n 128 -r 3 -ctk q8_0 -ctv q8_0 -fa 1
```

## References

- iacopPBK fork: https://github.com/iacopPBK/llama.cpp-gfx906
- The commit with all gfx906 optimizations: `eec153c086df6a9e7a69499bea3639597c085fff` ("2602.01 version")
- Skyne98 / unverbraucht fork (in our local checkout): `~/mygit/llama.cpp-gfx906`
- Our hipfire decode kernels:
  - `kernels/src/gemv_hfq4g256_residual_wave64.hip`
  - `kernels/src/fused_gate_up_hfq4g256_wave64.hip`
  - `kernels/src/fused_qkv_hfq4g256_wave64.hip`
  - `kernels/src/fused_qkvza_hfq4g256_wave64.hip`
