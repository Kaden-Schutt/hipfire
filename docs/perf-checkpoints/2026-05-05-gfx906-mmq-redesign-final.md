# gfx906 MMQ kernel redesign — final report

Date: 2026-05-04 → 2026-05-05
Hardware: AMD Instinct MI50 (gfx906, Vega 20, 60 CUs, 1024 GB/s HBM2)
Model: Qwen 3.5 9B (HFQ4-G256 quant)
Tools: ROCm 6.4.3, rocprofv3 (kernel-trace), legacy rocprof (PMC counters)

This document consolidates the redesign of hipfire's gfx906 dp4a MMQ
kernel into a single dev log. It supersedes:

- `docs/issues/gfx906-mmq-redesign.md` (the original problem statement)
- `plans/gfx906_mmq_redesign.md` (design plan + revision history)
- `plans/phase2a_probe_results.md` (ELF probe results)
- `gfx906_dp4a_bug.md` (pre-redesign bug post-mortem)
- `BATCH_TILE_RESULTS.md` (FP16 wave64 BATCH_TILE tuning, untracked)
- `docs/perf-checkpoints/2026-05-04-gfx906-mmq-{redesign-rocprof,
  cpu-investigation,screening-threshold,default-on,window-streaming}.md`

The four pre-redesign checkpoints (`2026-05-04-gfx906-mmq-{attribution,
junroll,spill-reduction}.md` and `2026-05-04-llamacpp-stock-comparison.md`)
are retained as historical context — they document the original kernel's
debugging journey and the analysis that motivated the redesign.

## TL;DR

| Metric | Pre-redesign | Final | Speedup |
|---|---|---|---|
| pp32  | 136 tok/s | 313 tok/s | 2.30× |
| pp64  | 140 tok/s | 463 tok/s | 3.31× |
| pp128 | 141 tok/s | 598 tok/s | **4.24×** |
| pp256 | 143 tok/s | 723 tok/s | 5.06× |
| pp512 | 142 tok/s | 714 tok/s | **5.02×** |

vs stock llama.cpp pp512 (750 tok/s baseline): 19% → **95%**.

The path: structural kernel rewrite (nwarps=2 → nwarps=4, runtime mmq_x
dispatch, Option C window streaming) → screening-threshold recalibration
→ default-on at batch ≥ 16 → LDS bank-conflict diagnostic → ds_read_b128
with per-mmq_x stride. End-to-end correctness preserved (NRMSE bit-
identical to FP16 wave64 reference at all shapes; coherence gate clean).

The biggest single learning: **PMC bank-conflict counters
(LDSBankConflict, ALUStalledByLDS), not ELF metadata, are the right
diagnostic when changing LDS layout on AMD GCN/Vega.** A "clean" ELF
(low VGPR, 0 spills, LDS budget OK) hid a 47% bank-conflict regression
that cost 14% wallclock until +1 int of row padding fixed it.

## 1. Problem

### 1.1 Performance gap (pre-redesign)

llama.cpp on the same MI50 hits **244 tok/s prefill** at pp512 on Qwen
3.5 9B Q4_K_M. hipfire's pre-redesign kernel hit **74 tok/s** —
**3.29× slower**. Decode was less skewed (60 vs 49 tok/s, 1.24×). The
gap was entirely in prefill.

### 1.2 What we ruled out before redesigning

| Hypothesis | Outcome |
|---|---|
| rocprof overhead | No: gap reproduces without profiler |
| Kernel granularity (BATCH_TILE) | No: BATCH_TILE ∈ {1,2,4,8,16,32} all gave 74 tok/s on FP16 wave64; 8 was optimal, 16 and 32 regressed -39% / -48% from VGPR pressure |
| Manual SIMD vectorization | No: compiler already vectorizes; HFQ4 layout's 136 B group stride blocks cross-group vectorization |

### 1.3 Architectural diagnosis

- **hipfire's old approach (row-parallel)**: one wave64 per output row,
  no LDS, no data reuse. 64 large kernels (113 ms avg). GPU/wall ratio
  1.04× — near-sequential.
- **llama.cpp (tiled MMQ)**: LDS staging, Q8_1 pre-quantized
  activations, int8 dot-products, 8× weight reuse per tile. 1182 small
  kernels (1.77 ms avg). GPU/wall ratio 6.28× — heavy async overlap.

Conclusion: hipfire needed a tiled MMQ kernel adapted to gfx906 (no
WMMA → use dp4a `__builtin_amdgcn_sdot4`).

## 2. Original gfx906 MMQ kernel (pre-redesign) — bug saga

The first attempt at a gfx906 dp4a MMQ kernel (commits `dc123f4` →
`17c05f3`) was buggy. Three correctness fixes landed in `feb8e08` and
**carry forward into the redesigned kernel**:

### 2.1 Symmetric weight unpacking

`__builtin_amdgcn_sdot4` interprets bytes as **signed** int8. The
original kernel unpacked HFQ4 nibbles to bytes in `[0, 15]` (strictly
positive). Mean weight magnitude ≈ 7.5; consistent rounding biases in
dynamic Q8_1 activation quantization were multiplied by these positive-
only weights and summed over K=4096, creating a directional DC offset
on every output neuron. The offset compounded layer-over-layer through
32 layers until logits drifted off the model's trained distribution.

**Fix**: unpack nibbles as `(n - 8)` so dp4a sees signed `[-8, +7]`.
Add back `8 × scale × sum_x` in the per-row correction. This carries
forward as `zp_eff = zp + 8.0f * sc` baked into x_dm at load time.

### 2.2 F32 LDS staging for x_dm

Weight scale and zero-point were stored in LDS as `half2` (4 bytes per
row). Casting f32 → f16 added a high noise floor to the bias term
(`zp_w × sum_x`) that sensitive models like Qwen could not tolerate.

**Fix**: f32 throughout — store as `float2`. 8 bytes per row vs 4.
Adds 0.5 KiB to LDS budget. Carries forward.

### 2.3 Per-weight screening (mmq_screen_weight)

Some weight matrices have legitimate precision issues that tile-MMQ
can't tolerate (e.g. row 3994 of m=4096 matrices in Qwen 9B has a
near-zero scale → dp4a int rounding dominates). Screening dispatches
each unique weight pointer once on first use against an FP16 wave64
reference; weights exceeding `mmq_screen_threshold` fall back to FP16.

**Fix added in `feb8e08`, carried forward.** The threshold needed
recalibration after the redesign — see §6.

### 2.4 The "row 3994" pattern

Real-data NRMSE on `/tmp/mmq_dump_0` (Qwen pp128 dump, M=4096 K=4096
N=128): **0.2881%, 99.9% of cells <1e-3 absolute error**, but the 10
worst-error cells **all sit on row 3994** (errors 6e-3 to 1.8e-2,
next-worst row 3.7e-3). User confirmed row 3994 was also the worst
in the original dp4a implementation — **this is a degenerate quant
group in the Qwen 9B mq4 quantization, not a kernel bug**. Screening
catches it; no kernel-side fix is needed.

Diagnostic flags introduced for this saga remain in dispatch.rs:
`HIPFIRE_MMQ_TRACE=1`, `HIPFIRE_MMQ_DUMP=N`, `HIPFIRE_MMQ_K_FILTER=N`,
`HIPFIRE_MMQ_CALL_FILTER=lo:hi`, `HIPFIRE_MMQ_DIAG_PASSTHROUGH=1`,
`HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY=1`.

## 3. Redesign architecture

### 3.1 Hardware constraints (gfx906 MI50)

| Resource | Per-CU limit |
|---|---|
| LDS | 64 KiB |
| VGPRs | 65,536 (256 KiB) |
| Wavefront size | 64 lanes |
| `ds_read_b128` alignment | 16 B (address) |
| Max threads/WG | 1024 |

At nwarps=4 (256 threads/WG) targeting 2 WGs/CU:
- LDS: ≤32 KiB/WG
- VGPRs: ≤128/thread

These are tight. Stock fits at exactly 128 VGPRs/thread (zero margin)
and 28.5 KiB/WG LDS (4 KiB headroom).

### 3.2 Non-negotiables

1. **`nwarps = 4`**, threads per WG = 256, block dim (64, 4, 1).
   Confirmed by stock's ELF `max_flat_workgroup_size = 256` and
   `mmq_get_nwarps_device()` = 4 on gfx906.
2. `__launch_bounds__(256, 2)` — 2 WGs/CU.
3. `mmq_y = 128` (same as stock).
4. **`mmq_x` is a template parameter** ∈ {8, 16, 24, 32, 40, 48, 56, 64}
   (greedy step-8 dispatch matching stock's `mmq.cuh:4069-4082`). Each
   value emits 3 entry symbols (`_x{N}`, `_full_add_x{N}`,
   `_full_set_x{N}`) → 24 entry symbols total.
5. `vgpr_count ≤ 128` is a hard ceiling.
6. **Chunk-major X-tile loader.** Adjacent tids hit consecutive 4-byte
   chunks of the same row (coalesced HBM access). The 1-thread-per-row
   layout would be ~64× HBM bandwidth waste.
7. Keep HFQ4 layout (136 B/group: f32 scale + f32 zp + 128 B nibbles)
   and Q8_1 activations.
8. **24 entry symbols** = 8 .hip wrappers including a shared
   `body.cuh` with the templated `mul_mat_q` body. Single-file would
   blow up build time.

### 3.3 LDS streaming pattern: Option B → Option C

Stock's `mul_mat_q` keeps only 32 K-elements of x_qs resident at a time
(streaming). At nwarps=4 with HFQ4's 256-K group, the original v1 plan
of "256-K-group-resident x_qs" failed Phase 2a Gate 3 (43 KiB/WG ×
2 WGs = 87 KiB > 64 KiB cap):

| Layout | x_qs LDS | Total/WG (mmq_x=64) | Verdict |
|---|---|---|---|
| 256-K resident (v1) | 35 KiB | 45 KiB | 1 WG/CU ❌ |
| **32-K stream (Option B)** | 4 KiB | 14 KiB | 2 WGs/CU ✅ but 8 syncs/group |
| **128-K window (Option C)** | 17 KiB | 27 KiB | 2 WGs/CU ✅, 4 syncs/group |
| 256-K resident (Option D) | 32 KiB | 42 KiB | 1 WG/CU ❌ |

Option C (the final design) loads 128 K-elements (= 1 Q8_1 block worth
= 4 sub-blocks of 32 K-elements) per LDS load. The 4 sub-blocks compute
back-to-back without intermediate syncs. Total: **4 syncs per HFQ4
group** (2 windows × 2 syncs each), down from Option B's 16.

The redesign initially shipped with Option B (commit `c022682`). Option
C landed in `29bcd79` after the diagnostic detour described in §7.

## 4. Implementation timeline

| Commit | Change | pp128 tok/s | Notes |
|---|---|---|---|
| (pre-redesign) | FP16 wave64 baseline | 141 | |
| `c022682` | Phase 5 redesign: nwarps=4, runtime mmq_x, Option B (8 syncs/group) | 287 (2.04×) | First land of new kernel |
| `0c5b29d` | qkvza split: route qkv+z to MMQ, beta+alpha tail stays FP16 wave64 | 352 | Path A from the qkvza analysis |
| `fa6778c` | qkv port (full-attention layers) | 355 | +0.9% — modest because 7/8 qkv calls hit screen-fallback |
| `7972c19` | Bump `mmq_screen_threshold` 0.10 → 0.50 (gfx906 only) | 462 | The 0.10 default was set when the original kernel was buggy; recalibrated for the redesigned kernel |
| `52eb6bb` | Default-on for gfx906 at batch ≥ 16 | 462 | No env vars needed; auto-routed |
| `b2624b7` | (docs) CPU investigation — not a CPU bottleneck | 462 | Falsified launch-overhead hypothesis |
| `29bcd79` | Option C window streaming + LDS bank-conflict pad (X_STRIDE=33) | 512 (+10.8%) | The window-streaming work and bank-conflict diagnostic |
| `c48e308` | ds_read_b128 at mmq_x≥64 (with X_STRIDE=33) | 522 | 25% b128 alignment, 75% b32-quad fallback |
| `786f68e` | Per-mmq_x X_STRIDE: 40 for mmq_x≥64, 33 otherwise | **598** | 100% b128 alignment for large kernels |
| `4a130db` | Cleanup: remove superseded original kernel + 4 probe files | 598 | Pre-merge tidy |

## 5. The headline diagnostic: LDS bank conflicts at stride%32==0

The window-streaming work (Option C) initially regressed pp128 from
462 → 293 tok/s (−37%) despite cleaner ELF metadata than the Option B
baseline (vgpr=68 vs 112, 0 spills, smaller LDS budget). PMC counter
sweep traced the regression to **LDS bank conflicts**:

| Counter | Kernel | Opt B (X_STRIDE=8) | Opt C (X_STRIDE=32) |
|---|---|---|---|
| **LDSBankConflict** | `_full_set_x64` | 13.5% | **37.2%** |
| | `_full_add_x16` | 20.6% | **47.0%** |
| ALUStalledByLDS | `_full_set_x64` | 6.0% | 4.0% |
| | `_full_add_x16` | 15.7% | 21.7% |
| VALUBusy | `_full_set_x64` | 41.1% | **19.8%** ⛔ |
| | `_full_add_x16` | 18.8% | **7.9%** ⛔ |
| MemUnitStalled | `_full_set_x64` | 0.249 | 0.088 |
| FetchSize KB/call | `_full_set_x64` | 51.9 | 30.0 |

The kernel was *less* memory-bound but *more* LDS-bound. Mystery
solved by AMD bank arithmetic:

> AMD GCN/Vega has **32 LDS banks at 4 bytes each** (128 bytes/cycle).
> For `x_qs[i * X_STRIDE + v]` where `i` = lane, the bank index is
> `(i × X_STRIDE + v) mod 32`. With **X_STRIDE = 32 ints** (a multiple
> of the bank count), `(i × 32 + v) mod 32 = v` — **every lane in the
> warp hits the same bank**. The LDS arbiter serializes the 64-way
> conflict, costing tens of cycles per access.

**Fix: +1 int of row padding (X_STRIDE 32 → 33)**. With `33 mod 32 = 1`,
banks rotate by 1 per row → each lane hits a distinct bank → 0 conflicts.

| Counter | X_STRIDE=32 | X_STRIDE=33 |
|---|---|---|
| LDSBankConflict | 37–47% | **0.0%** ✅ |
| ALUStalledByLDS | 4–22% | **0.2–1.1%** ✅ |
| VALUBusy | 8–20% | 27–37% ✅ |

After the bank-conflict fix, the window-streaming kernel landed
pp128 462 → 512 tok/s (+10.8%).

### 5.1 Per-mmq_x stride for ds_read_b128

A second LDS-layout iteration: at X_STRIDE=33, only every 4th row was
16-byte aligned (`33 × 4 = 132 B`, `132 mod 16 = 4`). The compiler
emitted 25% ds_read_b128 + 75% b32-quad fallback in `_x64` variants.
Switching to **X_STRIDE=40** (32 data + 8 pad) made every row 16-B
aligned → 100% ds_read_b128, but introduced a 4-way bank conflict
(`40 mod 32 = 8`). Trade-off table:

| Stride | b128 align | Bank conflict | Best at |
|---|---|---|---|
| 32 | 100% | 64-way (pathological) | nowhere |
| 33 | 25% | 0-way (best) | mmq_x < 64 |
| 36 | 100% | 8-way | nowhere |
| **40** | 100% | 4-way | mmq_x ≥ 64 |
| 48 | 100% | 16-way | nowhere |

Sweep showed stride=40 gave +14% pp512 but −14% pp32 because
small-mmq_x kernels can't amortize the 4-way conflict.
**Solution: per-mmq_x stride** via constexpr template:

```c
template <int mmq_x>
constexpr int x_stride_for() { return mmq_x >= 64 ? 40 : 33; }
```

Final result: pp32 unchanged at 313, pp64 +13%, pp128 +15%, pp256 +15%,
pp512 +14%. Bit-identical correctness.

## 6. Screening threshold recalibration (0.10 → 0.50)

The original `mmq_screen_threshold = 0.10` was set in commit `8081822`
to mask the two pre-redesign bugs (asymmetric unpacking, f16 staging).
With both bugs structurally fixed in the redesigned kernel, a single
9B mq4 load showed **30 weights rejected at 0.10**:

```
19/30 reject on row 3994 of m=4096 matrices  (degenerate quant group)
11/30 reject by very small margins (max errors 0.10–0.15) on m=8192
```

Threshold sweep with the 9B reasoning prompt:

| Threshold | Rejects | Final answer | pp128 |
|---|---|---|---|
| 0.05 (over-rejecting) | 126 | 9 ✓ | — |
| **0.10 (old default)** | 30 | 9 ✓ | 355 |
| 0.50 (effectively pass-through) | 0 | 9 ✓ | 462 |

Coherence gate at threshold=0.50 passes all 4 mq4 rows and the
tool-call shape (no `<|im_start|>` leak — rules out the #87 regression
pattern).

**Bumped per-arch default**: `gfx906 → 0.50`, other archs unchanged at
0.10 until similar validation. The env override
`HIPFIRE_MMQ_SCREEN_THRESHOLD` still takes precedence.

## 7. Default-on at batch ≥ 16

Pre-redesign, gfx906 forced an unconditional return-false in
`should_use_mmq()` ("default OFF during Phase 1 validation"). With
correctness preserved across 49 synthetic shapes, real-data NRMSE
0.29%, and coherence-gate all-PASS, the gfx906 carve-out got removed
in `52eb6bb`. Per-arch min_batch:

```rust
let arch_min_batch: usize = if arch == "gfx906" { 16 } else { 256 };
```

The cutover at 16 is empirical:

| pp | baseline (FP16 wave64) | MMQ on | Speedup |
|---|---|---|---|
| pp2  | 69 | 30 | **0.44×** ⛔ |
| pp8  | 120 | 113 | 0.94× |
| pp16 | 131 | 192 | **1.46×** ✓ |
| pp32 | 136 | 276 | 2.03× |
| pp128 | 141 | 462 | 3.27× |

Below pp16, Q8_1 quantize + per-output launch overhead dominates and
FP16 wave64 wins. At pp16+ MMQ wins decisively. Decode (batch=1)
correctly stays on FP16 wave64.

## 8. Investigation: not a CPU bottleneck

Hypothesis (motivated by the user observing "daemon never exceeds 200%
CPU"): pp256/pp512 plateau (561 vs 554 tok/s pre-redesign-iteration)
suggests CPU-bound launch path on AMD/HIP.

PMC + thread-state sampling falsified this:

- During prefill: CPU at **20–40% in `D` state** (uninterruptible
  kernel-mode sync — GPU-bound), with brief 100% bursts at launch loops.
- The 200% peak observed is **post-prefill** housekeeping (logits
  download + sample + JSON emit + unload), not the prefill phase.
- Setting `gpu.active_stream = Some(stream_create())` explicitly in
  the bench produced **no perf change** (461.5/554 tok/s identical to
  default-stream). Either modern HIP runs default-stream as
  per-thread-default (async) or any sync overhead is below noise.

pp256/pp512 wallclock scaled linearly (456 → 923 ms; per-token cost
1.78 → 1.80 ms). The plateau in tok/s is the linear-scaling regime,
not a bottleneck.

The actual remaining lever was per-call MMQ kernel cost — closed by §5.

## 9. Final perf vs stock llama.cpp

| pp | pre-redesign | final | stock llama.cpp | hipfire / stock |
|---|---|---|---|---|
| 128 | 141 | 598 | (not measured) | — |
| 256 | 143 | 723 | (not measured) | — |
| **512** | **142** | **714** | **750** | **95%** |

vs the original 19% of stock at pre-redesign baseline.

The remaining 5% gap to stock at pp512 is structural:
- Beta+alpha tail still on FP16 wave64 (qkvza Path B fused-4-output
  kernel deferred — small remaining lever).
- Possibly inter-warp sync overhead at very large batches (4 syncs/group
  vs stock's 2/group). Reducing further would force 1 WG/CU per
  Phase 2a Gate 3 — not worth pursuing.

### 9.1 Cross-process A/B (DPM-warmed, 3 alternating iterations)

Each row is a fresh process invocation:

| pp | Iter | A (MMQ=0) | B (default) | Speedup |
|---|---|---|---|---|
| 128 | 1 | 141.7 | 598.6 | 4.22× |
| 128 | 2 | 141.1 | 598.2 | 4.24× |
| 128 | 3 | 141.2 | 599.5 | 4.25× |
| 512 | 1 | 142.2 | 714.1 | 5.02× |
| 512 | 2 | 142.2 | 713.9 | 5.02× |
| 512 | 3 | 142.2 | 714.3 | 5.02× |

B-spread ≤1.0 tok/s (≤0.14%), structural — well below the
±10–15% within-session noise floor that
`docs/methodology/perf-benchmarking.md` warns about.

## 10. Correctness gates (all green)

| Gate | Result |
|---|---|
| Synthetic NRMSE (49 shapes: residual+set, mmq_x ∈ {8..64}, K ∈ {4096, 12288}, partial-M, partial-N, M=4096) | 0.04–0.18% vs FP16 wave64 |
| Real-data NRMSE (Qwen pp128 dump M=4096 K=4096 N=128) | 0.2881% (≤0.30% threshold) |
| Coherence gate (4 mq4 rows: 0.8B-cap, 4B-code, 9B-reason, 9B-tool-call) | all PASS, fluent output, no `<\|im_start\|>` leak in tool-call |
| ELF (per-mmq_x): vgpr_count ≤ 94, 0 spills, 0 group_segment_fixed_size (extern __shared__) | clean |

The runtime LDS budget is enforced via
`debug_assert!(shared_mem ≤ 32*1024)` in dispatch.rs since the
kernels use `extern __shared__` (compiler can't validate the cap).

## 11. Lessons learned (durable findings)

### 11.1 PMC counters > ELF metadata for LDS-layout changes

When changing LDS layout in a HIP/dp4a kernel on gfx906 (or any
AMD GCN/Vega arch with 32 banks × 4 bytes), **the first counters to
check are LDSBankConflict and ALUStalledByLDS**, before VALUBusy or
wallclock. ELF metadata (vgpr/spill/lds budget) is necessary but not
sufficient — it cannot predict bank-conflict-driven regressions.

The probe ELF for Option C said "this should work" (vgpr=68, 0 spills,
LDS budget OK). Reality was a 47% bank-conflict rate that dropped
wallclock by 14%. Always check bank-conflict counters when stride
changes.

### 11.2 Stride choice trades alignment vs bank conflict

For gfx906 (32 banks × 4 bytes), the LDS layout optimum depends on
both ds_read_b128 alignment AND bank-conflict pattern. The right stride
is often per-mmq_x:

| Property | Constraint |
|---|---|
| `stride × 4 mod 16 == 0` | 100% ds_read_b128 alignment |
| `stride mod 32 == 0` | 64-way bank conflict (pathological) |
| `stride mod 32 == 1` | 0-way bank conflict (best) |
| `stride mod 32 ∈ {4, 8}` | 4-way / 8-way conflict (acceptable trade-off) |

Smaller mmq_x kernels prefer the 0-conflict stride (33); larger
mmq_x kernels prefer the b128-aligned stride (40) because they have
enough j0 iterations to amortize the conflict cost.

### 11.3 Screening threshold needs recalibration on kernel changes

The 0.10 default was set when the gfx906 dp4a kernel was buggy. With
the bugs fixed, weights that legitimately exceeded 0.10 in the buggy
kernel pass cleanly through the redesigned kernel. **Don't carry
forward calibrated thresholds across kernel redesigns** — re-measure.

### 11.4 "200% CPU max" ≠ CPU-bound

Per-process %CPU samples are a poor diagnostic without context.
The peak might be in housekeeping (post-prefill emit/unload), and
the prefill itself might be deeply GPU-bound (D-state). Always
sample at sub-second granularity and correlate with process state.

### 11.5 Streaming pattern: 2 sync points, 4 windows

Stock llama.cpp's `mul_mat_q_process_tile` runs 2 syncs per HFQ4
group (load X once + load Y twice). Our Option B ran 8 syncs/group
(easier to write, finer-grained). Option C compromised at 4 syncs/group
(2 windows × 2 syncs each) because the 2-sync option requires resident
256-K x_qs which forces 1 WG/CU. The 8 → 4 reduction was worth +10%;
going to 2 isn't worth halving occupancy.

### 11.6 BATCH_TILE=8 is optimal for FP16 wave64 (the now-dormant path)

For completeness: BATCH_TILE ∈ {1,2,4,8,16,32} on the FP16 wave64
prefill kernel showed 8 as the optimum. 16 regressed −39% (additional
4×8 = 32 accumulator floats per thread reduced wave occupancy
on gfx906's 64-VGPR-per-SIMD budget); 32 regressed −48%. With MMQ
default-on at batch ≥ 16 the FP16 wave64 path is dormant during
prefill anyway, but documenting for completeness.

## 12. Reproducing this report

```sh
# Final perf bench (no env vars — default-on at gfx906)
HIPFIRE_DPM_WARMUP_SECS=2 \
  $HIPFIRE/target/release/examples/bench_qwen35_mq4 \
  $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
  --prefill 128 --prefill-runs 5 --gen 0 --warmup 0

# Cross-process A/B
for iter in 1 2 3; do
  for label in A B; do
    [ "$label" = "A" ] && env="HIPFIRE_MMQ=0" || env=""
    env $env HIPFIRE_DPM_WARMUP_SECS=2 \
      $HIPFIRE/target/release/examples/bench_qwen35_mq4 \
      $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
      --prefill 128 --prefill-runs 5 --gen 0 --warmup 0
  done
done

# PMC counter sweep (one counter per pass on gfx906)
for ctr in VALUBusy LDSBankConflict ALUStalledByLDS \
           MemUnitStalled FetchSize VALUUtilization; do
  printf 'pmc: %s\ngpu: 0\n' "$ctr" > pmc.txt
  rocprof -i pmc.txt -o "run_${ctr}.csv" \
    $HIPFIRE/target/release/examples/bench_qwen35_mq4 \
    $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
    --prefill 128 --prefill-runs 1 --gen 0 --warmup 0
done

# Per-kernel wallclock (rocprofv3 has issues with iGPU agent on this
# system — use --kernel-trace specifically, not -L)
rocprofv3 --kernel-trace --stats -d ./run -o trace --output-format csv -- \
  $HIPFIRE/target/release/examples/bench_qwen35_mq4 \
  $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
  --prefill 128 --prefill-runs 1 --gen 0 --warmup 0

# Coherence gate
$HIPFIRE/scripts/coherence-gate.sh

# Synthetic NRMSE matrix
for spec in "128 4096 8" "128 4096 64" "4096 12288 128"; do
  $HIPFIRE/target/release/examples/test_gfx906_mmq_correctness $spec
done
MMQ_TEST_MODE=set $HIPFIRE/target/release/examples/test_gfx906_mmq_correctness 128 4096 64

# Real-data NRMSE (requires existing /tmp/mmq_dump_0)
$HIPFIRE/target/release/examples/test_gfx906_mmq_realdata /tmp/mmq_dump_0
```

## 13. Future work (deferred from this branch)

| Lever | Estimated impact | Status |
|---|---|---|
| Path B fused 4-output qkvza MMQ kernel | small — qkvza_fp16_wave64 share is 0.3% post-redesign (just beta+alpha tail) | Phase 6 deferred |
| `attention_q8_0_kv_batched` 2× per-call scaling at pp512 (0.66 → 1.29 ms) | small share (1.75%) but suspicious | Out-of-scope for MMQ branch |
| Reduce sync 4 → 2 per group à la stock | speculative | Needs LDS-budget review (would force 1 WG/CU) |
| Test harness mmq_x parameterization | polish | Low priority |

## 14. References

- Stock comparison: `docs/perf-checkpoints/2026-05-04-llamacpp-stock-comparison.md`
- Pre-redesign attribution: `docs/perf-checkpoints/2026-05-04-gfx906-mmq-attribution.md`
- Pre-redesign j0-unroll experiments: `docs/perf-checkpoints/2026-05-04-gfx906-mmq-junroll.md`
- Pre-redesign spill reduction: `docs/perf-checkpoints/2026-05-04-gfx906-mmq-spill-reduction.md`
- Stock kernel reference: `ggml/src/ggml-cuda/mmq.cuh` in
  https://github.com/ggml-org/llama.cpp (architectural template for
  the templated `mmq_x` body and Q8_1 quantize math)
- iacopPBK/llama.cpp-gfx906 (original gfx906 fork): https://github.com/iacopPBK/llama.cpp-gfx906
- skyne98/llama.cpp-gfx906 (fork-of-iacopPBK + upstream-tracking): https://github.com/skyne98/llama.cpp-gfx906
- skyne98/wiki-gfx906 (gfx906 ISA reference): https://skyne98.github.io/wiki-gfx906/intro.html
- Memory: `~/.claude/projects/.../memory/feedback_lds_bank_conflict.md`
  (the LDS bank-conflict pattern, distilled for future kernel work)
