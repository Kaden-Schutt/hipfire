# 2026-05-04 stock llama.cpp comparison on gfx906

Hardware: AMD MI50 (gfx906), ROCm 6.4.3.
Stock build: `ggml-org/llama.cpp` master @ commit `a4701c98f` (build id 9020),
              `cmake -DGGML_HIP=ON -DGGML_HIP_GRAPHS=OFF -DAMDGPU_TARGETS=gfx906`
              (matches `mixa3607/ML-gfx906` Dockerfile).
Hipfire: branch `feat/gfx906-mmq-dp4a` @ commit `65cf82b` (gate_up MMQ wired).

Driven by `/local/git/mi50_benchmark.txt` (yesterday's report) showing stock
llama.cpp at 766 tok/s pp512 vs our ~148 tk/s — too big a gap to be inner-loop
microarchitecture. This checkpoint nails down the structural difference.

## Headline numbers

Same model (Qwen3.5-9B Q4_K_M), same hardware, `HIP_VISIBLE_DEVICES=0`:

| Batch | Stock llama.cpp | Hipfire MMQ (residual+gate_up) | Ratio |
|---|---|---|---|
| pp32 | 289.7 tok/s | — | — |
| pp64 | 405.2 tok/s | — | — |
| **pp128** | **500.5 tok/s** | **147.8 tok/s** | **3.39× behind** |
| pp256 | 646.8 tok/s | — | — |
| pp512 | 749.5 tok/s | — | — |
| pp1024 | 735.7 tok/s | — | — |
| **tg256** (decode) | **64.7 tok/s** | ~52 tok/s | **1.24× behind** |

`/local/git/mi50_benchmark.txt` headlined pp512 = 766.7 from `mixa3607` build;
our local stock build at `a4701c98f` measures 749.5 (within 2% — the deltas
are 30 stock master commits later).

**Stock prefill scales nearly linearly with batch up to pp512, then saturates.
Hipfire only measured at pp128 (the bench harness's --prefill 128).**

## Per-kernel attribution from `rocprof --hip-trace`

Stock `mul_mat_q` is one templated kernel handling all GEMM shapes:

| Stock kernel (template instantiation) | Calls | Avg per-call | Total | % of GPU time |
|---|---|---|---|---|
| `mul_mat_q<Q4_K, mmq_x=64, need_check=false>` | 258 | **0.85 ms** | 219 ms | 42.5% |
| `mul_mat_q<Q5_K, 64, false>` | 96 | 1.21 ms | 116 ms | 22.5% |
| `mul_mat_q<Q6_K, 64, false>` | 40 | 1.93 ms | 77 ms | 15.0% |
| `mul_mat_q<Q8_0, 64, true>` | 96 | 0.36 ms | 35 ms | 6.7% |
| `mul_mat_vec_q<Q6_K, 1>` | 2 | 1.49 ms | 3 ms | 0.6% |
| rocBLAS (`Cijk_*`) | 32 | 0.05 ms | 1.5 ms | 0.3% |

Stock total: 528 GEMM-class calls = 451 ms = **87.6% of total GPU time**.
rocBLAS contribution is **0.3%** — they don't win prefill via rocBLAS. They
win it via their custom `mul_mat_q` kernel.

For comparison, hipfire's prefill GEMM total: 256 calls = 944 ms (from prior
HIP trace at MMQ+screen). **Per-call: stock 0.85 ms vs hipfire 3.69 ms = 4.3×
slower per call.**

## The structural difference: WG topology

| Property | Stock (gfx906) | Hipfire MMQ |
|---|---|---|
| `nwarps` (warps per WG) | **4** | 2 |
| Threads per WG | **256** | 128 |
| `mmq_y` (rows per WG) | 128 | 128 |
| `mmq_x` (cols per WG) | **runtime-selected from {8,16,24,32,40,48,56,64,88,96,112,128}** | hardcoded 8 |
| `mmq_x` typical for prefill | **64** | 8 |
| Per-warp tile cols (`mmq_x/nwarps`) | 16 | 4 |
| Per-warp tile rows (`mmq_y/warp_size`) | 2 | 2 |
| Per-thread accumulator floats | **32** | 8 |
| WG block dim | **(64, 4, 1)** | (64, 2, 1) |
| `__launch_bounds__` | `256, 2` | `128, 2` |

Stock uses **2× the threads per WG and 8× the columns per WG**. With nwarps=4
sharing the work, each warp still owns the same 16×2 sub-tile that we'd have
if we tried mmq_x=8 with nwarps=2 — but stock fits **8 of those sub-tiles per
WG instead of 1**, amortizing the X tile load (loaded once, reused by 8 column
sub-tiles' worth of compute).

## Stock's kernel SPILLS too — and is still 4× faster

ELF metadata for `mul_mat_q<Q4_K, mmq_x=64, need_check=false>`:

| Property | Stock Q4_K mmq_x=64 | Hipfire MMQ_X=8 (committed) | Hipfire MMQ_X=64 (pre-j0-unroll) |
|---|---|---|---|
| `vgpr_count` | **128** | 60 | 128 |
| `vgpr_spill_count` | **144** | 0 | 144 |
| `private_segment_fixed_size` | **476 B** | 0 | 564 B |
| `wavefront_size` | 64 | 64 | 64 |

This **completely overturns the spill-elimination narrative** from this
session's commits 17c05f3 (j0 un-unroll) and `2026-05-04-gfx906-mmq-junroll.md`.
We treated 144 spilled VGPRs as catastrophic and shrank `mmq_x` 8× to
eliminate them. Stock ships **the same 144 spills** and runs 4× faster per
call. The spills are not the problem — the tile size and arithmetic
intensity are.

The j0 un-unroll commit (17c05f3) was a real local optimum *for our 2-warp
WG topology* (where mmq_x=8 is forced by single-warp tile width). It bought
+16% on a kernel that's still 4× behind a different design point. The
**bigger structural redesign supersedes it.**

## Why our spill-fear was a red herring

Looking at our HIP trace duration vs counter-driven duration earlier in the
session:

- `rocprof --pmc` (counter-sampling) reported MMQ K=4096 at 3.4 ms per call.
- `rocprof --hip-trace` (no counters) reported MMQ K=4096 at **1.3 ms per call**.

Counter-sampling inflated wallclock 50–100%. With counters off, the hipfire
MMQ residual kernel runs at 1.3 ms/call — and stock's `mul_mat_q<Q4_K, 64>`
runs at **0.85 ms/call**. The *real* gap on the residual kernel is 1.5×, not
4×. The 4.3× headline number is across all hipfire GEMM kernels (including
gate_up FP16 wave64 at 12 ms/call which is much slower than what stock's MMQ
achieves on the same shape).

Translating: **the bigger structural opportunity is porting MMQ to gate_up
and qkvza** (already partially done — gate_up shipped in `65cf82b`, qkvza
attempted and reverted because our 2-warp WG topology can't handle the M=16
beta/alpha matrices well). With a 4-warp WG topology, mmq_x can adapt to
narrow batches and qkvza becomes feasible.

## What stock has that we don't

From `mmq.cuh:298-313`:

```cpp
// Host (dispatch) side:
static int mmq_get_nwarps_host(const int cc, const int warp_size) {
    return amd_mfma_available(cc) ? 8 : 256/warp_size;  // gfx906 → 4
}
// Device side:
static constexpr __device__ int mmq_get_nwarps_device() {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    return 8;
#else
    return 256/ggml_cuda_get_physical_warp_size();  // gfx906 → 4
#endif
}
```

Stock's kernel is a **template on `mmq_x`** (compiled for {8, 16, 24, 32, 40,
48, 56, 64, 88, 96, 112, 128}) with `nwarps = 4` baked in via `__launch_bounds__`.
At dispatch time they pick the smallest mmq_x that holds the batch. This
gives them:
- Big tiles (high arithmetic intensity per LDS load) for big batches
- Small tiles (low waste of M-direction work) for small batches like
  alpha/beta in qkvza
- One kernel invocation handles the whole GEMM (one tile per WG, multiple
  WGs cover the batch)

Plus several PRs touching this kernel since unverbraucht's last sync:
- `66c4f9ded ds_read_b128 for q4_0/q4_1 mmq` — claimed +12.6% on MI50
- `9725a313b reduce MMQ stream-k overhead (fastdiv)` — minor on older arch
- `4eac5b450 refactor mma data loading for AMD` — up to 2.7× on MMQ on MI100

## Implications for hipfire's next move

The `gemm_hfq4g256_residual_mmq_gfx906.hip` kernel needs **structural
redesign**, not microoptimization:

1. **Switch to nwarps=4 (256 threads/WG)** — `__launch_bounds__(256, 2)` and
   `(64, 4, 1)` block dim.
2. **Template on mmq_x** — at minimum {8, 16, 32, 64, 128}, dispatched by
   batch size.
3. **Accept the spill** — at mmq_x=64 nwarps=4, expect ~144 spills/thread.
   That's the operating point stock found, and it's faster than our spill-
   free shape.
4. **Q8_1 activation layout matches stock's** — we already use `block_q8_1`
   (with the `_mmq_ds4` quantize variant); this part is right.
5. **Once redesigned, qkvza becomes natural** — small mmq_x (say 16) for the
   alpha/beta matrices, big mmq_x for qkv/z.

The ds_read_b128 conversion we tried (Follow-up 2 in junroll log) didn't
help us because our small-tile kernel had a small dp4a body where LDS-issue
waits weren't the bottleneck. In a 4-warp 64-col-mmq_x kernel with the body
8× larger, the LDS-issue waits do become significant — and the b128 PR's
+12.6% MI50 claim makes sense in *that* context.

## Estimated end-to-end target

Stock's prefill at pp128 = 500 tok/s. If hipfire reaches the same kernel
structure, we should land at the same or better (we have HFQ4-G256 vs
their Q4_K_M; the per-byte accuracy and dequant cost differ slightly).

**+3.4× from current 148 tok/s → ~500 tok/s** if we close the kernel-design
gap. That's a real opportunity but it's a multi-day kernel rewrite, not a
one-line edit.

## Cross-reference

- Yesterday's report: `/local/git/mi50_benchmark.txt`
- Stock kernel source: `/tmp/llama-stock/ggml/src/ggml-cuda/mmq.cuh`
- Stock binary: `/tmp/llama-stock/build/bin/llama-bench`
- Stock rocprof: `/tmp/rocprof_stock/stock_pp128.{csv,stats.csv,hip_stats.csv}`
- Hipfire current MMQ: `kernels/src/gemm_hfq4g256_residual_mmq_gfx906.hip`
- Hipfire latest perf checkpoint: `2026-05-04-gfx906-mmq-junroll.md`
- L2 prefetch plan (now superseded): `plans/gfx906_mmq_l2.md`
