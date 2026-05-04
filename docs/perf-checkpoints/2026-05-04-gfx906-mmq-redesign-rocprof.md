# gfx906 MMQ redesign — rocprof attribution + counters

Date: 2026-05-04
Hardware: gfx906 (MI50). HIP 6.4.3.
Build: branch `feat/gfx906-mmq-dp4a` at commit `c022682`.
Workload: Qwen 3.5 9B mq4, `bench_qwen35_mq4 --prefill 128 --prefill-runs 1`.
Tools: `rocprofv3 --kernel-trace --stats` for wallclock; `rocprof -i pmc.txt`
for hardware counters (one counter per pass — three exceeds gfx906 HW limit).

## TL;DR

**pp128 prefill on Qwen 3.5 9B mq4: 141 → 287 tok/s (2.04×).** Plan
target was ≥240 (1.6×) — beats it by 20 %. Cross-process A/B with DPM
warmup confirms the speedup is structural, not a within-session
artifact (B-side spread 0.04 % across 3 alternating iterations).

The new MMQ kernel itself is **neither compute-bound nor memory-bound**:
VALUBusy 27–41 % (low), MemUnitStalled ≤ 0.25 (essentially zero), VALU
utilization 100 %. Fetch is 24 KB/call (vs 69 KB for FP16 wave64 at the
same shape) — Q8_1 activation compression is paying off.

The remaining headroom is in **inter-warp synchronization**: 8 sub-iters
per HFQ4 group × `__syncthreads()` × many groups. ds_read_b128 (deferred
per plan §Q8) will not help here because the load isn't on the critical
path. Better follow-up lever: reduce barrier frequency (Option B's
8× sync per group is what bought the 2 WGs/CU LDS budget; relaxing it
needs care).

## End-to-end prefill bench

Qwen 3.5 9B mq4 on MI50, last of 5 runs/config (post-JIT). Same binary,
same model, same prompt — only `HIPFIRE_MMQ` env differs.

| Prefill | Baseline (MMQ=0) | New MMQ (MMQ=1, SCREEN=1) | Speedup |
|---|---|---|---|
| pp32  | 137 tok/s | 228 tok/s | 1.67× |
| pp64  | 140 tok/s | 264 tok/s | 1.89× |
| **pp128** | **141 tok/s** | **287 tok/s** | **2.04×** |
| pp256 | 143 tok/s | 307 tok/s | 2.15× |

Speedup grows with batch size, consistent with the greedy mmq_x ladder
picking larger tiles at higher batches: pp32 → mmq_x=32 (4 col-tiles
per row), pp128 → mmq_x=64 (2 col-tiles), pp256 → mmq_x=64 (4 col-tiles
across 2 row-tile passes).

### Cross-process A/B (alternating, fresh process, DPM-warmed)

Each row is a separate process invocation. A/B alternation defeats any
monotonic thermal/DPM drift; `HIPFIRE_DPM_WARMUP_SECS=2` stabilizes
clocks before the timed phase.

| Iter | A (MMQ=0) | B (MMQ=1) | B/A |
|---|---|---|---|
| 1 | 141.3 tok/s | 287.1 tok/s | 2.03× |
| 2 | 141.0 | 287.2 | 2.04× |
| 3 | 140.9 | 287.1 | 2.04× |
| **Median** | **141.0** | **287.1** | **2.04×** |

A spread 0.4 tok/s (0.3 %), B spread 0.1 tok/s (0.04 %), B/A identical
to 2 decimals across iterations. Per
`docs/methodology/perf-benchmarking.md`, this is well below the
±10–15 % within-session noise floor — the speedup is structural.

## Correctness gates (prerequisite for the perf claim)

Three layers of correctness validation cleared before this perf
result was accepted:

| Gate | Result |
|---|---|
| Synthetic NRMSE (49 shapes, residual+set, mmq_x ∈ {8..64}, K ∈ {4096, 12288}, partial-M, partial-N, M=4096) | 0.04–0.18 % vs FP16 wave64 |
| Real-data NRMSE (Qwen pp128 dump M=4096 K=4096 N=128) | 0.29 % (≤0.30 % threshold) |
| Coherence gate (4 mq4 rows: 0.8B-cap, 4B-code, 9B-reason, 9B-tool-call) | all PASS, fluent output, no `<\|im_start\|>` leak |

See `plans/gfx906_mmq_redesign.md` §"Phase 2b validation findings"
for full breakdown.

## Per-kernel wallclock (rocprofv3 --kernel-trace --stats)

Top GEMM kernels in the prefill, sorted by total time share:

| Kernel | Calls | Avg per-call | Share |
|---|---|---|---|
| `gemm_qkvza_hfq4g256_fp16_wave64`         | 48  | 6.30 ms | **28.87 %** |
| `gemm_hfq4g256_residual_fp16_wave64`      | 166 | 1.36 ms | 21.55 % |
| `gemm_hfq4g256_residual_mmq_gfx906_full_set_x64` (gate_up) | 128 | **1.40 ms** | 17.05 % |
| `gemm_hfq4g256_residual_mmq_gfx906_full_add_x64` (residual K=12288) | 90 | **1.52 ms** | 13.06 % |
| `gemm_qkv_hfq4g256_fp16_wave64`           | 16  | 5.25 ms | 8.02 % |
| `gemm_hfq4g256_residual_mmq_gfx906_full_add_x16` (residual K=4096) | 128 | **0.42 ms** | 5.15 % |

The new gfx906 MMQ entry symbols carry **35.3 %** of GEMM time
(17.05 + 13.06 + 5.15). The remaining 58.4 % is FP16 wave64
(`qkvza` + `qkv` + `residual` screening fallbacks). Per the redesign
plan §Phase 6, qkvza fused-4-output MMQ is the next-largest opportunity
(28.87 % alone).

### Per-call comparison vs prior attribution

From `2026-05-04-llamacpp-stock-comparison.md`:

| Path | Hipfire pre-redesign | Stock (Q4_K) | Hipfire post-redesign |
|---|---|---|---|
| Residual MMQ K=4096 | 1.30 ms | 0.85 ms | **0.42 ms** (mmq_x=16) |
| Residual MMQ K=12288 | (extrapolated 3.9 ms) | (extrapolated 2.55 ms) | **1.52 ms** (mmq_x=64) |

Stock numbers are extrapolated linearly from the K=4096 measurement.
Caveat: stock dispatches mmq_x=64 even at small batches (it has no
mmq_x=16 instantiation in the comparison data); a fairer comparison
would re-run stock at the exact shapes hipfire dispatches.

## Hardware counters (rocprof -i pmc.txt, one counter per pass)

`rocprof --pmc` inflates wallclock 50–100 % per
`2026-05-04-llamacpp-stock-comparison.md §"Why our spill-fear was a red
herring"`, so absolute timings here are not comparable to the
`--kernel-trace` data above. Counter ratios within a run are valid.

### VALUBusy (% of GPU time vector ALU instructions are processed)

| Kernel | Calls | VALUBusy |
|---|---|---|
| `gemm_qkvza_hfq4g256_fp16_wave64`         | 24  | **61.5 %** |
| `gemm_qkv_hfq4g256_fp16_wave64`           | 8   | 61.4 % |
| `gemm_hfq4g256_residual_fp16_wave64`      | 147 | 54.5 % |
| `..._mmq_gfx906_full_set_x64`             | 64  | **41.1 %** |
| `..._mmq_gfx906_full_add_x64`             | 45  | **27.4 %** |
| `..._mmq_gfx906_full_add_x16`             | 128 | 18.8 % |

The new MMQ kernel sits at 18–41 % VALUBusy — much lower than the
FP16 wave64 path's 54–61 %. Counter-intuitive but expected: dp4a does
4 multiply-accumulates per instruction; FP16 wave64 does 1. Lower
VALUBusy + 2× faster wallclock = the kernel is achieving more
arithmetic per cycle.

### VALUUtilization (% of vector lanes active per wave)

| Kernel | Avg |
|---|---|
| `..._mmq_gfx906_full_set_x64` | **100.00 %** |
| `..._mmq_gfx906_full_add_x64` | **100.00 %** |
| `..._mmq_gfx906_full_add_x16` | **100.00 %** |
| `qkvza_hfq4g256_fp16_wave64`  | 99.81 % |

100 % across all new MMQ entry symbols → no wave divergence. The
`sub_block = sub_iter % 4` indexing is uniform across the wave; the
chunk-major X loader's `task_id = tid * 2 + loop` distribution does
not introduce per-lane branches.

### MemUnitStalled (% of GPU time the memory unit is stalled)

| Kernel | Avg |
|---|---|
| `qkv_hfq4g256_fp16_wave64`    | 1.83 |
| `qkvza_hfq4g256_fp16_wave64`  | 1.80 |
| `residual_fp16_wave64`        | 1.11 |
| `..._mmq_gfx906_full_set_x64` | **0.25** |
| `..._mmq_gfx906_full_add_x64` | **0.08** |
| `..._mmq_gfx906_full_add_x16` | **0.08** |

MMQ is ~14× less memory-stalled than the FP16 wave64 path it replaces
on the same shape. Q8_1 activations + nibble-packed weights cut HBM
pressure dramatically.

### FetchSize (HBM bytes read per call)

| Kernel | Avg KB/call |
|---|---|
| `qkvza_hfq4g256_fp16_wave64`  | 480 |
| `qkv_hfq4g256_fp16_wave64`    | 399 |
| `residual_fp16_wave64`        | 69 |
| `..._mmq_gfx906_full_set_x64` | 52 |
| `..._mmq_gfx906_full_add_x16` | 24 |
| `..._mmq_gfx906_full_add_x64` | 24 |

At the same residual K=4096 shape, MMQ fetches **24 KB/call vs FP16
wave64's 69 KB/call** — 2.9× less HBM read. This is mostly the Q8_1
activation compression (4× smaller than FP16) plus better X-tile
reuse from the larger 4-warp WG.

## Why ds_read_b128 (plan §Q8 deferred) won't help much

The plan deferred ds_read_b128 vectorization with the rationale
"revisit after v1 lands." The counter data answers that question:

- ds_read_b128 reduces the *number* of LDS-load instructions per
  sub-iter (4× fewer issues). It would help if VALUBusy were saturated
  at the LDS-issue rate.
- VALUBusy is at 27–41 %, not saturated. LDS issue is not on the
  critical path.
- The bottleneck is **inter-warp synchronization** — Option B's
  8 `__syncthreads()` per HFQ4 group, × ~6 groups per K=12288 call,
  × 32 col-tiles per WG = ~1500 sync points per call.

Better follow-up lever: relax the sync requirement. Stock llama.cpp
uses the same 32-K streaming pattern but with **2 syncs per group**
instead of 8 (it loads X once per group, both Q8_1 blocks once per
group, and issues 4 sub-iter-equivalents between syncs). Adopting
this would need a larger LDS buffer (load 128-K of X up front) which
might push us back over the 32 KiB/WG cap; revisit with the b128 axis
together.

## What this validates

- Phase 4 acceptance gate ✅ — pp128 prefill 287 tok/s, 2.04× over
  baseline, hard-confirmed by cross-process A/B (B spread 0.04 %).
- VALUUtilization 100 % rules out wave divergence as a confounding
  variable for any future tuning experiment.
- MemUnitStalled ~0 rules out HBM bandwidth as the next bottleneck.

## What this leaves on the table

| Lever | Headroom estimate | Plan reference |
|---|---|---|
| qkvza fused-4-output MMQ (Phase 6) | ~28.87 % share × maybe 2× speedup → ~14 % end-to-end | §Phase 6 |
| qkv MMQ (currently FP16 wave64) | ~8.02 % share × maybe 2× → ~4 % end-to-end | implied by Phase 6 |
| Reduce sync frequency (8/group → 2/group) | speculative, needs LDS-budget analysis | not in plan |
| ds_read_b128 vectorization | likely ≤ 5 % (memory + LDS already cheap) | §Q8 |

The qkvza port is by far the largest remaining lever — 28.87 % share
is bigger than all three new MMQ entry symbols combined (35.3 %).

## Reproducing this report

```sh
mkdir -p /tmp/rocprof_mmq_redesign && cd /tmp/rocprof_mmq_redesign

# Wallclock per kernel
HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=1 \
rocprofv3 --kernel-trace --stats -d ./run1 -o trace --output-format csv -- \
  $HIPFIRE/target/release/examples/bench_qwen35_mq4 \
  $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
  --prefill 128 --prefill-runs 1 --gen 0 --warmup 0
# inspect ./run1/trace_kernel_stats.csv

# Hardware counter (one per pass — gfx906 PMC HW limit)
for ctr in VALUBusy VALUUtilization MemUnitStalled FetchSize WriteSize SALUBusy; do
  printf 'pmc: %s\ngpu: 0\n' "$ctr" > pmc.txt
  HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=1 \
    rocprof -i pmc.txt -o "run_${ctr}.csv" \
    $HIPFIRE/target/release/examples/bench_qwen35_mq4 \
    $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
    --prefill 128 --prefill-runs 1 --gen 0 --warmup 0
done
```

Note: legacy `rocprof` segfaults at process exit after data is written;
the data files are valid (segfault is post-flush). `rocprofv3 -L`
crashes on this system because of the iGPU agent (`gfx90c`); we use
legacy `rocprof` for counters as a workaround.
