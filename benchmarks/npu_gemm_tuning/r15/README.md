# R15 — GPU + NPU run in parallel on Phoenix (the concurrency gate: PASS)

The R5–R14 line proved the NPU can do a real W4A8 GEMM (1.44 TOPS). But the NPU is worth
using as a *second engine* only if it makes progress **concurrently** with the iGPU — the
value is parallelism (offload a prefill slice; run a pipelined spec-decode draft), not raw
speed. R15 measures whether the gfx1103 iGPU and the XDNA1 NPU actually overlap on the shared
Phoenix die, or whether one stalls the other.

Probe: `crates/hipfire-rdna/examples/npu_gpu_overlap.rs` runs a prefill-shaped GPU WMMA GEMM
(F16, M=1024 K=7168 B=256) and the R14 whole_array NPU GEMM dispatch, each in a time-boxed
busy loop — SOLO, then CONCURRENTLY (NPU on its own thread, GPU on main, started via a
barrier). If each engine keeps its solo throughput under concurrency, they overlap.

## Result (two trials, 4 s and 6 s windows)

|              | solo            | concurrent      | keeps |
|--------------|-----------------|-----------------|-------|
| GPU WMMA GEMM| ~845 GFLOP/s    | ~589 GFLOP/s    | ~70%  |
| NPU dispatch | ~476 disp/s     | ~373 disp/s     | ~78%  |
| NPU GEMM     | ~1.15 TOPS      | ~0.90 TOPS      |       |

**Overlap efficiency ≈ 48%, stable across trials.** The engines keep ~70% / ~78% of their
solo throughput when run together, and 70%+78% = 148% > 100% — i.e. this is **genuine
parallelism, not time-slicing** (which would cap each at ~50%). Combined useful work is
**~1.48× a single engine**, with ~25% mutual slowdown from **shared UMA memory bandwidth**
(the F16 GEMM at K=7168 is very BW-heavy — a near-worst-case contender; the NPU whole_array
pulls only ~5.6 GB/s, so this ~48% is a conservative floor — a more compute-bound GPU
workload would contend less).

## What this settles

The concurrency gate — the make-or-break for the "NPU as a parallel unit" thesis — **passes**:

- **Parallel prefill offload is viable.** The NPU adds ~0.9 TOPS of concurrent GEMM while
  costing the GPU ~30% of its rate → net ~+48% throughput. Split a prefill's GEMMs GPU+NPU.
- **A spec-decode draft on the NPU is viable — and strictly better than single-GPU spec
  decode.** Single-GPU spec decode already works while the draft steals the GPU's own
  bandwidth; moving the draft to the NPU offloads its compute entirely and costs the GPU only
  the ~30% UMA contention. Pipeline it: GPU verifies batch N while the NPU drafts N+1.

The remaining work is **software, not hardware**: async NPU dispatch on the inference hot path
(today's path is synchronous — the GPU stalls on the NPU, so this overlap is unused) plus a
scheduler that hands the draft / a prefill slice to the NPU. The physics is proven — the two
engines overlap.

Repro: build an R14 xclbin, then
`cargo run --release -p hipfire-rdna --example npu_gpu_overlap -- <r14-dir> <asz> <wsz> <csz> 4 <macs>`.
