# R4 — 8-column aggregate W4A8, measured end-to-end through hipfire

Goal: get the honest aggregate W4A8 compute number on halo's NPU, driven through
hipfire's own `NpuKernel` dispatch (no XRT), to decide whether NPU offload is worth
wiring into the runtime.

## R4a — 8-column compute ceiling (`r4a_cols_run.py` + `npu_gemm_bench`)

8 columns, each running the R2a W4A8 compute core (`../r2/r2a_gemm.cc`) pinned to
its own column (distinct compute tile + shim DMA channel), streaming int4 weight
tiles that are reused `(INNER+1)*NACC` times against `NACC` resident int8
activation tiles — R2a's II=1 fake-reuse recipe, so the cores are **compute-bound**.
This is the aggregate compute *ceiling*; real streaming GEMM (feed-bound) is lower.

`r4a_cols_run.py` builds the xclbin; `npu_gemm_bench` (crates/hipfire-xdna example)
loads it via `NpuKernel`, validates the all-ones result (C lane = `(INNER+1)*16`),
then times a dispatch loop.

### Result (halo, gfx1151 NPU / XDNA2, COLS=8 NACC=4 INNER=64)

| N_BTILES | MACs/dispatch | per-dispatch | TOPS |
|---|---|---|---|
| 256  | 0.55 G | 187 µs → 164 µs* | 5.8 → 6.7* |
| 1024 | 2.18 G | 428 µs | 10.2 |
| 4096 | 8.72 G | 1468 µs | 11.9 |

A linear fit over N_BTILES separates the two numbers:

- **Compute-bound rate ≈ 12.8 TOPS** (8 columns, W4A8, overhead-free). Scales ~8×
  the single-core R2a rate (~460 MAC/cyc/core), so the array scales near-linearly —
  but it lands **below the ~15.7-TOPS int8 whole_array reference** and only ~2× the
  ~7-TOPS DynamicDispatch int4 decode kernel.
- **Per-dispatch fixed overhead ≈ 78 µs** (`*` after caching the ERT command BO in
  `NpuKernel` instead of rebuilding it each call — CREATE_BO + mmap removed; the
  residual is inherent amdxdna submit latency: input syncs + EXEC_CMD + syncobj
  wait, ~5 ioctls). Overhead is only amortized for large GEMMs (N_BTILES ≳ 1000).

### Read

The honest 8-column W4A8 compute ceiling is **~12.8 TOPS**, and it is
(a) a *ceiling* — real streaming GEMM is feed-bound and lower, and (b) only
reachable on large GEMMs because of the ~78 µs dispatch latency. Against the GPU's
real W4A8 (~50 TOPS on gfx1151, see the iu4-GEMM tuning notes), a *concurrent* NPU
offload buys at most ~+20–25% aggregate prefill throughput, for large GEMMs only.
Real but modest — this measurement is the input to the runtime-offload go/no-go.

Reaching compute-bound *beyond* the reference ceiling would need the R4 weight-
broadcast/cascade array dataflow (unbuilt; findings.md shows no existing kernel on
this hardware demonstrates it).
