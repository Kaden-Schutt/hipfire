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

## R4b — FULL 32-core array via memtile routing (`r4b_grid_run.py`)

R4a used only **8 of 32 cores** (one tile row). Feeding more cores straight from
each column shim exhausts shim DMA capacity ("no ShimNOCTile with sufficient DMA
capacity" past ~1 core/column). R4b routes through the **memtile** (tile row 1) per
column — the R4 weight-stationary dataflow:

- **W broadcast** to all ROWS cores in the column (`.forward(Tile(col,1))` +
  multiple `.cons()`): one shim stream, weight reused ROWS× spatially.
- **A distribute** into ROWS distinct activation blocks (`.split(Tile(col,1))`).
- **C gather** from the ROWS cores (`.join(Tile(col,1))`).

3 shim channels/column regardless of ROWS, so it scales to the full 4×8 = 32 cores.

### Result (halo, 4 rows × 8 cols = 32 cores, through NpuKernel)

| config | reuse | TOPS |
|---|---|---|
| 8-core (R4a), INNER=64 | fake | 12.8 |
| **32-core (R4b), INNER=64** | **fake** | **~40** (42.4 @ NB1024, 39.4 @ NB4096) |
| 32-core (R4b), INNER=0 | spatial only, 16×16 tiles | 5.25 |
| IRON whole_array int8 reference (big tiles) | real | 15.7 |

### Read (corrects R4a's "modest")

- **The array scales.** 32 cores sustain ~3.1× the 8-core rate → a **~40-TOPS W4A8
  compute capacity** (fake-reuse ceiling). The silicon can do it, and the memtile
  weight-broadcast dataflow feeds all 32 cores. Beats the 15.7 int8 reference.
- **But real streaming GEMM is per-tile-overhead-bound, not compute-bound.** The
  INNER sweep is the proof: 0 → 64 moves 5.25 → 40 TOPS purely by adding
  compute-per-tile to amortize the fixed per-acquire fifo/DMA overhead. With tiny
  16×16 tiles the real (INNER=0) rate is only 5.25 — *below* the reference, which
  uses big 64×64 tiles to amortize overhead and reaches 15.7.
- **So the deliverable real-W4A8 ceiling is still ~15.7 TOPS** (reference dataflow),
  vs GPU ~50 → concurrent offload ~+30% at best. The ~40-TOPS capacity is real
  headroom, but capturing it for *real* GEMM needs bigger tiles + lower per-tile
  overhead (cross-core cascade / K-resident C) — the unbuilt dataflow. Next step to
  chase it: R4b with a larger mmul tile (m/k/n = 64) and measure the real (INNER=0)
  rate against the 15.7 reference.
