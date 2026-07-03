# NPU int8 GEMM tuning loop

A reproducible loop for tuning an int8 GEMM on the Strix Halo NPU (gfx1151 +
aie2p / XDNA2). Each iteration builds one `whole_array` matmul config, runs it
on the NPU through the **compiled C++ host**, and logs sustained throughput as a
CSV row so configs are directly comparable across runs.

## Why C++ host, not the python harness

On this box the mlir-aie python `test.py` harness segfaults loading the
`mlir_aie` native MLIR bindings under Python 3.14 (independent of the kernel and
of the source/wheel version). The compiled C++ host (`whole_array.exe`) has no
such dependency and is the only reliable run path. `tune.sh` uses it for both
build and run.

## Run

```bash
./tune.sh                                  # default config matrix
./tune.sh "4096 4096 4096 64 128 64 8"     # one explicit "M K N m k n cols"
WARMUP=10 ITERS=50 ./tune.sh               # steadier timing
```

Toolchain paths are LOCAL to halo and overridable via env (`MLIR_AIE_DIR`,
`HIPFIRE_NPU_VENV`, `XRT_SETUP`, `DEV`, `DTYPE_IN/OUT`, `PEAK_TOPS`). Defaults
match this machine; nothing here is committed assuming these paths exist
elsewhere.

## Baseline (2026-07-04)

Stock `whole_array`, int8 (i8/i8, int32 accumulate in-register):

| config | throughput | % of 55 TOPS peak |
|---|---|---|
| 4096³, 8 cols | ~6.2 TOPS | ~11% |
| 4096³, 4 cols | ~3.5 TOPS | ~6% |

Bottleneck (roofline): **not** DRAM-bound (AI≈2740 op/byte ≫ ~200 machine
balance) — the loss is on-chip **L2 memtile→L1 feeding + fill/drain**, not
compute width (8-col only helped 1.76× at 4096³, 0× at 2048³).

## Tuning levers, in priority order

1. **L2→L1 feeding / buffering depth** (the diagnosed bottleneck). Exposed knobs
   reachable from `tune.sh`: MAC tile `m/k/n` (reuse per L1 load) and layout
   `b/c_col_maj`. Deeper: the memtile double-buffer depth and tile shapes inside
   `whole_array.py` / `makefile-common` — edit the design, then re-measure here.
   The AMD `programming_examples/ml/{resnet,bottleneck}` designs are the
   reference for how to stage memtile→L1 well.
2. **Microkernel** (`aie_kernels/aie2p/mm.cc`) — already vectorized (`aie::mmul`
   + `chess_prepare_for_pipelining` + 2× unroll); lower priority.
3. **Columns** — already maxed at 8; diminishing returns confirm feeding-bound.

L1 is 64 KB/tile: `dtype_out=i8` (C buffer `m*n*1*2buf`), `i32` overflows.

## Output

`results/tune-<utc>.csv`:
`utc,dtype_in,dtype_out,M,K,N,m,k,n,cols,warmup,iters,avg_us,avg_gflops,tops,pct_peak,status`
