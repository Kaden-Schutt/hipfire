# R3 — real W4A8 GEMV / GEMM kernels (K-accumulating) + wire-in

R2a probed compute by repeating one weight tile; R3 does REAL K accumulation over
distinct streamed weights — the kernels an inference path would actually call.

## R3a — batched-decode W4A8 GEMV (`r3a_gemv.cc` + `r3a_run.py`)

M=4 (a spec-decode / small-batch column), one N-block. Activations and int4 weight
tiles stream as KCHUNK-deep super-tiles over the K contraction; each super-tile
does KCHUNK II=1 macs in a register accumulator (R2a named-accumulator recipe) and
folds the partial into a resident C (vector add — `aie::mmul` can't reseed its acc).

Measured (host-wall differential, single core/stream): **11.2 GB/s weight-feed,
0.18 TOPS**. As predicted for M=4 this is **bandwidth-bound** — throughput tracks
the R1 single-stream feed (~14.4 GB/s), not compute. The ~3 GB/s gap is the
co-streamed activations (own shim) + per-super-tile C load/add/store; in a real
multi-N-block GEMV the activation stays resident and reused, so W approaches the
full per-stream rate, and 8 columns give ~44 GB/s of weight feed. This is the case
where int4's half-weight-bytes matters: decode token rate ∝ feed / weight-bytes.

Open: bit-exact validation vs a numpy reference needs matching `aie::mmul`'s 16×16
tile layout (deferred); multi-N-block + 8-column scale-up; then GEMM (large M,
compute-bound, R2a path) and wiring both into the runtime.

## Wire-in status

The existing NPU dispatch (`hipfire-arch-qwen35/src/xdna1_ffi.rs`) dlopens
`libhipfire_xdna1.so` — a binary blob (symbols reverse-engineered from disasm) with
swiglu/rmsnorm/rope/etc. ops, **not on disk here**, and with no gemm/gemv symbol we
can add. So wiring the W4A8 gemv/gemm needs a **direct XRT-from-Rust** loader
(xclbin + instr → run), not that dormant blob — tracked as the next integration arc.

## R3b — prefill W4A8 GEMM (`r3b_gemm.cc` + `r3b_run.py`)

Real K accumulation with NACC row-blocks (M = NACC·4): one streamed int4 weight
tile is macced into NACC named accumulators (R2a recipe), so weight reuse = NACC
and AI = NACC·8 MACs/byte. Measured single-core, single-N-block:

| NACC (M) | TOPS | MAC/cyc | limiter |
|---|---|---|---|
| 4 (M=16) | 0.70 | 195 | feed-bound (W feed ~11 GB/s, matches R1) |
| 8 (M=32) | 0.26 | 71  | **A-stream-bound** (A super = 4× W bytes) |

The single-N-block probe streams the full activation for one output block, and at
M=32 that A stream (NACC·MR·K int8) is 4× the weight stream (K·16 int4) — so it
goes A-bound, not compute-bound. NACC=8 also L1-caps KCHUNK (A super 32 KB at
KCHUNK=64 overflows), forcing KCHUNK=16 which adds C-fold overhead.

**To reach compute-bound prefill, A must stay resident and be reused across N-blocks
(or K-tiled through the memtile)** — the standard AIE GEMM dataflow that the
reference kernels use to approach peak. That is the next step: A-resident, stream W
across many N-blocks, then scale to the 8×4 array. This is the same ~27%-efficiency
dataflow wall documented for the `whole_array` reference in ../findings.md.
