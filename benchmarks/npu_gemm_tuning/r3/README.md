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
