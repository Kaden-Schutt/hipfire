# OQ8 / OQ+ → XDNA1 NPU — Feasibility Spike Results & Go/No-Go

**Dates:** 2026-06-23 (plan) → 2026-06-24 (results)
**Box:** Ryzen 7 7840HS — gfx1103 (Radeon 780M) + XDNA1/Phoenix NPU (AIE2, 16 TOPS, 4×4 tiles)
**Status:** Spike complete. **Verdict: NO-GO** for an NPU OQ8/OQ+ GEMM path on this
APU as a speed win, under the current synchronous-dispatch model. Correctness was
fully proven; perf is not competitive. Details below.

## What was built (and proven)

The OQ-family grouped int8 GEMM runs **correctly** on the NPU:

- `tools/npu/oq_gemm_design.py` — int8·int8→int32 single-core matmul via IRON
  `kernels.linalg.mm` (the `aie::mmul<4,8,8,int8,int8,acc32>` micro-kernel — the same
  compute the GPU OQ8 path runs on `v_wmma_i32_16x16x16_iu8`), adapted from the upstream
  mlir-aie `single_core` matmul example. Self-contained env bootstrap.
- `tools/npu/test_oq_gemm_npu.py` — OQ8 (W8A8) and OQ+ (W4A8) oracle: per-256-group
  symmetric quant → per-group NPU int8 contraction → host f32 rescale.
- `tools/npu/bench_oq_gemm_npu.py` — single-core perf sweep.

**Correctness (NPU1/Phoenix, all bit-exact vs numpy int64 contraction):**

| Format | Shape | int32 contraction | f32 rescale |
|--------|-------|-------------------|-------------|
| OQ8 W8A8 | M64 K256 B64 (1 grp) | PASS | PASS |
| OQ8 W8A8 | M64 K512 B64 (2 grp) | PASS | PASS |
| OQ+ W4A8 | M64 K256 B64 | PASS | PASS |

The OQ8↔OQ+ delta is only the weight quant range (int8 vs int4-unpacked-to-int8),
confirming the shared-kernel approach. (FWHT rotation omitted — orthonormal pre-step
that does not affect matmul reproduction. On-tile int4 unpack — the real OQ+ DMA win —
is a follow-up; here weights are host-unpacked to int8.)

## Performance (the deciding data)

Representative shape: Qwen3.5-1.5B FFN down-proj **M=1536, K=8960 (35 groups)**, raw
int8 matmul (one dispatch over full K; the per-group f32 rescale is cheap host/epilogue
arithmetic, excluded to isolate the NPU compute).

**Single-core (1 of 16 AIE tiles), `bench_oq_gemm_npu.py`:**

| B | NPU µs | int8 GFLOP/s | weight BW |
|---|--------|--------------|-----------|
| 32 | 7128 | 124 | 1.9 GB/s |
| 64 | 12252 | 144 | 1.1 GB/s |
| 128 | 23277 | 151 | 0.6 GB/s |
| 256 | 45188 | 156 | 0.3 GB/s |

**Whole-array (16 AIE tiles), upstream `whole_array.py` run directly:**
`-M 1536 -K 8960 -N {128,256} -m 64 -k 64 -n 32 --n-aie-cols 4 --dtype_in i8 --dtype_out i32 --b-col-maj 1`

| B | NPU µs | int8 GFLOP/s |
|---|--------|--------------|
| 128 | 1864 | 1890 |
| 256 | 3507 | 2009 |

Multi-core scales ~12.9× over single-core (near-linear in the 16 tiles). The realistic
NPU ceiling here is **~2.0 TFLOP/s int8 — ~12% of the 16-TOPS peak**, weight-DMA-bound
at ~3.9 GB/s.

## Why NO-GO

1. **Synchronous dispatch makes any NPU time pure addition.** This is already
   established for elementwise (`NPU-RESULTS.md`: SwiGLU on 24 layers = −2.4%, the
   ~180 µs dispatch floor × layers exceeding the GPU op cost). A GEMM that takes
   **1.9–3.5 ms per matmul** is far worse — there are dozens of matmuls per layer ×
   28 layers. Synchronously, this cannot beat the GPU.
2. **No compute advantage.** The NPU's int8 peak (16 TOPS) is below the 780M's int8
   WMMA throughput, and the NPU achieves only ~12% of its own peak. The GPU runs the
   same OQ8 kernel at a small fraction of the NPU's per-matmul time.
3. **No bandwidth advantage.** It's a unified-memory APU — NPU and GPU share the same
   LPDDR5. The NPU's achieved weight BW (~4 GB/s) is *worse* than the GPU's, and the
   OQ8 matmul is weight-bandwidth-bound at decode.
4. **Decode (the dominant mode) is the worst case.** The 16-core design requires
   N ≥ n·cols = 128, so it **cannot run B=1 at all**; the single-core path at B=1 is
   dispatch-floor + full-weight-stream per token (thousands of µs/matmul). The GPU does
   an entire 1.5B decode step in ~16 ms (~62 tok/s). No contest.

## The only scenario with upside (and why it's not worth it now)

NPU value would require **async prefill offload** — the NPU running *concurrently* with
the GPU on different prefill matmuls to add aggregate throughput. That needs all of:
(a) non-blocking/async NPU dispatch infrastructure (does not exist — dispatch is
synchronous today), (b) a fused per-group-rescale epilogue kernel (custom `.cc`, not
stock `mm`), (c) on-tile int4 unpack for OQ+ to realize its DMA advantage, and
(d) a scheduler that splits matmuls across NPU+GPU. High effort for marginal upside:
prefill is already fast, and **decode — the actual bottleneck — gets no help** because
the NPU can't run small-batch GEMV competitively. Not recommended.

## Follow-ups (if revisited)

- Exact GPU OQ8 microbench for the same shape (`bench_batched_gemm.rs` family) to
  replace the reasoned GPU comparison with a measured one. (Conclusion is robust
  without it given the synchronous-dispatch and shared-memory facts above.)
- Toolchain notes for reproduction live in memory `project-npu-toolchain-this-box`:
  the IRON API drift fix and the user-space boost `LD_LIBRARY_PATH` workaround are
  required for any NPU build/run on this box.

## Reproduction

```
# correctness
python tools/npu/test_oq_gemm_npu.py --wbits 8 --M 64 --K 256 --B 64
python tools/npu/test_oq_gemm_npu.py --wbits 4 --M 64 --K 256 --B 64
# single-core perf
python tools/npu/bench_oq_gemm_npu.py --M 1536 --K 8960 --B 32,64,128,256
# whole-array perf (upstream example, env per memory note)
python whole_array.py -M 1536 -K 8960 -N 256 -m 64 -k 64 -n 32 \
    --n-aie-cols 4 --dtype_in i8 --dtype_out i32 --b-col-maj 1 --warmup 3 --iters 10
```
All NPU runs need: `LD_LIBRARY_PATH=~/.cache/hipfire-npu-deps/lib:/opt/xilinx/xrt/lib`
(the `oq_*` scripts bootstrap this internally).
