#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Bench the OQ8/OQ+ int8 GEMM on the XDNA1 NPU across a batch sweep.

Measures the raw int8·int8→int32 contraction (one dispatch over full K — the
NPU compute the OQ8 path needs; the per-group f32 rescale is cheap host/epilogue
arithmetic excluded here to isolate the matmul). Reports NPU hardware time,
end-to-end wall time, achieved int8 GFLOPS, and effective weight-DMA bandwidth
(M·K bytes streamed from shared LPDDR5 per matmul).

This is the go/no-go probe: prefill (B≫1) amortizes the weight DMA + ~180 µs
dispatch floor across B tokens; decode (B=1) does not.

Usage:
    python tools/npu/bench_oq_gemm_npu.py --M 1536 --K 8960 --B 32,64,128,256
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oq_gemm_design as design  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1536, help="weight rows / output dim")
    ap.add_argument("--K", type=int, default=8960, help="contraction dim (mult of 256)")
    ap.add_argument("--B", type=str, default="32,64,128,256", help="comma batch sweep")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    M, K = args.M, args.K
    Bs = [int(x) for x in args.B.split(",")]
    print(f"[bench_oq_gemm] device={design.NPU_DEVICE}  M={M} K={K} groups={K // 256}")
    print(f"  weight bytes (int8 M·K) = {M * K / 1e6:.2f} MB streamed per matmul")
    print()
    print(f"  {'B':>5} {'tile(m,k,n)':>14} {'npu_us':>9} {'e2e_us':>9} {'GFLOP/s':>8} {'wBW GB/s':>9} {'us/token':>9}")
    rng = np.random.default_rng(0)
    W = rng.integers(-64, 64, size=(M, K), dtype=np.int8)
    for B in Bs:
        X = rng.integers(-64, 64, size=(B, K), dtype=np.int8)
        _C, bench, tile = design.bench_npu(W, X, warmup=args.warmup, iters=args.iters)
        npu_us = bench.npu.avg_us
        e2e_us = bench.e2e.avg_us
        flops = 2.0 * M * K * B
        gflops = flops / (npu_us * 1e3)
        wbw = (M * K) / (npu_us * 1e3)  # bytes/us = GB/s (int8 weight stream)
        us_tok = e2e_us / B
        print(f"  {B:>5} {str(tile):>14} {npu_us:>9.1f} {e2e_us:>9.1f} {gflops:>8.1f} {wbw:>9.1f} {us_tok:>9.1f}")


if __name__ == "__main__":
    main()
