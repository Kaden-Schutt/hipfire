#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Correctness and benchmark test for the NPU BF16 row softmax kernel.

Tests: output[h, i] = softmax(input[h, :])[i]  (per-head, numerically stable)

Usage:
    python tools/npu/test_softmax_npu.py --n-heads 8 --ctx-len 256

Build xclbins first:
    python tools/npu/build_qwen35_softmax.py --n-heads 8 --ctx-lens 64,128,256,512
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path

_XRT_PYTHON = "/opt/xilinx/xrt/python"
_XRT_LIB = "/opt/xilinx/xrt/lib"
if _XRT_PYTHON not in sys.path:
    sys.path.insert(0, _XRT_PYTHON)
ctypes.CDLL(f"{_XRT_LIB}/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)

_VENV = Path.home() / ".venv" / "lib"
for p in _VENV.glob("python*/site-packages"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from ml_dtypes import bfloat16
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

KERNEL_NAME = "MLIR_AIE"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def softmax_ref(x: np.ndarray) -> np.ndarray:
    """Float32 reference: per-row softmax with max subtraction."""
    x32 = x.astype(np.float32)
    x32 -= x32.max(axis=-1, keepdims=True)
    e = np.exp(x32)
    return (e / e.sum(axis=-1, keepdims=True)).astype(bfloat16)


def run_test(
    xclbin_path: Path,
    instr_path: Path,
    n_heads: int,
    ctx_len: int,
    warmup: int = 20,
    timed: int = 200,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> dict:
    total = n_heads * ctx_len
    total_bytes = total * 2  # bf16

    rng = np.random.default_rng(42)
    # Simulate attention scores: small range after 1/sqrt(head_dim) scaling
    scores = rng.uniform(-4.0, 4.0, (n_heads, ctx_len)).astype(bfloat16)
    # Set last quarter to -inf to simulate causal masking
    n_valid = ctx_len * 3 // 4
    scores[:, n_valid:] = bfloat16(-1e9)

    ref = softmax_ref(scores)  # [n_heads, ctx_len]

    flat_in = scores.reshape(-1)
    t_in = XRTTensor(flat_in, dtype=bfloat16, device="cpu")
    t_out = XRTTensor((total,), dtype=bfloat16, device="cpu")

    npu_kernel = NPUKernel(
        xclbin_path=xclbin_path,
        insts_path=instr_path,
        kernel_name=KERNEL_NAME,
    )
    rt = XRTHostRuntime()
    handle = rt.load(npu_kernel)
    print(f"  hw_context: OK")

    # Buffer order: [t_in, t_out] (no tensor params)
    result = rt.run(handle, [t_in, t_out])
    if not result.is_success():
        raise RuntimeError(f"NPU kernel failed: {result.ret}")
    t_out.to("cpu")
    npu_out = t_out.numpy().reshape(n_heads, ctx_len).astype(bfloat16)

    ref32 = ref.astype(np.float32)
    npu32 = npu_out.astype(np.float32)
    abs_err = np.abs(ref32 - npu32)
    max_abs = float(abs_err.max())
    mean_abs = float(abs_err.mean())
    max_rel = float((abs_err / (np.abs(ref32) + 1e-6)).max())
    pass_ = bool((abs_err <= atol + rtol * np.abs(ref32)).all())

    # Verify masked positions are ~0
    masked_max = float(npu_out[:, n_valid:].astype(np.float32).max())

    print(
        f"  correctness: max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} "
        f"max_rel={max_rel:.4f}  → {'PASS' if pass_ else 'FAIL'}"
    )
    print(f"  masked positions max: {masked_max:.2e}  (should be ~0)")
    if not pass_:
        failing = np.argwhere(abs_err > atol + rtol * np.abs(ref32))
        for idx in failing[:8]:
            h, i = idx
            print(
                f"    [h={h}, i={i}] in={float(scores[h, i]):.4f} "
                f"ref={float(ref[h, i]):.5f} npu={float(npu_out[h, i]):.5f} "
                f"err={float(abs_err[h, i]):.5f}"
            )
        raise AssertionError(f"{len(failing)} elements exceed tolerance")

    for _ in range(warmup):
        rt.run(handle, [t_in, t_out])

    npu_times_us = []
    wall_times_us = []
    for _ in range(timed):
        t0 = time.perf_counter()
        res = rt.run(handle, [t_in, t_out])
        t1 = time.perf_counter()
        wall_times_us.append((t1 - t0) * 1e6)
        if hasattr(res, "npu_time") and res.npu_time:
            npu_times_us.append(res.npu_time / 1e3)

    wall_mean = float(np.mean(wall_times_us))
    wall_p50 = float(np.percentile(wall_times_us, 50))
    wall_p99 = float(np.percentile(wall_times_us, 99))
    bw_gb_s = (2 * total_bytes) / (wall_mean * 1e-6) / 1e9  # in + out

    npu_mean = npu_p50 = npu_p99 = float("nan")
    if npu_times_us:
        npu_mean = float(np.mean(npu_times_us))
        npu_p50 = float(np.percentile(npu_times_us, 50))
        npu_p99 = float(np.percentile(npu_times_us, 99))
        print(f"  npu_time: mean={npu_mean:.0f}µs p50={npu_p50:.0f}µs p99={npu_p99:.0f}µs")
    else:
        print(f"  npu_time: (not reported)")

    print(f"  wall:     mean={wall_mean:.0f}µs p50={wall_p50:.0f}µs p99={wall_p99:.0f}µs")
    print(f"  BW:       {bw_gb_s:.2f} GB/s  (2×{total_bytes}B / wall_mean)")

    return dict(
        n_heads=n_heads,
        ctx_len=ctx_len,
        npu_mean=npu_mean,
        npu_p50=npu_p50,
        npu_p99=npu_p99,
        wall_mean=wall_mean,
        wall_p50=wall_p50,
        wall_p99=wall_p99,
        bw_gb_s=bw_gb_s,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-heads", type=int, required=True)
    parser.add_argument("--ctx-len", type=int, required=True, help="Context length (must match a built xclbin)")
    parser.add_argument("--xclbin-dir", type=Path, default=REPO_ROOT / "target/npu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    ctx_len = args.ctx_len
    xclbin = args.xclbin_dir / f"qwen35-softmax-{args.n_heads}h{ctx_len}ctx.xclbin"
    instr = args.xclbin_dir / f"qwen35-softmax-{args.n_heads}h{ctx_len}ctx-instr.bin"

    for p in [xclbin, instr]:
        if not p.exists():
            sys.exit(
                f"{p} not found.\nRun: python tools/npu/build_qwen35_softmax.py "
                f"--n-heads {args.n_heads} --ctx-lens {ctx_len}"
            )

    print(f"=== Softmax NPU Test: n_heads={args.n_heads} ctx_len={ctx_len} ===")
    print(f"  xclbin: {xclbin}")
    print(f"  instr:  {instr}")

    run_test(xclbin, instr, args.n_heads, ctx_len, args.warmup, args.timed, args.atol, args.rtol)
    print("\n=== PASS ===")


if __name__ == "__main__":
    main()
