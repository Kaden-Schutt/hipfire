#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Correctness and benchmark test for the NPU attention output gate kernel.

Tests: out[i] = sigmoid(gate[i]) * x[i]

Usage:
    python tools/npu/test_attn_gate_npu.py --n-heads 8 --head-dim 256

Build xclbin first:
    python tools/npu/build_qwen35_attn_gate.py --n-heads 8 --head-dim 256
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path

_XRT_PYTHON = "/opt/xilinx/xrt/python"
_XRT_LIB    = "/opt/xilinx/xrt/lib"
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


def sigmoid_ref(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(bfloat16)


def sigmoid_mul_ref(gate: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (sigmoid_ref(gate).astype(np.float32) * x.astype(np.float32)).astype(bfloat16)


def run_test(xclbin_path: Path, instr_path: Path, q_dim: int,
             warmup: int = 20, timed: int = 200,
             atol: float = 0.02, rtol: float = 0.02) -> dict:
    total_bytes = q_dim * 2

    rng = np.random.default_rng(42)
    gate = rng.uniform(-2.0, 2.0, q_dim).astype(bfloat16)
    x    = rng.uniform(-2.0, 2.0, q_dim).astype(bfloat16)
    ref  = sigmoid_mul_ref(gate, x)

    t_gate = XRTTensor(gate, dtype=bfloat16, device="cpu")
    t_x    = XRTTensor(x,    dtype=bfloat16, device="cpu")
    t_out  = XRTTensor((q_dim,), dtype=bfloat16, device="cpu")

    npu_kernel = NPUKernel(
        xclbin_path=xclbin_path,
        insts_path=instr_path,
        kernel_name=KERNEL_NAME,
    )
    rt = XRTHostRuntime()
    handle = rt.load(npu_kernel)
    print(f"  hw_context: OK")

    result = rt.run(handle, [t_gate, t_x, t_out])
    if not result.is_success():
        raise RuntimeError(f"NPU kernel failed: {result.ret}")
    t_out.to("cpu")
    npu_out = t_out.numpy().astype(bfloat16)

    ref32 = ref.astype(np.float32)
    npu32 = npu_out.astype(np.float32)
    abs_err  = np.abs(ref32 - npu32)
    max_abs  = float(abs_err.max())
    mean_abs = float(abs_err.mean())
    max_rel  = float((abs_err / (np.abs(ref32) + 1e-6)).max())
    pass_    = bool((abs_err <= atol + rtol * np.abs(ref32)).all())

    print(f"  correctness: max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} "
          f"max_rel={max_rel:.4f}  → {'PASS' if pass_ else 'FAIL'}")
    if not pass_:
        failing = np.where(abs_err > atol + rtol * np.abs(ref32))[0]
        for i in failing[:8]:
            print(f"    [{i}] gate={float(gate[i]):.4f} x={float(x[i]):.4f} "
                  f"ref={float(ref[i]):.4f} npu={float(npu_out[i]):.4f} "
                  f"err={float(abs_err[i]):.4f}")
        raise AssertionError(f"{len(failing)} elements exceed tolerance")

    for _ in range(warmup):
        rt.run(handle, [t_gate, t_x, t_out])

    npu_times_us = []
    wall_times_us = []
    for _ in range(timed):
        t0 = time.perf_counter()
        res = rt.run(handle, [t_gate, t_x, t_out])
        t1 = time.perf_counter()
        wall_times_us.append((t1 - t0) * 1e6)
        if hasattr(res, "npu_time") and res.npu_time:
            npu_times_us.append(res.npu_time / 1e3)

    wall_mean = float(np.mean(wall_times_us))
    wall_p50  = float(np.percentile(wall_times_us, 50))
    wall_p99  = float(np.percentile(wall_times_us, 99))
    bw_gb_s   = (3 * total_bytes) / (wall_mean * 1e-6) / 1e9

    npu_mean = npu_p50 = npu_p99 = float("nan")
    if npu_times_us:
        npu_mean = float(np.mean(npu_times_us))
        npu_p50  = float(np.percentile(npu_times_us, 50))
        npu_p99  = float(np.percentile(npu_times_us, 99))
        print(f"  npu_time: mean={npu_mean:.0f}µs p50={npu_p50:.0f}µs p99={npu_p99:.0f}µs")
    else:
        print(f"  npu_time: (not reported)")

    print(f"  wall:     mean={wall_mean:.0f}µs p50={wall_p50:.0f}µs p99={wall_p99:.0f}µs")
    print(f"  BW:       {bw_gb_s:.2f} GB/s  (3×{total_bytes}B / wall_mean)")

    return dict(q_dim=q_dim,
                npu_mean=npu_mean, npu_p50=npu_p50, npu_p99=npu_p99,
                wall_mean=wall_mean, wall_p50=wall_p50, wall_p99=wall_p99,
                bw_gb_s=bw_gb_s, max_abs=max_abs, mean_abs=mean_abs, max_rel=max_rel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-heads",  type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--xclbin-dir", type=Path, default=REPO_ROOT / "target/npu")
    parser.add_argument("--warmup",   type=int, default=20)
    parser.add_argument("--timed",    type=int, default=200)
    parser.add_argument("--atol",     type=float, default=0.02)
    parser.add_argument("--rtol",     type=float, default=0.02)
    args = parser.parse_args()

    q_dim    = args.n_heads * args.head_dim
    xclbin   = args.xclbin_dir / f"qwen35-attn-gate-{args.n_heads}h{args.head_dim}d.xclbin"
    instr    = args.xclbin_dir / f"qwen35-attn-gate-{args.n_heads}h{args.head_dim}d-instr.bin"

    for p in [xclbin, instr]:
        if not p.exists():
            sys.exit(
                f"{p} not found.\nRun: python tools/npu/build_qwen35_attn_gate.py "
                f"--n-heads {args.n_heads} --head-dim {args.head_dim}"
            )

    print(f"=== Attn Gate NPU Test: n_heads={args.n_heads} head_dim={args.head_dim} q_dim={q_dim} ===")
    print(f"  xclbin: {xclbin}")
    print(f"  instr:  {instr}")

    run_test(xclbin, instr, q_dim, args.warmup, args.timed, args.atol, args.rtol)
    print("\n=== PASS ===")


if __name__ == "__main__":
    main()
