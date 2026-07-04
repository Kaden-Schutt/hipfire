#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Correctness + bench test for the NPU weighted RMSNorm kernel.

Runs rms_norm_weighted_bf16 on the NPU and compares against a numpy reference.
Requires the xclbin and instr.bin to already be built:
    python tools/npu/build_qwen35_rmsnorm.py --hidden-size 1536

Usage:
    python tools/npu/test_rmsnorm_npu.py --hidden-size 1536
    python tools/npu/test_rmsnorm_npu.py --hidden-size 3584
    python tools/npu/test_rmsnorm_npu.py --hidden-size 1536 --xclbin target/npu/qwen35-rmsnorm-1536.xclbin

Uses XRTHostRuntime from mlir_aie (bypasses aie.xrt.XCLBin which asserts
"RyzenAI-Phoenix" but XRT 2.25 reports "RyzenAI-npu1").
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

# ── pyxrt bootstrap ─────────────────────────────────────────────────────────
_XRT_PYTHON = "/opt/xilinx/xrt/python"
_XRT_LIB = "/opt/xilinx/xrt/lib"
if _XRT_PYTHON not in sys.path:
    sys.path.insert(0, _XRT_PYTHON)
ctypes.CDLL(os.path.join(_XRT_LIB, "libxrt_coreutil.so.2"), mode=ctypes.RTLD_GLOBAL)

_VENV = Path.home() / ".venv" / "lib"
for p in _VENV.glob("python*/site-packages"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pyxrt
from ml_dtypes import bfloat16
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

KERNEL_NAME = "MLIR_AIE"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EPS = 1e-5


def rmsnorm_ref(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    w32 = weight.astype(np.float32)
    rms = np.sqrt(np.mean(x32**2) + EPS)
    return ((x32 / rms) * w32).astype(bfloat16)


def run_test(
    xclbin_path: Path,
    instr_path: Path,
    hidden_size: int,
    n_warmup: int = 20,
    n_timed: int = 200,
    seed: int = 42,
    atol: float = 0.02,
    rtol: float = 0.02,
) -> bool:
    print(f"[test_rmsnorm_npu] hidden_size={hidden_size}")
    print(f"  xclbin: {xclbin_path}")
    print(f"  instr:  {instr_path}")

    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, hidden_size).astype(bfloat16)
    weight = rng.uniform(0.5, 1.5, hidden_size).astype(bfloat16)
    ref = rmsnorm_ref(x, weight)

    t_in = XRTTensor(x, dtype=bfloat16, device="cpu")
    t_w = XRTTensor(weight, dtype=bfloat16, device="cpu")
    t_out = XRTTensor((hidden_size,), dtype=bfloat16, device="cpu")

    npu_kernel = NPUKernel(
        xclbin_path=xclbin_path,
        insts_path=instr_path,
        kernel_name=KERNEL_NAME,
    )
    rt = XRTHostRuntime()
    handle = rt.load(npu_kernel)
    print(f"  hw_context: OK")

    # Warmup
    for _ in range(n_warmup):
        rt.run(handle, [t_in, t_w, t_out])

    # Timed
    npu_times = []
    wall_times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        result = rt.run(handle, [t_in, t_w, t_out])
        wall_times.append(time.perf_counter() - t0)
        if not result.is_success():
            raise RuntimeError(f"NPU kernel failed: {result.ret}")
        npu_times.append(result.npu_time)

    npu_mean = int(np.mean(npu_times)) // 1000
    npu_p50 = int(np.percentile(npu_times, 50)) // 1000
    npu_p99 = int(np.percentile(npu_times, 99)) // 1000
    wall_mean_us = np.mean(wall_times) * 1e6
    data_bytes = 3 * hidden_size * 2
    bw_gbs = data_bytes / (np.mean(npu_times) * 1e-9) / 1e9 if np.mean(npu_times) > 0 else 0.0

    print(f"  warmup={n_warmup} timed={n_timed}")
    print(f"  npu mean={npu_mean} µs  p50={npu_p50} µs  p99={npu_p99} µs")
    print(f"  wall mean={wall_mean_us:.0f} µs  BW={bw_gbs:.2f} GB/s")

    # Sync output and check
    t_out.to("cpu")
    npu_out = t_out.numpy().astype(bfloat16)

    ref32 = ref.astype(np.float32)
    npu32 = npu_out.astype(np.float32)
    abs_err = np.abs(ref32 - npu32)
    rel_err = abs_err / (np.abs(ref32) + 1e-6)

    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())
    mean_abs = float(abs_err.mean())
    pass_ = bool((abs_err <= atol + rtol * np.abs(ref32)).all())

    print(f"  max_abs_err={max_abs:.5f}  mean_abs_err={mean_abs:.5f}  max_rel_err={max_rel:.4f}")

    if pass_:
        print(f"  PASS (atol={atol}, rtol={rtol})")
    else:
        failing = np.where(abs_err > atol + rtol * np.abs(ref32))[0]
        print(f"  FAIL — {len(failing)} elements exceed tolerance")
        for i in failing[:8]:
            print(
                f"    [{i}] x={float(x[i]):.4f} w={float(weight[i]):.4f} "
                f"ref={float(ref[i]):.4f} npu={float(npu_out[i]):.4f} "
                f"err={float(abs_err[i]):.4f}"
            )

    return pass_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden-size", type=int, default=1536, help="Model hidden size (1536 for 1.5B, 3584 for 7B)")
    parser.add_argument("--xclbin", type=Path)
    parser.add_argument("--instr", type=Path)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-timed", type=int, default=200)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()

    npu_dir = REPO_ROOT / "target" / "npu"
    xclbin = args.xclbin or npu_dir / f"qwen35-rmsnorm-{args.hidden_size}.xclbin"
    instr = args.instr or npu_dir / f"qwen35-rmsnorm-{args.hidden_size}-instr.bin"

    if not xclbin.exists():
        sys.exit(
            f"xclbin not found: {xclbin}\nRun: python tools/npu/build_qwen35_rmsnorm.py --hidden-size {args.hidden_size}"
        )
    if not instr.exists():
        sys.exit(f"instr.bin not found: {instr}")

    ok = run_test(
        xclbin, instr, args.hidden_size, n_warmup=args.n_warmup, n_timed=args.n_timed, atol=args.atol, rtol=args.rtol
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
