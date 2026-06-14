#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Correctness test for the NPU SwiGLU kernel.

Runs silu_mul_bf16 on the NPU and compares against a numpy reference.
Requires the xclbin and instr.bin to already be built:
    python tools/npu/build_qwen35_swiglu.py --hidden-size 8960

Usage:
    python tools/npu/test_swiglu_npu.py
    python tools/npu/test_swiglu_npu.py --hidden-size 8960 --xclbin target/npu/qwen35-swiglu-8960.xclbin

Uses XRTHostRuntime from mlir_aie (bypasses aie.xrt.XCLBin which asserts
"RyzenAI-Phoenix" but XRT 2.25 reports "RyzenAI-npu1").
"""

import argparse
import ctypes
import os
import sys
from pathlib import Path

# ── pyxrt bootstrap ─────────────────────────────────────────────────────────
_XRT_PYTHON = "/opt/xilinx/xrt/python"
_XRT_LIB    = "/opt/xilinx/xrt/lib"
if _XRT_PYTHON not in sys.path:
    sys.path.insert(0, _XRT_PYTHON)
# libxrt_coreutil provides a weak vtable symbol that must be loaded before
# libxrt_core to avoid an "undefined symbol" crash under XRT 2.25.
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


def sigmoid_ref(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-x32))).astype(bfloat16)


def silu_mul_ref(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return (gate.astype(np.float32) * sigmoid_ref(gate).astype(np.float32)
            * up.astype(np.float32)).astype(bfloat16)


def run_test(xclbin_path: Path, instr_path: Path, hidden_size: int,
             seed: int = 42, atol: float = 0.02, rtol: float = 0.02) -> bool:
    print(f"[test_swiglu_npu] hidden_size={hidden_size}")
    print(f"  xclbin: {xclbin_path}")
    print(f"  instr:  {instr_path}")

    rng = np.random.default_rng(seed)
    # Values in [-2, 2]: well within the range where the tanh approximation of
    # sigmoid is accurate and bfloat16 doesn't catastrophically round.
    gate = rng.uniform(-2.0, 2.0, hidden_size).astype(bfloat16)
    up   = rng.uniform(-2.0, 2.0, hidden_size).astype(bfloat16)
    ref  = silu_mul_ref(gate, up)

    # ── Build XRTTensors ────────────────────────────────────────────────────
    # XRTHostRuntime.run() expects XRTTensor instances, allocated with host_only
    # flags + group_id=0 (the "magic 0" for RyzenAI DMA-accessible buffers).
    t_gate = XRTTensor(gate, dtype=bfloat16, device="cpu")
    t_up   = XRTTensor(up,   dtype=bfloat16, device="cpu")
    t_out  = XRTTensor((hidden_size,), dtype=bfloat16, device="cpu")

    # ── Run on NPU ──────────────────────────────────────────────────────────
    npu_kernel = NPUKernel(
        xclbin_path=xclbin_path,
        insts_path=instr_path,
        kernel_name=KERNEL_NAME,
    )
    rt = XRTHostRuntime()
    handle = rt.load(npu_kernel)
    print(f"  hw_context: OK")

    result = rt.run(handle, [t_gate, t_up, t_out])
    if not result.is_success():
        raise RuntimeError(f"NPU kernel failed: {result.ret}")
    print(f"  npu_time: {result.npu_time / 1000:.1f} µs")

    # Sync output back to CPU and read
    t_out.to("cpu")
    npu_out = t_out.numpy().astype(bfloat16)

    # ── Compare ─────────────────────────────────────────────────────────────
    ref32 = ref.astype(np.float32)
    npu32 = npu_out.astype(np.float32)
    abs_err = np.abs(ref32 - npu32)
    rel_err = abs_err / (np.abs(ref32) + 1e-6)

    max_abs  = float(abs_err.max())
    max_rel  = float(rel_err.max())
    mean_abs = float(abs_err.mean())
    pass_    = bool((abs_err <= atol + rtol * np.abs(ref32)).all())

    print(f"  max_abs_err={max_abs:.5f}  mean_abs_err={mean_abs:.5f}  max_rel_err={max_rel:.4f}")

    if pass_:
        print(f"  PASS (atol={atol}, rtol={rtol})")
    else:
        failing = np.where(abs_err > atol + rtol * np.abs(ref32))[0]
        print(f"  FAIL — {len(failing)} elements exceed tolerance")
        for i in failing[:8]:
            print(f"    [{i}] gate={float(gate[i]):.4f} up={float(up[i]):.4f} "
                  f"ref={float(ref[i]):.4f} npu={float(npu_out[i]):.4f} "
                  f"err={float(abs_err[i]):.4f}")

    return pass_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden-size", type=int, default=8960)
    parser.add_argument("--xclbin", type=Path)
    parser.add_argument("--instr", type=Path)
    parser.add_argument("--atol", type=float, default=0.02,
                        help="Absolute tolerance (default 0.02; bfloat16 unit ulp ~0.008 at 1.0)")
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()

    npu_dir = REPO_ROOT / "target" / "npu"
    xclbin = args.xclbin or npu_dir / f"qwen35-swiglu-{args.hidden_size}.xclbin"
    instr  = args.instr  or npu_dir / f"qwen35-swiglu-{args.hidden_size}-instr.bin"

    if not xclbin.exists():
        sys.exit(f"xclbin not found: {xclbin}\nRun: python tools/npu/build_qwen35_swiglu.py --hidden-size {args.hidden_size}")
    if not instr.exists():
        sys.exit(f"instr.bin not found: {instr}")

    ok = run_test(xclbin, instr, args.hidden_size, atol=args.atol, rtol=args.rtol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
