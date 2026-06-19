#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Correctness and benchmark test for the NPU RoPE rotation kernel.

Runs Q and K configs: checks against a float32 reference, then times
200 iterations and prints npu_time, wall time, and effective BW.

Usage:
    python tools/npu/test_rope_npu.py \\
        --n-heads 8 --n-kv-heads 2 --head-dim 256 --n-rot 64

The test loads pre-built xclbins from --xclbin-dir (default: target/npu/).
Build them first with:
    python tools/npu/build_qwen35_rope.py --n-heads 8 --n-kv-heads 2 --head-dim 256 --n-rot 64
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path
from typing import Any

# ── pyxrt bootstrap (must happen before any aie imports) ────────────────────
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


def make_cs_buf(n_rot: int, pos: int, freq_base: float = 500000.0) -> np.ndarray:
    """Build [cos_0..cos_{n_rot/2-1}, sin_0..sin_{n_rot/2-1}] at token position pos."""
    n_rot2 = n_rot // 2
    cos_vals = np.zeros(n_rot2, dtype=np.float32)
    sin_vals = np.zeros(n_rot2, dtype=np.float32)
    for i in range(n_rot2):
        freq = 1.0 / (freq_base ** (2.0 * i / n_rot))
        angle = pos * freq
        cos_vals[i] = np.cos(angle)
        sin_vals[i] = np.sin(angle)
    return np.concatenate([cos_vals, sin_vals]).astype(bfloat16)


def reference_rope(x_bf16: np.ndarray, n_heads: int, head_dim: int,
                   n_rot: int, pos: int, freq_base: float = 500000.0) -> np.ndarray:
    """Float32 half-split RoPE reference."""
    n_rot2 = n_rot // 2
    x32 = x_bf16.astype(np.float32).reshape(n_heads, head_dim)
    for i in range(n_rot2):
        freq = 1.0 / (freq_base ** (2.0 * i / n_rot))
        angle = pos * freq
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        xi = x32[:, i].copy()
        yi = x32[:, i + n_rot2].copy()
        x32[:, i]          = xi * c - yi * s
        x32[:, i + n_rot2] = yi * c + xi * s
    return x32.reshape(-1).astype(bfloat16)


def run_one(label: str, n_heads: int, head_dim: int, n_rot: int,
            xclbin_path: Path, instr_path: Path,
            warmup: int = 20, timed: int = 200,
            atol: float = 0.02, rtol: float = 0.02,
            pos: int = 1, freq_base: float = 500000.0) -> dict[str, Any]:
    total_elems = n_heads * head_dim
    total_bytes = total_elems * 2  # bfloat16

    print(f"\n[{label}] n_heads={n_heads} head_dim={head_dim} n_rot={n_rot} "
          f"total_elems={total_elems}")
    print(f"  xclbin: {xclbin_path}")
    print(f"  instr:  {instr_path}")

    rng = np.random.default_rng(42)
    x_bf16 = rng.standard_normal(total_elems).astype(np.float32).astype(bfloat16)
    cs_bf16 = make_cs_buf(n_rot, pos, freq_base)
    ref = reference_rope(x_bf16, n_heads, head_dim, n_rot, pos, freq_base)

    t_in  = XRTTensor(x_bf16,  dtype=bfloat16, device="cpu")
    t_cs  = XRTTensor(cs_bf16, dtype=bfloat16, device="cpu")
    t_out = XRTTensor((total_elems,), dtype=bfloat16, device="cpu")

    npu_kernel = NPUKernel(
        xclbin_path=xclbin_path,
        insts_path=instr_path,
        kernel_name=KERNEL_NAME,
    )
    rt = XRTHostRuntime()
    handle = rt.load(npu_kernel)
    print(f"  hw_context: OK")

    # Buffer order matches MLIR runtime_sequence: (in0, out, param0)
    # i.e. [input, output, cs_param] — output before cs param.
    result = rt.run(handle, [t_in, t_out, t_cs])
    if not result.is_success():
        raise RuntimeError(f"NPU kernel failed: {result.ret}")
    t_out.to("cpu")
    npu_out = t_out.numpy().astype(bfloat16)

    ref32 = ref.astype(np.float32)
    npu32 = npu_out.astype(np.float32)
    abs_err = np.abs(ref32 - npu32)
    max_abs  = float(abs_err.max())
    mean_abs = float(abs_err.mean())
    max_rel  = float((abs_err / (np.abs(ref32) + 1e-6)).max())
    pass_ = bool((abs_err <= atol + rtol * np.abs(ref32)).all())
    print(f"  correctness: max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} "
          f"max_rel={max_rel:.4f}  → {'PASS' if pass_ else 'FAIL'}")
    if not pass_:
        failing = np.where(abs_err > atol + rtol * np.abs(ref32))[0]
        for i in failing[:8]:
            print(f"    [{i}] ref={float(ref[i]):.4f} npu={float(npu_out[i]):.4f} "
                  f"err={float(abs_err[i]):.4f}")
        raise AssertionError(f"{label}: {len(failing)} elements exceed tolerance")

    # Warmup
    for _ in range(warmup):
        rt.run(handle, [t_in, t_out, t_cs])

    # Timed
    npu_times_us = []
    wall_times_us = []
    for _ in range(timed):
        t0 = time.perf_counter()
        res = rt.run(handle, [t_in, t_out, t_cs])
        t1 = time.perf_counter()
        wall_times_us.append((t1 - t0) * 1e6)
        if hasattr(res, "npu_time") and res.npu_time:
            npu_times_us.append(res.npu_time / 1e3)  # ns → µs

    wall_mean = float(np.mean(wall_times_us))
    wall_p50  = float(np.percentile(wall_times_us, 50))
    wall_p99  = float(np.percentile(wall_times_us, 99))
    bw_gb_s   = (3 * total_bytes) / (wall_mean * 1e-6) / 1e9

    if npu_times_us:
        npu_mean = float(np.mean(npu_times_us))
        npu_p50  = float(np.percentile(npu_times_us, 50))
        npu_p99  = float(np.percentile(npu_times_us, 99))
        print(f"  npu_time: mean={npu_mean:.0f}µs p50={npu_p50:.0f}µs p99={npu_p99:.0f}µs")
    else:
        npu_mean = float("nan")
        print(f"  npu_time: (not reported)")

    print(f"  wall:     mean={wall_mean:.0f}µs p50={wall_p50:.0f}µs p99={wall_p99:.0f}µs")
    print(f"  BW:       {bw_gb_s:.2f} GB/s  (3×{total_bytes}B / wall_mean)")

    return {
        "label": label, "npu_mean": npu_mean, "wall_mean": wall_mean,
        "npu_p50": npu_p50 if npu_times_us else float("nan"),
        "npu_p99": npu_p99 if npu_times_us else float("nan"),
        "wall_p50": wall_p50, "wall_p99": wall_p99,
        "bw_gb_s": bw_gb_s,
        "max_abs": max_abs, "mean_abs": mean_abs, "max_rel": max_rel,
    }


def test(n_heads: int, n_kv_heads: int, head_dim: int, n_rot: int,
         xclbin_dir: Path, warmup: int, timed: int) -> None:
    q_xclbin = xclbin_dir / f"qwen35-rope-q-{n_heads}h{head_dim}d.xclbin"
    q_instr  = xclbin_dir / f"qwen35-rope-q-{n_heads}h{head_dim}d-instr.bin"
    k_xclbin = xclbin_dir / f"qwen35-rope-k-{n_kv_heads}h{head_dim}d.xclbin"
    k_instr  = xclbin_dir / f"qwen35-rope-k-{n_kv_heads}h{head_dim}d-instr.bin"

    for p in [q_xclbin, q_instr, k_xclbin, k_instr]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Build first with:\n"
                f"  python tools/npu/build_qwen35_rope.py "
                f"--n-heads {n_heads} --n-kv-heads {n_kv_heads} "
                f"--head-dim {head_dim} --n-rot {n_rot}"
            )

    print(f"=== RoPE NPU Test: head_dim={head_dim} n_rot={n_rot} ===")

    rq = run_one("Q", n_heads,    head_dim, n_rot, q_xclbin, q_instr, warmup, timed)
    rk = run_one("K", n_kv_heads, head_dim, n_rot, k_xclbin, k_instr, warmup, timed)

    print("\n=== Summary ===")
    print(f"{'tensor':<6} {'n_heads':<7} {'npu_mean':>10} {'npu_p50':>8} {'npu_p99':>8} "
          f"{'wall_mean':>10} {'BW':>9} {'max_abs':>9} {'result'}")
    for r in [rq, rk]:
        nh = n_heads if r["label"] == "Q" else n_kv_heads
        print(f"{r['label']:<6} {nh:<7} "
              f"{r['npu_mean']:>9.0f}µs {r['npu_p50']:>7.0f}µs {r['npu_p99']:>7.0f}µs "
              f"{r['wall_mean']:>9.0f}µs {r['bw_gb_s']:>7.2f} GB/s "
              f"{r['max_abs']:>8.5f} PASS")
    print("\n=== PASS ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-heads",    type=int, required=True)
    parser.add_argument("--n-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim",   type=int, default=256)
    parser.add_argument("--n-rot",      type=int, default=64)
    parser.add_argument("--xclbin-dir", type=Path, default=REPO_ROOT / "target/npu")
    parser.add_argument("--warmup",     type=int, default=20)
    parser.add_argument("--timed",      type=int, default=200)
    args = parser.parse_args()
    test(args.n_heads, args.n_kv_heads, args.head_dim, args.n_rot,
         args.xclbin_dir, args.warmup, args.timed)


if __name__ == "__main__":
    main()
