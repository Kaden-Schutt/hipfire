#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Correctness and benchmark test for the fused NPU headnorm + RoPE kernel.

Tests: for each head, apply per-head RMSNorm (with shared weight) then
       RoPE rotation, all in a single NPU kernel invocation.

Buffer order for rt.run: [t_in, t_out, t_packed]
  t_packed = np.concatenate([weight, cs])  — shape [head_dim + n_rot]

Usage:
    python tools/npu/test_headnorm_rope_npu.py \\
        --n-heads 8 --n-kv-heads 2 --head-dim 256

Build xclbins first:
    python tools/npu/build_qwen35_headnorm_rope.py --n-heads 8 --n-kv-heads 2 --head-dim 256
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


def headnorm_rope_ref(x: np.ndarray, weight: np.ndarray, cs: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """Float32 reference: per-head RMSNorm then half-split RoPE."""
    n_rot = head_dim // 4
    n_rot2 = n_rot // 2

    x32 = x.astype(np.float32).reshape(n_heads, head_dim)
    w32 = weight.astype(np.float32)
    cs32 = cs.astype(np.float32)
    cos_v = cs32[:n_rot2]
    sin_v = cs32[n_rot2:]

    out = np.empty_like(x32)
    for h in range(n_heads):
        head = x32[h]
        # RMSNorm
        rms = np.sqrt(np.mean(head**2) + 1e-5)
        normed = head * w32 / rms
        # RoPE — half-split: pairs at (i, i+n_rot2)
        out_h = normed.copy()
        out_h[:n_rot2] = normed[:n_rot2] * cos_v - normed[n_rot2:n_rot] * sin_v
        out_h[n_rot2:n_rot] = normed[n_rot2:n_rot] * cos_v + normed[:n_rot2] * sin_v
        out[h] = out_h

    return out.reshape(-1).astype(bfloat16)


def run_test(
    label: str,
    xclbin_path: Path,
    instr_path: Path,
    n_heads: int,
    head_dim: int,
    weight: np.ndarray,
    cs: np.ndarray,
    warmup: int = 20,
    timed: int = 200,
    atol: float = 0.02,
    rtol: float = 0.02,
) -> dict:
    n_total = n_heads * head_dim
    n_rot = head_dim // 4

    rng = np.random.default_rng(7)
    x = rng.uniform(-2.0, 2.0, n_total).astype(bfloat16)

    ref = headnorm_rope_ref(x, weight, cs, n_heads, head_dim)

    packed = np.concatenate([weight, cs])  # [head_dim + n_rot]
    assert packed.shape == (head_dim + n_rot,), f"packed shape mismatch: {packed.shape}"

    t_in = XRTTensor(x, dtype=bfloat16, device="cpu")
    t_out = XRTTensor((n_total,), dtype=bfloat16, device="cpu")
    t_packed = XRTTensor(packed, dtype=bfloat16, device="cpu")

    npu_kernel = NPUKernel(
        xclbin_path=xclbin_path,
        insts_path=instr_path,
        kernel_name=KERNEL_NAME,
    )
    rt = XRTHostRuntime()
    handle = rt.load(npu_kernel)
    print(f"  [{label}] hw_context: OK")

    # Buffer order: [t_in, t_out, t_packed]  (tiled input, output, tensor param)
    result = rt.run(handle, [t_in, t_out, t_packed])
    if not result.is_success():
        raise RuntimeError(f"NPU kernel failed: {result.ret}")
    t_out.to("cpu")
    npu_out = t_out.numpy().astype(bfloat16)

    ref32 = ref.astype(np.float32)
    npu32 = npu_out.astype(np.float32)
    abs_err = np.abs(ref32 - npu32)
    max_abs = float(abs_err.max())
    mean_abs = float(abs_err.mean())
    max_rel = float((abs_err / (np.abs(ref32) + 1e-6)).max())
    pass_ = bool((abs_err <= atol + rtol * np.abs(ref32)).all())

    print(
        f"  [{label}] correctness: max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} "
        f"max_rel={max_rel:.4f}  → {'PASS' if pass_ else 'FAIL'}"
    )
    if not pass_:
        flat_in = x.astype(np.float32)
        flat_ref = ref32
        flat_npu = npu32
        failing = np.argwhere(abs_err > atol + rtol * np.abs(flat_ref)).flatten()
        for idx in failing[:8]:
            h, i = divmod(idx, head_dim)
            print(
                f"    [h={h}, i={i}] in={flat_in[idx]:.4f} "
                f"ref={flat_ref[idx]:.5f} npu={flat_npu[idx]:.5f} "
                f"err={float(abs_err[idx]):.5f}"
            )
        raise AssertionError(f"[{label}] {len(failing)} elements exceed tolerance")

    for _ in range(warmup):
        rt.run(handle, [t_in, t_out, t_packed])

    npu_times_us = []
    wall_times_us = []
    for _ in range(timed):
        t0 = time.perf_counter()
        res = rt.run(handle, [t_in, t_out, t_packed])
        t1 = time.perf_counter()
        wall_times_us.append((t1 - t0) * 1e6)
        if hasattr(res, "npu_time") and res.npu_time:
            npu_times_us.append(res.npu_time / 1e3)

    total_bytes = 2 * n_total * 2  # in + out, bf16
    wall_mean = float(np.mean(wall_times_us))
    wall_p50 = float(np.percentile(wall_times_us, 50))
    wall_p99 = float(np.percentile(wall_times_us, 99))
    bw_gb_s = total_bytes / (wall_mean * 1e-6) / 1e9

    npu_mean = npu_p50 = npu_p99 = float("nan")
    if npu_times_us:
        npu_mean = float(np.mean(npu_times_us))
        npu_p50 = float(np.percentile(npu_times_us, 50))
        npu_p99 = float(np.percentile(npu_times_us, 99))
        print(f"  [{label}] npu_time: mean={npu_mean:.0f}µs p50={npu_p50:.0f}µs p99={npu_p99:.0f}µs")
    else:
        print(f"  [{label}] npu_time: (not reported)")

    print(f"  [{label}] wall:     mean={wall_mean:.0f}µs p50={wall_p50:.0f}µs p99={wall_p99:.0f}µs")
    print(f"  [{label}] BW:       {bw_gb_s:.2f} GB/s  (2×{n_total * 2}B / wall_mean)")

    return dict(
        label=label,
        n_heads=n_heads,
        head_dim=head_dim,
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


def make_cs(n_rot: int, freq_base: float = 500000.0, pos: int = 1) -> np.ndarray:
    """Pre-compute [cos_0..cos_{n_rot2-1}, sin_0..sin_{n_rot2-1}] for a single position."""
    n_rot2 = n_rot // 2
    freqs = np.array([1.0 / (freq_base ** (2 * i / n_rot)) for i in range(n_rot2)], dtype=np.float32)
    angles = pos * freqs
    cs = np.concatenate([np.cos(angles), np.sin(angles)]).astype(bfloat16)
    return cs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-heads", type=int, required=True)
    parser.add_argument("--n-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--xclbin-dir", type=Path, default=REPO_ROOT / "target/npu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()

    head_dim = args.head_dim
    n_rot = head_dim // 4

    # Shared weight (all-ones in test — non-trivial norms tested by the values themselves)
    rng = np.random.default_rng(99)
    weight = rng.uniform(0.5, 1.5, head_dim).astype(bfloat16)
    # cos/sin for pos=1
    cs = make_cs(n_rot)

    print(f"=== Fused HeadNorm+RoPE NPU Test: head_dim={head_dim} n_rot={n_rot} ===")

    # Q xclbin
    q_xclbin = args.xclbin_dir / f"qwen35-headnorm-rope-q-{args.n_heads}h{head_dim}d.xclbin"
    q_instr = args.xclbin_dir / f"qwen35-headnorm-rope-q-{args.n_heads}h{head_dim}d-instr.bin"
    for p in [q_xclbin, q_instr]:
        if not p.exists():
            sys.exit(
                f"{p} not found.\nRun: python tools/npu/build_qwen35_headnorm_rope.py "
                f"--n-heads {args.n_heads} --n-kv-heads {args.n_kv_heads} --head-dim {head_dim}"
            )
    print(f"\n--- Q ({args.n_heads} heads) ---")
    run_test("q", q_xclbin, q_instr, args.n_heads, head_dim, weight, cs, args.warmup, args.timed, args.atol, args.rtol)

    # K xclbin
    k_xclbin = args.xclbin_dir / f"qwen35-headnorm-rope-k-{args.n_kv_heads}h{head_dim}d.xclbin"
    k_instr = args.xclbin_dir / f"qwen35-headnorm-rope-k-{args.n_kv_heads}h{head_dim}d-instr.bin"
    for p in [k_xclbin, k_instr]:
        if not p.exists():
            sys.exit(
                f"{p} not found.\nRun: python tools/npu/build_qwen35_headnorm_rope.py "
                f"--n-heads {args.n_heads} --n-kv-heads {args.n_kv_heads} --head-dim {head_dim}"
            )
    print(f"\n--- K ({args.n_kv_heads} heads) ---")
    run_test(
        "k", k_xclbin, k_instr, args.n_kv_heads, head_dim, weight, cs, args.warmup, args.timed, args.atol, args.rtol
    )

    print("\n=== PASS ===")


if __name__ == "__main__":
    main()
