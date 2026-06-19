#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Compile the NPU RoPE rotation kernel for Qwen3.5 Q and K tensors.

Produces four files (two per tensor, Q and K):
  - <out_dir>/qwen35-rope-q-{n_heads}h{head_dim}d.xclbin  / -instr.bin
  - <out_dir>/qwen35-rope-k-{n_kv_heads}h{head_dim}d.xclbin / -instr.bin

The kernel applies the half-split RoPE rotation (matching hipfire's GPU
rope_partial_halfsplit_f32) — pairs are at (d, d+n_rot/2) not (2d, 2d+1).
cos/sin for the current token position must be pre-computed by the caller
and packed into a cs buffer: [cos_0..cos_{n_rot/2-1}, sin_0..sin_{n_rot/2-1}].

Design: _transform_gen (single-core), tile_size=head_dim. The kernel is called
once per head; N_div_n = n_heads (Q) or n_kv_heads (K). The cs buffer is a
tensor param acquired once and reused for all head iterations.

Usage:
    python tools/npu/build_qwen35_rope.py \\
        --n-heads 8 --n-kv-heads 2 --head-dim 256 --n-rot 64
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# ── venv bootstrap ──────────────────────────────────────────────────────────
_VENV = Path.home() / ".venv" / "lib"
for p in _VENV.glob("python*/site-packages"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_XRT_BIN = Path("/opt/xilinx/xrt/bin")
if _XRT_BIN.is_dir() and str(_XRT_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_XRT_BIN) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from ml_dtypes import bfloat16
from aie.iron import ExternalFunction
from aie.iron.algorithms.transform import _transform_gen
from aie.iron.device import NPU1, NPU2
from aie.utils import set_current_device
from aie.utils.compile import compile_mlir_module, compile_external_kernel

SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_SRC = SCRIPT_DIR / "rope_rotate_bf16.cc"

_mlir_aie_pkg = next(
    (Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None
)
AIE_INCLUDE = _mlir_aie_pkg / "mlir_aie" / "include" if _mlir_aie_pkg else None

_NPU_DEVICES = {
    "npu1": (NPU1, "aie2"),
    "npu2": (NPU2, "aie2p"),
}

_NAME_TO_NPU = {
    "npu1":    "npu1",
    "Phoenix": "npu1",
    "npu4":    "npu2",
    "npu5":    "npu2",
    "npu6":    "npu2",
    "Strix":   "npu2",
    "Krackan": "npu2",
}


def detect_npu() -> str:
    import ctypes
    _xrt_lib = "/opt/xilinx/xrt/lib"
    ctypes.CDLL(f"{_xrt_lib}/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)
    _xrt_py = "/opt/xilinx/xrt/python"
    if _xrt_py not in sys.path:
        sys.path.insert(0, _xrt_py)
    import pyxrt
    device = pyxrt.device(0)
    name = device.get_info(pyxrt.xrt_info_device.name)
    for substr, key in _NAME_TO_NPU.items():
        if substr in name:
            return key
    raise RuntimeError(
        f"Cannot map device name {name!r} to a known NPU generation. "
        f"Pass --npu explicitly (npu1 or npu2)."
    )


def build_one(label: str, n_total: int, n_heads_label: int,
              head_dim: int, n_rot: int, out_dir: Path,
              target_arch: str, device_cls) -> None:
    """Compile one RoPE xclbin (Q or K)."""
    xclbin_name = f"qwen35-rope-{label}-{n_heads_label}h{head_dim}d.xclbin"
    instr_name  = f"qwen35-rope-{label}-{n_heads_label}h{head_dim}d-instr.bin"
    xclbin_path = out_dir / xclbin_name
    instr_path  = out_dir / instr_name

    print(f"[build_qwen35_rope] {label}: n_total={n_total} head_dim={head_dim} "
          f"n_rot={n_rot} N_div_n={n_total//head_dim} npu={target_arch}")
    print(f"  xclbin → {xclbin_path}")
    print(f"  instr  → {instr_path}")

    if n_rot % 32 != 0:
        raise ValueError(f"n_rot={n_rot} must be a multiple of 32 (n_rot/2 must be multiple of 16)")
    if (head_dim - n_rot) % 16 != 0:
        raise ValueError(f"head_dim-n_rot={head_dim-n_rot} must be a multiple of 16 (pass-through region)")
    if n_total % head_dim != 0:
        raise ValueError(f"n_total={n_total} must be divisible by head_dim={head_dim}")

    # tile_ty: one head (head_dim bfloat16 elements)
    tile_ty: Any = np.ndarray[(head_dim,), np.dtype[bfloat16]]
    # cs_tile_ty: n_rot bfloat16 elements [cos_0..cos_{n_rot/2-1}, sin_0..sin_{n_rot/2-1}]
    cs_tile_ty: Any = np.ndarray[(n_rot,), np.dtype[bfloat16]]

    include_dirs = []
    if AIE_INCLUDE and AIE_INCLUDE.is_dir():
        include_dirs.append(str(AIE_INCLUDE))

    # arg_types: (input, output, cs, head_dim=auto-appended-n)
    # n_rot is NOT a runtime param — the kernel derives it as head_dim/4 (Qwen3.5
    # partial_rotary_factor=0.25). The IRON _transform_gen framework only supports
    # tensor params + the auto-appended tile size as scalars.
    kernel = ExternalFunction(
        name="rope_rotate_bf16",
        source_file=str(KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, cs_tile_ty, np.int32],
        include_dirs=include_dirs,
    )

    qk_buf  = np.zeros(n_total, dtype=bfloat16)   # full Q or K tensor
    out_buf  = np.zeros(n_total, dtype=bfloat16)
    cs_buf   = np.zeros(n_rot,   dtype=bfloat16)   # tensor param: acquired once per dispatch

    # _transform_gen: single-core, tile_size=head_dim
    # N_div_n = n_total / head_dim = n_heads (or n_kv_heads)
    # cs_buf → tensor param (held for all head iterations)
    mlir_module = _transform_gen(
        kernel, [qk_buf], out_buf, cs_buf, tile_size=head_dim
    )

    with tempfile.TemporaryDirectory(prefix="hipfire_npu_build_") as tmp:
        tmp_path   = Path(tmp)
        tmp_xclbin = tmp_path / "final.xclbin"
        tmp_instr  = tmp_path / "insts.bin"

        compile_external_kernel(kernel, tmp_path, target_arch=target_arch)
        compile_mlir_module(
            mlir_module=mlir_module,
            insts_path=tmp_instr,
            xclbin_path=tmp_xclbin,
            work_dir=tmp_path,
        )
        shutil.copy2(tmp_xclbin, xclbin_path)
        shutil.copy2(tmp_instr,  instr_path)

    print(f"  xclbin: {xclbin_path.stat().st_size} bytes")
    print(f"  instr:  {instr_path.stat().st_size} bytes")


def build(n_heads: int, n_kv_heads: int, head_dim: int, n_rot: int,
          out_dir: Path, npu: str = "auto") -> None:
    if npu == "auto":
        npu = detect_npu()
        print(f"[build_qwen35_rope] detected NPU: {npu}")
    if npu not in _NPU_DEVICES:
        raise ValueError(f"--npu must be 'auto' or one of {list(_NPU_DEVICES)}, got {npu!r}")
    device_cls, target_arch = _NPU_DEVICES[npu]

    if not KERNEL_SRC.exists():
        raise FileNotFoundError(f"Kernel source not found: {KERNEL_SRC}")

    out_dir.mkdir(parents=True, exist_ok=True)
    set_current_device(device_cls())

    build_one("q", n_heads * head_dim, n_heads,    head_dim, n_rot, out_dir, target_arch, device_cls)
    build_one("k", n_kv_heads * head_dim, n_kv_heads, head_dim, n_rot, out_dir, target_arch, device_cls)

    print(f"\n[build_qwen35_rope] done")
    print("Set env vars to activate the NPU RoPE path:")
    print(f"  export HIPFIRE_QWEN35_ROPE_BF16=xdna1")
    print(f"  export HIPFIRE_QWEN35_XDNA1_ROPE_Q_XCLBIN={out_dir}/qwen35-rope-q-{n_heads}h{head_dim}d.xclbin")
    print(f"  export HIPFIRE_QWEN35_XDNA1_ROPE_K_XCLBIN={out_dir}/qwen35-rope-k-{n_kv_heads}h{head_dim}d.xclbin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-heads",    type=int, required=True, help="Number of Q heads")
    parser.add_argument("--n-kv-heads", type=int, required=True, help="Number of KV heads")
    parser.add_argument("--head-dim",   type=int, default=256,   help="Head dimension (default: 256)")
    parser.add_argument("--n-rot",      type=int, default=64,    help="Number of dims to rotate (default: 64)")
    parser.add_argument("--out-dir",    type=Path, default=Path("target/npu"))
    parser.add_argument("--npu", choices=["auto"] + list(_NPU_DEVICES), default="auto")
    args = parser.parse_args()
    build(args.n_heads, args.n_kv_heads, args.head_dim, args.n_rot, args.out_dir, args.npu)


if __name__ == "__main__":
    main()
