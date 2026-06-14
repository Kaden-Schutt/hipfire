#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Compile the fused NPU per-head QK norm + RoPE rotation kernel for Qwen3.5.

Produces four files (two per tensor, Q and K):
  - <out_dir>/qwen35-headnorm-rope-q-{n_heads}h{head_dim}d.xclbin  / -instr.bin
  - <out_dir>/qwen35-headnorm-rope-k-{n_kv_heads}h{head_dim}d.xclbin / -instr.bin

The kernel fuses headnorm (per-head RMSNorm with a shared weight) and RoPE
rotation into a single tile invocation, eliminating two NPU dispatches per
attention layer compared to running headnorm + rope separately.

Tensor param layout (packed_weight_cs, shape [head_dim + n_rot]):
  [0, head_dim)         = per-head norm weight (bf16)
  [head_dim, head_dim+n_rot) = cos/sin buffer (bf16) for the current token
                               layout: [cos_0..cos_{n_rot2-1}, sin_0..sin_{n_rot2-1}]

n_rot is derived inside the kernel as head_dim / 4 (Qwen3.5 partial_rotary_factor=0.25).

Design: _transform_gen (single-core), tile_size=head_dim, one tensor param.
N_div_n = n_heads (Q) or n_kv_heads (K).

Usage:
    python tools/npu/build_qwen35_headnorm_rope.py \\
        --n-heads 8 --n-kv-heads 2 --head-dim 256
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

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
KERNEL_SRC = SCRIPT_DIR / "headnorm_rope_bf16.cc"

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
    ctypes.CDLL("/opt/xilinx/xrt/lib/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)
    xrt_py = "/opt/xilinx/xrt/python"
    if xrt_py not in sys.path:
        sys.path.insert(0, xrt_py)
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
              head_dim: int, out_dir: Path,
              target_arch: str, device_cls) -> None:
    n_rot = head_dim // 4   # Qwen3.5 partial_rotary_factor = 0.25

    xclbin_name = f"qwen35-headnorm-rope-{label}-{n_heads_label}h{head_dim}d.xclbin"
    instr_name  = f"qwen35-headnorm-rope-{label}-{n_heads_label}h{head_dim}d-instr.bin"
    xclbin_path = out_dir / xclbin_name
    instr_path  = out_dir / instr_name

    print(f"[build_qwen35_headnorm_rope] {label}: n_total={n_total} head_dim={head_dim} "
          f"n_rot={n_rot} N_div_n={n_total//head_dim} npu={target_arch}")
    print(f"  xclbin → {xclbin_path}")
    print(f"  instr  → {instr_path}")

    if head_dim % 16 != 0:
        raise ValueError(f"head_dim={head_dim} must be a multiple of 16")
    if n_rot % 32 != 0:
        raise ValueError(
            f"n_rot={n_rot} must be a multiple of 32 "
            f"(n_rot/2 must be divisible by VEC=16)"
        )
    if (head_dim - n_rot) % 16 != 0:
        raise ValueError(
            f"head_dim-n_rot={head_dim-n_rot} must be a multiple of 16 "
            f"(passthrough region VEC alignment)"
        )
    if n_total % head_dim != 0:
        raise ValueError(f"n_total={n_total} must be divisible by head_dim={head_dim}")

    # Tile type: one head
    tile_ty = np.ndarray[(head_dim,), np.dtype[bfloat16]]
    # Tensor param: packed [weight (head_dim elems), cs (n_rot elems)]
    packed_param_ty = np.ndarray[(head_dim + n_rot,), np.dtype[bfloat16]]

    include_dirs = []
    if AIE_INCLUDE and AIE_INCLUDE.is_dir():
        include_dirs.append(str(AIE_INCLUDE))

    # arg_types: (input=tile_in, output=tile_out, packed_weight_cs=param0, head_dim=auto-n)
    kernel = ExternalFunction(
        name="headnorm_rope_bf16",
        source_file=str(KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, packed_param_ty, np.int32],
        include_dirs=include_dirs,
    )

    qk_buf     = np.zeros(n_total,           dtype=bfloat16)
    out_buf    = np.zeros(n_total,           dtype=bfloat16)
    packed_buf = np.zeros(head_dim + n_rot,  dtype=bfloat16)  # tensor param

    mlir_module = _transform_gen(
        kernel, [qk_buf], out_buf, packed_buf, tile_size=head_dim
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


def build(n_heads: int, n_kv_heads: int, head_dim: int,
          out_dir: Path, npu: str = "auto") -> None:
    if npu == "auto":
        npu = detect_npu()
        print(f"[build_qwen35_headnorm_rope] detected NPU: {npu}")
    if npu not in _NPU_DEVICES:
        raise ValueError(f"--npu must be 'auto' or one of {list(_NPU_DEVICES)}, got {npu!r}")
    device_cls, target_arch = _NPU_DEVICES[npu]

    if not KERNEL_SRC.exists():
        raise FileNotFoundError(f"Kernel source not found: {KERNEL_SRC}")

    out_dir.mkdir(parents=True, exist_ok=True)
    set_current_device(device_cls())

    build_one("q", n_heads    * head_dim, n_heads,    head_dim, out_dir, target_arch, device_cls)
    build_one("k", n_kv_heads * head_dim, n_kv_heads, head_dim, out_dir, target_arch, device_cls)

    print(f"\n[build_qwen35_headnorm_rope] done")
    print("Set env vars to activate the NPU fused headnorm+rope path:")
    print(f"  export HIPFIRE_QWEN35_HEADNORM_ROPE_BF16=xdna1")
    print(f"  export HIPFIRE_QWEN35_XDNA1_HEADNORM_ROPE_Q_XCLBIN="
          f"{out_dir}/qwen35-headnorm-rope-q-{n_heads}h{head_dim}d.xclbin")
    print(f"  export HIPFIRE_QWEN35_XDNA1_HEADNORM_ROPE_K_XCLBIN="
          f"{out_dir}/qwen35-headnorm-rope-k-{n_kv_heads}h{head_dim}d.xclbin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-heads",    type=int, required=True, help="Number of Q heads")
    parser.add_argument("--n-kv-heads", type=int, required=True, help="Number of KV heads")
    parser.add_argument("--head-dim",   type=int, default=256,   help="Head dimension (default: 256)")
    parser.add_argument("--out-dir",    type=Path, default=Path("target/npu"))
    parser.add_argument("--npu", choices=["auto"] + list(_NPU_DEVICES), default="auto")
    args = parser.parse_args()
    build(args.n_heads, args.n_kv_heads, args.head_dim, args.out_dir, args.npu)


if __name__ == "__main__":
    main()
