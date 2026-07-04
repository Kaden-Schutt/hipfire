#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Compile the NPU attention output gate kernel for Qwen3.5.

Produces two files:
  - <out_dir>/qwen35-attn-gate-{n_heads}h{head_dim}d.xclbin
  - <out_dir>/qwen35-attn-gate-{n_heads}h{head_dim}d-instr.bin

The kernel computes: out[i] = sigmoid(gate[i]) * x[i]
across all 4 NPU columns in parallel (BF16, element-wise).

This replaces gpu.sigmoid_f32 + gpu.mul_f32 in the Qwen3.5 forward pass
when config.attn_output_gate is true. Only dense-attention configs that set
attn_output_gate=true (inferred or explicit) need this kernel.

q_dim = n_heads × head_dim must be divisible by tile_size × 4.

Usage:
    python tools/npu/build_qwen35_attn_gate.py --n-heads 8 --head-dim 256
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

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
from aie.iron.algorithms.transform import transform_parallel_binary
from aie.iron.device import NPU1, NPU2
from aie.utils import set_current_device
from aie.utils.compile import compile_mlir_module, compile_external_kernel

SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_SRC = SCRIPT_DIR / "sigmoid_mul_bf16.cc"

_mlir_aie_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
AIE_INCLUDE = _mlir_aie_pkg / "mlir_aie" / "include" if _mlir_aie_pkg else None

_NPU_DEVICES = {
    "npu1": (NPU1, "aie2", "AIE2"),
    "npu2": (NPU2, "aie2p", "AIE2P"),
}

_NAME_TO_NPU = {
    "npu1": "npu1",
    "Phoenix": "npu1",
    "npu4": "npu2",
    "npu5": "npu2",
    "npu6": "npu2",
    "Strix": "npu2",
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
    raise RuntimeError(f"Cannot map device name {name!r}. Pass --npu explicitly.")


def build(n_heads: int, head_dim: int, out_dir: Path, tile_size: int = 16, npu: str = "auto") -> None:
    if npu == "auto":
        npu = detect_npu()
        print(f"[build_qwen35_attn_gate] detected NPU: {npu}")
    if npu not in _NPU_DEVICES:
        raise ValueError(f"--npu must be 'auto' or one of {list(_NPU_DEVICES)}, got {npu!r}")
    device_cls, target_arch, runtime_subdir = _NPU_DEVICES[npu]

    q_dim = n_heads * head_dim
    xclbin_name = f"qwen35-attn-gate-{n_heads}h{head_dim}d.xclbin"
    instr_name = f"qwen35-attn-gate-{n_heads}h{head_dim}d-instr.bin"
    xclbin_path = out_dir / xclbin_name
    instr_path = out_dir / instr_name

    print(
        f"[build_qwen35_attn_gate] n_heads={n_heads} head_dim={head_dim} "
        f"q_dim={q_dim} tile_size={tile_size} npu={npu} arch={target_arch}"
    )
    print(f"  xclbin → {xclbin_path}")
    print(f"  instr  → {instr_path}")

    if not KERNEL_SRC.exists():
        raise FileNotFoundError(f"Kernel source not found: {KERNEL_SRC}")

    set_current_device(device_cls())
    from aie.utils import get_current_device

    num_cols = get_current_device().cols
    min_multiple = tile_size * num_cols
    if q_dim % min_multiple != 0:
        raise ValueError(
            f"q_dim={q_dim} must be divisible by tile_size*num_cols={min_multiple} "
            f"(tile_size={tile_size}, num_cols={num_cols})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    runtime_lib = _mlir_aie_pkg / "mlir_aie" / "aie_runtime_lib" / runtime_subdir if _mlir_aie_pkg else None

    tile_ty: Any = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    include_dirs = []
    if AIE_INCLUDE and AIE_INCLUDE.is_dir():
        include_dirs.append(str(AIE_INCLUDE))
    if runtime_lib and runtime_lib.is_dir():
        include_dirs.append(str(runtime_lib))

    kernel = ExternalFunction(
        name="sigmoid_mul_bf16",
        source_file=str(KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, tile_ty, np.int32],
        include_dirs=include_dirs,
    )

    gate_buf = np.zeros(q_dim, dtype=bfloat16)
    x_buf = np.zeros(q_dim, dtype=bfloat16)
    out_buf = np.zeros(q_dim, dtype=bfloat16)

    mlir_module = transform_parallel_binary(kernel, gate_buf, x_buf, out_buf, tile_size=tile_size)

    with tempfile.TemporaryDirectory(prefix="hipfire_npu_build_") as tmp:
        tmp_path = Path(tmp)
        tmp_xclbin = tmp_path / "final.xclbin"
        tmp_instr = tmp_path / "insts.bin"

        compile_external_kernel(kernel, tmp_path, target_arch=target_arch)
        compile_mlir_module(
            mlir_module=mlir_module,
            insts_path=tmp_instr,
            xclbin_path=tmp_xclbin,
            work_dir=tmp_path,
        )
        shutil.copy2(tmp_xclbin, xclbin_path)
        shutil.copy2(tmp_instr, instr_path)

    print(f"  xclbin: {xclbin_path.stat().st_size} bytes")
    print(f"  instr:  {instr_path.stat().st_size} bytes")
    print("\nSet env vars to activate the NPU attn-gate path:")
    print(f"  export HIPFIRE_QWEN35_ATTN_GATE_BF16=xdna1")
    print(f"  export HIPFIRE_QWEN35_XDNA1_ATTN_GATE_XCLBIN={xclbin_path}")
    print(f"  export HIPFIRE_QWEN35_XDNA1_ATTN_GATE_INSTR={instr_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-heads", type=int, required=True, help="Number of Q heads")
    parser.add_argument("--head-dim", type=int, default=256, help="Head dimension (default: 256)")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("target/npu"))
    parser.add_argument("--npu", choices=["auto"] + list(_NPU_DEVICES), default="auto")
    args = parser.parse_args()
    if args.tile_size % 16 != 0:
        parser.error(f"--tile-size must be a multiple of 16, got {args.tile_size}")
    build(args.n_heads, args.head_dim, args.out_dir, args.tile_size, args.npu)


if __name__ == "__main__":
    main()
