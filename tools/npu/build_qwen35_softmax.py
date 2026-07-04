#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Compile the NPU BF16 row softmax kernel for Qwen3.5 attention scores.

One xclbin is produced per (n_heads, ctx_len) pair:
  <out_dir>/qwen35-softmax-{n_heads}h{ctx_len}ctx.xclbin
  <out_dir>/qwen35-softmax-{n_heads}h{ctx_len}ctx-instr.bin

The kernel computes:
  output[head, i] = exp(input[head, i] - max_head) / sum_j(exp(input[head, j] - max_head))

ctx_len must be a multiple of 16.  Caller pads to the next supported size and
fills invalid positions with -inf before calling.

Default configs (all Qwen3.5 dense Q heads, common context window sizes):
  --n-heads 8  --ctx-lens 64,128,256,512

Usage:
    python tools/npu/build_qwen35_softmax.py --n-heads 8 --ctx-lens 64,128,256,512
    python tools/npu/build_qwen35_softmax.py --n-heads 8 --ctx-lens 256
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
from aie.iron.algorithms.transform import _transform_gen
from aie.iron.device import NPU1, NPU2
from aie.utils import set_current_device
from aie.utils.compile import compile_mlir_module, compile_external_kernel

SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_SRC = SCRIPT_DIR / "softmax_bf16.cc"

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


def build_one(n_heads: int, ctx_len: int, out_dir: Path, npu: str) -> None:
    if npu not in _NPU_DEVICES:
        raise ValueError(f"--npu must be one of {list(_NPU_DEVICES)}, got {npu!r}")
    device_cls, target_arch, runtime_subdir = _NPU_DEVICES[npu]

    if ctx_len % 16 != 0:
        raise ValueError(f"ctx_len={ctx_len} must be a multiple of 16")

    xclbin_name = f"qwen35-softmax-{n_heads}h{ctx_len}ctx.xclbin"
    instr_name = f"qwen35-softmax-{n_heads}h{ctx_len}ctx-instr.bin"
    xclbin_path = out_dir / xclbin_name
    instr_path = out_dir / instr_name

    print(f"[build_qwen35_softmax] n_heads={n_heads} ctx_len={ctx_len} npu={npu} arch={target_arch}")
    print(f"  xclbin → {xclbin_path}")
    print(f"  instr  → {instr_path}")

    if not KERNEL_SRC.exists():
        raise FileNotFoundError(f"Kernel source not found: {KERNEL_SRC}")

    set_current_device(device_cls())

    out_dir.mkdir(parents=True, exist_ok=True)

    runtime_lib = _mlir_aie_pkg / "mlir_aie" / "aie_runtime_lib" / runtime_subdir if _mlir_aie_pkg else None

    tile_ty: Any = np.ndarray[(ctx_len,), np.dtype[bfloat16]]
    include_dirs = []
    if AIE_INCLUDE and AIE_INCLUDE.is_dir():
        include_dirs.append(str(AIE_INCLUDE))
    if runtime_lib and runtime_lib.is_dir():
        include_dirs.append(str(runtime_lib))

    # arg_types: (input_tile, output_tile, n=tile_size auto-appended)
    kernel = ExternalFunction(
        name="softmax_bf16",
        source_file=str(KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, np.int32],
        include_dirs=include_dirs,
    )

    total = n_heads * ctx_len
    input_buf = np.zeros(total, dtype=bfloat16)
    out_buf = np.zeros(total, dtype=bfloat16)

    # _transform_gen: tile_size=ctx_len → N_div_n=n_heads tile iterations
    # No tensor params → rt.run order: [t_in, t_out]
    mlir_module = _transform_gen(kernel, [input_buf], out_buf, tile_size=ctx_len)

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


def build(n_heads: int, ctx_lens: list[int], out_dir: Path, npu: str) -> None:
    if npu == "auto":
        npu = detect_npu()
        print(f"[build_qwen35_softmax] detected NPU: {npu}")
    for ctx_len in ctx_lens:
        build_one(n_heads, ctx_len, out_dir, npu)

    print("\nSet env vars to activate the NPU softmax path:")
    print(f"  export HIPFIRE_QWEN35_SOFTMAX_BF16=xdna1")
    for ctx_len in ctx_lens:
        tag = f"CTX{ctx_len}"
        print(
            f"  export HIPFIRE_QWEN35_XDNA1_SOFTMAX_{tag}_XCLBIN={out_dir}/qwen35-softmax-{n_heads}h{ctx_len}ctx.xclbin"
        )
        print(
            f"  export HIPFIRE_QWEN35_XDNA1_SOFTMAX_{tag}_INSTR="
            f"{out_dir}/qwen35-softmax-{n_heads}h{ctx_len}ctx-instr.bin"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument(
        "--ctx-lens",
        default="64,128,256,512",
        help="Comma-separated context window sizes to build (default: 64,128,256,512)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("target/npu"))
    parser.add_argument("--npu", choices=["auto"] + list(_NPU_DEVICES), default="auto")
    args = parser.parse_args()

    ctx_lens = []
    for s in args.ctx_lens.split(","):
        v = int(s.strip())
        if v % 16 != 0:
            parser.error(f"ctx_len={v} must be a multiple of 16")
        ctx_lens.append(v)

    build(args.n_heads, ctx_lens, args.out_dir, args.npu)


if __name__ == "__main__":
    main()
