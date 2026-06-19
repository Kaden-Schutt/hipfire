#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Compile the NPU SwiGLU kernel for Qwen3.5 dense FFN layers.

Produces two files consumed by `libhipfire_xdna1.so` via
`xdna1_bf16_swiglu_create(xclbin_path, instr_path, hidden_size)`:
  - <out_dir>/qwen35-swiglu-<hidden_size>.xclbin  (path1)
  - <out_dir>/qwen35-swiglu-<hidden_size>-instr.bin  (path2)

The kernel computes: out[i] = silu(gate[i]) * up[i] (BF16, elementwise)
across all NPU columns in parallel.

Usage:
    python tools/npu/build_qwen35_swiglu.py --hidden-size 8960          # auto-detects NPU
    python tools/npu/build_qwen35_swiglu.py --hidden-size 8960 --npu npu1
    python tools/npu/build_qwen35_swiglu.py --hidden-size 18944 --out-dir target/npu
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

# xclbinutil lives in the XRT bin directory which may not be on PATH.
_XRT_BIN = Path("/opt/xilinx/xrt/bin")
if _XRT_BIN.is_dir() and str(_XRT_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_XRT_BIN) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from ml_dtypes import bfloat16
from aie.iron import ExternalFunction, Program
from aie.iron.algorithms.transform import transform_parallel_binary
from aie.iron.device import NPU1, NPU2
from aie.utils import set_current_device
from aie.utils.compile import compile_mlir_module, compile_external_kernel

SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_SRC = SCRIPT_DIR / "silu_mul_bf16.cc"

# AIE API headers are bundled with mlir_aie
_mlir_aie_pkg = next(
    (Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None
)
AIE_INCLUDE = _mlir_aie_pkg / "mlir_aie" / "include" if _mlir_aie_pkg else None
AIE_RUNTIME_LIB = _mlir_aie_pkg / "mlir_aie" / "aie_runtime_lib" / "AIE2" if _mlir_aie_pkg else None


_NPU_DEVICES = {
    "npu1": (NPU1, "aie2",  "AIE2"),
    "npu2": (NPU2, "aie2p", "AIE2P"),
}

# Maps substrings of the pyxrt device name to NPU generation keys.
# Same mapping used by XRTHostRuntime.NPU_MODELS.
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
    """Query the installed NPU via pyxrt and return its generation key.

    Returns one of the keys in _NPU_DEVICES, or raises RuntimeError if the
    device cannot be detected (no hardware, driver not loaded, etc.).
    """
    import ctypes
    # libxrt_coreutil must be loaded before libxrt_core to avoid a missing
    # weak vtable symbol under XRT 2.25.
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


def build(hidden_size: int, out_dir: Path, tile_size: int = 16,
          npu: str = "auto") -> None:
    """Compile the SwiGLU kernel for `hidden_size` and write artifacts to `out_dir`."""
    if npu == "auto":
        npu = detect_npu()
        print(f"[build_qwen35_swiglu] detected NPU: {npu}")
    if npu not in _NPU_DEVICES:
        raise ValueError(f"--npu must be 'auto' or one of {list(_NPU_DEVICES)}, got {npu!r}")
    device_cls, target_arch, runtime_subdir = _NPU_DEVICES[npu]

    out_dir.mkdir(parents=True, exist_ok=True)

    xclbin_name = f"qwen35-swiglu-{hidden_size}.xclbin"
    instr_name  = f"qwen35-swiglu-{hidden_size}-instr.bin"
    xclbin_path = out_dir / xclbin_name
    instr_path  = out_dir / instr_name

    print(f"[build_qwen35_swiglu] hidden_size={hidden_size} tile_size={tile_size} npu={npu} arch={target_arch}")
    print(f"  xclbin → {xclbin_path}")
    print(f"  instr  → {instr_path}")

    if not KERNEL_SRC.exists():
        raise FileNotFoundError(f"Kernel source not found: {KERNEL_SRC}")

    set_current_device(device_cls())
    from aie.utils import get_current_device
    dev = get_current_device()
    num_cols = dev.cols
    min_multiple = tile_size * num_cols
    if hidden_size % min_multiple != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by "
            f"tile_size*num_cols={min_multiple} (tile_size={tile_size}, "
            f"num_cols={num_cols} from {npu.upper()})"
        )

    # Runtime lib differs per arch (AIE2 has LUT tanh, AIE2P does not)
    runtime_lib = (
        _mlir_aie_pkg / "mlir_aie" / "aie_runtime_lib" / runtime_subdir
        if _mlir_aie_pkg else None
    )

    tile_ty: Any = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    include_dirs = []
    if AIE_INCLUDE and AIE_INCLUDE.is_dir():
        include_dirs.append(str(AIE_INCLUDE))
    if runtime_lib and runtime_lib.is_dir():
        include_dirs.append(str(runtime_lib))
    kernel = ExternalFunction(
        name="silu_mul_bf16",
        source_file=str(KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, tile_ty, np.int32],
        include_dirs=include_dirs,
    )

    gate_buf = np.zeros(hidden_size, dtype=bfloat16)
    up_buf   = np.zeros(hidden_size, dtype=bfloat16)
    out_buf  = np.zeros(hidden_size, dtype=bfloat16)

    mlir_module = transform_parallel_binary(
        kernel, gate_buf, up_buf, out_buf, tile_size=tile_size
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

    print(f"[build_qwen35_swiglu] done")
    print(f"  xclbin: {xclbin_path.stat().st_size} bytes")
    print(f"  instr:  {instr_path.stat().st_size} bytes")
    print()
    print("Set env vars to activate the NPU path:")
    print(f"  export HIPFIRE_QWEN35_FFN_BF16=xdna1")
    print(f"  export HIPFIRE_QWEN35_XDNA1_XCLBIN={xclbin_path}")
    print(f"  export HIPFIRE_QWEN35_XDNA1_INSTR={instr_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hidden-size", type=int, required=True,
        help="FFN intermediate size (e.g. 8960 for Qwen3.5-1.5B, 18944 for 7B)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("target/npu"),
        help="Output directory for xclbin and instr.bin (default: target/npu)"
    )
    parser.add_argument(
        "--tile-size", type=int, default=16,
        help="Elements per objectfifo tile (must be multiple of 16, default: 16)"
    )
    parser.add_argument(
        "--npu", choices=["auto"] + list(_NPU_DEVICES), default="auto",
        help="Target NPU generation: auto=detect from hardware (default), npu1=AIE2/Phoenix, npu2=AIE2P/Strix"
    )
    args = parser.parse_args()

    if args.tile_size % 16 != 0:
        parser.error("--tile-size must be a multiple of 16")

    build(hidden_size=args.hidden_size, out_dir=args.out_dir,
          tile_size=args.tile_size, npu=args.npu)


if __name__ == "__main__":
    main()
