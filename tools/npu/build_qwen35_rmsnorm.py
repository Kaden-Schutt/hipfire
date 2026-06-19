#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Compile the NPU weighted RMSNorm kernel for Qwen3.5 hidden layers.

Produces two files consumed by `libhipfire_xdna1.so` via
`xdna1_bf16_rmsnorm_create(xclbin_path, instr_path, hidden_size)`:
  - <out_dir>/qwen35-rmsnorm-<hidden_size>.xclbin  (path1)
  - <out_dir>/qwen35-rmsnorm-<hidden_size>-instr.bin  (path2)

The kernel computes: out[i] = (x[i] / rms(x)) * weight[i] (BF16)
where rms(x) = sqrt(mean(x²) + eps).  Unlike the mlir_aie reference, this
kernel accepts a learned weight (gamma) tensor rather than hardcoding 1.0.

Because RMSNorm requires a full-row reduction before any output element can
be written, the entire hidden_size is mapped to a single tile (tile_size=hidden_size).
This ensures the AIE core sees all elements during the reduction pass.

Usage:
    python tools/npu/build_qwen35_rmsnorm.py --hidden-size 1536          # Qwen3.5-1.5B
    python tools/npu/build_qwen35_rmsnorm.py --hidden-size 3584          # Qwen3.5-7B
    python tools/npu/build_qwen35_rmsnorm.py --hidden-size 1536 --npu npu1
    python tools/npu/build_qwen35_rmsnorm.py --hidden-size 3584 --out-dir target/npu
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
from aie.iron.algorithms.transform import _transform_gen
from aie.iron.device import NPU1, NPU2
from aie.utils import set_current_device
from aie.utils.compile import compile_mlir_module, compile_external_kernel

SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_SRC = SCRIPT_DIR / "rms_norm_weighted_bf16.cc"

# AIE API headers are bundled with mlir_aie
_mlir_aie_pkg = next(
    (Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None
)
AIE_INCLUDE = _mlir_aie_pkg / "mlir_aie" / "include" if _mlir_aie_pkg else None

_NPU_DEVICES = {
    "npu1": (NPU1, "aie2",  "AIE2"),
    "npu2": (NPU2, "aie2p", "AIE2P"),
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
    """Query the installed NPU via pyxrt and return its generation key."""
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


def build(hidden_size: int, out_dir: Path, npu: str = "auto") -> None:
    """Compile the RMSNorm kernel for `hidden_size` and write artifacts to `out_dir`."""
    if npu == "auto":
        npu = detect_npu()
        print(f"[build_qwen35_rmsnorm] detected NPU: {npu}")
    if npu not in _NPU_DEVICES:
        raise ValueError(f"--npu must be 'auto' or one of {list(_NPU_DEVICES)}, got {npu!r}")
    device_cls, target_arch, _runtime_subdir = _NPU_DEVICES[npu]

    if hidden_size % 16 != 0:
        raise ValueError(f"hidden_size={hidden_size} must be a multiple of 16 (vector width)")

    out_dir.mkdir(parents=True, exist_ok=True)

    xclbin_name = f"qwen35-rmsnorm-{hidden_size}.xclbin"
    instr_name  = f"qwen35-rmsnorm-{hidden_size}-instr.bin"
    xclbin_path = out_dir / xclbin_name
    instr_path  = out_dir / instr_name

    print(f"[build_qwen35_rmsnorm] hidden_size={hidden_size} tile_size={hidden_size} (full-row) npu={npu} arch={target_arch}")
    print(f"  xclbin → {xclbin_path}")
    print(f"  instr  → {instr_path}")

    if not KERNEL_SRC.exists():
        raise FileNotFoundError(f"Kernel source not found: {KERNEL_SRC}")

    set_current_device(device_cls())

    # tile_size = hidden_size: the entire row fits in one AIE tile (≤32 KB SRAM).
    # At bfloat16: 3 × hidden_size × 2 bytes = 9 KB (1536) or 21 KB (3584) — both fit.
    # This produces a single-tile design so the reduction sees all elements at once.
    tile_size = hidden_size
    tile_ty: Any = np.ndarray[(tile_size,), np.dtype[bfloat16]]

    include_dirs = []
    if AIE_INCLUDE and AIE_INCLUDE.is_dir():
        include_dirs.append(str(AIE_INCLUDE))

    kernel = ExternalFunction(
        name="rms_norm_weighted_bf16",
        source_file=str(KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, tile_ty, np.int32],
        include_dirs=include_dirs,
    )

    input_buf  = np.zeros(hidden_size, dtype=bfloat16)
    weight_buf = np.zeros(hidden_size, dtype=bfloat16)
    output_buf = np.zeros(hidden_size, dtype=bfloat16)

    # _transform_gen: single-core design — sends the entire row to one AIE tile.
    # RMSNorm requires a global reduction before writing any output element, so
    # we cannot split the work across columns. With tile_size=hidden_size the
    # core receives all elements at once, computes sum(x²), then applies inv_rms.
    # The framework appends tile_size as the `cols` int32 arg automatically.
    mlir_module = _transform_gen(
        kernel, [input_buf, weight_buf], output_buf, tile_size=tile_size
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

    print(f"[build_qwen35_rmsnorm] done")
    print(f"  xclbin: {xclbin_path.stat().st_size} bytes")
    print(f"  instr:  {instr_path.stat().st_size} bytes")
    print()
    print("Set env vars to activate the NPU RMSNorm path:")
    print(f"  export HIPFIRE_QWEN35_RMSNORM_BF16=xdna1")
    print(f"  export HIPFIRE_QWEN35_XDNA1_RMSNORM_XCLBIN={xclbin_path}")
    print(f"  export HIPFIRE_QWEN35_XDNA1_RMSNORM_INSTR={instr_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hidden-size", type=int, required=True,
        help="Model hidden size (e.g. 1536 for Qwen3.5-1.5B, 3584 for 7B)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("target/npu"),
        help="Output directory for xclbin and instr.bin (default: target/npu)"
    )
    parser.add_argument(
        "--npu", choices=["auto"] + list(_NPU_DEVICES), default="auto",
        help="Target NPU generation: auto=detect from hardware (default), npu1=AIE2/Phoenix, npu2=AIE2P/Strix"
    )
    args = parser.parse_args()

    build(hidden_size=args.hidden_size, out_dir=args.out_dir, npu=args.npu)


if __name__ == "__main__":
    main()
