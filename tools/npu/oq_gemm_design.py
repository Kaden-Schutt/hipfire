#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# OQ8/OQ+ → NPU feasibility spike: int8 grouped GEMM on the XDNA1 (AIE2) NPU.
#
# The AIE compute core is the stock IRON int8 matmul kernel
# (`aie.iron.kernels.linalg.mm`, input=int8 output=int32, the same
# `aie::mmul<4,8,8,int8,int8,acc32>` micro-kernel the GPU OQ8 path uses on
# `v_wmma_i32_16x16x16_iu8`).  The dataflow harness below is adapted from the
# upstream mlir-aie single-core matrix-multiply IRON example
# (programming_examples/basic/matrix_multiplication/single_core/single_core.py,
# Copyright (C) 2025-2026 AMD, Apache-2.0 WITH LLVM-exception) — trimmed to a
# library entry point that hipfire's OQ8 oracle/bench harness drives.
#
# OQ8 maps to this kernel as: Y[b,m] = Σ_g scale_w[m,g]·scale_x[b,g]·Σ_{k∈g} W[m,k]·X[b,k].
# The inner int8·int8→int32 contraction is exactly `mm`; the per-group (G=256)
# f32 rescale is applied by the *caller* (see test_oq_gemm_npu.py).  We run the
# matmul with C[M,N] = A[M,K] · B[N,K]^T (b_col_maj=1) so A=W[M,K] and B=X[N,K]
# are both consumed in their natural row-major layout (N = batch B).
#
# ── Self-contained env bootstrap (this box) ──────────────────────────────────
# Building/running NPU kernels here needs three things the bare interpreter
# lacks (see memory project-npu-toolchain-this-box):
#   1. pyxrt on sys.path (/opt/xilinx/xrt/python) BEFORE aie.utils imports, else
#      it silently falls back to CPUOnlyTensor and device="npu" is rejected.
#   2. libxrt_coreutil preloaded RTLD_GLOBAL (weak-vtable ordering under XRT 2.25).
#   3. libboost_program_options.so.1.83.0 on LD_LIBRARY_PATH for xclbinutil
#      (vendored user-space copy under ~/.cache/hipfire-npu-deps/lib).
# All three are set up at import time so callers need no external env.

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, cast

# 3. boost for xclbinutil — must be on LD_LIBRARY_PATH before the aiecc
#    subprocess spawns (set in-process; propagates to children).
_BOOST_DEP = Path.home() / ".cache" / "hipfire-npu-deps" / "lib"
_XRT_LIB = "/opt/xilinx/xrt/lib"
_extra_ld = [str(p) for p in (_BOOST_DEP, Path(_XRT_LIB)) if p and Path(p).is_dir()]
if _extra_ld:
    _cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(_extra_ld + ([_cur] if _cur else []))

# xclbinutil lives in the XRT bin dir which may not be on PATH.
_XRT_BIN = "/opt/xilinx/xrt/bin"
if Path(_XRT_BIN).is_dir() and _XRT_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _XRT_BIN + os.pathsep + os.environ.get("PATH", "")

# venv site-packages (mlir_aie, ml_dtypes) when launched via a bare python.
for _p in (Path.home() / ".venv" / "lib").glob("python*/site-packages"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 2. coreutil preload, then 1. pyxrt on path — BEFORE importing aie.utils.
ctypes.CDLL(os.path.join(_XRT_LIB, "libxrt_coreutil.so.2"), mode=ctypes.RTLD_GLOBAL)
_XRT_PY = "/opt/xilinx/xrt/python"
if _XRT_PY not in sys.path:
    sys.path.insert(0, _XRT_PY)

import numpy as np  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import (  # noqa: E402
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import NPU1, NPU2  # noqa: E402
from aie.helpers.taplib import TensorTiler2D  # noqa: E402
from aie.utils import set_current_device  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402

# Set the active NPU device at import — without this, get_current_device() is
# empty in resolve_program() and the XRT run segfaults (run_design_cli does the
# equivalent set_current_device() before any run). Detect npu1/npu2 by name.
_NAME_TO_NPU = {"Phoenix": NPU1, "npu1": NPU1, "Strix": NPU2, "Krackan": NPU2, "npu4": NPU2, "npu5": NPU2, "npu6": NPU2}


def _detect_and_set_device():
    try:
        import pyxrt

        name = pyxrt.device(0).get_info(pyxrt.xrt_info_device.name)
        cls = next((c for sub, c in _NAME_TO_NPU.items() if sub in name), NPU1)
    except Exception:
        cls = NPU1  # this box is Phoenix/NPU1
    set_current_device(cls())
    return cls.__name__


NPU_DEVICE = _detect_and_set_device()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def int_matmul(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    dtype_in_str: CompileTime[str],
    dtype_out_str: CompileTime[str],
    b_col_maj: CompileTime[int] = 1,
):
    """Single-core integer matmul C[M,N] = A[M,K] · B[K,N] (B consumed col-major
    when b_col_maj=1, i.e. stored [N,K]).  Adapted from upstream single_core.py."""
    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert M % m == 0 and K % k == 0 and N % n == 0

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims
    assert m % r == 0 and k % s == 0 and n % t == 0

    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    tiles = M_div_m * N_div_n

    ndarray_ty = cast(Any, np.ndarray)
    dtype_ty = cast(Any, np.dtype)
    A_ty: Any = ndarray_ty[(M * K,), dtype_ty[dtype_in]]
    B_ty: Any = ndarray_ty[(K * N,), dtype_ty[dtype_in]]
    C_ty: Any = ndarray_ty[(M * N,), dtype_ty[dtype_out]]
    a_ty: Any = ndarray_ty[(m, k), dtype_ty[dtype_in]]
    b_ty: Any = ndarray_ty[(k, n), dtype_ty[dtype_in]]
    c_ty: Any = ndarray_ty[(m, n), dtype_ty[dtype_out]]

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    if b_col_maj:
        b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
    else:
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        stack_size=0xD00,
    )

    rows_per_block = 4
    A_tiles = TensorTiler2D.group_tiler((M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n, prune_step=False)
    if b_col_maj:
        b_tap = TensorTiler2D.group_tiler((N, K), (n, k), (N_div_n, K_div_k), prune_step=False)[0]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True, prune_step=False
        )[0]
    C_tiles = TensorTiler2D.group_tiler((M, N), (m, n), (rows_per_block // 2, N_div_n), prune_step=False)
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        tgs = []
        for tile_row_block in range(iron.ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                rt.drain(outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True)
                c_index += 1
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    return Program(iron.get_current_device(), rt).resolve_program()


# Default micro-tile (per-core) dims. int8 AIE2 mac_dims are (4,8,8); these
# tile sizes are multiples and keep one (m,k)+(k,n) pair resident in core SRAM.
DEFAULT_TILE = dict(m=32, k=64, n=64)


def _tiles_for(M, K, N, tile):
    """Pick per-core (m,k,n) that divide (M,K,N); the upstream ping-pong host
    loop additionally needs M//m even."""
    m, k, n = tile["m"], tile["k"], tile["n"]
    # shrink to divisors if needed
    while M % m or (M // m) % 2:
        m //= 2
        if m < 4:
            m = 4
            break
    while K % k:
        k //= 2
    while N % n:
        n //= 2
    return m, k, n


def matmul_npu(A_np, B_np, *, b_col_maj=1, tile=None):
    """Run C[M,N] = A[M,K] · B[N,K]^T on the NPU. A int8 [M,K], B int8 [N,K]
    (b_col_maj=1). Returns int32 C[M,N] as numpy. Compiles+runs (cached by shape).

    Routes through run_iters (warmup=0, iters=1): calling the @iron.jit design
    object directly segfaults the XRT runtime on this box, but run_iters drives
    the compile+run+device-sync protocol correctly."""
    C, _bench, tile_used = bench_npu(A_np, B_np, b_col_maj=b_col_maj, tile=tile, warmup=0, iters=1)
    return C, tile_used


def bench_npu(A_np, B_np, *, b_col_maj=1, tile=None, warmup=5, iters=20):
    """Benchmark the int8 matmul. Returns (C[M,N] int32, bench) where bench has
    .npu_time_us / .e2e_time_us avg fields from run_iters."""
    tile = tile or DEFAULT_TILE
    M, K = A_np.shape
    N = B_np.shape[0]
    m, k, n = _tiles_for(M, K, N, tile)
    A_t = iron.tensor(A_np.reshape(-1).astype(np.int8), dtype=np.int8, device="npu")
    B_t = iron.tensor(B_np.reshape(-1).astype(np.int8), dtype=np.int8, device="npu")
    C_t = iron.zeros(M * N, dtype=np.int32, device="npu")
    bench = run_iters(
        int_matmul,
        A_t,
        B_t,
        C_t,
        M=M,
        K=K,
        N=N,
        m=m,
        k=k,
        n=n,
        dtype_in_str="i8",
        dtype_out_str="i32",
        b_col_maj=b_col_maj,
        warmup=warmup,
        iters=iters,
    )
    # C_t.numpy() is a VIEW onto the XRT device buffer; it is freed when C_t
    # goes out of scope at return, so the caller would read freed memory
    # (segfault). Copy into host-owned memory before C_t dies.
    C = np.array(C_t.numpy().reshape(M, N), dtype=np.int32)
    return C, bench, (m, k, n)
