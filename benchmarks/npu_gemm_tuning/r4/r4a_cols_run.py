# R4a: 8-column aggregate W4A8 COMPUTE. COLS independent columns, each running the
# R2a compute core (../r2/r2a_gemm.cc) pinned to its own column so it uses a
# distinct compute tile + shim DMA channel. Each column streams N_BTILES int4
# weight tiles, reusing each (INNER+1)*NACC times against NACC resident int8
# activation tiles (R2a's II=1 fake-reuse recipe) — so the cores are COMPUTE-bound,
# giving the aggregate W4A8 compute CEILING across the array (the best-case number
# for "can the NPU beat the GPU"). Real streaming GEMM (feed-bound) is lower.
#
# Builds the xclbin+insts into the mlir-aie cache; the actual TOPS is measured
# through hipfire's NpuKernel (loop dispatch) — this host-wall print is a sanity
# check. Fresh process per run (pyxrt segfaults on repeat under py3.14).
import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.jit import jit
import aie.utils as aie_utils

INC = os.environ["MLIR_AIE_INC"]
COLS = int(os.environ.get("COLS", 8))
NACC = int(os.environ.get("NACC", 4))
INNER = int(os.environ.get("INNER", 64))
N_BTILES = int(os.environ["N_BTILES"])
SZ_A, SZ_C, SZ_Bb = 64, 64, 128  # int8 A tile / int32 C tile / packed-int4 W tile (128 B)
MACS = COLS * N_BTILES * (INNER + 1) * NACC * 1024

a_ty = np.ndarray[(NACC * SZ_A,), np.dtype[np.int8]]
w_ty = np.ndarray[(SZ_Bb,), np.dtype[np.int8]]
in_w_ty = np.ndarray[(N_BTILES * SZ_Bb,), np.dtype[np.int8]]
c_ty = np.ndarray[(NACC * SZ_C,), np.dtype[np.int32]]

flags = ["-std=c++20", "-O2", f"-DNACC={NACC}", f"-DINNER={INNER}"]
kern = ExternalFunction("r2a_mac", source_file="../r2/r2a_gemm.cc",
                        arg_types=[a_ty, w_ty, c_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r4a(A, W, C, *, kf):
    dev = aie_utils.get_current_device()
    fa = [ObjectFifo(a_ty, name=f"fa{i}", depth=1) for i in range(COLS)]
    fw = [ObjectFifo(w_ty, name=f"fw{i}", depth=4) for i in range(COLS)]
    fc = [ObjectFifo(c_ty, name=f"fc{i}", depth=1) for i in range(COLS)]

    def make_core(kf):
        def core(a_in, w_in, c_out, kf):
            a = a_in.acquire(1)                          # resident activations
            c = c_out.acquire(1)
            for _ in range_(N_BTILES):
                wt = w_in.acquire(1)
                kf(a, wt, c)
                w_in.release(1)
            a_in.release(1)
            c_out.release(1)
        return core

    # Pin each worker to its OWN column (row 2) so each uses a distinct compute tile
    # + shim DMA channel (auto-placement stacks them on col 0 — verified in R1b).
    workers = [Worker(make_core(kf), [fa[i].cons(), fw[i].cons(), fc[i].prod(), kf],
                      tile=Tile(col=i, row=2)) for i in range(COLS)]
    rt = Runtime()
    # 3 shared BOs (A, W-stream, C) — under XRT's ~5 inout cap; all columns read the
    # same DDR region (max shared-controller contention, the realistic aggregate).
    with rt.sequence(a_ty, in_w_ty, c_ty) as (a, wstream, c):
        for w in workers:
            rt.start(w)
        for i in range(COLS):
            rt.fill(fa[i].prod(), a)
        for i in range(COLS):
            rt.fill(fw[i].prod(), wstream)
        for i in range(COLS):
            rt.drain(fc[i].cons(), c, wait=True)
    return Program(dev, rt).resolve_program()


A = randint(-8, 8, (NACC * SZ_A,), dtype=np.int8)
W = randint(-8, 8, (N_BTILES * SZ_Bb,), dtype=np.int8)
C = zeros(NACC * SZ_C, dtype=np.int32)
t = time.perf_counter()
r4a(A, W, C, kf=kern)
dt = time.perf_counter() - t
print(f"CALLMS {dt*1e3:.4f} COLS {COLS} NACC {NACC} INNER {INNER} N_BTILES {N_BTILES} "
      f"MACS {MACS} TOPS_hostwall {2*MACS/dt/1e12:.4f}")
