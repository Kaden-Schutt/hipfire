# R4b: FULL-ARRAY aggregate W4A8 compute via per-column MEMTILE routing. R4a used
# only 8 of 32 cores because feeding each core straight from its column shim
# exhausts shim DMA capacity past ~1 core/column. Here each column's shim drives
# the MEMTILE (tile row 1), which fans out to the ROWS compute cores:
#   * W is BROADCAST to all ROWS cores (one shim stream, reused ROWS x — the R4
#     weight-stationary lever) via .forward() + multiple .cons();
#   * A is DISTRIBUTED as ROWS distinct activation blocks via .split();
#   * C is GATHERED from the ROWS cores via .join().
# So 3 shim channels/column regardless of ROWS. Cores run the R2a II=1 compute core
# (fake INNER reuse) so the array is COMPUTE-bound — measuring the whole-array
# ceiling. Build the xclbin here; measure TOPS through NpuKernel (npu_gemm_bench).
import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.jit import jit
import aie.utils as aie_utils

INC = os.environ["MLIR_AIE_INC"]
COLS = int(os.environ.get("COLS", 8))
ROWS = int(os.environ.get("ROWS", 4))
NACC = int(os.environ.get("NACC", 4))
INNER = int(os.environ.get("INNER", 64))
N_BTILES = int(os.environ["N_BTILES"])
N_CORES = ROWS * COLS
SZ_A, SZ_C, SZ_Bb = 64, 64, 128
MACS = N_CORES * N_BTILES * (INNER + 1) * NACC * 1024

a_ty = np.ndarray[(NACC * SZ_A,), np.dtype[np.int8]]           # per-core A block
w_ty = np.ndarray[(SZ_Bb,), np.dtype[np.int8]]                 # one int4 weight tile
in_w_ty = np.ndarray[(N_BTILES * SZ_Bb,), np.dtype[np.int8]]   # W stream (per column)
c_ty = np.ndarray[(NACC * SZ_C,), np.dtype[np.int32]]          # per-core C block
a_col_ty = np.ndarray[(ROWS * NACC * SZ_A,), np.dtype[np.int8]]  # ROWS A blocks / column
c_col_ty = np.ndarray[(ROWS * NACC * SZ_C,), np.dtype[np.int32]]  # ROWS C blocks / column

flags = ["-std=c++20", "-O2", f"-DNACC={NACC}", f"-DINNER={INNER}"]
kern = ExternalFunction("r2a_mac", source_file="../r2/r2a_gemm.cc",
                        arg_types=[a_ty, w_ty, c_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r4b(A, W, C, *, kf):
    dev = aie_utils.get_current_device()

    def make_core(kf):
        def core(a_in, w_in, c_out, kf):
            a = a_in.acquire(1)
            c = c_out.acquire(1)
            for _ in range_(N_BTILES):
                wt = w_in.acquire(1)
                kf(a, wt, c)
                w_in.release(1)
            a_in.release(1)
            c_out.release(1)
        return core

    workers = []
    wl3s, al3s, cl3s = [], [], []
    for col in range(COLS):
        mt = Tile(col=col, row=1)  # this column's mem tile

        # W: shim -> memtile forward, then broadcast to ROWS cores.
        wl3 = ObjectFifo(w_ty, name=f"wl3_{col}", depth=4)
        wl2 = wl3.cons().forward(tile=mt, obj_type=w_ty, name=f"wl2_{col}", depth=4)
        wl3s.append(wl3)

        # A: shim -> memtile split into ROWS distinct blocks.
        al3 = ObjectFifo(a_col_ty, name=f"al3_{col}", depth=1)
        al2 = al3.cons().split([r * NACC * SZ_A for r in range(ROWS)], tile=mt,
                               obj_types=[a_ty] * ROWS,
                               names=[f"al2_{col}_{r}" for r in range(ROWS)])
        al3s.append(al3)

        # C: ROWS cores -> memtile join -> shim.
        cl3 = ObjectFifo(c_col_ty, name=f"cl3_{col}", depth=1)
        cl2 = cl3.prod().join([r * NACC * SZ_C for r in range(ROWS)], tile=mt,
                              obj_types=[c_ty] * ROWS,
                              names=[f"cl2_{col}_{r}" for r in range(ROWS)])
        cl3s.append(cl3)

        for r in range(ROWS):
            workers.append(Worker(make_core(kf),
                                  [al2[r].cons(), wl2.cons(), cl2[r].prod(), kf],
                                  tile=Tile(col=col, row=2 + r)))

    rt = Runtime()
    with rt.sequence(a_col_ty, in_w_ty, c_col_ty) as (a, wstream, c):
        for w in workers:
            rt.start(w)
        for col in range(COLS):
            rt.fill(al3s[col].prod(), a)
            rt.fill(wl3s[col].prod(), wstream)
        for col in range(COLS):
            rt.drain(cl3s[col].cons(), c, wait=True)
    return Program(dev, rt).resolve_program()


A = randint(-8, 8, (ROWS * NACC * SZ_A,), dtype=np.int8)
W = randint(-8, 8, (N_BTILES * SZ_Bb,), dtype=np.int8)
C = zeros(ROWS * NACC * SZ_C, dtype=np.int32)
t = time.perf_counter()
r4b(A, W, C, kf=kern)
dt = time.perf_counter() - t
print(f"CALLMS {dt*1e3:.4f} CORES {N_CORES} (ROWS {ROWS} COLS {COLS}) NACC {NACC} INNER {INNER} "
      f"N_BTILES {N_BTILES} MACS {MACS} TOPS_hostwall {2*MACS/dt/1e12:.4f}")
