# R1b multi-column aggregate feed: COLS independent single-column feeds running
# CONCURRENTLY (each its own objectFIFO + worker + shim DMA channel + host BO
# slice), to find where the aggregate clears the NoC/mem-controller knee. Per
# docs/192-193 the aggregate is NOT COLS x single-column: the shim streams sum on
# a shared LPDDR5X/NoC fabric that saturates below the naive product.
#
# aggregate wall bandwidth = (COLS * per_column_bytes) / call_ms. Timed host-wall;
# use the differential slope (cols_sweep via sweep_r1b.py) to cancel the ~16 ms
# fixed per-call overhead, exactly as for the single column.
#
# Fresh process per measurement (pyxrt segfaults on repeat under py3.14).
import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.jit import jit
import aie.utils as aie_utils

INC = os.environ["MLIR_AIE_INC"]
TILE_N = int(os.environ.get("TILE_N", 4096))
N_TILES = int(os.environ["N_TILES"])
COLS = int(os.environ.get("COLS", 8))
DEPTH = int(os.environ.get("DEPTH", 4))
PER = TILE_N * N_TILES                 # bytes fed per column
TOTAL = PER * COLS                     # aggregate bytes

in_ty = np.ndarray[(PER,), np.dtype[np.int8]]
tile_ty = np.ndarray[(TILE_N,), np.dtype[np.int8]]
acc_ty = np.ndarray[(64,), np.dtype[np.int32]]

flags = ["-std=c++20", "-O2", f"-DTILE_N={TILE_N}"]
feed = ExternalFunction("feed_sum", source_file="r1b_feed.cc",
                        arg_types=[tile_ty, acc_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r1b_cols(A, Out, *, kf):
    dev = aie_utils.get_current_device()
    fins = [ObjectFifo(tile_ty, name=f"fin{i}", depth=DEPTH) for i in range(COLS)]
    fouts = [ObjectFifo(acc_ty, name=f"fout{i}", depth=1) for i in range(COLS)]

    def make_core(kf):
        def core(f_in, f_out, kf):
            acc = f_out.acquire(1)
            for _ in range_(N_TILES):
                t = f_in.acquire(1)
                kf(t, acc)
                f_in.release(1)
            f_out.release(1)
        return core

    # Pin each worker to its OWN column (row 2) so each feed uses a distinct shim
    # DMA channel — otherwise auto-placement stacks them on column 0 (verified) and
    # they share one shim, which is not a multi-column aggregate.
    workers = [Worker(make_core(kf), [fins[i].cons(), fouts[i].prod(), kf], tile=Tile(col=i, row=2))
               for i in range(COLS)]
    rt = Runtime()
    # XRT NPU caps inout buffers at ~5 (group_ids 3-7), so 2*COLS separate BOs
    # segfaults for COLS>=3. Share ONE input BO (all COLS shims read the same DDR
    # region -- maximal shared-controller contention, ideal for the knee) and ONE
    # output BO (drains are DCE guards; racing writes are fine).
    with rt.sequence(in_ty, acc_ty) as (a, o):
        for w in workers:
            rt.start(w)
        for i in range(COLS):
            rt.fill(fins[i].prod(), a)
        for i in range(COLS):
            rt.drain(fouts[i].cons(), o, wait=True)
    return Program(dev, rt).resolve_program()


A = randint(-8, 8, (PER,), dtype=np.int8)       # shared feed source (all COLS shims)
Out = zeros(64, dtype=np.int32)                 # shared DCE-guard sink
t = time.perf_counter()
r1b_cols(A, Out, kf=feed)                        # exactly one NPU run per process
dt = time.perf_counter() - t

agg_gbs = TOTAL / dt / 1e9
print(f"CALLMS {dt*1e3:.4f} TOTALB {TOTAL} COLS {COLS} PERCOL_B {PER} TILE_N {TILE_N} "
      f"NTILES {N_TILES} DEPTH {DEPTH} AGG_GBS {agg_gbs:.4f}")
