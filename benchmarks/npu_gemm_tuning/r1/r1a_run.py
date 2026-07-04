import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, Kernel, zeros, randint
from aie.iron.placers import SequentialPlacer
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.utils.jit import jit
import aie.utils as aie_utils

INC = os.environ["MLIR_AIE_INC"]
TILE_N = int(os.environ.get("TILE_N", 4096))          # int8 per tile
N_TILES = int(os.environ["N_TILES"])                  # tiles streamed from L3
TOTAL = TILE_N * N_TILES                               # total int8 bytes fed
DEPTH = int(os.environ.get("DEPTH", 4))

in_ty = np.ndarray[(TOTAL,), np.dtype[np.int8]]
tile_ty = np.ndarray[(TILE_N,), np.dtype[np.int8]]
acc_ty = np.ndarray[(64,), np.dtype[np.int32]]

kern = ExternalFunction(
    "feed_sum", source_file="r1a_feed.cc", arg_types=[tile_ty, acc_ty],
    include_dirs=[INC], compile_flags=["-std=c++20", "-O2", f"-DTILE_N={TILE_N}"] + (["-DMINIMAL"] if os.environ.get("MINIMAL") else []))


@jit(use_cache=True)
def r1a(A, Out, k):
    dev = aie_utils.get_current_device()
    of_in = ObjectFifo(tile_ty, name="fin", depth=DEPTH)
    of_out = ObjectFifo(acc_ty, name="fout", depth=1)

    def core(f_in, f_out, kk):
        acc = f_out.acquire(1)
        for _ in range_(N_TILES):
            t = f_in.acquire(1)
            kk(t, acc)
            f_in.release(1)
        f_out.release(1)

    w = Worker(core, [of_in.cons(), of_out.prod(), k])
    rt = Runtime()
    with rt.sequence(in_ty, acc_ty) as (a, o):
        rt.start(w)
        rt.fill(of_in.prod(), a)          # stream TOTAL bytes from L3 in TILE_N chunks
        rt.drain(of_out.cons(), o, wait=True)
    return Program(dev, rt).resolve_program(SequentialPlacer())


A = randint(-8, 8, (TOTAL,), dtype=np.int8)
Out = zeros(64, dtype=np.int32)
t = time.perf_counter()
r1a(A, Out, kern)
dt = time.perf_counter() - t
print(f"CALLMS {dt*1e3:.4f} TOTALB {TOTAL} NTILES {N_TILES}")
