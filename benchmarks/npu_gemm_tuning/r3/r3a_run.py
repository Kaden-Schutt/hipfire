# R3a: real K-accumulating W4A8 GEMV (M=4 batched decode), one N-block. Streams A
# and W super-tiles over K, folds partials into a resident C. Measures streamed
# weight bandwidth + TOPS; for M=4 this is bandwidth-bound (each weight reused only
# 4x), so throughput should track the R1 feed (~14 GB/s/stream) — the decode case
# where int4's half-weight-bytes advantage pays off.
#
# Sustained via host-wall differential across N_SUPER (cancels the ~16 ms fixed
# cost). Fresh process per point (pyxrt segfaults on repeat under py3.14).
import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.utils.jit import jit
import aie.utils as aie_utils

INC = os.environ["MLIR_AIE_INC"]
KCHUNK = int(os.environ.get("KCHUNK", 64))     # k-blocks per streamed super-tile
N_SUPER = int(os.environ["N_SUPER"])           # super-tiles (K = N_SUPER*KCHUNK*16)
HCLK_GHZ = float(os.environ.get("HCLK_GHZ", 1.8))
SZ_A, SZ_Wb, SZ_C = 64, 128, 64                # int8 A tile / packed-int4 W tile / int32 C
A_SUPER = KCHUNK * SZ_A
W_SUPER = KCHUNK * SZ_Wb
K_BLOCKS = N_SUPER * KCHUNK
MACS = K_BLOCKS * 1024                          # 4x16x16 per k-block
STREAMED_W = N_SUPER * W_SUPER

asuper_ty = np.ndarray[(A_SUPER,), np.dtype[np.int8]]
wsuper_ty = np.ndarray[(W_SUPER,), np.dtype[np.int8]]
c_ty = np.ndarray[(SZ_C,), np.dtype[np.int32]]
ina_ty = np.ndarray[(N_SUPER * A_SUPER,), np.dtype[np.int8]]
inw_ty = np.ndarray[(N_SUPER * W_SUPER,), np.dtype[np.int8]]

flags = ["-std=c++20", "-O2", f"-DKCHUNK={KCHUNK}"]
# The kernels share r3a_gemv_common.h, which lives next to this script (mlir-aie
# copies only the .cc into the build dir, so add our dir to the include path).
HERE = os.path.dirname(os.path.abspath(__file__))
incs = [INC, HERE]
# init RESEEDS the K accumulator on the first super-tile; matvec ACCUMULATES the
# rest. Peeling the first call keeps the resident C reuse-safe across dispatches
# (else the tile-local C carries stale state from the prior dispatch).
kern_init = ExternalFunction("r3a_matvec_init", source_file="r3a_gemv_init.cc",
                             arg_types=[asuper_ty, wsuper_ty, c_ty], include_dirs=incs, compile_flags=flags)
kern = ExternalFunction("r3a_matvec", source_file="r3a_gemv.cc",
                        arg_types=[asuper_ty, wsuper_ty, c_ty], include_dirs=incs, compile_flags=flags)


@jit(use_cache=True)
def r3a(A, W, C, k_init, k):
    dev = aie_utils.get_current_device()
    of_a = ObjectFifo(asuper_ty, name="fa", depth=2)
    of_w = ObjectFifo(wsuper_ty, name="fw", depth=4)
    of_c = ObjectFifo(c_ty, name="fc", depth=1)

    def core(a_in, w_in, c_out, kk_init, kk):
        c = c_out.acquire(1)
        # Peel the first super-tile: init RESEEDS c (reuse-safe), then accumulate
        # the remaining N_SUPER-1 super-tiles into the running K partial.
        asup = a_in.acquire(1)
        wsup = w_in.acquire(1)
        kk_init(asup, wsup, c)
        a_in.release(1)
        w_in.release(1)
        for _ in range_(N_SUPER - 1):
            asup = a_in.acquire(1)
            wsup = w_in.acquire(1)
            kk(asup, wsup, c)                    # folds this super-tile into C
            a_in.release(1)
            w_in.release(1)
        c_out.release(1)

    w = Worker(core, [of_a.cons(), of_w.cons(), of_c.prod(), k_init, k])
    rt = Runtime()
    with rt.sequence(ina_ty, inw_ty, c_ty) as (a, wstream, c):
        rt.start(w)
        rt.fill(of_a.prod(), a)
        rt.fill(of_w.prod(), wstream)
        rt.drain(of_c.cons(), c, wait=True)
    return Program(dev, rt).resolve_program()


A = randint(-8, 8, (N_SUPER * A_SUPER,), dtype=np.int8)
W = randint(-8, 8, (N_SUPER * W_SUPER,), dtype=np.int8)
C = zeros(SZ_C, dtype=np.int32)
t = time.perf_counter()
r3a(A, W, C, kern_init, kern)
dt = time.perf_counter() - t

gbs = STREAMED_W / dt / 1e9
tops = 2 * MACS / dt / 1e12
print(f"CALLMS {dt*1e3:.4f} KCHUNK {KCHUNK} NSUPER {N_SUPER} KBLOCKS {K_BLOCKS} "
      f"STREAMEDW {STREAMED_W} MACS {MACS} GBS {gbs:.4f} TOPS {tops:.4f}")
