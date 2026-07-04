# R3b: real K-accumulating W4A8 GEMM (prefill), one N-block, NACC row-blocks
# (M = NACC*4 rows). Streams A ([k][block]) and W super-tiles over K, folds into
# NACC resident C accumulators. NACC controls weight reuse (= arithmetic intensity
# NACC*8 MACs/byte): NACC=4 is feed-bound single-stream, NACC=8 compute-bound.
#
# Sustained via host-wall differential across N_SUPER. Fresh process per point.
import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.utils.jit import jit
import aie.utils as aie_utils

INC = os.environ["MLIR_AIE_INC"]
KCHUNK = int(os.environ.get("KCHUNK", 64))
NACC = int(os.environ.get("NACC", 8))
N_SUPER = int(os.environ["N_SUPER"])
HCLK_GHZ = float(os.environ.get("HCLK_GHZ", 1.8))
SZ_A, SZ_Wb, SZ_C = 64, 128, 64
A_SUPER = KCHUNK * NACC * SZ_A
W_SUPER = KCHUNK * SZ_Wb
K_BLOCKS = N_SUPER * KCHUNK
MACS = K_BLOCKS * NACC * 1024
STREAMED_W = N_SUPER * W_SUPER

asuper_ty = np.ndarray[(A_SUPER,), np.dtype[np.int8]]
wsuper_ty = np.ndarray[(W_SUPER,), np.dtype[np.int8]]
c_ty = np.ndarray[(NACC * SZ_C,), np.dtype[np.int32]]
ina_ty = np.ndarray[(N_SUPER * A_SUPER,), np.dtype[np.int8]]
inw_ty = np.ndarray[(N_SUPER * W_SUPER,), np.dtype[np.int8]]

flags = ["-std=c++20", "-O2", f"-DKCHUNK={KCHUNK}", f"-DNACC={NACC}"]
kern = ExternalFunction("r3b_mac", source_file="r3b_gemm.cc",
                        arg_types=[asuper_ty, wsuper_ty, c_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r3b(A, W, C, k):
    dev = aie_utils.get_current_device()
    of_a = ObjectFifo(asuper_ty, name="fa", depth=2)
    of_w = ObjectFifo(wsuper_ty, name="fw", depth=4)
    of_c = ObjectFifo(c_ty, name="fc", depth=1)

    def core(a_in, w_in, c_out, kk):
        c = c_out.acquire(1)
        for _ in range_(N_SUPER):
            asup = a_in.acquire(1); wsup = w_in.acquire(1)
            kk(asup, wsup, c)
            a_in.release(1); w_in.release(1)
        c_out.release(1)

    w = Worker(core, [of_a.cons(), of_w.cons(), of_c.prod(), k])
    rt = Runtime()
    with rt.sequence(ina_ty, inw_ty, c_ty) as (a, wstream, c):
        rt.start(w)
        rt.fill(of_a.prod(), a)
        rt.fill(of_w.prod(), wstream)
        rt.drain(of_c.cons(), c, wait=True)
    return Program(dev, rt).resolve_program()


A = randint(-8, 8, (N_SUPER * A_SUPER,), dtype=np.int8)
W = randint(-8, 8, (N_SUPER * W_SUPER,), dtype=np.int8)
C = zeros(NACC * SZ_C, dtype=np.int32)
t = time.perf_counter()
r3b(A, W, C, kern)
dt = time.perf_counter() - t

gbs = STREAMED_W / dt / 1e9
tops = 2 * MACS / dt / 1e12
mac_per_cyc = (MACS / dt) / (HCLK_GHZ * 1e9)
print(f"CALLMS {dt*1e3:.4f} NACC {NACC} M {NACC*4} KCHUNK {KCHUNK} NSUPER {N_SUPER} "
      f"MACS {MACS} WGBS {gbs:.4f} TOPS {tops:.4f} MAC_PER_CYC {mac_per_cyc:.0f}")
