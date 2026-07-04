# R3c: compute-bound prefill W4A8 GEMM. Activation A (M=NACC*4 x K=KFULL*16) stays
# RESIDENT; only weights stream, one N-block at a time, reusing A. Arithmetic
# intensity NACC*8 MACs/byte (NACC=8 -> 64) should push throughput to the ~512
# MAC/cyc compute ceiling, not the feed -- the real prefill case.
#
# Sustained via host-wall differential across N_NBLOCKS. Fresh process per point.
import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.utils.jit import jit
import aie.utils as aie_utils

INC = os.environ["MLIR_AIE_INC"]
KFULL = int(os.environ.get("KFULL", 32))
NACC = int(os.environ.get("NACC", 8))
N_NBLOCKS = int(os.environ["N_NBLOCKS"])
HCLK_GHZ = float(os.environ.get("HCLK_GHZ", 1.8))
SZ_A, SZ_Wb, SZ_C = 64, 128, 64
A_RES = KFULL * NACC * SZ_A       # resident activations
W_NB = KFULL * SZ_Wb              # one N-block's weights
MACS = N_NBLOCKS * KFULL * NACC * 1024
STREAMED_W = N_NBLOCKS * W_NB

a_ty = np.ndarray[(A_RES,), np.dtype[np.int8]]
wnb_ty = np.ndarray[(W_NB,), np.dtype[np.int8]]
c_ty = np.ndarray[(NACC * SZ_C,), np.dtype[np.int32]]
inw_ty = np.ndarray[(N_NBLOCKS * W_NB,), np.dtype[np.int8]]

flags = ["-std=c++20", "-O2", f"-DKFULL={KFULL}", f"-DNACC={NACC}"]
kern = ExternalFunction("r3c_mac", source_file="r3c_gemm.cc",
                        arg_types=[a_ty, wnb_ty, c_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r3c(A, W, C, k):
    dev = aie_utils.get_current_device()
    of_a = ObjectFifo(a_ty, name="fa", depth=1)
    of_w = ObjectFifo(wnb_ty, name="fw", depth=4)
    of_c = ObjectFifo(c_ty, name="fc", depth=1)

    def core(a_in, w_in, c_out, kk):
        a = a_in.acquire(1)            # resident activations, reused
        c = c_out.acquire(1)
        for _ in range_(N_NBLOCKS):
            wnb = w_in.acquire(1)
            kk(a, wnb, c)
            w_in.release(1)
        a_in.release(1)
        c_out.release(1)

    w = Worker(core, [of_a.cons(), of_w.cons(), of_c.prod(), k])
    rt = Runtime()
    with rt.sequence(a_ty, inw_ty, c_ty) as (a, wstream, c):
        rt.start(w)
        rt.fill(of_a.prod(), a)
        rt.fill(of_w.prod(), wstream)
        rt.drain(of_c.cons(), c, wait=True)
    return Program(dev, rt).resolve_program()


A = randint(-8, 8, (A_RES,), dtype=np.int8)
W = randint(-8, 8, (N_NBLOCKS * W_NB,), dtype=np.int8)
C = zeros(NACC * SZ_C, dtype=np.int32)
t = time.perf_counter()
r3c(A, W, C, kern)
dt = time.perf_counter() - t

tops = 2 * MACS / dt / 1e12
mac_per_cyc = (MACS / dt) / (HCLK_GHZ * 1e9)
gbs = STREAMED_W / dt / 1e9
print(f"CALLMS {dt*1e3:.4f} NACC {NACC} M {NACC*4} KFULL {KFULL} K {KFULL*16} NNB {N_NBLOCKS} "
      f"MACS {MACS} TOPS {tops:.4f} MAC_PER_CYC {mac_per_cyc:.0f} WGBS {gbs:.4f}")
