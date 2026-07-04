# R2a: single-core W4A8 GEMM compute fused with weight streaming. Streams N_BTILES
# int4 weight tiles (128 B each) from L3 into one core; per tile the core does
# (INNER+1)*NACC mac_4x16_16x16 instrs (1024 MACs each) against NACC resident int8
# activation tiles. Sustained MAC rate + arithmetic intensity (MACs/streamed-byte)
# show where the core crosses from feed-bound to compute-bound (II=1).
#
# Measured host-wall; use the differential across N_BTILES (cancels the ~16 ms
# fixed per-call overhead) for the true per-tile time. Fresh process per point
# (pyxrt segfaults on repeat under py3.14).
import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.utils.jit import jit
import aie.utils as aie_utils

INC = os.environ["MLIR_AIE_INC"]
NACC = int(os.environ.get("NACC", 8))
INNER = int(os.environ.get("INNER", 64))
N_BTILES = int(os.environ["N_BTILES"])
INT8W = bool(os.environ.get("INT8W"))
HCLK_GHZ = float(os.environ.get("HCLK_GHZ", 1.8))
# int4: mmul<4,16,16> size_A=64 i8, size_B=256 i4 (128 B), 1024 MACs/mac
# int8: mmul<8,8,8>   size_A=64 i8, size_B=64 i8 (64 B),   512 MACs/mac
SZ_A, SZ_C = 64, 64
SZ_Bb = 64 if INT8W else 128
MACS_PER_MAC = 512 if INT8W else 1024
MACS_PER_TILE = (INNER + 1) * NACC * MACS_PER_MAC
STREAMED_B = N_BTILES * SZ_Bb                       # weight bytes fed

a_ty = np.ndarray[(NACC * SZ_A,), np.dtype[np.int8]]
w_ty = np.ndarray[(SZ_Bb,), np.dtype[np.int8]]        # one weight tile, packed int4
in_ty = np.ndarray[(N_BTILES * SZ_Bb,), np.dtype[np.int8]]
c_ty = np.ndarray[(NACC * SZ_C,), np.dtype[np.int32]]

flags = ["-std=c++20", "-O2", f"-DNACC={NACC}", f"-DINNER={INNER}"] + (["-DINT8W"] if INT8W else [])
kern = ExternalFunction("r2a_mac", source_file="r2a_gemm.cc",
                        arg_types=[a_ty, w_ty, c_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r2a(A, W, C, k):
    dev = aie_utils.get_current_device()
    of_a = ObjectFifo(a_ty, name="fa", depth=1)
    of_w = ObjectFifo(w_ty, name="fw", depth=4)
    of_c = ObjectFifo(c_ty, name="fc", depth=1)

    def core(a_in, w_in, c_out, kk):
        a = a_in.acquire(1)                          # resident activations
        c = c_out.acquire(1)
        for _ in range_(N_BTILES):
            wt = w_in.acquire(1)
            kk(a, wt, c)
            w_in.release(1)
        a_in.release(1)
        c_out.release(1)

    w = Worker(core, [of_a.cons(), of_w.cons(), of_c.prod(), k])
    rt = Runtime()
    with rt.sequence(a_ty, in_ty, c_ty) as (a, wstream, c):
        rt.start(w)
        rt.fill(of_a.prod(), a)                      # activations once
        rt.fill(of_w.prod(), wstream)                # weight tiles streamed
        rt.drain(of_c.cons(), c, wait=True)
    return Program(dev, rt).resolve_program()


A = randint(-8, 8, (NACC * SZ_A,), dtype=np.int8)
W = randint(-8, 8, (N_BTILES * SZ_Bb,), dtype=np.int8)
C = zeros(NACC * SZ_C, dtype=np.int32)
t = time.perf_counter()
r2a(A, W, C, kern)
dt = time.perf_counter() - t

total_macs = N_BTILES * MACS_PER_TILE
tops = 2 * total_macs / dt / 1e12
gmacs = total_macs / dt / 1e9
mac_per_cyc = gmacs / HCLK_GHZ
wbits = 8 if INT8W else 4
print(f"CALLMS {dt*1e3:.4f} W{wbits}A8 NACC {NACC} INNER {INNER} NBTILES {N_BTILES} "
      f"MACS {total_macs} TOPS {tops:.4f} GMAC_S {gmacs:.1f} MAC_PER_CYC {mac_per_cyc:.0f}")
