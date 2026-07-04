import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.placers import SequentialPlacer
from aie.iron.kernel import ExternalFunction
from aie.utils.jit import jit
import aie.utils as aie_utils

INC = os.environ["MLIR_AIE_INC"]
ITERS = int(os.environ["RITERS"])
SA, SB = 32, 64
A_N, B_N, OUT_N = 2 * SA, 2 * SB, 64
A_ty = np.ndarray[(A_N,), np.dtype[np.int8]]
B_ty = np.ndarray[(B_N,), np.dtype[np.int8]]
OUT_ty = np.ndarray[(OUT_N,), np.dtype[np.int32]]

kern = ExternalFunction(
    "r0b_i8i8", source_file="r0b.cc", arg_types=[A_ty, B_ty, OUT_ty],
    include_dirs=[INC], compile_flags=["-std=c++20", "-O2", f"-DITERS={ITERS}"])


@jit(use_cache=True)
def r0b(A, B, Out, k):
    dev = aie_utils.get_current_device()
    of_a = ObjectFifo(A_ty, name="a", depth=1)
    of_b = ObjectFifo(B_ty, name="b", depth=1)
    of_o = ObjectFifo(OUT_ty, name="o", depth=1)

    def core(a_in, b_in, o_out, kk):
        ea = a_in.acquire(1); eb = b_in.acquire(1); eo = o_out.acquire(1)
        kk(ea, eb, eo)
        a_in.release(1); b_in.release(1); o_out.release(1)

    w = Worker(core, [of_a.cons(), of_b.cons(), of_o.prod(), k])
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, OUT_ty) as (a, b, o):
        rt.start(w)
        rt.fill(of_a.prod(), a)
        rt.fill(of_b.prod(), b)
        rt.drain(of_o.cons(), o, wait=True)
    return Program(dev, rt).resolve_program(SequentialPlacer())


A = randint(-8, 8, (A_N,), dtype=np.int8)
B = randint(-8, 8, (B_N,), dtype=np.int8)
Out = zeros(OUT_N, dtype=np.int32)
t = time.perf_counter()
r0b(A, B, Out, kern)                          # exactly one NPU run per process
dt = time.perf_counter() - t
print(f"CALLMS {dt*1e3:.4f} OUT0 {int(Out.numpy()[0])}")
