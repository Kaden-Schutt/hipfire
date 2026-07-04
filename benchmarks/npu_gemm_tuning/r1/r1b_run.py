# R1b M1: single-column L3->L1 feed, host-wall timed (same measurement R1a made),
# but revived on the current mlir-aie toolchain and driven across a DEPTH sweep.
#
# Why only M1 here (see r1/README.md "Three-way, and why M2 collapsed into M3"):
#   - M2 (in-kernel core-cycle timer) does NOT link on this toolchain:
#     aie::tile::current().cycles() lowers to an undefined ::get_cycles().
#   - Host-side 194 fencing is not cleanly reachable either: IRON's runner bundles
#     BO sync + execute + sync in one concrete run() call.
#   => the host-vs-device split needs the trace unit (M3, r1b_trace_run.py).
#
# What M1 + the DEPTH sweep still buys us: the busy-vs-idle proxy. If GB/s rises
# with FIFO DEPTH, the feed is handshake/latency-bound (core idling on acquire,
# more in-flight BDs help within the 16-BD budget); if GB/s is flat in DEPTH, the
# DMA is continuously busy = bandwidth-bound (lever: nd-descriptors / more columns).
#
# Fresh process per measurement (pyxrt segfaults on repeat under py3.14).
import os, time, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.utils.jit import jit
import aie.utils as aie_utils

# mlir-aie >= 2026-05 places tiles automatically (--aie-place-tiles pass);
# resolve_program() takes no placer (older builds needed SequentialPlacer()).
INC = os.environ["MLIR_AIE_INC"]
TILE_N = int(os.environ.get("TILE_N", 4096))          # int8 per tile
N_TILES = int(os.environ["N_TILES"])                  # tiles streamed from L3
TOTAL = TILE_N * N_TILES                               # total int8 bytes fed
DEPTH = int(os.environ.get("DEPTH", 4))
MINIMAL = bool(os.environ.get("MINIMAL"))

in_ty = np.ndarray[(TOTAL,), np.dtype[np.int8]]
tile_ty = np.ndarray[(TILE_N,), np.dtype[np.int8]]
acc_ty = np.ndarray[(64,), np.dtype[np.int32]]

flags = ["-std=c++20", "-O2", f"-DTILE_N={TILE_N}"] + (["-DMINIMAL"] if MINIMAL else [])
feed = ExternalFunction("feed_sum", source_file="r1b_feed.cc",
                        arg_types=[tile_ty, acc_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r1b(A, Out, kf):
    dev = aie_utils.get_current_device()
    of_in = ObjectFifo(tile_ty, name="fin", depth=DEPTH)
    of_out = ObjectFifo(acc_ty, name="fout", depth=1)

    def core(f_in, f_out, kf):
        acc = f_out.acquire(1)
        for _ in range_(N_TILES):
            t = f_in.acquire(1)
            kf(t, acc)
            f_in.release(1)
        f_out.release(1)

    w = Worker(core, [of_in.cons(), of_out.prod(), kf])
    rt = Runtime()
    with rt.sequence(in_ty, acc_ty) as (a, o):
        rt.start(w)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), o, wait=True)
    return Program(dev, rt).resolve_program()


A = randint(-8, 8, (TOTAL,), dtype=np.int8)
Out = zeros(64, dtype=np.int32)
t = time.perf_counter()
r1b(A, Out, feed)                             # exactly one NPU run per process
dt = time.perf_counter() - t

host_gbs = TOTAL / dt / 1e9
print(f"CALLMS {dt*1e3:.4f} TOTALB {TOTAL} TILE_N {TILE_N} NTILES {N_TILES} "
      f"DEPTH {DEPTH} MINIMAL {int(MINIMAL)} HOST_GBS {host_gbs:.4f}")
