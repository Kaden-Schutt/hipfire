# R1b stream-density probe: is ~56 GB/s a stream-DELIVERY ceiling or the real
# NoC/DDR ceiling? Single-column feed tops out at 8 B/cyc = one AIE stream's width,
# and a shim's DDR-read side is wider than one stream. We use 1 stream/shim; put
# ROWS streams per column (aie2p shim has 2 MM2S channels, rows 2..2+ROWS-1) so the
# shim issues more parallel DDR reads. If aggregate climbs past ~56 with 16 streams
# (COLS=8, ROWS=2), 56 was stream-delivery-bound and DDR has headroom; if it stays
# ~56, that's the shared NoC/DDR ceiling.
#
# S = COLS*ROWS streams, each feeds PER bytes; aggregate = S*PER / global span.
# Shared input BO (XRT ~5-buffer limit). Trace a subset (TRACE_N) — router can't
# route trace on all streams. Fresh process per run (pyxrt segfaults on repeat).
import os, sys, time, subprocess, json, numpy as np
from collections import defaultdict
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.jit import jit
import aie.utils as aie_utils
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import get_events_for_device, PortEvent
from aie.dialects.aie import WireBundle

DEV = os.environ.get("NPU_DEV", "npu2")
CoreEvent = get_events_for_device(DEV).CoreEvent
INC = os.environ["MLIR_AIE_INC"]
TILE_N = int(os.environ.get("TILE_N", 4096))
N_TILES = int(os.environ.get("N_TILES", 256))
COLS = int(os.environ.get("COLS", 8))
ROWS = int(os.environ.get("ROWS", 2))          # streams per column (shim MM2S ch)
S = COLS * ROWS
DEPTH = int(os.environ.get("DEPTH", 4))
HCLK_MHZ = float(os.environ.get("HCLK_MHZ", 1800))
TRACE_SIZE = int(os.environ.get("TRACE_SIZE", 262144))
TRACE_N = int(os.environ.get("TRACE_N", 4))    # how many streams to trace
DDR_ID = int(os.environ.get("DDR_ID", 4))
PER = TILE_N * N_TILES
TOTAL = PER * S
TRACE_TXT, TRACE_JSON = "trace_s.txt", "trace_s.json"

CORE_EVENTS = [PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True),
               PortEvent(CoreEvent.PORT_STALLED_0, port=WireBundle.DMA, channel=0, master=True),
               PortEvent(CoreEvent.PORT_IDLE_0, port=WireBundle.DMA, channel=0, master=True)]

in_ty = np.ndarray[(PER,), np.dtype[np.int8]]
tile_ty = np.ndarray[(TILE_N,), np.dtype[np.int8]]
acc_ty = np.ndarray[(64,), np.dtype[np.int32]]

flags = ["-std=c++20", "-O2", f"-DTILE_N={TILE_N}"]
feed = ExternalFunction("feed_sum", source_file="r1b_feed.cc",
                        arg_types=[tile_ty, acc_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r1b_streams(A, Out, kf, **_kw):
    dev = aie_utils.get_current_device()
    fins = [ObjectFifo(tile_ty, name=f"fin{i}", depth=DEPTH) for i in range(S)]
    fouts = [ObjectFifo(acc_ty, name=f"fout{i}", depth=1) for i in range(S)]

    def make_core(kf):
        def core(f_in, f_out, kf):
            acc = f_out.acquire(1)
            for _ in range_(N_TILES):
                t = f_in.acquire(1)
                kf(t, acc)
                f_in.release(1)
            f_out.release(1)
        return core

    # stream s -> column s//ROWS, compute row 2 + s%ROWS (so ROWS streams share a
    # column/shim, using its distinct MM2S channels).
    # spread the ROWS streams of a column across non-adjacent compute rows (2,4,..)
    # so the shim->core stream router doesn't collide on one South channel.
    row_of = [2, 4, 3, 5]
    workers = [Worker(make_core(kf), [fins[s].cons(), fouts[s].prod(), kf],
                      tile=Tile(col=s // ROWS, row=row_of[s % ROWS])) for s in range(S)]
    rt = Runtime()
    if TRACE_N > 0:
        rt.enable_trace(trace_size=TRACE_SIZE, workers=workers[:TRACE_N], ddr_id=DDR_ID,
                        coretile_events=CORE_EVENTS)
    with rt.sequence(in_ty, acc_ty) as (a, o):
        for w in workers:
            rt.start(w)
        for s in range(S):
            rt.fill(fins[s].prod(), a)
        for s in range(S):
            rt.drain(fouts[s].cons(), o, wait=True)
    return Program(dev, rt).resolve_program()


def intervals_by_pid(ev, name):
    byp = defaultdict(list)
    for x in ev:
        if x.get("name") == name and "ts" in x:
            byp[x.get("pid", 0)].append(x)
    out = defaultdict(list)
    for pid, recs in byp.items():
        recs.sort(key=lambda x: x["ts"]); op = None
        for x in recs:
            if x["ph"] == "B":
                op = x["ts"]
            elif x["ph"] == "E" and op is not None:
                out[pid].append((op, x["ts"])); op = None
    return out


A = randint(-8, 8, (PER,), dtype=np.int8)
Out = zeros(64, dtype=np.int32)
tc = TraceConfig(trace_size=TRACE_SIZE, trace_file=TRACE_TXT, ddr_id=DDR_ID)
t = time.perf_counter()
r1b_streams(A, Out, feed, trace_config=tc)
dt = time.perf_counter() - t

if TRACE_N == 0:
    # No trace: host-wall aggregate only (use a big total to clear ~16 ms fixed).
    print(f"COLS {COLS} ROWS {ROWS} STREAMS {S} TOTALB {TOTAL} NOTRACE CALLMS {dt*1e3:.3f} "
          f"HOST_AGG_GBS {TOTAL/dt/1e9:.4f}")
    sys.exit(0)

mlir = tc.physical_mlir_path
with open(TRACE_TXT) as f:
    n_pkts = sum(1 for ln in f if ln.strip() and ln.strip() != "00000000")
p = subprocess.run([sys.executable, "-m", "aie.utils.trace.parse",
                    "--input", TRACE_TXT, "--mlir", mlir, "--output", TRACE_JSON],
                   capture_output=True, text=True)
if p.returncode != 0 or not os.path.exists(TRACE_JSON):
    sys.exit(f"ERROR: parse rc={p.returncode}\n{p.stderr[-800:]}")

ev = json.load(open(TRACE_JSON))
run_by_pid = intervals_by_pid(ev, "PORT_RUNNING_0")
all_ts = [t for ivs in run_by_pid.values() for pair in ivs for t in pair]
if not all_ts:
    sys.exit(f"ERROR: no PORT_RUNNING (names={sorted({x.get('name') for x in ev})})")
span = max(all_ts) - min(all_ts)
busy = [sum(e - b for b, e in ivs) / (max(e for _, e in ivs) - min(b for b, _ in ivs))
        for ivs in run_by_pid.values() if len(ivs) > 1]
mean_busy = sum(busy) / len(busy) if busy else float("nan")
overflow = n_pkts >= (TRACE_SIZE // 4) - 2
agg = TOTAL / (span / (HCLK_MHZ * 1e6)) / 1e9 if span else float("nan")
print(f"COLS {COLS} ROWS {ROWS} STREAMS {S} TRACED {len(run_by_pid)} TOTALB {TOTAL} "
      f"SPAN {span} MEAN_BUSY {mean_busy:.3f} AGG_GBS {agg:.4f} PERSTREAM_GBS {agg/S:.4f}"
      f"{'  OVERFLOW' if overflow else ''}")
