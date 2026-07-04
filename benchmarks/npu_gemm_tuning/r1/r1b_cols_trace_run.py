# R1b multi-column AGGREGATE feed via the trace unit. COLS single-column feeds run
# concurrently, each pinned to its own column (own shim MM2S), all reading a shared
# DDR region. We trace every column's S2MM receive port (PORT_RUNNING/STALLED/IDLE)
# and take the GLOBAL span (first tile received on any column -> last on any column)
# as the concurrent wall:
#     agg_gbs = (COLS * per_col_bytes) / (global_span / hclk)   [on-NPU, sync-free]
# If the columns don't contend, global_span ~= single-column span and agg scales
# ~COLS x. If the shared LPDDR5X/NoC saturates, the concurrent span stretches (per
# port shows more STALL/IDLE) and agg flattens -- the knee (docs/192-193).
#
# BO budget: shared input + shared output + trace buffer stay within XRT's ~5
# inout group_ids regardless of COLS (2*COLS separate BOs segfaults for COLS>=3).
# One fresh process per run (pyxrt segfaults on repeat under py3.14).
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
from aie.helpers.taplib import TensorTiler2D

DEV = os.environ.get("NPU_DEV", "npu2")
CoreEvent = get_events_for_device(DEV).CoreEvent
INC = os.environ["MLIR_AIE_INC"]
TILE_N = int(os.environ.get("TILE_N", 4096))
N_TILES = int(os.environ.get("N_TILES", 256))
COLS = int(os.environ.get("COLS", 8))
# Trace only a subset of columns: 8 feed flows + 8 trace flows overrun the router.
# Traced columns feel the same shared-fabric contention, so per-col rate x COLS is
# the aggregate. Default caps at 4.
TRACE_COLS = int(os.environ.get("TRACE_COLS", min(COLS, 4)))
DEPTH = int(os.environ.get("DEPTH", 4))
HCLK_MHZ = float(os.environ.get("HCLK_MHZ", 1800))
TRACE_SIZE = int(os.environ.get("TRACE_SIZE", 262144))
DDR_ID = int(os.environ.get("DDR_ID", 4))
# DISTINCT=1: one big input BO of COLS*PER, each column reads its OWN offset slice
# (distinct DDR regions -> real bank/controller contention, no shared-region
# locality). DISTINCT=0: all columns read the same PER-byte BO (locality-biased).
DISTINCT = bool(int(os.environ.get("DISTINCT", 1)))
PER = TILE_N * N_TILES
STRIDE = int(os.environ.get("STRIDE", PER))     # address gap between columns' regions
TOTAL = PER * COLS                              # bytes actually read (aggregate)
IN_ELEMS = COLS * STRIDE if DISTINCT else PER   # BO size (regions may be spread out)
TRACE_TXT, TRACE_JSON = "trace_cols.txt", "trace_cols.json"

CORE_EVENTS = [PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True),
               PortEvent(CoreEvent.PORT_STALLED_0, port=WireBundle.DMA, channel=0, master=True),
               PortEvent(CoreEvent.PORT_IDLE_0, port=WireBundle.DMA, channel=0, master=True)]

in_ty = np.ndarray[(IN_ELEMS,), np.dtype[np.int8]]
tile_ty = np.ndarray[(TILE_N,), np.dtype[np.int8]]
acc_ty = np.ndarray[(64,), np.dtype[np.int32]]

flags = ["-std=c++20", "-O2", f"-DTILE_N={TILE_N}"]
feed = ExternalFunction("feed_sum", source_file="r1b_feed.cc",
                        arg_types=[tile_ty, acc_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r1b_cols_trace(A, Out, kf, **_kw):
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

    workers = [Worker(make_core(kf), [fins[i].cons(), fouts[i].prod(), kf], tile=Tile(col=i, row=2))
               for i in range(COLS)]
    rt = Runtime()
    rt.enable_trace(trace_size=TRACE_SIZE, workers=workers[:TRACE_COLS], ddr_id=DDR_ID,
                    coretile_events=CORE_EVENTS)
    with rt.sequence(in_ty, acc_ty) as (a, o):
        for w in workers:
            rt.start(w)
        # distinct: column i reads a PER-sized tile at offset i*STRIDE. STRIDE>PER
        # spreads columns far apart in address space to test whether they land on
        # different memory controllers (aggregate would rise if we were controller-
        # bound). simple_tiler gives the linear [1,1,1,PER] BD that lowers.
        region_taps = TensorTiler2D.simple_tiler([IN_ELEMS], [PER]) if DISTINCT else None
        step = max(1, STRIDE // PER)
        for i in range(COLS):
            rt.fill(fins[i].prod(), a, tap=(region_taps[i * step] if DISTINCT else None))
        for i in range(COLS):
            rt.drain(fouts[i].cons(), o, wait=True)
    return Program(dev, rt).resolve_program()


def intervals_by_pid(ev, name):
    """Per-pid list of (start,end) for B/E pairs of `name`."""
    out = defaultdict(list)
    byp = defaultdict(list)
    for x in ev:
        if x.get("name") == name and "ts" in x:
            byp[x.get("pid", 0)].append(x)
    for pid, recs in byp.items():
        recs.sort(key=lambda x: x["ts"])
        op = None
        for x in recs:
            if x["ph"] == "B":
                op = x["ts"]
            elif x["ph"] == "E" and op is not None:
                out[pid].append((op, x["ts"])); op = None
    return out


A = randint(-8, 8, (IN_ELEMS,), dtype=np.int8)
Out = zeros(64, dtype=np.int32)
tc = TraceConfig(trace_size=TRACE_SIZE, trace_file=TRACE_TXT, ddr_id=DDR_ID)
t = time.perf_counter()
r1b_cols_trace(A, Out, feed, trace_config=tc)
dt = time.perf_counter() - t

mlir = tc.physical_mlir_path
with open(TRACE_TXT) as f:
    n_pkts = sum(1 for ln in f if ln.strip() and ln.strip() != "00000000")
p = subprocess.run([sys.executable, "-m", "aie.utils.trace.parse",
                    "--input", TRACE_TXT, "--mlir", mlir, "--output", TRACE_JSON],
                   capture_output=True, text=True)
if p.returncode != 0 or not os.path.exists(TRACE_JSON):
    sys.exit(f"ERROR: trace parse failed rc={p.returncode}\n{p.stderr[-800:]}")

ev = json.load(open(TRACE_JSON))
run_by_pid = intervals_by_pid(ev, "PORT_RUNNING_0")
n_traced = len(run_by_pid)
all_ts = [t for ivs in run_by_pid.values() for pair in ivs for t in pair]
if not all_ts:
    sys.exit(f"ERROR: no PORT_RUNNING events (names={sorted({x.get('name') for x in ev})})")
global_span = max(all_ts) - min(all_ts)
# per-column busy fraction (mean over traced cols)
busy = []
for pid, ivs in run_by_pid.items():
    run = sum(e - b for b, e in ivs)
    span = max(e for _, e in ivs) - min(b for b, _ in ivs)
    busy.append(run / span if span else 0)
mean_busy = sum(busy) / len(busy) if busy else float("nan")
overflow = n_pkts >= (TRACE_SIZE // 4) - 2

agg_gbs = TOTAL / (global_span / (HCLK_MHZ * 1e6)) / 1e9 if global_span else float("nan")
per_col_gbs = agg_gbs / COLS if COLS else float("nan")
flag = "  OVERFLOW(prefix)" if overflow else ""
print(f"COLS {COLS} DISTINCT {int(DISTINCT)} TRACED {n_traced} TOTALB {TOTAL} PERCOL_B {PER} NTILES {N_TILES} "
      f"PKTS {n_pkts} GLOBAL_SPAN {global_span} MEAN_BUSY {mean_busy:.3f} "
      f"AGG_GBS {agg_gbs:.4f} PERCOL_GBS {per_col_gbs:.4f}{flag}")
