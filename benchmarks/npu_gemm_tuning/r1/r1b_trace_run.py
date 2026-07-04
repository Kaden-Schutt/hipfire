# R1b M3: on-NPU feed bandwidth via the trace unit -- the measurement that isolates
# the feed from host BO sync (host->device sync precedes kernel start, so it is NOT
# in the trace window). We trace the compute tile's S2MM ch0 (the feed-receive port)
# with PORT_RUNNING / PORT_STALLED / PORT_IDLE, giving both:
#   - span      = last_ts - first_ts over port events  = on-NPU feed duration
#                 => FEED_GBS = TOTAL / (span / hclk), host-sync-free
#   - busy_frac = sum(PORT_RUNNING intervals) / span   = DMA busy fraction
#                 high busy_frac => bandwidth-bound; low (high stall/idle) => the
#                 upstream shim feed can't keep the port fed (handshake/DDR-bound).
#
# The trace buffer is finite: keep TOTAL small enough that all port events fit
# (overflow is detected and reported). One fresh process per run (pyxrt segfaults
# on repeat under py3.14).
import os, sys, time, subprocess, json, numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker, zeros, randint
from aie.iron.kernel import ExternalFunction
from aie.iron.controlflow import range_
from aie.utils.jit import jit
import aie.utils as aie_utils
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import get_events_for_device, PortEvent
from aie.dialects.aie import WireBundle

DEV = os.environ.get("NPU_DEV", "npu2")
CoreEvent = get_events_for_device(DEV).CoreEvent
INC = os.environ["MLIR_AIE_INC"]
TILE_N = int(os.environ.get("TILE_N", 4096))
N_TILES = int(os.environ.get("N_TILES", 256))      # keep small so the trace fits
TOTAL = TILE_N * N_TILES
DEPTH = int(os.environ.get("DEPTH", 4))
HCLK_MHZ = float(os.environ.get("HCLK_MHZ", 1800))
TRACE_SIZE = int(os.environ.get("TRACE_SIZE", 65536))   # bytes; TRACE_SIZE/4 = max packets
DDR_ID = int(os.environ.get("DDR_ID", 4))
TRACE_TXT = os.environ.get("TRACE_TXT", "trace.txt")
TRACE_JSON = os.environ.get("TRACE_JSON", "trace.json")

# Feed-receive port = compute-tile S2MM ch0 (master=True => S2MM).
CORE_EVENTS = [PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True),
               PortEvent(CoreEvent.PORT_STALLED_0, port=WireBundle.DMA, channel=0, master=True),
               PortEvent(CoreEvent.PORT_IDLE_0, port=WireBundle.DMA, channel=0, master=True)]

in_ty = np.ndarray[(TOTAL,), np.dtype[np.int8]]
tile_ty = np.ndarray[(TILE_N,), np.dtype[np.int8]]
acc_ty = np.ndarray[(64,), np.dtype[np.int32]]

flags = ["-std=c++20", "-O2", f"-DTILE_N={TILE_N}"]
feed = ExternalFunction("feed_sum", source_file="r1b_feed.cc",
                        arg_types=[tile_ty, acc_ty], include_dirs=[INC], compile_flags=flags)


@jit(use_cache=True)
def r1b_trace(A, Out, kf, **_kw):    # **_kw absorbs trace_config (jit forwards it)
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
    rt.enable_trace(trace_size=TRACE_SIZE, workers=[w], ddr_id=DDR_ID, coretile_events=CORE_EVENTS)
    with rt.sequence(in_ty, acc_ty) as (a, o):
        rt.start(w)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), o, wait=True)
    return Program(dev, rt).resolve_program()


def sum_intervals(ev, name):
    """Sum (E.ts - B.ts) over B/E pairs for `name`, in ts order."""
    rec = sorted((x for x in ev if x.get("name") == name and "ts" in x), key=lambda x: x["ts"])
    total, open_ts = 0, None
    for x in rec:
        if x["ph"] == "B":
            open_ts = x["ts"]
        elif x["ph"] == "E" and open_ts is not None:
            total += x["ts"] - open_ts
            open_ts = None
    return total


A = randint(-8, 8, (TOTAL,), dtype=np.int8)
Out = zeros(64, dtype=np.int32)
tc = TraceConfig(trace_size=TRACE_SIZE, trace_file=TRACE_TXT, ddr_id=DDR_ID)
t = time.perf_counter()
r1b_trace(A, Out, feed, trace_config=tc)
dt = time.perf_counter() - t

mlir = tc.physical_mlir_path
if not mlir or not os.path.exists(mlir):
    sys.exit(f"ERROR: physical MLIR not set/found ({mlir}); trace not configured?")
with open(TRACE_TXT) as f:
    n_pkts = sum(1 for ln in f if ln.strip() and ln.strip() != "00000000")
p = subprocess.run([sys.executable, "-m", "aie.utils.trace.parse",
                    "--input", TRACE_TXT, "--mlir", mlir, "--output", TRACE_JSON],
                   capture_output=True, text=True)
if p.returncode != 0 or not os.path.exists(TRACE_JSON):
    sys.exit(f"ERROR: trace parse failed rc={p.returncode}\n{p.stderr[-800:]}")

ev = json.load(open(TRACE_JSON))
ts = [x["ts"] for x in ev if isinstance(x.get("ts"), (int, float)) and x.get("name", "").startswith("PORT")]
if not ts:
    sys.exit(f"ERROR: no PORT events in trace (got {sorted({x.get('name') for x in ev})})")
span = max(ts) - min(ts)
run_cyc = sum_intervals(ev, "PORT_RUNNING_0")
stall_cyc = sum_intervals(ev, "PORT_STALLED_0")
idle_cyc = sum_intervals(ev, "PORT_IDLE_0")
overflow = n_pkts >= (TRACE_SIZE // 4) - 2      # buffer full => only a prefix captured

busy_frac = run_cyc / span if span else float("nan")
feed_gbs_span = TOTAL / (span / (HCLK_MHZ * 1e6)) / 1e9 if span else float("nan")
feed_gbs_run = TOTAL / (run_cyc / (HCLK_MHZ * 1e6)) / 1e9 if run_cyc else float("nan")
host_gbs = TOTAL / dt / 1e9
flag = "  OVERFLOW(prefix-only)" if overflow else ""
print(f"CALLMS {dt*1e3:.4f} TOTALB {TOTAL} TILE_N {TILE_N} NTILES {N_TILES} DEPTH {DEPTH} "
      f"PKTS {n_pkts} SPANCYC {span} RUNCYC {run_cyc} STALLCYC {stall_cyc} IDLECYC {idle_cyc} "
      f"BUSY_FRAC {busy_frac:.3f} HOST_GBS {host_gbs:.4f} FEED_GBS_SPAN {feed_gbs_span:.4f} "
      f"FEED_GBS_RUN {feed_gbs_run:.4f}{flag}")
