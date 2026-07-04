# R1b M3 (SCAFFOLD, UNSEALED): shim-DMA PORT_RUNNING active-cycle trace.
#
# STATUS: written against the documented IRON trace API (aie.utils.trace) but NOT
# yet run on hardware (no in-tree example wires trace_config on this toolchain, so
# it needs an on-hw iteration loop to seal -- the way R0b was sealed; see git log).
# The VALIDATED measurement is the differential slope in sweep_r1b.py.
#
# Why M3 is now the REQUIRED decisive step (not just a refinement): on this
# toolchain the two cheaper on-device routes are both closed --
#   - in-kernel core timer: aie::tile::current().cycles() does not link
#     (undefined ::get_cycles()); aie2p kernels timestamp via event0/event1 + trace.
#   - host-side 194 fencing: IRON's concrete run() bundles BO sync + execute, so
#     the host timer cannot fence the feed apart from the sync.
# And the slope result MADE this the open question: the ~12 GB/s byte-proportional
# rate is nearly DEPTH-INSENSITIVE (depth 1 vs 8 within noise). Since FIFO depth is
# an on-NPU DMA knob, its irrelevance means the ~12 GB/s is dominated by the HOST BO
# SYNC (host->device copy, which precedes the kernel), not the on-NPU feed. The true
# feed is >= 12 GB/s and hidden above the sync. Only tracing the shim MM2S channel's
# PORT_RUNNING cycles isolates the feed's own busy-cycle bandwidth from the sync.
#
# Event to trace on the feed column's shim tile (MM2S = L3->array feed channel):
#   ShimTilePortEvent(ShimTileEvent.PORT_RUNNING_0, WireBundle.DMA, channel=0, master=False)  # MM2S busy
#   ShimTilePortEvent(ShimTileEvent.PORT_STALLED_0, WireBundle.DMA, channel=0, master=False)  # MM2S stalled
#
# Wiring outline (the part that needs on-hw iteration under IRON @jit):
#   1. Obtain the placed shim tile handle for the feed column (SequentialPlacer
#      puts the single-column feed on col 0 -> shim (0,0)).
#   2. configure_trace([shim_tile], shimtile_events=[PORT_RUNNING_0, PORT_STALLED_0])
#      at program-build time (inside the @jit body, after resolve/placement).
#   3. start_trace(trace_size=..., ddr_id=-1) INSIDE rt.sequence so trace data is
#      appended after the last tensor arg (avoids a separate BO).
#   4. After the run, slice the trace bytes, parse_trace(...) -> JSON, then sum
#      PORT_RUNNING intervals; busy_frac = running / (running + stalled + idle).
#      DEV_GBS from M2 divided by busy_frac = the DMA's busy-cycle bandwidth.
#
# Deliberately not executed here to avoid shipping an unvalidated number. The
# scaffold documents the exact events + routing so the seal is a mechanical run.
import sys

print("R1b-M3 is an UNSEALED scaffold; run r1b_run.py (M1+M2) for the decisive "
      "measurement. See this file's header for the trace-wiring outline.",
      file=sys.stderr)
sys.exit(2)
