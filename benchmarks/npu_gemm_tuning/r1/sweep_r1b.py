#!/usr/bin/env python3
# R1b differential-slope driver.
#
# On-hardware reality (measured 2026-07-05, halo aie2p, mlir-aie +886d932):
# a warm single-shot host-wall run is dominated by FIXED per-call overhead
# (device load + BO alloc + dispatch ~20 ms), NOT the feed. R1a's ~0.9 GB/s is a
# DIFFERENTIAL slope, which cancels that fixed cost. So we fit
#     call_ms(bytes) = fixed_ms + slope_ms_per_byte * bytes
# across an N_TILES sweep; SLOPE_GBS = 1e-6 / slope_ms_per_byte is the
# byte-proportional feed rate (== R1a's number), and FIXED_MS is the intercept
# (the per-call overhead a single-shot wrongly folds into the feed).
#
# Axes, all read on the SLOPE (not the confounded single-shot):
#   DEPTH   : busy-vs-idle proxy. Slope rises with FIFO depth => handshake/latency
#             bound (core idles on acquire; more in-flight BDs help, 16-BD budget).
#             Slope flat in depth => DMA continuously busy = bandwidth-bound
#             (lever: nd-descriptors / more columns, not depth).
#   MINIMAL : touch cost. Slope unchanged full-vs-minimal => feed-bound, not the
#             consume loop (R1a's core finding, re-checked on the slope).
#
# What the slope still cannot separate: genuine DMA feed vs byte-proportional host
# BO sync (both scale with bytes). That split needs the trace unit -- r1b_trace_run.py.
#
# One fresh subprocess per point (pyxrt segfaults on repeat under py3.14). Each
# distinct (TILE_N, N_TILES, DEPTH, MINIMAL) is a distinct xclbin: we warm it once
# (discarded) then take the min of REPEAT timed runs.
import os, sys, subprocess, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.environ.get("OUT") or os.path.join(
    HERE, "..", "results", f"r1b-slope-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.csv")
TILE_N = int(os.environ.get("TILE_N", 4096))
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "1 4 8").split()]
NT_SWEEP = [int(x) for x in os.environ.get("NT_SWEEP", "256 512 1024 2048").split()]
MINIMALS = [int(x) for x in os.environ.get("MINIMALS", "0 1").split()]
REPEAT = int(os.environ.get("REPEAT", 3))

LINE = re.compile(r"CALLMS ([0-9.]+) TOTALB ([0-9]+)")


def run_point(tile_n, n_tiles, depth, minimal):
    """Warm the xclbin once, then return (bytes, min call_ms) over REPEAT runs."""
    env = dict(os.environ, TILE_N=str(tile_n), N_TILES=str(n_tiles), DEPTH=str(depth))
    env["MINIMAL"] = "1" if minimal else ""
    best, total_b = None, None
    for i in range(REPEAT + 1):                       # i==0 is the discarded warmup
        p = subprocess.run([sys.executable, "r1b_run.py"], cwd=HERE, env=env,
                           capture_output=True, text=True, timeout=600)
        m = LINE.search(p.stdout)
        if not m:
            sys.stderr.write(f"  point tile={tile_n} nt={n_tiles} d={depth} m={minimal} FAILED\n{p.stderr[-400:]}\n")
            return None
        ms, total_b = float(m.group(1)), int(m.group(2))
        if i > 0:
            best = ms if best is None else min(best, ms)
    return total_b, best


def fit(points):
    """Least-squares ms = a + b*bytes; return (fixed_ms, slope_gbs, r2)."""
    n = len(points)
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx                                      # ms per byte
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    slope_gbs = (1e-6 / b) if b > 0 else float("nan")  # bytes/ms=1e-6/b GB/s
    return a, slope_gbs, r2


def main():
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    csv = open(CSV, "w")
    csv.write("utc,tile_n,minimal,depth,slope_gbs,fixed_ms,r2,points_bytes_ms\n")
    print(f"# R1b differential slope  TILE_N={TILE_N}  NT_SWEEP={NT_SWEEP}  REPEAT={REPEAT}")
    print(f"{'MIN':<4}{'DEPTH':<7}{'SLOPE_GBS':<11}{'FIXED_MS':<10}{'R2':<8}points(bytes:min_ms)")
    for minimal in MINIMALS:
        for depth in DEPTHS:
            pts = []
            for nt in NT_SWEEP:
                r = run_point(TILE_N, nt, depth, minimal)
                if r:
                    pts.append(r)
            if len(pts) < 2:
                print(f"{minimal:<4}{depth:<7}insufficient points")
                continue
            a, gbs, r2 = fit(pts)
            desc = " ".join(f"{b//1024}K:{ms:.2f}" for b, ms in pts)
            print(f"{minimal:<4}{depth:<7}{gbs:<11.4f}{a:<10.3f}{r2:<8.4f}{desc}")
            sys.stdout.flush()
            utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            csv.write(f"{utc},{TILE_N},{minimal},{depth},{gbs:.4f},{a:.3f},{r2:.4f},{desc.replace(',', '')}\n")
            csv.flush()
    csv.close()
    print(f"results -> {CSV}")


if __name__ == "__main__":
    main()
