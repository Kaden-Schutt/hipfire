#!/usr/bin/env python3
"""Line-level ddmin for the mlir-aie #3281 repro.
Oracle = interesting.sh (aie-opt objectFifo assert). Deletions that break the
parse make aie-opt fail early (rc<128) -> not interesting -> rejected, so ddmin
only keeps parse-valid subsets that still crash. ~0.2s/oracle.
"""
import subprocess, sys, os, tempfile, time

SRC = sys.argv[1]
ORACLE = os.path.join(os.path.dirname(os.path.abspath(SRC)) if False else ".",
                      "interesting.sh")
ORACLE = "./interesting.sh"
OUT = sys.argv[2] if len(sys.argv) > 2 else "reduced.mlir"

with open(SRC) as f:
    lines = f.readlines()

calls = 0
def interesting(cand):
    global calls
    calls += 1
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as t:
        t.writelines(cand); path = t.name
    try:
        return subprocess.run(["bash", ORACLE, path], check=False).returncode == 0
    finally:
        os.unlink(path)

assert interesting(lines), "seed is not interesting — oracle/seed mismatch"

# Zeller ddmin over lines.
n = 2
t0 = time.time()
while len(lines) >= 2:
    chunk = max(1, len(lines) // n)
    subsets = [lines[i:i+chunk] for i in range(0, len(lines), chunk)]
    reduced = False
    # try removing each subset's complement (keep one subset) then each subset
    for i, sub in enumerate(subsets):
        complement = lines[:i*chunk] + lines[i*chunk+len(sub):]
        if complement and interesting(complement):
            lines = complement; n = max(n-1, 2); reduced = True; break
    if not reduced:
        if n >= len(lines): break
        n = min(len(lines), n*2)
    print(f"  lines={len(lines)} calls={calls} n={n} t={time.time()-t0:.1f}s", flush=True)

with open(OUT, "w") as f:
    f.writelines(lines)
print(f"DONE: {len(lines)} lines, {calls} oracle calls, {time.time()-t0:.1f}s -> {OUT}")
