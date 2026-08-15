#!/usr/bin/env bash
# Static (CPU-only) evidence for the MQ3-Lloyd MoE-down row tile.
#
# Compiles the incumbent, the row-tiled candidate, and the gate_up sibling for
# gfx1201 across all five radiowave scheduler profiles, then prints
# VGPR/SGPR/spill/scratch/occupancy per profile. NO GPU IS USED — `radiowave
# compile` and `radiowave inspect` are host-side hipcc + ELF metadata reads.
#
# Also builds and runs the host-side bit-parity simulator, which asserts the
# row-tiled kernels are BITWISE identical to the incumbent per (row, krank).
#
# usage: bash docs/investigations/2026-08-05-mq3-down-rowtile/sweep_static.sh
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
HERE="$REPO/docs/investigations/2026-08-05-mq3-down-rowtile"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

declare -A SRC=(
  [incumbent]="kernels/src/gemv_mq3g256_lloyd_moe_down_indexed.hip"
  [rowtile]="kernels/src/gemv_mq3g256_lloyd_moe_down_indexed_r4.hip"
  [gate_up]="kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed.hip"
)

for tag in incumbent rowtile gate_up; do
  for p in default max-ilp iterative-ilp memory-clause pipeline-ilp; do
    cargo run -q -p radiowave -- compile \
      --source "$REPO/${SRC[$tag]}" \
      --output "$OUT/${tag}_${p}.hsaco" \
      --arch gfx1201 --wave32 --scheduler-profile "$p" \
      --manifest "$OUT/${tag}_${p}.json" >/dev/null
  done
done

python3 - "$OUT" <<'PY'
import json, glob, os, sys
VG, GR, CAP = 1536, 8, 16          # gfx1201 wave32: VGPR/SIMD, granularity, wave cap
d = sys.argv[1]
hdr = ["kernel", "profile", "VGPR", "SGPR", "vspill", "sspill", "scratch",
       "occ w/SIMD", "instr", "waits"]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    m = json.load(open(f))
    for k in m["inspection"]["kernels"]:
        v = k["vgpr_count"]
        alloc = ((v + GR - 1) // GR) * GR
        rows.append([k["name"].replace("gemv_", "")[:52], m["scheduler_profile"], v,
                     k["sgpr_count"], k["vgpr_spill_count"], k["sgpr_spill_count"],
                     k["private_segment_fixed_size"], min(CAP, VG // alloc),
                     k["instructions"]["static_instructions"],
                     k["instructions"]["wait_instructions"]])
rows.sort(key=lambda r: (r[0], r[1]))
w = [max(len(str(r[c])) for r in [hdr] + rows) for c in range(len(hdr))]
p = lambda r: "  ".join(str(x).ljust(w[c]) for c, x in enumerate(r))
print(p(hdr)); print("  ".join("-" * x for x in w))
for r in rows: print(p(r))
PY

echo
g++ -O2 -std=c++17 -include array "$HERE/mq3_down_rowtile_parity.cpp" -o "$OUT/parity"
"$OUT/parity"
