#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kevin Read
# hipfire — see LICENSE and NOTICE in the project root.

# Opt-2 layer-split rebalance bench for the Stage 2b PpMtp path.
#
# Question: the PP boundary handoff in forward_prefill_batch_multi_with_caps
# is fully serialized (gfx906 runs its band to completion, blocking peer
# copy, then gfx1031 runs its band). The 48,16 split puts 16 layers on the
# slow/small gfx1031. Shifting more trunk layers onto gfx906 (which has
# both more headroom AND, per the rocprof memory, is the faster card for
# the dominant GEMM) should shrink gfx1031's serial tail and narrow the
# PpMtp-vs-pp2-ar decode gap. This bench measures decode tok/s + τ at
# three splits, with pp=1 MTP and pp=2 AR references for context.
#
# Zero code change — pure HIPFIRE_PP_LAYERS sweep. Honest comparison rule:
# byte-identical prompt across all cells (md5 recorded), one warmup forward
# per cell before the measured run, fresh daemon process per cell.
#
# Usage: ./scripts/bench-ppmtp-split.sh
#   HIPFIRE_PP_MODEL / HIPFIRE_PP_MTP_HEAD override paths.
#   HIPFIRE_BENCH_MAX overrides decode length (default 256).

set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/daemon"
MODEL="${HIPFIRE_PP_MODEL:-/local/hipfire/qwen3.6-27b.mq4}"
MTP_HEAD="${HIPFIRE_PP_MTP_HEAD:-/data/hipfire/qwen3.6-27b-cvs16384.mtp}"
MAX="${HIPFIRE_BENCH_MAX:-256}"
OUT="${HIPFIRE_BENCH_OUT:-/tmp/ppmtp-split-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# Fixed prompt — committed bytes, no heredoc reflow. Recorded md5 below.
PROMPT="Write a Python function that implements an LRU cache with a configurable capacity, using an OrderedDict. Include get and put methods, and a docstring."

# ── build if stale ──
if [ ! -x "$EXE" ] || [ crates/hipfire-runtime/examples/daemon.rs -nt "$EXE" ] \
   || [ crates/hipfire-arch-qwen35/src/mtp_spec.rs -nt "$EXE" ] \
   || [ crates/hipfire-arch-qwen35/src/qwen35.rs -nt "$EXE" ]; then
    echo "bench-ppmtp-split: building daemon..." >&2
    cargo build --release --example daemon --features deltanet >&2 || { echo "build failed" >&2; exit 2; }
fi

[ -f "$MODEL" ]    || { echo "model not found: $MODEL" >&2; exit 2; }
[ -f "$MTP_HEAD" ] || { echo "mtp head not found: $MTP_HEAD" >&2; exit 2; }

if [ -r "$LOCK_SCRIPT" ]; then
    . "$LOCK_SCRIPT"
    gpu_acquire "bench-ppmtp-split" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

prompt_md5=$(printf '%s' "$PROMPT" | md5sum | awk '{print $1}')
prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$PROMPT")

{
    echo "# PpMtp layer-split rebalance (Opt 2) — $(date -Iseconds)"
    echo
    echo "**Model:** \`$MODEL\`  **MTP head:** \`$MTP_HEAD\`"
    echo "**Prompt md5:** \`$prompt_md5\`  **max_tokens:** $MAX  **kv_mode:** asym3  **temp:** 0.0"
    echo
    echo "| cell | pp | split | spec_path | tok/s | decode tok/s | τ | accept | cycles | wall(s) |"
    echo "|---|---|---|---|---|---|---|---|---|---|"
} > "$OUT"

# Args: label  pp  use_mtp  split(or "-")
run_cell() {
    local label="$1" pp="$2" use_mtp="$3" split="$4"
    local mtp_param=""
    [ "$use_mtp" -eq 1 ] && mtp_param=",\"mtp_head\":\"$MTP_HEAD\""

    local in_file out_file
    in_file=$(mktemp); out_file=$(mktemp)
    # One warmup generate (short) then the measured generate. Fresh load.
    cat > "$in_file" <<JL
{"type":"load","model":"$MODEL","params":{"max_seq":4096,"pp":$pp,"kv_mode":"asym3"$mtp_param}}
{"type":"generate","id":"warm","prompt":${prompt_json},"temperature":0.0,"max_tokens":16,"repeat_penalty":1.0}
{"type":"generate","id":"meas","prompt":${prompt_json},"temperature":0.0,"max_tokens":$MAX,"repeat_penalty":1.0}
{"type":"unload"}
JL

    local split_env=""
    [ "$split" != "-" ] && split_env="HIPFIRE_PP_LAYERS=$split"

    echo "== $label (pp=$pp mtp=$use_mtp split=$split) ==" >&2
    local t0 t1 wall
    t0=$(date +%s.%N)
    env HIPFIRE_ALLOW_MIXED_ARCH=1 $split_env \
        timeout 600 "$EXE" < "$in_file" > "$out_file" 2>&1
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")

    # The measured done event is the SECOND done (id=meas).
    local done_line
    done_line=$(grep -aE '"type":"done","id":"meas"' "$out_file" | head -1)
    python3 - "$done_line" "$label" "$pp" "$split" "$wall" "$OUT" <<'PY'
import sys, json, re
done, label, pp, split, wall, out = sys.argv[1:7]
def g(k, d="-"):
    m = re.search(r'"%s":([0-9.]+)' % k, done)
    return m.group(1) if m else d
sp = re.search(r'"spec_path":"([^"]+)"', done)
sp = sp.group(1) if sp else "ar"
row = f"| {label} | {pp} | {split} | {sp} | {g('tok_s')} | {g('decode_tok_s')} | {g('tau')} | {g('accept_rate')} | {g('cycles')} | {wall} |"
with open(out, "a") as f:
    f.write(row + "\n")
print(row, file=sys.stderr)
PY
    rm -f "$in_file" "$out_file"
}

# References
run_cell "pp1-mtp"   1 1 -
run_cell "pp2-ar"    2 0 48,16
# PpMtp split sweep
run_cell "ppmtp-48-16" 2 1 48,16
run_cell "ppmtp-52-12" 2 1 52,12
run_cell "ppmtp-56-08" 2 1 56,8

echo >> "$OUT"
echo "Cheapest-step note: byte-identical prompt (md5 $prompt_md5), 1 warmup + 1 measured per cell, fresh process per cell." >> "$OUT"
echo
echo "bench report: $OUT"
cat "$OUT"
