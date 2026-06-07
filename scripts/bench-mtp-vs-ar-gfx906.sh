#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kevin Read
# hipfire — see LICENSE and NOTICE in the project root.

# Single-gfx906 MTP-vs-AR uplift bench. Closes the gap in the dev log:
# every prior MTP measurement reported τ + absolute tok/s but NEVER sat
# next to a same-prompt same-binary single-GPU AR baseline, so the
# realized uplift was only ever inferred. This measures it directly.
#
# Both cells: pp=1, gfx906 only (HIP_VISIBLE_DEVICES=0), 27B, kv_mode q8,
# byte-identical LRU prompt (same md5 as note 14's split bench), greedy,
# 1 warmup + 1 measured generate, fresh load per cell.
#   - AR  cell: load WITHOUT mtp_head → pick_path falls to SpecPath::Ar.
#   - MTP cell: load WITH mtp_head + repeat_penalty 1.0 → SpecPath::Mtp.
#
# Uplift = MTP decode tok/s ÷ AR decode tok/s (the realized number; τ is
# the theoretical ceiling above it).

set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/daemon"
MODEL="${HIPFIRE_PP_MODEL:-/local/hipfire/qwen3.6-27b.mq4}"
MTP_HEAD="${HIPFIRE_PP_MTP_HEAD:-/data/hipfire/qwen3.6-27b-cvs16384.mtp}"
MAX="${HIPFIRE_BENCH_MAX:-256}"
OUT="${HIPFIRE_BENCH_OUT:-/tmp/mtp-vs-ar-gfx906-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# Byte-identical to note 14's split bench (same md5).
PROMPT="Write a Python function that implements an LRU cache with a configurable capacity, using an OrderedDict. Include get and put methods, and a docstring."

[ -x "$EXE" ] || { echo "daemon binary missing — build first" >&2; exit 2; }
[ -f "$MODEL" ]    || { echo "model not found: $MODEL" >&2; exit 2; }
[ -f "$MTP_HEAD" ] || { echo "mtp head not found: $MTP_HEAD" >&2; exit 2; }

if [ -r "$LOCK_SCRIPT" ]; then
    . "$LOCK_SCRIPT"
    gpu_acquire "bench-mtp-vs-ar" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

prompt_md5=$(printf '%s' "$PROMPT" | md5sum | awk '{print $1}')
prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$PROMPT")

{
    echo "# Single-gfx906 MTP-vs-AR uplift — $(date -Iseconds)"
    echo
    echo "**Model:** \`$MODEL\`  **MTP head:** \`$MTP_HEAD\`"
    echo "**Prompt md5:** \`$prompt_md5\`  **max_tokens:** $MAX  **kv_mode:** q8  **temp:** 0.0  **device:** gfx906 (HIP idx 0)"
    echo
    echo "| cell | spec_path | tok/s | decode tok/s | prefill tok/s | τ | cycles | wall(s) |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$OUT"

# Args: label  use_mtp(0/1)
run_cell() {
    local label="$1" use_mtp="$2"
    local mtp_param=""
    [ "$use_mtp" -eq 1 ] && mtp_param=",\"mtp_head\":\"$MTP_HEAD\""

    local in_file out_file
    in_file=$(mktemp); out_file=$(mktemp)
    cat > "$in_file" <<JL
{"type":"load","model":"$MODEL","params":{"max_seq":4096,"kv_mode":"q8"$mtp_param}}
{"type":"generate","id":"warm","prompt":${prompt_json},"temperature":0.0,"max_tokens":16,"repeat_penalty":1.0}
{"type":"generate","id":"meas","prompt":${prompt_json},"temperature":0.0,"max_tokens":$MAX,"repeat_penalty":1.0}
{"type":"unload"}
JL

    echo "== $label (mtp=$use_mtp) ==" >&2
    local t0 t1 wall
    t0=$(date +%s.%N)
    # Pin to gfx906 (HIP idx 0); no PP, no mixed-arch override needed.
    env HIP_VISIBLE_DEVICES=0 timeout 600 "$EXE" < "$in_file" > "$out_file" 2>&1
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")

    local done_line
    done_line=$(grep -aE '"type":"done","id":"meas"' "$out_file" | head -1)
    python3 - "$done_line" "$label" "$wall" "$OUT" <<'PY'
import sys, re
done, label, wall, out = sys.argv[1:5]
def g(k, d="-"):
    m = re.search(r'"%s":([0-9.]+)' % k, done)
    return m.group(1) if m else d
sp = re.search(r'"spec_path":"([^"]+)"', done)
sp = sp.group(1) if sp else "ar"
row = f"| {label} | {sp} | {g('tok_s')} | {g('decode_tok_s')} | {g('prefill_tok_s')} | {g('tau')} | {g('cycles')} | {wall} |"
with open(out, "a") as f:
    f.write(row + "\n")
print(row, file=sys.stderr)
# stash decode tok/s for the uplift calc
with open("/tmp/_mtpvsar_" + label, "w") as f:
    f.write(g('decode_tok_s', '0'))
PY
    rm -f "$in_file" "$out_file"
}

run_cell "ar"  0
run_cell "mtp" 1

# Uplift
python3 - "$OUT" <<'PY'
import sys
out = sys.argv[1]
def rd(n):
    try: return float(open("/tmp/_mtpvsar_"+n).read().strip())
    except: return 0.0
ar, mtp = rd("ar"), rd("mtp")
with open(out, "a") as f:
    f.write("\n")
    if ar > 0:
        f.write(f"**Realized decode uplift (MTP ÷ AR): {mtp/ar:.2f}×**  (AR {ar:.1f} → MTP {mtp:.1f} tok/s)\n")
    else:
        f.write("AR decode tok/s missing — could not compute uplift.\n")
print(f"\nRealized uplift: {mtp/ar:.2f}x  (AR {ar:.1f} -> MTP {mtp:.1f} tok/s)" if ar>0 else "no AR number", file=sys.stderr)
PY
rm -f /tmp/_mtpvsar_ar /tmp/_mtpvsar_mtp
echo
echo "report: $OUT"
cat "$OUT"
