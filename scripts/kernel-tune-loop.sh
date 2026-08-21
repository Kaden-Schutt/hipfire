#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors
#
# kernel-tune-loop.sh — Automated kernel optimization loop for hipfire.
#
# Drives the profile → diagnose → implement → validate → measure → decide
# cycle from the hipfire-kernel-tuning playbook. Automates everything
# around the kernel edit: profiling, ISA capture, correctness checking,
# fresh-process measurement, and ledger logging.
#
# Usage:
#   ./scripts/kernel-tune-loop.sh baseline          # Phase 0: establish ground floor
#   ./scripts/kernel-tune-loop.sh profile           # Phase 1: profile + diagnose data
#   ./scripts/kernel-tune-loop.sh validate          # Phase 4: correctness check
#   ./scripts/kernel-tune-loop.sh measure           # Phase 5: fresh-process A/B
#   ./scripts/kernel-tune-loop.sh decide <disposition> <notes>  # Phase 6: log outcome
#   ./scripts/kernel-tune-loop.sh status            # Show current loop state
#
# Configuration (env vars):
#   HIPFIRE_MODEL       Model path (default: ~/.hipfire/models/qwen3.5-4b.mq4)
#   HIPFIRE_KV_MODE     KV cache mode (default: q8)
#   HIPFIRE_ARCH        Target arch (default: auto-detect)
#   HIPFIRE_BENCH_RUNS  Bench runs (default: 5)
#   HIPFIRE_BENCH_WARMUPS  Bench warmups (default: 3)
#   HIPFIRE_BENCH_MAX_TOKENS  Max tokens per run (default: 128)
#   HIPFIRE_BENCH_BACKEND   Bench backend (default: noslots)
#   HIPFIRE_BENCH_WORKLOAD  Bench workload (default: stateless)
#   HIPFIRE_PROMPT_FILE  Prompt file for bench (default: benchmarks/prompts/bare_factual.txt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────

MODEL="${HIPFIRE_MODEL:-$HOME/.hipfire/models/qwen3.5-4b.mq4}"
KV_MODE="${HIPFIRE_KV_MODE:-q8}"
BENCH_RUNS="${HIPFIRE_BENCH_RUNS:-5}"
BENCH_WARMUPS="${HIPFIRE_BENCH_WARMUPS:-3}"
BENCH_MAX_TOKENS="${HIPFIRE_BENCH_MAX_TOKENS:-128}"
BENCH_BACKEND="${HIPFIRE_BENCH_BACKEND:-noslots}"
BENCH_WORKLOAD="${HIPFIRE_BENCH_WORKLOAD:-stateless}"
PROMPT_FILE="${HIPFIRE_PROMPT_FILE:-benchmarks/prompts/bare_factual.txt}"

OUT_BASE=".codeinsight+research/kernel-tune"
RUNS_DIR="$OUT_BASE/runs"
LEDGER_FILE="$OUT_BASE/ledger/ledger.jsonl"
STATE_FILE="$OUT_BASE/loop_state.json"

mkdir -p "$RUNS_DIR" "$OUT_BASE/ledger"

# ── Helpers ───────────────────────────────────────────────────────────────

timestamp() { date -u +%Y%m%dT%H%M%SZ; }
bench_date() { date -u +%Y-%m-%d; }

md5() { md5sum "$1" 2>/dev/null | awk '{print $1}' || echo "unknown"; }

detect_arch() {
    # Filter out CPU nodes (gfx_target_version 0) and take the first GPU
    local arch
    arch=$(grep 'gfx_target_version' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | grep -v ':0$' | head -1 | grep -oP 'gfx_target_version\s+\K[0-9]+')
    case "$arch" in
        110000) echo "gfx1100" ;;
        120000) echo "gfx1200" ;;
        120001) echo "gfx1201" ;;
        103000) echo "gfx1030" ;;
        103100) echo "gfx1031" ;;
        115000) echo "gfx1150" ;;
        115100) echo "gfx1151" ;;
        94000)  echo "gfx940"  ;;
        94200)  echo "gfx942"  ;;
        90600)  echo "gfx906"  ;;
        101000) echo "gfx1010" ;;
        101300) echo "gfx1013" ;;
        *) echo "unknown" ;;
    esac
}

ARCH="${HIPFIRE_ARCH:-$(detect_arch)}"

get_git_commit() { git rev-parse HEAD 2>/dev/null || echo "unknown"; }
get_git_dirty() { git status --porcelain 2>/dev/null | head -1; }
get_git_diff_md5() { git diff 2>/dev/null | md5sum | awk '{print $1}'; }

BINARY="./target/release/hipfire"
DAEMON="./target/release/daemon"
BENCH_EXAMPLE="./target/release/examples/bench_qwen35_mq4"
TEST_KERNELS="./target/release/examples/test_kernels"
record_identity() {
    local out_file="$1"
    cat > "$out_file" <<EOF
{
  "timestamp": "$(timestamp)",
  "bench_date": "$(bench_date)",
  "arch": "$ARCH",
  "model": "$MODEL",
  "model_md5": "$(md5 "$MODEL")",
  "binary": "$BINARY",
  "binary_md5": "$(md5 "$BINARY")",
  "daemon_md5": "$(md5 "$DAEMON")",
  "git_commit": "$(get_git_commit)",
  "git_dirty": "$(get_git_dirty)",
  "git_diff_md5": "$(get_git_diff_md5)",
  "kv_mode": "$KV_MODE",
  "bench_runs": $BENCH_RUNS,
  "bench_warmups": $BENCH_WARMUPS,
  "bench_max_tokens": $BENCH_MAX_TOKENS,
  "bench_backend": "$BENCH_BACKEND",
  "bench_workload": "$BENCH_WORKLOAD",
  "prompt_file": "$PROMPT_FILE",
  "prompt_md5": "$(md5 "$PROMPT_FILE")"
}
EOF
}

run_bench() {
    local extra_env="$1"
    local output_file="$2"
    # stderr goes to the log file; stdout (JSON) goes to both log and output
    eval "$extra_env HIPFIRE_DPM_WARMUP_SECS=10 \
        $BINARY bench $MODEL \
        --kv-mode $KV_MODE \
        --runs $BENCH_RUNS \
        --warmups $BENCH_WARMUPS \
        --max-tokens $BENCH_MAX_TOKENS \
        --backend $BENCH_BACKEND \
        --workload $BENCH_WORKLOAD \
        --json" > "$output_file" 2>"${output_file}.stderr"
}

extract_bench_metric() {
    local json_file="$1"
    local metric="$2"
    python3 -c "
import json, sys
try:
    with open('$json_file') as f:
        content = f.read()
    data = json.loads(content)
    stats = data.get('$metric', {})
    print(f\"median={stats.get('median',0):.1f} mean={stats.get('mean',0):.1f} min={stats.get('min',0):.1f} max={stats.get('max',0):.1f} stdev={stats.get('stdev',0):.3f}\")
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1
}

# ── Phase 0: Baseline ─────────────────────────────────────────────────────

phase_baseline() {
    local run_dir="$RUNS_DIR/baseline-$(timestamp)"
    mkdir -p "$run_dir"

    echo "=== Phase 0: Baseline Establishment ==="
    echo "  Model:   $MODEL"
    echo "  Arch:    $ARCH"
    echo "  KV mode: $KV_MODE"
    echo "  Output:  $run_dir"
    echo ""

    # Record identity
    record_identity "$run_dir/identity.json"
    echo "  Identity recorded."

    # Run baseline bench
    echo "  Running baseline bench ($BENCH_RUNS runs, $BENCH_WARMUPS warmups)..."
    run_bench "" "$run_dir/bench_baseline.json"
    local decode prefill
    decode=$(extract_bench_metric "$run_dir/bench_baseline.json" "decode_tok_s")
    prefill=$(extract_bench_metric "$run_dir/bench_baseline.json" "prefill_tok_s")
    echo "  Decode:  $decode"
    echo "  Prefill: $prefill"

    # Run profiled bench using bench_qwen35_mq4 (has per-kernel timers)
    echo "  Running profiled bench (bench_qwen35_mq4 + HIPFIRE_PROFILE=1)..."
    HIPFIRE_KV_MODE="$KV_MODE" HIPFIRE_PROFILE=1 HIPFIRE_PROFILE_DECODE=1 \
        HIPFIRE_DPM_WARMUP_SECS=10 \
        $BENCH_EXAMPLE "$MODEL" --prefill 32 --gen "$BENCH_MAX_TOKENS" --warmup 5 \
        > "$run_dir/bench_profiled.log" 2>&1 || echo "  (profiled bench failed — non-fatal)"
    echo "  Profile saved."

    # Extract and display hot kernels
    echo ""
    echo "  Hot prefill kernels:"
    python3 -c "
import re
try:
    with open('$run_dir/bench_profiled.log') as f:
        content = f.read()
    in_profile = False
    timings = []
    for line in content.split('\n'):
        if '=== PROFILE' in line:
            in_profile = True
            continue
        if in_profile and line.strip().startswith('TOTAL'):
            in_profile = False
            continue
        if in_profile:
            m = re.match(r'\s+(\S+)\s+(\d+)x\s+([\d.]+)ms\s+\(([\d.]+)µs/call\)\s+([\d.]+)%\s+([\d.]+)\s*GiB/s', line)
            if m:
                timings.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))))
    if timings:
        timings.sort(key=lambda x: x[2], reverse=True)
        for name, calls, total, percall, pct, bw in timings[:10]:
            print(f'    {name:<45} {calls:>4}x {total:>7.1f}ms {percall:>7.0f}µs {pct:>5.1f}% {bw:>8.1f}GB/s')
except Exception as e:
    print(f'    ERROR: {e}')
" 2>&1

    # Show summary
    grep -E "^SUMMARY|^PREFILL_SUMMARY" "$run_dir/bench_profiled.log" 2>/dev/null | sed 's/^/    /' || true

    # Run kernel profile inventory
    echo "  Running kernel profile..."
    $BINARY profile "$MODEL" --json > "$run_dir/kernel_profile.json" 2>/dev/null
    echo "  Kernel profile saved."

    # Run test_kernels for correctness baseline
    echo "  Running test_kernels..."
    if [ -x "$TEST_KERNELS" ]; then
        $TEST_KERNELS > "$run_dir/test_kernels.log" 2>&1 || true
        echo "  test_kernels saved."
    else
        echo "  WARNING: test_kernels not built. Run: cargo build --release --features deltanet --example test_kernels -p hipfire-runtime"
    fi

    # Save baseline state
    python3 -c "
import json
identity = json.load(open('$run_dir/identity.json'))
state = {
    'baseline_dir': '$run_dir',
    'baseline_decode': '$decode'.split('median=')[1].split(' ')[0] if 'median=' in '$decode' else '0',
    'baseline_prefill': '$prefill'.split('median=')[1].split(' ')[0] if 'median=' in '$prefill' else '0',
    'identity': identity,
    'phase': 'baseline_complete',
    'timestamp': '$(timestamp)',
}
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
print('  State saved to $STATE_FILE')
"

    # Write ledger entry
    cat >> "$LEDGER_FILE" <<EOF
{"phase":"baseline","timestamp":"$(timestamp)","bench_date":"$(bench_date)","arch":"$ARCH","model_md5":"$(md5 "$MODEL")","binary_md5":"$(md5 "$BINARY")","git_commit":"$(get_git_commit)","decode":"$decode","prefill":"$prefill","run_dir":"$run_dir"}
EOF

    echo ""
    echo "=== Baseline complete ==="
    echo "  Decode:  $decode"
    echo "  Prefill: $prefill"
    echo "  Run dir: $run_dir"
}

# ── Phase 1: Profile + Diagnose ───────────────────────────────────────────

phase_profile() {
    local iter
    iter=$(python3 -c "import json; s=json.load(open('$STATE_FILE')); print(s.get('iteration',0))" 2>/dev/null || echo "0")
    local run_dir="$RUNS_DIR/iter-${iter}-$(timestamp)"
    mkdir -p "$run_dir"

    echo "=== Phase 1: Profile + Diagnose (iteration $iter) ==="
    echo "  Output: $run_dir"
    echo ""

    # Record identity
    record_identity "$run_dir/identity.json"

    # Run profiled bench using bench_qwen35_mq4 (has per-kernel timers)
    echo "  Running profiled bench (bench_qwen35_mq4 + HIPFIRE_PROFILE=1)..."
    local bench_cmd="$BENCH_EXAMPLE $MODEL --prefill 32 --gen $BENCH_MAX_TOKENS --warmup 5"
    HIPFIRE_KV_MODE="$KV_MODE" HIPFIRE_PROFILE=1 HIPFIRE_PROFILE_DECODE=1 \
        HIPFIRE_DPM_WARMUP_SECS=10 \
        $bench_cmd > "$run_dir/bench_profiled.log" 2>&1 || echo "  (bench failed — check log)"

    # Also run with --emit-atlas for Atlas integration
    HIPFIRE_KV_MODE="$KV_MODE" HIPFIRE_PROFILE=1 \
        HIPFIRE_DPM_WARMUP_SECS=10 \
        $bench_cmd --emit-atlas "$run_dir/atlas_raw.jsonl" > "$run_dir/bench_atlas.log" 2>&1 || true

    # Extract hot kernels from profile output
    echo ""
    echo "  Hot prefill kernels (from profile):"
    python3 -c "
import re
try:
    with open('$run_dir/bench_profiled.log') as f:
        content = f.read()
    # Parse PROFILE section: kernel_name Nx  Xms  (Yµs/call)  Z%  W GiB/s
    in_profile = False
    timings = []
    for line in content.split('\n'):
        if '=== PROFILE' in line:
            in_profile = True
            continue
        if in_profile and line.strip().startswith('TOTAL'):
            in_profile = False
            continue
        if in_profile:
            m = re.match(r'\s+(\S+)\s+(\d+)x\s+([\d.]+)ms\s+\(([\d.]+)µs/call\)\s+([\d.]+)%\s+([\d.]+)\s*GiB/s', line)
            if m:
                timings.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))))
    if timings:
        timings.sort(key=lambda x: x[2], reverse=True)
        print(f'    {\"kernel\":<45} {\"calls\":>5} {\"total\":>8} {\"per-call\":>10} {\"%\":>6} {\"BW\":>10}')
        print('    ' + '-'*90)
        for name, calls, total, percall, pct, bw in timings[:15]:
            print(f'    {name:<45} {calls:>5} {total:>7.1f}ms {percall:>8.0f}µs {pct:>5.1f}% {bw:>8.1f}GB/s')
    else:
        print('    (No kernel timing lines found — check $run_dir/bench_profiled.log)')
except Exception as e:
    print(f'    ERROR: {e}')
" 2>&1

    # Extract decode profile
    echo ""
    echo "  Decode profile:"
    python3 -c "
import re
try:
    with open('$run_dir/bench_profiled.log') as f:
        content = f.read()
    # Find DECODE PROFILE section
    in_decode = False
    timings = []
    for line in content.split('\n'):
        if '=== DECODE PROFILE' in line:
            in_decode = True
            continue
        if in_decode and line.strip().startswith('TOTAL'):
            in_decode = False
            continue
        if in_decode:
            m = re.match(r'\s+(\S+)\s+(\d+)x\s+([\d.]+)ms\s+\(([\d.]+)µs/call\)\s+([\d.]+)%\s+([\d.]+)\s*GiB/s', line)
            if m:
                timings.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))))
    if timings:
        timings.sort(key=lambda x: x[2], reverse=True)
        print(f'    {\"kernel\":<45} {\"calls\":>5} {\"total\":>8} {\"per-call\":>10} {\"%\":>6} {\"BW\":>10}')
        print('    ' + '-'*90)
        for name, calls, total, percall, pct, bw in timings[:15]:
            print(f'    {name:<45} {calls:>5} {total:>7.1f}ms {percall:>8.0f}µs {pct:>5.1f}% {bw:>8.1f}GB/s')
    else:
        # Check for SUMMARY line
        m = re.search(r'SUMMARY\s+gen_tok_s=([\d.]+)', content)
        if m:
            print(f'    gen_tok_s={m.group(1)} (no per-kernel decode timers — decode kernels lack begin_timer)')
        else:
            print('    (No decode profile data found)')
        print('    NOTE: decode kernels may not have internal timers. Use rocprof for attribution.')
except Exception as e:
    print(f'    ERROR: {e}')
" 2>&1

    # Extract SUMMARY line
    echo ""
    echo "  Summary:"
    grep -E "^SUMMARY|^PREFILL_SUMMARY" "$run_dir/bench_profiled.log" 2>/dev/null | sed 's/^/    /' || echo "    (no summary found)"

    # Run kernel profile inventory
    echo ""
    echo "  Running kernel profile inventory..."
    $BINARY profile "$MODEL" --json > "$run_dir/kernel_profile.json" 2>/dev/null

    # Show low-occupancy / high-VGPR kernels
    echo ""
    echo "  Low-occupancy / high-VGPR kernels:"
    python3 -c "
import json
data = json.load(open('$run_dir/kernel_profile.json'))
kernels = data.get('kernels', [])
kernels_sorted = sorted(kernels, key=lambda k: k.get('occupancy',{}).get('pct',100))
for k in kernels_sorted:
    occ = k.get('occupancy',{}).get('pct',0)
    vgpr = k.get('vgprs',0)
    if occ < 100 or vgpr > 40:
        name = k.get('name','?')
        sgpr = k.get('sgprs',0)
        lds = k.get('lds_bytes',0)
        lim = k.get('occupancy',{}).get('limiter','?')
        print(f'    {name:<50} VGPR={vgpr:>3} SGPR={sgpr:>3} LDS={lds:>6} occ={occ:>5.1f}% lim={lim}')
" 2>&1

    # Try Atlas collect-ar with --profile-prefill --profile-decode
    echo ""
    echo "  Running Atlas collect-ar..."
    local atlas_out="$run_dir/atlas.jsonl"
    local isa_out="$run_dir/isa.json"
    python3 scripts/kernel_atlas.py collect-ar \
        --model "$MODEL" \
        --workload "qwen3.5-4b-tune-iter${iter}" \
        --model-size 4b \
        --quant mq4 \
        --prefill 32 \
        --gen "$BENCH_MAX_TOKENS" \
        --kv-mode "$KV_MODE" \
        --profile-prefill \
        --profile-decode \
        --isa-dir .hipfire_kernels \
        --isa-output "$isa_out" \
        --dispatch-provenance \
        --dispatch-output "$run_dir/dispatch.json" \
        --output "$atlas_out" 2>&1 || echo "  (Atlas collect failed — non-fatal)"

    # Try Atlas suggest
    if [ -f "$atlas_out" ]; then
        echo ""
        echo "  Atlas suggestions:"
        python3 scripts/kernel_atlas.py suggest \
            --row "$atlas_out" \
            --row-index 0 \
            --isa "$isa_out" \
            --format markdown 2>&1 | head -30 || echo "  (suggest failed — non-fatal)"
    fi

    # Update state
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['current_iter_dir'] = '$run_dir'
state['iteration'] = $iter
state['phase'] = 'profile_complete'
state['timestamp'] = '$(timestamp)'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
"

    echo ""
    echo "=== Profile complete ==="
    echo "  Data in: $run_dir"
    echo "  Next: Edit the target kernel, then run: $0 validate"
}

# ── Phase 4: Validate Correctness ─────────────────────────────────────────

phase_validate() {
    local run_dir
    run_dir=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('current_iter_dir',''))" 2>/dev/null || echo "")
    if [ -z "$run_dir" ]; then
        echo "ERROR: No current iteration. Run '$0 profile' first."
        exit 1
    fi

    echo "=== Phase 4: Correctness Validation ==="
    echo "  Iteration dir: $run_dir"
    echo ""

    # Run test_kernels
    echo "  Running test_kernels..."
    if [ -x "$TEST_KERNELS" ]; then
        $TEST_KERNELS 2>&1 | tee "$run_dir/test_kernels_candidate.log"
        local tk_exit=$?
        if [ $tk_exit -ne 0 ]; then
            echo ""
            echo "  ❌ test_kernels FAILED (exit $tk_exit)"
            echo "  Correctness gate: FAIL"
            python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['phase'] = 'correctness_fail'
state['correctness'] = 'fail'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
"
            exit 1
        fi
        echo "  ✅ test_kernels PASSED"
    else
        echo "  WARNING: test_kernels not built. Skipping."
        echo "  Build with: cargo build --release --features deltanet --example test_kernels -p hipfire-runtime"
    fi

    # Run serve harness for model-level check
    echo ""
    echo "  Running serve harness (battery)..."
    if [ -f "scripts/serve_harness.py" ]; then
        python3 scripts/serve_harness.py battery \
            --model "$MODEL" \
            --kv-mode "$KV_MODE" \
            --max-tokens 64 \
            2>&1 | tee "$run_dir/serve_harness.log" || echo "  (serve harness failed — check log)"
    else
        echo "  (serve_harness.py not found — skipping)"
    fi

    # Eyeball check — run a simple generation
    echo ""
    echo "  Eyeball check — generating sample output..."
    HIPFIRE_DPM_WARMUP_SECS=5 $BINARY bench "$MODEL" \
        --kv-mode "$KV_MODE" \
        --runs 1 --warmups 1 --max-tokens 64 \
        --backend noslots --workload stateless 2>&1 | tee "$run_dir/eyeball.log" | tail -5

    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['phase'] = 'correctness_pass'
state['correctness'] = 'pass'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
"

    echo ""
    echo "=== Correctness validation complete ==="
    echo "  Next: Run '$0 measure' for fresh-process A/B"
}

# ── Phase 5: Fresh-Process Measurement ────────────────────────────────────

phase_measure() {
    local run_dir
    run_dir=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('current_iter_dir',''))" 2>/dev/null || echo "")
    if [ -z "$run_dir" ]; then
        echo "ERROR: No current iteration. Run '$0 profile' first."
        exit 1
    fi

    local baseline_dir
    baseline_dir=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('baseline_dir',''))" 2>/dev/null || echo "")

    echo "=== Phase 5: Fresh-Process Measurement ==="
    echo "  Iteration dir: $run_dir"
    echo ""

    # Record candidate identity
    record_identity "$run_dir/identity_candidate.json"
    local candidate_bin_md5
    candidate_bin_md5=$(md5 "$BINARY")

    echo "  Candidate binary md5: $candidate_bin_md5"
    echo ""

    # Run candidate bench (fresh process)
    echo "  Running candidate bench ($BENCH_RUNS runs, $BENCH_WARMUPS warmups)..."
    run_bench "" "$run_dir/bench_candidate.json"
    local decode_c prefill_c
    decode_c=$(extract_bench_metric "$run_dir/bench_candidate.json" "decode_tok_s")
    prefill_c=$(extract_bench_metric "$run_dir/bench_candidate.json" "prefill_tok_s")
    echo "  Candidate decode:  $decode_c"
    echo "  Candidate prefill: $prefill_c"

    # Compare with baseline
    echo ""
    if [ -n "$baseline_dir" ] && [ -f "$baseline_dir/bench_baseline.json" ]; then
        local decode_b prefill_b
        decode_b=$(extract_bench_metric "$baseline_dir/bench_baseline.json" "decode_tok_s")
        prefill_b=$(extract_bench_metric "$baseline_dir/bench_baseline.json" "prefill_tok_s")
        echo "  Baseline decode:  $decode_b"
        echo "  Baseline prefill: $prefill_b"
        echo ""

        # Compute deltas
        python3 -c "
import re
def parse(s):
    m = re.search(r'median=([\d.]+)', s)
    return float(m.group(1)) if m else 0.0

d_b = parse('''$decode_b''')
d_c = parse('''$decode_c''')
p_b = parse('''$prefill_b''')
p_c = parse('''$prefill_c''')

d_delta = ((d_c - d_b) / d_b * 100) if d_b > 0 else 0
p_delta = ((p_c - p_b) / p_b * 100) if p_b > 0 else 0

print(f'  Decode delta:  {d_delta:+.1f}%  ({d_b:.1f} → {d_c:.1f} tok/s)')
print(f'  Prefill delta: {p_delta:+.1f}%  ({p_b:.1f} → {p_c:.1f} tok/s)')
print()

if d_delta > 2 or p_delta > 2:
    print('  ✅ Potential win — proceed to decide')
elif d_delta < -5 or p_delta < -5:
    print('  ❌ Regression — reject')
else:
    print('  ⚠️  Within noise band — consider ABBA or more samples')
" 2>&1
    else
        echo "  (No baseline found — run '$0 baseline' first)"
    fi

    # Save measurement to state
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['phase'] = 'measure_complete'
state['candidate_decode'] = '$decode_c'.split('median=')[1].split(' ')[0] if 'median=' in '$decode_c' else '0'
state['candidate_prefill'] = '$prefill_c'.split('median=')[1].split(' ')[0] if 'median=' in '$prefill_c' else '0'
state['candidate_bin_md5'] = '$candidate_bin_md5'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
"

    echo ""
    echo "=== Measurement complete ==="
    echo "  Next: Run '$0 decide <disposition> <notes>'"
    echo "  Disposition: win | reject | park | regression"
}

# ── Phase 6: Decide & Log ─────────────────────────────────────────────────

phase_decide() {
    local disposition="${1:-unknown}"
    local notes="${2:-}"
    local run_dir
    run_dir=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('current_iter_dir',''))" 2>/dev/null || echo "")

    echo "=== Phase 6: Decide & Log ==="
    echo "  Disposition: $disposition"
    echo "  Notes: $notes"
    echo ""

    # Write ledger entry
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
entry = {
    'phase': 'decide',
    'iteration': state.get('iteration', 0),
    'disposition': '$disposition',
    'notes': '''$notes''',
    'timestamp': '$(timestamp)',
    'bench_date': '$(bench_date)',
    'arch': '$ARCH',
    'model_md5': state.get('identity', {}).get('model_md5', 'unknown'),
    'baseline_bin_md5': state.get('identity', {}).get('binary_md5', 'unknown'),
    'candidate_bin_md5': state.get('candidate_bin_md5', 'unknown'),
    'git_commit': state.get('identity', {}).get('git_commit', 'unknown'),
    'baseline_decode': state.get('baseline_decode', '0'),
    'baseline_prefill': state.get('baseline_prefill', '0'),
    'candidate_decode': state.get('candidate_decode', '0'),
    'candidate_prefill': state.get('candidate_prefill', '0'),
    'run_dir': '$run_dir',
}
with open('$LEDGER_FILE', 'a') as f:
    f.write(json.dumps(entry) + '\n')
print('  Ledger entry written to $LEDGER_FILE')
" 2>&1

    # Increment iteration counter
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['iteration'] = state.get('iteration', 0) + 1
state['phase'] = 'ready_for_next'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
print('  Iteration counter incremented to', state['iteration'])
" 2>&1

    echo ""
    echo "=== Decision logged ==="
    echo "  Next: Run '$0 profile' for the next iteration"
}

# ── Status ────────────────────────────────────────────────────────────────

phase_status() {
    echo "=== Kernel Tune Loop Status ==="
    echo ""

    if [ -f "$STATE_FILE" ]; then
        python3 -c "
import json
state = json.load(open('$STATE_FILE'))
print(f'  Phase:      {state.get(\"phase\", \"unknown\")}')
print(f'  Iteration:  {state.get(\"iteration\", 0)}')
print(f'  Arch:       {state.get(\"identity\", {}).get(\"arch\", \"unknown\")}')
print(f'  Model:      {state.get(\"identity\", {}).get(\"model\", \"unknown\")}')
print(f'  KV mode:    {state.get(\"identity\", {}).get(\"kv_mode\", \"unknown\")}')
print(f'  Baseline:   decode={state.get(\"baseline_decode\", \"?\")} prefill={state.get(\"baseline_prefill\", \"?\")}')
if 'candidate_decode' in state:
    print(f'  Candidate:  decode={state.get(\"candidate_decode\", \"?\")} prefill={state.get(\"candidate_prefill\", \"?\")}')
print(f'  Baseline dir: {state.get(\"baseline_dir\", \"none\")}')
print(f'  Current iter: {state.get(\"current_iter_dir\", \"none\")}')
" 2>&1
    else
        echo "  No state file found. Run '$0 baseline' to start."
    fi

    echo ""
    if [ -f "$LEDGER_FILE" ]; then
        local n
        n=$(wc -l < "$LEDGER_FILE")
        echo "  Ledger entries: $n"
        echo ""
        echo "  Recent entries:"
        tail -5 "$LEDGER_FILE" | python3 -c "
import json, sys
for line in sys.stdin:
    try:
        e = json.loads(line.strip())
        disp = e.get('disposition', e.get('phase', '?'))
        ts = e.get('timestamp', '?')
        it = e.get('iteration', '-')
        print(f'    [{ts}] iter={it} {disp}')
    except: pass
" 2>&1
    else
        echo "  No ledger entries yet."
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────

case "${1:-help}" in
    baseline)
        phase_baseline
        ;;
    profile)
        phase_profile
        ;;
    validate)
        phase_validate
        ;;
    measure)
        phase_measure
        ;;
    decide)
        if [ $# -lt 2 ]; then
            echo "Usage: $0 decide <disposition> [notes]"
            echo "  disposition: win | reject | park | regression"
            exit 1
        fi
        phase_decide "$2" "${3:-}"
        ;;
    status)
        phase_status
        ;;
    help|--help|-h)
        echo "kernel-tune-loop.sh — Automated kernel optimization loop"
        echo ""
        echo "Usage:"
        echo "  $0 baseline          Phase 0: establish ground floor"
        echo "  $0 profile           Phase 1: profile + diagnose data"
        echo "  $0 validate          Phase 4: correctness check"
        echo "  $0 measure           Phase 5: fresh-process A/B"
        echo "  $0 decide <disp> [notes]  Phase 6: log outcome"
        echo "  $0 status            Show current loop state"
        echo ""
        echo "Configuration (env vars):"
        echo "  HIPFIRE_MODEL       Model path (default: ~/.hipfire/models/qwen3.5-4b.mq4)"
        echo "  HIPFIRE_KV_MODE     KV cache mode (default: q8)"
        echo "  HIPFIRE_ARCH        Target arch (default: auto-detect)"
        echo "  HIPFIRE_BENCH_RUNS  Bench runs (default: 5)"
        echo "  HIPFIRE_BENCH_WARMUPS  Bench warmups (default: 3)"
        echo "  HIPFIRE_BENCH_MAX_TOKENS  Max tokens (default: 128)"
        echo "  HIPFIRE_BENCH_BACKEND   Bench backend (default: noslots)"
        echo "  HIPFIRE_BENCH_WORKLOAD  Bench workload (default: stateless)"
        echo "  HIPFIRE_PROMPT_FILE  Prompt file (default: benchmarks/prompts/bare_factual.txt)"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage."
        exit 1
        ;;
esac
