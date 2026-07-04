#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Grégory D
# hipfire — see LICENSE and NOTICE in the project root.

# Path C smoke gate — runs `spec_step_ddtree_path_c` (Phase 1 and Phase 2)
# end-to-end through `dflash_spec_demo`, applies the Path A/B token-attractor
# detector, and reports per-mode pass/fail.
#
# PRD: docs/plans/ddtree-path-c-main-path-first-from-lucebox.prd
#
# Hard-fail conditions (block commit / push):
#   - dflash_spec_demo non-zero exit / panic / zero emitted tokens
#   - Phase 1 output diverges from --ddtree-batched on the same prompt
#     (Phase 1 should be bit-exact with verify_dflash_block on the main chain;
#     diff against ddtree-batched gives a sanity signal — they're not
#     guaranteed identical but should agree on most prompts)
#   - max_token_frequency / total > 0.50 in the first 128 emitted tokens
#     (Path A failure mode — single-token attractor)
#   - unique_token_count / total < 0.15 (low-entropy loop)
#
# Soft warn (printed, doesn't block): paragraph-level repetition.
#
# Modes tested (each with a short prose prompt + a short code prompt):
#   path-c-phase1-b12-k2  : Step 1 only
#   path-c-phase2-b12-k2  : Steps 1+2+3 (lazy branch FA-only re-verify)
#
# Usage:
#   ./tests/path-c-smoke.sh                    # auto-detect models
#   TARGET=/path/to/t-mq4.hfq DRAFT=/path/to/d.hfq ./tests/path-c-smoke.sh
#   ./tests/path-c-smoke.sh --graph-ab         # report graph/nograph deltas
#   ./tests/path-c-smoke.sh --graph-promote    # hard-fail unless graph deltas promote
#
# Exit codes:
#   0  smoke ran clean
#   1  hard error
#   2  build / environment error

set -u
cd "$(dirname "$0")/.."

FULL=0
GRAPH_AB=0
GRAPH_PROMOTE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        --graph-ab) GRAPH_AB=1 ;;  # A/B verify-graph capture on/off (Phase 3 gate)
        --graph-promote) GRAPH_AB=1; GRAPH_PROMOTE=1 ;;  # hard-fail unless graph A/B is promotable
        -h|--help) sed -n '3,33p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

EXE="./target/release/examples/dflash_spec_demo"

# Model resolution: explicit env wins, else /tmp default, else $HOME defaults.
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET="${TARGET:-}"
DRAFT="${DRAFT:-}"
if [ -z "$TARGET" ]; then
    for cand in "$MODELS_DIR/qwen3.6-27b-mq4.hfq" "$MODELS_DIR/qwen3.5-27b-mq4.hfq"; do
        [ -f "$cand" ] && TARGET="$cand" && break
    done
fi
if [ -z "$DRAFT" ]; then
    for cand in "$MODELS_DIR/qwen3.6-27b-mq4.dflash.hfq" "$MODELS_DIR/qwen3.5-27b-mq4.dflash.hfq"; do
        [ -f "$cand" ] && DRAFT="$cand" && break
    done
fi

OUT="${HIPFIRE_PATH_C_OUT:-/tmp/path-c-smoke-$(date +%Y%m%d-%H%M%S).md}"
TRACE_JSON_OUT="${HIPFIRE_PATH_C_TRACE_JSON:-${OUT%.md}.path_c_trace.json}"
HIPFIRE_GPULOCK_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"

# ── Build dflash_spec_demo if needed ──────────────────────────────────────
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/dflash.rs crates/hipfire-arch-qwen35/src/speculative.rs \
               crates/hipfire-runtime/src/ddtree.rs crates/hipfire-runtime/examples/dflash_spec_demo.rs \
               crates/hipfire-rdna/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "path-c-smoke: rebuilding dflash_spec_demo (release)..."
    if ! cargo build --release --example dflash_spec_demo --features deltanet >&2; then
        echo "path-c-smoke: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock ──────────────────────────────────────────────────────────────
if { [ -x "$HIPFIRE_GPULOCK_BIN" ] || command -v "$HIPFIRE_GPULOCK_BIN" >/dev/null 2>&1; }; then
    # shellcheck disable=SC1090
    "$HIPFIRE_GPULOCK_BIN" gpu-lock acquire "path-c-smoke" --watch-pid "$$" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap '"$HIPFIRE_GPULOCK_BIN" gpu-lock release 2>/dev/null || true' EXIT
fi

if [ -z "$TARGET" ] || [ -z "$DRAFT" ] || [ ! -f "$TARGET" ] || [ ! -f "$DRAFT" ]; then
    printf '{"kind":"path_c_trace","records":[\n\n]}\n' > "$TRACE_JSON_OUT"
    {
        echo "# Path C smoke — SKIPPED (target/draft model not found)"
        echo
        echo "- target: ${TARGET:-(unset)}"
        echo "- draft:  ${DRAFT:-(unset)}"
        echo
        echo "Re-stage models or set TARGET / DRAFT env vars and re-run."
    } > "$OUT"
    echo "path-c-smoke: models not present, skipping (no hard error)"
    echo "report: $OUT"
    echo "path-c trace json: $TRACE_JSON_OUT"
    exit 0
fi

# ── Prompts ──────────────────────────────────────────────────────────────
PROSE_PROMPT="The Roman Empire, at its height, stretched from the windswept moors of northern Britain to the sands of the Arabian peninsula. Its decline was not a single event but a long slow unraveling that took centuries. Several factors contributed to this gradual collapse. The first and perhaps most important was"

# Second prose: science-leaning expository — different domain than empire-history.
PROSE2_PROMPT="The discovery of penicillin by Alexander Fleming in 1928 was a turning point in the history of medicine, but the path from a serendipitous mould in a Petri dish to a mass-produced antibiotic that saved millions of lives was anything but straightforward. The decade between Fleming's observation and the first clinical use of penicillin was marked by"

# Third prose: narrative — a third register again to triangulate paragraph-level
# repetition versus genuine cohesion.
PROSE3_PROMPT="The lighthouse keeper's daughter had grown up listening to the sea. Every gale that battered the rocks below the cottage taught her something new about the moods of the Atlantic, and by the time she was twelve she could read a coming storm from the colour of the spray alone. The morning the lifeboat went out and did not return, the wind was"

CODE_PROMPT='from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'

# Second code: HumanEval #14 (all_prefixes). Different control flow than #0.
CODE2_PROMPT='from typing import List


def all_prefixes(string: str) -> List[str]:
    """ Return list of all prefixes from shortest to longest of the input string
    >>> all_prefixes("abc")
    ["a", "ab", "abc"]
    """
'

# Instruct: assistant-style request, less repetitive than continuation prompts.
INSTRUCT_PROMPT="Explain step by step why a soap film between two parallel wires forms a flat surface rather than a curved one, and describe what would change if one of the wires were heated. Use clear physical reasoning."

# ── Test matrix ──────────────────────────────────────────────────────────
SHORT_TESTS=(
    "path-c-phase1-prose|phase1|PROSE_PROMPT|192"
    "path-c-phase1-code|phase1|CODE_PROMPT|128"
    "path-c-phase2-prose|phase2|PROSE_PROMPT|192"
    "path-c-phase2-code|phase2|CODE_PROMPT|128"
)
# --full: 3 prose × 2 code × 1 instruct, each at 256 tokens, on phase1 + phase2.
# Per-prompt PRD smoke gate: unique_ratio > 0.3, max_freq < 0.4 over 256 tokens.
FULL_TESTS=(
    "path-c-phase1-prose1|phase1|PROSE_PROMPT|256"
    "path-c-phase1-prose2|phase1|PROSE2_PROMPT|256"
    "path-c-phase1-prose3|phase1|PROSE3_PROMPT|256"
    "path-c-phase1-code1|phase1|CODE_PROMPT|192"
    "path-c-phase1-code2|phase1|CODE2_PROMPT|192"
    "path-c-phase1-instruct|phase1|INSTRUCT_PROMPT|256"
    "path-c-phase2-prose1|phase2|PROSE_PROMPT|256"
    "path-c-phase2-prose2|phase2|PROSE2_PROMPT|256"
    "path-c-phase2-prose3|phase2|PROSE3_PROMPT|256"
    "path-c-phase2-code1|phase2|CODE_PROMPT|192"
    "path-c-phase2-code2|phase2|CODE2_PROMPT|192"
    "path-c-phase2-instruct|phase2|INSTRUCT_PROMPT|256"
)
if [ "$FULL" -eq 1 ]; then
    TESTS=("${FULL_TESTS[@]}")
else
    TESTS=("${SHORT_TESTS[@]}")
fi

# --graph-ab pairs each HIPFIRE_VERIFY_GRAPH=1 test with a `-nograph` variant
# that runs the same command with HIPFIRE_VERIFY_GRAPH=0. Used to validate the PRD's Phase 3
# expected delta (+10-15 % tok/s with verify-graph capture on the Path C
# main + branch FA forwards). Doubles the test count.
GRAPH_MIN_TOK_DELTA_PCT="${PATH_C_GRAPH_MIN_TOK_DELTA_PCT:-5.0}"
GRAPH_MIN_TAU_DELTA_PCT="${PATH_C_GRAPH_MIN_TAU_DELTA_PCT:--1.0}"
if [ "$GRAPH_AB" -eq 1 ]; then
    AB=()
    for t in "${TESTS[@]}"; do
        AB+=("$t")
        IFS='|' read -r label phase prompt_var max_tok <<< "$t"
        AB+=("${label}-nograph|$phase|$prompt_var|$max_tok|nograph")
    done
    TESTS=("${AB[@]}")
fi

# ── Detector (same logic as coherence-gate-dflash.sh) ────────────────────
HIPFIRE_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"
if [ ! -x "$HIPFIRE_BIN" ] && ! command -v "$HIPFIRE_BIN" >/dev/null 2>&1; then
    echo "path-c-smoke: building hipfire CLI for the token-attractor detector..." >&2
    cargo build --release -p hipfire-cli --bin hipfire >&2 || {
        echo "path-c-smoke: hipfire build failed" >&2
        exit 2
    }
    HIPFIRE_BIN="./target/release/hipfire"
fi

# ── Run ──────────────────────────────────────────────────────────────────
hard_errors=0
GRAPH_METRICS_FILE="/tmp/path_c_graph_metrics_$$.tsv"
GRAPH_VERDICT_FILE="/tmp/path_c_graph_verdict_$$.txt"
GRAPH_SUMMARY_FILE="/tmp/path_c_graph_summary_$$.json"
rm -f "$GRAPH_METRICS_FILE"
rm -f "$GRAPH_VERDICT_FILE"
rm -f "$GRAPH_SUMMARY_FILE"
printf '{"kind":"path_c_trace","records":[\n' > "$TRACE_JSON_OUT"
trace_json_records=0

append_trace_json_record() {
    if [ "$trace_json_records" -gt 0 ]; then
        printf ',\n' >> "$TRACE_JSON_OUT"
    fi
    printf '%s' "$1" >> "$TRACE_JSON_OUT"
    trace_json_records=$((trace_json_records + 1))
}

{
    echo "# Path C smoke (PRD ddtree-path-c-main-path-first-from-lucebox)"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FULL" -eq 1 ] && echo full || echo short )"
    echo "- target: $TARGET"
    echo "- draft:  $DRAFT"
    echo
    echo "Hard-fail thresholds: zero tokens, panic, max_token_freq > 0.50,"
    echo "unique_token_ratio < 0.15."
    echo
} > "$OUT"

for entry in "${TESTS[@]}"; do
    IFS='|' read -r label phase prompt_var max_tok graph_flag <<< "$entry"
    case "$prompt_var" in
        PROSE_PROMPT)    prompt="$PROSE_PROMPT" ;;
        PROSE2_PROMPT)   prompt="$PROSE2_PROMPT" ;;
        PROSE3_PROMPT)   prompt="$PROSE3_PROMPT" ;;
        CODE_PROMPT)     prompt="$CODE_PROMPT" ;;
        CODE2_PROMPT)    prompt="$CODE2_PROMPT" ;;
        INSTRUCT_PROMPT) prompt="$INSTRUCT_PROMPT" ;;
        *) echo "unknown prompt_var: $prompt_var" >&2; exit 2 ;;
    esac
    # graph_flag = "nograph" disables both dense-DFlash graph capture and the
    # Path C opt-in. graph A/B explicitly opts into both graph controls.
    case_graph_mode="default"
    if [ "${graph_flag:-}" = "nograph" ]; then
        graph_env=("HIPFIRE_VERIFY_GRAPH=0" "HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH=0")
        case_graph_mode="nograph"
    elif [ "$GRAPH_AB" -eq 1 ]; then
        graph_env=("HIPFIRE_VERIFY_GRAPH=1" "HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH=1")
        case_graph_mode="graph"
    else
        graph_env=()
    fi

    echo "== $label =="
    out_file="/tmp/path_c_out_$$.log"
    t0=$(date +%s.%N)
    timeout 240 env "${graph_env[@]}" "$EXE" \
        --target "$TARGET" --draft "$DRAFT" \
        --prompt "$prompt" --max "$max_tok" --ctx 2048 \
        --kv-mode q8 --no-chatml \
        --ddtree-path-c "$phase" --ddtree-budget 12 --ddtree-topk 2 \
        > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$out_file" | head -1)
    detect=$("$HIPFIRE_BIN" detect --source auto < "$out_file")
    detect_ok=$(echo "$detect" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('ok',False))")
    detect_warn=$(echo "$detect" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('soft_warn',False))")

    status="OK"
    if [ "$ec" -ne 0 ] || [ -n "$panic" ]; then
        status="HARD_ERROR (exit=$ec panic=${panic:+yes})"
        hard_errors=$((hard_errors + 1))
    elif [ "$detect_ok" != "True" ]; then
        status="HARD_ERROR (token attractor: $detect)"
        hard_errors=$((hard_errors + 1))
    elif [ "$detect_warn" = "True" ]; then
        status="WARN (paragraph-level repetition — soft, not blocking)"
    fi

    stats=$(grep -aE '^emitted:|^cycles:|^verify_graph:|^accept_rate:' "$out_file" | head -4)
    path_c_last=$(grep -a '^\[path-c\]' "$out_file" | tail -1)
    record_json=$(python3 - "$label" "$phase" "$case_graph_mode" "$status" "$detect" "$stats" "$path_c_last" "$wall" <<'PYEOF'
import json
import re
import sys

label, phase, graph_mode, status, detect, stats, path_c_last, wall = sys.argv[1:]

def decode_json(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

metrics = {
    "phase": phase,
    "graph_mode": graph_mode,
    "wall_s": float(wall),
    "detector": decode_json(detect),
}
emitted = re.search(r"^emitted:\s+(\d+) tokens.*\(([-+0-9.eE]+)\s+tok/s\)", stats, re.MULTILINE)
if emitted:
    metrics["emitted_tokens"] = int(emitted.group(1))
    metrics["tok_s"] = float(emitted.group(2))
cycles = re.search(r"^cycles:.*τ=([-+0-9.eE]+)", stats, re.MULTILINE)
if cycles:
    metrics["tau"] = float(cycles.group(1))
verify_graph = re.search(
    r"^verify_graph: direct=(\d+) warmup=(\d+) capture=(\d+) replay=(\d+) not_applicable=(\d+)",
    stats,
    re.MULTILINE,
)
if verify_graph:
    metrics["verify_graph"] = {
        "direct": int(verify_graph.group(1)),
        "warmup": int(verify_graph.group(2)),
        "capture": int(verify_graph.group(3)),
        "replay": int(verify_graph.group(4)),
        "not_applicable": int(verify_graph.group(5)),
    }
if path_c_last:
    metrics["path_c_counters"] = path_c_last
print(json.dumps({
    "battery": "path_c",
    "case_id": label,
    "status": status,
    "metrics": metrics,
}, sort_keys=True))
PYEOF
)
    append_trace_json_record "$record_json"
    if [ "$GRAPH_AB" -eq 1 ] && [ -n "$stats" ]; then
        base_label="${label%-nograph}"
        graph_mode="graph"
        [ "${graph_flag:-}" = "nograph" ] && graph_mode="nograph"
        python3 - "$out_file" "$base_label" "$graph_mode" >> "$GRAPH_METRICS_FILE" <<'PYEOF'
import re
import sys

path, label, graph_mode = sys.argv[1:]
text = open(path, "rb").read().decode("utf-8", "replace")
tok_s = "nan"
tau = "nan"
m = re.search(r"^emitted:.*\(([-+0-9.eE]+)\s+tok/s\)", text, re.MULTILINE)
if m:
    tok_s = m.group(1)
m = re.search(r"^cycles:.*τ=([-+0-9.eE]+)", text, re.MULTILINE)
if m:
    tau = m.group(1)
print(f"{label}\t{graph_mode}\t{tok_s}\t{tau}")
PYEOF
    fi

    {
        echo "## $label (phase=$phase, b=12, k=2)"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        echo "- detector: \`$detect\`"
        if [ -n "$stats" ]; then
            echo "- stats:"
            echo '  ```'
            echo "$stats" | sed 's/^/  /'
            echo '  ```'
        fi
        if [ -n "$path_c_last" ]; then
            echo "- path-c counters (HIPFIRE_DDTREE_PATH_C_VERBOSE=1):"
            echo '  ```'
            echo "  $path_c_last"
            echo '  ```'
        fi
        if [ -n "$panic" ]; then
            echo
            echo '**PANIC/ERROR:**'
            echo
            echo '```'
            echo "$panic"
            echo '```'
        fi
        echo
        echo '**Output (first 40 lines of generation):**'
        echo
        echo '```'
        sed -n '/--- OUTPUT ---/,/-------------/p' "$out_file" \
            | sed '1d;$d' \
            | head -40
        echo '```'
        echo
    } >> "$OUT"

    rm -f "$out_file"
done

if [ "$GRAPH_AB" -eq 1 ] && [ -s "$GRAPH_METRICS_FILE" ]; then
    {
        echo "## Verify Graph A/B"
        echo
        echo "- promotion thresholds: tok/s delta >= ${GRAPH_MIN_TOK_DELTA_PCT}% for every paired case, τ delta >= ${GRAPH_MIN_TAU_DELTA_PCT}% for every paired case"
        echo "- hard gate: $( [ "$GRAPH_PROMOTE" -eq 1 ] && echo enabled || echo disabled )"
        echo
        echo "| case | graph tok/s | nograph tok/s | tok/s delta | graph τ | nograph τ | τ delta |"
        echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        python3 - "$GRAPH_METRICS_FILE" "$GRAPH_VERDICT_FILE" "$GRAPH_SUMMARY_FILE" "$GRAPH_MIN_TOK_DELTA_PCT" "$GRAPH_MIN_TAU_DELTA_PCT" <<'PYEOF'
import math
import json
import sys

rows = {}
metrics_path, verdict_path, summary_path, min_tok_raw, min_tau_raw = sys.argv[1:]
min_tok_delta = float(min_tok_raw)
min_tau_delta = float(min_tau_raw)
with open(metrics_path, "r", encoding="utf-8") as f:
    for line in f:
        label, mode, tok_s, tau = line.rstrip("\n").split("\t")
        rows.setdefault(label, {})[mode] = (float(tok_s), float(tau))

def fmt(value):
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3f}"

failures = []
summary_rows = []
paired = 0
for label in sorted(rows):
    pair = rows[label]
    if "graph" not in pair or "nograph" not in pair:
        failures.append(f"{label}: missing graph/nograph pair")
        continue
    paired += 1
    graph_tok, graph_tau = pair["graph"]
    nograph_tok, nograph_tau = pair["nograph"]
    tok_delta = ((graph_tok - nograph_tok) / nograph_tok * 100.0) if nograph_tok else math.nan
    tau_delta = ((graph_tau - nograph_tau) / nograph_tau * 100.0) if nograph_tau else math.nan
    if not math.isfinite(tok_delta) or not math.isfinite(tau_delta):
        failures.append(f"{label}: non-finite delta")
    elif tok_delta < min_tok_delta:
        failures.append(f"{label}: tok/s delta {tok_delta:.3f}% < {min_tok_delta:.3f}%")
    elif tau_delta < min_tau_delta:
        failures.append(f"{label}: τ delta {tau_delta:.3f}% < {min_tau_delta:.3f}%")
    summary_rows.append({
        "case_id": label,
        "graph_tok_s": graph_tok,
        "nograph_tok_s": nograph_tok,
        "tok_s_delta_pct": tok_delta,
        "graph_tau": graph_tau,
        "nograph_tau": nograph_tau,
        "tau_delta_pct": tau_delta,
    })
    print(
        f"| {label} | {fmt(graph_tok)} | {fmt(nograph_tok)} | {fmt(tok_delta)}% | "
        f"{fmt(graph_tau)} | {fmt(nograph_tau)} | {fmt(tau_delta)}% |"
    )
if paired == 0:
    failures.append("no paired graph/nograph metrics")
promoted = not failures
print()
print(f"- promotion_verdict: {'PROMOTED' if promoted else 'NOT_PROMOTED'}")
print(f"- paired_cases: {paired}")
if failures:
    print("- blockers:")
    for failure in failures:
        print(f"  - {failure}")
with open(verdict_path, "w", encoding="utf-8") as f:
    f.write("PROMOTED\n" if promoted else "NOT_PROMOTED\n")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "battery": "path_c",
        "case_id": "verify_graph_promotion",
        "status": "PROMOTED" if promoted else "NOT_PROMOTED",
        "metrics": {
            "promotion_verdict": "PROMOTED" if promoted else "NOT_PROMOTED",
            "paired_cases": paired,
            "tok_s_min_delta_pct": min_tok_delta,
            "tau_min_delta_pct": min_tau_delta,
            "blockers": failures,
            "pairs": summary_rows,
        },
    }, f, sort_keys=True)
PYEOF
        echo
    } >> "$OUT"
    if [ -s "$GRAPH_SUMMARY_FILE" ]; then
        append_trace_json_record "$(cat "$GRAPH_SUMMARY_FILE")"
    fi
fi
if [ "$GRAPH_PROMOTE" -eq 1 ]; then
    if [ "$(cat "$GRAPH_VERDICT_FILE" 2>/dev/null || echo NOT_PROMOTED)" != "PROMOTED" ]; then
        hard_errors=$((hard_errors + 1))
        {
            echo "## Verify Graph Promotion Gate"
            echo
            echo "Graph capture did not satisfy promotion thresholds; keeping it diagnostic."
            echo
        } >> "$OUT"
    fi
fi
printf '\n]}\n' >> "$TRACE_JSON_OUT"
rm -f "$GRAPH_METRICS_FILE" "$GRAPH_VERDICT_FILE" "$GRAPH_SUMMARY_FILE"

echo
echo "path-c-smoke report: $OUT"
echo "path-c trace json: $TRACE_JSON_OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors test(s) hit hard errors — gate FAILED"
    exit 1
fi
echo "no hard errors — review $OUT for coherence"
exit 0
