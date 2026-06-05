#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Minimal GPU smoke for the unified hipfire-eval harness.
#
# Defaults to the smallest staged Qwen model and writes standard eval artifacts
# under ~/.hipfire/eval-results/smoke/. This is path validation, not a perf or
# quality claim.
#
# Environment:
#   HIPFIRE_EVAL_SMOKE_MODEL       target model path
#   HIPFIRE_EVAL_SMOKE_DRAFT       optional DFlash draft path; when unset, the
#                                   dflash group relies on hipfire-eval auto-discovery
#   HIPFIRE_EVAL_SMOKE_MAX_TOKENS  decode cap, default 4
#   HIPFIRE_EVAL_SMOKE_OUT_ROOT    output root, default ~/.hipfire/eval-results/smoke
#   HIPFIRE_EVAL_SMOKE_GROUP       shape|speed|dflash|all, default all
#   HIPFIRE_EVAL_BIN               optional hipfire-eval binary override

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${HIPFIRE_EVAL_SMOKE_MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b.mq4}"
DRAFT="${HIPFIRE_EVAL_SMOKE_DRAFT:-}"
MAX_TOKENS="${HIPFIRE_EVAL_SMOKE_MAX_TOKENS:-4}"
OUT_ROOT="${HIPFIRE_EVAL_SMOKE_OUT_ROOT:-$HOME/.hipfire/eval-results/smoke}"
GROUP="${HIPFIRE_EVAL_SMOKE_GROUP:-all}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

case "$GROUP" in
    shape|speed|dflash|all) ;;
    *)
        echo "eval-smoke: unknown HIPFIRE_EVAL_SMOKE_GROUP=$GROUP; expected shape|speed|dflash|all" >&2
        exit 2
        ;;
esac

if [ ! -f "$MODEL" ]; then
    echo "eval-smoke: model not found: $MODEL" >&2
    echo "eval-smoke: set HIPFIRE_EVAL_SMOKE_MODEL to a local .hfq/.mq artifact" >&2
    exit 2
fi

if [ -n "$DRAFT" ] && [ ! -f "$DRAFT" ]; then
    echo "eval-smoke: draft not found: $DRAFT" >&2
    exit 2
fi

pick_eval_bin() {
    if [ -n "${HIPFIRE_EVAL_BIN:-}" ] && [ -x "$HIPFIRE_EVAL_BIN" ]; then
        printf '%s\n' "$HIPFIRE_EVAL_BIN"
        return 0
    fi
    for candidate in \
        "$ROOT/target/release/hipfire-eval" \
        "$ROOT/target/debug/hipfire-eval"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "eval-smoke: hipfire-eval binary not found; run cargo build -p hipfire-runtime --bin hipfire-eval" >&2
    return 1
}

EVAL_BIN="$(pick_eval_bin)"
mkdir -p "$OUT_ROOT"

LOCK_SCRIPT="$ROOT/scripts/gpu-lock.sh"
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "eval-harness-gpu-smoke"
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

run_eval() {
    local label="$1"
    shift
    local out="$OUT_ROOT/${label}-${STAMP}"
    echo "eval-smoke: running $label"
    "$EVAL_BIN" "$@" --out "$out"
    python3 - "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines() if line.strip()]
manifest = json.loads((out / "manifest.json").read_text())
summary_md = (out / "summary.md").read_text()
print(json.dumps({
    "out": str(out),
    "rows": len(rows),
    "pass": sum(1 for r in rows if r["status"] == "pass"),
    "fail": sum(1 for r in rows if r["status"] == "fail"),
    "skip": sum(1 for r in rows if r["status"] == "skip"),
    "arch": manifest.get("arch"),
    "rocm": manifest.get("rocm"),
    "binary_hash_present": bool(manifest.get("binary_hash")),
}, sort_keys=True))
if any(r["status"] == "fail" for r in rows):
    raise SystemExit(1)
if not manifest.get("hipfire_version") or not manifest.get("git_commit"):
    raise SystemExit("eval-smoke: manifest missing hipfire version/git provenance")
for required in ["## Models", "## Datasets", "## Admission", "## Evidence Artifacts", "## Rows"]:
    if required not in summary_md:
        raise SystemExit(f"eval-smoke: summary missing {required}")
if "hipfire version:" not in summary_md or "git commit:" not in summary_md:
    raise SystemExit("eval-smoke: summary missing hipfire version/git provenance")
PY
}

if [ "$GROUP" = "all" ] || [ "$GROUP" = "shape" ]; then
    run_eval "shape" \
        --model "$MODEL" \
        --battery smoke,prompt_shape,structured \
        --executor examples \
        --max-tokens "$MAX_TOKENS" \
        --kv-mode q8
fi

if [ "$GROUP" = "all" ] || [ "$GROUP" = "speed" ]; then
    run_eval "speed-dflash-skip" \
        --model "$MODEL" \
        --battery speed,dflash \
        --executor examples \
        --max-tokens "$MAX_TOKENS" \
        --kv-mode q8 \
        --dflash auto
fi

if [ "$GROUP" = "all" ] || [ "$GROUP" = "dflash" ]; then
    if [ "$GROUP" = "all" ] && [ -z "$DRAFT" ]; then
        echo "eval-smoke: HIPFIRE_EVAL_SMOKE_DRAFT not set; skipping paired DFlash smoke in all group"
        exit 0
    fi
    DFLASH_ARGS=(
        --model "$MODEL" \
        --battery dflash \
        --executor examples \
        --max-tokens "$MAX_TOKENS" \
        --kv-mode q8 \
        --dflash auto
    )
    if [ -n "$DRAFT" ]; then
        DFLASH_ARGS+=(--draft "$DRAFT")
    fi
    out="$OUT_ROOT/paired-dflash-${STAMP}"
    echo "eval-smoke: running paired-dflash"
    "$EVAL_BIN" "${DFLASH_ARGS[@]}" --out "$out"
    python3 - "$out" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines() if line.strip()]
manifest = json.loads((out / "manifest.json").read_text())
trace = json.loads((out / "artifacts" / "dflash_trace.json").read_text())
summary_md = (out / "summary.md").read_text()

summary = {
    "out": str(out),
    "rows": len(rows),
    "pass": sum(1 for r in rows if r["status"] == "pass"),
    "fail": sum(1 for r in rows if r["status"] == "fail"),
    "skip": sum(1 for r in rows if r["status"] == "skip"),
    "arch": manifest.get("arch"),
    "rocm": manifest.get("rocm"),
    "binary_hash_present": bool(manifest.get("binary_hash")),
    "draft_hash_present": bool(manifest.get("draft_hash")),
}
print(json.dumps(summary, sort_keys=True))

if any(r["status"] == "fail" for r in rows):
    raise SystemExit("eval-smoke: paired DFlash rows contain failures")
if not manifest.get("hipfire_version") or not manifest.get("git_commit"):
    raise SystemExit("eval-smoke: paired DFlash manifest missing hipfire version/git provenance")
for required in ["## Models", "## Datasets", "## Admission", "## Evidence Artifacts", "## Rows"]:
    if required not in summary_md:
        raise SystemExit(f"eval-smoke: paired DFlash summary missing {required}")
if "hipfire version:" not in summary_md or "git commit:" not in summary_md:
    raise SystemExit("eval-smoke: paired DFlash summary missing hipfire version/git provenance")

ar_rows = [r for r in rows if r["battery"] == "dflash" and r["case_id"] == "ar_anchor"]
dflash_rows = [r for r in rows if r["battery"] == "dflash" and r["case_id"] == "dflash_anchor"]
if not ar_rows or any(r["status"] != "pass" for r in ar_rows):
    raise SystemExit("eval-smoke: paired DFlash smoke requires passing ar_anchor rows")
if len(dflash_rows) != 1 or dflash_rows[0]["status"] != "pass":
    reason = dflash_rows[0].get("reason") if dflash_rows else "missing dflash_anchor"
    raise SystemExit(f"eval-smoke: paired DFlash smoke requires passing dflash_anchor: {reason}")
if not manifest.get("draft_hash"):
    raise SystemExit("eval-smoke: paired DFlash smoke requires a recorded draft hash")

modes = {record.get("metrics", {}).get("mode") for record in trace.get("records", [])}
if not {"ar", "dflash"}.issubset(modes):
    raise SystemExit(f"eval-smoke: dflash_trace missing ar/dflash records: {sorted(modes)}")
PY
fi
