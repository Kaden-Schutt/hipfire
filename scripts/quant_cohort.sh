#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# hipfire — see LICENSE and NOTICE in the project root.

# quant_cohort.sh — Phase A Step 0 cohort runner.
#
# Orchestrates a per-format quality bench cohort. Per variant emits:
#   - Per-tensor MSE table (quant_quality_mse, vs BF16 safetensors)
#   - KLD vs FP16 reference  [STUB — fills in once chore/113-quant-eval-plan lands]
#   - PPL on wikitext-2-test [STUB — same source]
#   - HumanEval completion capture (bench_humaneval_completion.sh)
#   - Train-pursuit reasoning smoke
#
# Output layout matches chore/113-quant-eval-plan's results/ schema so
# kld_reduce.py can pick up the per-seq files unchanged once eval_hipfire
# starts producing them:
#
#   benchmarks/quality-baselines/results/YYYY-MM-DD-cohort-<label>/
#     ├── manifest.json                 # cohort metadata (variants, refs, flags)
#     ├── per-variant/
#     │   ├── <variant>__<arch>.mse.txt     # quant_quality_mse output
#     │   ├── <variant>__<arch>.humaneval.jsonl
#     │   ├── <variant>__<arch>.smoke.md    # train-pursuit verdict
#     │   └── <variant>__<arch>.kldseq      # ← lands here once eval_hipfire merges
#     ├── result-table.md               # aggregated markdown (built by this script)
#     └── result-table.md.in-progress   # written as we go, renamed on success
#
# Usage:
#   scripts/quant_cohort.sh <cohort-label> <bf16-ref-dir> <variant-spec.tsv> \
#       [--kldref <path>] [--max-chunks N] [--kv-mode <mode>] [--scoring-mode <mode>]
#
# variant-spec.tsv format (tab-separated; one row per variant):
#   <variant-name>  <hfq-path>  <arch>
# Example:
#   qwen35-9b-mq4-uniform  /local/hipfire/qwen3.5-9b.mq4         gfx906
#   qwen35-9b-mfp4         /local/hipfire/qwen3.5-9b.mfp4        gfx906
#   qwen35-9b-hfp4         /local/hipfire/qwen3.5-9b.hfp4        gfx906
#
# Optional flags:
#   --kldref <path>        Path to .kldref.bin BF16 reference for the
#                          model family. If set, KLD column populates via
#                          eval_hipfire. If unset, KLD column shows
#                          "(no-kldref)" and only MSE/smoke/HE run.
#                          Reference fetcher: scripts/fetch-eval-refs.sh
#                          (refs go to benchmarks/quality-baselines/refs/).
#   --max-chunks N         Cap eval_hipfire to N chunks (full slice is
#                          1175). 256 chunks ≈ 1.4 h per 9B variant on
#                          gfx906; full slice ≈ 9-11 h per variant.
#                          Use for during-development iteration; omit for
#                          "lock the result" cohorts.
#   --kv-mode <mode>       q8 | asym2 | asym3 | asym4 (default: asym3)
#   --scoring-mode <mode>  per-token | prefill (default: prefill, the
#                          canonical hipfire scoring path per #113)
#
# Example invocation:
#   scripts/quant_cohort.sh \
#     phase-a-step-0-baselines \
#     ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/SNAP \
#     /tmp/cohort_baselines.tsv \
#     --kldref benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.kldref.bin \
#     --max-chunks 256

set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
    echo "usage: $0 <cohort-label> <bf16-ref-dir-or-file> <variant-spec.tsv> [flags...]"
    echo
    echo "variant-spec.tsv format (TAB-separated):"
    printf '  <variant-name>\t<hfq-path>\t<arch>\n'
    echo
    echo "Optional flags: --kldref PATH, --max-chunks N, --kv-mode MODE, --scoring-mode MODE"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi
if [ $# -lt 3 ]; then
    usage
    exit 2
fi

LABEL="$1"
ST_DIR="$2"
SPEC="$3"
shift 3

# Optional flags.
KLDREF=""
MAX_CHUNKS=""
KV_MODE="asym3"
SCORING_MODE="prefill"
while [ $# -gt 0 ]; do
    case "$1" in
        --kldref)      KLDREF="$2"; shift 2 ;;
        --max-chunks)  MAX_CHUNKS="$2"; shift 2 ;;
        --kv-mode)     KV_MODE="$2"; shift 2 ;;
        --scoring-mode) SCORING_MODE="$2"; shift 2 ;;
        *) echo "error: unknown flag: $1"; exit 2 ;;
    esac
done

if [ -n "$KLDREF" ] && [ ! -f "$KLDREF" ]; then
    echo "error: --kldref path not found: $KLDREF"
    echo "       run scripts/fetch-eval-refs.sh to download (needs huggingface_hub in .venv)"
    exit 1
fi

if [ ! -e "$ST_DIR" ]; then
    echo "error: bf16 reference dir/file not found: $ST_DIR"
    exit 1
fi
if [ ! -f "$SPEC" ]; then
    echo "error: variant spec not found: $SPEC"
    exit 1
fi

DATE=$(date -u +%Y-%m-%d)
COHORT_DIR="benchmarks/quality-baselines/results/${DATE}-cohort-${LABEL}"
mkdir -p "${COHORT_DIR}/per-variant"

# ─── Manifest ─────────────────────────────────────────────────────────────
HOSTNAME=$(hostname)
GIT_HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)

python3 -c "
import json, os
manifest = {
    'cohort_label': '${LABEL}',
    'date_utc': '${DATE}',
    'host': '${HOSTNAME}',
    'git_head': '${GIT_HEAD}',
    'git_branch': '${GIT_BRANCH}',
    'bf16_ref': '${ST_DIR}',
    'variant_spec': '${SPEC}',
    'variants': [],
}
with open('${SPEC}') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) != 3:
            raise SystemExit(f'malformed spec row: {line!r} (expected 3 TAB-separated fields)')
        name, hfq_path, arch = parts
        manifest['variants'].append({'name': name, 'hfq_path': hfq_path, 'arch': arch})
with open('${COHORT_DIR}/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'  cohort with {len(manifest[\"variants\"])} variants')
"

# Build prerequisites once.
echo "Building prerequisites..."
cargo build --release --example quant_quality_mse --quiet 2>&1 | tail -3
if [ -n "$KLDREF" ]; then
    cargo build --release --example eval_hipfire --features deltanet --quiet 2>&1 | tail -3
fi

PROGRESS="${COHORT_DIR}/result-table.md.in-progress"
KLD_DESC=""
if [ -n "$KLDREF" ]; then
    KLDREF_BASENAME=$(basename "$KLDREF")
    if [ -n "$MAX_CHUNKS" ]; then
        KLD_DESC="**KLD ref:** \`${KLDREF_BASENAME}\` (max_chunks=${MAX_CHUNKS}; quick-slice)"
    else
        KLD_DESC="**KLD ref:** \`${KLDREF_BASENAME}\` (full slice 1175 chunks)"
    fi
else
    KLD_DESC="**KLD ref:** not provided — KLD column will show \`(no-kldref)\`"
fi
{
    echo "# Cohort: ${LABEL}"
    echo
    echo "**Date:** ${DATE}"
    echo "**Host:** ${HOSTNAME}"
    echo "**Git:** ${GIT_BRANCH} @ ${GIT_HEAD:0:8}"
    echo "**BF16 ref (safetensors / MSE):** \`${ST_DIR}\`"
    echo "${KLD_DESC}"
    echo "**KV mode:** ${KV_MODE} · **Scoring mode:** ${SCORING_MODE}"
    echo
    echo "## Per-variant metrics"
    echo
    echo "| Variant | Arch | MSE mean (4-bit qts) | KLD mean ± CI | KLD p99 | PPL | HE tokens (sum) | Smoke |"
    echo "|---|---|---:|---|---:|---:|---:|---|"
} > "$PROGRESS"

# Spiral-detection prompt (matches existing bench_quant_quality.sh; intentionally
# byte-identical so prior bench runs are comparable). Embedded here rather than
# read from disk to avoid newline-normalization drift.
PROMPT='A train leaves Station A traveling at 60 km/h. Two hours later, a second train leaves Station A on the same track traveling at 90 km/h. How long after the second train departs will it catch up to the first? Show your reasoning step by step.'

# wait_for_model_ready <hfq_path> [timeout_sec]
# Polls /v1/models until the requested model is registered. Returns 0 on success.
# Used after `hipfire serve -d` returns (the CLI already polls /health before
# exiting). The legacy `tail -1 serve.log | grep "warm-up complete"` check is
# broken because serve.log is opened O_APPEND — a stale "warm-up complete" line
# from the previous session passes the gate before the new daemon binds. And
# `pgrep -af "examples/daemon"` doesn't match anymore — the CLI now spawns
# `bun ... serve <port>`, not a standalone daemon binary.
wait_for_model_ready() {
    local hfq_path="$1"; local timeout="${2:-120}"
    local want; want=$(basename "$hfq_path")
    local start; start=$(date +%s)
    local tmp; tmp=$(mktemp)
    while [ $(( $(date +%s) - start )) -lt "$timeout" ]; do
        if curl -sS --max-time 3 -o "$tmp" http://127.0.0.1:8080/v1/models 2>/dev/null; then
            if python3 -c "
import sys, json
try:
    with open('$tmp') as f:
        d = json.load(f)
    sys.exit(0 if any(m.get('id','').endswith('$want') for m in d.get('data', [])) else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
                rm -f "$tmp"; return 0
            fi
        fi
        sleep 2
    done
    rm -f "$tmp"; return 1
}

# ─── Per-variant loop ────────────────────────────────────────────────────
while IFS=$'\t' read -r VARIANT HFQ_PATH ARCH; do
    # Skip blank/comment rows.
    [ -z "${VARIANT:-}" ] && continue
    case "$VARIANT" in '#'*) continue ;; esac

    echo
    echo "═══ Variant: ${VARIANT} (${ARCH}) ═══"
    if [ ! -e "$HFQ_PATH" ]; then
        echo "  SKIP: model file not found: $HFQ_PATH"
        echo "| ${VARIANT} | ${ARCH} | — | — | — | — | — | MODEL_MISSING | — |" >> "$PROGRESS"
        continue
    fi

    PV="${COHORT_DIR}/per-variant/${VARIANT}__${ARCH}"

    # ─── 1. Per-tensor MSE ──────────────────────────────────────────────
    echo "  [1/4] per-tensor MSE..."
    ./target/release/examples/quant_quality_mse "$ST_DIR" "$HFQ_PATH" > "${PV}.mse.txt" 2>&1 \
        || { echo "    (MSE run failed; output captured anyway)"; }

    # Extract the aggregate 4-bit MSE (mean over qts ∈ {13, 21, 24} == MQ4/HFP4/MFP4).
    # Falls back to "—" if none of those qts are present.
    MSE_MEAN=$(python3 -c "
import re, sys
qts_4bit = {'MQ4G256', 'HFP4G32', 'MFP4G32'}
text = open('${PV}.mse.txt').read()
# parse the 'Aggregate stats by quant type' table
m = re.search(r'=== Aggregate stats by quant type ===(.*?)(?:\n\n|\Z)', text, re.S)
if not m:
    print('—'); sys.exit()
lines = [l for l in m.group(1).strip().split('\n') if l and not l.startswith('-') and not l.startswith('qt')]
mses = []
for line in lines:
    parts = line.split()
    if len(parts) >= 4 and parts[0] in qts_4bit:
        try:
            mses.append(float(parts[3]))
        except ValueError:
            pass
if not mses:
    print('—')
else:
    avg = sum(mses) / len(mses)
    print(f'{avg:.2e}')
")

    # ─── 2. KLD / PPL — eval_hipfire ────────────────────────────────────
    KLD_MEAN="(no-kldref)"
    KLD_P99="—"
    PPL_VAL="—"
    if [ -n "$KLDREF" ]; then
        echo "  [2/4] KLD / PPL via eval_hipfire (kv-mode=${KV_MODE}, scoring=${SCORING_MODE}, max_chunks=${MAX_CHUNKS:-full})..."

        # Build invocation. eval_hipfire writes HFKSEQ to ${PV}.kldseq;
        # kld_reduce.py converts to row-level mean/CI/p99/PPL.
        EVAL_ARGS=(
            --model "$HFQ_PATH"
            --ref "$KLDREF"
            --output "${PV}.kldseq"
            --kv-mode "$KV_MODE"
            --scoring-mode "$SCORING_MODE"
        )
        if [ -n "$MAX_CHUNKS" ]; then
            EVAL_ARGS+=(--max-chunks "$MAX_CHUNKS")
        fi

        # GPU lock — eval_hipfire takes hours; coordinate with other jobs.
        if [ -f "scripts/gpu-lock.sh" ]; then
            # shellcheck disable=SC1091
            source scripts/gpu-lock.sh
            gpu_acquire "quant_cohort-${LABEL}-${VARIANT}" 2>&1 | tail -1 || true
        fi

        EVAL_START=$(date +%s)
        if ./target/release/examples/eval_hipfire "${EVAL_ARGS[@]}" \
                > "${PV}.eval.log" 2>&1; then
            EVAL_OK=1
        else
            EVAL_OK=0
            echo "    eval_hipfire FAILED (log: ${PV}.eval.log)"
        fi
        EVAL_WALL=$(( $(date +%s) - EVAL_START ))
        echo "    eval_hipfire wall: ${EVAL_WALL}s"

        if [ -f "scripts/gpu-lock.sh" ]; then
            gpu_release "quant_cohort-${LABEL}-${VARIANT}" 2>&1 | tail -1 || true
        fi

        if [ "$EVAL_OK" = "1" ] && [ -f "${PV}.kldseq" ]; then
            # Reduce single .kldseq file via the inline reducer pattern.
            # We can't use kld_reduce.py's directory-walking interface
            # for one file, so we call the Python read_per_seq_kld helper
            # directly. kldref_format.py lives in harness/.
            #
            # Use the repo's .venv/bin/python3 (per scripts/fetch-eval-refs.sh
            # convention) so numpy is available without polluting system
            # Python. Falls back to python3 if .venv isn't set up.
            if [ -x ".venv/bin/python3" ]; then
                REDUCE_PY=".venv/bin/python3"
            else
                REDUCE_PY="python3"
                echo "    (warning: .venv/bin/python3 not found; falling back to system python3)"
                echo "    (set up:  python3 -m venv .venv && .venv/bin/pip install numpy)"
            fi
            REDUCE_OUT=$("$REDUCE_PY" -c "
import sys, json
from pathlib import Path
sys.path.insert(0, 'benchmarks/quality-baselines/harness')
from kldref_format import read_per_seq_kld
import numpy as np

path = Path('${PV}.kldseq')
means, p99s, nlls = read_per_seq_kld(path)
means_arr = np.asarray(means, dtype=np.float64)
p99s_arr = np.asarray(p99s, dtype=np.float64)
nlls_arr = np.asarray(nlls, dtype=np.float64)

# Bootstrap 95% CI on slice-mean.
rng = np.random.default_rng(0)
idx = rng.integers(0, len(means_arr), size=(10_000, len(means_arr)))
boot_means = means_arr[idx].mean(axis=1)
ci_lo = float(np.percentile(boot_means, 2.5))
ci_hi = float(np.percentile(boot_means, 97.5))

slice_mean = float(means_arr.mean())
p99 = float(np.percentile(p99s_arr, 99))
finite_nll = nlls_arr[np.isfinite(nlls_arr)]
ppl = float(np.exp(finite_nll.mean())) if finite_nll.size else float('nan')

# Output as TSV for shell consumption.
print(f'{slice_mean:.4f}\t{ci_lo:.4f}\t{ci_hi:.4f}\t{p99:.3f}\t{ppl:.3f}')
" 2>"${PV}.reduce.err" || echo "err err err err err")
            IFS=$'\t' read -r SM CI_LO CI_HI P99 PPL <<< "$REDUCE_OUT"
            if [ "$SM" != "err" ]; then
                KLD_MEAN="${SM} (CI ${CI_LO}-${CI_HI})"
                KLD_P99="${P99}"
                PPL_VAL="${PPL}"
            else
                echo "    reduce failed (err log: ${PV}.reduce.err)"
            fi
        fi
    else
        echo "  [2/4] KLD / PPL: skipped (no --kldref provided)"
    fi

    # ─── 3. HumanEval completion capture ──────────────────────────────
    echo "  [3/4] HumanEval prompts..."
    if command -v hipfire >/dev/null 2>&1; then
        if bash scripts/bench_humaneval_completion.sh "$HFQ_PATH" "${PV}.humaneval.jsonl" \
                2>&1 | tail -20 > "${PV}.humaneval.log"; then
            HE_TOK_SUM=$(python3 -c "
import json
rows = [json.loads(l) for l in open('${PV}.humaneval.jsonl')]
s = sum(r.get('tokens_used', 0) for r in rows if 'error' not in r)
print(s)
")
        else
            HE_TOK_SUM="FAIL"
            echo "    (humaneval failed; log: ${PV}.humaneval.log)"
        fi
    else
        HE_TOK_SUM="(no CLI)"
        echo "    (skip: hipfire CLI not on PATH)"
    fi

    # ─── 4. Reasoning smoke ─────────────────────────────────────────────
    echo "  [4/4] reasoning smoke..."
    SMOKE="—"
    if command -v hipfire >/dev/null 2>&1; then
        hipfire stop 2>&1 | head -1 || true; sleep 2
        HIPFIRE_MODEL="$HFQ_PATH" hipfire serve 8080 -d 2>&1 | tail -1 >/dev/null
        if ! wait_for_model_ready "$HFQ_PATH" 300; then
            SMOKE="ERR_DAEMON_NOT_READY"
            hipfire stop 2>&1 | head -1 || true
            echo "| ${VARIANT} | ${ARCH} | ${MSE_MEAN} | ${KLD_MEAN} | ${KLD_P99} | ${PPL_VAL} | ${HE_TOK_SUM} | ${SMOKE} |" >> "$PROGRESS"
            continue
        fi

        MODEL_ID=$(curl -sS http://127.0.0.1:8080/v1/models 2>/dev/null \
            | python3 -c "import sys,json; ms=json.load(sys.stdin)['data']; n='$(basename "$HFQ_PATH")'; [print(m['id']) for m in ms if m['id'].endswith(n)]" \
            | head -1)
        [ -z "$MODEL_ID" ] && MODEL_ID="$(basename "$HFQ_PATH")"

        body=$(python3 -c "
import json
print(json.dumps({'model':'$MODEL_ID','messages':[{'role':'user','content':'''$PROMPT'''}],'temperature':0,'max_tokens':400}))
")
        timeout 300 curl -sS http://127.0.0.1:8080/v1/chat/completions \
            -H 'Content-Type: application/json' -d "$body" > "${PV}.smoke.json" 2>&1 || true

        SMOKE=$(python3 -c "
import json
try:
    d = json.load(open('${PV}.smoke.json'))
    c = d['choices'][0]['message']['content']
    n = len(c)
    if n == 0: print('SPIRAL')
    elif n > 800: print(f'COHERENT_{n}c')
    else: print(f'PARTIAL_{n}c')
except Exception as e:
    print(f'ERR_{e}')
")
        hipfire stop 2>&1 | head -1 || true
    fi

    echo "| ${VARIANT} | ${ARCH} | ${MSE_MEAN} | ${KLD_MEAN} | ${KLD_P99} | ${PPL_VAL} | ${HE_TOK_SUM} | ${SMOKE} |" >> "$PROGRESS"

done < "$SPEC"

# ─── Finalize ────────────────────────────────────────────────────────────
{
    echo
    echo "## Notes"
    echo
    echo "- **MSE mean (4-bit qts):** average per-tensor MSE across MQ4G256 / HFP4G32 / MFP4G32 tensors (excluding Q8/F16-promoted tensors). Lower is better for raw reconstruction. **Not a model-quality signal alone** — see \`docs/plans/hfp4-fivetide-rebuttal-perspective.md\`."
    echo "- **KLD mean ± CI:** slice-mean KLD vs the BF16 reference logits, with 95% bootstrap CI. From \`eval_hipfire\` (#113 infrastructure). Lower is better. Shows \`(no-kldref)\` when --kldref wasn't passed."
    echo "- **KLD p99:** 99th-percentile of per-sequence p99 KLD values. Catches tail-distribution divergence that the slice mean misses (the case fivetide's PPL vs KLD disagreement surfaced)."
    echo "- **PPL:** exp(mean NLL) across all scored tokens. Wikitext-2-test slice (1175 chunks, n_ctx=2048, slice md5 \`83b0205a\`; or fewer if --max-chunks was used)."
    echo "- **HE tokens (sum):** sum of completion tokens across in-tree humaneval prompts (3 prompts). Sanity signal for 'model produces non-zero output on code prompts'; pass@1 scoring is a follow-up."
    echo "- **Smoke:** train-pursuit reasoning attractor check at temp=0, max_tokens=400. \`SPIRAL\` = empty content after \`<think>\` strip; \`COHERENT_<N>c\` = N chars of reasoning; \`PARTIAL_<N>c\` = N chars but suspiciously short."
} >> "$PROGRESS"

mv "$PROGRESS" "${COHORT_DIR}/result-table.md"

echo
echo "═══ Cohort complete ═══"
echo "  Output: ${COHORT_DIR}/result-table.md"
echo "  Per-variant artifacts: ${COHORT_DIR}/per-variant/"
echo
cat "${COHORT_DIR}/result-table.md"
