#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEARCH_PATHS=(
    AGENTS.md
    README.md
    cli
    tests
    scripts
    crates
    docs
    benchmarks
)
EXISTING_SEARCH_PATHS=()
for path in "${SEARCH_PATHS[@]}"; do
    if [ -e "$path" ]; then
        EXISTING_SEARCH_PATHS+=("$path")
    fi
done

status=0

# benchmarks/results/** is generated run output: model paths + eval-dir names
# record what was actually evaluated (a historical log, e.g. older hfq4/hfq6
# runs). The gate keeps *authored* artifacts/scripts/docs canonical; it must not
# rewrite past experiment records, so those trees are excluded below.

echo "check-artifact-names: dotted quant artifact suffixes"
if rg -n \
    --glob '!target/**' \
    --glob '!**/*.lock' \
    --glob '!scripts/check-artifact-names.sh' \
    --glob '!benchmarks/results/**' \
    --glob '!**/findings-archive/**' \
    -- '\.hfq-(?:hf|mq)[1-8]|\.q[1-8]\.hfq|[-.]hfq[1-8]\.hfq' \
    "${EXISTING_SEARCH_PATHS[@]}"; then
    status=1
fi

echo "check-artifact-names: legacy dflash quant ordering"
if rg -n \
    --glob '!target/**' \
    --glob '!**/*.lock' \
    --glob '!scripts/check-artifact-names.sh' \
    --glob '!benchmarks/results/**' \
    --glob '!**/findings-archive/**' \
    -- '(?:qwen3[._-]?[56]|qwen3[56])-[A-Za-z0-9_.-]+-dflash-(?:hf|mq)[1-8]|dflash-(?:hf|mq)[1-8]' \
    "${EXISTING_SEARCH_PATHS[@]}"; then
    status=1
fi

if [ "$status" -ne 0 ]; then
    cat >&2 <<'EOF'
check-artifact-names: legacy artifact spelling found.
Use canonical names such as:
  qwen3.5-9b-mq4.hfq
  qwen3.5-9b-mq4.dflash.hfq
  deepseek-v4-flash-lloyd-mq2.hfq
EOF
fi

exit "$status"
