#!/usr/bin/env bash
# Regression prompts from issue #8 -- output correctness suite
# https://github.com/Kaden-Schutt/hipfire/issues/8
#
# Tests tokenizer integrity, structured output, and decode-path
# correctness. Run against any model to catch regressions.
#
# Usage: MODEL=qwen3.5:9b tests/regression_issue8.sh
#        MODEL=qwen3.5:35b-a3b tests/regression_issue8.sh

set -uo pipefail

MODEL=${MODEL:-qwen3.5:9b}
PASS=0
FAIL=0

run() { hipfire run "$MODEL" --max-tokens 64 --temp 0 "$1" 2>/dev/null; }

norm() { printf '%s' "$1" | tr -d '\r' | sed 's/[[:space:]]*$//; s/^[[:space:]]*//'; }

check_contains() {
    local name="$1" expected="$2" output="$3"
    if echo "$output" | grep -qF "$expected"; then
        echo "PASS $name"
        ((PASS++))
    else
        echo "FAIL $name"
        echo "  expected to contain: $expected"
        echo "  got: $output"
        ((FAIL++))
    fi
}

check_exact() {
    local name="$1" expected="$2" output="$3"
    if [[ "$(norm "$output")" == "$expected" ]]; then
        echo "PASS $name"
        ((PASS++))
    else
        echo "FAIL $name"
        echo "  expected: $expected"
        echo "  got: $(norm "$output")"
        ((FAIL++))
    fi
}

check_not_contains() {
    local name="$1" bad="$2" output="$3"
    if echo "$output" | grep -qF "$bad"; then
        echo "FAIL $name (found: $bad)"
        ((FAIL++))
    else
        echo "PASS $name"
        ((PASS++))
    fi
}

echo "=== Issue #8 regression suite -- $MODEL ==="

# 1. Primes: catches tokenizer byte-map bugs, digit concatenation (2,3,5711)
echo "[1] Primes"
OUT=$(run "Return the first 5 prime numbers, comma-separated. Nothing else.")
check_contains "primes: has 2, 3, 5, 7, 11" "2, 3, 5, 7, 11" "$OUT"
check_not_contains "primes: no concatenation" "5711" "$OUT"

# 2. Numbers: catches output assembly corruption, missing separators
echo "[2] Numbers 1-10"
OUT=$(run "Return numbers 1 to 10, one per line. Nothing else.")
LINES=$(echo "$OUT" | grep -cE '^[0-9]+$' || true)
if [[ "$LINES" -ge 10 ]]; then
    echo "PASS numbers: $LINES numeric lines"
    ((PASS++))
else
    echo "FAIL numbers: expected >=10 numeric lines, got $LINES"
    echo "  output: $OUT"
    ((FAIL++))
fi

# 3. JSON: catches escape handling, special character corruption
echo "[3] JSON"
OUT=$(run 'Return exactly this JSON, nothing else: {"a":1,"b":2}')
if echo "$OUT" | jq -e '. == {"a":1,"b":2}' >/dev/null 2>&1; then
    echo "PASS json: exact match"
    ((PASS++))
elif echo "$OUT" | jq -e '.a == 1 and .b == 2' >/dev/null 2>&1; then
    echo "PASS json: keys correct (whitespace differs)"
    ((PASS++))
else
    echo "FAIL json"
    echo "  got: $OUT"
    ((FAIL++))
fi

# 4. Single token: catches hallucination (1024 instead of 100)
echo "[4] Single token"
OUT=$(run "Return exactly the number 100. Nothing else, just the number.")
check_contains "single-token: has 100" "100" "$OUT"
check_not_contains "single-token: no hallucination" "1024" "$OUT"

# 5. Exact copy: catches whitespace collapse, tokenizer decode errors
echo "[5] Exact copy"
OUT=$(run 'Repeat exactly: alpha beta gamma delta')
check_contains "copy: preserved" "alpha beta gamma delta" "$OUT"

# 6. Isolation: prior prompt must not leak into next output
echo "[6] Isolation"
run "Remember this secret code: ZEBRA-9472" >/dev/null
OUT=$(run "What is 2+2? Answer with just the number.")
check_not_contains "isolation: no leak" "ZEBRA" "$OUT"
check_contains "isolation: correct answer" "4" "$OUT"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $((FAIL > 0 ? 1 : 0))
