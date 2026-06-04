#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Rust check =="
cargo check --workspace --examples

echo "== Eval harness check =="
cargo check -p hipfire-runtime --bin hipfire-eval

echo "== Rust no-GPU unit tests =="
cargo test -p rdna-compute --lib
cargo test -p hipfire-arch-qwen35 --lib moe_prefill
cargo test -p hipfire-runtime eval_harness --lib
cargo test -p hipfire-quantize xxh64_provenance_tests

echo "== Eval harness no-GPU smoke =="
cargo build -p hipfire-runtime --bin hipfire-eval
HIPFIRE_EVAL_BIN="$ROOT/target/debug/hipfire-eval" bash scripts/smoke/eval-harness-nogpu-smoke.sh

echo "== Python CPU tests =="
python3 -m pytest tests scripts/test_astrea.py

echo "== Env/docs drift check =="
python3 scripts/check-env-docs.py

echo "== Eval smoke script syntax =="
bash -n scripts/smoke/eval-harness-nogpu-smoke.sh
bash -n scripts/smoke/eval-harness-gpu-smoke.sh
bash -n scripts/smoke/eval-harness-model-eval-smoke.sh

if command -v bun >/dev/null 2>&1; then
    echo "== Bun tests/typecheck =="
    (
        cd cli
        bun test
        bun run typecheck
    )
else
    echo "no-gpu-ci: bun not found; skipping Bun checks" >&2
fi
