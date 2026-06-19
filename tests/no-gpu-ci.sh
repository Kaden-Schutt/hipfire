#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Rust check =="
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-D warnings" cargo check --workspace --examples

echo "== Eval harness check =="
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-D warnings" cargo check -p hipfire-eval

echo "== Rust no-GPU unit tests =="
cargo test -p rdna-compute --lib
cargo test -p hipfire-arch-qwen35 --lib moe_prefill
cargo test -p hipfire-eval --lib
cargo test -p hipfire-quantize xxh64_provenance_tests
cargo test -p hipfire-quantize fixture

echo "== Tiny-fixture round-trip (CPU: emit → quantize, no GPU) =="
bash tests/fixture-roundtrip-nogpu.sh

echo "== Eval harness no-GPU smoke =="
cargo build -p hipfire-eval
HIPFIRE_EVAL_BIN="$ROOT/target/debug/hipfire-eval" bash tests/smoke/eval-harness-nogpu-smoke.sh

echo "== Python CPU tests =="
python3 -m ruff check .
python3 -m mypy tests scripts benchmarks tools --config-file pyproject.toml
python3 -m pytest tests

echo "== Env/docs drift check =="
python3 scripts/check-env-docs.py

echo "== CLI docs freshness (docs/CLI.md + man/ vs clap definition) =="
cargo run -q -p hipfire-cli -- gen-docs --check

echo "== Artifact naming check =="
bash scripts/check-artifact-names.sh

echo "== Eval smoke script syntax =="
bash -n tests/smoke/eval-harness-nogpu-smoke.sh
bash -n tests/smoke/eval-harness-gpu-smoke.sh
bash -n tests/smoke/eval-harness-model-eval-smoke.sh

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
