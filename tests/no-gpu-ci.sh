#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${HIPFIRE_PYTHON:-python3}"

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
cargo test -p hipfire-arch-api --lib
cargo test -p hipfire-arch-specs --lib
cargo test -p hipfire-arch-template-spec --lib
cargo test -p hipfire-arch-template --lib
cargo test -p hipfire-archs --lib
cargo test -p hipfire-arch-llama --lib caps

echo "== Tiny-fixture round-trip (CPU: emit → quantize, no GPU) =="
bash tests/fixture-roundtrip-nogpu.sh

echo "== Eval harness no-GPU smoke =="
cargo build -p hipfire-eval
HIPFIRE_EVAL_BIN="$ROOT/target/debug/hipfire-eval" bash tests/smoke/eval-harness-nogpu-smoke.sh

echo "== Python CPU tests =="
"$PYTHON" -m ruff check .
"$PYTHON" -m mypy tests scripts benchmarks tools --config-file pyproject.toml
"$PYTHON" -m pytest tests

echo "== Env-var docs freshness (docs/env-vars.md + env_docs.rs vs source) =="
cargo run -q -p hipfire-cli -- gen-env-docs --check

echo "== CLI docs freshness (docs/CLI.md + man/ vs clap definition) =="
cargo run -q -p hipfire-cli -- gen-docs --check

echo "== Config schema freshness (docs/config-schema.* vs schema registry) =="
cargo run -q -p hipfire-cli -- gen-config-schema --format json --output docs/config-schema.json --check
cargo run -q -p hipfire-cli -- gen-config-schema --format toml --output docs/config-schema.toml --check

echo "== Model-support freshness (model_support_generated.rs + MODEL-SUPPORT.md vs docs/model-support.toml) =="
cargo run -q -p hipfire-cli -- gen-model-support --check
cargo run -q -p hipfire-cli -- gen-config-schema --format markdown --output docs/config-schema.md --check

echo "== Artifact naming check =="
bash scripts/check-artifact-names.sh

echo "== Arch capability-layer purity (no format tokens in arch-api / *-spec) =="
bash scripts/check-arch-spec-purity.sh

echo "== Eval smoke script syntax =="
bash -n tests/smoke/eval-harness-nogpu-smoke.sh
bash -n tests/smoke/eval-harness-gpu-smoke.sh
bash -n tests/smoke/eval-harness-model-eval-smoke.sh
bash -n tests/tiny-affected-gate.sh
bash -n tests/tiny-quant-gate.sh
bash -n tests/tiny-state-gate.sh
bash -n tests/tiny-spec-gate.sh
bash -n tests/smoke/diffusion-sdapi-smoke.sh
bash -n tests/smoke/diffusion-tiny-sd-hfq-admission.sh

echo "== Legacy CLI checks =="
echo "Legacy CLI support has been removed; no script-runtime checks are run."
