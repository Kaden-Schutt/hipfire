#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Rust check =="
cargo check --workspace --examples

echo "== Rust no-GPU unit tests =="
cargo test -p rdna-compute --lib
cargo test -p hipfire-arch-qwen35 --lib moe_prefill
cargo test -p hipfire-config -p hipfire-registry -p hipfire-client -p hipfire-cli -p hipfire-tui

echo "== Python CPU tests =="
if python3 -c 'import pytest, numpy' 2>/dev/null; then
    python3 -m pytest tests scripts/test_astrea.py autoresearch/ar/tests
elif command -v uv >/dev/null 2>&1; then
    uv run --with pytest --with numpy python -m pytest \
        tests scripts/test_astrea.py autoresearch/ar/tests
else
    echo "no-gpu-ci: pytest/numpy missing and uv unavailable" >&2
    exit 1
fi
python3 -m unittest tools.redline.tests.test_product_bench tools.redline.tests.test_golden tools.redline.tests.test_serve_diff tools.redline.tests.test_lower tools.redline.tests.test_dispatch_profile
python3 scripts/test_install_revision.py
python3 scripts/test_uninstall.py

echo "== Env/docs drift check =="
python3 scripts/check-env-docs.py
