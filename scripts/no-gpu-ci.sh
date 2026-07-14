#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Rust check =="
cargo check --workspace --examples

echo "== Rust no-GPU unit tests =="
cargo test -p rdna-compute --lib
cargo test -p hipfire-arch-qwen35 --lib moe_prefill

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

echo "== Env/docs drift check =="
python3 scripts/check-env-docs.py

if command -v bun >/dev/null 2>&1; then
    echo "== Bun tests/typecheck =="
    (
        cd cli
        bun install --frozen-lockfile
        bun test
        bun run typecheck
    )
else
    echo "no-gpu-ci: bun not found; skipping Bun checks" >&2
fi
