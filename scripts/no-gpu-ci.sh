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
python3 -m pytest tests scripts/test_astrea.py

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
