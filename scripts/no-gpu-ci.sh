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

echo "== Docs reliability unit tests =="
# Focused unittest for scripts/check-docs-reliability.py (no pytest/GPU).
python3 -m unittest tests.test_docs_reliability

echo "== Docs reliability checker =="
# Require an explicit comparison base. In CI the workflow must pass
# DOCS_DIFF_BASE (PR base SHA or push-before SHA). Locally, default to HEAD
# only for same-tree structural checking (target == base == working tree tip).
if [[ -z "${DOCS_DIFF_BASE:-}" ]]; then
    if [[ -n "${CI:-}${GITHUB_ACTIONS:-}" ]]; then
        echo "no-gpu-ci: DOCS_DIFF_BASE is required in CI (PR base SHA or push-before SHA)" >&2
        exit 1
    fi
    DOCS_DIFF_BASE=HEAD
    echo "no-gpu-ci: DOCS_DIFF_BASE unset; defaulting to HEAD (local same-tree structural check)"
fi
if ! git rev-parse --verify --quiet "${DOCS_DIFF_BASE}^{commit}" >/dev/null; then
    echo "no-gpu-ci: DOCS_DIFF_BASE='${DOCS_DIFF_BASE}' is not a resolvable commit" >&2
    exit 1
fi
python3 scripts/check-docs-reliability.py --target-ref HEAD --base-ref "$DOCS_DIFF_BASE"

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
