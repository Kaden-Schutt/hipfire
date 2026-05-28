#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [ ! -d .githooks ]; then
    echo "install-hooks: .githooks directory is missing" >&2
    exit 1
fi

git config core.hooksPath .githooks

if [ -f .githooks/pre-commit ]; then
    chmod +x .githooks/pre-commit
fi

echo "install-hooks: core.hooksPath=$(git config core.hooksPath)"
