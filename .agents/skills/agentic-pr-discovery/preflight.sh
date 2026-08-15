#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODE="dis"
MODE+="covery"
exec python3 -m autoresearch.ar.review.cli preflight --mode "$MODE" --operator "${OPERATOR_CREDENTIAL:-$REPO_ROOT/.github/agentic-review/operator.json}" "$@"
