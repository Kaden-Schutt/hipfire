#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Auto-detect repository from git remote if not provided
if [[ "$*" != *"--repository"* ]] && [[ "$*" != *"--pr"* ]]; then
  echo "Usage: run-inspector.sh --pr PR_NUM [--repository OWNER/REPO] [--capsule FILE] [--proposal FILE]"
  echo ""
  echo "Examples:"
  echo "  # Build capsule only:"
  echo "  run-inspector.sh --pr 123 --capsule capsule.json"
  echo ""
  echo "  # Build + infer + save proposal:"
  echo "  REVIEW_API_KEY=sk-... run-inspector.sh --pr 123 --proposal proposal.json --provider review-adapter"
  exit 1
fi
exec python3 -m autoresearch.ar.review.cli inspect "$@"
