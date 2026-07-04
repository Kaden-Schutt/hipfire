#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

exec python3 scripts/adhoc/kernel_atlas.py render-fit "$@"
