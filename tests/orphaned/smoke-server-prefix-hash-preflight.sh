#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPFIRE_REQUIRE_PREFIX_PREFLIGHT=1 exec "$ROOT/tests/smoke-server-prefix-checkpoint-reuse.sh"
