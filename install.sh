#!/usr/bin/env bash
# Thin source-checkout wrapper around scripts/install.sh. Keep install logic there.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HIPFIRE_DIR="${HIPFIRE_DIR:-$HOME/.hipfire}"
export CARGO_INSTALL_OPTS="${CARGO_INSTALL_OPTS:---force}"

if [ "$#" -gt 0 ]; then
    export CARGO_INSTALL_OPTS="$CARGO_INSTALL_OPTS $*"
fi

exec "$ROOT/scripts/install.sh"
