#!/usr/bin/env bash
set -euo pipefail

exec ./tests/verify-bind-thread.sh "$@"
