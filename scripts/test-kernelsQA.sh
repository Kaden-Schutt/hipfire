#!/bin/bash
# QA kernel harness mirror.
# Builds and runs the hardened QA kernel example with explicit status handling.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ARCH_EXPECTED=${1:-}
BUILD_TIMEOUT=${BUILD_TIMEOUT:-600}
RUN_TIMEOUT=${RUN_TIMEOUT:-180}
FEATURES=${FEATURES:-deltanet}
EXAMPLE=${EXAMPLE:-test_kernelsQA}

log() {
    printf '%s\n' "$*"
}

run_with_status() {
    local label=$1
    local timeout_s=$2
    shift 2

    log "$label"
    set +e
    timeout "$timeout_s" "$@"
    local rc=$?
    set -e
    return "$rc"
}

log "=== hipfire kernel QA harness ==="
if [[ -n "$ARCH_EXPECTED" ]]; then
    log "Expected arch: $ARCH_EXPECTED"
fi

set +e
timeout "$BUILD_TIMEOUT" cargo build --release --features "$FEATURES" --example "$EXAMPLE" -p hipfire-runtime
BUILD_RC=$?
set -e
if [[ $BUILD_RC -ne 0 ]]; then
    log "BUILD FAIL rc=$BUILD_RC"
    exit "$BUILD_RC"
fi

ARGS=()
if [[ -n "$ARCH_EXPECTED" ]]; then
    ARGS+=("--expected-arch" "$ARCH_EXPECTED")
fi

set +e
timeout "$RUN_TIMEOUT" ./target/release/examples/$EXAMPLE "${ARGS[@]}"
RUN_RC=$?
set -e

case "$RUN_RC" in
    0)
        log "=== QA KERNEL TESTS PASSED ==="
        ;;
    124)
        log "=== QA KERNEL TESTS TIMED OUT (${RUN_TIMEOUT}s) ==="
        exit "$RUN_RC"
        ;;
    *)
        log "=== QA KERNEL TESTS FAILED (rc=$RUN_RC) ==="
        exit "$RUN_RC"
        ;;
esac
