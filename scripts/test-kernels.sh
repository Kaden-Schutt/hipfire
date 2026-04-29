#!/bin/bash
# Comprehensive kernel test harness. Validates every dispatch path
# with synthetic data — no model loading required.
# Usage: ./scripts/test-kernels.sh [arch]   # arch defaults to detected
set -euo pipefail

SCRIPT_DIR_BIN="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect arch when no arg passed. Fall back to gfx1100 only if
# detection truly fails (no rocminfo / amdgpu-arch / offload-arch).
. "$SCRIPT_DIR_BIN/_detect-gpu.sh"
ARCH="${1:-${HIPFIRE_DETECTED_ARCH:-gfx1100}}"
echo "=== hipfire kernel test harness (${ARCH}) ==="

# Build the test binary
BUILD_LOG=$(mktemp)
if cargo build --release --features deltanet --example test_kernels -p engine >"$BUILD_LOG" 2>&1; then
    tail -2 "$BUILD_LOG"
else
    tail -40 "$BUILD_LOG"
    rm -f "$BUILD_LOG"
    exit 1
fi
rm -f "$BUILD_LOG"

echo ""
echo "Running kernel tests..."
timeout 120 ./target/release/examples/test_kernels 2>&1
EXIT=$?

if [ $EXIT -eq 0 ]; then
    echo ""
    echo "=== ALL TESTS PASSED ==="
else
    echo ""
    echo "=== TESTS FAILED (exit $EXIT) ==="
    exit $EXIT
fi
