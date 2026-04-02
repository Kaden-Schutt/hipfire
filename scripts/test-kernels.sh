#!/bin/bash
# Comprehensive kernel test harness. Validates every dispatch path
# with synthetic data — no model loading required.
# Usage: ./scripts/test-kernels.sh [arch]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

ARCH=${1:-gfx1010}
echo "=== hipfire kernel test harness (${ARCH}) ==="

# Build the test binary
cargo build --release --features deltanet --example test_kernels -p engine 2>&1 | tail -2

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
