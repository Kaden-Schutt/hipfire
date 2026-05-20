#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Generate .hash sidecar files for pre-compiled kernel blobs.
# Uses gen_kernel_hashes (Rust) to compute hashes matching compiler.rs.
# Run this after compile-kernels.sh to make pre-compiled blobs trusted.
#
# Usage: ./scripts/write-kernel-hashes.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

cargo run --release -p rdna-compute --example gen_kernel_hashes
