#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Björn Bösel
# hipfire — see LICENSE and NOTICE in the project root.

# Source this file to set LD_LIBRARY_PATH for ROCm/HIP.
# Tries versioned /opt/rocm installs, then /opt/rocm.

# Try specific versioned paths first (newest to oldest)
for v in 7.12 7.11 7.2 7.1 7.0; do
    if [ -d "/opt/rocm-$v/lib" ]; then
        export LD_LIBRARY_PATH="/opt/rocm-$v/lib:${LD_LIBRARY_PATH:-}"
        export PATH="/opt/rocm-$v/bin:${PATH:-}"
        return 0 2>/dev/null || true
    fi
done

if [ -d "/opt/rocm/lib" ]; then
    export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
    export PATH="/opt/rocm/bin:${PATH:-}"
    return 0 2>/dev/null || true
fi

echo "WARNING: libamdhip64.so not found. ROCm/HIP may not be installed." >&2
