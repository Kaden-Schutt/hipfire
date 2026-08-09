#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
#
# Profile a daemon launched indirectly by `hipfire serve`.
#
# `serve_harness.py` accepts a daemon path through HIPFIRE_DAEMON_BIN, but
# rocprof must launch the GPU process itself; wrapping the Python harness does
# not instrument the child daemon and ptrace attach may be disabled. Point
# HIPFIRE_DAEMON_BIN at this script and provide:
#
#   HIPFIRE_ROCPROF_DAEMON_TARGET=/absolute/path/to/examples/daemon
#   HIPFIRE_ROCPROF_OUTPUT_DIR=/absolute/path/to/evidence/rocprof
#
# All arguments supplied by `hipfire serve` are forwarded byte-for-byte.

set -euo pipefail

: "${HIPFIRE_ROCPROF_DAEMON_TARGET:?set the real daemon path}"
: "${HIPFIRE_ROCPROF_OUTPUT_DIR:?set the rocprof output directory}"

ROCPROF_BIN="${HIPFIRE_ROCPROF_BIN:-rocprofv3}"
mkdir -p "${HIPFIRE_ROCPROF_OUTPUT_DIR}"

exec "${ROCPROF_BIN}" \
    --kernel-trace \
    --rccl-trace \
    --stats \
    --output-format csv \
    -d "${HIPFIRE_ROCPROF_OUTPUT_DIR}" \
    -o daemon \
    -- "${HIPFIRE_ROCPROF_DAEMON_TARGET}" "$@"
