#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Stage A: CPU/network preparation safe to run while GPU work is active.
set -euo pipefail

export PATH=/root/.cargo/bin:/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
export HIP_PATH=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIPFIRE_FP16=0

log() { printf '[hermes-setup] %s\n' "$*"; }

cd /root/hipfire
if [ ! -x target/release/examples/daemon ]; then
    log "building daemon..."
    cargo build --release --features deltanet --example daemon -p hipfire-runtime 2>&1 | tail -3
fi
if [ ! -x target/release/hipfire ]; then
    log "building native CLI..."
    cargo build --release -p hipfire-cli 2>&1 | tail -3
fi

if ! command -v hermes >/dev/null 2>&1; then
    log "installing hermes-agent..."
    curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
    # shellcheck disable=SC1091
    source "$HOME/.bashrc" 2>/dev/null || true
else
    log "hermes-agent already installed"
fi

log "STAGE A COMPLETE"
log "  native CLI:    /root/hipfire/target/release/hipfire"
log "  daemon binary: /root/hipfire/target/release/examples/daemon"
log "  hermes-agent:  $(command -v hermes || echo NOT_FOUND)"
log "Next: run scripts/hermes_validate_run.sh after the GPU queue drains."
