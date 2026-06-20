#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# hermes_validate_setup.sh — Stage A (CPU/net prep, parallel to GPU work):
# build hipfire binaries, install hermes-agent.
# Safe to run while other GPU workloads are active.
#
# Stage B (run separately, after current GPU chain drains):
#   scripts/hermes_validate_run.sh — quantize Carnice-9b, cal agentic sidecars,
#   start daemon + hipfire serve, configure + invoke hermes-agent.

set -euo pipefail

export PATH=/root/.cargo/bin:/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
export HIP_PATH=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIPFIRE_FP16=0

log() { printf '[hermes-setup] %s\n' "$*"; }

# ── 1. Build hipfire binaries (if missing) ───────────────────────────
cd /root/hipfire
if [ ! -x target/release/hipfire-daemon ]; then
    log "building daemon..."
    cargo build --release --features deltanet -p hipfire-daemon --bin hipfire-daemon 2>&1 | tail -3
fi
if [ ! -x target/release/hipfire ]; then
    log "building Rust CLI..."
    cargo build --release -p hipfire-cli 2>&1 | tail -3
fi
log "daemon binary OK: $(ls -la target/release/hipfire-daemon)"
log "CLI binary OK: $(ls -la target/release/hipfire)"

# ── 2. Hermes-agent install ──────────────────────────────────────────
if ! command -v hermes >/dev/null 2>&1; then
    log "installing hermes-agent..."
    curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
    # shellcheck disable=SC1091
    source "$HOME/.bashrc" 2>/dev/null || true
else
    log "hermes-agent already installed"
fi

log "────────────────────────────────────────────────────"
log "STAGE A COMPLETE"
log "  daemon binary: /root/hipfire/target/release/hipfire-daemon"
log "  CLI binary:    /root/hipfire/target/release/hipfire"
log "  hermes-agent:  $(command -v hermes || echo NOT_FOUND)"
log ""
log "Next — run scripts/hermes_validate_run.sh after current GPU chain drains"
log "       (it quantizes Carnice-9b, cals agentic sidecars, starts daemon,"
log "        configures hermes-agent, runs a small agent-task battery)."
log "────────────────────────────────────────────────────"
