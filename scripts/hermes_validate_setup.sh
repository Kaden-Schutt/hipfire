#!/usr/bin/env bash
# hermes_validate_setup.sh — Stage A (CPU/net prep, parallel to GPU work):
# install Bun, build the hipfire daemon binary, install hermes-agent.
# Safe to run while other GPU workloads are active.
#
# Stage B (run separately, after current GPU chain drains):
#   scripts/hermes_validate_run.sh — quantize Carnice-9b, cal agentic sidecars,
#   start daemon + hipfire serve, configure + invoke hermes-agent.

set -euo pipefail

export PATH=/root/.cargo/bin:/root/.bun/bin:/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
export HIP_PATH=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIPFIRE_FP16=0

log() { printf '[hermes-setup] %s\n' "$*"; }

# ── 1. Bun ───────────────────────────────────────────────────────────
if ! command -v bun >/dev/null 2>&1; then
    log "installing Bun..."
    curl -fsSL https://bun.sh/install | bash
    # shellcheck disable=SC1091
    source "$HOME/.bashrc" 2>/dev/null || true
    export PATH="$HOME/.bun/bin:$PATH"
else
    log "Bun already installed: $(bun --version)"
fi

# ── 2. hipfire CLI deps (npm equivalent) ─────────────────────────────
cd /root/hipfire/cli
if [ ! -d node_modules ] || [ package.json -nt node_modules/.stamp ]; then
    log "installing CLI deps..."
    bun install
    touch node_modules/.stamp
fi
log "CLI deps OK"

# ── 3. Build daemon (if missing) ─────────────────────────────────────
cd /root/hipfire
if [ ! -x target/release/examples/daemon ]; then
    log "building daemon..."
    cargo build --release --features deltanet --example daemon 2>&1 | tail -3
fi
log "daemon binary OK: $(ls -la target/release/examples/daemon)"

# ── 4. Hermes-agent install ──────────────────────────────────────────
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
log "  Bun:           $(command -v bun || echo NOT_FOUND)"
log "  daemon binary: /root/hipfire/target/release/examples/daemon"
log "  hermes-agent:  $(command -v hermes || echo NOT_FOUND)"
log ""
log "Next — run scripts/hermes_validate_run.sh after current GPU chain drains"
log "       (it quantizes Carnice-9b, cals agentic sidecars, starts daemon,"
log "        configures hermes-agent, runs a small agent-task battery)."
log "────────────────────────────────────────────────────"
