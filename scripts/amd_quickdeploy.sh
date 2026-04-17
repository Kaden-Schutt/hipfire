#!/usr/bin/env bash
# amd_quickdeploy.sh — one-shot bring-up for a fresh rented AMD GPU box
# (MI300X/MI250X/MI210/Radeon VII Pro/V620/etc). Idempotent: safe to re-run.
#
# What it does:
#   1. Verifies ROCm is installed and parses its minor version.
#   2. Installs apt prereqs (python3-venv, build tools).
#   3. Installs rustup if missing; guarantees a stable toolchain.
#   4. Creates /root/pytorch_env and installs the torch wheel whose ROCm
#      minor version matches the installed stack. Falls back to the closest
#      supported channel if no exact match exists.
#   5. Installs the HF training stack (transformers, safetensors, datasets,
#      accelerate, numpy).
#   6. Optionally clones/updates hipfire to /root/hipfire and builds the
#      release binaries under the `deltanet` feature.
#
# What it does NOT do:
#   - Transfer models or calibration corpora (multi-GB, pick per job).
#   - Start any long-running workload (caller kicks those explicitly).
#
# Usage:
#   scp scripts/amd_quickdeploy.sh <host>:/root/
#   ssh <host> 'bash /root/amd_quickdeploy.sh'            # default targets
#   ssh <host> 'bash /root/amd_quickdeploy.sh --skip-build'
#   ssh <host> 'bash /root/amd_quickdeploy.sh --repo=git@github.com:user/hipfire.git'

set -euo pipefail

REPO_URL="${REPO_URL:-}"      # git URL — if empty, assumes repo already present
REPO_DIR="${REPO_DIR:-/root/hipfire}"
VENV_DIR="${VENV_DIR:-/root/pytorch_env}"
SKIP_BUILD=0
SKIP_HF=0
SKIP_TORCH=0

for arg in "$@"; do
    case "$arg" in
        --skip-build)  SKIP_BUILD=1 ;;
        --skip-hf)     SKIP_HF=1 ;;
        --skip-torch)  SKIP_TORCH=1 ;;
        --repo=*)      REPO_URL="${arg#--repo=}" ;;
        --help|-h)     sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

log() { printf '[quickdeploy] %s\n' "$*"; }

# ── 1. ROCm detection ─────────────────────────────────────────────────
if ! [ -x /opt/rocm/bin/rocminfo ] && ! command -v rocm-smi >/dev/null 2>&1; then
    log "ERROR: no ROCm install found at /opt/rocm. Install ROCm first."
    log "See https://rocm.docs.amd.com/projects/install-on-linux/ for the official installer."
    exit 1
fi

ROCM_VERSION="$(dpkg-query -W -f='${Version}' rocm-core 2>/dev/null | head -c5 || true)"
if [ -z "$ROCM_VERSION" ]; then
    ROCM_VERSION="$(/opt/rocm/bin/rocminfo 2>/dev/null | awk -F: '/^Runtime Version/ {gsub(/ /,""); print $2; exit}')"
fi
ROCM_MAJOR="${ROCM_VERSION%%.*}"
ROCM_MINOR="$(echo "$ROCM_VERSION" | cut -d. -f2)"
ROCM_MM="${ROCM_MAJOR}.${ROCM_MINOR}"
log "ROCm detected: $ROCM_VERSION (using channel rocm${ROCM_MM})"

GPU_NAME="$(/opt/rocm/bin/rocminfo 2>/dev/null | awk -F: '/Marketing Name/ {print $2; exit}' | xargs || echo "unknown")"
log "GPU: $GPU_NAME"

export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
export HIP_PATH=/opt/rocm
export ROCM_PATH=/opt/rocm

# ── 2. apt prereqs ────────────────────────────────────────────────────
log "installing apt prereqs..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -qq -y python3-venv build-essential pkg-config git rsync tmux htop

# ── 3. rustup ─────────────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1; then
    log "installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env" 2>/dev/null || true
log "cargo: $(cargo --version 2>&1 | head -1)"

# ── 4. PyTorch (ROCm-matched wheel) ──────────────────────────────────
if [ "$SKIP_TORCH" -eq 0 ]; then
    if [ ! -d "$VENV_DIR" ]; then
        log "creating venv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --quiet --upgrade pip

    # Try the exact channel first; fall back to the closest older one.
    # PyTorch releases ROCm wheels at rocm<MAJOR>.<MINOR>; when there is no
    # exact match, use the newest available channel ≤ installed ROCm.
    try_channels=(
        "rocm${ROCM_MM}"
        "rocm${ROCM_MAJOR}.$((ROCM_MINOR - 1))"
        "rocm${ROCM_MAJOR}.$((ROCM_MINOR - 2))"
        "rocm${ROCM_MAJOR}.0"
    )
    installed=0
    for ch in "${try_channels[@]}"; do
        log "trying torch wheel channel: $ch"
        if pip install --quiet torch --index-url "https://download.pytorch.org/whl/$ch" >/tmp/pip_torch.log 2>&1; then
            installed=1
            log "installed torch from $ch"
            break
        fi
        log "  (channel $ch not available; trying next)"
    done
    if [ "$installed" -eq 0 ]; then
        log "ERROR: no ROCm-matching torch wheel found across $(echo "${try_channels[@]}")"
        log "See https://pytorch.org/get-started/locally/ for current options."
        tail /tmp/pip_torch.log
        exit 3
    fi

    python3 -c '
import torch
print(f"torch={torch.__version__} hip={torch.version.hip}")
print(f"device_count={torch.cuda.device_count()} device0={torch.cuda.get_device_name(0) if torch.cuda.device_count() else None}")
t = torch.randn(1024, 1024, device="cuda")
print(f"sanity gemm ok, sum={(t @ t).sum().item():.3f}")
'
fi

# ── 5. HF stack ───────────────────────────────────────────────────────
if [ "$SKIP_HF" -eq 0 ]; then
    # shellcheck disable=SC1091
    [ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"
    log "installing HF training stack..."
    pip install --quiet --upgrade \
        numpy transformers safetensors datasets accelerate \
        huggingface-hub tokenizers tqdm
    python3 -c 'import transformers, datasets, accelerate, safetensors; print(f"hf stack ok: transformers={transformers.__version__}")'
fi

# ── 6. hipfire build ──────────────────────────────────────────────────
if [ "$SKIP_BUILD" -eq 0 ]; then
    if [ -n "$REPO_URL" ] && [ ! -d "$REPO_DIR/.git" ]; then
        log "cloning $REPO_URL → $REPO_DIR"
        git clone "$REPO_URL" "$REPO_DIR"
    elif [ -d "$REPO_DIR/.git" ]; then
        log "hipfire repo present at $REPO_DIR"
    else
        log "no REPO_URL given and $REPO_DIR has no .git — skipping build"
        log "either pass --repo=… or rsync the repo in manually, then re-run"
        exit 0
    fi
    cd "$REPO_DIR"

    # Nuke stale JIT kernel cache — the handoff note in
    # memory/feedback_dflash_nondeterminism.md explains this is mandatory
    # on kernel-touching rebuilds.
    rm -rf .hipfire_kernels "$HOME/.hipfire/bin/kernels/compiled" 2>/dev/null || true

    export HIPFIRE_FP16=0
    log "building hipfire (release, deltanet)..."
    cargo build --release --features deltanet --examples 2>&1 | tail -5
    log "build done — key binaries under $REPO_DIR/target/release/examples/"
fi

log "────────────────────────────────────────────────────"
log "QUICKDEPLOY COMPLETE"
log "ROCm:    $ROCM_VERSION"
log "GPU:     $GPU_NAME"
log "venv:    $VENV_DIR (activate with: source $VENV_DIR/bin/activate)"
if [ "$SKIP_BUILD" -eq 0 ] && [ -d "$REPO_DIR" ]; then
    log "hipfire: $REPO_DIR"
fi
log ""
log "Next steps (machine-specific):"
log "  - rsync your models and calibration corpora to this box."
log "  - For sidecar cals:"
log "      ./target/release/examples/triattn_validate <model.mq4> \\"
log "        --corpus <wikitext_calib.txt> --max-tokens 1000000 \\"
log "        --sidecar <model.mq4.triattn.bin>"
log "────────────────────────────────────────────────────"
