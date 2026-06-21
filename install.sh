#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# hipfire installer - builds from source and installs to ${HIPFIRE_DIR:-~/.hipfire}.
# Usage (from source checkout): ./install.sh
# Usage (remote):               curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/install.sh | bash
set -euo pipefail

HIPFIRE_DIR="${HIPFIRE_DIR:-$HOME/.hipfire}"
BIN_DIR="$HIPFIRE_DIR/bin"
MODELS_DIR="$HIPFIRE_DIR/models"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"
SRC_DIR="$HIPFIRE_DIR/src"
GITHUB_REPO="Kaden-Schutt/hipfire"
GITHUB_BRANCH="master"
CARGO_INSTALL_OPTS="${CARGO_INSTALL_OPTS:---force}"
if [ "$#" -gt 0 ]; then
    CARGO_INSTALL_OPTS="$CARGO_INSTALL_OPTS $*"
fi
INSTALL_OPTS=()
if [ -n "${CARGO_INSTALL_OPTS:-}" ]; then
    # shellcheck disable=SC2206
    INSTALL_OPTS=(${CARGO_INSTALL_OPTS})
fi

echo "=== hipfire installer ==="
echo ""

# ─── Interactive prompts (safe for curl|bash) ────────────
ask() {
    local prompt="$1" default="$2"
    if printf "%s" "$prompt" >/dev/tty 2>/dev/null; then
        local reply
        read -r reply </dev/tty 2>/dev/null || reply="$default"
        echo "${reply:-$default}"
    else
        echo "$default"
    fi
}

# Pick the right HIP runtime package name for dnf-based distros.
dnf_hip_pkg() {
    local id="" id_like=""
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        id="${ID:-}"
        id_like="${ID_LIKE:-}"
    fi
    case "$id" in
        fedora) echo "rocm-hip" ;;
        rhel|rocky|almalinux|centos|ol) echo "rocm-hip-runtime" ;;
        *)
            case "$id_like" in
                *fedora*) echo "rocm-hip" ;;
                *rhel*|*centos*) echo "rocm-hip-runtime" ;;
                *) echo "rocm-hip-runtime" ;;
            esac
            ;;
    esac
}

# ─── OS Detection ────────────────────────────────────────
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$OS" in
    linux) ;;
    darwin)
        echo "macOS is not supported (AMD GPUs only). Exiting."
        exit 1
        ;;
    mingw*|msys*|cygwin*)
        echo "Windows: use WSL2 or the PowerShell installer."
        exit 1
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac
echo "OS: $OS ($ARCH)"

# ─── GPU Detection ───────────────────────────────────────
echo ""
echo "Checking for AMD GPU..."
if [ ! -e /dev/kfd ]; then
    echo "ERROR: /dev/kfd not found. No AMD GPU detected."
    exit 1
fi
echo "  /dev/kfd: found ✓"

GPU_ARCH="unknown"
for node_props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ -f "$node_props" ] || continue
    ver=$(grep -oP 'gfx_target_version\s+\K\d+' "$node_props" 2>/dev/null || true)
    case "$ver" in
        90006)          GPU_ARCH="gfx906";  break ;;
        90008)          GPU_ARCH="gfx908";  break ;;
        100100)         GPU_ARCH="gfx1010"; break ;;
        100300|100302)  GPU_ARCH="gfx1030"; break ;;
        110000|110001)  GPU_ARCH="gfx1100"; break ;;
        110003)         GPU_ARCH="gfx1103"; break ;;
        110501)         GPU_ARCH="gfx1151"; break ;;
        120000)         GPU_ARCH="gfx1200"; break ;;
        120001)         GPU_ARCH="gfx1201"; break ;;
    esac
done

if [ "$GPU_ARCH" = "unknown" ] && command -v rocm-smi &>/dev/null; then
    GPU_ARCH=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
fi

if [ "$GPU_ARCH" = "unknown" ]; then
    echo "  WARNING: Could not detect GPU architecture."
    echo "  Supported: gfx906 gfx908 gfx1010 gfx1030 gfx1100 gfx1103 gfx1151 gfx1200 gfx1201"
    GPU_ARCH=$(ask "  Enter your GPU arch [or Enter to skip]: " "unknown")
fi
echo "  GPU arch: $GPU_ARCH"

# ─── HIP Runtime ─────────────────────────────────────────
echo ""
echo "Checking HIP runtime..."
HIP_FOUND=false
for dir in /opt/rocm/lib /opt/rocm/lib64 \
           /usr/lib /usr/lib64 \
           /usr/lib/x86_64-linux-gnu /usr/lib64/rocm; do
    for suffix in "" ".6" ".7" ".8"; do
        if [ -f "$dir/libamdhip64.so${suffix}" ]; then
            echo "  libamdhip64.so: found at $dir/libamdhip64.so${suffix} ✓"
            HIP_FOUND=true
            break 2
        fi
    done
done

if ! $HIP_FOUND; then
    ldconfig_hit=$(ldconfig -p 2>/dev/null | grep -m1 -E '\blibamdhip64\.so(\.[0-9]+)?\b' | awk '{print $NF}' || true)
    if [ -n "$ldconfig_hit" ] && [ -f "$ldconfig_hit" ]; then
        echo "  libamdhip64.so: found via ldconfig at $ldconfig_hit ✓"
        HIP_FOUND=true
    fi
fi

if ! $HIP_FOUND; then
    echo "  libamdhip64.so: NOT FOUND"
    PKG_CMD=""
    if command -v apt &>/dev/null; then
        PKG_CMD="sudo apt install -y rocm-hip-runtime"
    elif command -v dnf &>/dev/null; then
        PKG_CMD="sudo dnf install -y $(dnf_hip_pkg)"
    elif command -v pacman &>/dev/null; then
        PKG_CMD="sudo pacman -S --noconfirm rocm-hip-runtime"
    elif command -v zypper &>/dev/null; then
        PKG_CMD="sudo zypper install -y rocm-hip-runtime"
    fi

    if [ -n "$PKG_CMD" ]; then
        reply=$(ask "  Install HIP runtime now? ($PKG_CMD) [Y/n] " "Y")
        if [ "$reply" != "n" ] && [ "$reply" != "N" ]; then
            eval "$PKG_CMD" || {
                echo "  HIP runtime install failed."
                reply=$(ask "  Continue without HIP runtime? [y/N] " "N")
                [ "$reply" = "y" ] || [ "$reply" = "Y" ] || exit 1
            }
        fi
    else
        echo "  Install libamdhip64.so manually: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        reply=$(ask "  Continue without HIP runtime? [y/N] " "N")
        [ "$reply" = "y" ] || [ "$reply" = "Y" ] || exit 1
    fi
fi

# ─── Rust ────────────────────────────────────────────────
echo ""
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
echo "Rust: $(cargo --version) ✓"

# ─── Source checkout ─────────────────────────────────────
echo ""
REPO_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd 2>/dev/null)" || true
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/Cargo.toml" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR" && pwd)"
    echo "Source: $REPO_DIR (local checkout)"
else
    echo "Source: cloning from GitHub..."
    if [ ! -d "$SRC_DIR/.git" ]; then
        git clone --depth 1 --branch "$GITHUB_BRANCH" \
            "https://github.com/$GITHUB_REPO.git" "$SRC_DIR"
    else
        echo "  Existing clone at $SRC_DIR — updating..."
        if [ -n "$(git -C "$SRC_DIR" status --porcelain 2>/dev/null)" ]; then
            stamp=$(date -u +%Y-%m-%dT%H-%M-%SZ)
            git -C "$SRC_DIR" stash push --include-untracked -m "hipfire-install-${stamp}" >/dev/null 2>&1 || true
        fi
        git -C "$SRC_DIR" fetch origin "$GITHUB_BRANCH" --depth 1 2>/dev/null && \
        git -C "$SRC_DIR" reset --hard "origin/$GITHUB_BRANCH" 2>/dev/null || \
            echo "  Update failed (non-fatal). Using existing checkout."
    fi
    REPO_DIR="$SRC_DIR"
fi

# ─── Build & install ─────────────────────────────────────
mkdir -p "$BIN_DIR" "$MODELS_DIR"

echo ""
echo "Building and installing hipfire (release build — this may take several minutes)..."
cd "$REPO_DIR"

# hipfire-daemon: the GPU inference worker
cargo install "${INSTALL_OPTS[@]}" --path crates/hipfire-daemon --root "$HIPFIRE_DIR"

# Browser UIs (Leptos/WASM) — optional, embedded into `hipfire` when the
# wasm toolchain is present. Best-effort: a missing/broken trunk just falls
# back to the lightweight built-in routes.
UI_FEATURES=()
if command -v trunk &>/dev/null && \
   rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
    echo ""
    echo "Building admin console (Leptos/WASM)..."
    if ( cd "$REPO_DIR/crates/hipfire-admin-ui" && env -u NO_COLOR trunk build --release ); then
        UI_FEATURES+=(--features admin-ui-embed)
        echo "  Admin console built ✓"
    else
        echo "  Admin console build failed — installing without the embedded UI."
    fi

    echo ""
    echo "Building browser chat UI (Leptos/WASM)..."
    if ( cd "$REPO_DIR/crates/hipfire-chat-ui" && env -u NO_COLOR trunk build --release ); then
        UI_FEATURES+=(--features chat-ui-embed)
        echo "  Browser chat UI built ✓"
    else
        echo "  Browser chat UI build failed — installing without the embedded UI."
    fi
else
    echo ""
    echo "Browser UIs: trunk/wasm32 not found — skipping embedded UIs."
    echo "  Enable with: rustup target add wasm32-unknown-unknown && cargo install trunk"
fi

# hipfire: the CLI (serve / run / list)
cargo install "${INSTALL_OPTS[@]}" "${UI_FEATURES[@]}" --path crates/hipfire-cli --root "$HIPFIRE_DIR"

# hipfire-tui: optional terminal operator UI
cargo install "${INSTALL_OPTS[@]}" --path crates/hipfire-tui --root "$HIPFIRE_DIR"

# Auxiliary eval/runtime tools
cargo install --path crates/hipfire-eval \
    "${INSTALL_OPTS[@]}" \
    --root "$HIPFIRE_DIR"
cargo install --path crates/hipfire-runtime \
    --bin hipfire-host-profile \
    "${INSTALL_OPTS[@]}" \
    --root "$HIPFIRE_DIR"

echo ""
echo "Installed to $BIN_DIR:"
ls -1 "$BIN_DIR"/

# ─── Symlinks in ~/.local/bin ────────────────────────────
echo ""
echo "Creating symlinks in $LOCAL_BIN..."
mkdir -p "$LOCAL_BIN"
for bin in hipfire hipfire-daemon hipfire-tui hipfire-eval hipfire-host-profile; do
    if [ -f "$BIN_DIR/$bin" ]; then
        ln -sf "$BIN_DIR/$bin" "$LOCAL_BIN/$bin"
        echo "  $LOCAL_BIN/$bin -> $BIN_DIR/$bin ✓"
    fi
done

# ─── Kernels ─────────────────────────────────────────────
echo ""
if [ "$GPU_ARCH" != "unknown" ]; then
    KERNEL_DEST="$HIPFIRE_DIR/kernels/compiled/$GPU_ARCH"
    mkdir -p "$KERNEL_DEST"
    if [ -d "$REPO_DIR/kernels/compiled/$GPU_ARCH" ]; then
        cp "$REPO_DIR/kernels/compiled/$GPU_ARCH"/*.hsaco "$KERNEL_DEST/" 2>/dev/null || true
        cp "$REPO_DIR/kernels/compiled/$GPU_ARCH"/*.hash  "$KERNEL_DEST/" 2>/dev/null || true
        count=$(ls "$KERNEL_DEST"/*.hsaco 2>/dev/null | wc -l)
        echo "Pre-compiled kernels for $GPU_ARCH: $count copied to $KERNEL_DEST/ ✓"
    else
        echo "No pre-compiled kernels for $GPU_ARCH in repo — will JIT on first use."
    fi
fi

if [ -x "$BIN_DIR/hipfire-daemon" ]; then
    echo ""
    echo "Pre-compiling GPU kernels..."
    "$BIN_DIR/hipfire-daemon" --precompile && echo "  Pre-compile complete ✓" || \
        echo "  Pre-compile finished with warnings — missing kernels will JIT on first use."
fi

# ─── Config ──────────────────────────────────────────────
CONFIG="$HIPFIRE_DIR/config.json"
if [ ! -f "$CONFIG" ]; then
    cat > "$CONFIG" << CONF
{
  "temperature": 0.3,
  "top_p": 0.8,
  "max_tokens": 512,
  "gpu_arch": "$GPU_ARCH"
}
CONF
    echo ""
    echo "Created default config at $CONFIG"
fi

# ─── PATH check ──────────────────────────────────────────
echo ""
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    SHELL_RC=""
    case "$(basename "${SHELL:-bash}")" in
        bash) SHELL_RC="$HOME/.bashrc" ;;
        zsh)  SHELL_RC="$HOME/.zshrc"  ;;
    esac
    PATH_LINE="export PATH=\"\$HOME/.local/bin:\$PATH\""
    if [ -n "$SHELL_RC" ] && [ -f "$SHELL_RC" ]; then
        if ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
            reply=$(ask "Add ~/.local/bin to PATH in $SHELL_RC? [Y/n] " "Y")
            if [ "$reply" != "n" ] && [ "$reply" != "N" ]; then
                printf '\n# hipfire\n%s\n' "$PATH_LINE" >> "$SHELL_RC"
                echo "  Added to $SHELL_RC ✓"
            else
                echo "  Add manually: $PATH_LINE"
            fi
        fi
    else
        echo "Add to your shell profile: $PATH_LINE"
    fi
fi

echo ""
echo "=== hipfire installed ==="
echo ""
echo "Quick start:"
echo "  hipfire list                        # see local models"
echo "  hipfire run <model> \"Hello\"         # generate text"
echo "  hipfire serve                       # start OpenAI-compatible API"
echo "  hipfire-tui                         # open terminal operator UI"
echo ""
echo "To reinstall (force rebuild): re-run with CARGO_INSTALL_OPTS=--force ./install.sh"
echo "Models go in ~/.hipfire/models/"
echo ""
