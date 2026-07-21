#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# hipfire installer — detects GPU, installs deps, downloads binary + kernels.
# Usage: curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash
set -euo pipefail

HIPFIRE_DIR="$HOME/.hipfire"
BIN_DIR="$HIPFIRE_DIR/bin"
MODELS_DIR="$HIPFIRE_DIR/models"
SRC_DIR="$HIPFIRE_DIR/src"
GITHUB_REPO="Kaden-Schutt/hipfire"
GITHUB_BRANCH="master"

echo "=== hipfire installer ==="
echo ""

# ─── Interactive prompts (safe for curl|bash) ────────────
ask() {
    # Usage: result=$(ask "prompt [Y/n] " "Y")
    # Safe for curl|bash: reads from /dev/tty, falls back to default if non-interactive
    local prompt="$1" default="$2"
    if printf "%s" "$prompt" >/dev/tty 2>/dev/null; then
        local reply
        read -r reply </dev/tty 2>/dev/null || reply="$default"
        echo "${reply:-$default}"
    else
        echo "$default"
    fi
}

# Pick the right HIP runtime package name for dnf-based distros. Fedora's
# rocm-hip package is what ships libamdhip64.so.6; the rocm-hip-runtime
# meta-package only exists on RHEL / Rocky / Alma via AMD's repo. Detect
# via /etc/os-release ID + ID_LIKE. Reported in #64 (kamikazechaser).
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
        echo "Windows detected (via $OS)."
        echo ""
        echo "hipfire has native Windows support. Install options:"
        echo "  1. PowerShell (recommended):"
        echo "     irm https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/scripts/install.ps1 | iex"
        echo "  2. WSL2 (alternative):"
        echo "     wsl --install"
        echo "     # Then inside WSL:"
        echo "     curl -L https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/scripts/install.sh | bash"
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
    echo ""
    echo "Possible fixes:"
    echo "  - Install amdgpu driver: sudo apt install linux-firmware (Ubuntu)"
    echo "  - Reboot after driver install"
    echo "  - Check: lspci | grep -i amd"
    echo ""
    echo "Run 'hipfire diag' after install for automated troubleshooting."
    exit 1
fi
echo "  /dev/kfd: found ✓"

# Detect GPU arch via kfd topology (most reliable on modern kernels)
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
        110501)         GPU_ARCH="gfx1151"; break ;;
        120000)         GPU_ARCH="gfx1200"; break ;;
        120001)         GPU_ARCH="gfx1201"; break ;;
    esac
done

# Fallback: rocm-smi
if [ "$GPU_ARCH" = "unknown" ] && command -v rocm-smi &>/dev/null; then
    GPU_ARCH=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
fi

# Fallback: ask user
if [ "$GPU_ARCH" = "unknown" ]; then
    echo "  WARNING: Could not detect GPU architecture."
    echo "  Supported: gfx906 (Vega 20), gfx908 (MI100), gfx1010 (5700 XT), gfx1030 (6800 XT), gfx1100 (7900 XTX), gfx1151 (Strix Halo), gfx1200 (R9700), gfx1201 (9070 XT)"
    GPU_ARCH=$(ask "  Enter your GPU arch [or Enter to skip]: " "unknown")
fi
echo "  GPU arch: $GPU_ARCH"

# ─── HIP Runtime ─────────────────────────────────────────
echo ""
echo "Checking HIP runtime..."
HIP_FOUND=false
HIP_LIB=""
# Probe each known install directory for either the unversioned .so symlink
# or a versioned ABI variant (.so.6 / .so.7 / .so.8). Fedora's `rocm-hip`
# package installs only `libamdhip64.so.6` — the unversioned symlink ships
# in `rocm-hip-devel` which most users don't have. So checking only `.so`
# misses Fedora installs entirely.
for dir in /opt/rocm/lib /opt/rocm/lib64 \
           /usr/lib /usr/lib64 \
           /usr/lib/x86_64-linux-gnu /usr/lib64/rocm; do
    for suffix in "" ".6" ".7" ".8"; do
        lib="$dir/libamdhip64.so${suffix}"
        if [ -f "$lib" ]; then
            echo "  libamdhip64.so: found at $lib ✓"
            HIP_FOUND=true
            HIP_LIB="$lib"
            break 2
        fi
    done
done

# Fallback: ask ldconfig if none of the hardcoded paths matched. Match both
# unversioned (`libamdhip64.so`) and versioned (`libamdhip64.so.N`) entries
# — the trailing-space pattern from the previous version only matched the
# unversioned line, missing Fedora's `.so.6` SONAME.
if ! $HIP_FOUND; then
    ldconfig_hit=$(ldconfig -p 2>/dev/null | grep -m1 -E '\blibamdhip64\.so(\.[0-9]+)?\b' | awk '{print $NF}' || true)
    if [ -n "$ldconfig_hit" ] && [ -f "$ldconfig_hit" ]; then
        echo "  libamdhip64.so: found via ldconfig at $ldconfig_hit ✓"
        HIP_FOUND=true
        HIP_LIB="$ldconfig_hit"
    fi
fi

# Check HIP version matches GPU arch requirements
if $HIP_FOUND; then
    HIP_VER=""
    if command -v /opt/rocm/bin/hipconfig &>/dev/null; then
        HIP_VER=$(/opt/rocm/bin/hipconfig --version 2>/dev/null | grep -oP '^\d+\.\d+' || true)
    elif command -v hipconfig &>/dev/null; then
        HIP_VER=$(hipconfig --version 2>/dev/null | grep -oP '^\d+\.\d+' || true)
    fi

    if [ -n "$HIP_VER" ]; then
        HIP_MAJOR=$(echo "$HIP_VER" | cut -d. -f1)
        HIP_MINOR=$(echo "$HIP_VER" | cut -d. -f2)
        echo "  HIP version: $HIP_VER"

        # Minimum HIP versions per GPU arch
        MIN_MAJOR=5; MIN_MINOR=0
        case "$GPU_ARCH" in
            gfx1200|gfx1201) MIN_MAJOR=6; MIN_MINOR=4 ;;
            gfx1150|gfx1151|gfx1152) MIN_MAJOR=7; MIN_MINOR=2 ;;
            gfx1100|gfx1101) MIN_MAJOR=5; MIN_MINOR=5 ;;
        esac

        NEEDS_UPGRADE=false
        if [ "$HIP_MAJOR" -lt "$MIN_MAJOR" ] 2>/dev/null; then
            NEEDS_UPGRADE=true
        elif [ "$HIP_MAJOR" -eq "$MIN_MAJOR" ] && [ "$HIP_MINOR" -lt "$MIN_MINOR" ] 2>/dev/null; then
            NEEDS_UPGRADE=true
        fi

        if $NEEDS_UPGRADE; then
            echo ""
            echo "  WARNING: HIP $HIP_VER is too old for $GPU_ARCH (needs $MIN_MAJOR.$MIN_MINOR+)"
            echo "  Kernels may fail to load. Upgrading HIP runtime is recommended."
            PKG_CMD=""
            if command -v apt &>/dev/null; then
                PKG_CMD="sudo apt install -y rocm-hip-runtime"
            elif command -v dnf &>/dev/null; then
                PKG_CMD="sudo dnf install -y $(dnf_hip_pkg)"
            elif command -v pacman &>/dev/null; then
                PKG_CMD="sudo pacman -S --noconfirm rocm-hip-runtime"
            fi
            if [ -n "$PKG_CMD" ]; then
                reply=$(ask "  Upgrade now? ($PKG_CMD) [Y/n] " "Y")
                if [ "$reply" != "n" ] && [ "$reply" != "N" ]; then
                    echo "  Running: $PKG_CMD"
                    eval "$PKG_CMD" || echo "  Upgrade failed. You may need to add the ROCm repo first."
                fi
            else
                echo "  Upgrade manually: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
            fi
        fi
    fi
fi

if ! $HIP_FOUND; then
    echo "  libamdhip64.so: NOT FOUND"
    echo ""
    echo "  hipfire needs the HIP runtime library (libamdhip64.so)."
    echo "  This is a small package (~50MB), NOT the full ROCm SDK."

    # Detect package manager and offer guided install
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
        reply=$(ask "  Install now? ($PKG_CMD) [Y/n] " "Y")
        if [ "$reply" != "n" ] && [ "$reply" != "N" ]; then
            echo "  Running: $PKG_CMD"
            eval "$PKG_CMD" || {
                echo ""
                echo "  HIP runtime install failed. Try manually:"
                echo "    $PKG_CMD"
                echo "  Or see: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
                echo ""
                echo "  hipfire can still be installed, but won't run without libamdhip64.so."
                reply=$(ask "  Continue anyway? [y/N] " "N")
                if [ "$reply" != "y" ] && [ "$reply" != "Y" ]; then
                    exit 1
                fi
            }
        else
            echo "  Skipping. Install later: $PKG_CMD"
        fi
    else
        echo "  Unknown package manager. Install libamdhip64.so manually:"
        echo "  https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        reply=$(ask "  Continue without HIP runtime? [y/N] " "N")
        if [ "$reply" != "y" ] && [ "$reply" != "Y" ]; then
            exit 1
        fi
    fi
fi

# ─── Create directories ─────────────────────────────────
mkdir -p "$BIN_DIR" "$MODELS_DIR"

# ─── Determine install mode ──────────────────────────────
# Local: running from within a repo checkout (./scripts/install.sh)
# Remote: running via curl|bash — clone the repo
INSTALL_MODE="remote"
REPO_DIR=""
# INSTALL-F1: tracks whether the source tree was (re)fetched/cloned/reset this
# run. When source changed, an existing target/release/examples/daemon is STALE
# and MUST NOT be trusted — we rebuild. Only a genuinely fresh prebuilt binary
# (no source update this run) is allowed to short-circuit the build.
SOURCE_UPDATED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd 2>/dev/null)" || true
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../Cargo.toml" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    INSTALL_MODE="local"
fi

echo ""
if [ "$INSTALL_MODE" = "local" ]; then
    echo "Install mode: local (repo at $REPO_DIR)"
else
    echo "Install mode: remote (cloning repository)"

    if [ ! -d "$SRC_DIR/.git" ]; then
        if ! command -v git &>/dev/null; then
            echo "  ERROR: git is required for remote install."
            echo "  Install git and re-run, or clone manually:"
            echo "    git clone https://github.com/$GITHUB_REPO.git ~/.hipfire/src"
            exit 1
        fi
        echo "  Cloning https://github.com/$GITHUB_REPO.git ..."
        git clone --depth 1 --branch "$GITHUB_BRANCH" \
            "https://github.com/$GITHUB_REPO.git" "$SRC_DIR" || {
            echo "  Clone failed. Check your connection or try:"
            echo "    git clone https://github.com/$GITHUB_REPO.git $SRC_DIR"
            exit 1
        }
        echo "  Cloned ✓"
        SOURCE_UPDATED=1
    else
        echo "  Existing clone found at $SRC_DIR"
        # Stash any local modifications (Cargo.lock rewritten by cargo build,
        # autocrlf line-ending drift, user edits, etc.) so that the subsequent
        # reset can't abort with "local changes would be overwritten by merge".
        # The stash is named so the user can recover via `git stash pop`.
        if [ -n "$(git -C "$SRC_DIR" status --porcelain 2>/dev/null)" ]; then
            stamp=$(date -u +%Y-%m-%dT%H-%M-%SZ)
            stash_msg="hipfire-install-${stamp}"
            echo "  Local modifications detected — stashing as '$stash_msg'"
            if git -C "$SRC_DIR" stash push --include-untracked -m "$stash_msg" >/dev/null 2>&1; then
                echo "  Recover later with: git -C $SRC_DIR stash pop"
            else
                echo "  WARNING: git stash failed; reset may drop local changes."
            fi
        fi
        echo "  Updating..."
        # Fetch + hard-reset is safe now (tree is clean post-stash). Reset
        # handles both fast-forward and diverged-history cases uniformly.
        if git -C "$SRC_DIR" fetch origin "$GITHUB_BRANCH" --depth 1 2>/dev/null && \
           git -C "$SRC_DIR" reset --hard "origin/$GITHUB_BRANCH" 2>/dev/null; then
            # Source was reset to origin — any prior build artifact is now stale.
            SOURCE_UPDATED=1
        else
            echo "  Update failed (non-fatal). Using existing checkout."
        fi
    fi
    REPO_DIR="$SRC_DIR"
fi

# ─── Build / Install binaries ────────────────────────────
echo ""
echo "Installing hipfire..."

if ! command -v cargo &>/dev/null; then
    echo "  Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>/dev/null
    . "$HOME/.cargo/env"
fi

TARGET_DIR=$(cd "$REPO_DIR" && cargo metadata --format-version 1 | grep -oE '"target_directory" *: *"[^"]+"' | cut -d ':' -f 2- | tr -d '"')

# INSTALL-F1: only short-circuit on a prebuilt daemon when the source tree did
# NOT change this run (SOURCE_UPDATED=0). A leftover binary from a prior build
# against now-updated source is stale and must be rebuilt. HIPFIRE_FORCE_REBUILD=1
# (set by `hipfire update`) always forces a rebuild.
if [ -f "$TARGET_DIR/release/examples/daemon" ] && [ -f "$TARGET_DIR/release/hipfire" ] && [ "$SOURCE_UPDATED" = "0" ] && [ "${HIPFIRE_FORCE_REBUILD:-0}" = "0" ]; then
    echo "  Pre-built binaries found ✓"
else
    if [ -f "$TARGET_DIR/release/examples/daemon" ]; then
        echo "  Source updated — rebuilding (existing binary is stale)..."
    else
        echo "  No pre-built binaries. Building from source..."
    fi
    # MANDATORY build (daemon + inference examples). Stays inside the fatal
    # &&-chain: under `set -e` a failure here aborts the install, which is
    # correct — without the daemon there is nothing to install.
    (cd "$REPO_DIR" && \
        echo "  cargo build --release (this may take several minutes)..." && \
        cargo build --release --features deltanet --example daemon --example infer --example infer_hfq --example triattn_validate -p hipfire-runtime 2>&1 | tail -5)
    # The native Rust CLI is part of the mandatory product surface. Without it
    # model discovery, configuration, and daemon lifecycle are unavailable.
    (cd "$REPO_DIR" && \
        echo "  cargo build --release -p hipfire-cli..." && \
        cargo build --release -p hipfire-cli 2>&1 | tail -5)
    # OPTIONAL build (terminal UI). Run OUTSIDE the fatal chain so a tui build
    # failure does NOT trip `set -e` and abort the whole install after the
    # mandatory binaries already built. Warn clearly; the post-build copy step
    # surfaces the unavailability + remedy.
    (cd "$REPO_DIR" && \
        echo "  cargo build --release -p hipfire-tui (terminal UI)..." && \
        cargo build --release -p hipfire-tui 2>&1 | tail -5) \
        || echo "  WARNING: hipfire-tui (terminal UI) build failed — continuing without it."
    (cd "$REPO_DIR" && \
        echo "  cargo build --release -p hipfire-quantize (CPU quantizer)..." && \
        cargo build --release -p hipfire-quantize 2>&1 | tail -5) \
        || echo "  WARNING: hipfire-quantize build failed — continuing without quantize support."
    if [ ! -f "$TARGET_DIR/release/examples/daemon" ] || [ ! -f "$TARGET_DIR/release/hipfire" ]; then
        echo ""
        echo "  BUILD FAILED."
        echo "  Common causes:"
        echo "    - Missing ROCm SDK (needed to compile, not just run)"
        echo "    - Missing system libs (check error above)"
        echo ""
        echo "  After fixing, re-run this installer or build manually:"
        echo "    cd $REPO_DIR && cargo build --release --features deltanet --example daemon --example infer --example infer_hfq --example triattn_validate -p hipfire-runtime && cargo build --release -p hipfire-cli"
        exit 1
    fi
    echo "  Build complete ✓"
fi

# Copy binaries
cp "$TARGET_DIR/release/examples/daemon" "$BIN_DIR/daemon"
cp "$TARGET_DIR/release/hipfire" "$BIN_DIR/hipfire"
cp "$TARGET_DIR/release/examples/infer" "$BIN_DIR/infer" 2>/dev/null || true
cp "$TARGET_DIR/release/examples/infer_hfq" "$BIN_DIR/infer_hfq" 2>/dev/null || true
cp "$TARGET_DIR/release/examples/triattn_validate" "$BIN_DIR/triattn_validate" 2>/dev/null || true
if [ ! -f "$TARGET_DIR/release/hipfire-quantize" ] && command -v cargo &>/dev/null; then
    (cd "$REPO_DIR" && cargo build --release -p hipfire-quantize 2>&1 | tail -3) || true
fi
cp "$TARGET_DIR/release/hipfire-quantize" "$BIN_DIR/hipfire-quantize" 2>/dev/null || true
# Terminal UI (workspace bin — lives under target/release/, not examples/).
# Build it on demand if the pre-built tree didn't include it. The TUI is
# OPTIONAL — a failed build (or absent cargo) must not abort the install, but
# it must not be swallowed silently either: warn clearly so the user knows the
# terminal UI won't be available and how to get it (consistent w/ install.ps1).
if [ ! -f "$TARGET_DIR/release/hipfire-tui" ]; then
    if command -v cargo &>/dev/null; then
        (cd "$REPO_DIR" && cargo build --release -p hipfire-tui 2>&1 | tail -3) || true
    fi
fi
if [ -f "$TARGET_DIR/release/hipfire-tui" ]; then
    cp "$TARGET_DIR/release/hipfire-tui" "$BIN_DIR/hipfire-tui"
    echo "  hipfire-tui (terminal UI) installed ✓"
else
    echo "  WARNING: hipfire-tui (terminal UI) was not built — it will be unavailable."
    # Linux remedy: `hipfire update` is Linux-aware and builds + installs the
    # binary in one step. The direct cargo build is the fallback.
    echo "           To get it: install Rust and run \`hipfire update\` (recommended),"
    echo "           or build directly: \`cargo build --release -p hipfire-tui\`."
fi

# Remove the retired TypeScript runtime payload from older installations only
# after the native binary has been copied successfully above.
rm -rf "$HIPFIRE_DIR/cli"
chmod +x "$BIN_DIR/hipfire"
echo "  Native binaries installed to $BIN_DIR/ ✓"

# ─── Install kernels ────────────────────────────────────
# Engine probes for kernels at {exe_dir}/kernels/compiled/{arch}/
# so we place them at ~/.hipfire/bin/kernels/compiled/{arch}/
echo ""
if [ "$GPU_ARCH" != "unknown" ]; then
    echo "Setting up kernels for $GPU_ARCH..."
    KERNEL_DEST="$BIN_DIR/kernels/compiled/$GPU_ARCH"
    mkdir -p "$KERNEL_DEST"

    if [ -d "$REPO_DIR/kernels/compiled/$GPU_ARCH" ]; then
        cp "$REPO_DIR/kernels/compiled/$GPU_ARCH"/*.hsaco "$KERNEL_DEST/" 2>/dev/null
        cp "$REPO_DIR/kernels/compiled/$GPU_ARCH"/*.hash "$KERNEL_DEST/" 2>/dev/null
        count=$(ls "$KERNEL_DEST"/*.hsaco 2>/dev/null | wc -l)
        echo "  Copied $count kernels + hashes to $KERNEL_DEST/ ✓"
    else
        echo "  No pre-compiled kernels for $GPU_ARCH in repo — will JIT from source below."
    fi
else
    echo "Skipping pre-built kernel copy (GPU arch unknown) — daemon will still"
    echo "  auto-detect at runtime. Missing kernels compile on first use."
fi

# Fill in any kernels missing from the pre-compiled set, including MQ4/asym3
# defaults that aren't always shipped for newer arches. Uses hipcc in parallel
# and writes back to ~/.hipfire/bin/kernels/compiled/<arch>/ so first
# `hipfire run` isn't a 2-minute compile wall. Runs regardless of install-time
# arch detection — the daemon's own Gpu::init resolves the active arch.
if [ -x "$BIN_DIR/daemon" ]; then
    echo ""
    echo "Pre-compiling GPU kernels (first run will be instant afterward)..."
    if "$BIN_DIR/daemon" --precompile; then
        echo "  Pre-compile complete ✓"
    else
        echo "  Pre-compile finished with warnings — any missing kernels will compile on first use."
    fi
fi

# ─── Config ──────────────────────────────────────────────
CONFIG="$HIPFIRE_DIR/config.toml"
if [ ! -f "$CONFIG" ] && [ ! -f "$HIPFIRE_DIR/config.json" ]; then
    cat > "$CONFIG" << CONF
schema_version = 1
CONF
    echo ""
    echo "Config: $CONFIG"
fi

# ─── PATH setup ─────────────────────────────────────────
echo ""
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    SHELL_RC=""
    case "$(basename "${SHELL:-bash}")" in
        bash) SHELL_RC="$HOME/.bashrc" ;;
        zsh)  SHELL_RC="$HOME/.zshrc" ;;
    esac

    PATH_LINE="export PATH=\"\$HOME/.hipfire/bin:\$PATH\""
    if [ -n "$SHELL_RC" ] && [ -f "$SHELL_RC" ]; then
        if ! grep -q '.hipfire/bin' "$SHELL_RC" 2>/dev/null; then
            reply=$(ask "Add hipfire to PATH in $SHELL_RC? [Y/n] " "Y")
            if [ "$reply" != "n" ] && [ "$reply" != "N" ]; then
                printf '\n# hipfire\n%s\n' "$PATH_LINE" >> "$SHELL_RC"
                echo "  Added to $SHELL_RC ✓"
            else
                echo "  Add manually: $PATH_LINE"
            fi
        fi
    else
        echo "Add to your shell profile:"
        echo "  $PATH_LINE"
    fi
fi

echo ""
echo "=== hipfire installed ==="
echo ""
echo "Quick start:"
echo "  source ${SHELL_RC:-~/.bashrc}                    # reload PATH (or restart shell)"
echo "  hipfire list                                      # see local models"
echo "  hipfire run <model.hfq> \"Hello\"                  # generate text"
echo "  hipfire serve                                     # start OpenAI-compatible API"
echo ""
echo "Models go in ~/.hipfire/models/ or the repo's models/ directory."
echo ""
