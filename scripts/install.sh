#!/bin/bash
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
        100100)         GPU_ARCH="gfx1010"; break ;;
        100300|100302)  GPU_ARCH="gfx1030"; break ;;
        110000|110001)  GPU_ARCH="gfx1100"; break ;;
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
    echo "  Supported: gfx1010 (5700 XT), gfx1030 (6800 XT), gfx1100 (7900 XTX), gfx1200 (9060), gfx1201 (9070 XT)"
    GPU_ARCH=$(ask "  Enter your GPU arch [or Enter to skip]: " "unknown")
fi
echo "  GPU arch: $GPU_ARCH"

# ─── HIP Runtime ─────────────────────────────────────────
echo ""
echo "Checking HIP runtime..."
HIP_FOUND=false
HIP_LIB=""
for lib in /opt/rocm/lib/libamdhip64.so \
           /opt/rocm/lib64/libamdhip64.so \
           /usr/lib/libamdhip64.so \
           /usr/lib64/libamdhip64.so \
           /usr/lib/x86_64-linux-gnu/libamdhip64.so \
           /usr/lib64/rocm/libamdhip64.so; do
    if [ -f "$lib" ]; then
        echo "  libamdhip64.so: found at $lib ✓"
        HIP_FOUND=true
        HIP_LIB="$lib"
        break
    fi
done

# Fallback: ask ldconfig if none of the hardcoded paths matched
if ! $HIP_FOUND; then
    ldconfig_hit=$(ldconfig -p 2>/dev/null | grep -m1 'libamdhip64.so ' | awk '{print $NF}' || true)
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
                PKG_CMD="sudo dnf install -y rocm-hip-runtime"
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
        PKG_CMD="sudo dnf install -y rocm-hip-runtime"
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

# ─── Install Bun (needed for CLI) ───────────────────────
echo ""
if command -v bun &>/dev/null; then
    echo "Bun: found ✓"
else
    echo "Installing Bun (runtime for hipfire CLI)..."
    curl -fsSL https://bun.sh/install | bash 2>/dev/null || {
        echo "  Bun install failed. Visit https://bun.sh"
        echo "  hipfire CLI requires Bun to run."
        exit 1
    }
    # Source bun into current session
    export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
    export PATH="$BUN_INSTALL/bin:$PATH"
    if command -v bun &>/dev/null; then
        echo "  Bun installed ✓"
    else
        echo "  Bun installed but not in PATH. Restart your shell or run:"
        echo "    export PATH=\"\$HOME/.bun/bin:\$PATH\""
    fi
fi

# ─── Create directories ─────────────────────────────────
mkdir -p "$BIN_DIR" "$MODELS_DIR"

# ─── Determine install mode ──────────────────────────────
# Local: running from within a repo checkout (./scripts/install.sh)
# Remote: running via curl|bash — clone the repo
INSTALL_MODE="remote"
REPO_DIR=""

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
    else
        echo "  Existing clone found at $SRC_DIR"
        if [ -n "$(git -C "$SRC_DIR" status --porcelain 2>/dev/null)" ]; then
            echo "  WARNING: local modifications detected in $SRC_DIR"
            reply=$(ask "  Overwrite local changes and update? [y/N] " "N")
            if [ "$reply" != "y" ] && [ "$reply" != "Y" ]; then
                echo "  Keeping existing checkout as-is."
            else
                echo "  Updating..."
                git -C "$SRC_DIR" fetch origin "$GITHUB_BRANCH" --depth 1 2>/dev/null && \
                git -C "$SRC_DIR" reset --hard "origin/$GITHUB_BRANCH" 2>/dev/null || {
                    echo "  Update failed (non-fatal). Using existing checkout."
                }
            fi
        else
            echo "  Updating..."
            git -C "$SRC_DIR" fetch origin "$GITHUB_BRANCH" --depth 1 2>/dev/null && \
            git -C "$SRC_DIR" pull --ff-only origin "$GITHUB_BRANCH" 2>/dev/null || {
                echo "  Update failed (non-fatal). Using existing checkout."
            }
        fi
    fi
    REPO_DIR="$SRC_DIR"
fi

# ─── Build / Install binaries ────────────────────────────
echo ""
echo "Installing hipfire..."

if [ -f "$REPO_DIR/target/release/examples/daemon" ]; then
    echo "  Pre-built binaries found ✓"
else
    echo "  No pre-built binaries. Building from source..."
    if ! command -v cargo &>/dev/null; then
        echo "  Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>/dev/null
        . "$HOME/.cargo/env"
    fi
    (cd "$REPO_DIR" && \
        echo "  cargo build --release (this may take several minutes)..." && \
        cargo build --release --features deltanet --example daemon --example infer --example infer_hfq -p engine 2>&1 | tail -5)
    if [ ! -f "$REPO_DIR/target/release/examples/daemon" ]; then
        echo ""
        echo "  BUILD FAILED."
        echo "  Common causes:"
        echo "    - Missing ROCm SDK (needed to compile, not just run)"
        echo "    - Missing system libs (check error above)"
        echo ""
        echo "  After fixing, re-run this installer or build manually:"
        echo "    cd $REPO_DIR && cargo build --release --features deltanet --example daemon -p engine"
        exit 1
    fi
    echo "  Build complete ✓"
fi

# Copy binaries
cp "$REPO_DIR/target/release/examples/daemon" "$BIN_DIR/daemon"
cp "$REPO_DIR/target/release/examples/infer" "$BIN_DIR/infer" 2>/dev/null || true
cp "$REPO_DIR/target/release/examples/infer_hfq" "$BIN_DIR/infer_hfq" 2>/dev/null || true

# Copy CLI
mkdir -p "$HIPFIRE_DIR/cli"
cp "$REPO_DIR/cli/index.ts" "$HIPFIRE_DIR/cli/index.ts"
cp "$REPO_DIR/cli/package.json" "$HIPFIRE_DIR/cli/package.json"

# Create hipfire wrapper
cat > "$BIN_DIR/hipfire" << 'WRAPPER'
#!/bin/bash
exec bun run "$HOME/.hipfire/cli/index.ts" "$@"
WRAPPER
chmod +x "$BIN_DIR/hipfire"
echo "  Binaries + CLI installed to $BIN_DIR/ ✓"

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
        echo "  WARNING: No pre-compiled kernels for $GPU_ARCH in repo."
        echo "  Compile them: cd $REPO_DIR && scripts/compile-kernels.sh $GPU_ARCH"
        echo "  Then copy: cp kernels/compiled/$GPU_ARCH/*.hsaco $KERNEL_DEST/"
    fi
else
    echo "Skipping kernel setup (GPU arch unknown)."
    echo "  Re-run installer after fixing GPU detection, or copy kernels manually."
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
