#!/bin/bash
# hipfire installer — detects GPU, installs deps, downloads binary + kernels.
# Usage: curl -L hipfire.schutt.dev/install | sh
set -euo pipefail

HIPFIRE_DIR="$HOME/.hipfire"
BIN_DIR="$HIPFIRE_DIR/bin"
KERNELS_DIR="$HIPFIRE_DIR/kernels"
MODELS_DIR="$HIPFIRE_DIR/models"
GITHUB_REPO="autorocm/hipfire"  # TODO: update when public

echo "=== hipfire installer ==="
echo ""

# ─── OS Detection ────────────────────────────────────────
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$OS" in
    linux) ;;
    darwin) echo "macOS is not supported (AMD GPUs only). Exiting."; exit 1 ;;
    *) echo "Unsupported OS: $OS"; exit 1 ;;
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
    exit 1
fi
echo "  /dev/kfd: found ✓"

# Try to detect arch
GPU_ARCH="unknown"
if command -v rocm-smi &>/dev/null; then
    GPU_ARCH=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
fi
if [ "$GPU_ARCH" = "unknown" ] && [ -f /sys/class/drm/card0/device/gpu_id ]; then
    # Fallback: parse from sysfs
    GPU_ID=$(cat /sys/class/drm/card0/device/gpu_id 2>/dev/null || echo "")
    case "$GPU_ID" in
        *731f*|*7340*) GPU_ARCH="gfx1010" ;;
        *73bf*|*73df*) GPU_ARCH="gfx1030" ;;
        *744c*|*7480*) GPU_ARCH="gfx1100" ;;
        *150*|*154*) GPU_ARCH="gfx1200" ;;
    esac
fi
echo "  GPU arch: $GPU_ARCH"

# ─── HIP Runtime ─────────────────────────────────────────
echo ""
echo "Checking HIP runtime..."
HIP_FOUND=false
for lib in /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so /usr/lib/x86_64-linux-gnu/libamdhip64.so; do
    if [ -f "$lib" ]; then
        echo "  libamdhip64.so: found at $lib ✓"
        HIP_FOUND=true
        break
    fi
done

if ! $HIP_FOUND; then
    echo "  libamdhip64.so: NOT FOUND"
    echo ""
    echo "Installing ROCm HIP runtime..."
    if command -v apt &>/dev/null; then
        echo "  Running: sudo apt install -y rocm-hip-runtime"
        sudo apt install -y rocm-hip-runtime 2>/dev/null || {
            echo "  apt install failed. Try manually: sudo apt install rocm-hip-runtime"
            echo "  Or see: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        }
    elif command -v dnf &>/dev/null; then
        echo "  Running: sudo dnf install -y rocm-hip-runtime"
        sudo dnf install -y rocm-hip-runtime 2>/dev/null || echo "  dnf install failed."
    elif command -v pacman &>/dev/null; then
        echo "  Running: sudo pacman -S --noconfirm rocm-hip-runtime"
        sudo pacman -S --noconfirm rocm-hip-runtime 2>/dev/null || echo "  pacman install failed."
    else
        echo "  Unknown package manager. Install libamdhip64.so manually."
    fi
fi

# ─── Create directories ─────────────────────────────────
mkdir -p "$BIN_DIR" "$KERNELS_DIR" "$MODELS_DIR"

# ─── Download binary ─────────────────────────────────────
echo ""
echo "Downloading hipfire binary..."
# For alpha: copy from local build if available
if [ -f "target/release/examples/daemon" ]; then
    cp target/release/examples/daemon "$BIN_DIR/daemon"
    cp target/release/examples/infer "$BIN_DIR/infer" 2>/dev/null || true
    cp target/release/examples/infer_hfq "$BIN_DIR/infer_hfq" 2>/dev/null || true
    echo "  Copied from local build ✓"
else
    echo "  TODO: download from GitHub releases"
    echo "  For now: cargo build --release --features deltanet --example daemon --example infer --example infer_hfq -p engine"
fi

# ─── Download kernels ────────────────────────────────────
echo ""
echo "Setting up kernels for $GPU_ARCH..."
if [ -d "kernels/compiled/$GPU_ARCH" ]; then
    cp -r "kernels/compiled/$GPU_ARCH" "$KERNELS_DIR/$GPU_ARCH"
    echo "  Copied $(ls "$KERNELS_DIR/$GPU_ARCH"/*.hsaco 2>/dev/null | wc -l) kernels ✓"
else
    echo "  TODO: download from GitHub releases for $GPU_ARCH"
fi

# ─── Install Bun (if needed) ────────────────────────────
echo ""
if command -v bun &>/dev/null; then
    echo "Bun: found ✓"
else
    echo "Installing Bun..."
    curl -fsSL https://bun.sh/install | bash 2>/dev/null || echo "  Bun install failed. Visit https://bun.sh"
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

# ─── PATH ────────────────────────────────────────────────
echo ""
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "Add to your shell profile:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi

echo ""
echo "=== hipfire installed ==="
echo ""
echo "Next steps:"
echo "  hipfire list                              # see available models"
echo "  hipfire run <model.hfq> \"Hello\"          # generate text"
echo "  hipfire serve                             # start API server"
echo ""
