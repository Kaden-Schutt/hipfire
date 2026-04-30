#!/usr/bin/env bash
# Source this file to set LD_LIBRARY_PATH for ROCm/HIP.
# Tries (in order): existing LD_LIBRARY_PATH, /opt/rocm, Nix store.

if ldconfig -p 2>/dev/null | grep -q libamdhip64; then
    return 0 2>/dev/null || true  # already in system path
fi

if [ -d "/opt/rocm/lib" ]; then
    export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
    return 0 2>/dev/null || true
fi

# NixOS: find the clr package in the Nix store
NIX_HIP=$(find /nix/store -maxdepth 2 -name "libamdhip64.so" -path "*/clr-*/lib/*" 2>/dev/null | head -1)
if [ -n "$NIX_HIP" ]; then
    export LD_LIBRARY_PATH="$(dirname "$NIX_HIP"):${LD_LIBRARY_PATH:-}"
    return 0 2>/dev/null || true
fi

echo "WARNING: libamdhip64.so not found. ROCm/HIP may not be installed." >&2
