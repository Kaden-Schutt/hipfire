#!/usr/bin/env bash
# Build the hipfire gate-runner image and run a GPU gate inside it, with the
# host GPU (/dev/kfd + /dev/dri) passed through. This is the containerized
# PR / dev-build validation path — the same in-repo gates, reproducibly.
#
# Usage:
#   scripts/container-gate.sh [gate-script [args...]]
#
# Defaults to scripts/coherence-gate.sh. Examples:
#   scripts/container-gate.sh                              # coherence battery
#   scripts/container-gate.sh scripts/serve-multiturn-gate.sh
#   scripts/container-gate.sh scripts/coherence-gate-dflash.sh
#
# Environment:
#   HIPFIRE_CONTAINER=podman|docker   container runtime (default: podman)
#   HIPFIRE_MODELS_DIR=<path>         host models dir (default: ~/.hipfire/models)
#   HIPFIRE_IMAGE=<name>              image tag (default: hipfire-gate)
#   HIPFIRE_SKIP_BUILD=1              skip the image build, run the existing tag
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="${HIPFIRE_CONTAINER:-podman}"
IMAGE="${HIPFIRE_IMAGE:-hipfire-gate}"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"

GATE_CMD=("$@")
if [ ${#GATE_CMD[@]} -eq 0 ]; then
    # No default: the historical default (scripts/coherence-gate.sh) is retired
    # and absent from the checkout, so defaulting to it only produced a confusing
    # exit 127 inside the container. Fail closed here with a usable message.
    echo "[container-gate] no gate command given." >&2
    echo "  There is no universal gate; select a route from docs/VALIDATION.md and pass it, e.g.:" >&2
    echo "    scripts/container-gate.sh scripts/serve_harness.py battery --model <model>" >&2
    echo "    scripts/container-gate.sh scripts/redline_daemon_harness.py" >&2
    exit 2
fi

if [ ! -d "$MODELS_DIR" ]; then
    echo "[container-gate] models dir not found: $MODELS_DIR" >&2
    echo "  set HIPFIRE_MODELS_DIR, or pull models on the host first." >&2
    exit 2
fi

if [ "${HIPFIRE_SKIP_BUILD:-0}" != "1" ]; then
    echo "[container-gate] building $IMAGE (gate-runner target)..."
    "$RUNTIME" build -f "$ROOT/Containerfile" --target gate-runner \
        -t "$IMAGE" "$ROOT" || exit 1
fi

# Rootless podman needs --group-add keep-groups to keep the host render/video
# gids for /dev/kfd + /dev/dri access. Docker does not implement Podman's
# special keep-groups token (it treats it as a literal group name), and its
# default rootful mode does not need it. seccomp=unconfined avoids HSA syscall
# filtering. Share the host GPU lock file if it exists so a containerized gate
# coordinates with host GPU work (see scripts/gpu-lock.sh).
GPU_ARGS=(
    --device /dev/kfd
    --device /dev/dri
    --security-opt seccomp=unconfined
    -v "$MODELS_DIR:/root/.hipfire/models"
    -v hipfire-kcache:/var/cache/hipfire
)
if [ "$RUNTIME" = "podman" ]; then
    GPU_ARGS+=(--group-add keep-groups)
fi
LOCKFILE="${HIPFIRE_GPU_LOCKFILE:-/tmp/hipfire-gpu.lock}"
[ -e "$LOCKFILE" ] && GPU_ARGS+=(-v "$LOCKFILE:$LOCKFILE")

echo "[container-gate] running: ${GATE_CMD[*]}"
exec "$RUNTIME" run --rm -i "${GPU_ARGS[@]}" "$IMAGE" "${GATE_CMD[@]}"
