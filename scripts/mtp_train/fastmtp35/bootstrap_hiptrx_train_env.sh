#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-$PWD/.venv-rocm}"
# The host may have a user-site CPU-only torch that otherwise precedes the
# distro ROCm build even inside a --system-site-packages venv.
export PYTHONNOUSERSITE=1

if systemctl --user is-active --quiet hipfire-fastmtp-teacher.service; then
    echo "Stage 1 teacher service is still active; refusing to alter the ROCm training environment." >&2
    exit 2
fi

sudo apt-get update
sudo apt-get install -y python3-torch-rocm python3-venv
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -r scripts/mtp_train/fastmtp35/requirements.txt

HIP_RUNTIME="$(ldconfig -p | awk '$1 == "libamdhip64.so.7" { print $NF; exit }')"
[[ -n "$HIP_RUNTIME" ]] || {
    echo "the dynamic loader cannot resolve libamdhip64.so.7" >&2
    exit 2
}
HIP_RUNTIME="$(readlink -f "$HIP_RUNTIME")"
case "$HIP_RUNTIME" in
    /opt/rocm/core-*/lib/libamdhip64.so.*) ;;
    *)
        echo "loader no longer prefers the known-good /opt ROCm HIP runtime: $HIP_RUNTIME" >&2
        exit 2
        ;;
esac
echo "preferred HIP runtime: $HIP_RUNTIME"

"$VENV/bin/python" scripts/mtp_train/fastmtp35/validate_rocm_train_env.py \
    --mode devices \
    --expected-devices 4 \
    --arch-prefix gfx1201
"$VENV/bin/python" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=4 \
    scripts/mtp_train/fastmtp35/validate_rocm_train_env.py \
    --mode distributed \
    --expected-devices 4
