#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-$PWD/.venv-rocm}"

if systemctl --user is-active --quiet hipfire-fastmtp-teacher.service; then
    echo "Stage 1 teacher service is still active; refusing to alter the ROCm training environment." >&2
    exit 2
fi

sudo apt-get update
sudo apt-get install -y python3-torch-rocm python3-venv
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -r scripts/mtp_train/fastmtp35/requirements.txt

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
