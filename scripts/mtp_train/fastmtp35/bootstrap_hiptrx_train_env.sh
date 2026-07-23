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

"$VENV/bin/python" - <<'PY'
import torch

assert torch.version.hip, "installed torch is not a ROCm build"
assert torch.cuda.is_available(), "ROCm torch cannot see a GPU"
count = torch.cuda.device_count()
assert count == 4, f"expected four R9700s, found {count}"
arches = []
for index in range(count):
    props = torch.cuda.get_device_properties(index)
    arches.append(getattr(props, "gcnArchName", "unknown"))
assert all(str(arch).startswith("gfx1201") for arch in arches), arches
print(
    {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "devices": [torch.cuda.get_device_name(index) for index in range(count)],
        "arches": arches,
    }
)
PY
