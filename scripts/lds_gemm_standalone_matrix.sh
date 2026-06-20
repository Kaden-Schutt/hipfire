#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
OUT="${1:-/tmp/hipfire-lds-gemm-standalone-artifacts}"
VARIANT="${VARIANT:-tile6}"
MODE="${MODE:-full}"
LAUNCHES="${N_LAUNCH:-100}"
M="${M:-512}"
N="${N:-3072}"
K="${K:-3072}"
K_LIMIT="${K_LIMIT:-0}"
ARCH="${ARCH:-gfx1103}"
BUILD_ONLY="${BUILD_ONLY:-0}"

tag="${VARIANT}_${MODE}_n${LAUNCHES}_m${M}_n${N}_k${K}_klim${K_LIMIT}"
dest="$OUT/$tag"
mkdir -p "$dest"

bin="$dest/lds_gemm_standalone_probe"
temps="$dest/save-temps"
mkdir -p "$temps"

{
  echo "variant=$VARIANT"
  echo "mode=$MODE"
  echo "launches=$LAUNCHES"
  echo "shape=$M x $N x $K"
  echo "k_limit=$K_LIMIT"
  echo "arch=$ARCH"
  echo "build_only=$BUILD_ONLY"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname=$(uname -a)"
  if [[ -r /etc/os-release ]]; then
    grep '^PRETTY_NAME=' /etc/os-release || true
  fi
  echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
  echo "git_branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  echo "git_status_short=$(git -C "$REPO_ROOT" status --short 2>/dev/null | tr '\n' ';' || true)"
  echo "hipcc=$(/opt/rocm/bin/hipcc --version 2>/dev/null | head -1 || true)"
  echo "llvm_objdump=$(/opt/rocm/llvm/bin/llvm-objdump --version 2>/dev/null | head -1 || true)"
  echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-}"
  echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
  echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
  /opt/rocm/bin/rocminfo | sed -n '/Agent 2/,/Agent 3/p' | grep -E 'Name:|Marketing Name|Vendor Name' || true
  /opt/rocm/bin/rocm-smi --showproductname --showdriverversion || true
} > "$dest/meta.txt"

cp "$ROOT/lds_gemm_standalone_probe.hip" "$dest/lds_gemm_standalone_probe.hip"

/opt/rocm/bin/hipcc -O3 --offload-arch="$ARCH" -save-temps=obj \
  "$ROOT/lds_gemm_standalone_probe.hip" -o "$bin" > "$dest/build.log" 2>&1

find "$dest" "$ROOT" -maxdepth 2 -type f \( -name '*.hsaco' -o -name '*.o' -o -name '*.s' -o -name '*.ll' \) \
  > "$dest/generated-files.txt" 2>/dev/null || true

while IFS= read -r f; do
  [ -f "$f" ] || continue
  base="$(basename "$f")"
  cp "$f" "$temps/$base" 2>/dev/null || true
  if file "$f" | grep -qi ELF; then
    /opt/rocm/llvm/bin/llvm-readobj --notes --sections --symbols "$f" \
      > "$temps/$base.readobj.txt" 2>&1 || true
    /opt/rocm/llvm/bin/llvm-objdump -d --mcpu="$ARCH" "$f" \
      > "$temps/$base.isa.txt" 2>&1 || true
  fi
done < "$dest/generated-files.txt"

if [[ "$BUILD_ONLY" == "1" ]]; then
  echo "0" > "$dest/exit_code.txt"
  echo "build-only; kernel was not launched" > "$dest/run.log"
  exit 0
fi

dmesg --ctime > "$dest/dmesg.before.txt" 2>&1 || true
set +e
"$bin" "$VARIANT" "$MODE" "$LAUNCHES" "$M" "$N" "$K" "$K_LIMIT" > "$dest/run.log" 2>&1
rc=$?
set -e
dmesg --ctime > "$dest/dmesg.after.txt" 2>&1 || true
echo "$rc" > "$dest/exit_code.txt"

if [ "$rc" -ne 0 ] && [ -r /sys/class/drm/card0/device/devcoredump/data ]; then
  timeout 10s sudo -n dd if=/sys/class/drm/card0/device/devcoredump/data \
    of="$dest/devcoredump.data" bs=1M count=16 status=none || true
fi

exit "$rc"
