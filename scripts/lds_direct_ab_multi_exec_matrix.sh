#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-/tmp/hipfire-lds-direct-ab-multi-exec-artifacts}"
SUMMARY="${SUMMARY:-$ROOT/lds_direct_ab_artifact_summary.sh}"
BUILD_ONLY="${BUILD_ONLY:-1}"
ACTIVE_X="${ACTIVE_X:-6}"
ACTIVE_Y="${ACTIVE_Y:-6}"
BLOCK_X="${BLOCK_X:-$ACTIVE_X}"
BLOCK_Y="${BLOCK_Y:-$ACTIVE_Y}"
READS="${READS:-3}"
ITERS="${ITERS:-448}"
CHUNKS="${CHUNKS:-96,5}"
GRID_X="${GRID_X:-512}"
GRID_Y="${GRID_Y:-86}"
MODE="${MODE:-plain}"
ARCH="${ARCH:-gfx1103}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
ROCMINFO="${ROCMINFO:-/opt/rocm/bin/rocminfo}"
ROCMSMI="${ROCMSMI:-/opt/rocm/bin/rocm-smi}"
READOBJ="${READOBJ:-/opt/rocm/llvm/bin/llvm-readobj}"
OBJDUMP="${OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}"
CLEAR_COREDUMP="${CLEAR_COREDUMP:-0}"
WAIT_DEVCD_MS="${WAIT_DEVCD_MS:-8000}"

tag_chunks="${CHUNKS//,/_}"
tag="a${ACTIVE_X}x${ACTIVE_Y}_b${BLOCK_X}x${BLOCK_Y}_r${READS}_i${ITERS}_chunks${tag_chunks}_multi_${MODE}_g${GRID_X}x${GRID_Y}"
dest="$OUT/$tag"
mkdir -p "$dest/save-temps" "$dest/coredumps"

for tool in "$HIPCC" "$READOBJ" "$OBJDUMP"; do
  if [ ! -x "$tool" ]; then
    echo "missing executable: $tool" >&2
    exit 1
  fi
done
if [ ! -x "$SUMMARY" ]; then
  echo "missing executable: $SUMMARY" >&2
  exit 1
fi

if [ "$CLEAR_COREDUMP" = "1" ]; then
  for data in /sys/class/devcoredump/devcd*/data; do
    [ -e "$data" ] || continue
    echo 1 | sudo -n tee "$data" >/dev/null || true
  done
fi

start_epoch="$(date +%s)"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
start_since="$(date -u '+%Y-%m-%d %H:%M:%S')"

{
  echo "active=$ACTIVE_X x $ACTIVE_Y"
  echo "block=$BLOCK_X x $BLOCK_Y"
  echo "reads=$READS"
  echo "lds_bytes=$((8 * ACTIVE_X * ACTIVE_Y))"
  echo "active_threads=$((ACTIVE_X * ACTIVE_Y))"
  echo "block_threads=$((BLOCK_X * BLOCK_Y))"
  echo "iters=$ITERS"
  echo "chunks=$CHUNKS"
  echo "mode=$MODE"
  echo "grid=$GRID_X x $GRID_Y"
  echo "arch=$ARCH"
  echo "build_only=$BUILD_ONLY"
  echo "hipcc=$HIPCC"
  echo "clear_coredump=$CLEAR_COREDUMP"
  echo "wait_devcd_ms=$WAIT_DEVCD_MS"
  echo "date=$start_iso"
  "$HIPCC" --version 2>/dev/null | sed 's/^/hipcc_version=/' | head -1 || true
  "$ROCMINFO" | sed -n '/Agent 2/,/Agent 3/p' | grep -E 'Name:|Marketing Name|Vendor Name' || true
  "$ROCMSMI" --showproductname --showdriverversion || true
} > "$dest/meta.txt"

cp "$ROOT/lds_direct_ab_phase_probe.hip" "$dest/lds_direct_ab_phase_probe.hip"
cp "$ROOT/lds_direct_ab_multi_exec_parent.cpp" "$dest/lds_direct_ab_multi_exec_parent.cpp"

(
  cd "$dest"
  "$HIPCC" -O3 --offload-arch="$ARCH" -save-temps=obj \
    -DACTIVE_X="$ACTIVE_X" -DACTIVE_Y="$ACTIVE_Y" \
    -DBLOCK_X="$BLOCK_X" -DBLOCK_Y="$BLOCK_Y" \
    -DREADS="$READS" -DITERS="$ITERS" \
    "$ROOT/lds_direct_ab_phase_probe.hip" \
    -lhsa-runtime64 \
    -o "$dest/lds_direct_ab_phase_probe" > "$dest/build-phase.log" 2>&1

  "$HIPCC" -O2 "$ROOT/lds_direct_ab_multi_exec_parent.cpp" \
    -o "$dest/lds_direct_ab_multi_exec_parent" > "$dest/build-parent.log" 2>&1
)

find "$dest" -maxdepth 2 -type f \( -name '*.hsaco' -o -name '*.o' -o -name '*.s' -o -name '*.ll' \) \
  > "$dest/generated-files.txt" 2>/dev/null || true

while IFS= read -r f; do
  [ -f "$f" ] || continue
  base="$(basename "$f")"
  cp "$f" "$dest/save-temps/$base" 2>/dev/null || true
  if file "$f" | grep -qi ELF; then
    "$READOBJ" --notes --sections --symbols "$f" \
      > "$dest/save-temps/$base.readobj.txt" 2>&1 || true
    "$OBJDUMP" -d --mcpu="$ARCH" "$f" \
      > "$dest/save-temps/$base.isa.txt" 2>&1 || true
  fi
done < "$dest/generated-files.txt"

if [ "$BUILD_ONLY" = "1" ]; then
  {
    echo "build_only=1"
    echo "artifact=$dest"
    echo "phase_bin=$dest/lds_direct_ab_phase_probe"
    echo "parent_bin=$dest/lds_direct_ab_multi_exec_parent"
  } > "$dest/summary.txt"
  "$SUMMARY" "$OUT" "$OUT/direct-ab-artifact-summary" > "$dest/artifact-summary.log" 2>&1 || {
    cat "$dest/artifact-summary.log" >&2
    exit 1
  }
  cat "$dest/summary.txt"
  cat "$dest/artifact-summary.log"
  exit 0
fi

echo "[lds-direct-ab] risky run enabled: BUILD_ONLY=$BUILD_ONLY chunks=$CHUNKS grid=${GRID_X}x${GRID_Y}" >&2

dmesg --ctime > "$dest/dmesg.before.txt" 2>&1 || true
set +e
"$dest/lds_direct_ab_multi_exec_parent" \
  "$dest/lds_direct_ab_phase_probe" "$CHUNKS" "$GRID_X" "$GRID_Y" "$MODE" \
  > "$dest/run.log" 2>&1
rc=$?
set -e
dmesg --ctime > "$dest/dmesg.after.txt" 2>&1 || true
sudo -n dmesg --ctime --since "$start_since" > "$dest/dmesg.since.txt" 2>&1 || true
echo "$rc" > "$dest/exit_code.txt"

capture_devcd() {
  local label="$1"
  {
    echo "label=$label"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "generic paths:"
    sudo -n find /sys/class/devcoredump -maxdepth 2 -print 2>/dev/null || true
    echo "drm paths:"
    sudo -n find /sys/class/drm/card0/device -maxdepth 4 -path '*devcoredump*' -print 2>/dev/null || true
  } > "$dest/coredumps/$label.paths.txt"

  if [ -r /sys/class/drm/card0/device/devcoredump/data ]; then
    sudo -n timeout 10s dd if=/sys/class/drm/card0/device/devcoredump/data \
      of="$dest/coredumps/$label.drm-devcoredump.data" bs=1M count=16 status=none || true
  fi

  for data in /sys/class/devcoredump/devcd*/data; do
    [ -e "$data" ] || continue
    dev="$(basename "$(dirname "$data")")"
    sudo -n sh -c "find -L '$(dirname "$data")' -maxdepth 1 -type f -not -name data -print -exec sh -c 'for f; do echo ===\$f===; cat \$f 2>/dev/null || true; done' sh {} +" \
      > "$dest/coredumps/$label.$dev.meta.txt" 2>&1 || true
    sudo -n timeout 10s dd if="$data" of="$dest/coredumps/$label.$dev.data" bs=1M count=16 status=none || true
  done
}

if [ "$rc" -ne 0 ]; then
  capture_devcd immediate
  waited=0
  while [ "$waited" -lt "$WAIT_DEVCD_MS" ]; do
    sleep 1
    waited=$((waited + 1000))
    if sudo -n find /sys/class/devcoredump -maxdepth 1 -name 'devcd*' -print -quit 2>/dev/null | grep -q .; then
      capture_devcd "late_${waited}ms"
      break
    fi
  done
fi

"$SUMMARY" "$OUT" "$OUT/direct-ab-artifact-summary" > "$dest/artifact-summary.log" 2>&1 || {
  cat "$dest/artifact-summary.log" >&2
  exit 1
}
cat "$dest/artifact-summary.log"

exit "$rc"
