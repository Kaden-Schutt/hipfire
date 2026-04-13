#!/usr/bin/env bash
# hipfire-autoheal — gather baseline diagnostics + detect common failure modes.
# Emits structured sections that the interpret playbook maps to fixes.
#
# Usage: .skills/hipfire-autoheal/triage.sh [--json]
set -u

JSON=0
[ "${1:-}" = "--json" ] && JSON=1

run() { eval "$@" 2>/dev/null; }
exists() { command -v "$1" >/dev/null 2>&1; }
section() { [ "$JSON" = "1" ] && echo "{\"section\":\"$1\",\"content\":[" || echo "=== $1 ==="; }
section_end() { [ "$JSON" = "1" ] && echo "]},"; }
line() { [ "$JSON" = "1" ] && printf '"%s",\n' "$(echo "$1" | sed 's/"/\\"/g')" || echo "$1"; }

# ── 1. environment basics ──
section "environment"
line "$(uname -sr)"
line "shell: $SHELL"
line "PATH (hipfire-relevant): $(echo "$PATH" | tr ':' '\n' | grep -E 'rocm|hipfire|bun' | head -5 | tr '\n' ':')"
section_end

# ── 2. GPU stack ──
section "gpu_stack"
if [ -e /dev/kfd ]; then line "/dev/kfd: present"; else line "/dev/kfd: MISSING (install amdgpu-dkms)"; fi
if [ -e /dev/dri/renderD128 ]; then line "/dev/dri/renderD128: present"; else line "/dev/dri/renderD128: MISSING"; fi
for p in /opt/rocm/bin/hipcc /opt/rocm-6.*/bin/hipcc /opt/rocm-7.*/bin/hipcc; do
    if [ -x "$p" ]; then line "hipcc: $p"; break; fi
done
if exists rocminfo; then
    line "rocminfo: $(rocminfo 2>/dev/null | grep -E 'Name:\s*gfx' | head -1 | sed 's/^\s*//')"
fi
section_end

# ── 3. hipfire install ──
section "hipfire_install"
HIPFIRE_DIR="${HIPFIRE_DIR:-$HOME/.hipfire}"
[ -d "$HIPFIRE_DIR" ] && line "~/.hipfire: present" || line "~/.hipfire: MISSING (not installed)"
[ -x "$HIPFIRE_DIR/bin/daemon" ] && line "daemon: $(stat -c %y "$HIPFIRE_DIR/bin/daemon" 2>/dev/null | cut -d. -f1)" || line "daemon: MISSING"
[ -f "$HIPFIRE_DIR/cli/index.ts" ] && line "CLI: $(wc -l < "$HIPFIRE_DIR/cli/index.ts") lines"
[ -f "$HIPFIRE_DIR/config.json" ] && line "config: $(cat "$HIPFIRE_DIR/config.json" | tr -d '\n' | head -c 200)" || line "config: using defaults"
KBLOB_COUNT=$(find "$HIPFIRE_DIR/bin/kernels" -name '*.hsaco' 2>/dev/null | wc -l)
line "pre-compiled kernels: $KBLOB_COUNT blobs"
section_end

# ── 4. running state ──
section "running_state"
DAEMON_PIDS=$(pgrep -f 'bin/daemon' 2>/dev/null | tr '\n' ' ')
BUN_PIDS=$(pgrep -f 'cli/index.ts.*serve' 2>/dev/null | tr '\n' ' ')
line "daemon PIDs: ${DAEMON_PIDS:-none}"
line "bun serve PIDs: ${BUN_PIDS:-none}"
PID_FILE="$HIPFIRE_DIR/serve.pid"
if [ -f "$PID_FILE" ]; then
    PF_PID=$(cat "$PID_FILE")
    if kill -0 "$PF_PID" 2>/dev/null; then
        line "serve.pid: $PF_PID (alive)"
    else
        line "serve.pid: $PF_PID (STALE — process dead, rm this file)"
    fi
fi
if ss -tln 2>/dev/null | grep -q ':11435'; then
    line "port 11435: LISTENING"
    if curl -s --max-time 2 http://127.0.0.1:11435/health 2>/dev/null | grep -q '"status":"ok"'; then
        line "/health: responding OK"
    else
        line "/health: NOT responding"
    fi
else
    line "port 11435: free"
fi
section_end

# ── 5. recent errors from serve log ──
section "recent_errors"
LOG="$HIPFIRE_DIR/serve.log"
if [ -f "$LOG" ]; then
    line "log size: $(stat -c %s "$LOG") bytes"
    grep -Ei 'error|panic|failed|illegal memory|Kendall' "$LOG" 2>/dev/null | tail -5 | while read -r l; do line "$l"; done
else
    line "no serve.log"
fi
section_end

# ── 6. models ──
section "models"
if [ -d "$HIPFIRE_DIR/models" ]; then
    find "$HIPFIRE_DIR/models" -maxdepth 1 -type f \( -name '*.mq4' -o -name '*.mq6' -o -name '*.hf4' -o -name '*.hf6' \) 2>/dev/null | while read -r f; do
        line "$(basename "$f"): $(du -h "$f" | cut -f1)"
    done
fi
section_end

# ── 7. VRAM snapshot ──
section "vram"
if exists rocm-smi; then
    rocm-smi --showmeminfo vram 2>/dev/null | grep -iE 'vram|Total|Used' | head -6 | while read -r l; do line "$l"; done
else
    line "rocm-smi not installed"
fi
section_end

# ── 8. likely issue markers (one-line diagnoses) ──
section "likely_issues"
# Zombie serves
if [ -n "$DAEMON_PIDS" ] && [ "$(echo "$DAEMON_PIDS" | wc -w)" -gt 1 ]; then
    line "LIKELY: multiple daemon PIDs running (zombie serve contention — apply fix 1 from playbook)"
fi
# Stale pid file
if [ -f "$PID_FILE" ]; then
    PF_PID=$(cat "$PID_FILE")
    if ! kill -0 "$PF_PID" 2>/dev/null; then
        line "LIKELY: stale serve.pid (rm $PID_FILE and restart)"
    fi
fi
# Missing kernels
if [ "$KBLOB_COUNT" = "0" ]; then
    line "LIKELY: no pre-compiled kernels — first run will JIT every kernel (2-5 min on cold cache)"
fi
# hip_runtime.h issue (detect by looking for the error in recent log)
if grep -q "hip_runtime.h.*not found" "$LOG" 2>/dev/null; then
    line "LIKELY: ROCm include path not auto-injected — set HIPFIRE_HIPCC_EXTRA_FLAGS=-I/opt/rocm/include or hipfire update"
fi
# Multi-turn recall bug
if grep -q "Kendall" "$LOG" 2>/dev/null; then
    line "LIKELY: pre-0.1.5 givens4 KV — hipfire update + hipfire config set kv_cache asym3"
fi
section_end

if [ "$JSON" = "1" ]; then echo "{}"; fi
