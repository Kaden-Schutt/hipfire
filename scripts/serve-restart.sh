#!/usr/bin/env bash
# Cleanly stop, free the port, optionally restart `hipfire serve`.
# Usage: serve-restart.sh [port] [--kill-only] [-- <extra serve args>]
set -uo pipefail
PORT=11435; KILL_ONLY=0; EXTRA=()
while [ $# -gt 0 ]; do case "$1" in
  --kill-only) KILL_ONLY=1; shift;;
  --) shift; EXTRA=("$@"); break;;
  *) PORT="$1"; shift;; esac; done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "[serve-restart] killing serve/daemon, freeing :$PORT"
for pat in "cli/index.ts serve" "examples/daemon" "bun.*serve"; do
  for p in $(pgrep -f "$pat"); do kill -9 "$p" 2>/dev/null; done; done
fuser -k "$PORT/tcp" 2>/dev/null
rm -f ~/.hipfire/daemon.pid ~/.hipfire/serve.pid
# NB: do NOT rm /tmp/hipfire-gpu.lock — it is an flock'd file; unlinking it
# breaks mutual exclusion (a new acquirer would lock a fresh inode). The
# kernel auto-releases the flock when the holder dies, so no cleanup needed.
for i in $(seq 1 10); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "[serve-restart] WARN port still busy"; exit 1; }
echo "[serve-restart] clean"; rocm-smi --showmeminfo vram 2>/dev/null | grep Used | head
[ "$KILL_ONLY" = 1 ] && exit 0
echo "[serve-restart] launching"
rm -f ~/.hipfire/serve.log
setsid bun "$ROOT/cli/index.ts" serve 0.0.0.0 "$PORT" "${EXTRA[@]}" >~/.hipfire/serve.log 2>&1 & disown
for i in $(seq 1 60); do grep -qiE "warm-up complete|port in use|JSON Parse|FATAL" ~/.hipfire/serve.log && break; sleep 2; done
tail -3 ~/.hipfire/serve.log
