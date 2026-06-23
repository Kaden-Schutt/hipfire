#!/usr/bin/env bash
# Cleanly stop, free the port, optionally restart `hipfire serve`.
# Usage: serve-restart.sh [port] [--kill-only] [-- <extra serve args>]
#        serve-restart.sh --socket <path> [--kill-only] [-- <extra serve args>]
set -uo pipefail
PORT=11435; KILL_ONLY=0; EXTRA=(); SOCK=""
while [ $# -gt 0 ]; do case "$1" in
  --kill-only) KILL_ONLY=1; shift;;
  --socket) SOCK="$2"; shift 2;;
  --) shift; EXTRA=("$@"); break;;
  *) PORT="$1"; shift;; esac; done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "$SOCK" ]; then
  echo "[serve-restart] killing socket serve $SOCK"
  MATCHED=0
  # pgrep -f treats its pattern as an ERE, so escape regex metacharacters in the
  # path (a literal `.` must not match any char) before matching.
  SOCK_RE=$(printf '%s' "$SOCK" | sed -E 's/[][(){}.^$*+?|\\]/\\&/g')
  for pat in "--socket-path $SOCK_RE" "--unix-socket $SOCK_RE"; do
    pids=$(pgrep -f -- "$pat" || true)
    for p in $pids; do kill -9 "$p" 2>/dev/null && MATCHED=$((MATCHED+1)); done
  done
  # The daemon is a per-box flock singleton; at most one exists and it belongs
  # to the serve we just killed. Reap it ONLY if we matched our serve, so a
  # no-op --socket of an unrelated path never kills a live TCP serve's daemon.
  if [ "$MATCHED" -gt 0 ]; then
    pkill -f "examples/daemon" 2>/dev/null || true
  fi
  # Only clear the global pidfiles when we actually killed THIS socket's serve.
  # If MATCHED==0 (e.g. --socket A while only a TCP serve is live), touch NEITHER:
  # blindly rm'ing would untrack the live TCP serve and unlink daemon.pid out from
  # under a live flock holder (the anti-pattern the GPU-lock rule warns against).
  if [ "$MATCHED" -gt 0 ]; then
    rm -f "$HOME/.hipfire/daemon.pid" "$HOME/.hipfire/serve.pid"
  fi
  # A Unix socket file is MEANT to be unlinked for recovery: bind() recreates it.
  # This is NOT the flock'd GPU lock (never rm that) — different rule.
  rm -f "$SOCK"
  echo "[serve-restart] clean"
  [ "$KILL_ONLY" = 1 ] && exit 0
  echo "[serve-restart] launching"
  rm -f ~/.hipfire/serve.log
  setsid bun "$ROOT/cli/index.ts" serve --socket-path "$SOCK" "${EXTRA[@]}" >~/.hipfire/serve.log 2>&1 & disown
  for i in $(seq 1 60); do grep -qiE "warm-up complete|listening on unix:|port in use|JSON Parse|FATAL" ~/.hipfire/serve.log && break; sleep 2; done
  tail -3 ~/.hipfire/serve.log
else
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
fi
