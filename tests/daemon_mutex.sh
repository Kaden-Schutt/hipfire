#!/usr/bin/env bash
# Verify only one hipfire daemon can run at a time:
#   1. First daemon acquires ~/.hipfire/daemon.pid flock and waits on stdin.
#   2. Second attempt reads the first's PID and exits with FATAL.
#   3. Killing the first releases the lock; a third attempt succeeds.
#
# Does not touch the GPU — the lock is checked BEFORE Gpu::init(), so the
# second daemon fails fast without allocating VRAM.

set -uo pipefail
DAEMON="target/release/examples/daemon"
if [[ ! -x "$DAEMON" ]]; then echo "missing $DAEMON"; exit 2; fi

# Start daemon A — keep stdin open so it stays alive waiting for JSON.
# Use exec 3<&0 / exec 3<&- wrappers so we control stdin lifecycle without
# bash closing it on the child.
LOG_A=$(mktemp)
mkfifo /tmp/hipfire_mutex_stdin_a_$$
"$DAEMON" < /tmp/hipfire_mutex_stdin_a_$$ > /dev/null 2> "$LOG_A" &
PID_A=$!
# Open the fifo writable so the daemon sees an open stdin.
exec 9> /tmp/hipfire_mutex_stdin_a_$$
# Wait for A to register its PID in the lockfile (up to ~5s).
PIDFILE="$HOME/.hipfire/daemon.pid"
for i in $(seq 1 50); do
  if [[ -s "$PIDFILE" ]]; then
    REGPID=$(cat "$PIDFILE" | tr -d '[:space:]')
    if [[ "$REGPID" == "$PID_A" ]]; then break; fi
  fi
  sleep 0.1
done
if [[ "${REGPID:-}" != "$PID_A" ]]; then
  echo "FAIL: daemon A (pid $PID_A) didn't register in $PIDFILE (got '${REGPID:-}')"
  kill -9 $PID_A 2>/dev/null
  exec 9>&-; rm -f /tmp/hipfire_mutex_stdin_a_$$ "$LOG_A"
  exit 1
fi
echo "daemon A running, pid $PID_A, registered in pidfile"

cleanup() {
  kill -9 $PID_A 2>/dev/null || true
  exec 9>&- 2>/dev/null || true
  rm -f /tmp/hipfire_mutex_stdin_a_$$ "$LOG_A" /tmp/hipfire_mutex_out_b_$$ /tmp/hipfire_mutex_out_c_$$
}
trap cleanup EXIT

fails=0

# Case 1: second attempt must fail with FATAL naming PID A.
"$DAEMON" </dev/null > /dev/null 2> /tmp/hipfire_mutex_out_b_$$
EC_B=$?
ERR_B=$(cat /tmp/hipfire_mutex_out_b_$$)
echo "--- second daemon stderr ---"
echo "$ERR_B"
if [[ $EC_B -ne 0 ]] && echo "$ERR_B" | grep -q "FATAL: hipfire daemon already running" && echo "$ERR_B" | grep -q "PID $PID_A"; then
  echo "PASS case 1 (second daemon rejected, names PID $PID_A)"
else
  echo "FAIL case 1 (exit=$EC_B, stderr above)"
  fails=$((fails+1))
fi

# Case 2: kill A, the lock releases, a third daemon starts fine (but we'll
# kill it immediately once it registers, to avoid unwanted GPU init time).
kill -9 $PID_A
exec 9>&- 2>/dev/null || true
sleep 0.2  # let kernel release flock
mkfifo /tmp/hipfire_mutex_stdin_c_$$
"$DAEMON" < /tmp/hipfire_mutex_stdin_c_$$ > /dev/null 2> /tmp/hipfire_mutex_out_c_$$ &
PID_C=$!
exec 8> /tmp/hipfire_mutex_stdin_c_$$
for i in $(seq 1 50); do
  if [[ -s "$PIDFILE" ]]; then
    REGPID_C=$(cat "$PIDFILE" | tr -d '[:space:]')
    if [[ "$REGPID_C" == "$PID_C" ]]; then break; fi
  fi
  sleep 0.1
done
if [[ "${REGPID_C:-}" == "$PID_C" ]]; then
  echo "PASS case 2 (third daemon acquires lock after first is killed, pid $PID_C)"
else
  echo "FAIL case 2 (third daemon didn't register; got '${REGPID_C:-}')"
  cat /tmp/hipfire_mutex_out_c_$$
  fails=$((fails+1))
fi
kill -9 $PID_C 2>/dev/null || true
exec 8>&- 2>/dev/null || true
rm -f /tmp/hipfire_mutex_stdin_c_$$

if [[ $fails -eq 0 ]]; then echo "ALL PASS"; exit 0; else echo "$fails FAILS"; exit 1; fi
