#!/usr/bin/env bash
set -u
cd /home/bjoern/hipfire/.claude/worktrees/feature+device-mesh || exit 2
LOG=.agent-progress/phase0-cohgate.log; : > "$LOG"; exec >>"$LOG" 2>&1
echo "== phase0 coherence gate start $(date -Is) HEAD $(git rev-parse --short HEAD) =="
./scripts/coherence-gate.sh
echo "GATE exit: $?"
echo "== done $(date -Is) =="
