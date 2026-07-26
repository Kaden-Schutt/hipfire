#!/usr/bin/env bash
# Copyright (c) Kaden Schutt
# setup-gh-runner.sh — install a PERSISTENT, REBOOT-SAFE GitHub Actions self-hosted
# runner for the hipfire repo as a systemd service (Phase 0 of the Tier-3 GPU gate,
# docs/specs/2026-07-10-agentic-pr-merge-gate-design.md §16).
#
# The runner PROCESS is persistent (a systemd service that auto-starts on boot); the
# per-job PR CHECKOUT is ephemeral (the gpu-gates.yml gate job git-cleans it after
# each run). Idempotent: re-running stops/uninstalls the old service and reconfigures.
#
# Mint a registration token first (repo admin):
#   RUNNER_TOKEN="$(gh api -X POST repos/warpfront/hipfire/actions/runners/registration-token --jq .token)"
# Then, ON the target box:
#   RUNNER_TOKEN=<token> bash scripts/setup-gh-runner.sh <name> <labels> [work_dir]
# e.g.
#   RUNNER_TOKEN=... bash scripts/setup-gh-runner.sh hipx-gpu gfx1100,gfx1151
set -euo pipefail

REPO_URL="https://github.com/warpfront/hipfire"
NAME="${1:?runner name, e.g. hipx-gpu}"
LABELS="${2:?comma-separated labels, e.g. gfx1100,gfx1151}"
WORK="${3:-_work}"
DIR="${RUNNER_DIR:-$HOME/actions-runner}"
: "${RUNNER_TOKEN:?set RUNNER_TOKEN (gh api ... registration-token --jq .token)}"

mkdir -p "$DIR"; cd "$DIR"

if [ ! -x ./config.sh ]; then
  VER="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
         | grep -oP '"tag_name": "v\K[^"]+')"
  echo "==> downloading actions-runner v${VER}"
  curl -fsSL -o runner.tar.gz \
    "https://github.com/actions/runner/releases/download/v${VER}/actions-runner-linux-x64-${VER}.tar.gz"
  tar xzf runner.tar.gz && rm -f runner.tar.gz
fi

# Idempotent: tear down any prior service/registration for a clean reconfigure.
if [ -f .runner ]; then
  echo "==> existing runner found — stopping + reconfiguring"
  sudo ./svc.sh stop 2>/dev/null || true
  sudo ./svc.sh uninstall 2>/dev/null || true
  ./config.sh remove --token "$RUNNER_TOKEN" 2>/dev/null || true
fi

echo "==> configuring runner '${NAME}' (labels: ${LABELS})"
./config.sh --unattended --replace \
  --url "$REPO_URL" --token "$RUNNER_TOKEN" \
  --name "$NAME" --labels "$LABELS" --work "$WORK"

echo "==> installing as a reboot-safe systemd service"
sudo ./svc.sh install "$(whoami)"
sudo ./svc.sh start
sudo ./svc.sh status | head -5
echo "==> runner '${NAME}' (labels: ${LABELS}) is a systemd service — survives reboot."
