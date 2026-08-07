#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.
#
# Hard memory gate for SP1 attention harnesses and benchmarks.
#
# WHY THIS EXISTS
# ---------------
# On 2026-08-07 the SP1 test/bench binaries drove NINE global OOM kills between
# 18:41 and 19:14 on starling. The victims were the user's applications —
# steamwebhelper x4, teams-for-linux x3, slack, a Firefox tab — not our
# benchmark. The benchmark itself reported success.
#
# That is the failure mode this script removes. On Strix Halo the GPU's GTT is
# system RAM and this box has NO SWAP, so an allocation overshoot does not
# degrade, it goes straight to the global OOM killer, which picks victims by
# oom_score rather than by who caused the problem.
#
# Running under a cgroup with MemoryMax means the kernel reclaims and kills
# INSIDE OUR SCOPE first. We lose our own run instead of the user losing their
# desktop session.
#
# Symptom to look for if this ever slips: video/desktop stutter and apps
# silently disappearing. Between runs the box looks perfectly healthy (~60 GiB
# free), so a live `free` will NOT show it. Diagnose with:
#     journalctl -k | grep -E 'page allocation failure|Out of memory|oom-kill'
#
# USAGE
#   scripts/run-bounded.sh <command> [args...]
#   HIPFIRE_MEM_CAP=16G scripts/run-bounded.sh cargo run --release ...
#
# The default cap is 24 GiB: comfortably under the 32 GB R9700 deployment
# target (so a run that fits here is plausible on target) while leaving this
# 125 GiB box's desktop untouched.
#
# LIMITATION, STATED HONESTLY: amdgpu GTT pages are allocated by the kernel
# driver on behalf of the process. Most are charged to the process memcg, but
# this is not guaranteed for every allocation path on every kernel. The cgroup
# is therefore a strong backstop, NOT a proof. Keep the in-process preflight
# (kv_slots::preflight_alloc) as the first line of defence — it refuses before
# allocating rather than dying part-way through.

set -uo pipefail

CAP="${HIPFIRE_MEM_CAP:-24G}"

if [ $# -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
avail_gib=$(awk -v k="$avail_kb" 'BEGIN{printf "%.1f", k/1048576}')

# Refuse to start if the box is already under pressure. Starting a multi-GiB
# run with little headroom is how the 19:14 burst happened.
min_gib="${HIPFIRE_MEM_MIN_AVAIL_GIB:-12}"
if awk -v a="$avail_gib" -v m="$min_gib" 'BEGIN{exit !(a < m)}'; then
  echo "run-bounded: REFUSING — MemAvailable ${avail_gib} GiB is below the ${min_gib} GiB floor." >&2
  echo "run-bounded: something else is using this box; wait or raise HIPFIRE_MEM_MIN_AVAIL_GIB." >&2
  exit 3
fi

echo "run-bounded: MemAvailable ${avail_gib} GiB, cap ${CAP}, swap disabled inside scope"
echo "run-bounded: $*"

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "run-bounded: WARNING — systemd-run unavailable, running UNGATED." >&2
  echo "run-bounded: a runaway allocation can OOM-kill the user's applications." >&2
  exec "$@"
fi

# --scope runs in the caller's context (keeps cwd, env, tty) but inside a fresh
# cgroup carrying the limits. MemorySwapMax=0 is belt-and-braces: this box has
# no swap, but if any is ever added we still want to fail fast rather than
# thrash.
systemd-run --user --scope --quiet \
  -p MemoryMax="$CAP" \
  -p MemorySwapMax=0 \
  -- "$@"
rc=$?

if [ $rc -ne 0 ]; then
  echo "run-bounded: command exited $rc" >&2
  # 137 = SIGKILL, the signature of the cgroup OOM killer.
  if [ $rc -eq 137 ]; then
    echo "run-bounded: exit 137 = SIGKILL — this run exceeded the ${CAP} cap and was" >&2
    echo "run-bounded: killed INSIDE its own cgroup. That is the gate working as designed:" >&2
    echo "run-bounded: the run died instead of the user's desktop. Shrink the configuration" >&2
    echo "run-bounded: (fewer slots, shorter context) rather than raising the cap." >&2
  fi
fi
exit $rc
