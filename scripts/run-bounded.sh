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
# THE CGROUP DOES NOT CONTAIN GPU MEMORY. MEASURED, 2026-08-07.
# ------------------------------------------------------------
# An earlier version of this header called the cgroup a "strong backstop" and
# treated the in-process preflight as a first line of defence. That was the
# wrong way round, and the box proved it: at 02:04:45 a gated run of
# test_batched_attn_slots invoked the GLOBAL oom-killer
# (constraint=CONSTRAINT_NONE, not a memcg OOM) and killed slack plus three
# steamwebhelper processes -- while running inside this cgroup.
#
# amdgpu GTT pages are allocated by the kernel driver on behalf of the process
# and are NOT reliably charged to the process memcg. So MemoryMax bounds our
# host-side RSS and does essentially nothing about the GPU allocation that
# actually exhausts the machine.
#
# What this script can therefore still do:
#   1. Bound host-side RSS (real, but not the thing that OOMs the box).
#   2. REFUSE TO START unless there is genuinely enough headroom -- which is now
#      the primary protection, because we cannot rely on being killed first.
#
# What actually protects the box:
#   - kv_slots::preflight_alloc, called with the TOTAL held live at once
#     (device AND host), refusing before allocating.
#   - Not starting when MemAvailable is close to what the run needs.
#   - GTT accounting, below. Since the kernel will not charge GTT to a memcg,
#     we account for it ourselves: read the driver's own counter before and
#     after, refuse to start when GTT is already high, and report any GTT this
#     run leaked. A leak is invisible to `ps` and to `free`'s per-process view,
#     so nothing else would surface it.
#
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
# The floor scales with the cap: a run allowed to reach CAP needs CAP available
# plus a margin for the desktop, because the cgroup will NOT stop its GPU
# allocation. A flat floor let a 24 GiB-capped run start with 19 GiB free, and
# it took out slack and three steamwebhelper processes.
cap_gib=$(awk -v c="$CAP" 'BEGIN{ if (c ~ /[Gg]$/) {sub(/[Gg]$/,"",c); print c+0} else if (c ~ /[Mm]$/) {sub(/[Mm]$/,"",c); print (c+0)/1024} else print (c+0)/1073741824 }')
margin_gib="${HIPFIRE_MEM_MARGIN_GIB:-10}"
need_gib=$(awk -v c="$cap_gib" -v m="$margin_gib" 'BEGIN{printf "%.1f", c+m}')
if awk -v a="$avail_gib" -v n="$need_gib" 'BEGIN{exit !(a < n)}'; then
  echo "run-bounded: REFUSING — MemAvailable ${avail_gib} GiB, but a ${CAP} run needs" >&2
  echo "run-bounded: ${need_gib} GiB (cap + ${margin_gib} GiB desktop margin)." >&2
  echo "run-bounded: the cgroup does NOT contain amdgpu GTT, so starting anyway risks" >&2
  echo "run-bounded: a GLOBAL OOM that kills the user's applications. Wait, shrink the" >&2
  echo "run-bounded: run with a smaller HIPFIRE_MEM_CAP, or free memory first." >&2
  exit 3
fi

# ── GTT accounting: the kernel will not do it for us ────────────────────────
# amdgpu allocates GTT (GPU memory backed by system RAM) through TTM, outside
# memcg entirely, and the DRM cgroup memory controller was never merged
# upstream. So MemoryMax cannot see it. We read the driver counter directly.
gtt_file=$(ls /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)
gtt_before_gib=0
if [ -n "$gtt_file" ] && [ -r "$gtt_file" ]; then
  gtt_before_gib=$(awk '{printf "%.2f", $1/1073741824}' "$gtt_file")
  gtt_ceiling_gib="${HIPFIRE_GTT_CEILING_GIB:-20}"
  if awk -v g="$gtt_before_gib" -v c="$gtt_ceiling_gib" 'BEGIN{exit !(g > c)}'; then
    echo "run-bounded: REFUSING — ${gtt_before_gib} GiB of GPU GTT is already held," >&2
    echo "run-bounded: above the ${gtt_ceiling_gib} GiB ceiling. GTT is invisible to ps and to" >&2
    echo "run-bounded: the cgroup, so this is the only place it gets checked. Something else" >&2
    echo "run-bounded: is holding GPU memory — a resident model is the usual cause." >&2
    echo "run-bounded: Free it, or raise HIPFIRE_GTT_CEILING_GIB deliberately." >&2
    exit 4
  fi
  echo "run-bounded: GTT in use before run: ${gtt_before_gib} GiB (ceiling ${gtt_ceiling_gib} GiB)"
fi

echo "run-bounded: MemAvailable ${avail_gib} GiB, cap ${CAP} (need ${need_gib} GiB), swap off in scope"
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

# Report any GTT this run failed to release. A leak here is invisible to ps and
# to free's per-process view, so if we do not say it, nothing will.
if [ -n "$gtt_file" ] && [ -r "$gtt_file" ]; then
  # Let the driver settle before reading. amdgpu frees GTT asynchronously after
  # the process exits, so sampling immediately reports a phantom leak -- this
  # check's very first real use flagged +4.80 GiB that had fully drained a few
  # seconds later. Poll until it stops falling rather than trusting one sample.
  gtt_after_gib=$(awk '{printf "%.2f", $1/1073741824}' "$gtt_file")
  for _ in 1 2 3 4 5 6; do
    sleep 0.5
    gtt_now_gib=$(awk '{printf "%.2f", $1/1073741824}' "$gtt_file")
    # Stop early once it is back at or below where we started.
    if awk -v n="$gtt_now_gib" -v b="$gtt_before_gib" 'BEGIN{exit !(n <= b + 0.05)}'; then
      gtt_after_gib="$gtt_now_gib"
      break
    fi
    gtt_after_gib="$gtt_now_gib"
  done
  leak_gib=$(awk -v a="$gtt_after_gib" -v b="$gtt_before_gib" 'BEGIN{printf "%.2f", a-b}')
  if awk -v l="$leak_gib" 'BEGIN{exit !(l > 0.5)}'; then
    echo "run-bounded: WARNING — GTT rose ${gtt_before_gib} -> ${gtt_after_gib} GiB (+${leak_gib} GiB)." >&2
    echo "run-bounded: this run did not release its GPU memory. It will not show up in ps." >&2
  else
    echo "run-bounded: GTT after run: ${gtt_after_gib} GiB (delta ${leak_gib} GiB)"
  fi
fi

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
