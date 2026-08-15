#!/usr/bin/env bash
# ngram-mtp-abba.sh — resumable ABBA runner for Qwen3.6 A3B MQ4R + MTP/ngram campaigns.
#
# Developer-only shell orchestration around scripts/serve_harness.py.
# Evidence defaults outside the repo ($HOME/ngram-mtp-evidence). Failed arms are
# preserved; the campaign continues through the full ORDER.
#
# Required env:
#   A_DAEMON B_DAEMON CLI_BIN HARNESS MODEL PROMPT REGISTRY
# Optional env:
#   DEVICE OUT_ROOT LABEL ORDER A_NGRAM B_NGRAM WARMUP ALLOW_NO_GPU_GUARD RESUME
#   KERNEL_CACHE KV SAMPLING THINKING MAX_TOKENS MAX_SEQ MTP_NGRAM_MATCH MTP_NGRAM_MIN
#   MTP_NGRAM_MAX PORT SERVE_WARM_TIMEOUT_SECS MODE HOST_TIMING
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: ngram-mtp-abba.sh [--help]

Resumable ABBA runner for gradual Qwen3.6 A3B MQ4R + MTP/ngram optimization.
Launches a fresh serve process per arm through scripts/serve_harness.py.

Required environment:
  A_DAEMON     executable daemon binary for arm A
  B_DAEMON     executable daemon binary for arm B
  CLI_BIN      executable hipfire CLI binary
  HARNESS      readable serve_harness.py path
  MODEL        readable model artifact (.mq4 / .mq4r / ...)
  PROMPT       readable UTF-8 prompt file (--prompt-file)
  REGISTRY     readable model registry JSON

Derived:
  MODEL_MTP    sibling of MODEL with extension replaced by .mtp (required readable)

Optional environment (defaults):
  DEVICE                  0
  OUT_ROOT                $HOME/ngram-mtp-evidence
  LABEL                   UTC timestamp YYYYMMDDTHHMMSSZ
  ORDER                   A B B A   (one or more exact A B B A blocks only)
  A_NGRAM / B_NGRAM       on   (per-arm --mtp-ngram on|off)
  WARMUP                  1    (one unrecorded warmup per distinct arm when 1)
  RESUME                  0    (1 = allow existing $OUT_ROOT/$LABEL under byte-identical contract.txt)
  ALLOW_NO_GPU_GUARD      unset; set to 1 to skip rocm-smi --showpids requirement
  KERNEL_CACHE            $REPO/.hipfire_kernels (created; sealed into arm env)
  KV                      q8
  SAMPLING                greedy
  THINKING                off
  MAX_TOKENS              256
  MAX_SEQ                 4096
  MTP_NGRAM_MATCH         24
  MTP_NGRAM_MIN           48
  MTP_NGRAM_MAX           64
  PORT                    11520
  SERVE_WARM_TIMEOUT_SECS 180
  MODE                    battery
  HOST_TIMING             0    (1 = seal HIPFIRE_HOST_TIMING=1 into arm env for per-window MTP host/API timings)

Output layout under $OUT_ROOT/$LABEL/:
  contract.txt, identity.txt, git-status.txt, git-diff.txt, ledger.jsonl,
  warmup-A/ warmup-B/, 01-A/ 02-B/ ..., output-parity.txt, summary.txt

Each arm directory preserves harness stdout/stderr, report.json, serve.log,
timestamps, topology snapshots, exit status, and report-validation.txt; recorded
arms also write arm.txt, command.txt, and process status. Fresh campaigns create
LABEL atomically and refuse if it already exists unless RESUME=1. Resume requires
byte-identical contract.txt, never overwrites an existing arm/warmup directory,
skips completed slots (exit_status + end.iso + report.json present — failed or
not), and refuses incomplete slots (use a new LABEL). Resume appends to
ledger.jsonl (touch, never truncate).

Exit nonzero if any arm failed, any report was invalid, warmup failed, ledger is
incomplete/malformed, contract drifted, ORDER is non-ABBA, or output parity failed.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  printf 'ngram-mtp-abba: unexpected argument %q (try --help)\n' "$1" >&2
  exit 2
fi

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO=$(cd "$SCRIPT_DIR/.." && pwd)

die() {
  printf 'ngram-mtp-abba: %s\n' "$*" >&2
  exit 2
}

require_file() {
  local path=$1 kind=$2
  [[ -n "$path" ]] || die "missing required $kind"
  [[ -e "$path" ]] || die "$kind not found: $path"
  [[ -r "$path" ]] || die "$kind not readable: $path"
}

require_exec() {
  local path=$1 kind=$2
  require_file "$path" "$kind"
  [[ -x "$path" ]] || die "$kind not executable: $path"
}

# --- contract defaults (env-overridable) ---
DEVICE=${DEVICE:-0}
OUT_ROOT=${OUT_ROOT:-"$HOME/ngram-mtp-evidence"}
LABEL=${LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}
ORDER=${ORDER:-"A B B A"}
A_NGRAM=${A_NGRAM:-on}
B_NGRAM=${B_NGRAM:-on}
WARMUP=${WARMUP:-1}
RESUME=${RESUME:-0}
KV=${KV:-q8}
SAMPLING=${SAMPLING:-greedy}
THINKING=${THINKING:-off}
MAX_TOKENS=${MAX_TOKENS:-256}
MAX_SEQ=${MAX_SEQ:-4096}
MTP_NGRAM_MATCH=${MTP_NGRAM_MATCH:-24}
MTP_NGRAM_MIN=${MTP_NGRAM_MIN:-48}
MTP_NGRAM_MAX=${MTP_NGRAM_MAX:-64}
PORT=${PORT:-11520}
SERVE_WARM_TIMEOUT_SECS=${SERVE_WARM_TIMEOUT_SECS:-180}
MODE=${MODE:-battery}
HOST_TIMING=${HOST_TIMING:-0}
ALLOW_NO_GPU_GUARD=${ALLOW_NO_GPU_GUARD:-}
KERNEL_CACHE=${KERNEL_CACHE:-"$REPO/.hipfire_kernels"}

A_DAEMON=${A_DAEMON:-}
B_DAEMON=${B_DAEMON:-}
CLI_BIN=${CLI_BIN:-}
HARNESS=${HARNESS:-}
MODEL=${MODEL:-}
PROMPT=${PROMPT:-}
REGISTRY=${REGISTRY:-}

require_exec "$A_DAEMON" A_DAEMON
require_exec "$B_DAEMON" B_DAEMON
require_exec "$CLI_BIN" CLI_BIN
require_file "$HARNESS" HARNESS
require_file "$MODEL" MODEL
require_file "$PROMPT" PROMPT
require_file "$REGISTRY" REGISTRY

# Sibling MTP sidecar: replace MODEL extension with .mtp
MODEL_BASE=${MODEL%.*}
if [[ "$MODEL_BASE" == "$MODEL" ]]; then
  MODEL_MTP="${MODEL}.mtp"
else
  MODEL_MTP="${MODEL_BASE}.mtp"
fi
require_file "$MODEL_MTP" MODEL_MTP

case "$A_NGRAM" in on|off) ;; *) die "A_NGRAM must be on|off, got: $A_NGRAM" ;; esac
case "$B_NGRAM" in on|off) ;; *) die "B_NGRAM must be on|off, got: $B_NGRAM" ;; esac
case "$WARMUP" in 0|1) ;; *) die "WARMUP must be 0|1, got: $WARMUP" ;; esac
case "$RESUME" in 0|1) ;; *) die "RESUME must be 0|1, got: $RESUME" ;; esac
case "$HOST_TIMING" in 0|1) ;; *) die "HOST_TIMING must be 0|1, got: $HOST_TIMING" ;; esac

# ORDER must be one or more exact A B B A blocks (length multiple of 4).
read -r -a ORDER_ARMS <<< "$ORDER"
[[ ${#ORDER_ARMS[@]} -gt 0 ]] || die "ORDER is empty"
[[ $(( ${#ORDER_ARMS[@]} % 4 )) -eq 0 ]] || die "ORDER length must be a multiple of 4 (exact A B B A blocks), got ${#ORDER_ARMS[@]} arms: $ORDER"
for ((i = 0; i < ${#ORDER_ARMS[@]}; i += 4)); do
  b0=${ORDER_ARMS[i]}
  b1=${ORDER_ARMS[i + 1]}
  b2=${ORDER_ARMS[i + 2]}
  b3=${ORDER_ARMS[i + 3]}
  if [[ "$b0" != "A" || "$b1" != "B" || "$b2" != "B" || "$b3" != "A" ]]; then
    die "ORDER must be one or more exact A B B A blocks; bad block at index $i: $b0 $b1 $b2 $b3 (ORDER=$ORDER)"
  fi
done

# Absolute paths for identity / hashing stability
A_DAEMON=$(readlink -f "$A_DAEMON")
B_DAEMON=$(readlink -f "$B_DAEMON")
CLI_BIN=$(readlink -f "$CLI_BIN")
HARNESS=$(readlink -f "$HARNESS")
MODEL=$(readlink -f "$MODEL")
MODEL_MTP=$(readlink -f "$MODEL_MTP")
PROMPT=$(readlink -f "$PROMPT")
REGISTRY=$(readlink -f "$REGISTRY")
OUT_ROOT=$(readlink -f "$(mkdir -p "$OUT_ROOT" && printf '%s' "$OUT_ROOT")")
mkdir -p "$KERNEL_CACHE"
KERNEL_CACHE=$(readlink -f "$KERNEL_CACHE")

hash_pair() {
  # prints: path\tmd5\tsha256
  local path=$1
  local md sha
  md=$(md5sum "$path" | awk '{print $1}')
  sha=$(sha256sum "$path" | awk '{print $1}')
  printf '%s\t%s\t%s\n' "$path" "$md" "$sha"
}

# Deterministic frozen contract (byte-compared on RESUME=1 before any campaign touch).
write_contract_to() {
  local dest=$1
  local git_head git_status_md5 git_status_sha git_diff_md5 git_diff_sha
  local tmp_status tmp_diff
  tmp_status=$(mktemp)
  tmp_diff=$(mktemp)
  if git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
    git -C "$REPO" status --porcelain=v1 -b >"$tmp_status" 2>&1 || true
    git -C "$REPO" diff --no-ext-diff HEAD >"$tmp_diff" 2>&1 || true
  else
    git_head=not-a-git-repo
    printf 'not a git work tree\n' >"$tmp_status"
    printf 'not a git work tree\n' >"$tmp_diff"
  fi
  git_status_md5=$(md5sum "$tmp_status" | awk '{print $1}')
  git_status_sha=$(sha256sum "$tmp_status" | awk '{print $1}')
  git_diff_md5=$(md5sum "$tmp_diff" | awk '{print $1}')
  git_diff_sha=$(sha256sum "$tmp_diff" | awk '{print $1}')

  {
    printf 'ORDER=%s\n' "$ORDER"
    printf 'DEVICE=%s\n' "$DEVICE"
    printf 'KERNEL_CACHE=%s\n' "$KERNEL_CACHE"
    printf 'KV=%s\n' "$KV"
    printf 'SAMPLING=%s\n' "$SAMPLING"
    printf 'THINKING=%s\n' "$THINKING"
    printf 'MAX_TOKENS=%s\n' "$MAX_TOKENS"
    printf 'MAX_SEQ=%s\n' "$MAX_SEQ"
    printf 'MODE=%s\n' "$MODE"
    printf 'MTP=on\n'
    printf 'MTP_NGRAM_MATCH=%s\n' "$MTP_NGRAM_MATCH"
    printf 'MTP_NGRAM_MIN=%s\n' "$MTP_NGRAM_MIN"
    printf 'MTP_NGRAM_MAX=%s\n' "$MTP_NGRAM_MAX"
    printf 'PORT=%s\n' "$PORT"
    printf 'SERVE_WARM_TIMEOUT_SECS=%s\n' "$SERVE_WARM_TIMEOUT_SECS"
    printf 'WARMUP=%s\n' "$WARMUP"
    printf 'HOST_TIMING=%s\n' "$HOST_TIMING"
    printf 'A_NGRAM=%s\n' "$A_NGRAM"
    printf 'B_NGRAM=%s\n' "$B_NGRAM"
    printf 'A_DAEMON=%s\n' "$A_DAEMON"
    printf 'B_DAEMON=%s\n' "$B_DAEMON"
    printf 'CLI_BIN=%s\n' "$CLI_BIN"
    printf 'HARNESS=%s\n' "$HARNESS"
    printf 'MODEL=%s\n' "$MODEL"
    printf 'MODEL_MTP=%s\n' "$MODEL_MTP"
    printf 'PROMPT=%s\n' "$PROMPT"
    printf 'REGISTRY=%s\n' "$REGISTRY"
    printf 'HOME=%s\n' "$HOME"
    printf 'PATH=%s\n' "$PATH"
    printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}"
    printf 'git_head=%s\n' "$git_head"
    printf 'git_status_md5=%s\n' "$git_status_md5"
    printf 'git_status_sha256=%s\n' "$git_status_sha"
    printf 'git_diff_md5=%s\n' "$git_diff_md5"
    printf 'git_diff_sha256=%s\n' "$git_diff_sha"
    printf 'hash_runner=%s\n' "$(hash_pair "$SCRIPT_PATH")"
    printf 'hash_a_daemon=%s\n' "$(hash_pair "$A_DAEMON")"
    printf 'hash_b_daemon=%s\n' "$(hash_pair "$B_DAEMON")"
    printf 'hash_cli=%s\n' "$(hash_pair "$CLI_BIN")"
    printf 'hash_harness=%s\n' "$(hash_pair "$HARNESS")"
    printf 'hash_model=%s\n' "$(hash_pair "$MODEL")"
    printf 'hash_model_mtp=%s\n' "$(hash_pair "$MODEL_MTP")"
    printf 'hash_prompt=%s\n' "$(hash_pair "$PROMPT")"
    printf 'hash_registry=%s\n' "$(hash_pair "$REGISTRY")"
  } >"$dest"

  # Stash git snapshots for fresh identity write (caller may move these).
  CONTRACT_GIT_STATUS_TMP=$tmp_status
  CONTRACT_GIT_DIFF_TMP=$tmp_diff
}

CONTRACT_TMP=$(mktemp)
write_contract_to "$CONTRACT_TMP"

CAMPAIGN="$OUT_ROOT/$LABEL"
CAMPAIGN_RESUMED=0
if mkdir "$CAMPAIGN" 2>/dev/null; then
  CAMPAIGN_RESUMED=0
  mv "$CONTRACT_TMP" "$CAMPAIGN/contract.txt"
elif [[ -d "$CAMPAIGN" ]]; then
  if [[ "$RESUME" != "1" ]]; then
    rm -f "$CONTRACT_TMP"
    [[ -n "${CONTRACT_GIT_STATUS_TMP:-}" ]] && rm -f "$CONTRACT_GIT_STATUS_TMP" "$CONTRACT_GIT_DIFF_TMP"
    die "campaign directory already exists (set RESUME=1 to continue): $CAMPAIGN"
  fi
  if [[ ! -f "$CAMPAIGN/contract.txt" ]]; then
    rm -f "$CONTRACT_TMP"
    [[ -n "${CONTRACT_GIT_STATUS_TMP:-}" ]] && rm -f "$CONTRACT_GIT_STATUS_TMP" "$CONTRACT_GIT_DIFF_TMP"
    die "RESUME=1 but missing contract.txt (cannot prove equivalent campaign): $CAMPAIGN/contract.txt"
  fi
  if ! cmp -s "$CONTRACT_TMP" "$CAMPAIGN/contract.txt"; then
    rm -f "$CONTRACT_TMP"
    [[ -n "${CONTRACT_GIT_STATUS_TMP:-}" ]] && rm -f "$CONTRACT_GIT_STATUS_TMP" "$CONTRACT_GIT_DIFF_TMP"
    die "RESUME=1 contract mismatch vs $CAMPAIGN/contract.txt (refusing before touching campaign artifacts; use a new LABEL)"
  fi
  rm -f "$CONTRACT_TMP"
  [[ -n "${CONTRACT_GIT_STATUS_TMP:-}" ]] && rm -f "$CONTRACT_GIT_STATUS_TMP" "$CONTRACT_GIT_DIFF_TMP"
  CAMPAIGN_RESUMED=1
  printf 'ngram-mtp-abba: resuming campaign %s (contract identical)\n' "$CAMPAIGN"
else
  rm -f "$CONTRACT_TMP"
  [[ -n "${CONTRACT_GIT_STATUS_TMP:-}" ]] && rm -f "$CONTRACT_GIT_STATUS_TMP" "$CONTRACT_GIT_DIFF_TMP"
  die "failed to create campaign directory: $CAMPAIGN"
fi

# One shared Python report validator used by warmups, parity, and summary.
# Written every invocation (deterministic code, not evidence); arms keep their own diagnostics.
REPORT_VALIDATE_PY="$CAMPAIGN/report_validate.py"
cat >"$REPORT_VALIDATE_PY" <<'PY'
"""Shared ABBA report measurement contract validator.

CLI:
  python3 report_validate.py <report.json> <expected_ngram on|off> <host_timing 0|1> [diag_out]
Exit 0 iff VALID.

Import:
  from report_validate import load_first, validate_first, validate_report_file
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

TIMING_KINDS = frozenset({"ngram", "mtp", "ar"})
TIMING_US_FIELDS = (
    "wall_us",
    "draft_lookup_us",
    "launch_us",
    "h2d_us",
    "d2h_us",
    "d2d_us",
    "memset_us",
    "stream_sync_us",
    "event_sync_us",
    "device_sync_us",
    "graph_launch_us",
)


def is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def is_int_like(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return math.isfinite(v) and v == int(v)
    return False


def is_nonneg_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def load_first(report_path: Path) -> Tuple[Optional[dict], List[str]]:
    """Return (first_object_or_None, errors)."""
    errors: List[str] = []
    if not report_path.is_file():
        return None, [f"report missing: {report_path}"]
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, [f"json load failed: {type(e).__name__}: {e}"]
    if isinstance(data, list):
        if not data:
            return None, ["report array is empty"]
        first = data[0]
    elif isinstance(data, dict):
        first = data
    else:
        return None, [f"report root must be object or non-empty array, got {type(data).__name__}"]
    if not isinstance(first, dict):
        return None, [f"first report element is not an object: {type(first).__name__}"]
    return first, errors


def validate_mtp_window_timings(first: dict) -> List[str]:
    """HOST_TIMING=1 contract for mtp_window_timings records."""
    errors: List[str] = []
    cycles = first.get("cycles", None)
    if not (isinstance(cycles, int) and not isinstance(cycles, bool) and cycles >= 1):
        errors.append(f"cycles must be integer >= 1 when host_timing=1, got {cycles!r}")
        cycles_n: Optional[int] = None
    else:
        cycles_n = cycles

    timings = first.get("mtp_window_timings", None)
    if not isinstance(timings, list) or len(timings) == 0:
        errors.append(
            f"mtp_window_timings must be a non-empty list when host_timing=1, got {type(timings).__name__ if timings is not None else None}"
        )
        return errors

    if cycles_n is not None and len(timings) != cycles_n:
        errors.append(
            f"mtp_window_timings length must equal cycles ({cycles_n}), got {len(timings)}"
        )

    kind_counts = {"ngram": 0, "mtp": 0, "ar": 0}
    for i, rec in enumerate(timings):
        if not isinstance(rec, dict):
            errors.append(f"mtp_window_timings[{i}] must be an object, got {type(rec).__name__}")
            continue
        kind = rec.get("kind", None)
        if kind not in TIMING_KINDS:
            errors.append(
                f"mtp_window_timings[{i}].kind must be exactly ngram|mtp|ar, got {kind!r}"
            )
        else:
            kind_counts[str(kind)] += 1
        for field in TIMING_US_FIELDS:
            v = rec.get(field, None)
            if not is_nonneg_int(v):
                errors.append(
                    f"mtp_window_timings[{i}].{field} must be integer >= 0 (bool rejected), got {v!r}"
                )

    for key in ("ngram_mod_windows", "mtp_windows", "ar_windows"):
        raw = first.get(key, None)
        if not is_nonneg_int(raw):
            errors.append(
                f"{key} must be integer >= 0 when host_timing=1 (for kind counts), got {raw!r}"
            )

    ngram_w = first.get("ngram_mod_windows", None)
    mtp_w = first.get("mtp_windows", None)
    ar_w = first.get("ar_windows", None)
    if is_nonneg_int(ngram_w) and kind_counts["ngram"] != int(ngram_w):
        errors.append(
            f"mtp_window_timings kind=ngram count {kind_counts['ngram']} != ngram_mod_windows {int(ngram_w)}"
        )
    if is_nonneg_int(mtp_w) and kind_counts["mtp"] != int(mtp_w):
        errors.append(
            f"mtp_window_timings kind=mtp count {kind_counts['mtp']} != mtp_windows {int(mtp_w)}"
        )
    if is_nonneg_int(ar_w) and kind_counts["ar"] != int(ar_w):
        errors.append(
            f"mtp_window_timings kind=ar count {kind_counts['ar']} != ar_windows {int(ar_w)}"
        )

    return errors


def validate_first(
    first: Any, expected_ngram: str, host_timing: int = 0
) -> Tuple[bool, List[str]]:
    """Validate first report object against assigned ngram route ('on'|'off').

    When host_timing=1, also require ordered mtp_window_timings telemetry.
    """
    errors: List[str] = []
    want_ngram = expected_ngram == "on"
    if not isinstance(first, dict):
        return False, ["first report element is not an object"]

    ac = first.get("assistant_content", None)
    if not isinstance(ac, str) or ac == "":
        errors.append("assistant_content must be a nonempty string")

    dts = first.get("decode_tok_s", None)
    if not is_finite_number(dts) or float(dts) <= 0:
        errors.append(f"decode_tok_s must be numeric finite > 0, got {dts!r}")

    tau = first.get("tau", None)
    if not is_finite_number(tau) or float(tau) <= 0:
        errors.append(f"tau must be numeric finite > 0, got {tau!r}")

    mtp = first.get("mtp", None)
    if mtp is not True:
        errors.append(f"mtp must be true, got {mtp!r}")

    # Route: mtp_ngram must match assigned A_NGRAM/B_NGRAM (on->True, off->not True).
    mtp_ngram = first.get("mtp_ngram", None)
    if want_ngram:
        if mtp_ngram is not True:
            errors.append(f"mtp_ngram must be true for ngram-on route, got {mtp_ngram!r}")
    else:
        if mtp_ngram is True:
            errors.append(f"mtp_ngram must not be true for ngram-off route, got {mtp_ngram!r}")

    empty = first.get("empty", None)
    if empty is not False:
        errors.append(f"empty must be false, got {empty!r}")

    if want_ngram:
        wins = first.get("ngram_mod_windows", None)
        drafts = first.get("ngram_mod_drafts", None)
        accepted = first.get("ngram_mod_accepted", None)
        rate = first.get("ngram_mod_accept_rate", None)
        if not is_int_like(wins) or int(wins) <= 0:
            errors.append(f"ngram_mod_windows must be integer > 0, got {wins!r}")
        if not is_int_like(drafts) or int(drafts) <= 0:
            errors.append(f"ngram_mod_drafts must be integer > 0, got {drafts!r}")
        if not is_int_like(accepted) or int(accepted) < 0:
            errors.append(f"ngram_mod_accepted must be integer >= 0, got {accepted!r}")
        if not is_finite_number(rate) or not (0.0 <= float(rate) <= 1.0):
            errors.append(f"ngram_mod_accept_rate must be numeric finite in [0,1], got {rate!r}")

    if host_timing == 1:
        errors.extend(validate_mtp_window_timings(first))

    # Runaway / finish=length remains valid for this fixture (do not reject).
    return (not errors), errors


def validate_report_file(
    report_path: Path, expected_ngram: str, host_timing: int = 0
) -> Tuple[bool, List[str], Optional[dict]]:
    first, load_errs = load_first(report_path)
    if load_errs:
        return False, load_errs, None
    ok, errs = validate_first(first, expected_ngram, host_timing=host_timing)
    return ok, errs, first


def _timing_diag_lines(first: Optional[dict], host_timing: int) -> List[str]:
    lines = [f"host_timing={host_timing}"]
    if host_timing != 1 or not isinstance(first, dict):
        return lines
    timings = first.get("mtp_window_timings", None)
    if isinstance(timings, list):
        lines.append(f"mtp_window_timings_len={len(timings)}")
        kind_counts = {"ngram": 0, "mtp": 0, "ar": 0}
        for rec in timings:
            if isinstance(rec, dict):
                k = rec.get("kind")
                if k in kind_counts:
                    kind_counts[str(k)] += 1
        lines.append(f"kind_ngram={kind_counts['ngram']}")
        lines.append(f"kind_mtp={kind_counts['mtp']}")
        lines.append(f"kind_ar={kind_counts['ar']}")
    else:
        lines.append(f"mtp_window_timings_len=missing")
    lines.append(f"cycles={first.get('cycles')}")
    lines.append(f"ngram_mod_windows={first.get('ngram_mod_windows')}")
    lines.append(f"mtp_windows={first.get('mtp_windows')}")
    lines.append(f"ar_windows={first.get('ar_windows')}")
    return lines


def format_diag(
    ok: bool,
    expected_ngram: str,
    errs: List[str],
    first: Optional[dict],
    host_timing: int = 0,
) -> str:
    lines: List[str] = []
    if not ok:
        lines.append("VALID=0")
        lines.extend(_timing_diag_lines(first, host_timing))
        for e in errs:
            lines.append(f"error: {e}")
        return "\n".join(lines) + "\n"
    assert first is not None
    want_ngram = expected_ngram == "on"
    lines.append("VALID=1")
    lines.append(f"expected_ngram={expected_ngram}")
    lines.extend(_timing_diag_lines(first, host_timing))
    lines.append(f"assistant_content_len={len(first.get('assistant_content', ''))}")
    lines.append(f"decode_tok_s={first.get('decode_tok_s')}")
    lines.append(f"tau={first.get('tau')}")
    lines.append(f"mtp={first.get('mtp')}")
    lines.append(f"mtp_ngram={first.get('mtp_ngram')}")
    lines.append(f"empty={first.get('empty')}")
    if want_ngram:
        lines.append(f"ngram_mod_windows={first.get('ngram_mod_windows')}")
        lines.append(f"ngram_mod_drafts={first.get('ngram_mod_drafts')}")
        lines.append(f"ngram_mod_accepted={first.get('ngram_mod_accepted')}")
        lines.append(f"ngram_mod_accept_rate={first.get('ngram_mod_accept_rate')}")
    return "\n".join(lines) + "\n"


def _parse_host_timing(raw: str) -> int:
    if raw == "1":
        return 1
    if raw == "0":
        return 0
    raise ValueError(f"host_timing must be 0|1, got {raw!r}")


def main(argv: List[str]) -> int:
    if len(argv) < 4:
        print(
            "usage: report_validate.py <report.json> <on|off> <host_timing 0|1> [diag_out]",
            file=sys.stderr,
        )
        return 2
    report_path = Path(argv[1])
    expected = argv[2]
    try:
        host_timing = _parse_host_timing(argv[3])
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    diag_out = Path(argv[4]) if len(argv) > 4 else None
    ok, errs, first = validate_report_file(report_path, expected, host_timing=host_timing)
    text = format_diag(ok, expected, errs, first, host_timing=host_timing)
    if diag_out is not None:
        diag_out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY


daemon_for_arm() {
  case "$1" in
    A) printf '%s' "$A_DAEMON" ;;
    B) printf '%s' "$B_DAEMON" ;;
    *) die "bad arm: $1" ;;
  esac
}

ngram_for_arm() {
  case "$1" in
    A) printf '%s' "$A_NGRAM" ;;
    B) printf '%s' "$B_NGRAM" ;;
    *) die "bad arm: $1" ;;
  esac
}

# Shared report validator wrapper around $REPORT_VALIDATE_PY.
# Usage: validate_report <report.json> <expected_ngram on|off> [diag_out]
# Threads campaign HOST_TIMING (0|1) into the Python validator.
validate_report() {
  local report=$1 expected_ngram=$2 diag_out=${3:-}
  if [[ -n "$diag_out" ]]; then
    python3 "$REPORT_VALIDATE_PY" "$report" "$expected_ngram" "$HOST_TIMING" "$diag_out"
  else
    python3 "$REPORT_VALIDATE_PY" "$report" "$expected_ngram" "$HOST_TIMING" >/dev/null
  fi
}

# --- identity + git snapshots (fresh only; resume preserves original identity) ---
if [[ "$CAMPAIGN_RESUMED" -eq 0 ]]; then
  {
    printf 'campaign=%s\n' "$LABEL"
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'uname=%s\n' "$(uname -a)"
    printf 'repo=%s\n' "$REPO"
    printf 'runner=%s\n' "$SCRIPT_PATH"
    printf 'kernel_cache=%s\n' "$KERNEL_CACHE"
    printf 'resume=%s\n' "$RESUME"
    if git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      printf 'git_head=%s\n' "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
      printf 'git_describe=%s\n' "$(git -C "$REPO" describe --always --dirty 2>/dev/null || echo unknown)"
    else
      printf 'git_head=not-a-git-repo\n'
    fi
    printf '\n=== frozen contract (also contract.txt) ===\n'
    cat "$CAMPAIGN/contract.txt"
  } >"$CAMPAIGN/identity.txt"

  if [[ -n "${CONTRACT_GIT_STATUS_TMP:-}" && -f "${CONTRACT_GIT_STATUS_TMP:-}" ]]; then
    mv "$CONTRACT_GIT_STATUS_TMP" "$CAMPAIGN/git-status.txt"
    mv "$CONTRACT_GIT_DIFF_TMP" "$CAMPAIGN/git-diff.txt"
    {
      printf '\n=== git status (also git-status.txt) ===\n'
      cat "$CAMPAIGN/git-status.txt"
      printf '\n=== git diff HEAD (also git-diff.txt) ===\n'
      cat "$CAMPAIGN/git-diff.txt"
    } >>"$CAMPAIGN/identity.txt"
  else
    printf 'not a git work tree\n' >"$CAMPAIGN/git-status.txt"
    printf 'not a git work tree\n' >"$CAMPAIGN/git-diff.txt"
  fi

  : >"$CAMPAIGN/ledger.jsonl"
else
  {
    printf '\n=== resume %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'RESUME=1 contract=identical kernel_cache=%s\n' "$KERNEL_CACHE"
  } >>"$CAMPAIGN/identity.txt"
  touch "$CAMPAIGN/ledger.jsonl"
fi

require_gpu_clean() {
  local tag=$1
  if ! command -v rocm-smi >/dev/null 2>&1; then
    if [[ "$ALLOW_NO_GPU_GUARD" == "1" ]]; then
      printf 'ngram-mtp-abba: rocm-smi unavailable; ALLOW_NO_GPU_GUARD=1 continuing (%s)\n' "$tag" >&2
      return 0
    fi
    die "rocm-smi unavailable; set ALLOW_NO_GPU_GUARD=1 to override ($tag)"
  fi
  local attempt kfd_status
  for attempt in $(seq 1 90); do
    kfd_status=$(rocm-smi --showpids 2>&1 || true)
    if printf '%s\n' "$kfd_status" | grep -q "No KFD PIDs currently running"; then
      return 0
    fi
    sleep 2
  done
  printf 'ngram-mtp-abba: refusing %s: GPU not clean (KFD PIDs present)\n' "$tag" >&2
  rocm-smi --showpids >&2 || true
  exit 3
}

snapshot_topology() {
  local dest=$1
  {
    printf '=== rocm-smi ===\n'
    if command -v rocm-smi >/dev/null 2>&1; then
      rocm-smi --showproductname --showbus --showmeminfo vram --showclocks \
        --showpower --showtemp --showuse --showpids 2>&1 || true
    else
      printf 'rocm-smi unavailable\n'
    fi
    printf '\n=== amd-smi ===\n'
    if command -v amd-smi >/dev/null 2>&1; then
      amd-smi metric --gpu "$DEVICE" 2>&1 || amd-smi 2>&1 || true
    else
      printf 'amd-smi unavailable\n'
    fi
  } >"$dest"
}

build_harness_cmd() {
  # Sets global array HARNESS_CMD
  local ngram=$1 out_json=$2 serve_log=$3 home_dir=$4
  HARNESS_CMD=(
    python3 "$HARNESS"
    --model "$MODEL"
    --registry "$REGISTRY"
    --kv "$KV"
    --mtp on
    --mtp-ngram "$ngram"
    --mtp-ngram-match "$MTP_NGRAM_MATCH"
    --mtp-ngram-min "$MTP_NGRAM_MIN"
    --mtp-ngram-max "$MTP_NGRAM_MAX"
    --sampling "$SAMPLING"
    --thinking "$THINKING"
    --max-tokens "$MAX_TOKENS"
    --max-seq "$MAX_SEQ"
    --mode "$MODE"
    --prompt-file "$PROMPT"
    --port "$PORT"
    --home "$home_dir"
    --serve-log "$serve_log"
    --serve-warm-timeout-secs "$SERVE_WARM_TIMEOUT_SECS"
    --out "$out_json"
  )
}

# Slot is completed iff attempt finished with exit_status, end timestamp, and report
# (failed attempts count as completed — never overwrite or retry under same LABEL).
arm_completed() {
  local dir=$1
  [[ -f "$dir/exit_status" && -f "$dir/report.json" && -f "$dir/end.iso" ]]
}

# Directory exists but is not a completed attempt — resume must refuse (new LABEL).
arm_dir_incomplete() {
  local dir=$1
  [[ -e "$dir" ]] && ! arm_completed "$dir"
}

run_arm() {
  # $1=dir  $2=arm  $3=kind(warmup|recorded)  $4=run_index(or -)
  local dir=$1 arm=$2 kind=$3 run_index=${4:--}
  local daemon ngram rc
  daemon=$(daemon_for_arm "$arm")
  ngram=$(ngram_for_arm "$arm")

  # Never overwrite an existing arm/warmup directory (resume or accidental collision).
  if [[ -e "$dir" ]]; then
    die "refusing to overwrite existing arm directory (use a new LABEL): $dir"
  fi
  mkdir -p "$dir"

  require_gpu_clean "${kind}-${arm}-${run_index}"

  local out_json="$dir/report.json"
  local serve_log="$dir/serve.log"
  local home_dir="$dir/serve_home"
  local stdout_f="$dir/harness.stdout"
  local stderr_f="$dir/harness.stderr"
  mkdir -p "$home_dir"

  build_harness_cmd "$ngram" "$out_json" "$serve_log" "$home_dir"

  if [[ "$kind" == "recorded" ]]; then
    printf '%s\n' "$arm" >"$dir/arm.txt"
    {
      printf 'env=-i (sealed)\n'
      printf 'HOME=%s\n' "$HOME"
      printf 'PATH=%s\n' "$PATH"
      if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
      fi
      printf 'HIP_VISIBLE_DEVICES=%s\n' "$DEVICE"
      printf 'HIPFIRE_DAEMON_BIN=%s\n' "$daemon"
      printf 'HIPFIRE_CLI_BIN=%s\n' "$CLI_BIN"
      printf 'HIPFIRE_KERNEL_CACHE=%s\n' "$KERNEL_CACHE"
      if [[ "$HOST_TIMING" == "1" ]]; then
        printf 'HIPFIRE_HOST_TIMING=1\n'
      fi
      printf 'cwd=%s\n' "$REPO"
      printf 'kind=%s run_index=%s mtp_ngram=%s\n' "$kind" "$run_index" "$ngram"
      printf 'argv:'
      printf ' %q' "${HARNESS_CMD[@]}"
      printf '\n'
    } >"$dir/command.txt"
  fi

  date -u +%Y-%m-%dT%H:%M:%SZ >"$dir/start.iso"
  snapshot_topology "$dir/topology.before.txt"
  if [[ "$kind" == "recorded" ]]; then
    ps -ef >"$dir/ps.before.txt" 2>&1 || true
  fi

  # Sealed launch: no ambient HIPFIRE_* leakage into ABBA identity.
  # Capture exit status without aborting the campaign (no set -e).
  local -a sealed_env
  sealed_env=(
    env -i
    "HOME=$HOME"
    "PATH=$PATH"
    "HIP_VISIBLE_DEVICES=$DEVICE"
    "HIPFIRE_DAEMON_BIN=$daemon"
    "HIPFIRE_CLI_BIN=$CLI_BIN"
    "HIPFIRE_KERNEL_CACHE=$KERNEL_CACHE"
  )
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    sealed_env+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
  fi
  if [[ "$HOST_TIMING" == "1" ]]; then
    sealed_env+=("HIPFIRE_HOST_TIMING=1")
  fi
  (
    cd "$REPO" || exit 90
    "${sealed_env[@]}" "${HARNESS_CMD[@]}"
  ) >"$stdout_f" 2>"$stderr_f"
  rc=$?

  printf '%s\n' "$rc" >"$dir/exit_status"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$dir/end.iso"
  snapshot_topology "$dir/topology.after.txt"

  # Always validate report and preserve diagnostics (even on nonzero exit).
  validate_report "$out_json" "$ngram" "$dir/report-validation.txt" || true
  if [[ -f "$dir/report-validation.txt" ]] && grep -q '^VALID=1$' "$dir/report-validation.txt"; then
    printf 'valid=1\n' >"$dir/report-valid"
  else
    printf 'valid=0\n' >"$dir/report-valid"
  fi

  if [[ "$kind" == "recorded" ]]; then
    ps -ef >"$dir/ps.after.txt" 2>&1 || true
    {
      printf 'exit_status=%s\n' "$rc"
      printf 'start=%s\n' "$(cat "$dir/start.iso")"
      printf 'end=%s\n' "$(cat "$dir/end.iso")"
      if [[ -f "$out_json" ]]; then
        printf 'report=present\n'
      else
        printf 'report=missing\n'
      fi
      if [[ -f "$dir/report-valid" ]]; then
        cat "$dir/report-valid"
      fi
    } >"$dir/status.txt"
  fi

  return 0
}

append_ledger() {
  # $1=dir $2=run_index $3=arm
  # On failure: create ledger-error marker and return nonzero (caller continues campaign).
  local dir=$1 run_index=$2 arm=$3
  local daemon rc start end daemon_md5 daemon_sha ngram
  daemon=$(daemon_for_arm "$arm")
  ngram=$(ngram_for_arm "$arm")
  rc=$(cat "$dir/exit_status" 2>/dev/null || echo missing)
  start=$(cat "$dir/start.iso" 2>/dev/null || echo "")
  end=$(cat "$dir/end.iso" 2>/dev/null || echo "")
  daemon_md5=$(md5sum "$daemon" | awk '{print $1}')
  daemon_sha=$(sha256sum "$daemon" | awk '{print $1}')
  local report="$dir/report.json"
  if ! python3 - "$CAMPAIGN/ledger.jsonl" "$LABEL" "$run_index" "$arm" \
    "$daemon_md5" "$daemon_sha" "$rc" "$start" "$end" "$report" "$ngram" <<'PY'
import json, sys
ledger, campaign, run_index, arm, d_md5, d_sha, rc, start, end, report, ngram = sys.argv[1:]
entry = {
    "campaign": campaign,
    "run_index": int(run_index),
    "arm": arm,
    "expected_ngram": ngram,
    "daemon_md5": d_md5,
    "daemon_sha256": d_sha,
    "exit_status": int(rc) if str(rc).lstrip("-").isdigit() else rc,
    "start": start,
    "end": end,
}
try:
    with open(report, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            entry["metrics"] = first
        else:
            entry["parse_error"] = "first report element is not an object"
    elif isinstance(data, dict):
        entry["metrics"] = data
    else:
        entry["parse_error"] = "report is empty or not a JSON object/array"
except Exception as e:
    entry["parse_error"] = f"{type(e).__name__}: {e}"
with open(ledger, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, sort_keys=True) + "\n")
PY
  then
    {
      printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'run_index=%s arm=%s dir=%s\n' "$run_index" "$arm" "$dir"
      printf 'error=append_ledger python failed\n'
    } >>"$CAMPAIGN/ledger-error"
    printf 'ngram-mtp-abba: append_ledger failed for run_index=%s arm=%s (ledger-error written)\n' \
      "$run_index" "$arm" >&2
    return 1
  fi
  return 0
}

WARMUP_FAILED=0

# --- warmups (one per distinct arm, unrecorded but preserved) ---
declare -A WARMED=()
if [[ "$WARMUP" == "1" ]]; then
  for arm in "${ORDER_ARMS[@]}"; do
    if [[ -n "${WARMED[$arm]:-}" ]]; then
      continue
    fi
    WARMED[$arm]=1
    wdir="$CAMPAIGN/warmup-$arm"
    ngram=$(ngram_for_arm "$arm")
    if arm_completed "$wdir"; then
      # Skip only after preserved validation status is considered.
      if [[ -f "$wdir/report-validation.txt" ]] && grep -q '^VALID=1$' "$wdir/report-validation.txt"; then
        printf 'ngram-mtp-abba: skip completed warmup-%s (valid)\n' "$arm"
      else
        # Re-validate preserved report if validation artifact missing/stale.
        if [[ ! -f "$wdir/report-validation.txt" ]]; then
          validate_report "$wdir/report.json" "$ngram" "$wdir/report-validation.txt" || true
        fi
        if grep -q '^VALID=1$' "$wdir/report-validation.txt" 2>/dev/null; then
          printf 'ngram-mtp-abba: skip completed warmup-%s (valid after check)\n' "$arm"
        else
          printf 'ngram-mtp-abba: completed warmup-%s has invalid report; marking warmup-failure\n' "$arm" >&2
          WARMUP_FAILED=1
          {
            printf 'arm=%s\n' "$arm"
            printf 'reason=invalid_or_failed_warmup_report\n'
            printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
          } >>"$CAMPAIGN/warmup-failure"
        fi
      fi
      continue
    fi
    if arm_dir_incomplete "$wdir"; then
      die "incomplete warmup directory exists (refusing resume overwrite; use a new LABEL): $wdir"
    fi
    printf 'ngram-mtp-abba: warmup-%s\n' "$arm"
    run_arm "$wdir" "$arm" warmup -
    # Warmup failure/invalid report does not erase evidence; continue and force FAIL.
    wrc=$(cat "$wdir/exit_status" 2>/dev/null || echo missing)
    wvalid=0
    if [[ -f "$wdir/report-validation.txt" ]] && grep -q '^VALID=1$' "$wdir/report-validation.txt"; then
      wvalid=1
    fi
    if [[ "$wrc" != "0" || "$wvalid" -ne 1 ]]; then
      WARMUP_FAILED=1
      {
        printf 'arm=%s\n' "$arm"
        printf 'exit_status=%s\n' "$wrc"
        printf 'report_valid=%s\n' "$wvalid"
        printf 'reason=warmup_failed_or_invalid_report\n'
        printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      } >>"$CAMPAIGN/warmup-failure"
      printf 'ngram-mtp-abba: warmup-%s failed or invalid (continuing; OVERALL will FAIL)\n' "$arm" >&2
    fi
  done
fi

# --- recorded ORDER ---
run_i=0
for arm in "${ORDER_ARMS[@]}"; do
  run_i=$((run_i + 1))
  rdir=$(printf '%s/%02d-%s' "$CAMPAIGN" "$run_i" "$arm")
  if arm_completed "$rdir"; then
    printf 'ngram-mtp-abba: skip completed %02d-%s\n' "$run_i" "$arm"
    # Ensure validation artifact exists for summary/parity.
    ngram=$(ngram_for_arm "$arm")
    if [[ ! -f "$rdir/report-validation.txt" ]]; then
      validate_report "$rdir/report.json" "$ngram" "$rdir/report-validation.txt" || true
    fi
    # Ensure ledger has a row if resuming mid-campaign without one
    if ! grep -q "\"run_index\": ${run_i}[,}]" "$CAMPAIGN/ledger.jsonl" 2>/dev/null; then
      append_ledger "$rdir" "$run_i" "$arm" || true
    fi
    continue
  fi
  if arm_dir_incomplete "$rdir"; then
    die "incomplete arm directory exists (refusing resume overwrite; use a new LABEL): $rdir"
  fi
  printf 'ngram-mtp-abba: recorded %02d-%s\n' "$run_i" "$arm"
  run_arm "$rdir" "$arm" recorded "$run_i"
  append_ledger "$rdir" "$run_i" "$arm" || true
done

# Expected ORDER as space-separated for Python.
ORDER_STR="${ORDER_ARMS[*]}"
A_NGRAM_EXP=$A_NGRAM
B_NGRAM_EXP=$B_NGRAM

# --- output parity (assistant_content only across successful validated recorded reports) ---
python3 - "$CAMPAIGN" "$ORDER_STR" "$A_NGRAM_EXP" "$B_NGRAM_EXP" "$HOST_TIMING" <<'PY' >"$CAMPAIGN/output-parity.txt"
import hashlib, sys
from pathlib import Path

campaign = Path(sys.argv[1])
sys.path.insert(0, str(campaign))
from report_validate import validate_report_file  # noqa: E402

order = sys.argv[2].split()
a_ngram = sys.argv[3]
b_ngram = sys.argv[4]
host_timing = 1 if sys.argv[5] == "1" else 0

def expected_ngram(arm: str) -> str:
    return a_ngram if arm == "A" else b_ngram

contents = []  # (run_name, exit_status, ok, md5, err)
for idx, arm in enumerate(order, start=1):
    name = f"{idx:02d}-{arm}"
    run = campaign / name
    try:
        rc = int((run / "exit_status").read_text().strip())
    except Exception:
        rc = None
    report = run / "report.json"
    ngram = expected_ngram(arm)
    if rc != 0 or not report.is_file():
        contents.append((name, rc, False, None, "arm_failed_or_missing_report"))
        continue
    ok, errs, first = validate_report_file(report, ngram, host_timing=host_timing)
    if not ok or first is None:
        contents.append((name, rc, False, None, "; ".join(errs) if errs else "invalid_report"))
        continue
    text = first["assistant_content"]
    md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
    contents.append((name, rc, True, md5, None))

print(f"campaign={campaign.name}")
print(f"recorded_runs={len(contents)}")
print(f"expected_order={' '.join(order)}")
success = [c for c in contents if c[2]]
failed = [c for c in contents if not c[2]]
for name, rc, ok, md5, err in contents:
    if ok:
        print(f"{name}\texit={rc}\tassistant_content_md5={md5}\tOK")
    else:
        print(f"{name}\texit={rc}\tFAIL\t{err}")

parity = "PASS"
if failed:
    parity = "FAIL"
    print(f"parity=FAIL reason=missing_or_invalid_reports n_failed={len(failed)}")
elif not success:
    parity = "FAIL"
    print("parity=FAIL reason=no_successful_reports")
else:
    ref = success[0][3]
    mismatch = [c for c in success if c[3] != ref]
    if mismatch:
        parity = "FAIL"
        print(f"parity=FAIL reason=assistant_content_md5_mismatch ref={ref}")
        for c in mismatch:
            print(f"  mismatch {c[0]} md5={c[3]}")
    else:
        print(f"parity=PASS assistant_content_md5={ref} n_success={len(success)}")

if failed:
    print(f"failed_or_invalid_arms={len(failed)}")
print(f"OUTPUT_PARITY={parity}")
sys.exit(0)
PY

# --- summary ---
python3 - "$CAMPAIGN" "$ORDER_STR" "$A_NGRAM_EXP" "$B_NGRAM_EXP" "$WARMUP_FAILED" "$HOST_TIMING" <<'PY' >"$CAMPAIGN/summary.txt"
import json, sys
from collections import Counter
from pathlib import Path

campaign = Path(sys.argv[1])
sys.path.insert(0, str(campaign))
from report_validate import validate_first, validate_report_file  # noqa: E402

order = sys.argv[2].split()
a_ngram = sys.argv[3]
b_ngram = sys.argv[4]
warmup_failed_flag = sys.argv[5] == "1"
host_timing = 1 if sys.argv[6] == "1" else 0

def expected_ngram(arm: str) -> str:
    return a_ngram if arm == "A" else b_ngram

# ABBA ORDER check
order_ok = len(order) > 0 and len(order) % 4 == 0
if order_ok:
    for i in range(0, len(order), 4):
        if order[i : i + 4] != ["A", "B", "B", "A"]:
            order_ok = False
            break

parity_txt = (campaign / "output-parity.txt").read_text(encoding="utf-8", errors="replace")
parity_pass = "OUTPUT_PARITY=PASS" in parity_txt.splitlines()

warmup_failure = warmup_failed_flag or (campaign / "warmup-failure").is_file()
ledger_error = (campaign / "ledger-error").is_file()

print(f"campaign={campaign.name}")
print(f"path={campaign}")
print(f"order={' '.join(order)}")
print(f"order_abba_ok={order_ok}")
print("--- runs ---")
any_arm_fail = False
any_report_invalid = False
any_route_mismatch = False

for idx, arm in enumerate(order, start=1):
    name = f"{idx:02d}-{arm}"
    run = campaign / name
    ngram = expected_ngram(arm)
    want_on = ngram == "on"
    try:
        rc = int((run / "exit_status").read_text().strip())
    except Exception:
        rc = "missing"
        any_arm_fail = True
    decode = tau = ngram_s = "n/a"
    report_ok = False
    report = run / "report.json"
    # Prefer preserved validation artifact when present; still re-check file.
    val_path = run / "report-validation.txt"
    if val_path.is_file():
        val_txt = val_path.read_text(encoding="utf-8", errors="replace")
        if "VALID=1" not in val_txt.splitlines():
            any_report_invalid = True
    if report.is_file():
        ok, errs, first = validate_report_file(report, ngram, host_timing=host_timing)
        if ok and first is not None:
            report_ok = True
            decode = first.get("decode_tok_s", "n/a")
            tau = first.get("tau", "n/a")
            if want_on:
                acc = first.get("ngram_mod_accepted")
                drf = first.get("ngram_mod_drafts")
                rate = first.get("ngram_mod_accept_rate")
                ngram_s = f"{acc}/{drf}@{rate}"
            else:
                ngram_s = "off"
        else:
            report_ok = False
            any_report_invalid = True
            if any("mtp_ngram" in e for e in errs):
                any_route_mismatch = True
            if first is not None:
                decode = first.get("decode_tok_s", "n/a")
                tau = first.get("tau", "n/a")
    else:
        if rc == 0:
            any_report_invalid = True
        report_ok = False
    if rc != 0:
        any_arm_fail = True
    if not report_ok and rc == 0:
        any_report_invalid = True
    status = "OK" if (rc == 0 and report_ok) else "FAIL"
    print(f"{name}\tarm={arm}\tstatus={status}\texit={rc}\tdecode_tok_s={decode}\ttau={tau}\tngram={ngram_s}\texpected_ngram={ngram}")

# Ledger completeness
print("--- ledger ---")
ledger_path = campaign / "ledger.jsonl"
ledger_ok = True
ledger_rows = []
ledger_parse_errors = 0
if not ledger_path.is_file():
    ledger_ok = False
    print("ledger=missing")
else:
    for line_no, line in enumerate(ledger_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            ledger_rows.append(row)
        except Exception as e:
            ledger_parse_errors += 1
            ledger_ok = False
            print(f"ledger_line_{line_no}=parse_error {type(e).__name__}: {e}")

expected_n = len(order)
indices = []
for row in ledger_rows:
    if "parse_error" in row:
        ledger_ok = False
        ledger_parse_errors += 1
    try:
        indices.append(int(row.get("run_index")))
    except Exception:
        ledger_ok = False

# Exactly one row per expected run index, unique, expected arm mapping, metrics pass contract.
cnt = Counter(indices)
if len(indices) != expected_n:
    ledger_ok = False
    print(f"ledger_row_count={len(indices)} expected={expected_n}")
else:
    print(f"ledger_row_count={len(indices)} expected={expected_n}")

for i in range(1, expected_n + 1):
    if cnt.get(i, 0) != 1:
        ledger_ok = False
        print(f"ledger_index_{i}_count={cnt.get(i, 0)} expected=1")

# Map first occurrence of each index for arm/metrics checks
by_index = {}
for row in ledger_rows:
    try:
        ri = int(row.get("run_index"))
    except Exception:
        continue
    if ri not in by_index:
        by_index[ri] = row

for idx, arm in enumerate(order, start=1):
    row = by_index.get(idx)
    if row is None:
        ledger_ok = False
        print(f"ledger_missing_index={idx}")
        continue
    if row.get("arm") != arm:
        ledger_ok = False
        print(f"ledger_arm_mismatch index={idx} got={row.get('arm')!r} expected={arm}")
    if "parse_error" in row:
        ledger_ok = False
        print(f"ledger_parse_error index={idx} {row.get('parse_error')}")
        continue
    metrics = row.get("metrics")
    ok, errs = validate_first(metrics, expected_ngram(arm), host_timing=host_timing)
    if not ok:
        ledger_ok = False
        print(f"ledger_metrics_invalid index={idx} errs={','.join(errs)}")

if ledger_error:
    ledger_ok = False
    print("ledger_error_marker=present")
print(f"ledger_ok={ledger_ok}")
print(f"ledger_parse_errors={ledger_parse_errors}")

print("--- parity ---")
for line in parity_txt.splitlines():
    print(line)

print("--- gates ---")
print(f"warmup_failure={warmup_failure}")
print(f"any_arm_fail={any_arm_fail}")
print(f"any_report_invalid={any_report_invalid}")
print(f"any_route_mismatch={any_route_mismatch}")
print(f"output_parity_pass={parity_pass}")
print(f"order_abba_ok={order_ok}")
print(f"ledger_ok={ledger_ok}")

overall = "PASS"
if (
    any_arm_fail
    or any_report_invalid
    or any_route_mismatch
    or not parity_pass
    or warmup_failure
    or not order_ok
    or not ledger_ok
    or ledger_error
):
    overall = "FAIL"
print(f"OVERALL={overall}")
PY

# Final exit code from summary
if grep -q '^OVERALL=PASS$' "$CAMPAIGN/summary.txt"; then
  printf 'ngram-mtp-abba: OVERALL=PASS campaign=%s\n' "$CAMPAIGN"
  exit 0
fi
printf 'ngram-mtp-abba: OVERALL=FAIL campaign=%s\n' "$CAMPAIGN" >&2
exit 1
