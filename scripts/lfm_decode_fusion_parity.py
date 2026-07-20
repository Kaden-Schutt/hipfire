#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Fresh-process behavior-equivalence oracle for gfx1201 LFM2.5-350M MQ4 decode fusion.

Builds and runs two independent arms of
`lfm_decode_fusion_parity` (Rust example) under the exact env contract:

  baseline:  HIPFIRE_LFM2_GRAPH=0 HIPFIRE_FORWARD_LOWERED=1
             HIPFIRE_REPLAY_BACKEND=hip HIPFIRE_LFM2_GFX1201_DECODE_FUSION=0
  candidate: same with HIPFIRE_LFM2_GFX1201_DECODE_FUSION=1

Fusion is a process-global LazyLock, so arms MUST be separate fresh processes.

Gates (committed; design § correctness):
  * model md5 == cb5284b8ad5c6f9e4ca859c0aff0bcd0
  * gpu.arch == gfx1201, arch_id == 11
  * fusion marker ONLY in candidate stderr; absent from baseline
  * per position:
      - finite logits (both arms)
      - exact argmax
      - logits cosine >= 0.999999
      - logits max_abs <= 0.05
      - KL(softmax(base)||softmax(cand)) <= 5e-4
      - exact n_tokens
      - every conv tail cosine >= 0.999999 and max_abs <= 0.01
      - dequantized written Q8 KV cosine >= 0.99999 and max_abs <= 0.05
  * mean KL across positions <= 1e-4

Emits one machine-readable JSON report on stdout; exit 0 on pass, nonzero on fail.

Usage:
  python3 scripts/lfm_decode_fusion_parity.py
  python3 scripts/lfm_decode_fusion_parity.py --model /path/to/lfm2.5-350m.mq4
  python3 scripts/lfm_decode_fusion_parity.py --bin /path/to/lfm_decode_fusion_parity
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / ".hipfire" / "models" / "lfm2.5-350m.mq4"
EXPECTED_MODEL_MD5 = "cb5284b8ad5c6f9e4ca859c0aff0bcd0"
EXPECTED_ARCH = "gfx1201"
EXPECTED_ARCH_ID = 11
FUSION_MARKER = (
    "[lfm2moe] exact gfx1201 350m decode fusion active: shared RMSNorm+FWHT"
)
TOKENS = [1, 17, 42, 256, 1024, 4096, 8191, 7, 511, 2048, 63, 30000]

# Committed numeric gates — do not loosen after observing candidate output.
LOGIT_COS_MIN = 0.999999
LOGIT_MAX_ABS = 0.05
KL_MAX = 5e-4
KL_MEAN_MAX = 1e-4
CONV_COS_MIN = 0.999999
CONV_MAX_ABS = 0.01
KV_COS_MIN = 0.99999
KV_MAX_ABS = 0.05

EXAMPLE_NAME = "lfm_decode_fusion_parity"
PACKAGE = "hipfire-arch-lfm2moe"


def file_md5(path: Path) -> str:
    out = subprocess.check_output(["md5sum", str(path)], text=True)
    return out.split()[0]


def decode_f32_le_b64(blob: str) -> list[float]:
    raw = base64.b64decode(blob, validate=True)
    if len(raw) % 4 != 0:
        raise ValueError(f"f32_le_b64 length {len(raw)} not multiple of 4")
    n = len(raw) // 4
    return list(struct.unpack(f"<{n}f", raw))


def cosine_and_max_abs(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) != len(b):
        raise ValueError(f"length mismatch {len(a)} vs {len(b)}")
    if not a:
        return 1.0, 0.0
    dot = 0.0
    aa = 0.0
    bb = 0.0
    max_abs = 0.0
    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("non-finite value in metric vectors")
        xf = float(x)
        yf = float(y)
        dot += xf * yf
        aa += xf * xf
        bb += yf * yf
        max_abs = max(max_abs, abs(xf - yf))
    if aa == 0.0 and bb == 0.0:
        return 1.0, max_abs
    return dot / (math.sqrt(aa) * math.sqrt(bb)), max_abs


def kl_softmax(ref: list[float], cand: list[float]) -> float:
    """KL(softmax(ref) || softmax(cand)) with log-sum-exp stability."""
    if len(ref) != len(cand):
        raise ValueError("kl length mismatch")
    rmax = max(ref)
    cmax = max(cand)
    rsum = 0.0
    csum = 0.0
    for rv, cv in zip(ref, cand):
        rsum += math.exp(float(rv) - rmax)
        csum += math.exp(float(cv) - cmax)
    log_rsum = math.log(rsum) + rmax
    log_csum = math.log(csum) + cmax
    kl = 0.0
    for rv, cv in zip(ref, cand):
        log_p = float(rv) - log_rsum
        p = math.exp(log_p)
        if p > 0.0:
            log_q = float(cv) - log_csum
            kl += p * (log_p - log_q)
    if kl < 0.0 and kl > -1e-9:
        return 0.0
    return kl


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HIPFIRE_LFM2_GRAPH"] = "0"
    env["HIPFIRE_FORWARD_LOWERED"] = "1"
    env["HIPFIRE_REPLAY_BACKEND"] = "hip"
    # Keep legacy aliases off so only the exact gfx1201 flag is authoritative.
    env["HIPFIRE_LFM2_DECODE_FUSION"] = "0"
    env["HIPFIRE_LFM2_350M_MQ4_DECODE_FUSION"] = "0"
    return env


def build_example() -> Path:
    """Release-build the oracle example; return the binary path."""
    cmd = [
        "cargo",
        "build",
        "-p",
        PACKAGE,
        "--release",
        "--features",
        "deltanet",
        "--example",
        EXAMPLE_NAME,
    ]
    print(f"[build] {' '.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"cargo build failed rc={proc.returncode}")
    candidates = [
        REPO / "target" / "release" / "examples" / EXAMPLE_NAME,
        REPO / "target" / "release" / EXAMPLE_NAME,
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    # cargo may place examples under a deps hash path; ask cargo for the path.
    locate = subprocess.run(
        [
            "cargo",
            "build",
            "-p",
            PACKAGE,
            "--release",
            "--features",
            "deltanet",
            "--example",
            EXAMPLE_NAME,
            "--message-format=json",
        ],
        cwd=str(REPO),
        text=True,
        capture_output=True,
    )
    for line in locate.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "compiler-artifact" and msg.get("executable"):
            exe = Path(msg["executable"])
            if exe.is_file():
                return exe
    raise SystemExit(f"built binary not found for example {EXAMPLE_NAME}")


def run_arm(
    binary: Path,
    model: Path,
    fusion: str,
    label: str,
) -> tuple[dict[str, Any], str, str, int]:
    env = base_env()
    env["HIPFIRE_LFM2_GFX1201_DECODE_FUSION"] = fusion
    cmd = [str(binary), "--model", str(model)]
    print(
        f"[arm {label}] fusion={fusion} cmd={' '.join(cmd)}",
        file=sys.stderr,
        flush=True,
    )
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
    )
    stdout = proc.stdout
    stderr = proc.stderr
    if proc.returncode != 0:
        sys.stderr.write(f"[arm {label}] rc={proc.returncode}\n")
        sys.stderr.write(stderr)
        # Still try to surface any JSON for debugging.
    # The arm prints a single JSON object on stdout (may be large).
    payload = stdout.strip()
    if not payload:
        raise SystemExit(f"arm {label}: empty stdout (rc={proc.returncode})")
    # Tolerate trailing noise: take the last complete JSON object.
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        # Find outermost object by scanning from first '{'.
        start = payload.find("{")
        if start < 0:
            raise SystemExit(
                f"arm {label}: no JSON object in stdout (rc={proc.returncode})"
            )
        data = json.loads(payload[start:])
    return data, stdout, stderr, proc.returncode


def compare(base: dict[str, Any], cand: dict[str, Any], base_err: str, cand_err: str) -> dict[str, Any]:
    failures: list[str] = []
    position_rows: list[dict[str, Any]] = []

    # --- fixture / arch identity ---
    for arm_name, arm in (("baseline", base), ("candidate", cand)):
        md5 = arm.get("model_md5")
        if md5 != EXPECTED_MODEL_MD5:
            failures.append(
                f"{arm_name}: model_md5 {md5} != {EXPECTED_MODEL_MD5}"
            )
        arch = arm.get("arch")
        if arch != EXPECTED_ARCH:
            failures.append(f"{arm_name}: arch {arch} != {EXPECTED_ARCH}")
        arch_id = arm.get("arch_id")
        if arch_id != EXPECTED_ARCH_ID:
            failures.append(
                f"{arm_name}: arch_id {arch_id} != {EXPECTED_ARCH_ID}"
            )
        if not arm.get("retained_fixture_evidence", False):
            failures.append(f"{arm_name}: retained_fixture_evidence is false")

    if base.get("fusion_requested") is not False:
        failures.append(
            f"baseline: fusion_requested={base.get('fusion_requested')} (want false)"
        )
    if cand.get("fusion_requested") is not True:
        failures.append(
            f"candidate: fusion_requested={cand.get('fusion_requested')} (want true)"
        )

    base_has_marker = FUSION_MARKER in base_err
    cand_has_marker = FUSION_MARKER in cand_err
    if base_has_marker:
        failures.append("baseline stderr contains fusion marker (must be absent)")
    if not cand_has_marker:
        failures.append(
            "candidate stderr missing fusion marker "
            f"(need exact: {FUSION_MARKER!r})"
        )

    base_pos = base.get("positions") or []
    cand_pos = cand.get("positions") or []
    if len(base_pos) != len(TOKENS) or len(cand_pos) != len(TOKENS):
        failures.append(
            f"position count base={len(base_pos)} cand={len(cand_pos)} "
            f"expected={len(TOKENS)}"
        )

    kls: list[float] = []
    n = min(len(base_pos), len(cand_pos), len(TOKENS))
    global_logit_min_cos = 1.0
    global_logit_max_abs = 0.0
    global_conv_min_cos = 1.0
    global_conv_max_abs = 0.0
    global_kv_min_cos = 1.0
    global_kv_max_abs = 0.0

    for i in range(n):
        bp = base_pos[i]
        cp = cand_pos[i]
        row: dict[str, Any] = {"pos": i, "token": TOKENS[i], "ok": True, "checks": {}}
        pos_fail: list[str] = []

        if bp.get("token") != TOKENS[i] or cp.get("token") != TOKENS[i]:
            pos_fail.append(
                f"token mismatch base={bp.get('token')} cand={cp.get('token')} "
                f"expected={TOKENS[i]}"
            )
        if bp.get("pos") != i or cp.get("pos") != i:
            pos_fail.append("pos field mismatch")

        # n_tokens exact
        expected_n = i + 1
        bn = bp.get("n_tokens")
        cn = cp.get("n_tokens")
        n_ok = bn == expected_n and cn == expected_n
        row["checks"]["n_tokens"] = {
            "baseline": bn,
            "candidate": cn,
            "expected": expected_n,
            "ok": n_ok,
        }
        if not n_ok:
            pos_fail.append(f"n_tokens base={bn} cand={cn} expected={expected_n}")

        # finite + argmax
        b_fin = bool(bp.get("logits_finite"))
        c_fin = bool(cp.get("logits_finite"))
        row["checks"]["logits_finite"] = {
            "baseline": b_fin,
            "candidate": c_fin,
            "ok": b_fin and c_fin,
        }
        if not (b_fin and c_fin):
            pos_fail.append(f"non-finite logits base={b_fin} cand={c_fin}")

        b_am = bp.get("argmax")
        c_am = cp.get("argmax")
        am_ok = b_am == c_am
        row["checks"]["argmax"] = {
            "baseline": b_am,
            "candidate": c_am,
            "ok": am_ok,
        }
        if not am_ok:
            pos_fail.append(f"argmax mismatch base={b_am} cand={c_am}")

        # logits metrics
        try:
            b_logits = decode_f32_le_b64(bp["logits_f32_le_b64"])
            c_logits = decode_f32_le_b64(cp["logits_f32_le_b64"])
            cos, mx = cosine_and_max_abs(b_logits, c_logits)
            kl = kl_softmax(b_logits, c_logits)
        except Exception as exc:  # noqa: BLE001 — surface decode errors as gate fails
            cos, mx, kl = float("nan"), float("inf"), float("inf")
            pos_fail.append(f"logits decode/metric error: {exc}")

        kls.append(kl)
        global_logit_min_cos = min(global_logit_min_cos, cos if math.isfinite(cos) else -1.0)
        global_logit_max_abs = max(global_logit_max_abs, mx if math.isfinite(mx) else float("inf"))

        logit_ok = (
            math.isfinite(cos)
            and cos >= LOGIT_COS_MIN
            and math.isfinite(mx)
            and mx <= LOGIT_MAX_ABS
            and math.isfinite(kl)
            and kl <= KL_MAX
        )
        row["checks"]["logits"] = {
            "cosine": cos,
            "max_abs": mx,
            "kl": kl,
            "cosine_min": LOGIT_COS_MIN,
            "max_abs_max": LOGIT_MAX_ABS,
            "kl_max": KL_MAX,
            "ok": logit_ok,
        }
        if not logit_ok:
            pos_fail.append(
                f"logits gate fail cos={cos} max_abs={mx} kl={kl} "
                f"(need cos>={LOGIT_COS_MIN}, max_abs<={LOGIT_MAX_ABS}, kl<={KL_MAX})"
            )

        # conv tails
        b_conv = bp.get("conv_tails") or []
        c_conv = cp.get("conv_tails") or []
        conv_min_cos = 1.0
        conv_max_abs = 0.0
        conv_ok = bool(bp.get("conv_finite", True)) and bool(cp.get("conv_finite", True))
        if len(b_conv) != len(c_conv):
            conv_ok = False
            pos_fail.append(f"conv slot count base={len(b_conv)} cand={len(c_conv)}")
        for bi, ci in zip(b_conv, c_conv):
            try:
                bv = decode_f32_le_b64(bi["f32_le_b64"])
                cv = decode_f32_le_b64(ci["f32_le_b64"])
                ccos, cmx = cosine_and_max_abs(bv, cv)
            except Exception as exc:  # noqa: BLE001
                ccos, cmx = float("nan"), float("inf")
                conv_ok = False
                pos_fail.append(f"conv decode error slot={bi.get('slot')}: {exc}")
            conv_min_cos = min(conv_min_cos, ccos if math.isfinite(ccos) else -1.0)
            conv_max_abs = max(conv_max_abs, cmx if math.isfinite(cmx) else float("inf"))
            if not (
                math.isfinite(ccos)
                and ccos >= CONV_COS_MIN
                and math.isfinite(cmx)
                and cmx <= CONV_MAX_ABS
            ):
                conv_ok = False
        global_conv_min_cos = min(global_conv_min_cos, conv_min_cos)
        global_conv_max_abs = max(global_conv_max_abs, conv_max_abs)
        row["checks"]["conv"] = {
            "min_cosine": conv_min_cos,
            "max_abs": conv_max_abs,
            "cosine_min": CONV_COS_MIN,
            "max_abs_max": CONV_MAX_ABS,
            "n_slots": len(b_conv),
            "ok": conv_ok,
        }
        if not conv_ok:
            pos_fail.append(
                f"conv gate fail min_cos={conv_min_cos} max_abs={conv_max_abs} "
                f"(need cos>={CONV_COS_MIN}, max_abs<={CONV_MAX_ABS})"
            )

        # dequantized written Q8 KV
        b_kv = bp.get("kv_written") or []
        c_kv = cp.get("kv_written") or []
        kv_min_cos = 1.0
        kv_max_abs = 0.0
        kv_ok = bool(bp.get("kv_finite", True)) and bool(cp.get("kv_finite", True))
        if len(b_kv) != len(c_kv):
            kv_ok = False
            pos_fail.append(f"kv entry count base={len(b_kv)} cand={len(c_kv)}")
        # Index candidate by (kind, slot) for stable pairing.
        c_kv_map = {(e.get("kind"), e.get("slot")): e for e in c_kv}
        for be in b_kv:
            key = (be.get("kind"), be.get("slot"))
            ce = c_kv_map.get(key)
            if ce is None:
                kv_ok = False
                pos_fail.append(f"kv missing candidate entry {key}")
                continue
            try:
                bv = decode_f32_le_b64(be["f32_le_b64"])
                cv = decode_f32_le_b64(ce["f32_le_b64"])
                kcos, kmx = cosine_and_max_abs(bv, cv)
            except Exception as exc:  # noqa: BLE001
                kcos, kmx = float("nan"), float("inf")
                kv_ok = False
                pos_fail.append(f"kv decode error {key}: {exc}")
            kv_min_cos = min(kv_min_cos, kcos if math.isfinite(kcos) else -1.0)
            kv_max_abs = max(kv_max_abs, kmx if math.isfinite(kmx) else float("inf"))
            if not (
                math.isfinite(kcos)
                and kcos >= KV_COS_MIN
                and math.isfinite(kmx)
                and kmx <= KV_MAX_ABS
            ):
                kv_ok = False
        global_kv_min_cos = min(global_kv_min_cos, kv_min_cos)
        global_kv_max_abs = max(global_kv_max_abs, kv_max_abs)
        row["checks"]["kv"] = {
            "min_cosine": kv_min_cos,
            "max_abs": kv_max_abs,
            "cosine_min": KV_COS_MIN,
            "max_abs_max": KV_MAX_ABS,
            "n_entries": len(b_kv),
            "ok": kv_ok,
        }
        if not kv_ok:
            pos_fail.append(
                f"kv gate fail min_cos={kv_min_cos} max_abs={kv_max_abs} "
                f"(need cos>={KV_COS_MIN}, max_abs<={KV_MAX_ABS})"
            )

        if pos_fail:
            row["ok"] = False
            row["failures"] = pos_fail
            failures.extend(f"pos={i}: {msg}" for msg in pos_fail)
        position_rows.append(row)

    mean_kl = sum(kls) / len(kls) if kls else float("inf")
    max_kl = max(kls) if kls else float("inf")
    mean_kl_ok = math.isfinite(mean_kl) and mean_kl <= KL_MEAN_MAX
    if not mean_kl_ok:
        failures.append(
            f"mean KL {mean_kl} exceeds {KL_MEAN_MAX} (max_kl={max_kl})"
        )

    passed = len(failures) == 0
    report = {
        "oracle": "lfm_decode_fusion_parity",
        "pass": passed,
        "fixture": {
            "model_path_baseline": base.get("model_path"),
            "model_path_candidate": cand.get("model_path"),
            "model_md5_baseline": base.get("model_md5"),
            "model_md5_candidate": cand.get("model_md5"),
            "expected_model_md5": EXPECTED_MODEL_MD5,
            "arch_baseline": base.get("arch"),
            "arch_candidate": cand.get("arch"),
            "arch_id_baseline": base.get("arch_id"),
            "arch_id_candidate": cand.get("arch_id"),
            "expected_arch": EXPECTED_ARCH,
            "expected_arch_id": EXPECTED_ARCH_ID,
            "tokens": TOKENS,
        },
        "route": {
            "baseline": base.get("route"),
            "candidate": cand.get("route"),
        },
        "fusion_marker": {
            "text": FUSION_MARKER,
            "baseline_present": base_has_marker,
            "candidate_present": cand_has_marker,
            "ok": (not base_has_marker) and cand_has_marker,
        },
        "thresholds": {
            "logit_cosine_min": LOGIT_COS_MIN,
            "logit_max_abs": LOGIT_MAX_ABS,
            "kl_max": KL_MAX,
            "kl_mean_max": KL_MEAN_MAX,
            "conv_cosine_min": CONV_COS_MIN,
            "conv_max_abs": CONV_MAX_ABS,
            "kv_cosine_min": KV_COS_MIN,
            "kv_max_abs": KV_MAX_ABS,
        },
        "summary": {
            "positions": n,
            "logit_min_cosine": global_logit_min_cos,
            "logit_max_abs": global_logit_max_abs,
            "kl_mean": mean_kl,
            "kl_max": max_kl,
            "kl_mean_ok": mean_kl_ok,
            "conv_min_cosine": global_conv_min_cos,
            "conv_max_abs": global_conv_max_abs,
            "kv_min_cosine": global_kv_min_cos,
            "kv_max_abs": global_kv_max_abs,
            "failure_count": len(failures),
        },
        "positions": position_rows,
        "failures": failures,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("HIPFIRE_LFM_FUSION_MODEL", DEFAULT_MODEL)),
        help=f"exact MQ4 fixture (default {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--bin",
        type=Path,
        default=None,
        help="prebuilt lfm_decode_fusion_parity binary (skip cargo build)",
    )
    ap.add_argument(
        "--keep-logs",
        action="store_true",
        help="write per-arm stdout/stderr under a temp dir and print the path",
    )
    args = ap.parse_args()

    model: Path = args.model.expanduser().resolve()
    if not model.is_file():
        print(
            json.dumps(
                {
                    "oracle": "lfm_decode_fusion_parity",
                    "pass": False,
                    "failures": [f"model missing: {model}"],
                }
            )
        )
        return 2

    md5 = file_md5(model)
    if md5 != EXPECTED_MODEL_MD5:
        print(
            json.dumps(
                {
                    "oracle": "lfm_decode_fusion_parity",
                    "pass": False,
                    "fixture": {
                        "model_path": str(model),
                        "model_md5": md5,
                        "expected_model_md5": EXPECTED_MODEL_MD5,
                    },
                    "failures": [
                        f"model md5 {md5} != {EXPECTED_MODEL_MD5} before arm launch"
                    ],
                }
            )
        )
        return 2

    binary = args.bin.expanduser().resolve() if args.bin else build_example()
    if not binary.is_file():
        print(
            json.dumps(
                {
                    "oracle": "lfm_decode_fusion_parity",
                    "pass": False,
                    "failures": [f"binary missing: {binary}"],
                }
            )
        )
        return 2

    log_dir = None
    if args.keep_logs:
        log_dir = Path(tempfile.mkdtemp(prefix="lfm_decode_fusion_parity_"))

    base_data, base_out, base_err, base_rc = run_arm(binary, model, "0", "baseline")
    cand_data, cand_out, cand_err, cand_rc = run_arm(binary, model, "1", "candidate")

    if log_dir is not None:
        (log_dir / "baseline.stdout").write_text(base_out)
        (log_dir / "baseline.stderr").write_text(base_err)
        (log_dir / "candidate.stdout").write_text(cand_out)
        (log_dir / "candidate.stderr").write_text(cand_err)
        print(f"[logs] {log_dir}", file=sys.stderr, flush=True)

    pre_failures: list[str] = []
    if base_rc != 0:
        pre_failures.append(f"baseline arm rc={base_rc}")
    if cand_rc != 0:
        pre_failures.append(f"candidate arm rc={cand_rc}")

    report = compare(base_data, cand_data, base_err, cand_err)
    if pre_failures:
        report["pass"] = False
        report["failures"] = pre_failures + list(report.get("failures") or [])
        report["summary"]["failure_count"] = len(report["failures"])

    report["arms"] = {
        "baseline_rc": base_rc,
        "candidate_rc": cand_rc,
        "binary": str(binary),
        "model_md5_host": md5,
    }

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
