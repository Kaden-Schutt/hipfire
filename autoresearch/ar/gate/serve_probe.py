# Copyright (c) Kaden Schutt
"""ar.gate.serve_probe — drive the on-box A/B through scripts/serve_harness.py.

The gate measures a (model, arch) cell by running the committed ``serve_harness.py``
greedy battery against the BASE daemon and the HEAD daemon and comparing their
per-prompt output. serve_harness already spawns the daemon (``HIPFIRE_DAEMON_BIN``),
warms it, runs the genre battery, and writes a per-turn JSON with everything the gate
needs — so the gate reuses it rather than re-implementing raw-daemon parity + rocprof
(the untested LiveServeRunner arms that returned empty samples on the fleet):

  * parity     — ``assistant_content`` byte-exact base-vs-head (greedy ⇒ deterministic)
  * perf       — ``decode_tok_s`` / ``wall_s`` per prompt → the WIN-gate classifier
  * coherence  — the ``attractor`` flag (uniq-ratio + 3-gram density) per prompt

``grade_cell`` is pure and unit-tested; ``run_serve_harness`` is the on-box seam
(subprocess → serve_harness.py) exercised live on the fleet.
"""
from __future__ import annotations

import json
import os
import subprocess

from .perf_policy import _delta_pct, classify_perf


def run_serve_harness(daemon_bin, model_path, dev, *, repo, kv="q8", max_tokens=128,
                      port=11540, timeout=1200, run=None) -> list:
    """Run one greedy serve_harness battery against ``daemon_bin`` and return its
    per-turn rows (parsed from ``--out``). Raises ``RuntimeError`` on a spawn/parse
    failure so the caller can map it to an ERROR verdict (never a silent empty pass)."""
    out = os.path.join("/tmp", f"gate_sh_{os.path.basename(daemon_bin)}_{os.path.basename(model_path)}_{port}.json")
    argv = ["python3", os.path.join(repo, "scripts", "serve_harness.py"),
            "--model", model_path, "--sampling", "greedy", "--kv", kv,
            "--max-tokens", str(max_tokens), "--mode", "battery",
            "--registry", os.path.join(repo, "cli", "registry.json"),
            "--out", out, "--port", str(port)]
    env = dict(os.environ, HIPFIRE_DAEMON_BIN=daemon_bin, HIP_VISIBLE_DEVICES=str(dev))
    runner = run or (lambda a, e, t: subprocess.run(a, env=e, timeout=t, capture_output=True, text=True))
    proc = runner(argv, env, timeout)
    rc = getattr(proc, "returncode", 1)
    if rc != 0:
        tail = (getattr(proc, "stderr", "") or "")[-1500:]
        raise RuntimeError(f"serve_harness rc={rc} for {os.path.basename(daemon_bin)}/"
                           f"{os.path.basename(model_path)}: {tail}")
    try:
        with open(out) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"serve_harness produced no parsable --out ({out}): {e}")


def _content(rows) -> list:
    return [(r.get("assistant_content") or "") for r in rows]


def grade_cell(base_rows, head_rows, *, arch, model, floor) -> dict:
    """Grade one (model, arch) cell from base/head serve_harness rows. Order: parity →
    coherence → perf (a value change or a new attractor is a hard REJECT; only then is
    perf classified — NEUTRAL and IMPROVEMENT both PASS). Empty output on EITHER side is
    a hard REJECT (a daemon that generates nothing is not a pass)."""
    common = {"arch": arch, "model": model}
    if not base_rows or not head_rows:
        return {**common, "gate_verdict": "REJECT", "reason": "empty_generation", "tok_delta_pct": 0.0}

    # 1. PARITY — greedy content must match byte-exact, per prompt (positional).
    bc, hc = _content(base_rows), _content(head_rows)
    if any(not c for c in bc + hc):
        return {**common, "gate_verdict": "REJECT", "reason": "empty_generation", "tok_delta_pct": 0.0}
    if bc != hc:
        return {**common, "gate_verdict": "REJECT", "reason": "parity", "tok_delta_pct": 0.0}

    # 2. COHERENCE — a NEW attractor on the head (not already on base) is a regression.
    if any(h.get("attractor") and not b.get("attractor") for b, h in zip(base_rows, head_rows)):
        return {**common, "gate_verdict": "REJECT", "reason": "coherence", "tok_delta_pct": 0.0}

    # 3. PERF — conjunctive WIN-gate mirror over decode_tok_s (+) and wall_s (duration).
    bt = [r["decode_tok_s"] for r in base_rows if isinstance(r.get("decode_tok_s"), (int, float))]
    ht = [r["decode_tok_s"] for r in head_rows if isinstance(r.get("decode_tok_s"), (int, float))]
    bw = [r["wall_s"] for r in base_rows if isinstance(r.get("wall_s"), (int, float))]
    hw = [r["wall_s"] for r in head_rows if isinstance(r.get("wall_s"), (int, float))]
    tok_d = _delta_pct(bt, ht)
    if bt and ht and bw and hw:
        pclass = classify_perf(bt, ht, bw, hw, floor=floor)
    else:
        pclass = "NEUTRAL"      # no usable perf samples → don't fail on perf, just PASS neutral
    if pclass == "REGRESSION":
        return {**common, "gate_verdict": "REJECT", "reason": "perf_regression", "tok_delta_pct": tok_d}
    return {**common, "gate_verdict": "PASS", "reason": pclass.lower(), "tok_delta_pct": tok_d}
