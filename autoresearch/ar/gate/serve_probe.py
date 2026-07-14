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
import statistics
import subprocess

from .perf_policy import _delta_pct, classify_perf
from ..certify import verdict as V


def run_serve_harness(daemon_bin, model_path, dev, *, repo, kv="q8", max_tokens=128,
                      port=11540, timeout=300, run=None) -> list:
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
    try:
        proc = runner(argv, env, timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"serve_harness timeout after {timeout}s for {os.path.basename(daemon_bin)}/"
            f"{os.path.basename(model_path)}"
        ) from e
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


def _samples(rows, key) -> list:
    return [r[key] for r in rows if isinstance(r.get(key), (int, float))]


def _median(xs):
    return round(statistics.median(xs), 2) if xs else None


def _cell_row(verdict, *, arch, model, base_rows, head_rows, tok_d=0.0,
              parity=None, coherence=None) -> dict:
    """A self-describing ledger row for one (model, arch) gate cell — the loop's
    ``verdict.make_row`` schema (so the gate ledger + BOD read like the loop's), with a
    gate verdict vocabulary: PASS / PARITY_FAIL / COHERENCE_FAIL / REGRESSION / EMPTY."""
    bt, ht = _samples(base_rows, "decode_tok_s"), _samples(head_rows, "decode_tok_s")
    return V.make_row(
        arch, kernel=None, lever="pr-gate", verdict=verdict,
        parity=parity, coherence=coherence, perf_delta=tok_d,
        extra={"model": model, "cell_pass": verdict == "PASS",
               "tok_delta_pct": tok_d, "base_decode": _median(bt), "var_decode": _median(ht),
               "base_runs": bt, "var_runs": ht})


def grade_cell(base_rows, head_rows, *, arch, model, floor) -> dict:
    """Grade one (model, arch) cell from base/head serve_harness rows → a ledger ROW
    (``_cell_row``). Order parity → coherence → perf: a value change or a NEW attractor
    is a hard fail; only then is perf classified (NEUTRAL and IMPROVEMENT both PASS).
    Empty output on EITHER side fails (a daemon that generates nothing is not a pass).
    The row is per (model, arch), so a change that breaks 27b but not a3b yields a
    PARITY_FAIL row for 27b and a PASS row for a3b — itemized straight into the BOD."""
    if not base_rows or not head_rows:
        return _cell_row("EMPTY", arch=arch, model=model, base_rows=base_rows, head_rows=head_rows,
                         parity={"content_exact": None, "empty": True})

    bc, hc = _content(base_rows), _content(head_rows)
    if any(not c for c in bc + hc):
        return _cell_row("EMPTY", arch=arch, model=model, base_rows=base_rows, head_rows=head_rows,
                         parity={"content_exact": None, "empty": True})
    if bc != hc:
        return _cell_row("PARITY_FAIL", arch=arch, model=model, base_rows=base_rows,
                         head_rows=head_rows, parity={"content_exact": False})

    if any(h.get("attractor") and not b.get("attractor") for b, h in zip(base_rows, head_rows)):
        return _cell_row("COHERENCE_FAIL", arch=arch, model=model, base_rows=base_rows,
                         head_rows=head_rows, parity={"content_exact": True},
                         coherence={"pass": False, "new_attractor": True})

    bt, ht = _samples(base_rows, "decode_tok_s"), _samples(head_rows, "decode_tok_s")
    bw, hw = _samples(base_rows, "wall_s"), _samples(head_rows, "wall_s")
    tok_d = _delta_pct(bt, ht)
    pclass = classify_perf(bt, ht, bw, hw, floor=floor) if (bt and ht and bw and hw) else "NEUTRAL"
    verdict = "REGRESSION" if pclass == "REGRESSION" else "PASS"
    return _cell_row(verdict, arch=arch, model=model, base_rows=base_rows, head_rows=head_rows,
                     tok_d=tok_d, parity={"content_exact": True},
                     coherence={"pass": True})


# BOD blocker kind per gate-cell verdict (spec §10 itemization vocabulary).
_CELL_KIND = {"PARITY_FAIL": "parity", "COHERENCE_FAIL": "coherence",
              "REGRESSION": "perf_regression", "EMPTY": "empty_generation"}


def cell_blocker(row) -> dict:
    """One itemized BOD blocker from a failing cell row — names the (arch, model, kind)
    so the contributor sees exactly which (model × arch) broke (27b vs a3b)."""
    kind = _CELL_KIND.get(row["verdict"], row["verdict"].lower())
    m = row.get("model", "?")
    detail = f"{m} @ {row['arch']}: {kind}"
    if kind == "perf_regression" and row.get("tok_delta_pct") is not None:
        detail += f" ({row['tok_delta_pct']:.1f}% tok/s)"
    return {"kind": kind, "arch": row["arch"], "model": m, "detail": detail}
