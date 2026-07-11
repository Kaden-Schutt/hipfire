# Copyright (c) Kaden Schutt
"""ar.gate.engine — gate orchestration over the reused certify arms.

gate_cell: one (model, arch) certification. Order: parity -> perf -> coherence,
but — unlike the loop's certify() — perf-neutral does NOT short-circuit;
coherence always runs (a perf-neutral PR must still be coherent). The verdict is
gate-shaped (PASS / REJECT), reusing verdict.parity_result / coherence_result and
gate.perf_policy.classify_perf.
"""
from __future__ import annotations

from ..certify import perf
from ..certify import verdict as V
from ..certify.orchestrator import DEFAULT_SEEDS, ServeRunner, measurement_hash
from .perf_policy import _delta_pct, classify_perf

__all__ = ["gate_cell", "ServeRunner"]


def _row(*, gate_verdict, reason, perf_class, arch, model, base_daemon, var_daemon,
         kv, maxtok, prompt_md5, tok_delta_pct=None, dur_delta_pct=None):
    return {
        "gate_verdict": gate_verdict, "reason": reason, "perf_class": perf_class,
        "gpu_arch": arch, "model": model, "base_sha": base_daemon, "variant_sha": var_daemon,
        "kv": kv, "maxtok": maxtok, "prompt_md5": prompt_md5,
        "measurement_hash": measurement_hash(arch, model, base_daemon, var_daemon,
                                             prompt_md5, kv, maxtok),
        "tok_delta_pct": tok_delta_pct, "dur_delta_pct": dur_delta_pct,
    }


def gate_cell(runner, *, base_daemon, var_daemon, arch, model, base_ref, kv, maxtok,
              prompt_md5, floor=perf.FLOOR, seeds=None, expects=None) -> dict:
    """Certify one (model, arch) cell: PASS unless parity/coherence fail or a
    significant perf regression. Neutral and improvement both PASS."""
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    common = dict(perf_class=None, arch=arch, model=model, base_daemon=base_daemon,
                  var_daemon=var_daemon, kv=kv, maxtok=maxtok, prompt_md5=prompt_md5)

    # 1. PARITY — a value change is a hard reject; short-circuit.
    p_ok, _ = V.parity_result(runner.parity_gens(base_daemon), runner.parity_gens(var_daemon))
    if not p_ok:
        return _row(gate_verdict="REJECT", reason="parity", **common)

    # 2. PERF — classify, but do NOT short-circuit on neutral.
    base_tok, base_dur = runner.perf_measure(base_daemon)
    var_tok, var_dur = runner.perf_measure(var_daemon)
    pclass = classify_perf(base_tok, var_tok, base_dur, var_dur, floor=floor)
    tok_d, dur_d = _delta_pct(base_tok, var_tok), _delta_pct(base_dur, var_dur)
    common.update(perf_class=pclass, tok_delta_pct=tok_d, dur_delta_pct=dur_d)

    # 3. COHERENCE — always (a neutral PR must still be coherent).
    c_ok, _ = V.coherence_result(runner.coherence_gens(base_daemon, seeds),
                                 runner.coherence_gens(var_daemon, seeds), expects=expects)
    if not c_ok:
        return _row(gate_verdict="REJECT", reason="coherence", **common)

    if pclass == "REGRESSION":
        return _row(gate_verdict="REJECT", reason="perf_regression", **common)
    return _row(gate_verdict="PASS", reason=pclass.lower(), **common)
