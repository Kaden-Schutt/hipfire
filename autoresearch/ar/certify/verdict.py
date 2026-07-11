# Copyright (c) Kaden Schutt
"""verdict — the certify decision arms + verdict combiner.

Consolidates the three arm results into one certify verdict + ledger row. Ports the combiner logic
from ``certify_verdict.py`` and the arm assembly (parity / coherence) from ``ab_certify_serve.py``,
and REPLACES the legacy duration-only perf arm with the CONJUNCTIVE gate (Task 2.1 contract):

  a WIN requires kernel_decode_tok_s UP *and* rocprof kernel-duration DOWN, each by an independent
  one-sided Mann-Whitney U. A gain in only one statistic ⇒ DEAD.

Precedence (v2 spec): PARITY (value-preservation, short-circuits) → PERF (filter) → COHERENCE
(hard gate). Both perf statistics (`tok_delta_pct`, `dur_delta_pct`) are written to every row.

Pure stdlib, no GPU — unit-testable in the no-GPU CI.
"""
from . import coherence as ca
from . import perf as perf_mod

VERDICTS = ("WIN", "DEAD", "INCONCLUSIVE", "VOID", "PARITY_FAIL", "COHERENCE_FAIL",
            "BUILD_FAIL", "VARIANT_BUILD_FAIL", "BASELINE_BUILD_FAIL")

# The perf arm's own verdict alphabet (Task 2.1 canonical set).
PERF_VERDICTS = ("WIN", "DEAD", "PARITY_FAIL", "INCONCLUSIVE")


# ---------- verdict combiner (ported from certify_verdict.py) ----------

def decide(parity_ok, coherence_ok, perf_verdict):
    """parity_ok/coherence_ok: bool. perf_verdict: one of WIN/DEAD/INCONCLUSIVE/VOID.
    Returns the final certify verdict. Precedence: parity → coherence → perf."""
    if not parity_ok:
        return "PARITY_FAIL"
    if not coherence_ok:
        return "COHERENCE_FAIL"
    if perf_verdict not in ("WIN", "DEAD", "INCONCLUSIVE", "VOID"):
        raise ValueError(f"bad perf_verdict {perf_verdict!r}")
    return perf_verdict          # WIN here == parity ∧ coherence ∧ perf-gain


def is_bankable(verdict):
    """Only a WIN advances the agent's baseline B_a and commits to loop/cardN."""
    return verdict == "WIN"


def make_row(arch, kernel, lever, verdict, *, parity=None, perf_delta=None, perf_f=None,
             coherence=None, base_ref=None, seeds=None, mcnemar=None, extra=None):
    """Assemble the ledger row. Carries the v2 fields (parity/coherence/base_ref) alongside the
    existing shape so `ar ingest` + the digest keep working."""
    row = {
        "arch": arch, "kernel": kernel, "lever": lever, "label": lever, "verdict": verdict,
        "WIN": verdict == "WIN",
        "parity": parity,             # {"fp32_exact": bool, "q8_tol": bool|None}
        "perf_delta": perf_delta,     # kernel-duration delta % (primary), variant vs B_a
        "mwu_dominance": perf_f,
        "coherence": coherence,       # {"pass": bool, "b": int, "c": int, "p": float, "seeds": N}
        "base_ref": base_ref,         # B_a@sha the variant was measured against (advancing baseline)
        "seeds": seeds,
        "mcnemar": mcnemar,
    }
    if extra:
        row.update(extra)
    return row


# ---------- ARM: parity (raw-daemon short greedy, token-id exact) ----------

def parity_result(base_gens, var_gens):
    """Value-preserving iff the variant's committed TOKEN-IDS equal the baseline's on every parity
    prompt. Runs on the RAW-DAEMON short greedy path — a plain-prompt, no-thinking, <=48-token greedy
    run IS reproducible, so exact token-id comparison is valid; a value-changing kernel flips a token
    → PARITY_FAIL. Empty baseline ids = the daemon produced nothing (infra), not a pass.

    Gens are matched by ``prompt_id`` when present, else positionally (a runner may emit anonymous
    single-prompt gens). Returns (ok: bool, detail: dict)."""
    have_pid = all("prompt_id" in g for g in base_gens) and all("prompt_id" in g for g in var_gens)
    if have_pid:
        bmap = {g["prompt_id"]: (g.get("token_ids") or []) for g in base_gens}
        items = [(vg.get("prompt_id"), bmap.get(vg.get("prompt_id"), []), vg.get("token_ids") or [])
                 for vg in var_gens]
    else:
        items = [(i, (base_gens[i].get("token_ids") or []) if i < len(base_gens) else [],
                  vg.get("token_ids") or []) for i, vg in enumerate(var_gens)]
    mismatches, empty = [], []
    for pid, b, v in items:
        if not b:
            empty.append(pid)
        elif b != v:
            mismatches.append(pid)
    ok = len(mismatches) == 0 and len(empty) < len(var_gens)
    return (ok, {"token_id_exact": len(mismatches) == 0, "mismatches": mismatches, "empty": empty})


# ---------- ARM: perf (CONJUNCTIVE: tok/s UP and duration DOWN) ----------

def perf_result(base_tok, var_tok, base_dur, var_dur, base_clk=None, var_clk=None, alpha=0.05):
    """The conjunctive perf gate. Returns a dict:
        {verdict, tok_delta_pct, dur_delta_pct, tok_p, dur_p}

    verdict == "WIN"  iff  var tok/s stochastically GREATER than base (one-sided MWU p<alpha, delta>0)
                     AND   var duration stochastically LESS than base (one-sided MWU p<alpha, delta<0).
    Either statistic failing ⇒ "DEAD" (a thermal tok/s bump with flat duration, or a duration win
    hiding a tok/s regression, is not a real kernel win). Missing samples or a clock-skew VOID ⇒
    "INCONCLUSIVE" (can't be trusted either way — do NOT fake a DEAD).

    tok_delta_pct: median %Δ of tok/s (variant vs base), positive = faster.
    dur_delta_pct: median %Δ of duration (variant vs base), negative = faster.
    """
    tok_delta = perf_mod.median_delta_pct(base_tok, var_tok)
    dur_delta = perf_mod.median_delta_pct(base_dur, var_dur)
    if not base_tok or not var_tok or not base_dur or not var_dur:
        return {"verdict": "INCONCLUSIVE", "tok_delta_pct": tok_delta, "dur_delta_pct": dur_delta,
                "tok_p": 1.0, "dur_p": 1.0}
    # clock-skew guard: a delta produced by a DPM/clock difference is not a kernel effect
    if base_clk is not None and var_clk is not None and perf_mod.clock_void(base_clk, var_clk):
        return {"verdict": "INCONCLUSIVE", "tok_delta_pct": tok_delta, "dur_delta_pct": dur_delta,
                "tok_p": 1.0, "dur_p": 1.0}
    tok_p = perf_mod.mwu(base_tok, var_tok)     # H1: var_tok > base_tok
    dur_p = perf_mod.mwu(var_dur, base_dur)     # H1: base_dur > var_dur  (i.e. var faster)
    tok_up = tok_p < alpha and tok_delta > 0
    dur_down = dur_p < alpha and dur_delta < 0
    verdict = "WIN" if (tok_up and dur_down) else "DEAD"
    return {"verdict": verdict, "tok_delta_pct": tok_delta, "dur_delta_pct": dur_delta,
            "tok_p": tok_p, "dur_p": dur_p}


# ---------- ARM: coherence (Q8 seed-set, paired McNemar rate test) ----------

def _gen_fails(gen, expect):
    """Does this single generation fail the coherence bar? (attractor OR runaway OR empty OR a
    semantic validator the prompt asked for). Returns (fail: bool, reason: str)."""
    if gen.get("empty") or not (gen.get("token_ids") or gen.get("text")):
        return True, "empty"
    if gen.get("finish") == "length":
        return True, "runaway"
    toks = gen.get("token_ids")
    if toks:
        is_a, reason = ca.detect_attractor(toks)
        if is_a:
            return True, f"attractor:{reason}"
    ok, fails = ca.run_validators(gen.get("genre", ""), gen.get("text", ""),
                                  tool_calls=gen.get("tool_calls"), expect=expect)
    if not ok:
        return True, "validator:" + ";".join(fails)
    return False, "ok"


def coherence_result(base_gens, var_gens, expects=None, alpha=0.05):
    """base_gens/var_gens: generations over the seed-SET × battery, aligned by (prompt_id, seed).
    expects: {prompt_id: {validator params}}. Builds paired (base_fail, var_fail) per (prompt,seed)
    and runs McNemar — the variant must not introduce MORE failures than the baseline.

    Falls back to positional pairing when gens lack prompt_id/seed keys (an anonymous single-gen
    runner). Returns (ok: bool, detail: dict)."""
    expects = expects or {}
    keyed = all(("prompt_id" in g and "seed" in g) for g in base_gens + var_gens)
    pairs = []
    if keyed:
        bkey = {(g["prompt_id"], g["seed"]): g for g in base_gens}
        for vg in var_gens:
            bg = bkey.get((vg["prompt_id"], vg["seed"]))
            if bg is None:
                continue
            exp = expects.get(vg["prompt_id"], {})
            bf, _ = _gen_fails(bg, exp)
            vf, _ = _gen_fails(vg, exp)
            pairs.append((bf, vf))
    else:
        for bg, vg in zip(base_gens, var_gens):
            exp = expects.get(vg.get("prompt_id"), {})
            bf, _ = _gen_fails(bg, exp)
            vf, _ = _gen_fails(vg, exp)
            pairs.append((bf, vf))
    worse, p, b, c = ca.mcnemar_worse(pairs, alpha=alpha)
    seeds = len({g.get("seed") for g in var_gens})
    return (not worse, {"pass": not worse, "b": b, "c": c, "p": p, "seeds": seeds, "trials": len(pairs)})
