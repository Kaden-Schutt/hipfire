#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""ab_certify_serve — the v2 certify orchestrator (serve_harness-only, three arms).

Ties the decision-core modules (coherence_arm / perf / certify_verdict) into the three-arm gate the
v2 spec defines. The GPU work — spinning a `hipfire serve` for a daemon binary and running
serve_harness against it — is behind a `ServeRunner` seam (dependency-injected), so ALL of the
orchestration (arm assembly, parity short-circuit, coherence hard-gate, B_a advancement, ledger row)
is unit-testable no-GPU with a mock runner. The real runner is a thin adapter (built next).

Measurement layout (v2 spec):
  PARITY    : FP32-det serve, greedy, battery -> byte-exact text vs B_a (+ Q8 tolerance, when wired)
  COHERENCE : Q8 serve, registry temp>0, seed-SET over battery+guards -> McNemar paired rate test
  PERF      : Q8 serve, interleaved kernel-duration on one warmed prompt -> MWU + clock-VOID
Verdict precedence: PARITY (short-circuit) -> COHERENCE (hard gate) -> PERF.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import coherence_arm as ca
import perf as perf_mod
import certify_verdict as cv


# ---------- a generation result (what the runner returns per prompt/seed) ----------
# dict shape (kept a plain dict so the runner can build it from serve_harness --out JSON):
#   {prompt_id, genre, seed, text, token_ids:[int], tool_calls:[...], finish, empty,
#    kernel_duration_ms, kernel_decode_tok_s, clk}
# `expect` (per-genre validator params) is carried alongside the prompt set, not the gen.


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


# ---------- ARM: parity (FP32-det greedy, byte-exact text) ----------

def parity_result(base_gens, var_gens):
    """base_gens/var_gens: per-prompt deterministic (greedy) generations, aligned by prompt_id.
    Value-preserving iff the variant's text is byte-identical to B_a's on every prompt."""
    bmap = {g["prompt_id"]: g for g in base_gens}
    mismatches = []
    for vg in var_gens:
        bg = bmap.get(vg["prompt_id"])
        if bg is None or bg.get("text", None) != vg.get("text", None):
            mismatches.append(vg["prompt_id"])
    return (len(mismatches) == 0, {"fp32_exact": len(mismatches) == 0, "mismatches": mismatches})


# ---------- ARM: coherence (Q8 seed-set, paired McNemar rate test) ----------

def coherence_result(base_gens, var_gens, expects=None, alpha=0.05):
    """base_gens/var_gens: generations over the seed-SET × battery, aligned by (prompt_id, seed).
    expects: {prompt_id: {validator params}}. Builds paired (base_fail, var_fail) per (prompt,seed)
    and runs McNemar — the variant must not introduce MORE failures than the baseline."""
    expects = expects or {}
    bkey = {(g["prompt_id"], g["seed"]): g for g in base_gens}
    pairs = []
    for vg in var_gens:
        k = (vg["prompt_id"], vg["seed"])
        bg = bkey.get(k)
        if bg is None:
            continue
        exp = expects.get(vg["prompt_id"], {})
        bf, _ = _gen_fails(bg, exp)
        vf, _ = _gen_fails(vg, exp)
        pairs.append((bf, vf))
    worse, p, b, c = ca.mcnemar_worse(pairs, alpha=alpha)
    seeds = len({g["seed"] for g in var_gens})
    return (not worse, {"pass": not worse, "b": b, "c": c, "p": p, "seeds": seeds, "trials": len(pairs)})


# ---------- ARM: perf (Q8 interleaved kernel-duration) ----------

def perf_result(base_dur_ms, var_dur_ms, base_clk=None, var_clk=None):
    """Kernel-DURATION samples (lower=better) -> orient -> MWU + clock-VOID. Returns (verdict,f,delta%).
    delta% is reported on the ORIGINAL duration (negative = faster) for the ledger."""
    b = perf_mod.orient_lower_is_better(base_dur_ms)
    v = perf_mod.orient_lower_is_better(var_dur_ms)
    verdict, f, _ = perf_mod.resolve(b, v, base_clk, var_clk)
    # report the human-facing duration delta (variant vs base), negative = faster
    dur_delta = perf_mod.median_delta_pct(base_dur_ms, var_dur_ms)
    return (verdict, f, dur_delta)


# ---------- orchestration ----------

class ServeRunner:
    """GPU seam. A real runner spins a `hipfire serve` for a daemon binary and drives serve_harness.
    Overridden by the GPU adapter; mocked in tests. All methods return lists of generation dicts /
    duration lists as described above."""
    def parity_gens(self, daemon):        raise NotImplementedError   # -> per-prompt greedy FP32-det
    def coherence_gens(self, daemon, seeds): raise NotImplementedError  # -> per (prompt,seed) Q8 temp>0
    def perf_durations(self, daemon):     raise NotImplementedError   # -> list[float] kernel-duration ms
    def clocks(self, daemon):             return []                    # -> list[int] sclk samples


def certify(runner, *, arch, kernel, lever, base_daemon, var_daemon, base_ref,
            seeds, expects=None):
    """Run the full three-arm gate for `var_daemon` vs `base_daemon` (= B_a). Returns the ledger row
    (and verdict). Short-circuits: parity first (cheap FP32-det), then coherence (hard gate), then
    perf. Only a WIN should advance B_a (caller checks certify_verdict.is_bankable)."""
    # 1. PARITY (FP32-det greedy) — short-circuit on fail
    p_ok, p_detail = parity_result(runner.parity_gens(base_daemon), runner.parity_gens(var_daemon))
    if not p_ok:
        return cv.make_row(arch, kernel, lever, "PARITY_FAIL", parity=p_detail, base_ref=base_ref)

    # 2. COHERENCE (Q8 seed-set) — hard gate
    c_ok, c_detail = coherence_result(runner.coherence_gens(base_daemon, seeds),
                                      runner.coherence_gens(var_daemon, seeds), expects=expects)
    if not c_ok:
        return cv.make_row(arch, kernel, lever, "COHERENCE_FAIL",
                           parity=p_detail, coherence=c_detail, base_ref=base_ref, seeds=len(seeds))

    # 3. PERF (Q8 interleaved kernel-duration)
    pv, f, delta = perf_result(runner.perf_durations(base_daemon), runner.perf_durations(var_daemon),
                               runner.clocks(base_daemon), runner.clocks(var_daemon))
    verdict = cv.decide(parity_ok=True, coherence_ok=True, perf_verdict=pv)
    return cv.make_row(arch, kernel, lever, verdict, parity=p_detail, perf_delta=delta, perf_f=f,
                       coherence=c_detail, base_ref=base_ref, seeds=len(seeds))


def _main():
    """Measurement entry point: given two ALREADY-BUILT daemon binaries, run the three-arm gate and
    print the verdict row JSON. The bash wrapper (ab_certify_serve.sh) builds the daemons + owns the
    git/B_a/ledger mechanics; this owns the serve-path measurement. Keeps the GPU seam in serve_runner."""
    import argparse
    from serve_runner import LiveServeRunner
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--dev", type=int, required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-daemon", required=True, dest="base_daemon")
    ap.add_argument("--var-daemon", required=True, dest="var_daemon")
    ap.add_argument("--base-ref", default="?", dest="base_ref")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--kv", default="q8")
    ap.add_argument("--prompts-file", default=None, dest="prompts_file")
    a = ap.parse_args()
    runner = LiveServeRunner(a.model, a.arch, a.dev, prompts_file=a.prompts_file, kv=a.kv)
    row = certify(runner, arch=a.arch, kernel=a.kernel, lever=a.label,
                  base_daemon=a.base_daemon, var_daemon=a.var_daemon,
                  base_ref=a.base_ref, seeds=list(range(a.seeds)))
    print(json.dumps(row))


if __name__ == "__main__":
    _main()
