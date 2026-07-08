#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""ab_certify_serve — the v2 certify orchestrator (serve_harness-only, three arms).

Ties the decision-core modules (coherence_arm / perf / certify_verdict) into the three-arm gate the
v2 spec defines. The GPU work — spinning a `hipfire serve` for a daemon binary and running
serve_harness against it — is behind a `ServeRunner` seam (dependency-injected), so ALL of the
orchestration (arm assembly, parity short-circuit, coherence hard-gate, B_a advancement, ledger row)
is unit-testable no-GPU with a mock runner. The real runner is a thin adapter (built next).

Measurement layout:
  PARITY    : value-preservation. RAW DAEMON short single greedy, token-id EXACT. Daemon voodoo is
              permitted ONLY here — a plain-prompt no-thinking <=48-tok greedy run IS reproducible
              (the serve path's thinking trace is not). A value change -> PARITY_FAIL. Cheapest, first.
  PERF      : rocprof pinned-clock kernel-DURATION (profile_standard) -> MWU. Filter (no gain -> DEAD).
  COHERENCE : the PRIMARY correctness gate on the REAL user path — SERVE, thinking ON, sampled,
              multiturn seed-SET + semantic validators -> McNemar. NO voodoo here.
Order: parity -> perf -> coherence. WIN = value-preserving AND faster AND stays coherent.
"""
import sys, os, json
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
    """Value-preserving iff the variant's committed TOKEN-IDS equal the baseline's on every parity
    prompt. Runs on the RAW-DAEMON short greedy path (serve_runner.parity_gens): a plain-prompt,
    no-thinking, <=48-token greedy run IS reproducible, so exact token-id comparison is valid and a
    value-changing kernel flips a token -> PARITY_FAIL. (Serve-path byte-exact was the mistake — its
    thinking trace is non-deterministic; raw-daemon short is the permitted value-preservation voodoo.)
    Empty baseline ids = the daemon produced nothing (infra), not a pass."""
    bmap = {g["prompt_id"]: (g.get("token_ids") or []) for g in base_gens}
    mismatches, empty = [], []
    for vg in var_gens:
        pid = vg["prompt_id"]
        b = bmap.get(pid, [])
        v = vg.get("token_ids") or []
        if not b:
            empty.append(pid)
        elif b != v:
            mismatches.append(pid)
    ok = len(mismatches) == 0 and len(empty) < len(var_gens)
    return (ok, {"token_id_exact": len(mismatches) == 0, "mismatches": mismatches, "empty": empty})


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


# ---------- ARM: perf (rocprof pinned-clock kernel-duration) ----------

def perf_result(base_dur_ms, var_dur_ms, base_clk=None, var_clk=None):
    """Kernel-DURATION samples (lower=better) -> orient -> MWU + clock-VOID. Returns (verdict,f,delta%).
    delta% is reported on the ORIGINAL duration (negative = faster) for the ledger."""
    if not base_dur_ms or not var_dur_ms:
        # profiling produced no samples (e.g. rocprof/GPU hiccup) — do NOT fake a DEAD/0.0; the
        # kernel-name match or the trace failed, so the perf arm is UNRESOLVED, not a no-op.
        return ("INCONCLUSIVE", 0.5, 0.0)
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
    """Gate for `var_daemon` vs `base_daemon` (= B_a). Returns the ledger row. Three gates, in order:

      1. PARITY  — value-preservation. RAW-DAEMON short greedy, token-id EXACT (daemon voodoo, allowed
                   ONLY here: short+no-thinking IS reproducible). A value change -> PARITY_FAIL.
      2. PERF    — rocprof pinned-clock kernel-duration. No gain -> DEAD (don't spend coherence).
      3. COHERENCE — the real user path: SERVE, thinking ON, sampled, multiturn seed-SET + validators.
                   A perf-winner that breaks output -> COHERENCE_FAIL. NO voodoo here.

    WIN = value-preserving AND faster AND stays coherent. Parity is cheapest so it goes first; coherence
    is the most expensive so it only runs on parity-clean perf-winners."""
    # 1. PARITY (raw-daemon short greedy token-id exact) — short-circuit a value change.
    p_ok, p_detail = parity_result(runner.parity_gens(base_daemon), runner.parity_gens(var_daemon))
    if not p_ok:
        return cv.make_row(arch, kernel, lever, "PARITY_FAIL", parity=p_detail, base_ref=base_ref)

    # 2. PERF (rocprof) — filter.
    pv, f, delta = perf_result(runner.perf_durations(base_daemon), runner.perf_durations(var_daemon),
                               runner.clocks(base_daemon), runner.clocks(var_daemon))
    if pv != "WIN":
        return cv.make_row(arch, kernel, lever, pv, parity=p_detail, perf_delta=delta, perf_f=f,
                           base_ref=base_ref)

    # 3. COHERENCE (serve, sampled — PRIMARY correctness gate) — a perf-winner must stay coherent.
    c_ok, c_detail = coherence_result(runner.coherence_gens(base_daemon, seeds),
                                      runner.coherence_gens(var_daemon, seeds), expects=expects)
    verdict = "WIN" if c_ok else "COHERENCE_FAIL"
    return cv.make_row(arch, kernel, lever, verdict, parity=p_detail, coherence=c_detail,
                       perf_delta=delta, perf_f=f, base_ref=base_ref, seeds=len(seeds))


def certify_coherence(runner, *, arch, kernel, lever, base_daemon, var_daemon, base_ref,
                      seeds, expects=None):
    """Coherence-ONLY gate — the ROLLOVER pass. The per-round screen (ab_certify_v2p) already enforced
    parity + perf on each win; at FOLD time we run the SERVE coherence pass (thinking ON, sampled,
    multiturn seed-SET + validators) ONCE on the composed stack (loop/<arch>) vs trunk. Returns
    COHERENT / COHERENCE_FAIL. This is the ONLY place the serve/CLI path is used."""
    c_ok, c_detail = coherence_result(runner.coherence_gens(base_daemon, seeds),
                                      runner.coherence_gens(var_daemon, seeds), expects=expects)
    verdict = "COHERENT" if c_ok else "COHERENCE_FAIL"
    return cv.make_row(arch, kernel, lever, verdict, coherence=c_detail, base_ref=base_ref,
                       seeds=len(seeds))


def load_expects(prompts_file):
    """Per-genre validator params from a --prompts-file: {genre: expect_dict}. A prompt's `expect`
    field (e.g. {"number": 42}, {"sentences": 1}) drives run_validators so the coherence arm catches
    fluent-but-WRONG output, not just attractors. Genres with no `expect` get no semantic validator
    (attractor/empty/runaway still apply). Without this the live gate ran with expects=None -> a
    confidently-wrong answer passed coherence."""
    if not prompts_file:
        return {}
    rows = json.load(open(prompts_file))
    return {r.get("genre", "prose"): r.get("expect", {}) for r in rows if r.get("expect")}


def _main():
    """Measurement entry point: given two ALREADY-BUILT daemon binaries, run the three-arm gate and
    print the verdict row JSON. The bash wrapper (ab_certify_serve.sh) builds the daemons + owns the
    git/B_a/ledger mechanics; this owns the serve-path measurement. Keeps the GPU seam in serve_runner."""
    import argparse
    from serve_runner import LiveServeRunner
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--dev", type=int, required=True)
    ap.add_argument("--card", type=int, default=None)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-daemon", required=True, dest="base_daemon")
    ap.add_argument("--var-daemon", required=True, dest="var_daemon")
    ap.add_argument("--base-ref", default="?", dest="base_ref")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--kv", default="q8")
    ap.add_argument("--prompts-file", default=None, dest="prompts_file")
    ap.add_argument("--mode", default="full", choices=["full", "coherence"],
                    help="full = parity+perf+coherence (per-round); coherence = serve coherence only (rollover)")
    a = ap.parse_args()
    runner = LiveServeRunner(a.model, a.arch, a.dev, prompts_file=a.prompts_file, kv=a.kv,
                             kernel=a.kernel, card=a.card)
    fn = certify_coherence if a.mode == "coherence" else certify
    row = fn(runner, arch=a.arch, kernel=a.kernel, lever=a.label,
             base_daemon=a.base_daemon, var_daemon=a.var_daemon,
             base_ref=a.base_ref, seeds=list(range(a.seeds)),
             expects=load_expects(a.prompts_file))
    print(json.dumps(row))


if __name__ == "__main__":
    _main()
