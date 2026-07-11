# Copyright (c) Kaden Schutt
"""orchestrator — the v2 certify orchestrator (serve_harness-only, three arms).

Ties the decision arms (verdict.parity_result / perf_result / coherence_result) into the three-arm
gate the v2 spec defines. The GPU work — spinning a `hipfire serve` for a daemon binary and running
serve_harness against it — is behind the ``ServeRunner`` seam (dependency-injected), so ALL of the
orchestration (arm assembly, parity short-circuit, coherence hard-gate, self-describing ledger row)
is unit-testable no-GPU with a mock runner. The real runner is the ``serve_runner.LiveServeRunner``
adapter.

Measurement layout:
  PARITY    : value-preservation. RAW DAEMON short single greedy, token-id EXACT. Daemon voodoo is
              permitted ONLY here (a plain-prompt no-thinking <=48-tok greedy run IS reproducible).
              A value change -> PARITY_FAIL. Cheapest, first.
  PERF      : CONJUNCTIVE — kernel_decode_tok_s (serve_harness) UP *and* rocprof pinned-clock
              kernel-DURATION (profile_standard) DOWN, each by an independent one-sided Mann-Whitney
              U. Either alone -> DEAD. Both stats written to the row.
  COHERENCE : the PRIMARY correctness gate on the REAL user path — SERVE, thinking ON, sampled,
              multiturn seed-SET + semantic validators -> McNemar. NO voodoo here.
Order: parity -> perf -> coherence. WIN = value-preserving AND faster (both axes) AND stays coherent.

Every row is SELF-DESCRIBING: it carries gpu_arch/model/base_sha/variant_sha/prompt_md5/kv/maxtok
and measurement_hash = sha256("|".join([...]))[:16].
"""
import hashlib

from . import verdict as V

# Default coherence seed-set size when the caller doesn't pass one (the canonical certify signature
# omits seeds/expects; they remain optional so later phases can pass a tuned set).
DEFAULT_SEEDS = list(range(8))


class ServeRunner:
    """GPU seam. A real runner spins a `hipfire serve` for a daemon binary and drives serve_harness.
    Overridden by the GPU adapter (serve_runner.LiveServeRunner); mocked in tests."""

    def parity_gens(self, daemon):                 # -> per-prompt greedy token-id gens
        raise NotImplementedError

    def coherence_gens(self, daemon, seeds):       # -> per (prompt,seed) sampled gens
        raise NotImplementedError

    def perf_measure(self, daemon):                # -> (tok_list, dur_list): kernel_decode_tok_s, dur ms
        raise NotImplementedError

    def clocks(self, daemon):                      # -> list[int] sclk samples (empty if pinned)
        return []


def measurement_hash(gpu_arch, model, base_sha, var_sha, prompt_md5, kv, maxtok):
    """sha256("|".join([gpu_arch, model, base_sha, var_sha, prompt_md5, kv, maxtok]))[:16]."""
    payload = "|".join([str(gpu_arch), str(model), str(base_sha), str(var_sha),
                        str(prompt_md5), str(kv), str(maxtok)])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _row(*, arch, kernel, lever, verdict, base_ref, base_sha, var_sha, model, kv, maxtok,
         prompt_md5, parity=None, coherence=None, perf_delta=None, perf_f=None, seeds=None,
         tok_delta_pct=None, dur_delta_pct=None, tok_p=None, dur_p=None):
    """Assemble a self-describing ledger row (base make_row + the self-describing block)."""
    extra = {
        "gpu_arch": arch, "model": model, "base_sha": base_sha, "variant_sha": var_sha,
        "prompt_md5": prompt_md5, "kv": kv, "maxtok": maxtok,
        "measurement_hash": measurement_hash(arch, model, base_sha, var_sha, prompt_md5, kv, maxtok),
        "tok_delta_pct": tok_delta_pct, "dur_delta_pct": dur_delta_pct,
        "tok_p": tok_p, "dur_p": dur_p,
    }
    return V.make_row(arch, kernel, lever, verdict, parity=parity, coherence=coherence,
                      perf_delta=perf_delta if perf_delta is not None else dur_delta_pct,
                      perf_f=perf_f, base_ref=base_ref, seeds=seeds, extra=extra)


def certify(runner, *, arch, kernel, lever, base_daemon, var_daemon, base_ref, model, kv, maxtok,
            prompt_md5, seeds=None, expects=None):
    """Gate for `var_daemon` vs `base_daemon` (= B_a). Returns the self-describing ledger row.

    Three gates, in order (cheapest-first):
      1. PARITY  — value-preservation (raw-daemon short greedy, token-id EXACT). Value change ->
                   PARITY_FAIL, short-circuit (don't spend perf/coherence on a wrong variant).
      2. PERF    — CONJUNCTIVE: tok/s UP *and* duration DOWN. No gain on either -> DEAD.
      3. COHERENCE — the real user path (serve, thinking ON, sampled, seed-SET + validators).
                   A perf-winner that breaks output -> COHERENCE_FAIL.

    base_daemon/var_daemon double as the base_sha/variant_sha identity on the row (the LiveServeRunner
    passes SHA-named daemon binaries)."""
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    base_sha, var_sha = base_daemon, var_daemon
    common = dict(arch=arch, kernel=kernel, lever=lever, base_ref=base_ref, base_sha=base_sha,
                  var_sha=var_sha, model=model, kv=kv, maxtok=maxtok, prompt_md5=prompt_md5)

    # 1. PARITY (raw-daemon short greedy token-id exact) — short-circuit a value change.
    p_ok, p_detail = V.parity_result(runner.parity_gens(base_daemon), runner.parity_gens(var_daemon))
    if not p_ok:
        return _row(verdict="PARITY_FAIL", parity=p_detail, **common)

    # 2. PERF (conjunctive tok/s UP + duration DOWN) — filter before the expensive coherence arm.
    base_tok, base_dur = runner.perf_measure(base_daemon)
    var_tok, var_dur = runner.perf_measure(var_daemon)
    pr = V.perf_result(base_tok, var_tok, base_dur, var_dur,
                       runner.clocks(base_daemon), runner.clocks(var_daemon))
    perf_kw = dict(tok_delta_pct=pr["tok_delta_pct"], dur_delta_pct=pr["dur_delta_pct"],
                   tok_p=pr["tok_p"], dur_p=pr["dur_p"])
    if pr["verdict"] != "WIN":
        return _row(verdict=pr["verdict"], parity=p_detail, **perf_kw, **common)

    # 3. COHERENCE (serve, sampled — PRIMARY correctness gate) — a perf-winner must stay coherent.
    c_ok, c_detail = V.coherence_result(runner.coherence_gens(base_daemon, seeds),
                                        runner.coherence_gens(var_daemon, seeds), expects=expects)
    final = "WIN" if c_ok else "COHERENCE_FAIL"
    return _row(verdict=final, parity=p_detail, coherence=c_detail, perf_f=None,
                seeds=len(seeds), **perf_kw, **common)


def certify_coherence(runner, *, arch, kernel, lever, base_daemon, var_daemon, base_ref, model, kv,
                      maxtok, prompt_md5, seeds=None, expects=None):
    """Coherence-ONLY gate — the ROLLOVER pass. The per-round certify already enforced parity+perf on
    each win; at FOLD time we run the SERVE coherence pass ONCE on the composed stack vs trunk.
    Returns COHERENT / COHERENCE_FAIL."""
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    c_ok, c_detail = V.coherence_result(runner.coherence_gens(base_daemon, seeds),
                                        runner.coherence_gens(var_daemon, seeds), expects=expects)
    final = "COHERENT" if c_ok else "COHERENCE_FAIL"
    return _row(verdict=final, coherence=c_detail, seeds=len(seeds), arch=arch, kernel=kernel,
                lever=lever, base_ref=base_ref, base_sha=base_daemon, var_sha=var_daemon,
                model=model, kv=kv, maxtok=maxtok, prompt_md5=prompt_md5)
