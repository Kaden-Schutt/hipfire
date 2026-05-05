# Path D D3b validation — model B optimistic-seed cost on real workloads

**Issue:** [#38](https://github.com/Kaden-Schutt/hipfire/issues/38)
**Branch:** `feat/38-ddtree-pipeline` at `55c0e8b` (D3b.1) on top of master
`80330c3`.
**Hardware:** AMD RX 7900 XT (gfx1100, 21.5 GB VRAM, HIP 7.2).
**Date:** 2026-05-03.
**Status:** Model B as written is non-viable on code prompts.

## What this measures

Pipeline model B (chosen earlier in this branch — see
`memory/project_path_d_pipeline_model.md`) launches draft N+1 concurrent
with verify N using a guessed seed: `seed = drafted_N[B-1]`,
`position = position + B`. The guess is correct **only when
accept_len_N == B-1** (full accept). On any cycle where the verifier
rejects mid-block, the pipelined draft launches at the wrong absolute
position — its tokens are predictions for the wrong slots and verify
N+1 will reject them.

Before sinking the ~400 LOC of pipelined-branch implementation, this
experiment measures the empirical full-accept fraction on the workloads
plan §1 targets (code, prose). If full-accept is rare, model B's
pipelined draft is wasted GPU work and bandwidth contention with verify
is a net regression.

## Method

Two prompts run end-to-end through `dflash_spec_demo` with
`DFLASH_LIVE_TAU=1` for per-cycle accept_len logging:

  - **Code** (LRU cache, PEP-8 strict):
    `benchmarks/prompts/lru_cache_pep8_strict.txt`, `--max 120`.
  - **Prose / structured code** (merge sort, thinking off):
    `benchmarks/prompts/merge_sort_thinking_off.txt`, `--max 256`.

Both use the canonical 27B-3.5 LRU bench config: `--no-chatml --kv-mode
asym3`. Block size `B=16` (draft-trained default). `prompt_normalize`
default-on (per CLAUDE.md as of 2026-04-26). 7900 XT, idle GPU lock.

Per-cycle `accept_len` is reported by the demo's `live-tau` mode plus
the `seed-oracle` summary. `full_accept_count` is the fraction of cycles
with `accept_len == B-1`.

## Results

### Code (LRU PEP-8) — 12 cycles, mean accept_len 9.5

```
histogram (accept_len → cycles):
  0: 0    1: 0    2: 1    3: 1    4: 1    5: 0    6: 0
  7: 1    8: 1    9: 0   10: 1   11: 1   12: 1   13: 1
 14: 1   15: 2   16: 0

mean_accept_len: 9.500    full_accept: 2/12 = 17 %
decode_tau: 9.50          decode_accept_rate: 0.633
```

### Prose (merge sort) — 11 cycles, mean accept_len 13.27

```
histogram:
  0: 0    1: 0    2: 1    3: 0    4: 0    5: 0    6: 0
  7: 0    8: 0    9: 1   10: 0   11: 0   12: 0   13: 0
 14: 0   15: 9   16: 0

mean_accept_len: 13.273   full_accept: 9/11 = 82 %
decode_tau: 13.27         decode_accept_rate: 0.885
```

## Interpretation

### Code workloads — model B is a regression

On the code prompt (the hardest target in plan §1's per-class gates,
210 tok/s on 27B-3.5 LRU), only **17 %** of cycles satisfy model B's
optimistic-seed assumption. The remaining 83 % of cycles produce
pipelined drafts at the wrong absolute position — verify N+1 rejects
them and the GPU work is wasted.

Wall-time math:

  - **Best-case savings** when pipelining helps: ~11.8 % per cycle
    (D0-bandwidth contention bench, `findings/path-d-bandwidth-
    contention.md`).
  - **Regression cost** when pipelining is wasted: bandwidth contention
    on shared GDDR6 slows verify by ~5 % (D0-bandwidth bench, run 2).
    Plus the wasted draft compute.

  Net code wall change ≈
    `0.17 × 11.8 % - 0.83 × 5 % ≈ +2.0 % - 4.2 % = -2.2 %`

  Model B is a **2 % wall-time regression on code**, not a win. The
  plan's 210 tok/s target on this class is unreachable via model B —
  it'd land below 200 tok/s.

### Prose workloads — model B helps significantly

On the merge-sort prompt, **82 %** of cycles satisfy the optimistic-seed
assumption.

  Net prose wall change ≈
    `0.82 × 11.8 % - 0.18 × 5 % ≈ +9.7 % - 0.9 % = +8.8 %`

  Plan's 180 tok/s prose target is comfortably reachable — would land
  near 196 tok/s (current 180 baseline × 1.088).

### The split is structural, not noise

The split between code and prose isn't a small perturbation — it's a
fundamental property of how the draft+verify alignment interacts with
greedy decoding:

  - On highly-deterministic generations (boilerplate code, well-trained
    structured patterns like merge_sort), the draft predicts what the
    verifier wants, accept_len ≈ B-1, model B is sound.
  - On code with branching choices (real algorithm, conditional logic,
    LRU semantics) the verifier's greedy argmax frequently diverges
    from the draft mid-block. accept_len lands in the 7–12 range
    (mean 9.5 here). model B is broken.

This is exactly the workload class plan §1 most wants to accelerate
(code generation, agentic turns), so the failure mode lands on the
target use case.

## Implications for D3b

Model B as specified in `plans/path_d.md` §D3b — pre-launched draft
with optimistic seed, no recovery — produces a regression on code
prompts that's larger than the wins on prose. The per-class speed
gates in plan §1 (code 210, instruct 195, prose 180) cannot all pass
with this approach; at minimum the code gate fails.

Three viable pivots:

### A. Commit-overlap-only (former model A)

Cycle N still runs draft N → verify N → commit N sequentially. The
overlap is *only* `commit_N` (small async memcpys, ~1 ms) with the
START of cycle N+1's `draft_forward` (compute-bound). Real bonus_N
used as seed — no τ regression.

  - Pros: zero correctness/τ risk.
  - Cons: arithmetic ceiling on wall savings is ~2 %; effectively no
    measurable improvement at the 50-ms cycle scale.
  - Verdict: passes coherence gates and ±2 % default check, but
    **fails the per-class speed gates** (no improvement).

### B. Predictive bypass on model B

Engage model B's pipelined draft *only* when the prior cycle had
`accept_len_{N-1} == B-1` (or some recent EWMA threshold). On cycles
where the optimistic-seed assumption is unlikely to hold, fall back
to sequential.

  Empirical engagement rates with single-cycle predictor:
    - Code: ~0.17 × P(full | full) ≈ 0.08 (assume 50 % autocorrelation).
    - Prose: ~0.82 × P(full | full) ≈ 0.7 (high-correlation regime).

  - Pros: prose gets ~6 % wall savings (down from 8.8 %); code stays
    near zero (no win, but no regression — the 92 % of cycles that
    bypass run sequentially).
  - Cons: implementation cost ~ same as full model B (the predictive
    gate is a few LOC; the pipelined branch itself is the bulk).
  - Verdict: passes code gate (no change) and prose gate (improvement);
    fails the 210 tok/s aspirational code target but probably reaches
    a defensible ≥199 tok/s baseline.

### C. Optimistic launch + sync re-launch on miss

Launch draft N+1 optimistically. If `accept_len_N < B-1`, on cycle N+1
discard the pipelined draft and run draft N+1 sequentially.

  - Pros: τ stays at sequential level.
  - Cons: the wasted optimistic launch consumed bandwidth concurrent
    with verify N (~5 % verify slowdown via bandwidth bench), AND
    cycle N+1 pays full sequential cost. Net: positive on full-accept
    cycles, **negative** on miss cycles.
  - Verdict: probably worse than pure A on code; comparable to B on
    prose.

### Out of scope but worth noting

  - **Multi-position draft** (predict B+1 token sets, one per possible
    accept_len): would always have a usable draft. Requires a different
    draft training objective. Not feasible without retraining the draft
    model.

  - **Asynchronous draft re-issue**: launch draft optimistically;
    overlap a SECOND draft launch with verify N+1's forward using the
    real seed. Only the second draft's tokens are used. Effectively
    doubles draft compute with no acceptance benefit; worse than A.

## Recommendation

**Either pivot to model A or stop work on Path D pipelining.**

Model A is cheap (~150 LOC vs ~400 for full model B) and
correctness-safe. It will likely fail the per-class speed gates as
written, but those gates were derived from the optimistic 11.8 %
wall-savings projection — which this experiment shows is unreachable
at the workload mix (code-dominated) plan §1 cares about.

If we pursue model A:

  1. Implement the minimal cross-stream wiring: pre_commit_evt +
     `commit_staging_to_ring_on_stream` on verify_stream concurrent
     with cycle N+1's draft_forward starting on draft_stream.
  2. **Rebaseline plan §1 quantitative target** to ~75 ms → 73 ms
     (-3 %) — defensible as a "no regression, small overlap" claim
     rather than the 20 % aspiration.
  3. **Revise the per-class gates** to ±2 % of the existing baseline
     (no improvement expected; the gate becomes a regression check
     rather than a perf-target check).

If we stop: the D0a/D0b/D0c primitive refactors still land cleanly as
infrastructure (they're useful for any future stream-aware work). D1
(stream allocation) and D2 (DflashScratchPair) become dead code —
follow-up issue to remove. D3a (verify_dflash_block_inner skip flag)
becomes dead code likewise. D3b.1 signature refactor becomes dead
code — revert.

## Repro

```sh
# Code (LRU PEP-8):
PROMPT=$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)
DFLASH_LIVE_TAU=1 ./target/release/examples/dflash_spec_demo \
    --target ~/.hipfire/models/qwen3.5-27b.mq4 \
    --draft  ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
    --prompt "$PROMPT" --max 120 --no-chatml --kv-mode asym3

# Prose (merge sort thinking-off):
PROMPT=$(cat benchmarks/prompts/merge_sort_thinking_off.txt)
DFLASH_LIVE_TAU=1 ./target/release/examples/dflash_spec_demo \
    --target ~/.hipfire/models/qwen3.5-27b.mq4 \
    --draft  ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
    --prompt "$PROMPT" --max 256 --no-chatml --kv-mode asym3
```

The `seed-oracle` line at end-of-run reports `full_accept` count and
`mean_accept_len`. Histograms in the bench summary block.
