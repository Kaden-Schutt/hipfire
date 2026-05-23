# MTP GDN-tape replay fix — new baseline (Goal A starting point)

**Date:** 2026-05-23
**Branch:** `mtp-hiptrx-rocprof`
**Hardware:** k9lin / 7900 XTX (gfx1100), Qwen3.6-27B MQ4 trunk.
**Prompt:** `benchmarks/prompts/lru_cache_pep8_strict.txt`, md5 `1e74f17934fe759468dbe1471b732067`.

## The fix (Codex diagnosis, Claude empirical verification)

The MTP compressed-serial cycle, on every NON-full-accept cycle (~25-35% of
cycles at the canonical recipe), ran a *full trunk forward replay* — all 64
layers' GEMM + attention — purely to repair the DeltaNet recurrent state so it
reflects only the accepted prefix. DFlash long ago solved the identical problem
with an innovation "GDN tape": capture the conv/DeltaNet recurrence during the
verify forward, and on rollback restore the pre-verify state and replay ONLY the
cheap GDN recurrence for the accepted prefix — no GEMM/attention re-run.

The MTP path simply never adopted DFlash's tape. Codex's change (`mtp_spec.rs`,
two sites): allocate a `trunk_gdn_tape: GdnTape` in `MtpSpecState`, pass it as
the tape-capture argument to the verify `forward_prefill_batch_with_pbs`, and on
non-full-accept replace the full `forward_prefill_batch`/`forward_scratch` replay
with `trunk_gdn_tape.replay_gdn(...)`. Pure reuse of proven DFlash primitives.

This was Claude's session-long blindspot: the replay was visible in the cycle
trace and dismissed as "~3 ms amortized" when it was actually the dominant
non-verify cost (~15-20 ms per non-full-accept cycle). The user's hypothesis B —
"the agent is spinning its wheels missing a lever hidden in plain sight" — was
correct. Codex's GPU-less sandbox root-caused it from the in-repo profile docs.

## Empirical verification (the bench methodology this goal mandates)

Back-to-back A/B, identical warm conditions, 5-run fresh-process medians, canonical
recipe (K=2 here for the isolated A/B; the GDN change is what is being measured):

```
Pre-fix  (git-stash to baseline code, rebuilt): 53.20 tok/s median, τ≈2.6
Post-fix (Codex GDN tape):                      61.99 tok/s median, τ≈2.6
Delta:                                          +16.5%, τ UNCHANGED
```

τ-invariance is the mechanism proof: the fix changes *how* state is repaired on
partial-accept cycles, not *which* tokens are accepted. Pure cycle-wall reduction.
Well outside the ±1-3% session noise band; back-to-back so not drift.

Coherence: clean. MTP battery at the K=3 recipe across code (LRU), prose
(trains-meet), and agentic (multistep) prompts — all fluent, on-topic, no token
attractor or loop. The mandated `coherence-gate-dflash.sh --fast` (shared-`GdnTape`
infra regression check on the unchanged DFlash path) exited 0, no hard errors.

## K-sweep finding — K=3 is the new sweet spot

With the rollback cost decoupled from chain length, higher K finally pays off
(it never did pre-fix, because each partial-accept cycle's full replay penalty
scaled with K and cancelled the higher-τ benefit). p_min held at 0.65, 3-run
medians, post-fix:

| K | Median tok/s | τ | vs AR (44.6) |
|---|---|---|---|
| 2 | 60.4 | ~2.6 | 1.35x |
| **3** | **68.3** | **~3.1** | **1.53x** |
| 4 | 66.6 | ~3.0 | 1.49x |
| 5 | 65.9 | ~3.0 | 1.48x |
| 6 | 66.9 | ~3.0 | 1.50x |

Every K=3 run beat every K=2 run despite wide per-cell variance. K=4-6 plateau:
marginal extra drafts cost MTP-head forward time + hit more p_min truncation
without proportional acceptance gain.

## New canonical MTP-solo recipe (Goal A starting baseline)

```
--max-n 3 --mtp-p-min 0.65 --compressed-serial --temp 0.0 --no-chatml --kv-mode q8
prompt md5 1e74f17934fe759468dbe1471b732067, max=480
```
Expected ~68 tok/s = **1.53x AR** — into the published reference 1.5-3x band.

## The arc so far

```
AR baseline:                  44.6 tok/s   (1.00x)
MTP K=2, pre-GDN-fix:         ~53 tok/s    (1.19x)  ← session was stuck here, ~1.1-1.2x
MTP K=2, post-GDN-fix:        ~60 tok/s    (1.35x)
MTP K=3, post-fix (baseline): ~68 tok/s    (1.53x)  ← Goal A starts here
Goal A floor: 100 tok/s; target 120.
```

## Falsified / dead-end levers (do not re-walk)

- **MTP-head weight retraining** (v1-v7 this session: single-step CE, code-heavy
  CE, KL, recursive K=2/3/5). Offline argmax agreement improved +15-18pp but
  runtime tok/s was flat-to-negative — the offline metric does not transfer
  through the runtime gating. **The wins are in runtime/dispatch structure, not
  weights.** Documented at `docs/plans/mtp-fastmtp-v1-v5-complete-2026-05-22.md`
  and `mtp-fastmtp-v6-v7-progress-2026-05-22.md`.

## Goal A candidate levers (next, cheapest-first; runtime/dispatch only)

1. Profile the K=3 cycle to find the new dominant cost now that replay is cheap.
2. hipGraph capture of the K-cycle to eliminate per-cycle host round-trips
   (per-draft argmax D2H + sampling sync serialize the chain).
3. p_min micro-tune around K=3 (Dre Dyson pairing suggests slightly lower p_min
   with higher K).
4. Larger verify-tile fill — the trunk verify at M=K+1=4 still under-fills the
   16×16 WMMA tile; anything that packs the verify batch closer to 16 is free
   throughput (this is the structural lever DFlash exploits).
