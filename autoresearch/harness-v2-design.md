# Autoresearch Harness v2 — design spec

Status: DRAFT for review (2026-07-04). Supersedes the v1 loop (`loop_driver_v2.sh` +
`rollover.sh` + `ab_certify_swarm.sh`) once approved. v1 produced `baseline_v2`
(≈+17.4% a3b-mq4r decode on gfx1201) and, more importantly, the data that motivates v2.

## Why v2 (what v1 taught us)

v1 works but has three structural limits, all confirmed from its own ledger (144 rows):

1. **Fixed magnitude floor discards real wins.** WIN requires `delta > 0.3%` AND
   `f ≥ 0.90`. But the ledger shows the discarded value isn't *small* wins (there are
   **zero** consistent sub-0.3% positives — f≥0.85 count = 0), it's **high-variance
   marginal-big wins**: `gemm_grouped +2.97% (f=0.78)`, `fused_qkvza +2.06% (f=0.69)`
   — real 2–3% effects that a fixed 4/8-round A/B can't resolve because these big
   kernels have wide timing variance that dilutes `f`. We're losing 2–3% wins to
   **insufficient sampling**, not to the floor.
2. **Fixed run/time limit is the wrong stop condition.** The right condition is
   *per-kernel exhaustion* → drive every kernel to its roofline, then stop.
3. **Loose-file provenance is fragile.** `/tmp/wins/*.hip` + shared source reverts
   caused a real bug (untracked stray folded into a baseline; a card left on the old
   baseline). And ~6 WIN verdicts had no saved `.hip` → unfoldable.

## Locked decisions

- **Acceptance = significance, not magnitude.** Replace the fixed-rounds + 0.3% floor
  with **adaptive sampling**.
- **Per-agent wins-only branches + JSONL log** (not full-log branches).
- **Self-exhausting** — no fixed round/time limit; stop on global exhaustion.
- **Ledger-persistent + tunable** — a re-run reads the corpus (doesn't re-skip
  everything) but thresholds are tunable so it can re-examine under new settings.
- Build as v2 for the **next** campaign (not hot-swapped mid-run).

## Core change 1 — Adaptive sampling (acceptance)

Per candidate variant, instead of a fixed 4 (+4 confirm) rounds:

```
sample 4 rounds -> compute f (MWU dominance)
while  0.65 < f < 0.90  and  rounds < CAP(=16):
    sample 2 more rounds ; recompute f
verdict: f>=0.90 -> WIN ; f<=0.65 -> DEAD ; else (hit CAP, unresolved) -> INCONCLUSIVE
```

- Resolves the real 2–3% marginals (`gemm`/`qkvza`) by *measuring harder*, without
  admitting noise (a coin-flip effect stays f≈0.5 forever and hits CAP → not banked).
- Keeps clock-match + coherence gating exactly as v1 (a fast-but-incoherent or
  clock-skewed variant is still killed regardless of f).
- **Delta floor drops to ~just-above-clock-noise** (e.g. 0.15%) since `f` now carries
  the reliability — but INCONCLUSIVE candidates are *not* banked (they log a negative).
- Cost: hard-kernel rounds get longer (that's the trade — resolution over throughput).
  `CAP` bounds it so a true-noise candidate can't sample forever.

## Core change 2 — Per-agent wins-only branches

- 4 worktrees, each on its **own branch** `loop/card{0..3}` (not shared detached HEAD).
- Each certify: clean to baseline → apply variant → `git add -A && commit` (tracks the
  file, **no untracked stray possible**) → adaptive-sample → annotate commit w/ verdict.
  - **WIN** → the commit stays on `loop/cardN` as a **provisional win** (not baseline).
  - **DEAD/INCONCLUSIVE** → `git reset --hard baseline` (clean); the negative + its
    "why" go to the **JSONL** (the full log lives there, branch stays wins-only).
- Fixes the stray/collision class structurally, and every WIN has a committed source
  (no lost wins).

## Core change 3 — Banking stays at rollover (filter + can-overturn)

Unchanged in spirit from v1, now git-native:
- Rollover harvests the standing win-commits from `loop/card{0..3}`, best-per-kernel.
- **Composed re-measure is the filter/overturn** — survivors fold into `baseline_v{N+1}`;
  the rest stay on their branch as records, never merged (explicit overturn).
- After fold: force-align all agent branches to the new baseline, **re-census** (bod
  refresh — already baked into v1's rollover), append manifest.
- Composed re-measure remains the **anti-phantom guard**: adaptive sampling can be
  aggressive because nothing "counts" until the composed baseline confirms it.

## Core change 4 — Self-termination + roofline metric

- **Per-kernel exhaustion:** a kernel is EXHAUSTED when K consecutive diverse
  adaptive-sampled attempts all resolve DEAD/INCONCLUSIVE. Mark it, drop it from the
  target pool.
- **Global stop:** when every kernel above a wall%-threshold is EXHAUSTED, the run
  terminates itself — no fixed limit.
- **Computed roofline ceiling (the "are we there" number):** from the census, sum each
  kernel's theoretical-min time (traffic ÷ peak BW for mem-bound; else compute roofline)
  and invert. Report `current / ceiling` each rollover. When measured baseline ≈ ceiling
  AND all kernels exhausted, we can *claim* near-roofline — not by vibes.

## Core change 5 — Tried-levers digest (dedup)

- v1 re-treads (gemm 15 / qkvza 14 attempts). Codex re-reads raw JSONL.
- v2 injects a per-kernel **digest** into the generate prompt: `{kernel: [levers tried
  → verdict]}`, so Codex avoids re-deriving dead levers and gets a sharp "this kernel is
  near-exhausted" signal.

## Reused from v1 (do NOT rebuild)

- `rollover.sh` compose→gate→advance→**census** machinery (hardened: clean-strays,
  force-align, verify-all-4, `sudo -n` detached census). Adapt the *harvest source*
  (branches vs `/tmp/wins`) only.
- `oracle_profile.sh` census (`sudo -n`, profile_standard PMC + kernel-trace wall%).
- MWU `f` + clock-match + coherence (`var_coh`) gating.
- Homogeneous-only guard, gap-anti-thrash, folded-list prompt injection.
- On-box detached driver via a script (script-on-box beats inline-nohup FD-hang).

## Tunability / re-run semantics

- Corpus (ledger + branches + baselines) persists. A re-run:
  - reads exhaustion state so it doesn't re-hammer exhausted kernels,
  - BUT `--reexamine` (or a bumped CAP / lowered floor) lets it re-open kernels under
    tighter sampling — i.e. tune without wiping, exactly like v1's ledger-awareness.

## Open (for reviewer)

- `CAP` value (16?) and exhaustion `K` (3 consecutive?) — tune from v1 variance data.
- Whether INCONCLUSIVE-at-CAP candidates get one escalated re-visit later (bigger CAP)
  or are parked.
- Roofline-ceiling model fidelity (simple traffic÷BW vs per-kernel measured L2/DRAM mix).
