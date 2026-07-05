# v3-queue: brainstorm-then-implement autoresearch loop

The generation of the loop that produced **baseline_v3** (attention register-ring,
+6.95% gfx11 / +2.9% gfx12 a3b decode, byte-exact). It splits the old open-ended
`driver_v3` loop (which blindly generated kernel variants → a bajillion loser
certify-runs) into two phases with a clean division of labor:

```
JETFUEL — Claude / ultracode Workflow (creative, cheap LLM tokens, NO GPU)
  R1 brainstorm: 12 lenses → synthesize → 3-way adversarial refute      (autoresearch/workflows/…-wf_a7f…js)
  R2 deepen / combine / rescue / gap-fill on R1                          (…-r2-wf_da5…js)
  R3 harden → certify queue: red-team ISA & coherence, impl-specs,       (…-r3-wf_66c…js)
     Amdahl-sanity-check deltas, bake the measurement protocol
        ↓  certify_queue.json  (vetted, readiness-tagged, honest deltas)
IGNITION — Codex loop (implementation, shrewd/token-stingy, autonomous subscription)
  queue_driver.sh — resume-mode, drain detected from the PROGRESS LOG (not codex prose),
                    stall-guard, then an AUTONOMOUS FOLD (rollover_v2) at the end
    each fresh round, codex:
      read queue → IMPLEMENT the next item from its spec (do NOT brainstorm)
      → ISA precheck (skip GPU if the disasm says no-op / wave-dropping regression)
      → ab_certify_v2 (adaptive MWU A/B, clock-VOID resample, isolate-decode-window, coherence gate)
      → PERMUTATION SWEEP if verbatim misses — codex tunes the parameter space Claude framed
        (prefetch depth, ring P, which-loads, VGPR budget), bounded to ≤3 tuned shots
      → commit WIN → loop/card<N>
        ↓
  rollover_v2 (default ADVANCES now) → composes branch wins → A/B-gates → advances
  loop_baseline_$ARCH → re-censuses the BOD → the next brainstorm's input.  Versions
  are PERSISTENT per-arch (baseline_manifest_$ARCH.txt): v1 → v2 → v3 → v4 …
```

## The division is not two exclusive categories
Claude opines the **frame** (which loops, which mechanism, *why* it hides latency).
Codex is shrewd **within** it — it won't over-spend, and it explores permutations of
that frame efficiently (the sweep). The v3 win proved the frame dominates: MLP levers
win **only** on loops the compiler can't auto-pipeline (attention's `__shfl_xor`-serialized
score/V loops, moe_down's OOB-guarded tail) — every compiler-pipelineable gather
(rmsnorm, residual, gfx12 multirow) correctly no-op'd. Codex then found the right `P=4`
inside that frame.

## Hard-won config (footguns removed)
- `rollover_v2` **advances by default**; `DRY_RUN=1` is opt-in preview only (a default-dry-run
  fold silently refuses to commit — a liability, not a safety).
- drain is detected from the progress log, never from codex's reasoning text (which echoes
  prompt words like "queue_drained" and caused a false early-terminate).
- lineage manifest is **persistent + per-arch** so baseline versions continue (never restart at v1);
  the anti-thrash gap is bypassed with `MIN_ROUNDS_GAP=0` + a high round arg.
- `MIN_NAIVE_SUM=2.0` so real-but-small (e.g. +2.9%) coherent wins still compound.

## Files
`queue_driver.sh` codex loop + autonomous fold · `loop_round_prompt_queue.txt` the codex
prompt (implement + precheck + bounded permutation sweep) · `rollover_v2.sh` fold/advance/re-census
· `ab_certify_v2.sh` adaptive coherence-gated A/B certify · `oracle_profile.sh` rocprof census
· `stack_verify.sh` fresh isolate-decode-window stack A/B · `gen_digest.py`/`check_exhausted.py`/`update_exhaustion.py`/`campaign_stats.py`
helpers · `example_certify_queue_v3.json` the v3 queue that produced the attention win.

## Measurement discipline (baked into the certify protocol)
Isolate the decode window (`dwall = tokens/decode_tok_s`, sample only the last dwall seconds —
a whole-run busy median is ~35% load/gap garbage). Adaptive-resample to beat gfx11's ~30%
DPM clock-VOID. Byte-identical prompt (record md5). Coherence/token-id-parity gate on any
numeric change. Verify combined stacks with `stack_verify.sh`, not by summing solo deltas.
