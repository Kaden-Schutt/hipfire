# Path D vs Path C — chain mode vs tree mode on 27B-3.5

**Issue:** [#38](https://github.com/Kaden-Schutt/hipfire/issues/38) (Path D)
in light of [#41](https://github.com/Kaden-Schutt/hipfire/issues/41) +
PR #72 (Path C tree-mode orchestrator on gfx1100).
**Branch:** `feat/38-ddtree-pipeline` at `b38ce06` on top of master `80330c3`.
**Hardware:** AMD RX 7900 XT (gfx1100, 21.5 GB VRAM, HIP 7.2).
**Date:** 2026-05-03.

## Why this exists

The model-B validation in `findings/path-d-model-b-validation.md`
measured chain-mode `spec_step_dflash` only. The PR's recommendation
hedged: *"if Path C tree mode lifts the full-accept fraction enough,
the chain-mode regression on code might not apply to the production-
default path."* This experiment closes that loop with empirical
chain-vs-tree data on the same Qwen3.5-27B + DFlash MQ4 draft pair.

## Method

Same prompts as the chain-mode validation:

  - Code: `benchmarks/prompts/lru_cache_pep8_strict.txt`, `--max 120`.
  - Prose: `benchmarks/prompts/merge_sort_thinking_off.txt`, `--max 256`.

All runs: `--no-chatml --kv-mode asym3` (canonical 27B-3.5 LRU bench
config, plan §1). DPM warmup 3 s for representative tok/s.

Three modes:

  - **Chain** — default `spec_step_dflash` (B=16 chain block).
  - **Path C phase 1** — `--ddtree-budget 12 --ddtree-topk 2
    --ddtree-path-c phase1`. Tree mode, main-path-first linear verify.
  - **Path C phase 2** — same flags with `phase2`. Adds lazy
    branch FA-only re-verify on the unique structurally-acceptable
    candidate.

## Results

### Code prompt (LRU PEP-8)

| Mode | tok/s | τ | Full-path-accept | Mode-specific full-block size |
|---|---:|---:|---|---:|
| Chain                  | **153.75** | 8.85 | 2/13 = 15 % | 15 |
| Path C phase 1         | 138.64     | 8.69 | 5/13 = 38 % | 12 |
| Path C phase 2         | 131.48     | 8.69 | 3/13 = 23 % | 12 |

### Prose prompt (merge sort)

| Mode | tok/s | τ | Full-path-accept | Mode-specific full-block size |
|---|---:|---:|---|---:|
| Chain                  | **228.99** | 13.27 | 8/11 = 73 % | 15 |
| Path C phase 1         | 177.55     | 11.08 | 11/13 = 85 % | 12 |
| Path C phase 2         | 177.29     | 11.08 | 11/13 = 85 % | 12 |

(Full-path-accept counts cycles where the longest accepted path equals
the mode-specific maximum: B-1 = 15 for chain, budget = 12 for Path C.)

### Sanity check: env=both (chain pipelining + tree mode)

`HIPFIRE_DFLASH_PIPELINE=1 ... --ddtree-path-c phase1` on the code
prompt: 138.70 tok/s, τ=8.69. **Within run-to-run noise of Path C
phase 1 alone** (138.64 / 8.69). Confirms the demo dispatches to
`spec_step_ddtree_path_c` when `--ddtree-path-c` is set; our chain-
mode pipelining never engages. The pair's `b` half is allocated
(~64 MB extra VRAM, observed bump 16.56 → 17.69 GB) but unused.

## Interpretation

### Path C is slower than chain on 27B-3.5 in our config

  - **Code:** Path C phase 1 is **−10 %** vs chain (138.64 vs 153.75
    tok/s). Phase 2 is even slower at −15 %.
  - **Prose:** Path C phase 1 is **−22 %** vs chain (177.55 vs 228.99
    tok/s).

This is consistent with #41's TL;DR observation:

> "DDTree b12-k2 on 27B-3.5 code = 109 tok/s vs plain DFlash 185 tok/s
> (-41 %)."

Path C unblocks the *correctness* problem (no attractor failure mode
that broke Paths A and B per #41). It does **not** deliver perf wins
on Qwen3.5-27B at the canonical LRU config — the tree-mode per-cycle
overhead exceeds the τ benefit. This holds even though full-path-
accept fraction is higher under Path C (38 % code, 85 % prose vs
chain's 15 % / 73 %).

The +30–40 % τ wins for tree mode cited in #41's body ("memory:
asym3 KV b12-k2 wins +28 % prose / +29 % instruct") were
**pre-Phase-1** numbers — measured before the Phase 1 prompt
normalization (default-on as of 2026-04-26 per `CLAUDE.md`) raised
chain-mode τ on PEP-8 / structured prompts by ~24 %. Chain mode in
the post-Phase-1 regime is a stronger baseline than the historical
DDTree comparisons assumed.

### Implications for the Path D PR's recommendation

The PR's hedge — "if Path C becomes production default, model B's
chain-mode regression might not apply" — is **empirically refuted**.
Path C is slower than chain mode on Qwen3.5-27B at canonical config,
so users won't migrate to it as a default for these models. Chain
mode remains the production-default path, and the chain-mode model-B
finding (regression on code, win on prose) is the relevant data.

This **strengthens option C** (stop pursuing pipelining):

  - Chain mode is the production-default path on Qwen3.5-27B.
  - Pipelining model B regresses on chain-mode code by ~2.2 %
    (`findings/path-d-model-b-validation.md`).
  - Path C tree mode doesn't change this conclusion — it's a slower
    alternative on the model pair plan §1 targets.

Path C remains useful for Qwen3.6 + matching draft pair (per #41
comment's smoke results, where chain-mode τ is much lower at ~1.5–8
and Path C's tree-mode wins materialize). But that's a different
model regime than plan §1's 27B-3.5 LRU bench targets.

### Implications for tree-mode pipelining (P3 follow-up)

The full-path-accept rates under Path C are higher than chain mode
(38 % vs 15 % on code, 85 % vs 73 % on prose). If we *did* pipeline
Path C with model B, the optimistic-seed assumption would hold more
often.

But the per-cycle wall time is also longer in Path C (tree-verify
overhead). The pipelining wall-savings calculation:

  - Path C code: `0.38 × ~12 % overlap - 0.62 × ~5 % contention
    ≈ +4.6 % - 3.1 % = +1.5 %`. Marginal positive.
  - Path C prose: `0.85 × ~12 % - 0.15 × ~5 %  ≈ +10.2 % - 0.75 % =
    +9.4 %`. Solid positive.

But the absolute tok/s ceiling under Path C is already lower than
chain mode (138 / 178 vs 153 / 229). Even with a +9 % wall savings
on Path C prose, you'd land at ~194 tok/s — still under chain mode's
229. **Tree-mode pipelining would not catch chain mode** on
Qwen3.5-27B prose.

So tree-mode pipelining on this model regime is structurally
dominated by chain mode. Worth pursuing only if there's a model
regime where Path C beats chain (e.g. Qwen3.6 + matching draft per
#41 comment data) — and even then, the engineering cost is non-
trivial since `spec_step_ddtree_path_c` would need a D3b-equivalent
refactor.

## Recommendation update

**Lock in option C** (stop pursuing pipelining). The "Path C might
flip the picture" hedge in the PR is empirically refuted on the
model regime that matters. Path C is correct but slower; users won't
adopt it as default; chain-mode pipelining model B's chain-mode
regression is the real data; that data argues against shipping Path
D as the plan §1 perf lever.

Specific PR actions:

  1. **Recommendation flips from "C with a hedge" to "C, locked in."**
  2. **Strike the test plan bullet** asking to re-run model-B
     validation under Path C — done, results above.
  3. **Plan rebaseline** under option C stays as `75 ms → 75 ms (0 %)`,
     no perf claim. Plan §1's 210 / 195 / 180 tok/s gates were
     aspirational under the assumption pipelining would deliver
     11–20 %. They're unreachable on this hardware/model pair via
     either pipelining (this experiment + chain validation) or Path
     C alone (this experiment).
  4. **Plan §6 (Out of scope)** should add: "Tree-mode pipelining
     on `spec_step_ddtree_path_c`. Empirically dominated by chain
     mode on 27B-3.5 (this finding); only worth pursuing on model
     regimes where Path C beats chain."

## Repro

```sh
# Chain mode (baseline):
HIPFIRE_DPM_WARMUP_SECS=3 ./target/release/examples/dflash_spec_demo \
    --target ~/.hipfire/models/qwen3.5-27b.mq4 \
    --draft  ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq \
    --prompt "$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)" \
    --max 120 --no-chatml --kv-mode asym3

# Path C phase 1 (add):
    --ddtree-budget 12 --ddtree-topk 2 --ddtree-path-c phase1

# Path C phase 2 (substitute):
    --ddtree-budget 12 --ddtree-topk 2 --ddtree-path-c phase2

# env=both sanity:
HIPFIRE_DFLASH_PIPELINE=1 [chain mode flags] [Path C flags]
```

Histograms read from the bench summary block at end-of-run.
Tok/s + τ from the `emitted:` and `cycles:` lines.
