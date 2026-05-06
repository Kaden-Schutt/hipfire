# Path C on Qwen3.6 — does the lower-τ regime flip the picture?

**Issue:** [#151](https://github.com/Kaden-Schutt/hipfire/issues/151) (follow-up to
[#131](https://github.com/Kaden-Schutt/hipfire/pull/131) /
[#38](https://github.com/Kaden-Schutt/hipfire/issues/38) close-out).
**Branch:** `feat/issue-151-qwen36-pathc` at `262e5f6` (master).
**Hardware:** AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, Strix Halo APU,
137 GB shared LPDDR5x, ROCm 7.12).
**Date:** 2026-05-06.

## Why this exists

`findings/path-d-vs-path-c.md` (gfx1100, Qwen3.5-27B) closed pipelining as a dead
end *on that regime* but explicitly hedged:

> "Tree-mode pipelining on this model regime is structurally dominated by chain
> mode. Worth pursuing only if there's a model regime where Path C beats chain
> (e.g. Qwen3.6 + matching draft per #41 comment data)."

The claim from #41 is that chain-mode τ on Qwen3.6 + matching draft is much
lower (~1.5–8) than Qwen3.5 LRU's τ ~10. Lower chain-mode τ → more draft
work per accepted token → tree mode (Path C) has more τ headroom to reclaim.
This finding tests that hypothesis empirically on Qwen3.6-27B + the
co-shipped DFlash draft.

## Method

Mirrors `findings/path-d-vs-path-c.md` methodology with the model pair changed:

  - Target: `~/.hipfire/models/qwen3.6-27b.mq4` (15 GB, 64-layer Qwen3.6 with
    1:4 LinearAttention:FullAttention pattern, KV asym3, max_seq=544).
  - Draft: `~/.hipfire/models/qwen36-27b-dflash-mq4.hfq` (919 MB, 5-layer
    co-shipped DFlash draft with `target_layers=[1, 16, 31, 46, 61]`, MQ4
    weights with FWHT rotation).
  - Code prompt: `benchmarks/prompts/lru_cache_pep8_strict.txt`, `--max 120`.
  - Prose prompt: `benchmarks/prompts/merge_sort_thinking_off.txt`, `--max 256`.

All runs: `--no-chatml --kv-mode asym3 HIPFIRE_DPM_WARMUP_SECS=3` (canonical
post-Phase-1-norm bench config). Each run is a fresh process (model load
~125 s + DPM warmup 3 s + bench).

Three modes:

  - **Chain** — default `spec_step_dflash` (B=16 chain block).
  - **Path C phase 1** — `--ddtree-budget 12 --ddtree-topk 2 --ddtree-path-c phase1`.
    Tree mode, main-path-first linear verify.
  - **Path C phase 2** — same flags with `phase2`. Adds lazy branch FA-only
    re-verify on the unique structurally-acceptable candidate.

## Results

### Code prompt (LRU PEP-8)

| Mode | tok/s | τ | accept_rate | full-path-accept | Mode-max bin |
|---|---:|---:|---:|---|---:|
| Chain          | **36.73** | 5.737 | 0.383 | 1/19 =  5 % | 15 |
| Path C phase 1 | 34.51     | 5.722 | 0.381 | 0/18 =  0 % | 12 |
| Path C phase 2 | 29.20     | 5.316 | 0.354 | 0/19 =  0 % | 12 |

(Mode-max bin: B-1 for chain, budget for Path C. Full-path-accept counts
cycles where the longest accepted path equals the mode-specific maximum.)

### Prose prompt (merge sort, thinking-off)

| Mode | tok/s | τ | accept_rate | full-path-accept | Mode-max bin |
|---|---:|---:|---:|---|---:|
| Chain          | **69.36** | 9.500 | 0.633 |  7/16 = 44 % | 15 |
| Path C phase 1 | 65.20     | 9.500 | 0.633 | 10/16 = 63 % | 12 |
| Path C phase 2 | 59.32     | 9.125 | 0.608 |  8/16 = 50 % | 12 |

Both prose runs hit `<|endoftext|>` after 169 / 169 / 163 tokens (model
finishes the function naturally before max=256), so the prose comparison
is on equal-length output.

## Interpretation

### Path C is slower than chain on Qwen3.6-27B at canonical config

  - **Code:** Path C phase 1 is **−6.0 %** vs chain (34.51 vs 36.73 tok/s);
    phase 2 is **−20.5 %** (29.20 vs 36.73).
  - **Prose:** Path C phase 1 is **−6.0 %** (65.20 vs 69.36); phase 2 is
    **−14.5 %** (59.32 vs 69.36).

Same structural conclusion as `findings/path-d-vs-path-c.md` on Qwen3.5-27B
(gfx1100): tree-verify per-cycle overhead exceeds the τ benefit, even when
the full-path-accept fraction rises (prose: 44 % chain → 63 % Path C
phase 1).

### The "Qwen3.6 lower-τ regime might flip the picture" hypothesis is refuted

The premise survives — chain-mode τ on this Qwen3.6 + draft pair is **τ=5.74
on code** (cleanly inside #41's "1.5–8" range, vs Qwen3.5's ~10) and **τ=9.5
on prose** (top of that range). Code is the lower-τ regime where Path C had
the most room to win.

The hypothesis fails empirically:

  - τ does not lift meaningfully under Path C on either prompt.
    - Code: chain 5.737 → phase 1 5.722 → phase 2 5.316. Flat or
      regressive.
    - Prose: chain 9.500 → phase 1 9.500 → phase 2 9.125. Flat or
      regressive.
  - Even where full-path-accept rises (prose 44 → 63 %), absolute tok/s
    falls because the per-cycle wall-time penalty exceeds the τ headroom.

So the Qwen3.6 regime does not unlock a Path C win on this hardware/draft
pair.

### Cross-arch caveat (gfx1151 vs gfx1100)

This bench was run on gfx1151 (Strix Halo APU, shared LPDDR5x). The
prior `findings/path-d-vs-path-c.md` measurements are on gfx1100 (RX
7900 XT, GDDR6). Absolute tok/s differs by ~4×, which is expected
hardware delta — the comparison here is **within-host** (chain vs Path C
on the same gfx1151 machine in the same session), not cross-arch.

Whether Path C might still beat chain on Qwen3.6-27B + 27B-draft on a
*different* host class (e.g. gfx1100 / 7900 XT) is not directly answered
by this finding. However, the gfx1100 / Qwen3.5 result already shows
Path C losing by similar relative margins (−10 % / −22 %) under a higher
absolute throughput regime, so the structural conclusion (per-cycle
tree-verify > τ benefit) is consistent across both hardware classes
tested.

### What this means for #38

Per #151's action items:

> 2. If Path C beats chain on either Qwen3.6 variant, file a Path D /
>    pipelining track scoped to that regime
> 3. If Path C also loses on Qwen3.6, close #38 permanently

Path C loses on Qwen3.6-27B. The **A3B variant** (`qwen3.6-35b-a3b.mq4`)
is present in `~/.hipfire/models/`, but **no matching DFlash draft is
available**: the only Qwen3.6 draft on hand is `qwen36-27b-dflash-mq4.hfq`,
trained against the 27B target. A3B is a different topology (35B-param MoE
with ~3B active) so its draft-target distribution alignment is not implied
by the 27B draft, and τ on the cross-pair would conflate draft mismatch
with regime characteristics.

A3B is therefore **untestable on this hardware until a co-trained or
appropriately-distilled DFlash draft is produced**. With the 27B variant
showing Path C losing by −6 / −20 % (code) and −6 / −15 % (prose), and
no untested variant available to flip the result, **the empirical answer
for #151's gating question on this hardware is: Path C does not beat chain
on the testable Qwen3.6 regime → close #38**.

Caveat for the future: if someone produces a Qwen3.6-A3B-matching DFlash
draft, this conclusion should be re-tested. The A3B regime might still
flip the picture (different active-parameter dynamics could shift τ
characteristics enough to favor tree-mode tree exploration).

## Repro

```sh
source scripts/rocm-env.sh
source scripts/gpu-lock.sh && gpu_acquire "your-branch-tag"

# Chain mode (baseline):
HIPFIRE_DPM_WARMUP_SECS=3 ./target/release/examples/dflash_spec_demo \
    --target ~/.hipfire/models/qwen3.6-27b.mq4 \
    --draft  ~/.hipfire/models/qwen36-27b-dflash-mq4.hfq \
    --prompt "$(cat benchmarks/prompts/lru_cache_pep8_strict.txt)" \
    --max 120 --no-chatml --kv-mode asym3

# Path C phase 1 (add):
    --ddtree-budget 12 --ddtree-topk 2 --ddtree-path-c phase1

# Path C phase 2 (substitute):
    --ddtree-budget 12 --ddtree-topk 2 --ddtree-path-c phase2

# For prose, substitute:
    --prompt "$(cat benchmarks/prompts/merge_sort_thinking_off.txt)" --max 256

gpu_release
```

Tok/s + τ from the `=== BENCH METRICS ===` block at end-of-run.
Full-path-accept counted from the histogram entry at the mode-max bin
(15 for chain B=16, 12 for Path C budget=12).

## Bench-host quirks

- Process exits with SIGSEGV during ROCm teardown after metrics print on
  this gfx1151 host. Bench numbers are uncorrupted (printed before
  teardown), but exit codes from the demo can be 139 even on successful
  runs. Cleanup-path bug, not investigated here.
