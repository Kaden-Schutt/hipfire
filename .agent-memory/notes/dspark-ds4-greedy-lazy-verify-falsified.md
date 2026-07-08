---
title: ds4 DSpark lazy-verify — greedy +46% FALSIFIED (head <1% window); temp>0 BEATS AR +20% on code / -14% prose (deconfounded 4-seed) → supports_temp=false is a blunt gate, not pure artifact
date: 2026-07-02
tags: [dspark,spec-decode,deepseek4,lazy-verify,falsification,perf,tau]
---

Branch feature/dspark-qwen3. Validating commit 0709898e's claim that the greedy
lazy prefix-stop verify gives deepseek4 DSpark **+46%** (6.12→8.92 tok/s, code).
Method: env toggle `HIPFIRE_DEEPSEEK4_DSPARK_LAZY_VERIFY=0/1` gating BOTH capture
paths (greedy: lazy per-position loop vs eager BATCHED head `dspark_verify_argmax`;
sampled: break vs no-break), one binary, byte-identical prompt (md5 recorded),
warm (throwaway run + warmup=24), 3 fresh-proc runs/cell, median. Model
`deepseek-v4-flash.mq2lloyd` + `-dspark` sidecar on gfx1151. Toggle REVERTED after.

## Verdict: the +46% does NOT reproduce. Greedy lazy is a no-op-to-slight-LOSS.
| cell | τ | lazy=1 (lazy) | lazy=0 (batched) | Δ |
|---|---|---|---|---|
| greedy code (md5 5f90af57, conf0.3) | 3.636 | 22.03 | **22.76** | **−3.2%** |
| greedy prose (low-τ) | 1.905 | 12.65 | 12.62 | +0.2% |
| temp0.7 code | 4.324/3.556 | **24.12** | 18.94 | +27% raw / ~+5% deconfounded |

- Greedy output **byte-identical** lazy vs eager (matching token-id md5, identical
  windows/τ/accept) — correctness fine, it's purely a perf question.
- **Root cause the +46% is impossible from this path:** per-window arithmetic —
  low-τ prose skips ~2.4 head GEMVs/window yet saves only ~0.5 ms of a ~150 ms
  window ⇒ **the lm_head is <1% of the ds4 DSpark verify window; the MoE trunk
  forward dominates** (even more extreme than [[mtp-lmhead-not-the-lever]]'s 3.4%
  on 27B qwen35). Lazy only skips lm_heads, so it CANNOT move tok/s meaningfully.
- **Why lazy actively LOSES at τ3.6:** lazy REQUIRES a per-position loop (to
  early-exit), which can't batch the head; the pre-lazy baseline (af6e7ff5,
  parent of the lazy commit) already called the BATCHED head (`dspark_verify_argmax`
  → `final_norm_and_argmax_all_batched`, default since d63bd790/1c480ff5). Batched
  reads the 152k-vocab weight ONCE for all b + one D2H; the lazy loop pays
  per-position launch/D2H for ~0 skip benefit ⇒ −3%.
- **The commit's 6.12→8.92 was a DPM/throttle/warmup artifact**, not a compute win
  — exactly the documented ds4 greedy run-variance ("use parity + by-construction
  identity, not raw-stats A/B"; [[dspark-v4-deepseek4-port]] "warmup dropped
  17.6→14.9"). My warm numbers are 2× the commit's absolutes (22 vs ~9) and rock
  stable within-cell (22.03/22.03/22.06), i.e. better-warmed.

## temp>0 (sampled) — the PRODUCTION axis (greedy is bench-only, nobody serves it)
Right comparison is DSpark-temp0.7 vs **AR** (what supports_temp switches to), NOT vs
eager-sampled-spec. Measured properly: warmup=**128** (not the cold 24 my first pass
used — 24 reads ~12% low), 4 RNG seeds (`HIPFIRE_DSPARK_RNG_SEED`, since the default
0x13579BDF is hardcoded), fresh-proc. AR is greedy trunk decode (temp-invariant speed).

| prompt | AR | DSpark-temp0.7 mean (4 seeds) | range | Δ vs AR |
|---|---|---|---|---|
| **code** | 11.94 | **14.31** | 11.96–15.92 (τ1.98–2.81) | **+20%** (seed-variable: +0.2%…+33%) |
| **prose** | 12.22 | 10.49 | 10.09–10.87 (τ1.43–1.60) | **−14%** (consistent all 4 seeds) |

- **temp>0 DSpark genuinely BEATS AR on code (+20% mean, deconfounded).** My first-pass
  single-seed "24.12" was a LUCKY outlier — the hardcoded default seed draws τ4.324;
  typical code τ is ~2.3 (mean 14.31). Still a real +20% over AR. Both regimes coherent
  (no attractor; prose seed=42 = clean Borges pastiche).
- **temp>0 DSpark LOSES to AR on prose (−14%, consistent)** — drafter can't predict
  free-form text (τ~1.5, accept ~0.13). This is exactly the "prose-loses" reason for
  supports_temp=false ([[dspark-conf-and-temp-verify]]).
- Greedy DSpark (τ3.636) actually beats AR HARDER on code (~22 vs 11.94, ~+85%) because
  an argmax target is more predictable than a sampled one — but greedy isn't served.

## supports_temp=false is a BLUNT gate, not a pure artifact
It correctly protects prose (−14%) but bluntly forfeits the code win (+20%) by turning
temp>0 spec OFF for ALL workloads. Options: (a) flip supports_temp=true for a
code-focused deployment (accept the prose regression); (b) BETTER — τ-adaptive fallback:
serve temp>0 spec, auto-drop to AR when acceptance stays low (prose self-detects at
τ~1.5). (b) captures the code win without the prose loss. NOT a simple flag flip.

## Actionable
- **Greedy lazy-verify: revert for ds4** (route greedy `verify_block_capture_gpu` back
  through batched `dspark_verify_argmax`) — small regression, zero upside (head <1% of
  window). KEEP the sampled lazy path (helps temp compete with AR).
- **supports_temp for ds4: reconsider** — not an artifact but too blunt; τ-adaptive is
  the real fix. Conf-0.3 default (a85a072d, post-lazy) also interacts with τ.
  **Plan handover: `docs/superpowers/plans/2026-07-02-dspark-tau-adaptive-fallback.md`**
  (break-even τ*≈2.0; adapt spec↔AR in `generate_spec`; genre is an unreliable proxy —
  the `carriers.rs:973-976` comment has code/prose backwards vs measured).
- **qwen3 DSpark (+16% greedy claim) UNVALIDATED** — smaller dense trunk ⇒ bigger head
  fraction, lazy may genuinely help there; needs its own warm A/B.
- Logs: `/home/bjoern/ds4-lazy-validate/` (summary*.txt). Diagnostic seed/lazy env
  toggles were REVERTED after measuring (no knobs left in tree).

## temp0.7 drafter/mode matrix — AR / DSpark-lazy / DSpark-nonlazy / MTP × code/prose (2026-07-02, deconfounded 3-pass, fixed seed)
Re-added `HIPFIRE_DEEPSEEK4_DSPARK_LAZY_VERIFY` toggle, ran 3 passes (τ/windows
deterministic under the fixed seed; median tok/s; throwaway warmup → AR back to
11.95, not the cold 11.06). **ms/window (time/windows) is the deconfounded metric** —
lazy vs non-lazy diverge in RNG trajectory (different τ), so end-to-end tok/s can't
rank them; per-window cost can. Toggle REVERTED after (tree clean).

| genre | mode | tok/s | Δ vs AR | τ | ms/window |
|---|---|---|---|---|---|
| code | DSpark-lazy | 14.10 | **+18%** | 2.353 | 166.8 |
| code | MTP (greedy) | 13.22 | +11% | 2.192 | 165.8 |
| code | AR | 11.95 | — | 1.0 | 83.7 |
| code | DSpark-nonlazy | 11.33 | **−5%** | 2.105 | 185.8 |
| prose | AR | 12.22 | — | 1.0 | 81.8 |
| prose | MTP (greedy) | 11.10 | −9% | 1.839 | 165.7 |
| prose | DSpark-lazy | 9.78 | −20% | 1.441 | 147.4 |
| prose | DSpark-nonlazy | 9.34 | −24% | 1.569 | 167.9 |

- **KEEP lazy; it is NOT a τ-switch axis.** Deconfounded ms/window: lazy ~10–12%
  cheaper/window on BOTH genres (166.8<185.8 code, 147.4<167.9 prose) — real, because
  sampled per-position head+`sample_top_p_pf` (152k top_p/top_k sort) is expensive and
  lazy skips the rejected tail. **CORRECTS the greedy "head <1% of window" → that holds
  for GREEDY argmax only; sampled lazy is a genuine ~10–12%/window win.** Non-lazy loses
  to AR even on code (−5%) → strictly dominated. The lazy/non-lazy tok/s ranking flips
  by trajectory luck (smoke had non-lazy winning) — not a per-window property.
- **MTP tight-band (+11%/−9%) but NEVER optimal** — 2nd in both genres (DSpark wins
  code, AR wins prose). A 3-way spec↔MTP↔AR controller adds nothing over 2-way. (Note:
  simpler MTP head is more robust on unpredictable prose, τ1.839 > DSpark 1.441.)
- **Controller = binary DSpark-lazy ↔ sampled-AR at τ≈2.0.** ms/window confirms a
  DSpark-lazy window ≈ 1.8–2.0× an AR forward (167/84 code, 147/82 prose) → break-even
  τ≈1.8–2.0.
- **Block-size modulation is a DISTINCT unmeasured lever, NOT subsumed by lazy.** Lazy
  early-break shrinks the post-trunk HEAD loop only; the MoE **trunk** still runs over
  the full drafted block first (that's the ~2× cost). Shrinking the block shrinks the
  trunk; block=1 ≈ AR → block-size is the *continuous* form of the binary switch.
  Whether an intermediate block (2–3) beats binary spec↔AR at temp0.7 is UNMEASURED.
