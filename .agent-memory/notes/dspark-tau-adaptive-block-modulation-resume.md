---
title: DSpark τ-adaptive controller — block-modulation BEATS binary spec↔AR, CONFIRMED post-batched-kernel (code peak block=4 +44.6% vs block=5 +20.2%); design controller next
date: 2026-07-02
tags: [dspark,spec-decode,deepseek4,tau-adaptive,block-size,batched-sampler,perf]
---

## POST-BATCHED VERDICT (HEAD 2f744c11, clean 3-pass, temp0.7) — block-modulation WINS
Batched fused sample+accept kernel (commit 2f744c11) LANDED + coherence-confirmed +
block sweep RE-RUN. τ/windows byte-identical pre↔post (kernel is sampling-equivalent);
only per-window time changed. **Batched kernel = modest ~+2% on code, ~0% on prose** —
NOT the ranking-flip the max=64 smoke misleadingly suggested (short-run/high-τ artifact).
head+sampler is a small slice of the trunk-dominated window, so eliminating per-position
D2H can't recover more. **It did NOT restore binary spec↔AR.**

CODE (AR 11.95): b1 +22.8 / b2 +34.6 / b3 +31.3 / **b4 +44.6 (peak)** / b5 +20.2%.
PROSE (AR 12.23): **b1 −1.9 (peak,≈AR)** / b2 −10.5 / b3 −11.4 / b4 −23.4 / b5 −20.4%.
⇒ **Continuous τ-driven BLOCK MODULATION is the switch axis** (NOT binary spec↔AR):
best block beats the block=5 full-spec endpoint by +20% (code) / +23% (prose), and
block=1≈AR−1.9% means NO separate AR path is needed (drafter always runs, block∈[1,5]).
Code wants block 3-4 (high τ), prose wants block 1 (low τ). CAVEAT: exact peak block=4
is fixed-seed/prompt-specific (b4 drew highest τ2.581); robust claim = "intermediate
block 2-4 >> block5 on code; block1 best on prose". NEXT: design the τ→block controller
(resume brainstorming BS3→writing-plans). Data: summary_block_sweep.txt (post) +
summary_block_sweep.prebatched.txt (pre).

## DESIGN (approved 2026-07-02) — full spec in gitignored docs/superpowers/specs/2026-07-02-dspark-tau-adaptive-block-controller-design.md
Control law = **marginal-accept hill-climb on the draft block**, in the generic
`DsparkDrafter` core (dspark_core.rs, adjusts `self.block` feeding
`cfg.block_size.min(k)` @:1056 → covers deepseek4 + qwen3 DSpark, no generate_spec
change). Per window: EMA the **full-accept rate** `p` = P(accept_len == n_proposed)
(drafted count AFTER conf-truncation, NOT the cap); grow block +1 above threshold,
shrink −1 below, clamp [1,cfg.block_size]; reset block+EMA per request (no #462 bleed);
composes with conf_threshold (hill-climb = cap, conf trims within).
**Threshold is LIVE-MEASURED, not hard-coded** (portability, Rule 4): break-even
`p* = Δt_position / t_AR_forward` is model×quant×arch-specific, but being a *ratio of
same-thermal-state timings* it's thermal-invariant → measure from the profiler (fit
Δt slope of verify_time vs n_proposed as the hill-climb varies the block; t_AR from
n=1/bootstrap). Bootstrap prior p*=0.18 (gfx1151-mq2lloyd) + clamp [0.05,0.5]; fallback
= once-per-model startup calibration. Both signal (count) and threshold (ratio) are
thermal-robust. Staging: **A) greedy first** (already served, no daemon-gate change) →
**B) temp>0** (relax daemon.rs:6727 + flip carriers.rs:977 + thread temp through
generate_deepseek4_spec). Default-on. NEXT: writing-plans → implement.

Branch `feature/dspark-qwen3`. Investigation for the τ-adaptive DSpark spec/AR
controller (handover `docs/superpowers/plans/2026-07-02-dspark-tau-adaptive-fallback.md`).
PAUSED mid-block-sweep by user. Resume in the ORDER below. See sibling note
[[dspark-ds4-greedy-lazy-verify-falsified]] for the premise validation + the
lazy/non-lazy/MTP matrix that preceded this.

## RESUME PLAN (do in this order)
1. **Land the batched sample+accept kernel FIRST** (in flight — see "Parallel WIP"
   below). It replaces the per-position host D2H loop in the sampled verify, which
   is the real temp>0 overhead → it SHIFTS ms/window → the block-size economics
   below WILL CHANGE. Do NOT design the controller on the pre-batched numbers.
2. **Re-run the block sweep** on the new binary (`run_block_sweep.sh`, 3 passes,
   finish prose b4/b5 which are missing). Re-add the diagnostic block override
   first (exact edit below). Compare the post-batched curve to the pre-batched one
   recorded here.
3. **Decide the switch axis** (binary spec↔AR vs continuous block-modulation) on
   the post-batched sweep.
4. **Design the controller** (resume brainstorming → writing-plans). Granularity
   (per-request-probe vs continuous+hysteresis) is the next open question; fixed
   seed ⇒ per-prompt-deterministic τ favors a simple per-request probe.

## The decision in flight: block-modulation is BEATING binary spec↔AR
The controller's switch axis was narrowed to two live options. Partial block-size
sweep (temp0.7 DSpark-lazy, `HIPFIRE_DEEPSEEK4_DSPARK_BLOCK` override, PRE-batched-
kernel) says an intermediate block beats the binary endpoint:

**CODE (AR 11.94)** — reliable (2 passes): b1 +21% / **b2 +31% (peak)** / b3 +28% /
b4 +23% / b5 +18%. Peak block=2 beats block=5 full-spec by **+11%**.
**PROSE (AR 12.23)** — b1–b3 reliable, b4/b5 single-sample (thermal-suspect): **b1
−2.5% (near-AR)** / b2 −9.9% / b3 −3.3% / b4 −17.5%⚠ / b5 −34.3%⚠ (3-pass matrix
says b5=−20%, trust that). Smaller block = closer to AR on low-τ prose.
⇒ A τ-driven block schedule (block≈2 high-τ, block=1 low-τ) plausibly captures the
code peak AND stays ~AR on prose WITHOUT a separate AR path (block=1 ≈ AR−2.5%).
Data: `/home/bjoern/ds4-lazy-validate/summary_block_sweep.txt` (22/36 runs: pass 1
complete, pass 2 missing prose b4/b5, pass 3 not run). Parser: `parse_block_sweep.py`.

## Parallel WIP (another agent, UNCOMMITTED — do not clobber; `git status` for live state)
Batched fused sample+accept being implemented AND wired — this IS resume step 1:
- NEW `kernels/src/dspark_sample_accept_lazy.hip`: fused on-GPU sample+accept —
  samples all `n` verify positions on-device from resident `[n×vocab]` batched-head
  logits (top-K→softmax(temp)→top-p→xorshift32 draw), returns accepted prefix + RNG,
  killing the per-position 8-byte D2H host round-trips.
- `crates/rdna-compute/src/sampling.rs` (+93): `pub fn sample_accept_lazy_f32(...)`
- `crates/rdna-compute/src/kernels.rs` (+10): `DSPARK_SAMPLE_ACCEPT_LAZY_SRC` include
- `crates/hipfire-arch-deepseek4/src/forward.rs` (M): `final_norm_and_sample_all_batched_lazy`
  being rewired to call `.sample_accept_lazy_f32(...)` (replaces the per-position host loop
  at the old forward.rs:7947-7971).
  On resume: confirm this landed/committed + coherence-gated, THEN re-sweep. NOTE the
  block override re-add site (dspark_bench.rs) is unaffected by these edits.

## Established (do NOT re-derive)
- **Premise VALIDATED** (fresh 4-seed→fixed-seed reproduction): temp0.7 DSpark code
  +20% vs AR (τ2.35), prose −14/−20% (τ1.44); break-even τ*≈2.0. tok/s is thermally
  noisy (±15-40% on byte-identical work) ⇒ **controller keys on τ, NOT tok/s**.
- **KEEP lazy; it is NOT a switch axis.** Sampled lazy ~10-12% cheaper/window than
  non-lazy (deconfounded via ms/window; corrects greedy "<1%"). Non-lazy loses to AR
  even on code (−5%). MTP is tight-band (+11/−9) but NEVER optimal (2nd both genres).
- **AR-step primitive EXISTS** — `SpecTarget::spec_advance(&[seed])` (greedy) /
  `verify_block_sampled(&[seed])` (sampled); no new primitive needed. Sampled AR is
  the right fallback (honors temp, same speed as greedy AR).
- **REAL ds4 gate = daemon.rs:6727** `spec_mode = deepseek4_spec_requested(m) &&
  temp <= 1e-6` — NOT just `supports_temp=false` at carriers.rs:977 (handover missed
  this). temp>0 ds4 currently routes to the bespoke `generate_deepseek4` AR sampler
  and NEVER reaches `generate_spec`. Staging step 1 must relax 6727 AND flip 977 AND
  thread temp>0 sampling through generate_deepseek4_spec→generate_spec.
- **Fixed prod RNG seed 0x13579BDF** (dspark_core.rs:1334) ⇒ per-prompt-deterministic
  τ ⇒ stable controller signal, per-request probe reliable for homogeneous requests.
- **Bench↔daemon gap**: bench omits the daemon's `</think>` non-thinking scaffold
  (daemon.rs:9768, default NonThink); re-confirm τ* on the daemon in staging step 1.
- **Controller hook** = shared `generate_spec` (daemon.rs:4273), per-window
  `step.accepted` at 4659-4660; ds4 `generate_deepseek4_spec` (9800) delegates there
  ⇒ arch-generic (also benefits qwen3 DSpark + qwen35 DFlash sampled routes).

## Tree state / diagnostics (BOTH reverted — tree clean at HEAD 839f8346 except parallel WIP)
- Lazy toggle (`HIPFIRE_DEEPSEEK4_DSPARK_LAZY_VERIFY`) — REVERTED. Re-add: gate the
  break in `forward.rs::final_norm_and_sample_all_batched_lazy` (~7975) on a
  `lazy_stop = env!="0"` flag.
- Block override (`HIPFIRE_DEEPSEEK4_DSPARK_BLOCK`) — REVERTED. Re-add in
  `examples/dspark_bench.rs` right after `let block = …cfg.block_size…`:
  `let block = env("HIPFIRE_DEEPSEEK4_DSPARK_BLOCK").parse().map(|b| b.clamp(1,block)).unwrap_or(block);`
  (draft length is `cfg.block_size.min(k)` at dspark_core.rs:1056, so k<5 drafts fewer).
- Harnesses in `/home/bjoern/ds4-lazy-validate/`: `run_lazy_matrix.sh` (drafter/mode
  matrix), `run_block_sweep.sh` (block sweep), `parse_*.py`, `summary_*.txt` logs.
