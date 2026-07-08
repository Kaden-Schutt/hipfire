---
title: DSpark conf_threshold sweep (qwen3=0.1) + CLI-shadow fix + temp>0 sampled verify (fused GPU sampler, still loses to AR)
date: 2026-07-02
tags: [dspark,spec-decode,conf-threshold,temp,sampling,qwen3,wiring]
---

Branch feature/dspark-qwen3. Follows [[dspark-gpu-resident-p4-p5]].

## DSpark runtime params
- **k / draft length = `block_size`** — baked in the sidecar JSON
  (`dspark_block_size`: qwen3=7, ds4=5), clamped [1,8]. It's the drafter's
  TRAINED horizon — NOT a runtime knob (no env/CLI; `MtpSpeculator` uses
  `arch.k()`==block_size). Nothing to sweep.
- **"budget" = `conf_threshold`** — the only real tunable. Confidence cutoff that
  truncates the drafted block before verify. Ladder: env
  `HIPFIRE_{QWEN3,DEEPSEEK4}_DSPARK_CONF_THRESHOLD` > CLI/config > per-arch carrier
  default (qwen3 0.1, ds4 0.5).

## conf sweep (gfx1151, qwen3-8b, max=256, fresh-proc, warmed)
Flat-to-cliff: **0.1 is optimal** on both prompt types; higher over-truncates.
code: 0.05→26.7 0.1→26.7 0.2→26.8 0.3→26.3 0.5→22.6 0.7→18.8.
prose: 0.05→26.5 0.1→26.5 0.2→23.8 0.3→22.8 0.5→17.7 0.7→16.8.
AR baseline 23.8 → DSpark greedy@0.1 is +12% vs AR.

## CLI-shadow bug FIXED (066d084a)
CLI unconditionally forwarded `dspark_conf_threshold=0.5`, and the qwen3 carrier
ranks that above its 0.1 default → qwen3 DSpark via the CLI silently ran at 0.5
(−15% code / −33% prose on the DEFAULT greedy path). Also the `run` flag only set
the ds4 env var (no-op on qwen3). Fix: CLI default `null`, forward only when the
user sets it (per-arch carrier default applies); `run` flag sets both arch env
vars; docs corrected ("deepseek4-only" was wrong — drives both).

## temp>0 sampled verify (bc4df7c2 + adb90438)
DSpark was greedy-only (`requires_greedy`); temp>0 bypassed it to AR. Added
distribution-preserving sampled verify: drafter stays a point-mass guess, TARGET
samples t_i~p_T(temp,top_p,top_k), accept draft iff ==sample. Wiring:
`SpecTarget::verify_block_sampled_capture_gpu` (default Err; llama impl, ds4
pending), `MtpDrafter::set_sampling`/`supports_temp_verify`, DsparkDrafter branches
greedy/sampled on temp, `build_dspark_speculator(supports_temp)`. bench gains
`HIPFIRE_QWEN3_TEMP/TOP_P/TOP_K`.
**Fused the softmax into the existing `sample_top_p_pf` GPU kernel** (softmax +
nucleus + top_k + categorical in ONE launch, 4-byte D2H) — same sampler AR uses,
so committed tokens are distribution-IDENTICAL to AR (not an approximation).
First cut host-softmaxed (want_logits D2H + host exp) = 17.2; fused GPU = 20.3.

Progression (qwen3 code, top_p0.95/topk40, warm): host-softmax 17.2 → fused-GPU
20.3 → **LAZY 29.6** (temp0.7) / 28.8 (temp1.0). AR 23.8, greedy 26.7.

**LAZY PREFIX SAMPLING is the win (0f007882).** Acceptance is a prefix: once a
verify position's sample != its drafted token, all later positions reject — so
STOP the per-row head+sample loop at the first mismatch (pad rejected picks;
accept_greedy_prefix only reads up to the mismatch; the batched forward already
captured all b hidden, so P5 unaffected). The expensive 152k-vocab lm_head then
runs ~τ times/window instead of b. Per-window committed output identical to eager
(only the RNG stream diverges). **qwen3 temp>0 now BEATS AR (+24%) AND greedy** —
it does far fewer lm_head GEMVs than either. temp=0 unchanged (greedy path
untouched). Applies ONLY to the sample branch (argmax has multiple consumers that
read all picks — do NOT blanket-lazy the argmax path).

**Earlier "temp>0 fundamentally loses" was WRONG** — it assumed you must sample
all b positions. You don't. Lazy fixes it.

## deepseek4 (Stage 2, 3fe37f27) — verify_block_sampled_capture_gpu +
`final_norm_and_sample_all_batched_lazy` (per-position fused sampler + lazy stop).
CLEAN same-session CODE (max160, warm, 2-run stable): AR 11.32 / greedy 8.94
(τ1.35) / temp0.7 **11.40** (τ1.84, accept0.18 — HIGHER than greedy!) / temp1.0
9.58 (τ1.46). So on code temp0.7 ~TIES AR and beats greedy; temp1.0 loses to AR,
beats greedy. (temp0.7 fastest because the markov draft aligns with the SAMPLED
target better than the argmax target here.) PROSE (natural-EOS, noisy) AR 12.5 /
greedy ~14 / temp>0 8.9-11.7 — loses to both (ds4's prose win is greedy's high
accept 0.32, eroded by sampling). Net: ds4 temp>0 is coherent but ~break-even to
losing (no clear win like qwen3) → serving stays gated (supports_temp=false);
bench-enabled. Note greedy code here is 8.94 = post greedy-lazy (was 6.12).

## serving-enable DONE (af6e7ff5) — qwen3 temp>0 ON
Carrier supports_temp=true (qwen3); daemon `llama_dflash_route` (arch 0/1) now
also engages on `chain_sample_route`, so temp>0 routes to the spec loop. KEY
semantic: the daemon's `supports_temp_verify()` flag means "ddtree-SWOR,
TEMPERATURE-ONLY" — DSpark is chain-like (honors temp+top_p+top_k via
sample_top_p_pf), so DsparkDrafter keeps supports_temp_verify()=**false** and
signals sampling via requires_greedy()==false (→ `spec_can_sample` →
chain_sample_route). Validated: daemon temp=0.7 → "dflash":true (routed, not AR),
coherent, conf 0.10. ds4 stays gated (supports_temp=false: prose-loses/code-murky).

## GREEDY lazy prefix-stop DONE (0709898e) — both arches, byte-identical
Applied the lazy prefix-stop to the greedy argmax verify too, gated behind a
`lazy` bool on the shared llama body so ONLY the DSpark capture paths take it
(verify_block_argmax/logits stay eager — n-gram/DFlash-chain/tree consumers read
all picks). deepseek4 got final_norm_and_argmax_all_batched_lazy +
dspark_verify_argmax_lazy. Measured: qwen3 greedy 26.72→31.04 (+16%, A/B decoded
output byte-IDENTICAL); ds4 greedy code 6.12→8.92 (+46%, parity PASS; still <AR
10.19 — head-bound at τ1.35). Committed output byte-identical (accept reads only
to the mismatch; rejected picks padded u32::MAX).

TRAP: greedy-lazy lives in llama_spec's shared body (qwen3); deepseek4 has a
SEPARATE head (final_norm_and_argmax_all_batched) — must lazy each arch's head
independently. ds4 greedy has pre-existing run variance (mq2lloyd/Q8) so use
parity + by-construction identity, not a raw-stats A/B.

## ds4 conf budget with lazy verify (a85a072d) — default 0.5→0.3
With the cheaper (lazy) verify, does a higher drafter budget (lower conf → less
truncation) pay off? The drafter always drafts all block_size(=5); conf truncates
how many reach verify. Lazy skips per-position HEADS after the first mismatch, but
the verify TRUNK FORWARD still scales with #proposed — so extra drafts are ~free
only if they get ACCEPTED. Result is prompt- AND temp-dependent:
- GREEDY (fixed-len, warm): 0.3 is the sweet spot. code 0.5→8.95 / 0.3→9.47 /
  0.1→9.45; fiction-prose 0.5→10.09 / 0.3→**12.13** / 0.1→11.27. 0.5 was cutting
  correct drafts; 0.1 over-proposes wrong ones (forward cost, no accept). → default 0.3.
- temp>0: OPPOSITE — higher budget LOSES. temp0.7 code 0.5→11.41 / 0.1→10.31
  (−10%): sampled target less predictable → extra drafts wrong. temp>0 wants MORE
  truncation. (Moot for serving — ds4 temp>0 gated off; greedy optimum governs.)
- TRAP: the FIRST short-factual-prose test (natural EOS) falsely showed 0.1
  regressing (14.25→10.70) — an EOS/non-determinism artifact (token count 81 vs
  71). RAW fixed-length is the clean way; it flipped the verdict (0.3 wins prose).
  High-acceptance prompts genuinely prefer higher conf, low-accept prefer ~0.3.
- Greedy conf is a PURE SPEED knob (output = target argmax regardless of #proposed).

## traps
- AR decode speed is temp-invariant (per-token sampler cost ≈ forward-bound), so
  AR greedy tok/s is a fair temp>0 baseline.
- ds4 dspark_bench needs unique EOS-free comparison; natural-EOS prose gens vary
  token count → noisy tok/s. Use raw/fixed-max for clean A/B.
- grep trap: `grep "temp="` matches the bench's header line → `head -1` hides the
  result; filter on `tokens=.*tok/s`.
- ds4 conf sweep optimum still unknown (never cleanly measured).
