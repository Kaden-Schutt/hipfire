---
title: Sampled MTP fix = port shipped DFlash machinery (INDEPENDENT per-side nuclei); MTP vocab already tied; DFlash KV-bloat FIXED
date: 2026-06-23
tags: [spec-decode, mtp, sampled, dflash, kv-bloat, design, qwen35]
---

Design + adversarial review of the sampled + tightened MTP path (verdict: feasible-with-corrections).
Full design: docs/plans/mtp-sampled-tighten-design-2026-06-23.md.

**Sampled MTP bug (confirmed):** draft sampled from a nucleus (top_k=20 + top_p, mtp_spec.rs:2511-2520)
but the accept-ratio probs are gathered over the FULL UNTRUNCATED vocab (mtp_spec.rs:2879-2913); bonus
sampled from the full trunk, not the residual. The sampled code is compiled but gated off
(temp<=1e-6, daemon.rs:7125 + "no sampler wired" daemon.rs:12231).

**Fix = port shipped DFlash machinery** (mostly, not new algorithm): promote `sample_residual`
(speculative.rs:2001) + `softmax_temp_into` (1880) to pub(crate) [same crate, just visibility]; swap
the per-step full-vocab gather to the GPU truncated softmax `softmax_temp_topp_batched_into_f32` (this
IS also the #1 tightening T1 -- the distribution fix and the perf fix are the SAME kernel swap, ~89%
host-softmax cut); rewrite accept (u*p_d < p_t) + residual to match DFlash; dispatch-gate mirrors
DFlash (`temp<=1e-6 || (mtp_fast_sample_on && !topk_or_minp)`), thread temp/top_p, cactus_delta=0
(lossless), top_k/min_p -> AR.

**CRITICAL review correction (do NOT skip):** DFlash uses INDEPENDENT per-side nuclei -- the draft
truncates to its OWN nucleus, the target truncates to ITS OWN top_p (speculative.rs:3733-3747), and the
residual subtracts the draft-nucleus dist from the target-nucleus dist across MISMATCHED supports
(speculative.rs:3828). The design's first-pass "same-nucleus (apply draft-tau to target) + new kernel"
is a THIRD, unproven scheme -- WRONG. Build the independent-nuclei version (NO new kernel). It is
simpler, cheaper, and the coherence-validated convention.

**Vocab-tie / updated-head: DO NOTHING.** The MTP head has NO own lm_head/embed -- it reuses the trunk's
by construction (mtp_head.rs:104-106; vocab asserted == trunk at mtp_spec.rs:569-578). Vocab is ALREADY
tied; draft == target post-projection. There is no mismatch to fix and no retrain needed -- any sampled-
tau shortfall is the distribution bug above, NOT head quality. Retrain is a Phase-3 contingency, gated
on sampled tau trailing greedy AFTER the math is byte-identical to DFlash and coherence-clean.

**DFlash KV-bloat: FIXED, none open.** S-tape ~10GB gated off plain-prefill (d7471243); all-64-layer KV
alloc -> *_filtered (33fe5ab4); mq_x_rot FWHT 1.74GB -> chunked ~100MB (1b16aade). Non-bloats (by
design): verify-block stale KV recycled next cycle, rejected-draft KV monotonic ring (DN-state-only
rollback, speculative.rs:5659). Caveat: when MTP KV goes bundle-resident (tightening T6) apply the
*_filtered discipline + a fresh VRAM audit so the 48 LinearAttention layers do not re-bloat.

**Tightening (T1-T7):** T1 = the fix (free). T3 (GPU greedy-accept default-on) needs a byte-identical
greedy A/B (default-behavior change). T4 (verify hipGraph capture) DEMOTED to contingency -- ROCm 7.2
kernarg-snapshot fragile + hipGraph repeatedly net-loss on this stack. T5 (proposal-graph auto) cheap.
T6 (bundle-resident state) = #462 cross-request-bleed risk -> standalone commit + serve-multiturn-gate
(AR + DFlash arms). Effort: Phase 1 sampled fix ~3-5 days (review: design's 2-3 was optimistic; sampled
code has ZERO runtime coverage today). MANDATORY: coherence-gate-dflash three-tier; measure on the
DAEMON not mtp_only_demo; byte-identical prompts + md5. Related: [[mtp-serve-status]],
[[measure-spec-decode-on-the-daemon]], [[spec-decode-verify-kernel-ceiling]].
