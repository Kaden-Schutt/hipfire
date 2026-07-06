---
title: MTP spec-decode IS daemon-wired + serve-validated (greedy/gfx11/27B-3.6); sampled MTP is the gap
date: 2026-06-23
tags: [spec-decode, mtp, daemon, serve, qwen35, status]
---

Resolves the contradictory logs ("daemon-wired 1.26-1.73×" vs "mtp_only_demo-only, daemon-wire pending"):
**greedy MTP via serve is GOOD + validated**; the demo-only state is superseded.

**Daemon-wired: YES (greedy only).** Qwen3.5/3.6 (arch 5/6) MTP routes via `generate_qwen35_mtp`
(daemon.rs:5193), gated `HIPFIRE_QWEN_MTP=1 && qwen35_mtp_head.is_some() && temp<=1e-6 && (arch 5||6)
&& !budgeted_thinking_needs_ar` (daemon.rs:7122). Shipped fd717e5d. **temp>0 falls through to DFlash/AR.**
ds4 (arch 9) MTP is wired but a runtime-erroring STUB (spec_decode.rs:25-27) — ignore.

**Serve-validated: YES (narrow).** Durability measured on the DAEMON path (not mtp_only_demo), per
docs/spec-decode-durability-2026-06-23.md: lossless (byte-identical to AR @temp=0), no state-bleed
(4-req session), all genres >=1.15x. Caveat: gfx11 / 27B-3.6 / greedy only. Matrix (gfx1100, q8 KV,
AR~43): code 1.93x/t3.38, reason 1.73x/t3.13, instruct 1.57-1.64x/t2.89, prose 1.31-1.63x/t2.43,
fiction 1.26-1.48x/t2.37-2.79. MTP's distinct value = prose/fiction durability WITHOUT retrain.

**Gap to fully-good-via-serve:** (1) **sampled (temp>0) MTP NOT served** — hard temp<=1e-6 gate; the
sampled path exists (mtp_spec.rs:2839-2936) but needs a truncated-accept-ratio + residual-bonus fix.
THE big gap at temp>0 defaults — exact parallel to the sampled-DFlash we shipped this session.
(2) No MTP arm in serve-multiturn-gate.sh (state-bleed #462 class uncaught). (3) Single-arch (p_min=0
off + unvalidated on gfx1151/gfx12). (4) 27B-3.5 instruct 1.25x (3.6 passes). (5) pp>1 blocked.

**Re-pin needed:** per-genre multiplier spreads ~+-0.2x between two doc tables -> one canonical fresh
run (27B-3.6, k9lin gfx1100, HIPFIRE_QWEN_MTP=1, greedy, q8, max256, K=3/p_min=0.4, byte-identical
prompts + md5, coherence-gate + dflash-gate). Tight-stddev is SUSPICIOUS; print decoded text + run
the three-tier DFlash gate even though MTP is greedy/lossless. Related: [[measure-spec-decode-on-the-daemon]].
