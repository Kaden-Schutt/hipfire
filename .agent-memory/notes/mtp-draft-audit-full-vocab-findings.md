---
title: Served qwen3.6-27b MTP is FULL-VOCAB TIED (not compressed); ds4-class position bug ABSENT; clunk = compressed-path(A3B) + sampling + function tangle
date: 2026-06-24
tags: [spec-decode, mtp, qwen35, ds4, audit, full-vocab, compression]
---

Phase 0+1 read-only audit of the qwen35 MTP draft vs ds4's well-tuned reference (full:
docs/plans/mtp-draft-audit-vs-ds4-2026-06-24.md).

**Phase 0 (CONFIRMED on-disk, k9lin ~/.hipfire/models):** the served qwen3.6-27b MTP sidecars
(qwen3.6-27b.mtp 216M + qwen3.6-27b.mq4-mtp 15G) have `has_compressed_lm_head_draft=false`,
compressed_vocab_size=0 -> **FULL-VOCAB TIED** (use_full_vocab=true). NOT compressed. The ONLY
compressed sidecar on disk is qwen3.6-35b-a3b.moe-mtp-mq4-cvs16384.mtp (A3B MoE, cvs16384). So the
dense 27B-3.6 served path is full-vocab; compression is an A3B-only concern. (tie_word_embeddings=false
everywhere = trunk embed/lm_head not aliased; the MTP head still reuses the trunk lm_head, no separate
MTP projection — the "compressed-serial" function name does NOT mean compression is active.)

**Phase 1 headline (verified):** the ds4-class position/RoPE/SWA off-by-one is **ABSENT** in qwen35 —
every draft path passes `cur_pos+k` (mtp_spec.rs:2405/2418/2448/2461/900; RoPE mtp_head.rs:1543; SWA
1574/1592, seq_len_hint=pos+1 @1425). So the clunk is NOT a draft-position bug. The served 27B-3.6
greedy MTP (full-vocab, position-correct, hidden-state chaining structurally identical to ds4) is CLEAN
on the τ axis — the durability numbers (code 1.93x etc.) come from this clean path.

**The actual clunk (localized):**
- [W1 ★ latent] sampled accept-ratio support mismatch: p_draft over draft support vs p_target UNTRUNCATED
  full-vocab, min(1,p_t/p_d) across incommensurable supports — WORSE than DFlash (no target truncation
  at all). Unreachable today (daemon temp>0 -> AR). The bug F5 fixes.
- [W2 −14-20% τ] cvs-space p_min distortion — **COMPRESSED-PATH ONLY (A3B), NOT 27B-3.6.**
- [W4] spec_step_mtp_compressed_serial = 5 intertwined bool flags / 939 lines / runtime panic@2263 — the
  maintainability tangle that hides W1/W2.
- [W3] lossy embedding-arm chaining is DEMO-ONLY (spec_step_mtp / _compressed); the production serial
  path re-embeds discrete tokens (clean). [W6/W7] dead host-sample code (~1MB/call) + 3 entry points.
- qwen35 MTP head (single GQA+SwiGLU block) is shallower than ds4 (full MLA+MoE+HC) — architectural
  CHOICE, not a bug; no head retrain indicated.

**Phase 2 (for sampled-MTP on 27B-3.6) — reframed:** the served base is ALREADY a clean full-vocab path,
so sampling operates on aligned supports (draft+target both full-vocab) — the compressed-misalignment
worry is MOOT for 27B-3.6. Plan: **F3** decompose the tangled function (extract a clean sampled fn; kill
the panic) -> **F4** cleanup (dead code, entry points) -> **F5** the sampling fix (DFlash convention:
independent per-side nuclei + sample_residual) on the full-vocab path. **F2** compression-demote is
A3B-only / lower priority. Validate: greedy re-pin (full-vocab baseline τ) + coherence-gate-dflash
three-tier at temp>0 BEFORE lifting the daemon temp<=1e-6 gate; byte-identical prompts + md5; daemon-
measured. Related: [[mtp-serve-status]], [[mtp-sampled-fix-and-tightening]], [[measure-spec-decode-on-the-daemon]].
