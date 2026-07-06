---
title: MTP lm_head precision is NOT the lever — trunk verify forward is 84% of cycle BW, draft lm_head only 3.4%; stay full-vocab
date: 2026-06-24
tags: [spec-decode, mtp, bw, lm_head, cvs, int4, falsification, qwen35]
---

Investigated whether a cheaper draft lm_head (cvs sidecar / int4 native instructions / MQ2 storage)
speeds MTP on 27B. Verdict: NO — the lm_head is not the bottleneck.

**BW breakdown (verified-from-code, K=5, 27B):** trunk verify FORWARD = **84.3%** (10.6 GB/cycle —
the full 27B model over the K+1 verify positions); draft lm_head = **3.4%** (426 MB compressed,
~21% full-vocab); rest = MTP head block + verify lm_head. EVERY lm_head-targeted lever optimizes a
3–5% slice that is never the bottleneck.

**Q1 cvs-for-27b:** YES — arch-agnostic, ZERO code (`mtp_extract --vocab-sidecar`; h_dim=5120 is
256-aligned → MQ4G256; 2 CPU-only steps). Never a3b-restricted. **Q2 need it:** NO — costs τ
(out-of-top-K forced rejects) for a 3% BW slice. Do not generate a 27B cvs sidecar.

**Q3 int4 / lower-precision draft head:** int4 native INSTRUCTIONS don't help — BW-bound on weights
(same wall as int8/dp4a-verify-falsified). The draft is batch=1, so WMMA `iu4` (matrix, needs B≥16)
never fires, and RDNA3 has no int4 scalar dot for a GEMV. The ONLY real lm_head BW lever is an
**MQ2-storage FULL-VOCAB draft head** (~2×, 527→267 MB/pass; lossless-greedy because verify stays
full-precision) — but no `gemm_mq2g256_batched_lmhead` kernel exists (new kernel + dispatch +
quantizer + weight field), τ at 2 bpw is unmeasured (MQ3 is the quality floor), and it shaves
~260 MB off a 3–4% slice. NOT worth it for 27B. (qwen35 MTP already uses
`shared_lm_head_with_trunk` = the EAGLE design → no separate draft head exists; "int4 draft head"
means SPLITTING the shared one for a tiny win.)

**Rank-2 fused-argmax GEMV:** ~zero BW win (weight read unchanged; saves only the logits write
~3.6 MB + the separate-argmax launch); greedy-only. Harmless micro-opt, not a lever.

**The real lever** for MTP throughput is the trunk verify forward (84%), and the way to shrink it
is **τ** — accept more tokens/cycle → fewer verify forwards (the drafter-quality lever, consistent
with [[spec-decode-verify-kernel-ceiling]]). lm_head precision is a dead-end micro-opt. Stay
full-vocab. Related: [[mtp-draft-audit-full-vocab-findings]], [[mtp-serve-status]].
