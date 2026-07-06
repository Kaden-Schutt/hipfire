---
title: LFM2.5-ColBERT-350M hipfire port — feasible-with-corrections (~2-3 weeks); B1 may need ZERO new kernels
date: 2026-06-23
tags: [embedding, colbert, lfm2, retrieval, dots-ocr, design, phase2, parked]
---

Phase-2 ("ripgrep for agent memory" engine) port of LiquidAI/LFM2.5-ColBERT-350M into hipfire.
recon -> design -> adversarial-review workflow. Verdict: **feasible-with-corrections, ~2-3 weeks**.
Full design: docs/plans/lfm2colbert-port-design-2026-06-23.md.

**Shape:** bidirectional LFM2 encoder (10 conv + 6 attn, hidden 1024) + bias-free Dense 1024->128
per-token projection; late-interaction MaxSim host-side. NOT a decoder — one-shot batched prefill,
no KV/sampler. Closer to the dots.ocr vision tower than any text decoder.

**Big de-risk (review-confirmed):** the "one new kernel" (bidirectional attention) can route through
the **EXISTING** `attention_dflash.hip` (already non-causal GQA flash) via `KernelKey::AttnFullF32`;
`rope_batched_f32` (positions-vector RoPE) already exists; the 128-d projection is one reused
`linear_f16`. So **Phase B1 may need zero new HIP kernels** — mostly wiring + parity. New: crate
`hipfire-arch-lfm2colbert` (arch_id 14), `hipfire-retrieval` (CPU MaxSim = the divorce-first/portable
seam), an encode daemon arm.

**Review corrections (the design body has these errors):** RoPE = use existing `rope_batched_f32`
(positions vector), NOT `rope_f32` (scalar pos_buf[0]); conv = NEW work not reuse (no batched conv
kernel — seq-loop the decode kernel or build `conv1d_gated_prefill_f32`); `rope_theta=1e6` not the
lfm2moe 5e6; `intermediate_size` auto-adjusts ~4608 not 6656; drop Q8-KV (F32 flash, no KvCache).

**Top risk:** bidirectional-attn numerical correctness (GQA map + per-row RoPE + QK-norm-before-RoPE
order) — 5%-class error -> cosine ~0.95, no crash. Mitigation: FP32-deterministic per-layer hidden-state
parity bisect vs PyLate, with a CpuColbertBackend as the algorithm-vs-kernel discriminator.

**LICENSE gate:** LFM Open License v1.0 caps commercial use at $10M revenue — product go/no-go for the
standalone-engine vision, not a code blocker. Clear before any production ship.
Related: [[measure-spec-decode-on-the-daemon]], [[spec-decode-verify-kernel-ceiling]].
