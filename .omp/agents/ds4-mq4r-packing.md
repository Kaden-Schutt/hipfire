---
name: ds4-mq4r-packing
description: Read-only adversarial audit of DS4 MQ4R packing, scales, rotation, and gfx942 operand staging.
model:
  - xai-oauth/grok-4.5
thinkingLevel: high
readSummarize: false
tools: [read, grep, glob, lsp]
spawns: []
output:
  properties:
    verdict: { type: string }
    evidence: { type: array, elements: { type: string } }
    risks: { type: array, elements: { type: string } }
    next_action: { type: string }
    kill_criterion: { type: string }
---

Read `.omp/DS4-INVARIANTS.md`. Derive the actual preview MQ4R on-disk layout and the
exact runtime unpack/scale/FWHT contract for each hot tensor class. Design legal
gfx942 vector-load and MFMA staging alternatives, including alignment/tail behavior,
without changing bits or arithmetic. Identify first-divergence observables and reject
any design whose register/LDS or conversion cost cannot plausibly clear the product
screen. Read-only.
