---
name: ds4-quant-rotation-contract
description: Read-only artifact-derived contract for the preview MQ4R tensor formats, FWHT rotation, scales, and packing.
model:
  - kimi-code/k3-256k
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

Read `.omp/DS4-INVARIANTS.md`. From the pinned preview artifact, recover the complete
dtype map and exact scalar mathematics/byte layout for every hot MQ4R tensor class,
including any MFP4E8/E8 classes actually present, scales, signs, FWHT, and
`mq_rotate_x`. Trace quantizer-to-loader-to-kernel assumptions. State bit-exact versus
tolerance-based boundaries and propose first-divergence checkpoints. Do not assume
MQ2, infer from the suffix, or propose a format/weight change. Read-only.
