---
name: ds4-e8-packing
description: Read-only adversarial design of MFP4E8 decode and gfx942 MFMA operand staging.
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

Read `.omp/DS4-INVARIANTS.md`. Focus solely on the exact MFP4G32E8SOA decode path:
mantissa/sign extraction, shared exponent/scales, FP8/FNUZ conversion semantics,
packing, lane mapping, and staging into gfx942 MFMA. Compare native-instruction and
software expansion recipes without changing results. Rank by emitted instructions,
registers, LDS, and expected weighted benefit. Read-only.
