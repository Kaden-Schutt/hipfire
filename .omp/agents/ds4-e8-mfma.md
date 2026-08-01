---
name: ds4-e8-mfma
description: Read-only competing gfx942 MFMA designs for MFP4E8 dense projections.
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

Read `.omp/DS4-INVARIANTS.md`. Design and rank dense MFP4G32E8SOA gfx942 routes for
batch-1 decode and prefill. Compare scalar/software expansion, FP8 staging, and legal
MFMA operand forms without changing arithmetic or format. Size decode, packing,
memory, registers, LDS, wave count, and expected reuse at real DS4 shapes. State the
first cheap experiment and abandonment threshold. Read-only.
