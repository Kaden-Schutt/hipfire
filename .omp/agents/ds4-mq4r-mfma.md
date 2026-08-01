---
name: ds4-mq4r-mfma
description: Read-only design and ranking of gfx942 MFMA routes for the actual DS4 preview MQ4R tensor families.
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

Read `.omp/DS4-INVARIANTS.md`. Inspect the artifact-derived MQ4R tensor mix and rank
gfx942 wave64 MFMA designs for its hot expert and dense projection shapes. Do not
assume MQ2 or infer a format from the suffix. Model unpack/dequant bytes, MFMA operand
layout, reuse, registers/AGPR/LDS, occupancy, launches, and end-to-end Amdahl impact
for both batch-1 decode and prefill. Preserve exact arithmetic and packing. Give the
cheapest channel test and a quantitative abandonment threshold. Read-only.
