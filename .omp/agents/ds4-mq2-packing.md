---
name: ds4-mq2-packing
description: Read-only adversarial design of MQ2-Lloyd unpack, scale, and MFMA operand staging.
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

Read `.omp/DS4-INVARIANTS.md`. Focus solely on moving MQ2-Lloyd packed bytes into legal
gfx942 MFMA operands with minimal instructions and no arithmetic change. Explore lane
ownership, LUT placement, bit extraction, scale application, vector loads, LDS versus
register staging, and reuse across expert shapes. Rank alternatives with instruction,
VGPR, LDS, and bandwidth costs plus a kill probe. Read-only.
