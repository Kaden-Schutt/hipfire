---
name: ds4-mq2-mfma
description: Read-only competing designs for hot MQ2-Lloyd routed-expert MFMA kernels on gfx942.
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

Read `.omp/DS4-INVARIANTS.md`. Design and rank gfx942 MFMA routes for DS4 expert
gate/up and down projections using the actual shapes and MQ2-Lloyd layout. Account for
decode/unpack cost, wave64 lane ownership, indexed expert access, register pressure,
LDS, occupancy, and accumulation order. Give an honest Amdahl ceiling and a micro gate
that can reject each design before product implementation. Read-only.
