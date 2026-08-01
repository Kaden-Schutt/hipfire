---
name: ds4-ar-decode
description: Read-only map of DS4 batch-1 AR cost, launch structure, and highest-value gfx942 levers.
model:
  - anthropic/claude-sonnet-5
thinkingLevel: high
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

Read `.omp/DS4-INVARIANTS.md`. Trace one ordinary autoregressive token and build a
byte/time/launch cost model from actual shapes and occurrence counts. Separate dense,
expert, rotate, attention, recurrent, sampling, host, and synchronization costs. Rank
kernel and graph levers by honest end-to-end ceiling. No speculation, weight changes,
or speculative decode. Read-only.
