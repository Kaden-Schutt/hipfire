---
name: ds4-prefill
description: Read-only audit and optimization map for DS4 gfx942 prefill GEMM/MFMA routes.
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

Read `.omp/DS4-INVARIANTS.md`. Map the real DS4 prefill route, shapes, batching,
fallbacks, attention, expert grouping, and compilation dispatch on gfx942. Identify
where GEMV fallback or unsuitable tiling remains. Rank only arithmetic-preserving
MFMA/fusion/launch levers, with measured-denominator requirements and adjacent-arch
guards. Do not conflate prefill wins with AR wins. Read-only.
