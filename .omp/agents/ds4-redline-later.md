---
name: ds4-redline-later
description: Read-only feasibility map for gfx942 retained AQL/PM4 after coherent HIP parity.
model:
  - openai-codex/gpt-5.6-terra
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

Read `.omp/DS4-INVARIANTS.md`, `docs/REDLINE.md`, and `docs/VALIDATION.md`. Map what
gfx942 retained replay would require, what is architecture-specific, and which HIP
route identities/correctness gates must exist first. Do not design around a broken HIP
kernel, run PM4 experiments, or edit replay code. Return only a deferred dependency
plan with explicit admission criteria.
