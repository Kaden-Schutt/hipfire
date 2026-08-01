---
name: ds4-shape-census
description: Read-only census of actual DS4 kernel shapes, occurrences, bytes, and route coverage.
model:
  - anthropic/claude-sonnet-5
thinkingLevel: high
tools: [read, grep, glob, lsp, bash]
spawns: []
output:
  properties:
    verdict: { type: string }
    evidence: { type: array, elements: { type: string } }
    risks: { type: array, elements: { type: string } }
    next_action: { type: string }
    kill_criterion: { type: string }
---

Read `.omp/DS4-INVARIANTS.md`. Derive the in-model kernel × shape × occurrence table
for DS4 prefill and one AR token from durable profiles, launch-shape corpora, and source
routing. Flag stale or wrong-architecture data. Report weighted coverage and which
shapes dominate bytes and time. Do not benchmark, build, edit, or infer missing values
as zero; shell use is read-only analysis.
