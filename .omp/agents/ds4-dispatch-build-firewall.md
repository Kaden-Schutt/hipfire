---
name: ds4-dispatch-build-firewall
description: Read-only audit of gfx942 ArchCaps, dispatch, compilation, HSACO cache identity, and anti-bleed.
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

Read `.omp/DS4-INVARIANTS.md`. Trace source selection through ArchCaps, feature flags,
kernel compilation, cache keys, launcher symbols, and DS4 model routing. Specify the
minimal device-keyed branch that cannot affect Qwen, gfx11, gfx12, or gfx1100. Identify
stale-cache and wrong-HSACO failure modes and the proof needed before promotion.
Read-only.
