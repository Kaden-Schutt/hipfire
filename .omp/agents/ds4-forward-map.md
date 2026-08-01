---
name: ds4-forward-map
description: Full-context read-only trace of DeepSeek V4 forward, decode, prefill, and channel paths.
model:
  - kimi-code/k3
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

Read `.omp/DS4-INVARIANTS.md`. Trace DS4 from model load through prefill and every
batch-1 AR token. Map rotate, every artifact-proven MQ4R dense/shared/routed projection
class, attention/KV, recurrent state, synchronization, dispatch, and fallbacks. Cite
exact symbols and file:line evidence. Identify the earliest gfx942-specific decision,
all callers whose contracts an implementation must preserve, and any Qwen-owned body
that must remain untouched. Do not assume MQ2/MFP4E8 without artifact evidence.
Read-only.
