---
name: ds4-model-delta-0731
description: Read-only structural and semantic diff of DeepSeek V4 Flash preview versus the 0731 checkpoint.
model:
  - kimi-code/k3-256k
thinkingLevel: high
readSummarize: false
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

Read `.omp/DS4-INVARIANTS.md`. Compare the pinned preview config/tokenizer/tensor
inventory with `deepseek-ai/DeepSeek-V4-Flash-0731` without quantizing anything.
Report source revisions, shard hashes/completeness, architecture/config deltas, tensor
renames/shapes/classes, expert/router changes, tokenizer/RoPE/KV changes, and which
quantizer/runtime assumptions must be updated. Distinguish verified facts from absent
files. Do not use preview logits as a quality reference. Shell use is read-only.
