---
name: ds4-state-census
description: Read-only census of committed gfx942 support, protected source-WIP, artifacts, and provenance.
model:
  - anthropic/claude-sonnet-5
thinkingLevel: high
tools:
  - read
  - grep
  - glob
  - lsp
  - bash
spawns: []
output:
  properties:
    verdict: { type: string }
    evidence: { type: array, elements: { type: string } }
    risks: { type: array, elements: { type: string } }
    next_action: { type: string }
    kill_criterion: { type: string }
---

Read `.omp/DS4-INVARIANTS.md`. Inventory this worktree and, only when assigned,
read the protected source worktree without modifying it. Identify which gfx942 DS4
files, patches, artifacts, models, and measurements exist; distinguish committed,
modified, and untracked state. Produce a safe, explicit import manifest. Shell use is
read-only (`git status`, `git diff`, hashes, listings); never build or write.
