---
name: agent-memory
description: In-repo git-tracked lexical agent memory via .agent-memory/ and scripts/mem.sh. Use before answering project/architecture questions, when starting work that may repeat past findings, or after learning a durable decision/falsification/gotcha.
---

# agent-memory

Cross-session **project** memory: plain markdown notes + ripgrep recall. No DB, no embeddings, no MCP process.

## Canonical owners (verify before use)

| Path | Role |
|---|---|
| [`scripts/mem.sh`](../../../scripts/mem.sh) | CLI: `recall` / `remember` / `list` / `path` |
| [`.agent-memory/README.md`](../../../.agent-memory/README.md) | Contributor intro + conventions |
| [`.agent-memory/notes/*.md`](../../../.agent-memory/notes/) | One finding per note (YAML frontmatter) |

If `scripts/mem.sh` or `.agent-memory/` is missing, **fail closed**: report the missing path, do not invent a substitute store, and point the user at restoring those owners from git history.

## Reach for this when

- Answering "how does X work", "did we try Y", architecture/hotspot questions
- Starting work that resembles past kernel/dispatch/perf investigations
- After learning something durable: finding, falsification, decision, gotcha

## Commands

From repo root:

```sh
scripts/mem.sh recall <key terms...>
scripts/mem.sh remember <slug> "<one-line title>" [tag1,tag2]
scripts/mem.sh list
scripts/mem.sh path
```

- **`recall`** — ripgrep-ranks `.agent-memory/notes/*.md` by match density; open the relevant notes. Prefer this over loading a flat index.
- **`remember`** — creates `notes/<slug>.md`; edit the body (terse; link related notes by slug), then commit **with the work it describes**. One finding per note. Prefer recording **why** something failed.
- **`list` / `path`** — inventory and notes directory.

Requires `rg` for ranked recall; without it the script falls back to plain `grep` paths.

## What goes where

- **Here (committed):** hipfire findings, falsifications, decisions, kernel/dispatch facts, perf verdicts — shared, diffable, travels with the code.
- **Global agent memory (not committed):** personal preferences, machine/fleet config, ssh hosts, cross-project lessons.

## Consolidation

Notes accrue. Periodically `list` + `recall` a topic; merge overlaps or set `superseded_by: <slug>` on stale frontmatter and commit. Keep titles short; detail lives in the body.

## Other memory axes (not substitutes)

- **Harness long-term memory** (`retain` / `recall` / `reflect` tools) — session/agent durable facts outside this repo store.
- **`codebase-memory` MCP** — code structure (call graphs, symbols). Different axis from note memory.
