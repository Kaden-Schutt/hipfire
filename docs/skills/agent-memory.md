# Skill: agent-memory (git-tracked, lexical recall)

Cross-session project memory as in-repo markdown notes with ripgrep recall — no DB,
no embeddings, no MCP process. Store + helper live at `.agent-memory/` and
`scripts/mem.sh`; `.agent-memory/README.md` is the contributor-facing intro.

**Reach for this when:** you're about to answer a project/architecture question
("how does X work", "did we try Y"), starting a task that resembles past work, or
you've just learned something durable (a finding, a falsification, a decision, a
gotcha).

## Recall before answering
Before answering a project/architecture question or starting a hotspot task, run:

```sh
scripts/mem.sh recall <key terms>
```

It ripgrep-ranks `.agent-memory/notes/*.md` by match density and returns the top
notes (title + path + snippet). Open the relevant ones. This **replaces loading a
flat index** — recall on demand, so there is no size ceiling to nag about.

## Remembering
When you learn something worth keeping across sessions:

```sh
scripts/mem.sh remember <slug> "<one-line title>" tag1,tag2
```

Edit the created file's body (terse; link related notes by slug), then commit it
**with the work it describes**. One finding per note. Prefer recording **why**
something failed — those notes are the ones that save future cycles.

## What goes where
- **Here (in-repo, committed):** hipfire findings, falsifications, decisions,
  kernel/dispatch facts, perf verdicts — anything a contributor or future session
  would want. Shared, diffable, travels with the code.
- **Global agent memory (not committed):** personal preferences, machine/fleet
  config, ssh hosts, cross-project lessons.

## Consolidation
Notes accrue. Periodically `mem.sh list` + `recall` a topic; if several notes
overlap or one supersedes another, merge them or add `superseded_by: <slug>` to the
stale one's frontmatter, and commit. Keep titles short — detail lives in the body.

## Relationship to other memory
- The **global flat index** (`~/.claude/.../memory/MEMORY.md`) stays small — personal
  + cross-project pointers only. Project knowledge lives *here* now.
- This is *note* memory (decisions/findings). For *code-structure* questions (call
  graphs, symbol relations) the `codebase-memory` MCP server is the complementary
  tool — different axis, not a replacement.
