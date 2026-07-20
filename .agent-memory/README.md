# .agent-memory — git-tracked agent memory

Cross-session memory for AI coding agents (Claude Code, Codex, etc.) working on
hipfire, kept as **plain markdown with YAML frontmatter**, in-repo and version-
controlled — so the agent's accumulated findings, falsifications, and decisions
are **shared with contributors, diffable in PRs, and travel with the code**. This
is the project's "git commit everything; the history IS the research" rule applied
to agent memory.

**No database, no embeddings, no service.** Recall is **lexical (ripgrep)** — the
right tool for a few-hundred terse, keyword-dense engineering notes (kernel names,
env flags, commit SHAs). Embeddings only earn their cost on large, paraphrase-heavy
corpora; we are nowhere near that, and a substring match is faster and zero-dep.

## Layout
- `notes/<slug>.md` — one finding per file. Frontmatter: `title`, `date`, `tags`
  (optionally `superseded_by: <slug>`).
- `../scripts/mem.sh` — the `recall` / `remember` / `list` helper.

## Use
```sh
scripts/mem.sh recall verify graph int8         # ripgrep-ranked recall
scripts/mem.sh remember verify-ceiling "DFlash verify is at its kernel ceiling" perf,spec-decode
scripts/mem.sh list
```

## Conventions
- **Project findings go here** (committed, shared). Personal/machine-specific notes
  (fleet config, ssh hosts, preferences) stay in your *global* agent memory — not
  committed here.
- **One finding per note; terse.** Detail goes in the body, not in long titles.
  Link related notes by slug in prose.
- **Record _why_ a thing failed**, not just that it did — the falsifications are the
  most valuable notes (they narrow the search space).
- **Commit notes alongside the work they describe.** A note + its code change in one
  commit is a self-documenting decision log.
- **Recall before answering**, remember after learning. Helper: `scripts/mem.sh`. Sole skill root is `.agents/skills/` — `docs/skills/` is retired/removed and is not a second reference root.

## Why lexical, not embeddings (and what would change that)
At this scale, `rg` beats a vector model on speed, footprint, and zero-config. If the
corpus ever grows to thousands of fuzzy notes where paraphrase recall matters, the
upgrade path is a semantic re-rank tier over the same markdown notes — but that's a
deliberate future step, not a day-one dependency.
