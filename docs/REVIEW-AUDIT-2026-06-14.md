# Markdown Review Audit (2026-06-14)

## Scope and outcome

- Repository markdown inventory: **276 files** (`.md`) at the time of audit.
- Canonical documentation surface has been separated to:
  - `docs/` (clean, active pointers and status)
  - `docs-old/` (historical/archive copy of previous documentation corpus)
- Active canonical files are listed in this directory and linked from `docs/README.md`.
- `docs/ARCHIVE-INDEX.md` includes the complete archive map for traceability.

## Repository-wide distribution

- 180 files in `docs-old/` (archived historical docs)
- 13 files in `docs/` (canonical surface)
- 96 files outside `docs` and `docs-old` (benchmarks, experiments, crates, tests, scripts, and third_party notes)

## Non-archive top-level markdown

- `AGENTS.md`
- `AGENTS.local.md`
- `BUGS.md`
- `CHANGELOG.md`
- `CLAUDE.md`
- `CREDITS.md`
- `README.md`
- `TODO.md`

This list now excludes `BUGS-GEMINI.md`, `DOCS-GEMINI.md`, and `MANUAL_REVIEW.md`, which have been superseded and removed.

## Non-archive subtrees

- `benchmarks/`
- `crates/`
- `experiments/`
- `scripts/`
- `tests/`
- `third_party/`

## Review policy for this state

1. Do not edit historical files in `docs-old` unless explicitly needed for migration tasks.
2. Add or revise canonical guidance only in `docs/`.
3. For active work in `benchmarks/`, `experiments/`, `crates/`, and `tests/`, treat each markdown file as a status artifact tied to local code, and refresh links/headings if they feed canonical decisions.
4. When a sub-tree document contains reusable standard process, migrate it into `docs/` and replace the old file with a pointer note.

## Next consolidation target

- Produce a curated second pass for high-leverage non-archive markdown roots:
  - `README.md`, `TODO.md`, `BUGS.md`
  - modularization-related notes in `benchmarks/`, `experiments/`, and `crates/`
  - test procedure docs in `tests/README.md` and `tests/smoke/README.md`
- Promote any process that drives repeated decisions into `docs/reference/` and `docs/plans/`.
