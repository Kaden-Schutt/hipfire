# Documentation Overview

`docs/` now holds the canonical, low-friction documentation set for active engineering and release work.
`docs-old/` is the historical archive.

## Why this split

The old documentation tree mixed:
- current decisions,
- superseded architecture threads,
- lab notebooks,
- benchmark transcripts,
- and ad hoc experiments.

Keeping both in one folder made it difficult to determine source of truth.

## What is active here

`docs/plans/ARCHITECTURE-PLAN.md` is the anchor for the modularization trajectory and current constraints.
`docs/reference/STATUS.md` tracks drift and evidence coverage.
`docs/ARCHIVE-INDEX.md` tracks all moved historical Markdown files.

## How to keep it coherent

1. Update canonical plan files first.
2. Update status/checklist second.
3. Reference older work only through links into `docs-old`.
4. Delete or rename stale canonical pages only with an explicit migration note.
