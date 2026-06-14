# Documentation Status

## Canonical surface

Current docs surface:
- `docs/` (active, authoritative)
- `docs-old/` (archival)

## Active-vs-archive split at time of this edit

- Active markdown files: 13
- Archived markdown files: 180

## What was done

- `./docs` was renamed to `./docs-old` and retained for evidence preservation.
- A fresh `docs` was created with:
  - `docs/README.md`
  - `docs/OVERVIEW.md`
  - `docs/ARCHIVE-INDEX.md`
  - `docs/plans/ARCHITECTURE-PLAN.md`
  - `docs/reference/CHECKLIST.md`
  - `docs/reference/STATUS.md`

## Drift and review pass notes

- A grep pass found many markers indicating supersession, deprecation, or stale notes in the archive.
- Plans and investigation docs are intentionally preserved as historical context, not as active truth.

## Canonical pointers added

- Added thin canonical pointer pages in `docs/plans/` for modularization and serving architecture plans.

## Pending actions (recommended)

- Add dedicated canonical docs for CLI and serve usage if teams need active references there.
- Add a short glossary for feature-state terms used in plan tables.
- Add a quarterly docs curation pass before major release checkpoints.
