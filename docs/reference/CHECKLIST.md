# Documentation Hygiene Checklist

## Before merging any docs-facing architecture change

- [ ] Update `plans/ARCHITECTURE-PLAN.md` with the scope change.
- [ ] Update `reference/STATUS.md` if any canonical file moved or retired.
- [ ] If runtime/serve behavior changes, add/update a canonical status file in `docs/plans/`.
- [ ] Add a matching entry in `docs/plans/session-serving-feature-chart.md` for changed evidence.
- [ ] If feature gates changed, note the gating script and command in this check list.

## When a historical page becomes stale

- [ ] Confirm an archival replacement exists in `docs`.
- [ ] Leave old copy in `docs-old` and avoid in-place edits.
- [ ] Add a short link-and-reason in `ARCHIVE-INDEX.md` only if needed.
