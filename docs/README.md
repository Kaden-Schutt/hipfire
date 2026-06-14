# hipfire Documentation

This directory is the authoritative documentation surface for hipfire engineering workflow.

Use this space for
- stable design guidance,
- modularization execution state,
- operational checklists,
- and an evidence-linked archive pointer.

Do not add one-off experiments here unless they are accepted as canonical. Those belong in
`docs-old/` for traceability.

## Canonical Pages

- [OVERVIEW.md](./OVERVIEW.md) — one-page statement of the current documentation organization.
- [ARCHIVE-INDEX.md](./ARCHIVE-INDEX.md) — complete catalog of everything moved to `docs-old` with stable links.
- [plans/ARCHITECTURE-PLAN.md](./plans/ARCHITECTURE-PLAN.md) — current architecture + modularization plan.
- [reference/STATUS.md](./reference/STATUS.md) — current doc quality and drift state.
- [reference/CHECKLIST.md](./reference/CHECKLIST.md) — required doc updates for active work.

## Canonical Modularization Set (read first for ongoing implementation)

- `docs/plans/modular-runtime-architecture.md` → canonical pointer to archive plan
- `docs/plans/session-serving-feature-chart.md` → canonical pointer to archive plan
- `docs/plans/multi-model-session-state-serving.md` → canonical pointer to archive plan
- `docs/plans/priority-microbatching-scheduler.md` → canonical pointer to archive plan
- `docs/plans/rust-cli-server-port.md` → canonical pointer to archive plan
- `docs/plans/stabilize-before-extraction.md` → canonical pointer to archive plan
- `docs/plans/v1-architecture-roadmap.md` → canonical pointer to archive plan

These files are canonical pointers to the corresponding documents in `docs-old`.

## Archive policy

- `docs-old/` preserves the full historical record; do not edit historical files during active implementation.
- Add new canonical docs under `docs/` with narrow purpose and explicit scope.
- Move obsolete copies here only when they materially conflict with canonical guidance.
- If a historical file is used as active reference, either migrate it or explicitly link it from a canonical wrapper document.

## Quick navigation

- Server/runtime docs: check README and root `README.md` sections.
- Skills: `docs-old/skills/...` (historical) and `docs/reference/CHECKLIST.md` for when to apply one.
- Governance / relicense style notes: `docs-old/governance/`.
