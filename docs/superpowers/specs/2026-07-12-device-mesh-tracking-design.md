# Device-Mesh Refactor Tracking Design

## Purpose

Establish one durable source of truth for completing the device-mesh refactor while preserving the branch's chronological design, implementation, and validation record.

The current documentation contains accurate historical evidence mixed with stale forward-looking statements. Search-driven readers can therefore encounter contradictory claims such as "not started" and "complete" for the same milestone. The cleanup must remove ambiguity without rewriting history.

## Source Of Truth

`.agent-progress/device-mesh-refactor-tracker.md` is the authoritative status document.

PR #527 mirrors the tracker's open checklist for contributor visibility. If the PR and committed tracker differ, the committed tracker is authoritative. Every status-changing commit must update the tracker; the PR mirror should be refreshed after the commit is pushed.

Historical handovers, task reports, and ledgers remain evidence, not current status authorities.

## Tracker Structure

The tracker contains:

1. Scope and explicit completion definition.
2. Current status summary.
3. Completed foundations with commit and validation evidence.
4. Open milestones with stable task IDs.
5. Dependency order and parallelizable work.
6. Hardware-validation requests.
7. Final merge-readiness gate.
8. Update protocol.

Each open task records:

- stable ID;
- status: `blocked`, `ready`, `in progress`, or `complete`;
- goal and bounded scope;
- dependencies;
- acceptance criteria;
- required tests or validation commands;
- hardware requirements;
- evidence links or commit hashes when complete.

The tracker uses checkboxes only for actionable completion criteria. Historical achievements are recorded as completed evidence rather than mixed into the active queue.

## Milestones

The open work is grouped into these milestones:

1. Physical hardware validation for EP, dense PP, Qwen35 PP, and TP teardown.
2. Correctness and ownership debts: `mtp_k`, reset totality, parser finalization, and `SessionState` ownership.
3. Remaining ordinary generation coverage: Qwen35 arch-resident PP and DeepSeek4 single-GPU fallback.
4. Shared request lifecycle above AR and speculative strategies.
5. Multimodal post-prefill lifecycle adoption for Qwen35-VL and dots.ocr.
6. Step/manifest adoption for DeltaNet, MoE, recurrent convolution, and remaining architecture forwards.
7. Additional parallel architecture support, including model-family TP/EP decisions.
8. Optional composed TP x EP, gated on a concrete deployment requirement.
9. Documentation consolidation and final merge readiness.

Hardware validation is tracked first because further structural work must not obscure unproven physical topology behavior.

## Documentation Cleanup

Cleanup is surgical:

- Add a prominent superseded banner to stale top-level progress and handover documents.
- Link each banner to `.agent-progress/device-mesh-refactor-tracker.md`.
- Preserve dated historical bodies and validation evidence unchanged unless a statement falsely claims to be current at the document top level.
- Label `.superpowers/sdd/progress.md` as a chronological, append-only implementation ledger.
- Resolve contradictory current-status headings in the god-struct handover while retaining its historical sections.
- Update stale review-finding status where current code conclusively closed the finding.
- Do not delete historical reports or rewrite old implementation decisions as if they had always been known.

Documents requiring cleanup include:

- `.agent-progress/device-mesh-HANDOVER.md`
- `.agent-progress/device-mesh-status.md`
- `.agent-progress/device-mesh-phase0.md`
- `.agent-memory/notes/device-mesh-next-followups.md`
- `.agent-memory/notes/device-mesh-review-findings-2026-07-10.md`
- `.agent-memory/notes/device-mesh-pivot-execute-steps-spine.md`
- `.agent-memory/notes/daemon-god-struct-archdispatch-design.md`
- `.agent-memory/notes/godstruct-collapse-handover-2026-07-11.md`
- `.superpowers/sdd/progress.md`

Task reports remain immutable historical artifacts unless their top-level title incorrectly identifies them as current work.

## PR Synchronization

PR #527 keeps the full architecture report and contributor hardware request. A new "Authoritative Work Tracker" section is appended containing:

- a link to the committed tracker;
- the current milestone checklist using the same stable IDs;
- the update rule stating that the committed tracker wins on divergence.

The PR checklist is a collaboration surface, not an independent planning document. Detailed acceptance criteria stay in the committed tracker to avoid exceeding PR-body limits and duplicating large blocks of text.

## Update Protocol

For every tracked task:

1. Mark exactly one task `in progress` before implementation begins.
2. Record newly discovered blockers or follow-up tasks immediately.
3. Do not mark a task complete until all acceptance criteria and required validation pass.
4. Add commit hashes, hardware details, and validation evidence when completing it.
5. Push the tracker update with the implementation or validation commit.
6. Refresh the PR checklist after push.

Tasks blocked on unavailable hardware remain open and explicitly list the required topology. Emulation evidence never closes a physical-hardware acceptance criterion.

## Validation

Documentation cleanup is validated by:

- searching stale documents for unqualified current-status phrases;
- confirming every superseded document links to the canonical tracker;
- checking that tracker task IDs are unique;
- checking that every open task has acceptance criteria and dependencies;
- verifying the PR body contains the tracker link and matching open milestone IDs;
- running Markdown-sensitive checks available in the repository, if any.

No runtime behavior changes are part of this documentation cleanup.

## Non-Goals

- Rewriting chronological implementation history.
- Deleting forensic validation records.
- Creating one GitHub issue per task.
- Implementing any open refactor item as part of documentation cleanup.
- Claiming production readiness based on emulated multi-GPU validation.
