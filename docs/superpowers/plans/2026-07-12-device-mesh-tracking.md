# Device-Mesh Refactor Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one committed device-mesh completion tracker, clearly archive stale status documents, and mirror the active checklist in PR #527.

**Architecture:** `.agent-progress/device-mesh-refactor-tracker.md` becomes the sole status authority. Historical notes retain their dated evidence but receive prominent superseded banners; PR #527 mirrors only the active milestone checklist and links back to the committed tracker.

**Tech Stack:** Markdown, Git, GitHub CLI, ripgrep-compatible repository search.

---

## File Structure

- Create `.agent-progress/device-mesh-refactor-tracker.md`: authoritative completion definition, evidence, active task IDs, dependencies, acceptance criteria, and update protocol.
- Modify `.agent-progress/device-mesh-HANDOVER.md`: add superseded banner.
- Modify `.agent-progress/device-mesh-status.md`: add superseded banner.
- Modify `.agent-progress/device-mesh-phase0.md`: add superseded banner.
- Modify `.agent-memory/notes/device-mesh-next-followups.md`: add superseded banner and current-state correction.
- Modify `.agent-memory/notes/device-mesh-review-findings-2026-07-10.md`: add superseded banner and close the stale EP-manifest finding.
- Modify `.agent-memory/notes/device-mesh-pivot-execute-steps-spine.md`: label as historical execution spine.
- Modify `.agent-memory/notes/daemon-god-struct-archdispatch-design.md`: label as historical design/status ledger.
- Modify `.agent-memory/notes/godstruct-collapse-handover-2026-07-11.md`: make the completed status unambiguous while preserving history.
- Modify `.superpowers/sdd/progress.md`: label as chronological append-only ledger.
- Modify PR #527 body: append authoritative tracker link and mirrored milestone checklist.

### Task 1: Create The Canonical Tracker

**Files:**
- Create: `.agent-progress/device-mesh-refactor-tracker.md`

- [ ] **Step 1: Write the tracker header and authority rule**

Add a current-status header naming this file as authoritative, linking [PR #527](https://github.com/Kaden-Schutt/hipfire/pull/527), and stating that committed tracker state wins on divergence.

- [ ] **Step 2: Record completed foundations as evidence**

Record the completed DeviceMesh, manifest, TP/PP/EP, `ArchDispatch`, `StreamParser`, `ModelParallel`, and `LoadedModel` milestones with representative commit ranges and validation evidence. Do not use active checkboxes for completed history.

- [ ] **Step 3: Add stable active task IDs**

Use these milestone prefixes:

```text
HW   physical hardware validation
COR  correctness and ownership debt
GEN  remaining ordinary generation coverage
SPEC shared AR/speculative request lifecycle
VL   multimodal lifecycle
STEP Step/manifest architecture adoption
PAR  additional parallel architecture support
COMP optional composed-topology scope decisions and conditional implementation
DOC  documentation cleanup, final validation, and merge readiness
```

Keep optional TP x EP composition under `COMP-*`. Keep documentation cleanup and the final completion/merge gate under `DOC-*`; do not use `PAR-*` for composition or `COMP-*` for final completion.

The tracker contains one immutable bootstrap migration table because `7115135e` and the initial PR mirror published `PAR-003` for optional TP x EP composition and `COMP-001` for the final gate before implementation began. Their corrected IDs are `COMP-001` and `DOC-002`; the old ID-to-meaning mappings are retired aliases that must never be reused, and IDs must not be renamed after this recorded correction. The simultaneous correction creates one documented token collision: current `COMP-001` means optional composition only, while final completion is exclusively `DOC-002`.

- [ ] **Step 4: Add complete task contracts**

For every active task include status, dependencies, goal, acceptance criteria, validation commands, hardware requirements, and evidence. Status must be exactly one lowercase schema value: `blocked`, `ready`, `in progress`, or `complete`. Initialize unfinished evidence fields to `Pending`. Use `None` where a dependency or hardware requirement does not exist; do not omit fields.

- [ ] **Step 5: Add dependency order and update protocol**

State which tasks can proceed in parallel and require exactly one `in progress` task per implementation stream. State that emulation cannot close physical hardware criteria.

- [ ] **Step 6: Validate tracker structure**

Run:

```bash
rg -n '^### [A-Z]+-[0-9]+' .agent-progress/device-mesh-refactor-tracker.md
rg -n 'Status:|Dependencies:|Acceptance criteria:|Validation:|Hardware:|Evidence:' .agent-progress/device-mesh-refactor-tracker.md
```

Expected: every task ID appears once, every task has all seven contract fields, and every status matches the lowercase schema.

### Task 2: Archive Stale Top-Level Status Documents

**Files:**
- Modify: `.agent-progress/device-mesh-HANDOVER.md:1`
- Modify: `.agent-progress/device-mesh-status.md:1`
- Modify: `.agent-progress/device-mesh-phase0.md:1`
- Modify: `.agent-memory/notes/device-mesh-next-followups.md:1`
- Modify: `.agent-memory/notes/device-mesh-pivot-execute-steps-spine.md:1`
- Modify: `.agent-memory/notes/daemon-god-struct-archdispatch-design.md:1`
- Modify: `.superpowers/sdd/progress.md:1`

- [ ] **Step 1: Add a uniform superseded banner**

After any YAML frontmatter, add:

```markdown
> **Historical document.** This file preserves dated implementation and validation evidence. Current status and remaining work are tracked only in [device-mesh-refactor-tracker.md](../../.agent-progress/device-mesh-refactor-tracker.md).
```

Adjust the relative link for files already under `.agent-progress` or `.agent-memory` so it resolves to `.agent-progress/device-mesh-refactor-tracker.md`.

- [ ] **Step 2: Label the SDD ledger append-only**

Add one sentence to `.superpowers/sdd/progress.md` stating that chronological entries may contradict later entries and must not be interpreted as current status.

- [ ] **Step 3: Preserve historical bodies**

Review the diff and confirm no dated implementation evidence was deleted or rewritten.

- [ ] **Step 4: Verify all archived documents link to the tracker**

Run:

```bash
rg -L 'device-mesh-refactor-tracker\.md' \
  .agent-progress/device-mesh-HANDOVER.md \
  .agent-progress/device-mesh-status.md \
  .agent-progress/device-mesh-phase0.md \
  .agent-memory/notes/device-mesh-next-followups.md \
  .agent-memory/notes/device-mesh-pivot-execute-steps-spine.md \
  .agent-memory/notes/daemon-god-struct-archdispatch-design.md \
  .superpowers/sdd/progress.md
```

Expected: no output.

### Task 3: Resolve Known Contradictory Status Claims

**Files:**
- Modify: `.agent-memory/notes/godstruct-collapse-handover-2026-07-11.md:1-81`
- Modify: `.agent-memory/notes/device-mesh-review-findings-2026-07-10.md:1-51`

- [ ] **Step 1: Preserve existing collaborator edits**

Read both current worktree versions before editing. Do not replace or discard the existing uncommitted updates.

- [ ] **Step 2: Fix the god-struct top-level contradiction**

Change the title and current-status preamble to state that the field collapse completed at `9c57148d`. Add a historical-section marker immediately before text that still says `ImmutableMeta` remains.

- [ ] **Step 3: Correct the review-finding roll-up**

Mark EP manifest replication as closed by `4f55a274`, `8c441c76`, and `be5c4bdb`. Keep the original finding text as historical evidence. Keep physical PP validation open.

- [ ] **Step 4: Add tracker links**

Add the same current-status authority link used by the archived documents.

- [ ] **Step 5: Inspect the focused diff**

Run:

```bash
git diff -- \
  .agent-memory/notes/godstruct-collapse-handover-2026-07-11.md \
  .agent-memory/notes/device-mesh-review-findings-2026-07-10.md
```

Expected: only status clarification, tracker links, and pre-existing collaborator edits; no historical evidence removed.

### Task 4: Validate Documentation Consistency

**Files:**
- Test: all files modified by Tasks 1-3

- [ ] **Step 1: Scan for unqualified stale authority claims**

Run:

```bash
rg -n 'NOT started|NEXT unit|REMAINING god-struct|sole source|authoritative' \
  .agent-progress .agent-memory/notes .superpowers/sdd
```

Expected: historical claims remain only in documents carrying the historical banner; the canonical tracker contains the sole current authority claim.

- [ ] **Step 2: Check Markdown whitespace**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 3: Review the complete documentation diff**

Run:

```bash
git diff --stat
```

Expected: one tracker, archival banners, and targeted status corrections only.

- [ ] **Step 4: Commit the documentation cleanup**

Stage only the tracker, approved spec/plan, and intended cleaned documents. Preserve unrelated worktree changes unless they are the reviewed collaborator updates included by Task 3.

Commit message:

```text
docs(device-mesh): establish canonical completion tracker

Assisted-by: OpenCode:openai/gpt-5.6-sol
```

### Task 5: Push And Synchronize PR #527

**Files:**
- Modify remotely: PR #527 body

- [ ] **Step 1: Push the documentation commits**

Run:

```bash
git push origin feature/device-mesh
```

Expected: branch update succeeds without force push.

- [ ] **Step 2: Build the PR tracker section**

Append an `## Authoritative Work Tracker` section containing:

- a repository link to `.agent-progress/device-mesh-refactor-tracker.md` on `feature/device-mesh`;
- the authority/divergence rule;
- one checkbox per active stable task ID;
- hardware-help labels beside blocked `HW-*` tasks.

- [ ] **Step 3: Update the PR body without replacing the architecture report**

Use `gh pr edit 527 --repo Kaden-Schutt/hipfire --body-file <file>` with the existing body plus the appended tracker section.

- [ ] **Step 4: Verify PR synchronization**

Run:

```bash
gh pr view 527 --repo Kaden-Schutt/hipfire --json isDraft,state,body,url
```

Expected: PR remains open and draft; body retains the full report and contains every active tracker task ID exactly once.

- [ ] **Step 5: Verify final branch state**

Run:

```bash
git status --short --branch
```

Expected: branch is synchronized with `origin/feature/device-mesh`; only explicitly excluded pre-existing worktree changes may remain.
