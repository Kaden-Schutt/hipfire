# Agentic PR Review Workflow Design

**Date:** 2026-07-15
**Status:** Approved for planning

## Purpose

Add a lightweight, manually invoked PR-review workflow that supplements
the existing heavyweight GitHub gate system. It identifies open pull
requests whose current head has not received this workflow's static review,
performs one static review, and leaves durable GitHub-native evidence for
future review and hardware-validation agents.

The initial scope is deliberately limited to two agent skills:

1. discovery of PRs needing review; and
2. static review of one PR.

It does not replace existing gates, execute tests, select hardware, or
approve PRs.

## Scope and decisions

- Skills are manually invoked only.
- Discovery scans every open PR, including drafts and fork PRs.
- `needs-review` means the current PR head SHA has no accepted static-review
  report from this workflow.
- A completed static review removes `needs-review` whether it is clean or
  requests changes. A new head SHA makes prior review evidence stale; the
  next discovery run reapplies the label.
- Static review posts a visible PR report. Findings also submit GitHub's
  `request changes` review state. Clean reviews never approve the PR.
- Static review performs no test execution. Test and hardware validation are
  separate downstream concerns.
- Only reports from writers or higher may satisfy discovery. This supports
  whichever GitHub token is available on the machine running the skill.

## Architecture

### Discovery skill

The discovery skill enumerates open PRs and reads their current head SHAs,
labels, and comments. It recognizes only a valid, versioned report metadata
block attached to a visible report comment where:

- the report is from this workflow;
- the recorded SHA equals the current head SHA; and
- the comment author's current effective repository permission is `write`,
  `maintain`, or `admin`.

For each PR without such a report, it adds `needs-review` idempotently. It
does not remove labels based on ambiguous, malformed, untrusted, or
unverifiable reports. It summarizes every PR it could not inspect fully.

### Static-review skill

The review skill accepts one open PR currently labelled `needs-review`. It
reviews the diff and repository context statically, without executing PR
code or tests. It assesses:

- correctness and behavioral regressions;
- safety, data-integrity, compatibility, and error-handling risks;
- architectural debt and boundary violations introduced by the change;
- test coverage, test rigidity, and whether tests prove the intended
  behavior; and
- relevant API, documentation, and performance implications.

It posts a visible report pinned to the reviewed head SHA. If it finds
actionable issues, it separately submits a GitHub `request changes` review;
it never submits approval. Before it removes `needs-review`, it refetches
the PR head. If the head changed, it leaves the label in place and reports
that its result is stale.

### Future validation agents

Separate machines periodically scan static-review reports. They match
generic validation capabilities against their available hardware, trust
policy, and capacity, then run their own validation and publish separate
SHA-bound results. They do not revise or overwrite the static assessment.

This keeps code-aware impact assessment with the reviewer while keeping
volatile fleet topology and concrete machine routing with hardware agents.
There is no separate triage skill initially.

## Review report protocol

Each report has two layers:

1. a human-readable Markdown comment; and
2. a compact, versioned machine metadata block.

The visible comment states the reviewed SHA, verdict (`clean` or
`changes-requested`), findings grouped by severity and criterion, and a
static impact assessment. The metadata uses an `agentic-review/v1` schema
and contains at least:

- producer and schema version;
- PR number and reviewed head SHA;
- verdict and completion status;
- affected subsystems and architecture families;
- supporting diff locations and confidence;
- generic validation capability requests; and
- stable request IDs for downstream validation.

Validation requests are capability-level rather than machine-level, for
example `rdna3-smoke`, `gfx1151-kernel-validation`, or
`dflash-coherence`. Each is bound to the reviewed SHA. Any new or
force-pushed head supersedes all requests and results for the prior SHA.

The initial request lifecycle records at least `pending`, `satisfied`, and
`superseded`. Stable IDs reserve a path to later `claimed`, `waived`, and
lease-based processing without requiring an event-sourcing service now.

## Trust model

Report author identity cannot depend on a single bot account because each
manual skill run uses the token available on its machine. Discovery therefore
queries the report author's effective repository permission and accepts only
`write`, `maintain`, or `admin`.

This deliberately treats anyone with repository write access as a trusted
maintenance principal. Such a user could already change labels and review
state; a signed report format adds little protection against that authority.
Reports from readers, triagers, external contributors, unknown identities,
or authors whose permission cannot be verified are ignored. The label stays
or is reapplied in those cases.

## Failure behavior

The workflow is idempotent and fails closed:

- Failed comment parsing, permission checks, or head-SHA fetches never clear
  `needs-review`.
- The reviewer exits without mutation when the PR is closed or unlabelled.
- GitHub mutation failures are reported explicitly; no operation is inferred
  to have succeeded.
- A changed head during review leaves the PR reviewable for the next run.
- Missing or unsupported downstream validation mapping is recorded as an
  explicit uncertainty, never silently interpreted as no validation needed.

## Verification strategy

The skill implementation must fixture-test the shared protocol and GitHub
interaction layer for:

- a PR with no report;
- a valid report for the current SHA;
- stale-SHA reports after a push or force-push;
- malformed metadata;
- insufficient or unverifiable reporter permission;
- duplicate discovery and review runs;
- clean and changes-requested reports;
- a head change between review start and label removal; and
- stale validation requests.

These tests verify workflow behavior only. They do not execute target
hardware, model, or PR tests.

## Follow-on work deliberately deferred

- Scheduling discovery in GitHub Actions.
- A dedicated triage skill.
- Exact hardware machine selection and test execution.
- A versioned subsystem/path-to-capability policy manifest.
- Full append-only obligation lifecycle, claims, leases, waivers, and result
  aggregation.
- Approval automation or merge gating.
