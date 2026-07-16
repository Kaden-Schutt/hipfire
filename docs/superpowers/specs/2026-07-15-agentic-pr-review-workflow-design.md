# Agentic PR Review Workflow Design

**Date:** 2026-07-15
**Status:** Revised with tool-less provider inference; pending approval

## Purpose

Add a lightweight, manually invoked PR-review workflow that supplements
the existing heavyweight GitHub gate system. It identifies open pull
requests whose current complete review target has not received this workflow's static review,
performs one static review, and leaves GitHub-native evidence for future
review and hardware-validation agents. PR-controlled data is always treated
as hostile input.

The initial scope is deliberately limited to two agent skills:

1. discovery of PRs needing review; and
2. static review of one PR.

It does not replace existing gates, execute tests, select hardware, or
approve PRs.

## Scope and decisions

- Skills are manually invoked only.
- Discovery scans every open PR, including drafts and fork PRs.
- `needs-review` means the current complete PR review target has no accepted
  static-review report from this workflow.
- A complete static review removes `needs-review` whether it is clean or
  requests changes; an incomplete report never does. A changed head, base,
  target ref, or merge-base makes prior evidence stale; the next discovery
  run reapplies the label.
- Static review posts a visible PR report. Findings also submit GitHub's
  `request changes` review state. Clean reviews never approve the PR.
- Static review performs no test execution. Test and hardware validation are
  separate downstream concerns.
- Only reports from authors whose effective repository permission normalizes
  to `write` or `admin` may satisfy discovery. This supports whichever
  sufficiently privileged GitHub token is available on the machine running
  the skill.
- A report covers a complete review target, not only a head SHA: repository
  and PR number, head repository and SHA, base ref and SHA, and merge-base
  SHA. A change to any field makes it stale.
- A report is accepted only when its content coverage is complete. Oversized,
  paginated, unavailable, or truncated diff data produces an incomplete
  report that cannot clear `needs-review`.

## Architecture

### Discovery skill

The discovery skill exhaustively paginates open PRs, their comments, and
reviews. It reads the complete review target, labels, and report comments.
It recognizes only a valid, versioned completion record where:

- the report is from this workflow;
- the report target equals the current PR target in every field;
- the report comment node ID and canonical metadata digest match its
  completion record; and
- both the report and completion-record authors independently satisfy the
  trusted-principal rule.

For each PR without such a report, it adds `needs-review` idempotently. It
does not remove labels based on ambiguous, incomplete, malformed, untrusted,
or unverifiable reports. It re-fetches the target after every label mutation;
if the target changed, it reconciles the label again. Permission results are
cached per author during one run. Any pagination, rate-limit, or retrieval
failure makes the scan incomplete and prevents a successful-scan claim.

Each discovery scan also enumerates workflow-owned review records and their
attempt IDs. It dismisses only reviews whose attempt has no canonical intent
or is revoked, duplicate, or stale; it never dismisses a human or unrelated
automation review. A scan that cannot perform this reconciliation fails
closed, retains `needs-review`, and reports the unresolved workflow review.

### Static-review skill

The static-review skill has two isolated stages.

The read-only **review controller** accepts one open PR currently labelled
`needs-review`. It has a read-only GitHub credential and a model-provider
credential, but no GitHub mutation credential. It reads only fixed GitHub API
endpoints and never checks out, executes, or exposes PR code. It builds a
strict `ReviewCapsule`, sends it as one tool-less structured-inference request,
and validates the resulting `ReviewProposal`. It assesses:

- correctness and behavioral regressions;
- safety, data-integrity, compatibility, and error-handling risks;
- architectural debt and boundary violations introduced by the change;
- test coverage, test rigidity, and whether tests prove the intended
  behavior; and
- relevant API, documentation, and performance implications.

Before inference, the controller builds a pinned merge-base/head manifest
from Git-tree and Git-blob API objects. It records every changed path, mode,
base and head blob OID, and byte size, then retrieves every required changed
blob by OID. Tree truncation, API file caps, a missing blob, an unavailable
or truncated patch/blob, or any manifest mismatch produces an incomplete
capsule. Counts alone never prove coverage.

The trusted, non-LLM **publisher** independently fetches the selected PR
target, validates the proposal against a closed schema, and permits only the
selected PR, its current target, a verdict enum, findings, the impact
envelope, and the closed validation vocabulary. It rejects arbitrary PR
numbers, URLs, commands, credentials, machine selectors, uncontrolled
Markdown, and every field outside that schema. It renders visible Markdown
itself from escaped structured fields rather than accepting inspector Markdown.
The publisher creates a
deterministic target key and a unique attempt ID.

Before any GitHub review mutation, the publisher appends a minimal,
authenticated attempt-intent record containing the target key, attempt ID,
and principal identity. It re-lists intents and elects the earliest valid,
non-revoked one by creation time then node ID. Only that canonical intent may
publish a visible report, completion record, or review-state mutation. Every
such record must carry the canonical intent node ID and attempt ID; discovery
rejects records that do not match the current canonical intent. A later
publisher resumes the canonical incomplete intent rather than creating a
competing one; a trusted revocation is the only way to replace it.

An attempt is complete only after its visible report completion record exists
and, for a `changes-requested` verdict, a matching GitHub review has been
submitted with `commit_id` exactly equal to the recorded head SHA. It never
submits approval.

Before and after removing `needs-review`, it refetches the complete target.
If any target field changed, it immediately reapplies the label and records
the attempt as stale. Retries reconcile existing attempts, report digests,
and review IDs before attempting a mutation. GitHub creates are not atomic:
concurrent attempts may exist temporarily, so the publisher elects the
earliest valid, non-revoked completion record by creation time then node ID
as canonical and marks later attempts duplicate. Before accepting a canonical
attempt, it reconciles and dismisses every workflow-owned
`REQUEST_CHANGES` review belonging to duplicate or revoked attempts for that
target key. A trusted publisher may append a revocation record containing the
target key, revoked attempt ID, reason, author identity, and canonical digest;
the record is authenticated under the same rule as a completion record. The
next valid non-revoked attempt becomes canonical.

### Future validation agents

Separate machines periodically scan accepted static-review reports. They
match generic validation capabilities against their available hardware,
trust policy, and capacity, then run their own validation and publish
separate results. They do not revise or overwrite the static assessment.

This keeps code-aware impact assessment with the reviewer while keeping
volatile fleet topology and concrete machine routing with hardware agents.
There is no separate triage skill initially.

## Review report protocol

Each run has append-only layers:

1. an authenticated attempt-intent record;
2. an optional incomplete report comment while a run is being finalized;
3. a human-readable Markdown report comment; and
4. a compact, versioned completion record that references the report comment
   node ID and canonical metadata digest.

Only the completion record is authoritative to discovery and downstream
validators. It is emitted after every required GitHub review mutation and
stale workflow-review dismissal has succeeded. The visible report states the full review target, verdict
(`clean`, `changes-requested`, or `incomplete`), findings grouped by severity
and criterion, and a static impact assessment. `clean` means zero actionable
findings. `changes-requested` means at least one actionable finding and a
successful matching `REQUEST_CHANGES` review. Nonblocking observations are
reported separately.

The `agentic-review/v1` metadata contains at least:

- producer and schema version;
- deterministic target key and unique attempt ID;
- canonical intent node ID;
- repository and PR identity; head repository and SHA; base ref and SHA; and
  merge-base SHA;
- verdict and completion status;
- report comment node ID and canonical metadata digest;
- report-author and completion-record-author node IDs, logins, and principal
  types;
- for App principals: app ID, installation ID, repository ID, and the trusted
  publisher credential-attestation digest in both the report and completion
  records;
- GitHub review ID when a changes-requested review is required;
- retrieved and expected file/content counts plus an explicit completeness
  flag;
- affected subsystems and architecture families;
- supporting diff locations and confidence;
- generic validation capability requests; and
- stable request IDs for downstream validation.

Every workflow-owned GitHub review also contains publisher-rendered machine
metadata with the complete review target, target key, attempt ID, canonical
intent node ID, exact `commit_id`, and a digest. Discovery authenticates its
author and validates all of those fields before treating it as workflow-owned
for reconciliation; a review without valid metadata is never dismissed.

The report and completion record are never edited by the skills after
publication. The canonical payload is RFC 8785 canonical JSON containing all
machine metadata except `metadata_digest` plus the SHA-256 of the exact UTF-8
visible Markdown body excluding the metadata block. `metadata_digest` is the
lowercase SHA-256 hex of that canonical payload. Consumers recompute both
hashes and reject altered or deleted records. The implementation supplies
byte-level digest test vectors. A replacement is a new attempt; a valid
replacement needs an explicit trusted revocation record for the canonical
attempt it replaces.

Validation requests are capability-level rather than machine-level. The
protected-default-branch `agentic-review/v1` registry owns the immutable
contract for every capability version. Each contract has a SHA-256 digest and
defines the empty parameter schema, eligible hardware, allowed suite
revisions, required check identifiers, required result artifacts, and the
pass condition. A request includes that contract digest; a validator can
satisfy it only when its result repeats the digest, uses an allowed suite
revision, supplies every required artifact, and passes every required check.

v1 has the following normative registry entries; each takes an empty
parameter object only:

| Capability | Eligible hardware | Required suite and satisfaction contract |
| --- | --- | --- |
| `hipfire/rdna3-smoke@1` | an RDNA3/gfx11 worker | The registry contract pins the `rdna3-smoke@1` suite revision, required checks, and evidence. |
| `hipfire/gfx1151-kernel-validation@1` | a gfx1151 worker | The registry contract pins the `gfx1151-kernel-validation@1` suite revision, required checks, and evidence. |
| `hipfire/dflash-coherence@1` | a supported GPU worker | The registry contract pins the canonical DFlash coherence-gate revision, hard-pass criterion, and evidence. |

Requests never contain shell commands, paths, environment variables, or
machine selectors. Unknown capabilities remain pending and unsatisfied. Each
request binds to the complete review target and report digest. A target
change supersedes all prior requests and results.

v1 static reports create immutable `pending` obligations only. A separate,
trusted validator result must reference the report digest, request ID, full
target, and its validator identity before any consumer can derive
`satisfied`. Supersession is derived from a changed current target, not by
editing a static report. Stable IDs reserve a path to later `claimed`,
`waived`, and lease-based processing without requiring an event-sourcing
service now.

## Hostile-input isolation

Fork and branch contents, PR title/body, comments, filenames, patches, and
linked content are untrusted data. Static review uses GitHub API reads pinned
to the recorded immutable SHAs; it must not use `git checkout`, `gh pr
checkout`, clone the PR, execute commands suggested by PR content, initialize
submodules, invoke Git hooks, fetch LFS objects, or execute any repository
code.

The controller presents PR-controlled text as quoted data and never treats it
as instructions. There is no local agent runtime, shell, tool registry, or
checkout to which PR content can issue commands. The controller is fixed code:
it exposes only allowlisted GitHub API reads, one configured model request,
and the publisher's separately implemented GitHub mutations.

## Tool-less provider inference

The static-review skill supplies `run-inspector.sh`, a shorthand launcher for
the controller's provider-neutral inference client. It accepts a canonical
`ReviewCapsule` and writes one schema-valid `ReviewProposal`; it neither runs
Codex/OpenCode nor starts a container image.

The protected-default-branch provider configuration selects an approved model
adapter, model identifier, endpoint, request deadline, capsule byte ceiling,
`max_output_tokens`, and per-run cost ceiling. Each adapter must send exactly
one model request with tools, function calling, browser access, code execution,
and provider-specific arbitrary headers disabled or absent. It sends fixed
review instructions plus the escaped capsule, never a PR-supplied instruction
or destination. The model-provider credential belongs only to the controller;
the model receives no GitHub credential and has no host runtime to inspect.

The client rejects redirects, unknown provider configuration, additional model
requests, incomplete streams, oversized input/output, deadline or cost-limit
violations, and any response that fails the `ReviewProposal` schema. It records
the capsule digest, provider adapter/version, model identifier, and response
digest in the proposal so the publisher can bind a report to the exact
inference input. Provider configuration and limits never come from the PR.

## Trust model

Report and completion-record author identity cannot depend on a single bot
account because each manual skill run uses the token available on its machine.
Discovery independently authenticates every intent, report, completion, and
revocation author. For user principals, it queries
`GET /repos/{owner}/{repo}/collaborators/{username}/permission` and accepts
only normalized `write` or `admin`. GitHub's `maintain` normalizes to
`write`; `triage` normalizes to `read`.

For GitHub App principals, discovery accepts only a configured app ID and bot
login. It verifies the configured app ID, installation ID, repository ID, and
credential-attestation digest in both records against the protected-default-
branch trusted-publishers configuration, then verifies that the installation
is currently scoped to the target repository. Unsupported principal types and
any unverifiable result are rejected.

This deliberately treats anyone with repository write access as a trusted
maintenance principal. Such a user could already change labels and review
state; a signed report format adds little protection against that authority.
Reports from readers, triagers, external contributors, unknown identities,
or authors whose permission cannot be verified are ignored. The label stays
or is reapplied in those cases.

Each skill package contains a read-only token-preflight script. It validates
token identity and target-repository access, probes every read endpoint used
by that skill with bounded pagination, and checks response permission headers
when available. Discovery requires pull-request reads, issue-comment reads,
the collaborator-permission endpoint, and issues write access only when it
will label PRs. The review controller needs immutable content reads and its
configured model-provider credential only; it receives no GitHub mutation
credential. Discovery and the publisher need pull-request writes to reconcile workflow-owned review
records and to submit or dismiss `REQUEST_CHANGES` respectively. Discovery's
credential must be explicitly attested as authorized to dismiss the
workflow's reviews on the target branch; otherwise it delegates reconciliation
to an equally attested publisher, retains `needs-review`, and reports the
unresolved state. The publisher also needs issue-comment writes.

The preflight never probes mutations. It fails closed on 401, 403, 404,
missing required read capability, unsupported principal, or incomplete
response. It rejects observable broad classic-PAT `repo` scopes under the
least-privilege policy and requires a repository-scoped fine-grained PAT or
GitHub App installation token in production. Because GitHub does not expose a
universal runtime manifest proving write grants or exclusivity for every
token type, every principal that can submit or dismiss a review additionally
requires an operator-attested credential manifest declaring its exact
repository and write permissions. The preflight proves required read
capabilities and observable scope limits; deployment configuration and the
attested manifest enforce the remaining least-privilege boundary.

## Failure behavior

The workflow is idempotent and fails closed:

- Failed comment parsing, permission checks, or head-SHA fetches never clear
  `needs-review`.
- The reviewer exits without mutation when the PR is closed or unlabelled.
- GitHub mutation failures are reported explicitly; no operation is inferred
  to have succeeded.
- A changed review target during review leaves the PR reviewable for the next
  run and supersedes all prior workflow-owned validation requests.
- Before publishing an authoritative completion record, the publisher
  dismisses only stale or noncanonical `REQUEST_CHANGES` reviews whose
  recorded review ID belongs to this workflow, including reviews from
  duplicate or revoked attempts. For a new changes-requested verdict, it
  creates the new review before dismissal. A dismissal failure leaves the
  attempt incomplete and labelled. Dismissal is never approval and is not
  applied to human or other automation reviews.
- A missing, truncated, or capped diff creates an `incomplete` report; it
  cannot be accepted and cannot remove the label.
- Missing or unsupported downstream validation mapping is recorded as an
  explicit uncertainty, never silently interpreted as no validation needed.

## Verification strategy

The skill implementation must fixture-test the shared protocol and GitHub
interaction layer for:

- a PR with no report;
- a valid report for the current complete target;
- stale targets after a push, force-push, base-branch advancement, or PR
  retargeting;
- malformed metadata;
- insufficient or unverifiable reporter permission;
- duplicate discovery and review runs;
- clean, incomplete, and changes-requested reports;
- a matching versus mismatching GitHub review `commit_id`;
- incomplete→complete publication and retry recovery;
- a target change between review start, report publication, and label
  removal;
- stale workflow-owned review dismissal without dismissing human reviews;
- altered or deleted report comments and digest mismatch;
- concurrent duplicate attempts, canonical-attempt election, revocation, and
  replacement, including a crashed noncanonical attempt that submits a review
  after canonical completion and discovery-side dismissal of its review;
- pagination, rate limiting, a 3,000-file cap, and unavailable/truncated
  patch content;
- a pinned merge-base/head Git-tree-and-blob manifest, including every changed path,
  mode, blob OID, size, missing blob, tree truncation, and manifest mismatch;
- completion records from an untrusted contributor, a trusted user, and a
  configured GitHub App principal, including invalid app/installation/
  repository/attestation bindings;
- the normative capability registry and incompatible/unknown capability
  requests;
- stale validation requests.

The token-preflight tooling is tested with mocked success, insufficient
capability, broad classic-scope, unsupported-principal, rate-limit, and
incomplete-response cases. It performs no mutation during verification.

These tests verify workflow behavior only. They do not execute target
hardware, model, or PR tests.

## Follow-on work deliberately deferred

- Scheduling discovery in GitHub Actions.
- A dedicated triage skill.
- Exact hardware machine selection and test execution.
- An expanded versioned subsystem/path-to-capability policy manifest beyond
  the v1 closed capability vocabulary.
- Full append-only obligation lifecycle, claims, leases, waivers, and result
  aggregation.
- Approval automation or merge gating.
