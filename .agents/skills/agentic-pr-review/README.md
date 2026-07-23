---
name: agentic-pr-review
description: Full lifecycle skill for the agentic PR review workflow. Orchestrates preflight, discovery, one-shot review (build capsule → LLM inference → publish), and capsule inspection. Produces review comments with hardware validation triage and verify-<arch> labels. Use as the top-level entry point for automated PR review.
---

# Agentic PR review

Full lifecycle skill for the hipfire agentic PR review workflow.

Requires: Python 3.11+, `gh` CLI with fine-grained PAT, an LLM provider API key.

## Setup

```bash
# 1. Provider config — edit .github/agentic-review/providers.json
#    See the PR's Usage section for the example.

# 2. API key
export REVIEW_API_KEY="sk-..."
```

## Agent workflow

### 1. Preflight

Validate connectivity, credentials, and config:

```bash
python3 -m autoresearch.ar.review.cli preflight \
  --mode discovery --repository OWNER/REPO
```

Use `--config-ref feature-branch` when the review policy files haven't been
merged to the default branch yet.

### 2. Discovery

Scan open PRs and reconcile `needs-review` labels:

```bash
python3 -m autoresearch.ar.review.cli discover \
  --repository OWNER/REPO \
  --operator .github/agentic-review/operator-credentials.json
```

Outputs JSON with `needs_review`, `reviewed`, `labelled`, `clean`, and
`errors` arrays.  Exit code 1 means the scan was incomplete.

### 3. Review a PR (one-shot)

Build capsule → run inference → publish report → apply `verify-*` labels:

```bash
python3 -m autoresearch.ar.review.cli review \
  --pr 123 \
  --repository OWNER/REPO \
  --operator .github/agentic-review/operator-credentials.json \
  --provider review-adapter
```

The `review` command:
1. Builds the capsule (PR diff + file contents)
2. Runs toolless inference via the configured provider
3. Publishes the review as a PR comment with:
   - Verdict and findings
   - **Hardware validation triage** (impacted model families, hardware, coverage decision)
   - **`verify-<arch>` labels** applied to the PR (e.g. `verify-gfx1151`)

### 4. Inspect a PR (capsule build only, no publish)

For debugging or manual review before publishing:

```bash
# Build capsule only (no API key needed):
python3 -m autoresearch.ar.review.cli inspect \
  --pr 123 --repository OWNER/REPO \
  --capsule capsule.json

# Build + infer + save proposal (API key needed):
export REVIEW_API_KEY="sk-..."
python3 -m autoresearch.ar.review.cli inspect \
  --pr 123 --repository OWNER/REPO \
  --capsule capsule.json --proposal proposal.json \
  --provider review-adapter
```

## Coverage decision reference

The LLM analyzes the diff and sets `coverage_decision` in the triage output:

| Decision | Meaning |
|---|---|
| `all-impacted` | Every impacted model family needs hardware validation (shared-code change like dispatch, forward pass, kernels) |
| `representative-only` | Testing any one impacted model suffices (model-specific or narrow change) |
| `none` | No hardware validation needed (docs, CI, tooling only) |

## verify-* labels

Each impacted hardware architecture gets a `verify-<arch>` label on the PR.
Downstream agents discover validation tasks by scanning for these labels:

```
verify-gfx1100    verify-gfx1101    verify-gfx1102
verify-gfx1150    verify-gfx1151    verify-gfx1200
verify-gfx1201    verify-gfx94x
```

## Shared flags

All commands accept:

- `--token <ghx_...>` — GitHub token override
- `--config-ref <branch>` — config branch (needed when policy files aren't merged)
