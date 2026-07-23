---
name: agentic-pr-review
description: Full lifecycle skill for the agentic PR review workflow. Orchestrates preflight, discovery, one-shot review (build capsule -> LLM inference -> publish), and capsule inspection. Produces review comments with hardware validation triage and verify-<arch> labels. Use as the top-level entry point for automated PR review.
---

# Agentic PR review

Full lifecycle skill for the hipfire agentic PR review workflow.

Requires: Python 3.11+, `gh` CLI with fine-grained PAT, an LLM provider API key.

## Provider configuration

Provider credentials are loaded from two files:

| File | Status | Purpose |
|---|---|---|
| `.github/agentic-review/providers.json` | **Checked in** | Schema, version, optional example entries |
| `.github/agentic-review/providers.local.json` | **Gitignored** | Per-developer real credentials |

The local file uses the same JSON schema. Its providers replace checked-in entries with the same `id` and append new ids. This lets the checked-in file stay minimal and public while developers keep their API keys local.

**Example checked-in file** (`.github/agentic-review/providers.json`):

```json
{
  "schema": "hipfire.agentic-review.providers",
  "version": 1,
  "providers": []
}
```

**Example local override** (`.github/agentic-review/providers.local.json`):

```json
{
  "schema": "hipfire.agentic-review.providers",
  "version": 1,
  "providers": [
    {
      "id": "review-adapter",
      "adapter_id": "openai-compatible",
      "adapter_version": "1",
      "endpoint": "https://api.deepseek.com/v1/chat/completions",
      "model": "deepseek-chat",
      "api_key_env": "DEEPSEEK_TOKEN",
      "max_requests": 1,
      "request_deadline_seconds": 120,
      "max_capsule_bytes": 262144,
      "max_response_bytes": 1048576,
      "max_tokens": 4096,
      "max_cost_usd": 0.5
    }
  ]
}
```

Set the API key: `export DEEPSEEK_TOKEN="sk-..."` (or `export REVIEW_API_KEY="sk-..."` if your provider uses that env var name).

## Agent workflow

### 1. Preflight

Validate connectivity, credentials, and config:

```bash
python3 -m autoresearch.ar.review.cli preflight \
  --mode discovery --repository OWNER/REPO
```

Use `--config-ref feature-branch` when the review policy files haven't been merged to the default branch yet.

### 2. Discovery

Scan open PRs and reconcile `needs-review` labels:

```bash
python3 -m autoresearch.ar.review.cli discover \
  --repository OWNER/REPO \
  --operator .github/agentic-review/operator-credentials.json
```

Outputs JSON with `needs_review`, `reviewed`, `labelled`, `clean`, and `errors` arrays. Exit code 1 means the scan was incomplete.

### 3. Review a PR (one-shot)

Build capsule -> run inference -> publish report -> apply `verify-*` labels:

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
export DEEPSEEK_TOKEN="sk-..."
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

Each impacted hardware architecture gets a `verify-<arch>` label on the PR. Downstream agents discover validation tasks by scanning for these labels:

```
verify-gfx1100    verify-gfx1101    verify-gfx1102
verify-gfx1150    verify-gfx1151    verify-gfx1200
verify-gfx1201    verify-gfx94x
```

## Shared flags

All commands accept:

- `--token <ghx_...>` — GitHub token override
- `--config-ref <branch>` — config branch (needed when policy files aren't merged)
