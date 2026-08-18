---
name: agentic-pr-static-review
description: Run bounded toolless inference on a review capsule and produce a review proposal with hardware validation triage. Use when a review capsule has been built and needs inference, or to run the full inspect pipeline (build → infer) on a PR. Outputs a structured ReviewProposal with triage data for downstream agent consumption.
---

# Agentic PR static review

This skill is **manual-only** and operates as a read-only controller: it
does not mutate GitHub. It reads a bounded capsule JSON and writes or
reports a structured proposal JSON. It uses toolless inference only; no
provider may receive tools or execute repository commands.

The controller must not run `git checkout`; test execution is out of scope.
It does not inspect arbitrary branches or invoke a shell-backed coding agent.

## Provider configuration

Credentials and endpoints are configured in `.github/agentic-review/providers.json`.
For local per-developer overrides (not checked in), create
`.github/agentic-review/providers.local.json` with the same schema — it merges
into the checked-in provider list (same `id` replaces, new `id` appends).
See `.agents/skills/agentic-pr-review/README.md` for examples.

The `api_key_env` field names the environment variable to read the API key from.
For DeepSeek this is typically `DEEPSEEK_TOKEN`; set it with:
`export DEEPSEEK_TOKEN="sk-..."`.  The examples below use `REVIEW_API_KEY` as a
generic var — substitute your provider's actual env var name.

## Commands

### Build a capsule from a PR (no inference, no provider key needed):

```text
python3 -m autoresearch.ar.review.cli inspect --pr 123 --repository OWNER/REPO --capsule capsule.json
```

### Build capsule + run inference + save proposal:

```text
export REVIEW_API_KEY="sk-..."
python3 -m autoresearch.ar.review.cli inspect --pr 123 --repository OWNER/REPO \
  --capsule capsule.json --proposal proposal.json --provider review-adapter
```

### Full one-shot review (build + infer + publish):

```text
export REVIEW_API_KEY="sk-..."
python3 -m autoresearch.ar.review.cli review --pr 123 --repository OWNER/REPO \
  --operator .github/agentic-review/operator-credentials.json --provider review-adapter
```

Use `preflight.sh` in `controller` mode to validate protected configuration,
read-only API access, and capsule source access before inspection. The
controller and publisher are separate: only a publisher with the required
write-permission operator credential may perform GitHub mutations.

The `--config-ref <branch>` flag points config authentication at a non-default
branch (needed when policy files haven't been merged yet). All commands accept
`--token <ghx_...>` to override the GitHub token.
