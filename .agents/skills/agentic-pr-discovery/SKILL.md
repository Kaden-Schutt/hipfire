# Agentic PR discovery

This skill is **manual-only**. It scans open pull requests, including
drafts and pull requests from forks, and reconciles the repository-owned
`needs-review` label.

The operator must provide a write-permission operator credential manifest.
The repository is read from `GITHUB_REPOSITORY` unless `--repository` is
provided. Protected configuration is loaded from `.github/agentic-review`.

Discovery does not check out branches and does not run tests. It uses the
bounded GitHub client through:

```text
python3 -m autoresearch.ar.review.cli discover --operator FILE [--repository OWNER/REPO]
```

The command prints JSON containing the `DiscoverySummary` fields
`reviewed`, `needs_review`, `labelled`, `clean`, `incomplete`, `errors`, and
`complete`. Each item contains a pull request number and reason. It exits
with status 1 when the scan is incomplete.

Use `preflight.sh` before discovery to validate the credential, protected
configuration, and read/write API boundary without selecting a model or
provider.
