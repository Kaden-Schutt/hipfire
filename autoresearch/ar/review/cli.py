#!/usr/bin/env python3
"""CLI entry points for the agentic review workflow — discovery and inspection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .capsule import ReviewCapsule, build_review_capsule
from .config import (
    load_operator_credential_manifest,
    load_review_configuration,
)
from .discovery import discover_pull_requests
from .github import GitHubClient, preflight_read_only
from .models import ReviewProposal, ReviewTarget
from .publisher import PublishResult, publish_review


def _root() -> Path:
    root = os.environ.get("GITHUB_WORKSPACE") or os.environ.get("REVIEW_REPO_ROOT")
    if root:
        return Path(root)
    # Walk up from this file to find git root.
    candidate = Path(__file__).resolve().parent
    for _ in range(10):
        if (candidate / ".git").exists():
            return candidate
        candidate = candidate.parent
    return Path.cwd()


def _operator_manifest(path: str, root: Path) -> dict[str, Any]:
    """Load a repository-relative manifest or an explicitly supplied file."""
    manifest_path = Path(path)
    if manifest_path.is_absolute():
        with manifest_path.open(encoding="utf-8") as stream:
            return json.load(stream)
    return load_operator_credential_manifest(root, manifest_path=path)


def _authenticated_configuration(client: Any, repository: str, root: Path):
    repository_data = client.get_repository(repository).data
    default_branch = repository_data.get("default_branch")
    commit_sha = client.get_branch_head(repository, default_branch)
    source = client.authenticated_config_source(
        repository, commit_sha=commit_sha, repository_root=str(root)
    )
    return load_review_configuration(root, source=source)


def cmd_discover(args: argparse.Namespace) -> None:
    root = _root()
    repo = args.repository or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("error: --repository or GITHUB_REPOSITORY required", file=sys.stderr)
        raise SystemExit(2)
    client = GitHubClient()
    config = _authenticated_configuration(client, repo, root)
    operator = _operator_manifest(args.operator, root)
    summary = discover_pull_requests(
        client, repo, configuration=config, operator_credential=operator
    )
    print(json.dumps({
        "reviewed": [{"number": item.number, "reason": item.reason} for item in summary.reviewed],
        "needs_review": [{"number": item.number, "reason": item.reason} for item in summary.needs_review],
        "labelled": [{"number": item.number, "reason": item.reason} for item in summary.labelled],
        "clean": [{"number": item.number, "reason": item.reason} for item in summary.clean],
        "incomplete": [{"number": item.number, "reason": item.reason} for item in summary.incomplete],
        "errors": [{"number": item.number, "reason": item.reason} for item in summary.errors],
        "complete": summary.complete,
    }, indent=2))
    if not summary.complete:
        raise SystemExit(1)


def cmd_preflight(args: argparse.Namespace) -> None:
    root = _root()
    repo = args.repository or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("error: --repository or GITHUB_REPOSITORY required", file=sys.stderr)
        raise SystemExit(2)
    client = GitHubClient()
    config = _authenticated_configuration(client, repo, root)
    operator = _operator_manifest(args.operator, root) if args.operator else None
    result = preflight_read_only(
        client,
        repo,
        mode=args.mode,
        configuration=config,
        operator_manifest=operator,
    )
    print(json.dumps({
        "login": result.login,
        "principal_type": result.principal_type,
        "repository": result.repository,
        "scopes": list(result.scopes),
    }))


def cmd_inspect(args: argparse.Namespace) -> None:
    # Inference remains deliberately outside this thin wrapper.  The capsule is
    # parsed here to fail early, while a controller/provider integration can
    # consume the structured input without granting the wrapper extra powers.
    with open(args.capsule, encoding="utf-8") as fh:
        capsule_data = json.load(fh)
    print(json.dumps({
        "status": "not-implemented",
        "capsule_file": args.capsule,
        "proposal_file": args.proposal,
        "capsule_loaded": isinstance(capsule_data, dict),
    }))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="review")
    sub = parser.add_subparsers(dest="command", required=True)

    disc = sub.add_parser("discover", help="Scan open PRs and reconcile needs-review labels")
    disc.add_argument("--repository", help="owner/repo (default: $GITHUB_REPOSITORY)")
    disc.add_argument("--operator", required=True, help="Path to operator credential manifest JSON")

    pre = sub.add_parser("preflight", help="Validate credentials, configuration, and API access")
    pre.add_argument("--mode", required=True, choices=["discovery", "controller", "publisher"])
    pre.add_argument("--repository")
    pre.add_argument("--operator")

    insp = sub.add_parser("inspect", help="Run toolless inference on a review capsule")
    insp.add_argument("--capsule", required=True, help="Path to review capsule JSON file")
    insp.add_argument("--proposal", required=True, help="Path to write the review proposal JSON")

    ns = parser.parse_args(argv)
    if ns.command == "discover":
        cmd_discover(ns)
    elif ns.command == "preflight":
        cmd_preflight(ns)
    elif ns.command == "inspect":
        cmd_inspect(ns)


if __name__ == "__main__":
    main()
