#!/usr/bin/env python3
"""CLI entry points for the agentic review workflow — full lifecycle.

An agent workflow looks like:

  1. review preflight --mode discovery --repository OWNER/REPO
  2. review discover --repository OWNER/REPO --operator creds.json
  3. review review --pr 123 --repository OWNER/REPO --operator creds.json

Step 3 does build-capsule → infer → publish in one shot.  The LLM provider
API key must be set in the REVIEW_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .capsule import ReviewCapsule, build_review_capsule
from .config import (
    _SOURCE_PROOF,
    AuthenticatedConfigSource,
    configuration_source_digest,
    load_operator_credential_manifest,
    load_review_configuration,
)
from .discovery import discover_pull_requests
from .github import GitHubClient, preflight_read_only
from .inference import BoundedHttpTransport, ToollessReviewAdapter
from .models import ReviewProposal, ReviewTarget
from .publisher import PublishResult, publish_review, render_report


def _root() -> Path:
    root = os.environ.get("GITHUB_WORKSPACE") or os.environ.get("REVIEW_REPO_ROOT")
    if root:
        return Path(root)
    candidate = Path(__file__).resolve().parent
    for _ in range(10):
        if (candidate / ".git").exists():
            return candidate
        candidate = candidate.parent
    return Path.cwd()


def _operator_manifest(path: str, root: Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_absolute():
        with manifest_path.open(encoding="utf-8") as stream:
            return json.load(stream)
    return load_operator_credential_manifest(root, manifest_path=path)


def _config(client: GitHubClient, repository: str, root: Path, config_ref: str | None = None):
    """Load review configuration, optionally from a non-default branch.

    The AuthenticatedConfigSource always binds to the default-branch Git SHA
    (required by the authentication boundary), but the policy bytes themselves
    are read from *local disk* in the checked-out working tree.  For production
    use against the default branch this matches; for ``config_ref`` (dev use
    before policy files are merged), the caller must have the feature branch
    checked out locally so the local files match the intended policy.
    """
    repo_data = client.get_repository(repository).data
    default_branch = repo_data.get("default_branch")
    default_sha = client.get_branch_head(repository, default_branch)
    from .config import _PROVIDERS, _CAPABILITIES, _TRUSTED
    provider_bytes = (root / _PROVIDERS).read_bytes()
    capabilities_bytes = (root / _CAPABILITIES).read_bytes()
    trusted_bytes = (root / _TRUSTED).read_bytes()
    config_digest = configuration_source_digest(provider_bytes, capabilities_bytes, trusted_bytes)
    source = AuthenticatedConfigSource._from_authenticated_boundary(
        _SOURCE_PROOF, repository, default_branch, default_sha, config_digest, str(root),
    )
    return load_review_configuration(root, source=source)


def _github_client(token: str | None = None) -> GitHubClient:
    # GitHubClient reads GH_TOKEN from the environment by default.
    # A --token flag overrides (injects via env before import, or the
    # client finds it; for simplicity we rely on the default gh auth).
    return GitHubClient()


def cmd_discover(args: argparse.Namespace) -> None:
    root = _root()
    repo = args.repository or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("error: --repository or GITHUB_REPOSITORY required", file=sys.stderr)
        raise SystemExit(2)
    client = _github_client(args.token)
    c = _config(client, repo, root, config_ref=args.config_ref)
    operator = _operator_manifest(args.operator, root)
    summary = discover_pull_requests(client, repo, configuration=c, operator_credential=operator)
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
    client = _github_client(args.token)
    c = _config(client, repo, root, config_ref=args.config_ref)
    operator = _operator_manifest(args.operator, root) if args.operator else None
    result = preflight_read_only(client, repo, mode=args.mode, configuration=c, operator_manifest=operator)
    print(json.dumps({
        "login": result.login,
        "principal_type": result.principal_type,
        "repository": result.repository,
        "scopes": list(result.scopes),
    }))


def cmd_inspect(args: argparse.Namespace) -> None:
    """Build a review capsule from a PR (and optionally run inference)."""
    root = _root()
    repo = args.repository or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("error: --repository or GITHUB_REPOSITORY required", file=sys.stderr)
        raise SystemExit(2)
    client = _github_client(args.token)

    from dataclasses import replace
    target = client.get_review_target(repo, args.pr)
    target = replace(target, head_repository=repo)  # resolve blobs via base repo
    capsule = build_review_capsule(client, target)

    if not capsule.complete:
        print(json.dumps({"status": "incomplete-capsule",
                          "files": len(capsule.manifest), "reason": "blob fetch incomplete"}))
        raise SystemExit(1)

    # Optionally write capsule to file
    if args.capsule:
        Path(args.capsule).write_text(capsule.canonical_json().decode("utf-8"))

    output = {
        "status": "capsule-ready",
        "target": {"repository": target.repository, "number": target.number,
                    "head_sha": target.head_sha, "base_sha": target.base_sha},
        "capsule_digest": capsule.digest,
        "files": len(capsule.manifest),
    }

    # Optionally run inference
    if args.provider:
        api_key = os.environ.get("REVIEW_API_KEY")
        if not api_key:
            print("error: REVIEW_API_KEY required for inference", file=sys.stderr)
            raise SystemExit(2)
        c = _config(client, repo, root, config_ref=args.config_ref)
        transport = BoundedHttpTransport()
        adapter = ToollessReviewAdapter.from_configuration(
            c, args.provider, transport, {"REVIEW_API_KEY": api_key}, github_client=client,
        )
        proposal = adapter.review(capsule)
        if args.proposal:
            from .canonical import canonical_json
            Path(args.proposal).write_text(canonical_json(proposal.to_mapping()).decode("utf-8"))
        output["status"] = "inferred"
        output["verdict"] = proposal.verdict
        output["findings_count"] = len(proposal.findings)
        if proposal.scope:
            output["scope"] = {"model_architectures": list(proposal.scope.model_architectures),
                               "hardware_architectures": list(proposal.scope.hardware_architectures)}
        if proposal.hardware_validation_triage:
            t = proposal.hardware_validation_triage
            output["hardware_validation_triage"] = {
                "impacted_model_families": list(t.impacted_model_families),
                "impacted_hardware": list(t.impacted_hardware),
                "coverage_decision": t.coverage_decision,
                "rationale": t.rationale,
            }
            if t.coverage_decision != "none":
                output["verify_labels"] = ["verify-" + arch for arch in t.impacted_hardware]
        # Always print the rendered report for human reading
        print(render_report(proposal))
        print("---")
    else:
        print("--- capsule built (no inference, pass --provider to infer) ---")

    print(json.dumps(output, indent=2))


def cmd_review(args: argparse.Namespace) -> None:
    """One-shot: build capsule → run inference → publish on a PR."""
    api_key = os.environ.get("REVIEW_API_KEY")
    if not api_key:
        print("error: REVIEW_API_KEY environment variable required", file=sys.stderr)
        raise SystemExit(2)
    root = _root()
    repo = args.repository or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("error: --repository or GITHUB_REPOSITORY required", file=sys.stderr)
        raise SystemExit(2)
    client = _github_client(args.token)
    c = _config(client, repo, root, config_ref=args.config_ref)
    from dataclasses import replace
    target = client.get_review_target(repo, args.pr)
    target = replace(target, head_repository=repo)
    capsule = build_review_capsule(client, target)
    if not capsule.complete:
        print(json.dumps({"status": "incomplete-capsule", "files": len(capsule.manifest)}))
        raise SystemExit(1)
    transport = BoundedHttpTransport()
    adapter = ToollessReviewAdapter.from_configuration(
        c, args.provider, transport, {"REVIEW_API_KEY": api_key}, github_client=client,
    )
    proposal = adapter.review(capsule)
    operator = _operator_manifest(args.operator, root)
    result = publish_review(client, proposal, target, configuration=c, operator_credential=operator)
    output = {"status": result.status, "attempt_id": result.attempt_id, "verdict": proposal.verdict}
    if result.reason:
        output["reason"] = result.reason
    if proposal.hardware_validation_triage:
        t = proposal.hardware_validation_triage
        output["hardware_validation_triage"] = {
            "impacted_model_families": list(t.impacted_model_families),
            "impacted_hardware": list(t.impacted_hardware),
            "coverage_decision": t.coverage_decision,
            "rationale": t.rationale,
        }
        if t.coverage_decision != "none":
            output["verify_labels"] = ["verify-" + arch for arch in t.impacted_hardware]
    print(json.dumps(output, indent=2))
    if result.status not in ("complete", "duplicate"):
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="review", description="Agentic PR review workflow for hipfire")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(p):
        p.add_argument("--repository", help="owner/repo (default: $GITHUB_REPOSITORY)")
        p.add_argument("--token", help="GitHub token (default: gh auth token)")
        p.add_argument("--config-ref", help="Branch for config policy files (default: default branch; "
                       "needed when policy files haven't been merged yet)")

    # preflight
    p = sub.add_parser("preflight", help="Validate credentials, configuration, and API access")
    p.add_argument("--mode", required=True, choices=["discovery", "controller", "publisher"])
    p.add_argument("--operator")
    add_shared(p)

    # discover
    p = sub.add_parser("discover", help="Scan open PRs and reconcile needs-review labels")
    p.add_argument("--operator", required=True, help="Path to operator credential manifest JSON")
    add_shared(p)

    # inspect — build capsule (and optionally run inference)
    p = sub.add_parser("inspect", help="Build a capsule from a PR (and optionally run inference)")
    p.add_argument("--pr", type=int, required=True, help="PR number")
    p.add_argument("--provider", help="Provider ID from config (default: none; set to run inference)")
    p.add_argument("--capsule", help="Write capsule JSON to this file")
    p.add_argument("--proposal", help="Write proposal JSON to this file")
    add_shared(p)

    # review — full one-shot
    p = sub.add_parser("review", help="Full one-shot: build capsule → infer → publish on a PR")
    p.add_argument("--pr", type=int, required=True, help="PR number")
    p.add_argument("--operator", required=True, help="Path to operator credential manifest JSON")
    p.add_argument("--provider", default="review-adapter",
                   help="Provider ID from config (default: review-adapter)")
    add_shared(p)

    ns = parser.parse_args(argv)
    try:
        if ns.command == "discover":
            cmd_discover(ns)
        elif ns.command == "preflight":
            cmd_preflight(ns)
        elif ns.command == "inspect":
            cmd_inspect(ns)
        elif ns.command == "review":
            cmd_review(ns)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
