# Copyright (c) Kaden Schutt
"""Focused tests for CLI configuration provenance."""

from __future__ import annotations

from pathlib import Path

from autoresearch.ar.review import cli
from autoresearch.ar.review.config import (
    AuthenticatedConfigSource,
    _SOURCE_PROOF,
    configuration_source_digest,
)


ROOT = Path(__file__).parents[3]
REPO = "owner/repo"
CONFIG_PATHS = (
    ".github/agentic-review/providers.json",
    ".github/agentic-review/capabilities-v1.json",
    ".github/agentic-review/trusted-publishers.json",
)


class ConfigClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def get_repository(self, repository: str):
        self.calls.append(("repository", repository))
        return type("Response", (), {"data": {"default_branch": "main"}})()

    def get_branch_head(self, repository: str, branch: str) -> str:
        self.calls.append(("branch", (repository, branch)))
        return "c" * 40

    def authenticated_config_source(self, repository: str, *, commit_sha: str, repository_root: str):
        self.calls.append(("authenticated_source", (repository, commit_sha, repository_root)))
        contents = tuple((Path(repository_root) / path).read_bytes() for path in CONFIG_PATHS)
        return AuthenticatedConfigSource._from_authenticated_boundary(
            _SOURCE_PROOF,
            repository,
            "main",
            commit_sha,
            configuration_source_digest(*contents),
            repository_root,
        )


def test_cli_loads_repository_config_through_authenticated_source():
    client = ConfigClient()

    configuration = cli._authenticated_configuration(client, REPO, ROOT)

    assert configuration.is_protected
    assert configuration.source is not None
    assert configuration.source.repository == REPO
    assert [name for name, _ in client.calls] == [
        "repository", "branch", "authenticated_source",
    ]
