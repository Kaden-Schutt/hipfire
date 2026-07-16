# Copyright (c) Kaden Schutt
import json
from pathlib import Path
import subprocess

import pytest

from autoresearch.ar.review.github import (
    GitHubBoundaryError,
    GitHubClient,
    PreflightError,
    preflight_read_only,
)
from autoresearch.ar.review.config import load_operator_credential_manifest, load_review_configuration


ROOT = Path(__file__).parents[3]
REPO = "owner/repo"


def result(payload, *, headers=None, returncode=0, stderr=""):
    headers = headers or {"X-OAuth-Scopes": "read:user, repo:status"}
    header_text = "HTTP/2 200\r\n" + "".join(f"{key}: {value}\r\n" for key, value in headers.items()) + "\r\n"
    return subprocess.CompletedProcess(["gh"], returncode, header_text + json.dumps(payload), stderr)


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv, input_data=None):
        self.calls.append((list(argv), input_data))
        response = self.responses.pop(0)
        return response() if callable(response) else response


def user(login="review-bot", principal_type="Bot"):
    return {"id": 7, "node_id": "U_7", "login": login, "type": principal_type}


def repository():
    return {"id": 8, "node_id": "R_8", "full_name": REPO, "private": True}


def pull(number=42):
    return {
        "id": 9,
        "node_id": "PR_9",
        "number": number,
        "head": {"repo": {"full_name": REPO}, "sha": "head-sha"},
        "base": {"ref": "main", "sha": "base-sha"},
        "merge_commit_sha": "merge-sha",
    }


def record(node_id="IC_1", *, updated_at="2026-01-01T00:00:00Z"):
    return {
        "id": 11,
        "node_id": node_id,
        "user": {"login": "review-bot", "type": "Bot"},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": updated_at,
        "body": "{\"schema\": \"agentic-review/v1\"}",
    }


def test_path_and_method_allowlist_rejects_before_subprocess():
    runner = FakeRunner([])
    client = GitHubClient(runner)

    with pytest.raises(GitHubBoundaryError):
        client._request("GET", "/repos/owner/repo/hooks")
    with pytest.raises(GitHubBoundaryError):
        client._request("PATCH", "/user")
    with pytest.raises(GitHubBoundaryError):
        client.get_tree(REPO, "tree?recursive=1")
    assert runner.calls == []


@pytest.mark.parametrize(
    "response, message",
    [
        (result({}, returncode=1, stderr="boom"), "exit"),
        (subprocess.CompletedProcess(["gh"], 0, "not json", ""), "JSON"),
        (subprocess.CompletedProcess(["gh"], 0, "HTTP/2 200\r\n\r\n{}", ""), "scope"),
        (subprocess.CompletedProcess(["gh"], 0, "HTTP/2 401\r\nX-OAuth-Scopes: repo\r\n\r\n{}", ""), "401"),
        (subprocess.CompletedProcess(["gh"], 0, "HTTP/2 403\r\nX-OAuth-Scopes: read:user\r\n\r\n{}", ""), "403"),
        (subprocess.CompletedProcess(["gh"], 0, "HTTP/2 404\r\nX-OAuth-Scopes: read:user\r\n\r\n{}", ""), "404"),
    ],
)
def test_runner_failures_and_headers_fail_closed(response, message):
    with pytest.raises(GitHubBoundaryError, match=message):
        GitHubClient(FakeRunner([response])).get_authenticated_user()


def test_paginated_pull_requests_are_flattened_and_bounded():
    runner = FakeRunner([result([pull(1)]), result([pull(2)])])
    client = GitHubClient(runner)

    # gh --paginate is represented by one response per invocation in this fake.
    # The second page is consumed by the client's pagination callback.
    pulls = client.list_pull_requests(REPO, pages=2)
    assert [item["number"] for item in pulls.data] == [1, 2]
    assert all("--paginate" in call[0] for call in runner.calls)
    assert all("per_page=1" in " ".join(call[0]) for call in runner.calls)


def test_paginated_http_output_has_a_fixed_page_bound():
    pages = []
    for page in range(17):
        pages.append("HTTP/2 200\r\nX-OAuth-Scopes: read:user\r\n\r\n[]")
    response = subprocess.CompletedProcess(["gh"], 0, "\r\n".join(pages), "")

    with pytest.raises(GitHubBoundaryError, match="bound|page"):
        GitHubClient(FakeRunner([response]))._request(
            "GET", f"/repos/{REPO}/pulls", query={"per_page": 1}, paginate=True
        )


def test_envelope_uses_server_fields_and_rejects_edited_records():
    runner = FakeRunner([result([record()])])
    client = GitHubClient(runner)
    envelope = client.comment_envelope(REPO, 42, {"record_id": "logical", "record_type": "intent"})
    assert envelope.node_id == "IC_1"
    assert envelope.author == "review-bot"
    assert envelope.author_type == "Bot"
    assert envelope.created_at == envelope.updated_at
    assert envelope.payload["record_id"] == "logical"

    edited = FakeRunner([result([record(updated_at="2026-01-01T00:01:00Z")])])
    with pytest.raises(GitHubBoundaryError, match="edited"):
        GitHubClient(edited).comment_envelope(REPO, 42, {"record_id": "logical"})


def test_envelope_builder_requires_the_api_record_shape():
    incomplete = dict(record())
    del incomplete["id"]
    with pytest.raises(GitHubBoundaryError, match="comment"):
        GitHubClient(FakeRunner([])).envelope_from_comment(incomplete, {"record_id": "logical"})


def test_review_envelope_is_constructed_from_authenticated_review():
    review = dict(record("PRR_1"), id=7, state="APPROVED", commit_id="head-sha")
    envelope = GitHubClient(FakeRunner([result([review])])).review_envelope(
        REPO, 42, 7, {"record_id": "logical", "record_type": "review-metadata"}
    )
    assert envelope.node_id == "PRR_1"
    assert envelope.author_type == "Bot"


def test_effective_permission_is_normalized():
    response = result({"user": user(), "permissions": {"pull": True, "push": False, "admin": False}})
    permission = GitHubClient(FakeRunner([response])).collaborator_effective_permission(REPO, "review-bot")
    assert permission.login == "review-bot"
    assert permission.principal_type == "Bot"
    assert permission.permission == "pull"


def test_create_review_sends_exact_commit_id():
    runner = FakeRunner([result({"id": 17, "node_id": "PRR_17"})])
    GitHubClient(runner).create_pull_request_review(
        REPO, 42, body="review", event="COMMENT", commit_id="exact-head-sha"
    )
    argv, _ = runner.calls[0]
    assert "commit_id=exact-head-sha" in argv
    assert "event=COMMENT" in argv


def test_config_loader_rejects_absolute_and_traversal_overrides(tmp_path):
    for override in ("/etc/providers.json", "../providers.json", ".github/agentic-review/../../providers.json"):
        with pytest.raises(ValueError, match="path|root|travers"):
            load_review_configuration(tmp_path, providers_path=override)


def test_operator_manifest_loader_is_repository_root_relative(tmp_path):
    manifest = {
        "schema": "hipfire.agentic-review.operator-credentials",
        "version": 1,
        "principal": {"login": "review-bot", "type": "Bot"},
        "allowed_operations": ["publish"],
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    path = tmp_path / "custom-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_operator_credential_manifest(tmp_path, manifest_path="custom-manifest.json") == manifest
    for override in (str(path), "../custom-manifest.json"):
        with pytest.raises(ValueError, match="path|root|travers"):
            load_operator_credential_manifest(tmp_path, manifest_path=override)


def test_config_loader_uses_task_one_validators():
    configuration = load_review_configuration(ROOT)
    assert configuration.capabilities["schema"] == "hipfire.agentic-review.capabilities"
    assert configuration.providers["providers"] == []


def preflight_responses(*, scopes="read:user, repo:status", trust_ok=True):
    headers = {"X-OAuth-Scopes": scopes}
    return [result(user(), headers=headers), result(repository(), headers=headers), result([pull()], headers=headers)]


def test_preflight_probes_only_read_endpoints_with_bounded_pages_and_explicit_principal():
    runner = FakeRunner(preflight_responses())
    configuration = load_review_configuration(ROOT)
    # The repository fixture has no trusted apps, so provide a minimal valid
    # configuration copy for the preflight's trust check.
    configuration = configuration.with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    outcome = preflight_read_only(GitHubClient(runner), REPO, mode="discovery", configuration=configuration)
    assert outcome.principal_type == "Bot"
    assert len(runner.calls) == 3
    assert "--method" in runner.calls[0][0]
    assert "per_page=1" in " ".join(runner.calls[2][0])
    assert all(call[0][1] == "api" for call in runner.calls)


def test_preflight_rejects_classic_repo_scope_and_empty_trust():
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    with pytest.raises(PreflightError, match="classic|scope"):
        preflight_read_only(
            GitHubClient(FakeRunner(preflight_responses(scopes="repo, read:user"))),
            REPO,
            mode="discovery",
            configuration=configuration,
        )


def test_preflight_rejects_malformed_scope_header():
    configuration = load_review_configuration(ROOT)
    with pytest.raises(PreflightError, match="scope"):
        preflight_read_only(
            GitHubClient(FakeRunner(preflight_responses(scopes="read:user,,repo:status"))),
            REPO,
            mode="discovery",
            configuration=configuration,
        )


def test_read_only_preflight_accepts_task_one_empty_apps():
    configuration = load_review_configuration(ROOT)
    outcome = preflight_read_only(
        GitHubClient(FakeRunner(preflight_responses())),
        REPO,
        mode="discovery",
        configuration=configuration,
    )
    assert outcome.login == "review-bot"


def test_controller_preflight_uses_effective_permission_without_static_apps():
    configuration = load_review_configuration(ROOT)
    runner = FakeRunner(preflight_responses() + [
        result({"user": user(), "permissions": {"pull": True, "push": False}})
    ])
    outcome = preflight_read_only(
        GitHubClient(runner),
        REPO,
        mode="controller",
        configuration=configuration,
    )
    assert outcome.login == "review-bot"
    assert len(runner.calls) == 4


def test_publisher_preflight_requires_matching_app_and_operator_manifest():
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "different-app", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    manifest = {
        "schema": "hipfire.agentic-review.operator-credentials",
        "version": 1,
        "principal": {"login": "review-bot", "type": "Bot"},
        "allowed_operations": ["publish"],
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    with pytest.raises(PreflightError, match="matching|App"):
        preflight_read_only(
            GitHubClient(FakeRunner(preflight_responses() + [result({"user": user(), "permissions": {"push": True}})])),
            REPO,
            mode="publisher",
            configuration=configuration,
            operator_manifest=manifest,
        )


def test_publisher_preflight_accepts_matching_app_and_operator_manifest():
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    manifest = {
        "schema": "hipfire.agentic-review.operator-credentials",
        "version": 1,
        "principal": {"login": "review-bot", "type": "Bot"},
        "allowed_operations": ["publish"],
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    runner = FakeRunner(preflight_responses() + [
        result({"user": user(), "permissions": {"push": True}})
    ])
    preflight_read_only(
        GitHubClient(runner), REPO, mode="publisher", configuration=configuration, operator_manifest=manifest
    )
    assert all("--method" in call[0] and call[0][call[0].index("--method") + 1] == "GET" for call in runner.calls)


@pytest.mark.parametrize("bad_user", [{"id": 1, "login": "bot"}, {"id": 1, "login": "bot", "type": "Robot"}])
def test_preflight_rejects_missing_or_unsupported_principal_type(bad_user):
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    with pytest.raises(PreflightError, match="principal|type"):
        preflight_read_only(
            GitHubClient(FakeRunner([result(bad_user)])), REPO, mode="discovery", configuration=configuration
        )


def test_preflight_rejects_incomplete_page_and_bad_repository():
    configuration = load_review_configuration(ROOT)
    configuration = configuration.with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    with pytest.raises(PreflightError, match="page|pull"):
        preflight_read_only(
            GitHubClient(FakeRunner([result(user()), result(repository()), result({})])),
            REPO, mode="discovery", configuration=configuration,
        )
