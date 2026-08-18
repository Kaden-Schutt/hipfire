# Copyright (c) Kaden Schutt
import json
import hashlib
from pathlib import Path
import subprocess
import base64
import sys
import time

import pytest

import autoresearch.ar.review.github as github
from autoresearch.ar.review.canonical import canonical_digest
from autoresearch.ar.review.github import (
    decode_protocol_body,
    encode_protocol_body,
    GitHubBoundaryError,
    GitHubClient,
    PreflightError,
    _subprocess_runner,
    preflight_read_only,
)
from autoresearch.ar.review.config import (
    configuration_source_digest,
    load_operator_credential_manifest,
    load_review_configuration,
    validate_operator_credential_manifest,
)
from autoresearch.ar.review.models import ReviewTarget


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


def human_user(login="reviewer"):
    return user(login=login, principal_type="User")


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


def body_payload(record_id="logical"):
    target = ReviewTarget(REPO, 42, REPO, "head-sha", "main", "base-sha", "merge-sha")
    payload = {
        "schema": "agentic-review/v1",
        "record_type": "intent",
        "record_id": record_id,
        "target": {
            "repository": REPO,
            "number": 42,
            "head_repository": REPO,
            "head_sha": "head-sha",
            "base_ref": "main",
            "base_sha": "base-sha",
            "merge_base_sha": "merge-sha",
        },
        "target_key": target.target_key(),
        "attempt_id": "attempt-1",
    }
    payload["canonical_digest"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "canonical_digest"}
    )
    return payload


def record(node_id="IC_1", *, updated_at="2026-01-01T00:00:00Z", author_login="review-bot", author_type="Bot"):
    payload = body_payload()
    return {
        "id": 11,
        "node_id": node_id,
        "user": {"login": author_login, "type": author_type},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": updated_at,
        "body": json.dumps(payload, separators=(",", ":")),
    }


def review_record(*, submitted_at="2026-01-01T00:00:00Z", state="APPROVED"):
    review = dict(record("PRR_1"), id=7, state=state, commit_id="head-sha")
    review.pop("created_at")
    review.pop("updated_at")
    review["submitted_at"] = submitted_at
    return review


def permission(login="review-bot", role="pull", principal_type="Bot"):
    return {
        "user": {**user(login=login, principal_type=principal_type), "permissions": {}},
        "permission": role,
        "role_name": role,
    }


def installation_repositories(repositories=None, *, total_count=1):
    return {"total_count": total_count, "repositories": [repository()] if repositories is None else repositories}


def app_manifest(operation="publish", *, login="review-bot", digest=None):
    return {
        "schema": "hipfire.agentic-review.operator-credentials",
        "version": 1,
        "repository": REPO,
        "principal": {"login": login, "type": "Bot"},
        "allowed_operations": [operation],
        "write_permissions": {"issues": "write", "pull_requests": "write"},
        "credential_attestation_digest": digest or "sha256:" + "a" * 64,
    }


def discovery_manifest():
    return {
        "schema": "hipfire.agentic-review.operator-credentials",
        "version": 1,
        "repository": REPO,
        "principal": {"login": "review-bot", "type": "User"},
        "allowed_operations": ["discover"],
        "write_permissions": {"issues": "write", "pull_requests": "write"},
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }


def tree():
    return {"sha": "base-sha", "tree": [{"path": "README.md", "mode": "100644", "type": "blob", "sha": "blob-sha"}]}


def blob():
    return {"sha": "blob-sha", "encoding": "base64", "content": "cmVhZG1lCg=="}


def test_path_and_method_allowlist_rejects_before_subprocess():
    runner = FakeRunner([])
    client = GitHubClient(runner)

    assert not hasattr(client, "request")
    with pytest.raises(GitHubBoundaryError):
        client._request("GET", "/repos/owner/repo/hooks")
    with pytest.raises(GitHubBoundaryError):
        client._request("PATCH", "/user")
    with pytest.raises(GitHubBoundaryError):
        client.get_tree(REPO, "tree?recursive=1")
    assert runner.calls == []


@pytest.mark.parametrize(
    "call",
    [
        lambda client: client.get_repository("owner/../repo"),
        lambda client: client.get_repository("owner/repo?bad"),
        lambda client: client.get_tree(REPO, "../tree"),
        lambda client: client.get_blob(REPO, "blob?bad"),
        lambda client: client.collaborator_effective_permission(REPO, "bad/login"),
        lambda client: client.remove_label(REPO, 42, "../label"),
    ],
)
def test_unsafe_endpoint_identifiers_are_rejected_before_subprocess(call):
    runner = FakeRunner([])
    with pytest.raises(GitHubBoundaryError):
        call(GitHubClient(runner))
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


def test_runner_timeout_and_output_bounds_fail_closed():
    class TimeoutRunner:
        def __call__(self, argv, input_data=None):
            raise subprocess.TimeoutExpired(argv, 30)

    with pytest.raises(GitHubBoundaryError, match="timed out"):
        GitHubClient(TimeoutRunner()).get_authenticated_user()
    huge = subprocess.CompletedProcess(["gh"], 0, "x" * (16 * 1024 * 1024 + 1), "")
    with pytest.raises(GitHubBoundaryError, match="stdout|size"):
        GitHubClient(FakeRunner([huge])).get_authenticated_user()


def test_subprocess_runner_stops_streaming_process_at_output_bound():
    producer = "import sys; sys.stdout.write('x' * (17 * 1024 * 1024)); sys.stdout.flush()"
    with pytest.raises(GitHubBoundaryError, match="stdout|size|bound"):
        _subprocess_runner([sys.executable, "-c", producer])


def test_subprocess_runner_terminates_child_when_streams_close_first(monkeypatch):
    monkeypatch.setattr(github, "_SUBPROCESS_TIMEOUT_SECONDS", 0.05)
    producer = "import sys, time; sys.stdout.close(); sys.stderr.close(); time.sleep(10)"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _subprocess_runner([sys.executable, "-c", producer])
    assert time.monotonic() - started < 2


@pytest.mark.parametrize(
    "runner_result",
    [
        ("0", "{}", ""),
        (0, 1, ""),
        (0, "{}", 1),
        (0, "{}", "x" * (1 << 20) + "x"),
    ],
)
def test_malformed_runner_results_fail_closed(runner_result):
    with pytest.raises(GitHubBoundaryError):
        GitHubClient(FakeRunner([runner_result])).get_authenticated_user()


def test_paginated_pull_requests_are_flattened_and_bounded():
    next_page = '<https://api.github.com/repos/owner/repo/pulls?page=2>; rel="next"'
    runner = FakeRunner([
        result([pull(1)], headers={"X-OAuth-Scopes": "read:user", "Link": next_page}),
        result([pull(2)]),
    ])
    client = GitHubClient(runner)

    pulls = client.list_pull_requests(REPO, max_pages=2)
    assert [item["number"] for item in pulls.data] == [1, 2]
    assert all("--paginate" not in call[0] for call in runner.calls)
    assert all("per_page=100" in " ".join(call[0]) for call in runner.calls)


def test_merge_base_compare_endpoint_is_allowlisted():
    runner = FakeRunner([
        result({
            "base_commit": {"sha": "base-sha"},
            "merge_base_commit": {"sha": "merge-sha"},
        }),
    ])

    assert GitHubClient(runner).get_merge_base_sha(REPO, "base-sha", "head-sha") == "merge-sha"
    assert runner.calls[0][0][-1] == "/repos/owner/repo/compare/base-sha...head-sha"


def test_issue_labels_follow_bounded_link_pagination():
    next_page = '<https://api.github.com/repos/owner/repo/issues/42/labels?page=2>; rel="next"'
    runner = FakeRunner([
        result([{"name": "other"}], headers={"X-OAuth-Scopes": "read:user", "Link": next_page}),
        result([{"name": "needs-review"}], headers={"X-OAuth-Scopes": "read:user"}),
    ])

    labels = GitHubClient(runner).list_issue_labels(REPO, 42)

    assert [item["name"] for item in labels.data] == ["other", "needs-review"]
    assert all("per_page=100" in " ".join(call[0]) for call in runner.calls)


def test_issue_labels_fail_closed_at_pagination_bound():
    next_page = '<https://api.github.com/repos/owner/repo/issues/42/labels?page=2>; rel="next"'
    runner = FakeRunner([
        result([], headers={"X-OAuth-Scopes": "read:user", "Link": next_page})
        for _ in range(16)
    ])

    with pytest.raises(GitHubBoundaryError, match="labels pagination"):
        GitHubClient(runner).list_issue_labels(REPO, 42)


def test_pagination_fails_closed_when_link_exceeds_configured_bound():
    next_page = '<https://api.github.com/repos/owner/repo/pulls?page=2>; rel="next"'
    with pytest.raises(GitHubBoundaryError, match="pagination|page|bound"):
        GitHubClient(FakeRunner([
            result([pull(1)], headers={"X-OAuth-Scopes": "read:user", "Link": next_page}),
        ])).list_pull_requests(REPO, max_pages=1)


def test_exhaustive_pull_listing_fails_with_explicit_incomplete_scan_at_page_cap():
    responses = []
    for page in range(1, 17):
        link = f'<https://api.github.com/repos/owner/repo/pulls?page={page + 1}>; rel="next"'
        responses.append(result([pull(page)], headers={"X-OAuth-Scopes": "read:user", "Link": link}))
    with pytest.raises(GitHubBoundaryError, match="incomplete|page|bound"):
        GitHubClient(FakeRunner(responses)).list_pull_requests(REPO, max_pages=16)


def test_paginated_http_output_has_a_fixed_page_bound():
    pages = []
    for page in range(17):
        pages.append("HTTP/2 200\r\nX-OAuth-Scopes: read:user\r\n\r\n[]")
    response = subprocess.CompletedProcess(["gh"], 0, "\r\n".join(pages), "")

    with pytest.raises(GitHubBoundaryError, match="bound|page"):
        GitHubClient(FakeRunner([response]))._request(
            "GET", f"/repos/{REPO}/pulls", query={"per_page": 1}, paginate=True
        )


def test_envelope_uses_exact_server_endpoint_and_rejects_edited_records():
    runner = FakeRunner([result(record())])
    client = GitHubClient(runner)
    envelope = client.comment_envelope(REPO, 11)
    assert envelope.node_id == "IC_1"
    assert envelope.author == "review-bot"
    assert envelope.author_type == "Bot"
    assert envelope.created_at == envelope.updated_at
    assert envelope.payload["record_id"] == "logical"
    assert runner.calls[0][0][-1] == "/repos/owner/repo/issues/comments/11"

    edited = FakeRunner([result(record(updated_at="2026-01-01T00:01:00Z"))])
    with pytest.raises(GitHubBoundaryError, match="edited"):
        GitHubClient(edited).comment_envelope(REPO, 11)


@pytest.mark.parametrize("method", ["comment_envelope", "review_envelope"])
def test_envelope_rejects_a_record_with_a_different_server_id(method):
    payload = record() if method == "comment_envelope" else review_record()
    runner = FakeRunner([result(dict(payload, id=99))])
    with pytest.raises(GitHubBoundaryError, match="ID|id"):
        if method == "comment_envelope":
            GitHubClient(runner).comment_envelope(REPO, 11)
        else:
            GitHubClient(runner).review_envelope(REPO, 42, 7)


def test_envelope_factories_are_not_public_record_mapping_apis():
    client = GitHubClient(FakeRunner([]))
    assert not hasattr(client, "envelope_from_comment")
    assert not hasattr(client, "envelope_from_review")


def test_envelope_acquisition_uses_server_author_for_later_app_bot_trust():
    runner = FakeRunner([result(record(author_login="repository-owner", author_type="User"))])
    envelope = GitHubClient(runner).comment_envelope(REPO, 11)
    assert envelope.author == "repository-owner"
    assert envelope.author_type == "User"


def test_api_shaped_app_record_uses_body_provenance_not_top_level_fields():
    payload = body_payload()
    payload.update({
        "app_id": 7,
        "installation_id": 8,
        "repository_id": 9,
        "credential_attestation_digest": "sha256:" + "a" * 64,
    })
    payload["canonical_digest"] = canonical_digest({key: value for key, value in payload.items() if key != "canonical_digest"})
    raw = record()
    raw["body"] = json.dumps(payload, separators=(",", ":"))
    raw.update(app_id=999, installation_id=999, repository_id=999)

    envelope = GitHubClient(FakeRunner([result(raw)])).comment_envelope(REPO, 11)

    assert envelope.payload["app_id"] == 7
    assert envelope.payload["installation_id"] == 8
    assert envelope.payload["repository_id"] == 9
    assert not hasattr(envelope, "app_id")


def test_review_envelope_is_constructed_from_authenticated_review():
    review = review_record()
    runner = FakeRunner([result(review)])
    envelope = GitHubClient(runner).review_envelope(
        REPO, 42, 7
    )
    assert envelope.node_id == "PRR_1"
    assert envelope.author_type == "Bot"
    assert envelope.created_at == "2026-01-01T00:00:00Z"
    assert envelope.updated_at == envelope.created_at
    assert "/pulls/42/reviews/7" in runner.calls[0][0][-1]


@pytest.mark.parametrize("submitted_at", [None, "not-a-timestamp"])
def test_review_envelope_rejects_missing_or_invalid_submitted_timestamp(submitted_at):
    review = review_record(submitted_at=submitted_at)
    with pytest.raises(GitHubBoundaryError, match="timestamp|submitted"):
        GitHubClient(FakeRunner([result(review)])).review_envelope(REPO, 42, 7)


def test_pull_review_listing_accepts_pending_review_without_submitted_timestamp():
    pending = review_record(state="PENDING")
    pending.pop("submitted_at")
    response = GitHubClient(FakeRunner([result([pending])])).list_pull_reviews(REPO, 42)
    assert response.data[0]["state"] == "PENDING"


def test_pull_review_listing_rejects_non_pending_review_without_submitted_timestamp():
    review = review_record(state="APPROVED")
    review.pop("submitted_at")
    with pytest.raises(GitHubBoundaryError, match="timestamp|submitted"):
        GitHubClient(FakeRunner([result([review])])).list_pull_reviews(REPO, 42)


def test_pending_pull_review_is_rejected_when_building_authenticated_envelope():
    pending = review_record(state="PENDING")
    pending.pop("submitted_at")
    with pytest.raises(GitHubBoundaryError, match="timestamp|submitted"):
        GitHubClient(FakeRunner([result(pending)])).review_envelope(REPO, 42, 7)


def test_pending_pull_review_with_timestamp_is_rejected_as_an_authenticated_envelope():
    pending = review_record(state="PENDING")
    with pytest.raises(GitHubBoundaryError, match="pending|submitted"):
        GitHubClient(FakeRunner([result(pending)])).review_envelope(REPO, 42, 7)


def test_protocol_body_recursion_failure_is_a_bounded_boundary_error():
    nested = "[" * 2000 + "]" * 2000
    hostile = dict(record(), body=nested)
    with pytest.raises(GitHubBoundaryError, match="protocol payload|body"):
        GitHubClient(FakeRunner([result(hostile)])).comment_envelope(REPO, 11)


def test_protocol_body_round_trip_preserves_report_visible_prefix_exactly():
    payload = body_payload("report-visible")
    payload.update({
        "record_type": "report",
        "report_body": "Line | <tag>\r\nsecond",
        "report_body_sha256": hashlib.sha256("Line | <tag>\r\nsecond".encode()).hexdigest(),
    })
    body = encode_protocol_body(payload, visible_body=payload["report_body"])
    assert decode_protocol_body(body) == payload
    with pytest.raises(GitHubBoundaryError, match="visible|report_body"):
        encode_protocol_body(payload, visible_body="Line | <tag>\nsecond")
    with pytest.raises(GitHubBoundaryError, match="visible|report_body"):
        decode_protocol_body(body.replace("second", "tampered", 1))


def test_protocol_body_preserves_pure_machine_readable_comments_without_stripping():
    payload = body_payload("machine-only")
    body = encode_protocol_body(payload)
    assert body == json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    assert decode_protocol_body(body) == payload


def test_ledger_bearing_machine_only_report_is_rejected_at_encode_but_legacy_report_is_allowed():
    ledger_report = {"schema": "agentic-review/v1", "record_type": "report", "validation_ledger": []}
    with pytest.raises(GitHubBoundaryError, match="visible|protocol body"):
        encode_protocol_body(ledger_report)
    legacy_report = {"schema": "agentic-review/v1", "record_type": "report", "report_body": "legacy"}
    assert decode_protocol_body(encode_protocol_body(legacy_report)) == legacy_report


def test_protocol_comment_size_bound_is_checked_before_mutation_boundary():
    base = {"record_type": "intent", "blob": ""}
    overhead = len(encode_protocol_body(base).encode("utf-8"))
    exact = {"record_type": "intent", "blob": "x" * (65_536 - overhead)}
    assert len(encode_protocol_body(exact).encode("utf-8")) == 65_536
    with pytest.raises(GitHubBoundaryError, match="65,536|size|bound"):
        encode_protocol_body({"record_type": "intent", "blob": "x" * 65_536})


@pytest.mark.parametrize("include_http_headers", [False, True])
def test_api_json_recursion_failure_is_a_bounded_boundary_error(include_http_headers):
    nested = "[" * 10000 + "0" + "]" * 10000
    payload = nested
    if include_http_headers:
        payload = "HTTP/2 200\r\nX-OAuth-Scopes: read:user\r\n\r\n" + payload
    response = subprocess.CompletedProcess(["gh"], 0, payload, "")
    with pytest.raises(GitHubBoundaryError, match="JSON|recursion|depth"):
        GitHubClient(FakeRunner([response]))._request("GET", "/user")


@pytest.mark.parametrize("include_http_headers", [False, True])
def test_api_json_depth_check_ignores_brackets_inside_strings(include_http_headers):
    payload = json.dumps("[" * 10000 + "]" * 10000)
    if include_http_headers:
        payload = "HTTP/2 200\r\nX-OAuth-Scopes: read:user\r\n\r\n" + payload
    response = subprocess.CompletedProcess(["gh"], 0, payload, "")

    assert GitHubClient(FakeRunner([response]))._request("GET", "/user").data == "[" * 10000 + "]" * 10000


def test_api_json_depth_check_ignores_escaped_quotes_and_deep_text_inside_strings():
    value = 'escaped quote: " ' + "[{" * 10000 + "}]" * 10000
    payload = json.dumps(value)
    response = subprocess.CompletedProcess(["gh"], 0, payload, "")

    assert GitHubClient(FakeRunner([response]))._request("GET", "/user").data == value


def test_api_json_depth_check_handles_even_and_odd_backslash_parity_before_nested_json():
    string_fields = json.dumps(
        {"even": "ends with a backslash\\", "odd": 'contains an escaped " quote'},
        separators=(",", ":"),
    )[:-1]
    nested = '{"value":' * 64 + "0" + "}" * 64
    payload = string_fields + ',"nested":' + nested + "}"
    response = subprocess.CompletedProcess(["gh"], 0, payload, "")

    data = GitHubClient(FakeRunner([response]))._request("GET", "/user").data

    assert data["even"] == "ends with a backslash\\"
    assert data["odd"] == 'contains an escaped " quote'
    nested_data = data["nested"]
    for _ in range(63):
        nested_data = nested_data["value"]
    assert nested_data["value"] == 0


@pytest.mark.parametrize(
    "opening, closing, expected_type",
    [("[", "]", list), ('{"value":', "}", dict)],
)
@pytest.mark.parametrize("depth, accepted", [(256, True), (257, False)])
def test_api_json_depth_check_enforces_exact_depth_boundary(opening, closing, expected_type, depth, accepted):
    payload = opening * depth + "0" + closing * depth
    response = subprocess.CompletedProcess(["gh"], 0, payload, "")

    if accepted:
        data = GitHubClient(FakeRunner([response]))._request("GET", "/user").data
        assert isinstance(data, expected_type)
    else:
        with pytest.raises(GitHubBoundaryError, match="JSON|depth|recursion"):
            GitHubClient(FakeRunner([response]))._request("GET", "/user")


@pytest.mark.parametrize(
    "payload",
    [
        '{"unterminated":"value}',
        '{"items":[1,2}',
        '[{"item":1]}',
    ],
)
def test_api_json_depth_check_rejects_unterminated_strings_and_mismatched_delimiters(payload):
    response = subprocess.CompletedProcess(["gh"], 0, payload, "")

    with pytest.raises(GitHubBoundaryError):
        GitHubClient(FakeRunner([response]))._request("GET", "/user")


def test_effective_permission_is_normalized():
    response = result({"user": {**user(), "permissions": {"pull": True, "push": False, "admin": False}}})
    permission = GitHubClient(FakeRunner([response])).collaborator_effective_permission(REPO, "review-bot")
    assert permission.login == "review-bot"
    assert permission.principal_type == "Bot"
    assert permission.permission == "read"


@pytest.mark.parametrize("role, expected", [("push", "write"), ("maintain", "write"), ("triage", "read"), ("pull", "read")])
def test_effective_permission_roles_are_normalized(role, expected):
    response = result({"user": {**user(), "permissions": {}}, "role_name": role})
    assert GitHubClient(FakeRunner([response])).collaborator_effective_permission(REPO, "review-bot").permission == expected


def test_effective_permission_verifies_requested_login():
    response = result({"user": {**user(login="other"), "permissions": {"pull": True}}})
    with pytest.raises(GitHubBoundaryError, match="login"):
        GitHubClient(FakeRunner([response])).collaborator_effective_permission(REPO, "review-bot")


def test_pull_tree_and_blob_responses_are_bound_to_requested_ids():
    with pytest.raises(GitHubBoundaryError, match="number"):
        GitHubClient(FakeRunner([result(pull(41))])).get_pull_request(REPO, 42)
    with pytest.raises(GitHubBoundaryError, match="sha"):
        GitHubClient(FakeRunner([result(dict(tree(), sha="other-sha"))])).get_tree(REPO, "base-sha")
    with pytest.raises(GitHubBoundaryError, match="sha"):
        GitHubClient(FakeRunner([result(dict(blob(), sha="other-sha"))])).get_blob(REPO, "blob-sha")


def test_commit_tree_is_read_from_github_top_level_tree_field():
    response = result({"sha": "commit-sha", "tree": {"sha": "tree-sha", "url": "https://api.invalid/tree"}})
    commit = GitHubClient(FakeRunner([response])).get_commit(REPO, "commit-sha")
    assert commit.data["tree"]["sha"] == "tree-sha"

    nested = result({"sha": "commit-sha", "commit": {"tree": {"sha": "tree-sha"}}})
    with pytest.raises(GitHubBoundaryError, match="missing|commit"):
        GitHubClient(FakeRunner([nested])).get_commit(REPO, "commit-sha")


def test_branch_head_allows_safe_slash_refs_and_rejects_dot_segments():
    runner = FakeRunner([result({"ref": "refs/heads/release/stable", "object": {"sha": "c" * 40, "type": "commit"}})])
    assert GitHubClient(runner).get_branch_head(REPO, "release/stable") == "c" * 40
    assert runner.calls[0][0][-1] == "/repos/owner/repo/git/ref/heads/release/stable"

    runner = FakeRunner([])
    with pytest.raises(GitHubBoundaryError):
        GitHubClient(runner).get_branch_head(REPO, "release/../stable")
    assert runner.calls == []


def test_branch_head_accepts_git_plus_and_rejects_invalid_ref_constructs():
    runner = FakeRunner([result({"ref": "refs/heads/release+stable", "object": {"sha": "c" * 40, "type": "commit"}})])
    assert GitHubClient(runner).get_branch_head(REPO, "release+stable") == "c" * 40
    assert runner.calls[0][0][-1].endswith("/git/ref/heads/release%2Bstable")
    for branch in ("@", "release..stable", "release@{stable}", "release~stable", "release:stable", "release/.lock", "/release", "release/"):
        with pytest.raises(GitHubBoundaryError):
            GitHubClient(FakeRunner([])).get_branch_head(REPO, branch)


def test_authenticated_config_source_binds_branch_commit_and_policy_blobs(tmp_path):
    paths = (
        ".github/agentic-review/providers.json",
        ".github/agentic-review/capabilities-v1.json",
        ".github/agentic-review/trusted-publishers.json",
    )
    contents = tuple((ROOT / path).read_bytes() for path in paths)
    blob_ids = [hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() for content in contents]
    tree_entries = [
        {"path": path, "mode": "100644", "type": "blob", "sha": oid}
        for path, oid in zip(paths, blob_ids)
    ]
    responses = [
        result({"id": 8, "full_name": REPO, "default_branch": "main"}),
        result({"ref": "refs/heads/main", "object": {"sha": "c" * 40, "type": "commit"}}),
        result({"sha": "c" * 40, "tree": {"sha": "t" * 40}}),
        result({"sha": "t" * 40, "tree": tree_entries, "truncated": False}),
        *(result({"sha": oid, "encoding": "base64", "content": base64.b64encode(content).decode(), "size": len(content)})
          for oid, content in zip(blob_ids, contents)),
    ]
    source = GitHubClient(FakeRunner(responses)).authenticated_config_source(
        REPO, commit_sha="c" * 40, repository_root=str(ROOT)
    )
    assert source.authenticated
    assert source.config_digest == configuration_source_digest(*contents)


def test_authenticated_config_source_accepts_slash_default_branch(tmp_path):
    paths = (
        ".github/agentic-review/providers.json",
        ".github/agentic-review/capabilities-v1.json",
        ".github/agentic-review/trusted-publishers.json",
    )
    contents = tuple((ROOT / path).read_bytes() for path in paths)
    blob_ids = [hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() for content in contents]
    responses = [
        result({"id": 8, "full_name": REPO, "default_branch": "release/stable"}),
        result({"ref": "refs/heads/release/stable", "object": {"sha": "c" * 40, "type": "commit"}}),
        result({"sha": "c" * 40, "tree": {"sha": "t" * 40}}),
        result({"sha": "t" * 40, "tree": [
            {"path": path, "mode": "100644", "type": "blob", "sha": oid}
            for path, oid in zip(paths, blob_ids)
        ], "truncated": False}),
        *(result({"sha": oid, "encoding": "base64", "content": base64.b64encode(content).decode(), "size": len(content)})
          for oid, content in zip(blob_ids, contents)),
    ]
    source = GitHubClient(FakeRunner(responses)).authenticated_config_source(
        REPO, commit_sha="c" * 40, repository_root=str(tmp_path)
    )
    assert source.default_branch == "release/stable"


def test_authenticated_config_source_rejects_stale_head_and_unverified_blob():
    responses = [
        result({"id": 8, "full_name": REPO, "default_branch": "main"}),
        result({"ref": "refs/heads/main", "object": {"sha": "d" * 40, "type": "commit"}}),
    ]
    with pytest.raises(GitHubBoundaryError, match="live|head"):
        GitHubClient(FakeRunner(responses)).authenticated_config_source(
            REPO, commit_sha="c" * 40, repository_root=str(ROOT)
        )

    content = b"{}"
    bad_oid = "0" * 40
    entries = [{"path": ".github/agentic-review/providers.json", "mode": "100644", "type": "blob", "sha": bad_oid}]
    responses = [
        result({"id": 8, "full_name": REPO, "default_branch": "main"}),
        result({"ref": "refs/heads/main", "object": {"sha": "c" * 40, "type": "commit"}}),
        result({"sha": "c" * 40, "tree": {"sha": "t" * 40}}),
        result({"sha": "t" * 40, "tree": entries + [
            {"path": ".github/agentic-review/capabilities-v1.json", "mode": "100644", "type": "blob", "sha": bad_oid},
            {"path": ".github/agentic-review/trusted-publishers.json", "mode": "100644", "type": "blob", "sha": bad_oid},
        ], "truncated": False}),
        result({"sha": bad_oid, "encoding": "base64", "content": base64.b64encode(content).decode(), "size": len(content)}),
        result({"sha": bad_oid, "encoding": "base64", "content": base64.b64encode(content).decode(), "size": len(content)}),
        result({"sha": bad_oid, "encoding": "base64", "content": base64.b64encode(content).decode(), "size": len(content)}),
    ]
    with pytest.raises(GitHubBoundaryError, match="object hash"):
        GitHubClient(FakeRunner(responses)).authenticated_config_source(
            REPO, commit_sha="c" * 40, repository_root=str(ROOT)
        )


def test_create_review_sends_exact_commit_id():
    runner = FakeRunner([result({"id": 17, "node_id": "PRR_17"})])
    GitHubClient(runner).create_pull_request_review(
        REPO, 42, body="@file", event="COMMENT", commit_id="exact-head-sha"
    )
    argv, input_data = runner.calls[0]
    assert "--field" not in argv
    assert "--input" in argv and "-" in argv
    assert json.loads(input_data) == {"body": "@file", "event": "COMMENT", "commit_id": "exact-head-sha"}


def test_mutation_labels_are_json_and_at_file_is_not_a_file_reference():
    runner = FakeRunner([result([{"id": 1, "node_id": "L_1", "name": "@file"}])])
    GitHubClient(runner).add_labels(REPO, 42, ["@file"])
    argv, input_data = runner.calls[0]
    assert "--field" not in argv
    assert json.loads(input_data) == {"labels": ["@file"]}


@pytest.mark.parametrize("method", ["create_issue_comment", "create_pull_request_review"])
def test_mutation_response_ids_are_required(method):
    runner = FakeRunner([result({"body": "ok"})])
    with pytest.raises(GitHubBoundaryError, match="id"):
        if method == "create_issue_comment":
            GitHubClient(runner).create_issue_comment(REPO, 42, "comment")
        else:
            GitHubClient(runner).create_pull_request_review(
                REPO, 42, body="comment", event="COMMENT", commit_id="head-sha"
            )


def test_mutation_body_has_a_fixed_input_bound():
    runner = FakeRunner([])
    with pytest.raises(GitHubBoundaryError, match="body|size|bound"):
        GitHubClient(runner).create_issue_comment(REPO, 42, "x" * ((1 << 20) + 1))
    assert runner.calls == []


def test_config_loader_rejects_absolute_and_traversal_overrides(tmp_path):
    for override in ("/etc/providers.json", "../providers.json", ".github/agentic-review/../../providers.json"):
        with pytest.raises(ValueError, match="path|root|travers"):
            load_review_configuration(tmp_path, providers_path=override)


def test_operator_manifest_loader_is_repository_root_relative(tmp_path):
    manifest = {
        "schema": "hipfire.agentic-review.operator-credentials",
        "version": 1,
        "repository": REPO,
        "principal": {"login": "review-bot", "type": "Bot"},
        "allowed_operations": ["publish"],
        "write_permissions": {"issues": "write", "pull_requests": "write"},
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    path = tmp_path / "custom-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_operator_credential_manifest(tmp_path, manifest_path="custom-manifest.json") == manifest
    for override in (str(path), "../custom-manifest.json"):
        with pytest.raises(ValueError, match="path|root|travers"):
            load_operator_credential_manifest(tmp_path, manifest_path=override)


@pytest.mark.parametrize(
    "change",
    [
        {"repository": "../repo"},
        {"write_permissions": {"contents": "write"}},
        {"write_permissions": {"issues": "read", "pull_requests": "write"}},
    ],
)
def test_operator_manifest_declares_exact_repository_and_intended_write_permissions(change):
    manifest = app_manifest()
    manifest.update(change)
    with pytest.raises(ValueError, match="repository|permission"):
        validate_operator_credential_manifest(manifest)


def test_config_loader_uses_task_one_validators():
    configuration = load_review_configuration(ROOT)
    assert configuration.capabilities["schema"] == "hipfire.agentic-review.capabilities"
    assert configuration.providers["providers"] == ()


def preflight_responses(*, scopes="read:user, repo:status", accepted=None, probe=True, tree_probe=False, principal_type="Bot"):
    headers = {"X-OAuth-Scopes": scopes}
    if accepted is not None:
        headers = {"X-Accepted-GitHub-Permissions": accepted}
        if scopes != "read:user, repo:status":
            headers["X-OAuth-Scopes"] = scopes
    responses = [result(user(principal_type=principal_type), headers=headers), result(repository(), headers=headers), result([pull()], headers=headers)]
    if probe:
        responses.append(result(pull(), headers=headers))
        if tree_probe:
            responses.extend([result(tree(), headers=headers), result(blob(), headers=headers)])
        else:
            responses.extend([result([record()], headers=headers), result([review_record()], headers=headers)])
        responses.append(result(permission(principal_type=principal_type), headers=headers))
    return responses


def app_preflight_responses(*, link=None):
    accepted = "metadata=read, pull_requests=write, issues=write"
    headers = {"X-Accepted-GitHub-Permissions": accepted}
    pull_headers = dict(headers)
    if link is not None:
        pull_headers["Link"] = link
    return [
        result(repository(), headers=headers),
        result([pull()], headers=pull_headers),
        result(pull(), headers=headers),
        result([record()], headers=headers),
        result([review_record()], headers=headers),
        result(installation_repositories(), headers=headers),
    ]


def human_preflight_responses():
    headers = {
        "X-OAuth-Scopes": "",
        "X-Accepted-GitHub-Permissions": "metadata=read, pull_requests=write, issues=write",
    }
    return [
        result(human_user(), headers=headers),
        result(repository(), headers=headers),
        result([pull()], headers=headers),
        result(pull(), headers=headers),
        result([record()], headers=headers),
        result([review_record()], headers=headers),
        result(permission(login="reviewer", role="push", principal_type="User"), headers=headers),
    ]


def test_app_token_repository_enumeration_follows_link_to_target_beyond_first_page():
    next_page = '<https://api.github.com/installation/repositories?page=2>; rel="next"'
    first_page = result(
        installation_repositories([dict(repository(), id=9)], total_count=2),
        headers={"X-Accepted-GitHub-Permissions": "metadata=read", "Link": next_page},
    )
    second_page = result(
        installation_repositories([repository()], total_count=2),
        headers={"X-Accepted-GitHub-Permissions": "metadata=read"},
    )
    response = GitHubClient(FakeRunner([first_page, second_page])).list_installation_repositories()
    assert [item["id"] for item in response.data["repositories"]] == [9, 8]


def test_app_token_repository_enumeration_fails_when_link_remains_at_page_cap():
    responses = []
    for page in range(1, 17):
        link = f'<https://api.github.com/installation/repositories?page={page + 1}>; rel="next"'
        responses.append(result(
            installation_repositories([dict(repository(), id=page)], total_count=16),
            headers={"X-Accepted-GitHub-Permissions": "metadata=read", "Link": link},
        ))
    with pytest.raises(GitHubBoundaryError, match="pagination|page|bound"):
        GitHubClient(FakeRunner(responses)).list_installation_repositories()


def test_preflight_probes_only_read_endpoints_with_bounded_pages_and_explicit_principal():
    runner = FakeRunner(preflight_responses(principal_type="User"))
    configuration = load_review_configuration(ROOT)
    # The repository fixture has no trusted apps, so provide a minimal valid
    # configuration copy for the preflight's trust check.
    configuration = configuration.with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    outcome = preflight_read_only(
        GitHubClient(runner), REPO, mode="discovery", configuration=configuration,
        operator_manifest=discovery_manifest(),
    )
    assert outcome.principal_type == "User"
    assert len(runner.calls) == 7
    assert "--method" in runner.calls[0][0]
    assert "per_page=1" in " ".join(runner.calls[2][0])
    assert all(call[0][1] == "api" for call in runner.calls)
    assert all(call[0][call[0].index("--method") + 1] == "GET" for call in runner.calls)


def test_preflight_rejects_classic_repo_scope_and_empty_trust():
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    with pytest.raises(PreflightError, match="classic|scope"):
        preflight_read_only(
            GitHubClient(FakeRunner(preflight_responses(scopes="repo, read:user", probe=False, principal_type="User"))),
            REPO,
            mode="discovery",
            configuration=configuration,
            operator_manifest=discovery_manifest(),
        )


def test_preflight_rejects_malformed_scope_header():
    configuration = load_review_configuration(ROOT)
    with pytest.raises(PreflightError, match="scope"):
        preflight_read_only(
            GitHubClient(FakeRunner(preflight_responses(scopes="read:user,,repo:status", probe=False, principal_type="User"))),
            REPO,
            mode="discovery",
            configuration=configuration,
            operator_manifest=discovery_manifest(),
        )


def test_read_only_preflight_accepts_task_one_empty_apps():
    configuration = load_review_configuration(ROOT)
    outcome = preflight_read_only(
        GitHubClient(FakeRunner(preflight_responses(principal_type="User"))),
        REPO,
        mode="discovery",
        configuration=configuration,
        operator_manifest=discovery_manifest(),
    )
    assert outcome.login == "review-bot"


def test_controller_preflight_uses_effective_permission_without_static_apps():
    configuration = load_review_configuration(ROOT)
    runner = FakeRunner(preflight_responses(tree_probe=True))
    outcome = preflight_read_only(
        GitHubClient(runner),
        REPO,
        mode="controller",
        configuration=configuration,
    )
    assert outcome.login == "review-bot"
    assert len(runner.calls) == 7


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
        "repository": REPO,
        "principal": {"login": "review-bot", "type": "Bot"},
        "allowed_operations": ["publish"],
        "write_permissions": {"issues": "write", "pull_requests": "write"},
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    with pytest.raises(PreflightError, match="matching|App"):
        preflight_read_only(
            GitHubClient(FakeRunner(app_preflight_responses())),
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
        "repository": REPO,
        "principal": {"login": "review-bot", "type": "Bot"},
        "allowed_operations": ["publish"],
        "write_permissions": {"issues": "write", "pull_requests": "write"},
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    runner = FakeRunner(app_preflight_responses())
    preflight_read_only(
        GitHubClient(runner), REPO, mode="publisher", configuration=configuration, operator_manifest=manifest
    )
    assert all("--method" in call[0] and call[0][call[0].index("--method") + 1] == "GET" for call in runner.calls)
    assert runner.calls[-1][0][-1].startswith("/installation/repositories?")
    assert all("/installation" not in call[0][-1] or call[0][-1].startswith("/installation/repositories?") for call in runner.calls)


def test_dismissal_preflight_requires_dismissal_attestation():
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    manifest = {
        "schema": "hipfire.agentic-review.operator-credentials",
        "version": 1,
        "repository": REPO,
        "principal": {"login": "review-bot", "type": "Bot"},
        "allowed_operations": ["dismiss-workflow-review"],
        "write_permissions": {"issues": "write", "pull_requests": "write"},
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    preflight_read_only(
        GitHubClient(FakeRunner(app_preflight_responses())), REPO, mode="dismissal",
        configuration=configuration, operator_manifest=manifest,
    )
    with pytest.raises(PreflightError, match="operation|dismiss"):
        preflight_read_only(
            GitHubClient(FakeRunner(app_preflight_responses())), REPO, mode="dismissal",
            configuration=configuration, operator_manifest={**manifest, "allowed_operations": ["publish"]},
        )


def test_publisher_preflight_accepts_attested_human_fine_grained_pat():
    configuration = load_review_configuration(ROOT)
    manifest = {
        "schema": "hipfire.agentic-review.operator-credentials",
        "version": 1,
        "repository": REPO,
        "principal": {"login": "reviewer", "type": "User"},
        "allowed_operations": ["publish"],
        "write_permissions": {"issues": "write", "pull_requests": "write"},
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    outcome = preflight_read_only(
        GitHubClient(FakeRunner(human_preflight_responses())), REPO, mode="publisher",
        configuration=configuration, operator_manifest=manifest,
    )
    assert outcome.login == "reviewer"
    assert outcome.principal_type == "User"


def test_app_publisher_preflight_requires_manifest_before_api_calls():
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    runner = FakeRunner(app_preflight_responses())
    with pytest.raises(PreflightError, match="manifest|attest"):
        preflight_read_only(GitHubClient(runner), REPO, mode="publisher", configuration=configuration)
    assert runner.calls == []


def test_app_publisher_preflight_with_attestation_avoids_user_and_repo_installation_endpoints():
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    runner = FakeRunner(app_preflight_responses())
    outcome = preflight_read_only(
        GitHubClient(runner), REPO, mode="publisher", configuration=configuration,
        operator_manifest=app_manifest(),
    )
    assert outcome.login == "review-bot"
    assert all(call[0][-1] != "/user" for call in runner.calls)
    assert runner.calls[-1][0][-1].startswith("/installation/repositories?")
    assert all("/repos/owner/repo/installation" not in call[0] for call in runner.calls)


@pytest.mark.parametrize("mode", ["discovery", "publisher", "dismissal"])
def test_write_preflight_requires_operator_manifest_for_configured_app(mode):
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    runner = FakeRunner([])
    with pytest.raises(PreflightError, match="manifest|attest"):
        preflight_read_only(GitHubClient(runner), REPO, mode=mode, configuration=configuration)
    assert runner.calls == []


def test_publisher_preflight_does_not_claim_get_permission_proves_write_authority():
    configuration = load_review_configuration(ROOT)
    manifest = {
        "schema": "hipfire.agentic-review.operator-credentials",
        "version": 1,
        "repository": REPO,
        "principal": {"login": "reviewer", "type": "User"},
        "allowed_operations": ["publish"],
        "write_permissions": {"issues": "write", "pull_requests": "write"},
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    runner = FakeRunner(human_preflight_responses()[:-1])
    outcome = preflight_read_only(
        GitHubClient(runner), REPO, mode="publisher", configuration=configuration,
        operator_manifest=manifest,
    )
    assert outcome.login == "reviewer"
    assert all("collaborators" not in call[0][-1] for call in runner.calls)


def test_discovery_preflight_probes_effective_permission_and_rejects_inaccessible_response():
    configuration = load_review_configuration(ROOT)
    responses = preflight_responses(principal_type="User")
    responses[-1] = result({}, returncode=1, stderr="forbidden")
    with pytest.raises(PreflightError, match="exit|forbidden|permission"):
        preflight_read_only(
            GitHubClient(FakeRunner(responses)), REPO, mode="discovery",
            configuration=configuration, operator_manifest=discovery_manifest(),
        )


def test_publisher_preflight_rejects_manifest_repository_mismatch():
    configuration = load_review_configuration(ROOT)
    manifest = {**app_manifest(), "repository": "other/repo"}
    with pytest.raises(PreflightError, match="repository|manifest"):
        preflight_read_only(
            GitHubClient(FakeRunner([])), REPO, mode="publisher", configuration=configuration,
            operator_manifest=manifest,
        )


def test_preflight_sample_accepts_next_link_without_claiming_exhaustive_discovery():
    configuration = load_review_configuration(ROOT).with_trusted_publishers(
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [
            {"app_id": 1, "login": "review-bot", "installation_id": 2, "repository_id": 8,
             "credential_attestation_digest": "sha256:" + "a" * 64}
        ]}
    )
    next_page = '<https://api.github.com/repos/owner/repo/pulls?page=2>; rel="next"'
    outcome = preflight_read_only(
        GitHubClient(FakeRunner(app_preflight_responses(link=next_page))),
        REPO,
        mode="publisher",
        configuration=configuration,
        operator_manifest=app_manifest(),
    )
    assert outcome.login == "review-bot"


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
            GitHubClient(FakeRunner([result(bad_user)])), REPO, mode="discovery", configuration=configuration,
            operator_manifest=discovery_manifest(),
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
            REPO, mode="discovery", configuration=configuration, operator_manifest=discovery_manifest(),
        )


def test_preflight_has_explicit_no_open_pr_behavior():
    configuration = load_review_configuration(ROOT)
    with pytest.raises(PreflightError, match="open|pull request"):
        preflight_read_only(
            GitHubClient(FakeRunner(preflight_responses(probe=False, principal_type="User")[:2] + [result([])])),
            REPO, mode="discovery", configuration=configuration, operator_manifest=discovery_manifest(),
        )


def test_preflight_accepts_fine_grained_permission_headers_and_requires_needed_permission():
    configuration = load_review_configuration(ROOT)
    accepted = "metadata=read, pull_requests=read, issues=read, contents=read"
    outcome = preflight_read_only(
        GitHubClient(FakeRunner(preflight_responses(accepted=accepted, principal_type="User"))),
        REPO, mode="discovery", configuration=configuration, operator_manifest=discovery_manifest(),
    )
    assert outcome.scopes == ()
    with pytest.raises(PreflightError, match="permission"):
        preflight_read_only(
            GitHubClient(FakeRunner(preflight_responses(accepted="metadata=read", principal_type="User"))),
            REPO, mode="discovery", configuration=configuration, operator_manifest=discovery_manifest(),
        )


def test_preflight_rejects_visible_classic_repo_even_with_fine_grained_permissions():
    configuration = load_review_configuration(ROOT)
    with pytest.raises(PreflightError, match="classic|scope"):
        preflight_read_only(
            GitHubClient(FakeRunner(preflight_responses(
                scopes="repo",
                accepted="metadata=read, pull_requests=write, issues=write",
                principal_type="User",
            ))),
            REPO,
            mode="discovery",
            configuration=configuration,
            operator_manifest=discovery_manifest(),
        )


def test_record_pagination_fails_instead_of_returning_partial_data_at_page_cap():
    next_page = '<https://api.github.com/repos/owner/repo/issues/42/comments?page=17>; rel="next"'
    responses = [result([], headers={"X-OAuth-Scopes": "read:user", "Link": next_page}) for _ in range(16)]
    with pytest.raises(GitHubBoundaryError, match="pagination|page|bound"):
        GitHubClient(FakeRunner(responses)).list_issue_comments(REPO, 42)
