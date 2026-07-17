# Copyright (c) Kaden Schutt
"""Focused contract tests for exhaustive PR review discovery."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from autoresearch.ar.review.discovery import DiscoverySummary, discover_pull_requests
from autoresearch.ar.review.github import GitHubBoundaryError
from autoresearch.ar.review.canonical import canonical_digest, metadata_digest
from autoresearch.ar.review.models import ReviewProposal, ReviewTarget
from autoresearch.ar.review.publisher import ReviewPublisher
from autoresearch.ar.tests.test_review_publisher import (
    FakeGitHub,
    OPERATOR as BOT_OPERATOR,
    TARGET as PUBLISH_TARGET,
    _configuration,
    _proposal,
)


REPO = "owner/repo"
TARGET = ReviewTarget(REPO, 42, "fork/repo", "head", "main", "base", "merge")
OPERATOR = {
    "schema": "hipfire.agentic-review.operator-credentials",
    "version": 1,
    "repository": REPO,
    "principal": {"login": "review-bot", "type": "Bot"},
    "allowed_operations": ["discover", "dismiss-workflow-review"],
    "write_permissions": {"issues": "write", "pull_requests": "write"},
    "credential_attestation_digest": "sha256:" + "a" * 64,
}
DISCOVERY_BOT = {**BOT_OPERATOR, "allowed_operations": ["discover", "dismiss-workflow-review"]}
DISCOVERY_HUMAN = {
    **OPERATOR,
    "principal": {"login": "reviewer", "type": "User"},
}


class Client:
    def __init__(self, pulls=None):
        self.pulls = pulls if pulls is not None else [{"number": TARGET.number, "draft": False}]
        self.target = TARGET
        self.targets = {}
        self.labels = set()
        self.calls = []
        self.fail_add = False
        self.permission = "write"

    def list_pull_requests(self, repository, *, max_pages=16):
        self.calls.append(("list", max_pages))
        return SimpleNamespace(data=list(self.pulls), headers={})

    def get_review_target(self, repository, number):
        self.calls.append(("target", number))
        return getattr(self, "targets", {}).get(number, self.target)

    def list_issue_comments(self, repository, number):
        self.calls.append(("comments", number))
        return SimpleNamespace(data=[])

    def list_pull_reviews(self, repository, number):
        self.calls.append(("reviews", number))
        return SimpleNamespace(data=[])

    def list_issue_labels(self, repository, number):
        return SimpleNamespace(data=[{"name": name} for name in sorted(self.labels)])

    def add_labels(self, repository, number, labels):
        self.calls.append(("add", tuple(labels)))
        if self.fail_add:
            raise GitHubBoundaryError("label API failed")
        self.labels.update(labels)
        return SimpleNamespace(data=[])

    def remove_label(self, repository, number, label):
        self.calls.append(("remove", label))
        self.labels.discard(label)
        return SimpleNamespace(data={})

    def collaborator_effective_permission(self, repository, login):
        return SimpleNamespace(login=login, principal_type="User", permission=self.permission)

    def get_authenticated_user(self):
        return SimpleNamespace(data={"id": 1, "login": "review-bot", "type": "Bot"})

    def get_repository(self, repository):
        return SimpleNamespace(data={"id": 8, "full_name": repository})


def manifest(login="reviewer", principal_type="User"):
    return {**OPERATOR, "principal": {"login": login, "type": principal_type}}


def configuration():
    return _configuration()


def app_configuration():
    result = configuration().with_trusted_publishers({
        "schema": "hipfire.agentic-review.trusted-publishers",
        "version": 1,
        "apps": [{
            "app_id": 1,
            "login": "review-bot",
            "installation_id": 2,
            "repository_id": 8,
            "credential_attestation_digest": DISCOVERY_BOT["credential_attestation_digest"],
        }],
    })
    source = result.source
    object.__setattr__(result, "_loaded_from_protected_paths", True)
    object.__setattr__(result, "_loaded_source_digest", source.config_digest)
    object.__setattr__(result, "_loaded_root_identity", source.root_identity)
    return result


def multi_app_configuration():
    result = app_configuration()
    apps = list(result.trusted_publishers["apps"])
    apps.append({
        **apps[0],
        "login": "other-bot",
        "app_id": 2,
        "installation_id": 3,
    })
    policy = {
        "schema": "hipfire.agentic-review.trusted-publishers",
        "version": 1,
        "apps": apps,
    }
    result = result.with_trusted_publishers(policy)
    source = result.source
    object.__setattr__(result, "_loaded_from_protected_paths", True)
    object.__setattr__(result, "_loaded_source_digest", source.config_digest)
    object.__setattr__(result, "_loaded_root_identity", source.root_identity)
    return result


def completed_client(verdict="clean"):
    client = FakeGitHub()
    publish_target = PUBLISH_TARGET
    client.pull = client._pull(publish_target)
    result = __import__("autoresearch.ar.review.publisher", fromlist=["ReviewPublisher"]).ReviewPublisher(
        client, configuration=configuration(), operator_credential=BOT_OPERATOR
    ).publish(_proposal(verdict), publish_target)
    assert result.status == "complete", result.reason
    for record in [*client.comments, *client.reviews]:
        record.update(
            app_id=1,
            installation_id=2,
            repository_id=8,
            credential_attestation_digest=DISCOVERY_BOT["credential_attestation_digest"],
        )
    client.list_pull_requests = lambda repository, *, max_pages=16: SimpleNamespace(
        data=[{"number": TARGET.number, "draft": False}], headers={}
    )
    client.get_repository = lambda repository: SimpleNamespace(data={"id": 8, "full_name": repository})
    client.list_installation_repositories = lambda: SimpleNamespace(
        data={"repositories": [{"id": 8}]}
    )
    return client


def legacy_completed_client():
    client = completed_client()
    payloads = {
        item["id"]: json.loads(client.payload_from_body(item["body"])) for item in client.comments
    }
    by_type = {payload["record_type"]: payload for payload in payloads.values()}
    for payload in payloads.values():
        payload["schema"] = "agentic-review/v1"
        for field in (
            "retrieved_file_count", "expected_file_count", "retrieved_blob_count", "expected_blob_count",
            "retrieved_content_count", "expected_content_count", "coverage_complete",
        ):
            payload.pop(field, None)
    intent = by_type["intent"]
    intent["canonical_digest"] = canonical_digest({key: value for key, value in intent.items() if key != "canonical_digest"})
    by_type["report"]["canonical_intent_digest"] = intent["canonical_digest"]
    metadata = by_type["review-metadata"]
    metadata["canonical_intent_digest"] = intent["canonical_digest"]
    metadata["report_digest"] = canonical_digest(by_type["report"])
    metadata["metadata_digest"] = metadata_digest(metadata)
    completion = by_type["completion"]
    completion["canonical_intent_digest"] = intent["canonical_digest"]
    completion["report_digest"] = canonical_digest(by_type["report"])
    completion["metadata_digest"] = metadata["metadata_digest"]
    for item in client.comments:
        payload = payloads[item["id"]]
        item["body"] = json.dumps(payload)
    return client


def test_no_report_is_needing_review_and_labelled_idempotently():
    client = Client()
    first = discover_pull_requests(client, REPO, configuration=configuration(), operator_credential=manifest())
    second = discover_pull_requests(client, REPO, configuration=configuration(), operator_credential=manifest())

    assert [item.number for item in first.needs_review] == [42]
    assert [item.number for item in first.labelled] == [42]
    assert [item.number for item in second.needs_review] == [42]
    assert [item.number for item in second.labelled] == []
    assert sorted(client.labels) == ["needs-review"]


def test_drafts_and_fork_heads_are_not_filtered():
    client = Client([{"number": 2, "draft": True}, {"number": 1, "draft": False}])
    client.targets = {
        1: replace(TARGET, number=1, head_repository="fork/one"),
        2: replace(TARGET, number=2, head_repository="fork/two"),
    }
    summary = discover_pull_requests(client, REPO, configuration=configuration(), operator_credential=manifest())

    assert [item.number for item in summary.needs_review] == [1, 2]
    assert not summary.incomplete
    assert all("mismatched" not in item.reason for item in summary.needs_review)


def test_each_authorized_human_record_is_checked_and_accepted():
    client = completed_client()
    for record in [*client.comments, *client.reviews]:
        record["user"] = {"login": "other-reviewer", "type": "User"}
        for field in ("app_id", "installation_id", "repository_id", "credential_attestation_digest"):
            record.pop(field, None)
    client.collaborator_effective_permission = lambda repository, login: SimpleNamespace(
        login=login, principal_type="User", permission="write"
    )
    summary = discover_pull_requests(
        client, REPO, configuration=configuration(), operator_credential=DISCOVERY_HUMAN
    )

    assert [item.number for item in summary.clean] == [42]


def test_each_configured_app_record_is_checked_against_installation_scope():
    client = completed_client()
    for record in [*client.comments, *client.reviews]:
        record["user"] = {"login": "other-bot", "type": "Bot"}
        record.update(app_id=2, installation_id=3)
    summary = discover_pull_requests(
        client, REPO, configuration=multi_app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.clean] == [42]


def test_current_completion_requires_explicit_complete_coverage_evidence():
    client = completed_client()
    completion = next(
        item for item in client.comments
        if json.loads(client.payload_from_body(item["body"]))["record_type"] == "completion"
    )
    payload = json.loads(client.payload_from_body(completion["body"]))
    payload["coverage_complete"] = False
    completion["body"] = json.dumps(payload)
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.needs_review] == [42]


def test_legacy_completion_without_coverage_evidence_requires_review():
    client = legacy_completed_client()
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.needs_review] == [42]
    assert "coverage" in summary.needs_review[0].reason


def test_published_records_carry_strict_coverage_evidence():
    client = completed_client()
    payloads = [json.loads(client.payload_from_body(item["body"])) for item in client.comments]
    for payload in payloads:
        if payload["record_type"] in {"report", "review-metadata", "completion"}:
            assert payload["coverage_complete"] is True
            assert payload["retrieved_file_count"] == payload["expected_file_count"]
            assert payload["retrieved_blob_count"] == payload["expected_blob_count"]
            assert payload["retrieved_content_count"] == payload["expected_content_count"]


def test_publisher_rejects_proposal_without_explicit_coverage():
    client = FakeGitHub()
    values = {
        "target": PUBLISH_TARGET,
        "target_key": PUBLISH_TARGET.target_key(),
        "capsule_digest": "sha256:" + "a" * 64,
        "adapter_id": "adapter", "adapter_version": "1", "model": "model",
        "response_digest": "sha256:" + "c" * 64,
        "verdict": "clean", "findings": (),
    }
    legacy = ReviewProposal(
        PUBLISH_TARGET, values["capsule_digest"], "sha256:" + canonical_digest(values), "clean", (),
        "adapter", "1", "model", values["response_digest"],
    )
    result = ReviewPublisher(client, configuration=configuration(), operator_credential=BOT_OPERATOR).publish(
        legacy, PUBLISH_TARGET
    )

    assert result.status in {"error", "incomplete"}
    assert not any(item["record_type"] == "completion" for item in [
        json.loads(client.payload_from_body(comment["body"])) for comment in client.comments
    ])


def test_trusted_malformed_workflow_record_needs_review_but_untrusted_spoof_is_ignored():
    client = Client()
    client.list_issue_comments = lambda repository, number: SimpleNamespace(data=[
        {"id": 1, "body": "{malformed", "user": {"login": "reviewer", "type": "User"}},
        {"id": 2, "body": "{malformed", "user": {"login": "attacker", "type": "User"}},
    ])
    summary = discover_pull_requests(client, REPO, configuration=configuration(), operator_credential=manifest())

    assert summary.needs_review[0].number == 42
    assert "malformed" in summary.needs_review[0].reason


def test_incomplete_scan_is_explicit_and_has_no_review_success():
    class Broken(Client):
        def list_pull_requests(self, repository, *, max_pages=16):
            raise GitHubBoundaryError("pagination reached fixed page bound")

    summary = discover_pull_requests(Broken(), REPO, configuration=configuration(), operator_credential=manifest())

    assert summary.incomplete
    assert not summary.reviewed
    assert not summary.clean


def test_pagination_cap_is_passed_to_existing_bounded_github_component():
    client = Client()
    summary = discover_pull_requests(
        client, REPO, configuration=configuration(), operator_credential=manifest(), max_pages=3
    )

    assert not summary.incomplete
    assert ("list", 3) in client.calls


def test_label_failure_is_an_error_and_needs_review_is_not_claimed_clean():
    client = Client()
    client.fail_add = True
    summary = discover_pull_requests(client, REPO, configuration=configuration(), operator_credential=manifest())

    assert summary.errors[0].number == 42
    assert not summary.clean
    assert summary.needs_review[0].number == 42


def test_dynamic_human_permission_must_be_write_or_admin():
    client = Client()
    client.permission = "read"
    summary = discover_pull_requests(client, REPO, configuration=configuration(), operator_credential=manifest())
    assert summary.incomplete


def test_valid_current_clean_completion_is_clean_and_reconciles_label():
    client = completed_client()
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.clean] == [42], summary
    assert not summary.needs_review
    assert not summary.incomplete


@pytest.mark.parametrize("field", ["head_sha", "base_sha", "merge_base_sha"])
def test_stale_full_target_requires_review(field):
    client = completed_client()
    client.target = replace(replace(TARGET, head_repository=REPO), **{field: "new-" + field})
    client.pull = client._pull(client.target)
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.needs_review] == [42]
    assert "completion" in summary.needs_review[0].reason or "history" in summary.needs_review[0].reason


def test_stale_workflow_cleanup_preserves_human_review():
    client = completed_client("changes-requested")
    client.reviews.append({
        "id": 999,
        "node_id": "human-review",
        "user": {"login": "alice", "type": "User"},
        "submitted_at": "2026-01-01T00:20:00Z",
        "body": "human decision",
        "state": "CHANGES_REQUESTED",
        "commit_id": replace(TARGET, head_repository=REPO).head_sha,
    })
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.clean] == [42]
    assert ("dismiss", 999) not in client.calls


def test_stale_workflow_review_is_dismissed_before_clean_label_removal():
    client = completed_client("changes-requested")
    client.labels.add("needs-review")
    client.inject_review_on_labels = True
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.clean] == [42]
    assert ("dismiss", 902) in client.calls
    assert client.calls.index(("dismiss", 902)) < [
        index for index, call in enumerate(client.calls) if call == ("remove_label", "needs-review")
    ][-1]


def test_workflow_review_is_dismissed_without_current_completion():
    client = completed_client("changes-requested")
    completion = next(
        item for item in client.comments
        if json.loads(client.payload_from_body(item["body"]))["record_type"] == "completion"
    )
    client.comments.remove(completion)
    review_id = client.reviews[0]["id"]
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.needs_review] == [42]
    assert ("dismiss", review_id) not in client.calls


def test_newly_observed_workflow_review_fetch_failure_fails_closed():
    client = completed_client("changes-requested")
    client.labels.add("needs-review")
    client.inject_review_on_labels = True
    original = client.get_pull_review_record

    def failing_fetch(repository, number, review_id):
        if review_id == 902:
            raise RuntimeError("exact review fetch failed")
        return original(repository, number, review_id)

    client.get_pull_review_record = failing_fetch
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert summary.incomplete
    assert summary.errors
    assert "needs-review" in client.labels


def test_label_only_discovery_does_not_require_dismissal_authority():
    client = Client()
    operator = {**manifest(), "write_permissions": {"issues": "write"}}
    summary = discover_pull_requests(
        client, REPO, configuration=configuration(), operator_credential=operator
    )

    assert [item.number for item in summary.needs_review] == [42]
    assert [item.number for item in summary.labelled] == [42]
    assert not summary.incomplete


def test_app_record_envelope_must_bind_configured_app_identity():
    client = completed_client()
    for record in [*client.comments, *client.reviews]:
        record["user"] = {"login": "review-bot", "type": "Bot"}
        record["app_id"] = 999
        record["installation_id"] = 2
        record["repository_id"] = 8
        record["credential_attestation_digest"] = DISCOVERY_BOT["credential_attestation_digest"]
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.needs_review] == [42]
    assert not summary.clean


def test_discovery_uses_public_reconciliation_not_private_publisher_helper(monkeypatch):
    client = completed_client()
    monkeypatch.setattr(ReviewPublisher, "_remove_label", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("private helper used")))
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.clean] == [42]


def test_missing_mutation_authority_is_incomplete_and_retains_label():
    client = completed_client("changes-requested")
    client.labels.add("needs-review")
    client.inject_review_on_labels = True
    no_dismiss = {**DISCOVERY_BOT, "allowed_operations": ["discover"]}
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=no_dismiss
    )

    assert summary.incomplete, client.calls
    assert [item.number for item in summary.needs_review] == [42]
    assert "needs-review" in client.labels


def test_invalid_trust_configuration_returns_deterministic_incomplete_summary():
    summary = discover_pull_requests(
        Client(), REPO, configuration=configuration(), operator_credential={"invalid": True}
    )

    assert summary.incomplete
    assert summary.errors == tuple(sorted(summary.errors, key=lambda item: (item.number, item.reason)))


@pytest.mark.parametrize("mutation", ["edited", "deleted"])
def test_edited_or_deleted_trusted_record_is_incomplete(mutation):
    client = completed_client()
    report_id = next(
        item["id"] for item in client.comments
        if '"record_type":"report"' in item["body"]
    )
    if mutation == "edited":
        client.edited_comment_ids.add(report_id)
    else:
        client.deleted_comment_ids.add(report_id)
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.needs_review] == [42]
    assert not summary.clean


def test_invalid_active_requested_change_review_needs_review():
    client = completed_client("changes-requested")
    client.reviews[0]["state"] = "COMMENTED"
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.needs_review] == [42]
    assert "active requested-change" in summary.needs_review[0].reason


def test_target_mutation_during_reconciliation_retains_safety_label():
    client = completed_client("changes-requested")
    client.change_target_on_labels = replace(replace(TARGET, head_repository=REPO), merge_base_sha="advanced-merge")
    summary = discover_pull_requests(
        client, REPO, configuration=app_configuration(), operator_credential=DISCOVERY_BOT
    )

    assert [item.number for item in summary.needs_review] == [42]
    assert "needs-review" in client.labels


def test_unconfigured_app_cannot_become_trusted_by_spoofed_login():
    client = Client()
    summary = discover_pull_requests(client, REPO, configuration=configuration(), operator_credential=DISCOVERY_BOT)
    assert summary.incomplete
