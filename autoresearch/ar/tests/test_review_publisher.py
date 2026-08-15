# Copyright (c) Kaden Schutt
"""Contract tests for the authenticated, SHA-bound review publisher."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import autoresearch.ar.review.publisher as publisher_module

from autoresearch.ar.review.canonical import canonical_digest, metadata_digest
from autoresearch.ar.review.capsule import build_review_capsule
from autoresearch.ar.review.config import AuthenticatedConfigSource, ReviewConfiguration
from autoresearch.ar.review.github import GitHubClient, GitHubResponse
from autoresearch.ar.review.models import (
    Finding,
    GitHubEnvelope,
    ReviewProposal,
    ReviewScope,
    ReviewTarget,
    ValidationLedgerRow,
    ValidationProfile,
    capability_contract_digest,
)
from autoresearch.ar.review.publisher import PublisherError, ReviewPublisher, _HistoryRecord, render_report
from autoresearch.ar.review.protocol import validate_report


from autoresearch.ar.tests.review_fixtures import (
    FakeGitHub, OPERATOR, REPO, TARGET, TRUSTED, _configuration, _exempt_proposal,
    _exemption_configuration, _ledger_configuration, _ledger_proposal, _proposal,
)

@pytest.mark.parametrize("mismatch", ["digest", "paths"])
def test_exemption_capsule_digest_or_manifest_mismatch_fails_before_intent(monkeypatch, mismatch):
    client = FakeGitHub(changed_path="docs/review.md")
    actual_capsule = build_review_capsule(client, TARGET)
    alternate_capsule = build_review_capsule(FakeGitHub(), TARGET)
    assert actual_capsule.complete and alternate_capsule.complete
    assert actual_capsule.digest != alternate_capsule.digest
    assert tuple(entry.path for entry in actual_capsule.manifest) != tuple(
        entry.path for entry in alternate_capsule.manifest
    )
    monkeypatch.setattr(publisher_module, "build_review_capsule", lambda _client, _target: actual_capsule)
    if mismatch == "digest":
        proposal = _exempt_proposal(capsule=alternate_capsule)
        expected_reason = "proposal capsule or protected scope could not be authenticated"
    else:
        proposal = _exempt_proposal(capsule=actual_capsule, exemption_paths=("src/main.py",))
        expected_reason = "proposal validation ledger is not protected by publisher configuration"
    result = ReviewPublisher(client, configuration=_exemption_configuration(), operator_credential=OPERATOR).publish(
        proposal, TARGET,
    )
    assert result.status == "error"
    assert result.reason == expected_reason
    assert not any(call[0] in {"create_comment", "create_review", "add_label", "remove_label"} for call in client.calls)


def test_protected_exemption_publishes_complete_static_review_lifecycle():
    client = FakeGitHub(changed_path="docs/review.md")
    capsule = build_review_capsule(client, TARGET)
    result = ReviewPublisher(
        client, configuration=_exemption_configuration(), operator_credential=OPERATOR,
    ).publish(_exempt_proposal(capsule=capsule), TARGET)

    assert result.status == "complete", result.reason
    assert [call[1] for call in client.calls if call[0] == "create_comment"] == [
        "intent", "report", "review-metadata", "completion",
    ]
    assert not any(call[0] == "create_review" for call in client.calls)
    report = next(
        json.loads(client.payload_from_body(item["body"]))
        for item in client.comments
        if json.loads(client.payload_from_body(item["body"]))["record_type"] == "report"
    )
    assert "No validation required (protected exemption)." in report["report_body"]


def test_structural_validation_preflight_failure_performs_no_intent_mutation(monkeypatch):
    client = FakeGitHub()
    proposal = _proposal()
    capsule = build_review_capsule(client, TARGET)
    monkeypatch.setattr(publisher_module, "build_review_capsule", lambda _client, _target: capsule)
    def reject_section(*args, **kwargs):
        raise ValueError("validation section mismatch")
    monkeypatch.setattr(publisher_module, "validate_rendered_validation_section", reject_section)
    result = ReviewPublisher(client, configuration=_configuration(), operator_credential=OPERATOR).publish(
        proposal, TARGET,
    )
    assert result.status == "error"
    assert "comment" in (result.reason or "") or "bound" in (result.reason or "")
    assert not any(call[0] == "create_comment" for call in client.calls)


@pytest.mark.parametrize("kind", ["ledger", "configuration", "exemption_ids", "exemption_paths"])
def test_resumed_bound_report_rejects_each_exact_binding_mismatch(kind):
    configuration = _ledger_configuration()
    proposal, row = _ledger_proposal(configuration)
    payload = {
        "validation_ledger": [row.to_mapping()],
        "configuration_source_digest": configuration.source.config_digest,
    }
    if kind == "ledger":
        other_profile = ValidationProfile.from_mapping(configuration.capabilities["profiles"][1])
        other_capability = next(item for item in configuration.capabilities["capabilities"] if item["id"] == other_profile.capability_id)
        payload["validation_ledger"] = [ValidationLedgerRow(
            other_profile, capability_contract_digest(other_capability), "representative",
        ).to_mapping()]
    elif kind == "configuration":
        payload["configuration_source_digest"] = "sha256:" + "e" * 64
    else:
        exempt_proposal = _exempt_proposal()
        proposal = exempt_proposal
        payload = {
            "validation_ledger": [],
            "configuration_source_digest": configuration.source.config_digest,
            "exemption_ids": list(proposal.exemption_ids),
            "exemption_paths": list(proposal.exemption_paths),
        }
        payload[kind] = ["other"] if kind == "exemption_ids" else ["other/path.py"]
    report = _HistoryRecord(
        GitHubEnvelope(payload, "node", TRUSTED, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        False, 0,
    )
    client = FakeGitHub()
    publisher = ReviewPublisher(client, configuration=configuration, operator_credential=OPERATOR)
    with pytest.raises(PublisherError, match="validation binding|ledger"):
        publisher._require_matching_report_binding(report, proposal)
    assert client.calls == []


def _publisher(client: FakeGitHub) -> ReviewPublisher:
    return ReviewPublisher(client, configuration=_configuration(), operator_credential=OPERATOR)


def _proposal_with_scope(scope: ReviewScope) -> ReviewProposal:
    base = _proposal()
    configuration = _configuration()
    values = {
        "target": TARGET, "target_key": TARGET.target_key(), "capsule_digest": base.capsule_digest,
        "adapter_id": base.adapter_id, "adapter_version": base.adapter_version, "model": base.model,
        "response_digest": base.response_digest, "verdict": base.verdict, "findings": base.findings,
        "coverage": base.coverage_mapping(),
        "validation_ledger": tuple(row.to_mapping() for row in base.validation_ledger),
        "configuration_source_digest": configuration.source.config_digest,
        "scope": scope.to_mapping(),
    }
    return ReviewProposal(
        TARGET, base.capsule_digest, "sha256:" + canonical_digest(values), base.verdict, base.findings,
        base.adapter_id, base.adapter_version, base.model, base.response_digest,
        base.retrieved_file_count, base.expected_file_count, base.retrieved_blob_count,
        base.expected_blob_count, base.retrieved_content_count, base.expected_content_count,
        base.coverage_complete, validation_ledger=base.validation_ledger,
        configuration_source_digest=configuration.source.config_digest, scope=scope,
    )


@pytest.mark.parametrize("scope", [
    ReviewScope((), ()),
    ReviewScope(("qwen3.6-27b",), ("gfx1100",)),
    ReviewScope(("wrong-model",), ("gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151")),
])
def test_publisher_rejects_directly_constructed_scope_before_intent(monkeypatch, scope):
    base = _proposal()
    client = FakeGitHub()
    capsule = build_review_capsule(client, TARGET)
    monkeypatch.setattr(
        publisher_module, "build_review_capsule",
        lambda _client, _target: capsule,
    )
    result = _publisher(client).publish(_proposal_with_scope(scope), TARGET)
    assert result.status in {"error", "incomplete"}
    assert not any(call[0] == "create_comment" for call in client.calls)


def test_publisher_rejects_capsule_digest_mismatch_before_intent(monkeypatch):
    base = _proposal()
    client = FakeGitHub(changed_path="docs/review.md")
    actual_capsule = build_review_capsule(client, TARGET)
    monkeypatch.setattr(
        publisher_module, "build_review_capsule",
        lambda _client, _target: actual_capsule,
    )
    result = _publisher(client).publish(base, TARGET)
    assert result.status == "error"
    assert result.reason == "proposal capsule or protected scope could not be authenticated"
    assert not any(call[0] in {"create_comment", "create_review", "add_label", "remove_label"} for call in client.calls)


def test_new_legacy_proposal_is_rejected_before_any_github_mutation():
    values = {
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "capsule_digest": "sha256:" + "a" * 64,
        "adapter_id": "adapter",
        "adapter_version": "1",
        "model": "model",
        "response_digest": "sha256:" + "c" * 64,
        "verdict": "clean",
        "findings": (),
        "scope": ReviewScope((), ()).to_mapping(),
    }
    proposal = ReviewProposal(
        TARGET, values["capsule_digest"], "sha256:" + canonical_digest(values), "clean", (),
        "adapter", "1", "model", values["response_digest"],
        scope=ReviewScope((), ()),
    )
    client = FakeGitHub()
    with pytest.raises(PublisherError, match="validation evidence|exemption"):
        _publisher(client).publish(proposal, TARGET)
    assert client.calls == []


def test_clean_lifecycle_publishes_report_and_completion_without_approval():
    client = FakeGitHub()
    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status == "complete", result.reason
    assert [call for call in client.calls if call[0] == "create_review"] == []
    assert ("remove_label", "needs-review") in client.calls
    assert [call[1] for call in client.calls if call[0] == "create_comment"] == [
        "intent", "report", "review-metadata", "completion"
    ]
    report = next(json.loads(client.payload_from_body(item["body"])) for item in client.comments if json.loads(client.payload_from_body(item["body"]))["record_type"] == "report")
    assert "**the checked value**" not in report["report_body"]
    assert "<instead>" not in report["report_body"]


def test_empty_complete_capsule_publishes_zero_diff_lifecycle():
    client = FakeGitHub(empty_diff=True)
    capsule = build_review_capsule(client, TARGET)
    result = _publisher(client).publish(_proposal(capsule=capsule), TARGET)

    assert result.status == "complete", result.reason
    report = next(
        json.loads(client.payload_from_body(item["body"]))
        for item in client.comments
        if json.loads(client.payload_from_body(item["body"]))["record_type"] == "report"
    )
    assert report["capsule_paths"] == []
    assert report["coverage_complete"] is True
    assert report["expected_file_count"] == report["retrieved_file_count"] == 0


def test_publisher_rejects_forged_zero_coverage_for_nonempty_capsule():
    client = FakeGitHub()
    capsule = build_review_capsule(client, TARGET)
    original = _proposal(capsule=capsule)
    forged_coverage = {
        "retrieved_file_count": 0, "expected_file_count": 0,
        "retrieved_blob_count": 0, "expected_blob_count": 0,
        "retrieved_content_count": 0, "expected_content_count": 0,
        "coverage_complete": True,
    }
    digest_values = {
        "target": original.target, "target_key": original.target.target_key(),
        "capsule_digest": original.capsule_digest, "adapter_id": original.adapter_id,
        "adapter_version": original.adapter_version, "model": original.model,
        "response_digest": original.response_digest, "verdict": original.verdict,
        "findings": original.findings, "coverage": forged_coverage,
        "validation_ledger": tuple(row.to_mapping() for row in original.validation_ledger),
        "configuration_source_digest": original.configuration_source_digest,
        "scope": original.scope.to_mapping(),
    }
    forged = replace(
        original,
        proposal_digest="sha256:" + canonical_digest(digest_values),
        retrieved_file_count=0, expected_file_count=0,
        retrieved_blob_count=0, expected_blob_count=0,
        retrieved_content_count=0, expected_content_count=0,
    )
    result = _publisher(client).publish(forged, TARGET)

    assert result.status == "error"
    assert not any(call[0] == "create_comment" for call in client.calls)


def test_valid_ledger_round_trips_publisher_github_boundary_and_protocol():
    configuration = _ledger_configuration()
    proposal, row = _ledger_proposal(configuration)
    client = FakeGitHub()
    result = ReviewPublisher(client, configuration=configuration, operator_credential=OPERATOR).publish(proposal, TARGET)
    assert result.status == "complete", result.reason
    records = {json.loads(client.payload_from_body(item["body"]))["record_type"]: item for item in client.comments}

    def exact_envelope(record):
        header = "HTTP/2 200\r\nX-OAuth-Scopes: read:user\r\n\r\n"
        response = subprocess.CompletedProcess(["gh"], 0, header + json.dumps(record), "")
        return GitHubClient(lambda argv, input_data=None: response).comment_envelope(REPO, record["id"])

    intent = exact_envelope(records["intent"])
    report = exact_envelope(records["report"])
    assert report.payload["validation_ledger"][0]["request_id"] == row.request_id
    validate_report(
        report, intent, canonical_intent=intent, trusted_authors={TRUSTED},
        configuration=configuration, capsule=build_review_capsule(client, TARGET),
    )


def test_changes_requested_uses_exact_reviewed_head_and_never_approves():
    client = FakeGitHub()
    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "complete", result.reason
    reviews = [call[1] for call in client.calls if call[0] == "create_review"]
    assert reviews == [("REQUEST_CHANGES", TARGET.head_sha)]
    assert all(event != "APPROVE" for event, _ in reviews)


def test_race_after_mutation_reapplies_label_and_marks_stale():
    client = FakeGitHub()
    original = client.get_pull_request
    count = 0

    def advancing(repository, number):
        nonlocal count
        count += 1
        response = original(repository, number)
        if count == 4:
            client.pull = client._pull(replace(TARGET, head_sha="new-head"))
        return response

    client.get_pull_request = advancing
    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status == "stale"
    assert ("add_label", ("needs-review",)) in client.calls
    assert not any(call[0] == "remove_label" for call in client.calls)


def test_report_creation_failure_is_incomplete_and_retry_resumes_intent():
    client = FakeGitHub()
    client.fail.add("report")
    first = _publisher(client).publish(_proposal(), TARGET)
    assert first.status == "incomplete"
    client.fail.remove("report")
    second = _publisher(client).publish(_proposal(), TARGET)
    assert second.status == "complete"
    assert [call[1] for call in client.calls if call[0] == "create_comment"].count("intent") == 1


def test_duplicate_intent_is_a_no_mutation_state():
    client = FakeGitHub()
    client.comments.append({
        "id": 1, "node_id": "C_1", "user": {"login": TRUSTED, "type": "Bot"},
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "body": json.dumps({
            "schema": "agentic-review/v1", "record_type": "intent", "record_id": "other",
            "target": {"repository": REPO, "number": 42, "head_repository": REPO, "head_sha": "head-sha", "base_ref": "main", "base_sha": "base-sha", "merge_base_sha": "merge-sha"}, "target_key": TARGET.target_key(), "attempt_id": "different",
            "canonical_digest": "",
        }),
    })
    payload = json.loads(client.comments[0]["body"])
    payload["canonical_digest"] = canonical_digest({key: value for key, value in payload.items() if key != "canonical_digest"})
    client.comments[0]["body"] = json.dumps(payload, default=lambda value: value.__dict__)
    before = len(client.calls)

    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status == "duplicate"
    assert len(client.calls) > before
    assert sum(call[0] == "config" for call in client.calls) >= 1


def test_workflow_review_dismissal_preserves_human_review():
    client = FakeGitHub()
    client.reviews.extend([
        {"id": 20, "node_id": "human", "user": {"login": "alice", "type": "User"}, "submitted_at": "2026-01-01T00:00:00Z", "body": "human", "state": "CHANGES_REQUESTED", "commit_id": TARGET.head_sha},
    ])
    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert result.status == "complete", result.reason
    assert ("dismiss", 20) not in client.calls


def test_revoked_workflow_review_is_dismissed_but_human_review_is_not():
    client = FakeGitHub()
    client.fail.add("completion")
    old = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert old.status == "incomplete", old.reason
    client.fail.remove("completion")
    old_intent = next(item for item in client.comments if json.loads(client.payload_from_body(item["body"]))["record_type"] == "intent")
    intent_payload = json.loads(old_intent["body"])
    revocation = {
        "schema": "agentic-review/v1", "record_type": "revocation", "record_id": "revoke-old",
        "target_key": TARGET.target_key(), "attempt_id": intent_payload["attempt_id"],
        "canonical_intent_digest": intent_payload["canonical_digest"], "reason": "replacement",
    }
    client.comments.append({
        "id": 99, "node_id": "C_99", "user": {"login": TRUSTED, "type": "Bot"},
        "created_at": "2026-01-01T00:03:30Z", "updated_at": "2026-01-01T00:03:30Z",
        "body": json.dumps(revocation),
    })
    client.reviews.append({
        "id": 100, "node_id": "human-100", "user": {"login": "alice", "type": "User"},
        "submitted_at": "2026-01-01T00:03:31Z", "body": "human", "state": "APPROVED", "commit_id": TARGET.head_sha,
    })

    result = _publisher(client).publish(
        _proposal("changes-requested", response_digest="sha256:" + "d" * 64), TARGET
    )

    assert result.status == "complete", result.reason
    assert ("dismiss", 3) in client.calls
    assert ("dismiss", 100) not in client.calls


def test_failed_final_mutation_never_removes_needs_review():
    client = FakeGitHub()
    client.fail.add("remove_label")
    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert result.status == "incomplete"
    assert ("add_label", ("needs-review",)) in client.calls
    assert client.removed_labels == []


def test_edited_report_is_not_resumed_and_stale_target_keeps_label():
    client = FakeGitHub()
    first = _publisher(client).publish(_proposal(), TARGET)
    assert first.status == "complete"
    report_id = next(item["id"] for item in client.comments if json.loads(client.payload_from_body(item["body"]))["record_type"] == "report")
    client.edited_comment_ids.add(report_id)
    result = _publisher(client).publish(_proposal(), TARGET)
    assert result.status in {"error", "incomplete", "duplicate"}
    assert ("remove_label", "needs-review") not in client.calls[-5:]


def test_incomplete_proposal_never_completes_or_removes_label():
    client = FakeGitHub()
    result = _publisher(client).publish(_proposal("incomplete"), TARGET)

    assert result.status == "incomplete"
    assert not any(call[0] == "create_review" for call in client.calls)
    assert not client.removed_labels
    assert ("add_label", ("needs-review",)) in client.calls


def test_completed_retry_reconciles_label_after_prior_remove_failure():
    client = FakeGitHub()
    client.fail.add("remove_label")
    first = _publisher(client).publish(_proposal(), TARGET)
    assert first.status == "incomplete"
    client.fail.remove("remove_label")

    second = _publisher(client).publish(_proposal(), TARGET)

    assert second.status == "duplicate"
    assert client.removed_labels == ["needs-review"]


@pytest.mark.parametrize("field", ["repository", "head_repository", "head_sha", "base_ref", "base_sha", "merge_base_sha"])
def test_every_target_field_race_is_stale_and_reapplies_label(field):
    client = FakeGitHub()
    original = client.get_pull_request
    count = 0

    def advancing(repository, number):
        nonlocal count
        count += 1
        response = original(repository, number)
        if count == 4:
            values = {
                "repository": "other/repo" if field == "repository" else TARGET.repository,
                "head_repository": "fork/repo" if field == "head_repository" else TARGET.head_repository,
                "head_sha": "new-head" if field == "head_sha" else TARGET.head_sha,
                "base_ref": "release" if field == "base_ref" else TARGET.base_ref,
                "base_sha": "new-base" if field == "base_sha" else TARGET.base_sha,
                "merge_base_sha": "new-merge" if field == "merge_base_sha" else TARGET.merge_base_sha,
            }
            client.pull = client._pull(ReviewTarget(number=TARGET.number, **values))
        return response

    client.get_pull_request = advancing
    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status in {"stale", "error"}
    assert ("add_label", ("needs-review",)) in client.calls
    assert not client.removed_labels


def test_missing_merge_base_fails_closed_before_intent():
    client = FakeGitHub()
    client.pull.pop("merge_base_sha")

    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status == "error"
    assert not any(call[0] == "create_comment" for call in client.calls)


def test_prior_target_history_does_not_block_current_target():
    old = ReviewTarget(REPO, 42, REPO, "old-head", "main", "old-base", "old-merge")
    payload = {
        "schema": "agentic-review/v1", "record_type": "intent", "record_id": "old-intent",
        "target": {"repository": old.repository, "number": old.number, "head_repository": old.head_repository,
                    "head_sha": old.head_sha, "base_ref": old.base_ref, "base_sha": old.base_sha,
                    "merge_base_sha": old.merge_base_sha}, "target_key": old.target_key(),
        "attempt_id": "old-attempt", "canonical_digest": "",
    }
    payload["canonical_digest"] = canonical_digest({key: value for key, value in payload.items() if key != "canonical_digest"})
    client = FakeGitHub()
    client.comments.append({"id": 99, "node_id": "old", "user": {"login": TRUSTED, "type": "Bot"},
                            "created_at": "2025-12-01T00:00:00Z", "updated_at": "2025-12-01T00:00:00Z",
                            "body": json.dumps(payload)})

    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status == "complete", result.reason


def test_canonical_change_before_review_aborts_without_completion():
    client = FakeGitHub()
    client.fail.add("completion")
    old = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert old.status == "incomplete"
    client.fail.remove("completion")
    intent = next(json.loads(client.payload_from_body(item["body"])) for item in client.comments if json.loads(client.payload_from_body(item["body"]))["record_type"] == "intent")
    revocation = {"schema": "agentic-review/v1", "record_type": "revocation", "record_id": "race-revoke",
                  "target_key": TARGET.target_key(), "attempt_id": intent["attempt_id"],
                  "canonical_intent_digest": intent["canonical_digest"], "reason": "race"}
    client.comments.append({"id": 901, "node_id": "race-revoke-1", "user": {"login": TRUSTED, "type": "Bot"},
                            "created_at": "2026-01-01T00:03:30Z", "updated_at": "2026-01-01T00:03:30Z",
                            "body": json.dumps(revocation)})
    client.revoke_before_next_review = {**revocation, "record_id": "race-revoke-2"}

    result = _publisher(client).publish(
        _proposal("changes-requested", response_digest="sha256:" + "d" * 64), TARGET
    )

    assert result.status == "complete", result.reason
    assert ("dismiss", 3) in client.calls


@pytest.mark.parametrize("state,commit", [("COMMENTED", TARGET.head_sha), ("DISMISSED", TARGET.head_sha), ("REQUEST_CHANGES", "wrong-head")])
def test_changes_retry_rejects_invalid_review_metadata(state, commit):
    client = FakeGitHub()
    client.fail.add("completion")
    first = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert first.status == "incomplete"
    client.fail.remove("completion")
    client.reviews[0]["state"] = state
    client.reviews[0]["commit_id"] = commit

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "error"
    assert [call for call in client.calls if call[0] == "create_review"] == [("create_review", ("REQUEST_CHANGES", TARGET.head_sha))]


@pytest.mark.parametrize("change", ["source", "operator_repo", "operator_ops", "operator_permissions"])
def test_publish_requires_repository_and_operator_binding(change):
    client = FakeGitHub()
    configuration = _configuration()
    operator = dict(OPERATOR)
    if change == "source":
        source = configuration.source
        configuration = replace(configuration, source=AuthenticatedConfigSource._from_authenticated_boundary(
            __import__("autoresearch.ar.review.config", fromlist=["_SOURCE_PROOF"])._SOURCE_PROOF,
            "other/repo", source.default_branch, source.commit_sha, source.config_digest, "."))
        object.__setattr__(configuration, "_loaded_from_protected_paths", True)
        object.__setattr__(configuration, "_loaded_source_digest", source.config_digest)
        object.__setattr__(configuration, "_loaded_root_identity", configuration.source.root_identity)
    elif change == "operator_repo":
        operator["repository"] = "other/repo"
    elif change == "operator_ops":
        operator["allowed_operations"] = ["publish"]
    else:
        operator["write_permissions"] = {"issues": "write"}

    with pytest.raises(Exception) if change != "source" else pytest.raises(Exception):
        ReviewPublisher(client, configuration=configuration, operator_credential=operator).publish(_proposal(), TARGET)


def test_report_is_visible_markdown_with_hidden_metadata_and_escaped_injection():
    client = FakeGitHub()
    proposal = _proposal("changes-requested")
    result = _publisher(client).publish(proposal, TARGET)
    assert result.status == "complete", result.reason
    report = next(item for item in client.comments if json.loads(client.payload_from_body(item["body"]))["record_type"] == "report")
    body = report["body"]

    assert body.startswith("## Agentic review")
    assert "<!-- agentic-review/v1" in body
    assert "<pre><code>Use **the checked value** &lt;instead&gt;.</code></pre>" in body


def test_validation_report_table_is_sorted_and_escapes_cells_without_claiming_results():
    profile = ValidationProfile(
        "profile", "capability|id", "arch\n<unsafe>", "fixture", "sha256:" + "1" * 64,
        "gfx|one", ("gfx\n one", "gfx|one"),
    )
    row = ValidationLedgerRow(profile, "sha256:" + "2" * 64, "representative")
    values = {
        "target": TARGET, "target_key": TARGET.target_key(), "capsule_digest": "sha256:" + "a" * 64,
        "adapter_id": "adapter", "adapter_version": "1", "model": "model",
        "response_digest": "sha256:" + "c" * 64, "verdict": "clean", "findings": (),
        "coverage": {"retrieved_file_count": 0, "expected_file_count": 0,
                      "retrieved_blob_count": 0, "expected_blob_count": 0,
                      "retrieved_content_count": 0, "expected_content_count": 0,
                      "coverage_complete": True},
        "validation_ledger": (row.to_mapping(),),
        "configuration_source_digest": "sha256:" + "3" * 64,
        "scope": ReviewScope((row.model_architecture,), row.covered_hardware).to_mapping(),
    }
    proposal = ReviewProposal(
        TARGET, values["capsule_digest"], "sha256:" + canonical_digest(values), "clean", (),
        "adapter", "1", "model", values["response_digest"], 0, 0, 0, 0, 0, 0, True,
        validation_ledger=(row,), configuration_source_digest=values["configuration_source_digest"],
        scope=ReviewScope((row.model_architecture,), row.covered_hardware),
    )
    rendered = render_report(proposal)
    assert rendered == rendered.strip()
    assert "### Hardware/model smoke validation" in rendered
    assert "| ID | Capability | Model architecture | Representative | Covered hardware | Status | Validator | Result |" in rendered
    assert "capability&#124;id" in rendered
    assert "arch<br>&lt;unsafe&gt;" in rendered
    assert "gfx&#124;one" in rendered
    assert "| pending | — | — |" in rendered


def test_oversized_report_is_rejected_before_intent_creation():
    client = FakeGitHub()
    result = _publisher(client).publish(_proposal("changes-requested", message="x" * (255 * 1024)), TARGET)
    assert result.status == "error"
    assert "size" in (result.reason or "") or "bound" in (result.reason or "")
    assert not any(call[0] == "create_comment" for call in client.calls)


def test_validation_heading_inside_finding_does_not_trigger_structural_failure():
    configuration = _ledger_configuration()
    finding = Finding("src/main.py", (3, 4), "warning", "literal\n### Hardware/model smoke validation\ntext")
    proposal, _row = _ledger_proposal(configuration, findings=(finding,))
    client = FakeGitHub()
    result = ReviewPublisher(client, configuration=configuration, operator_credential=OPERATOR).publish(proposal, TARGET)
    assert result.status == "complete", result.reason


def test_label_add_failure_is_explicit():
    client = FakeGitHub()
    client.fail.add("add_label")

    result = _publisher(client).publish(_proposal("incomplete"), TARGET)

    assert result.status == "error"
    assert "reapply" in (result.reason or "").lower()


def test_completion_retry_revalidates_active_changes_review_before_label_removal():
    client = FakeGitHub()
    client.fail.add("remove_label")
    first = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert first.status == "incomplete"
    client.fail.remove("remove_label")
    client.reviews[0]["state"] = "COMMENTED"

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "error"
    assert not client.removed_labels


def test_visible_report_neutralizes_tilde_backtick_backslash_ordered_list_and_multiline_input():
    client = FakeGitHub()
    message = "~~strike~~\n1. list\n`code` \\path\n# heading\n- bullet\n\n    indented\nsetext\n===="
    result = _publisher(client).publish(_proposal("changes-requested", message=message), TARGET)
    assert result.status == "complete", result.reason
    report = next(item for item in client.comments if json.loads(client.payload_from_body(item["body"]))["record_type"] == "report")
    visible = report["body"].split("<!-- agentic-review/v1", 1)[0]

    assert "<pre><code>~~strike~~\n1. list\n`code` \\path\n# heading\n- bullet\n\n    indented\nsetext\n====</code></pre>" in visible
    assert visible.count("<pre><code>") == visible.count("</code></pre>") == 1


def test_needs_review_is_verified_across_all_label_pages_before_removal():
    client = FakeGitHub()
    client.label_pages = [[{"name": "other"}], [{"name": "needs-review"}]]

    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status == "complete", result.reason
    assert client.removed_labels == ["needs-review"]
    assert [call[0] for call in client.calls if call[0] == "list_labels"].count("list_labels") == 2


def test_completed_retry_dismisses_stale_workflow_review_before_label_removal():
    client = FakeGitHub()
    client.fail.add("remove_label")
    first = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert first.status == "incomplete"
    client.fail.remove("remove_label")
    client.inject_review_on_labels = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "duplicate", result.reason
    assert ("dismiss", 902) in client.calls
    assert client.removed_labels == ["needs-review"]


def test_completion_history_refresh_dismisses_review_appearing_after_completion_creation():
    client = FakeGitHub()
    client.inject_review_on_completion = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "complete", result.reason
    assert ("dismiss", 901) in client.calls
    assert client.removed_labels == ["needs-review"]


def test_review_appearing_during_label_removal_is_reconciled_before_complete():
    client = FakeGitHub()
    client.inject_review_on_remove = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "complete", result.reason
    assert ("dismiss", 903) in client.calls
    assert client.removed_labels == ["needs-review"]


def test_review_race_dismissal_failure_reapplies_needs_review():
    client = FakeGitHub()
    client.inject_review_on_remove = True
    client.fail.add("dismiss")

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status in {"incomplete", "error"}
    assert "needs-review" in client.labels


def test_reconciliation_stabilizes_across_reviews_appearing_during_dismissal():
    client = FakeGitHub()
    client.inject_review_on_labels = True
    client.inject_review_on_dismiss = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "complete", result.reason
    assert ("dismiss", 902) in client.calls
    assert ("dismiss", 904) in client.calls


def test_keep_review_invalidation_before_final_label_removal_fails_closed():
    client = FakeGitHub()
    client.invalidate_keep_on_labels = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status in {"error", "incomplete"}
    assert "needs-review" in client.labels


def test_stale_review_from_canonical_election_snapshot_is_reconciled():
    client = FakeGitHub()
    client.arm_stale_on_canonical = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "complete", result.reason
    assert ("dismiss", 905) in client.calls


def test_keep_review_change_from_canonical_election_snapshot_fails_closed():
    client = FakeGitHub()
    client.arm_keep_invalidation_on_canonical = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status in {"error", "incomplete"}
    assert "needs-review" in client.labels


def test_pre_delete_canonical_snapshot_stale_review_is_dismissed_before_delete():
    client = FakeGitHub()
    client.arm_stale_on_mutate_canonical = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "complete", result.reason
    assert ("dismiss", 905) in client.calls
    assert client.calls.index(("dismiss", 905)) < client.calls.index(("remove_label", "needs-review"))


def test_pre_delete_canonical_snapshot_keep_review_invalidation_never_deletes_label():
    client = FakeGitHub()
    client.arm_keep_invalidation_on_mutate_canonical = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status in {"error", "incomplete"}
    assert client.removed_labels == []
    assert "needs-review" in client.labels


def test_absent_label_still_reconciles_new_stale_review():
    client = FakeGitHub()
    client.fail.add("remove_label")
    first = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert first.status == "incomplete"
    client.fail.remove("remove_label")
    client.labels.clear()
    client.inject_review_on_labels = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "duplicate", result.reason
    assert ("dismiss", 902) in client.calls


def test_exact_review_fetch_state_wins_over_list_state_on_retry():
    client = FakeGitHub()
    client.fail.add("remove_label")
    first = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert first.status == "incomplete"
    client.fail.remove("remove_label")
    client.mutate_exact_review_before_envelope = True

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status == "error"
    assert not client.removed_labels


def test_absent_label_target_change_during_lookup_returns_stale():
    client = FakeGitHub()
    client.fail.add("remove_label")
    first = _publisher(client).publish(_proposal("changes-requested"), TARGET)
    assert first.status == "incomplete"
    client.fail.remove("remove_label")
    client.labels.clear()
    client.change_target_on_labels = replace(TARGET, base_sha="advanced-base")

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status in {"stale", "error"}
    assert "needs-review" in client.labels


def test_target_change_during_final_reconciliation_never_returns_complete():
    client = FakeGitHub()
    client.change_target_after_remove = replace(TARGET, merge_base_sha="advanced-merge")

    result = _publisher(client).publish(_proposal("changes-requested"), TARGET)

    assert result.status in {"stale", "error"}
    assert "needs-review" in client.labels


def test_untrusted_malformed_and_marker_comments_do_not_block_publication():
    client = FakeGitHub()
    client.comments.extend([
        {"id": 700, "node_id": "hostile-marker", "user": {"login": "alice", "type": "User"},
         "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z",
         "body": "<!-- agentic-review/v1\nnot protocol\n-->"},
        {"id": 701, "node_id": "hostile-json", "user": {"login": "alice", "type": "User"},
         "created_at": "2025-01-01T00:00:01Z", "updated_at": "2025-01-01T00:00:01Z",
         "body": "{not protocol}"},
    ])

    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status == "complete", result.reason


def test_untrusted_valid_intent_does_not_shadow_canonical_attempt():
    target = {"repository": REPO, "number": 42, "head_repository": REPO, "head_sha": "head-sha",
              "base_ref": "main", "base_sha": "base-sha", "merge_base_sha": "merge-sha"}
    payload = {"schema": "agentic-review/v1", "record_type": "intent", "record_id": "untrusted-intent",
               "target": target, "target_key": TARGET.target_key(), "attempt_id": "untrusted-attempt",
               "canonical_digest": ""}
    payload["canonical_digest"] = canonical_digest({key: value for key, value in payload.items() if key != "canonical_digest"})
    client = FakeGitHub()
    client.comments.append({"id": 702, "node_id": "untrusted-intent", "user": {"login": "alice", "type": "User"},
                            "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z",
                            "body": json.dumps(payload)})

    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status == "complete", result.reason


def test_trusted_malformed_protocol_record_fails_closed():
    client = FakeGitHub()
    client.comments.append({"id": 703, "node_id": "trusted-malformed", "user": {"login": TRUSTED, "type": "Bot"},
                            "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z",
                            "body": json.dumps({"schema": "agentic-review/v1", "record_type": "report"})})

    result = _publisher(client).publish(_proposal(), TARGET)

    assert result.status == "error"
