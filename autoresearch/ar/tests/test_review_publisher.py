# Copyright (c) Kaden Schutt
"""Contract tests for the authenticated, SHA-bound review publisher."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from autoresearch.ar.review.canonical import canonical_digest, metadata_digest
from autoresearch.ar.review.config import AuthenticatedConfigSource, ReviewConfiguration
from autoresearch.ar.review.github import GitHubResponse
from autoresearch.ar.review.models import Finding, GitHubEnvelope, ReviewProposal, ReviewTarget
from autoresearch.ar.review.publisher import ReviewPublisher


REPO = "owner/repo"
TARGET = ReviewTarget(REPO, 42, REPO, "head-sha", "main", "base-sha", "merge-sha")
TRUSTED = "review-bot"
OPERATOR = {
    "schema": "hipfire.agentic-review.operator-credentials",
    "version": 1,
    "repository": REPO,
    "principal": {"login": TRUSTED, "type": "Bot"},
    "allowed_operations": ["publish", "dismiss-workflow-review"],
    "write_permissions": {"issues": "write", "pull_requests": "write"},
    "credential_attestation_digest": "sha256:" + "a" * 64,
}


def _configuration() -> ReviewConfiguration:
    source = AuthenticatedConfigSource._from_authenticated_boundary(
        __import__("autoresearch.ar.review.config", fromlist=["_SOURCE_PROOF"])._SOURCE_PROOF,
        REPO,
        "main",
        "config-sha",
        "sha256:" + "b" * 64,
        ".",
    )
    configuration = ReviewConfiguration(
        {},
        {},
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": []},
        source,
    )
    object.__setattr__(configuration, "_loaded_from_protected_paths", True)
    object.__setattr__(configuration, "_loaded_source_digest", source.config_digest)
    object.__setattr__(configuration, "_loaded_root_identity", source.root_identity)
    return configuration


def _proposal(verdict: str = "clean", response_digest: str = "sha256:" + "c" * 64,
              message: str = "Use **the checked value** <instead>.") -> ReviewProposal:
    findings = () if verdict == "clean" else (
        Finding("src/main.py", (3, 4), "error", message),
    )
    values = {
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "capsule_digest": "sha256:" + "a" * 64,
        "adapter_id": "adapter",
        "adapter_version": "1",
        "model": "model",
        "response_digest": response_digest,
        "verdict": verdict,
        "findings": findings,
        "coverage": {
            "retrieved_file_count": 0,
            "expected_file_count": 0,
            "retrieved_blob_count": 0,
            "expected_blob_count": 0,
            "retrieved_content_count": 0,
            "expected_content_count": 0,
            "coverage_complete": True,
        },
    }
    return ReviewProposal(
        TARGET,
        values["capsule_digest"],
        "sha256:" + canonical_digest(values),
        verdict,
        findings,
        "adapter",
        "1",
        "model",
        values["response_digest"],
        0, 0, 0, 0, 0, 0, True,
    )


class FakeGitHub:
    def __init__(self) -> None:
        self.pull = self._pull(TARGET)
        self.comments: list[dict] = []
        self.reviews: list[dict] = []
        self.calls: list[tuple[str, object]] = []
        self.next_id = 1
        self.clock = 0
        self.fail: set[str] = set()
        self.removed_labels: list[str] = []
        self.labels = {"needs-review"}
        self.label_pages: list[list[dict]] | None = None
        self.mutate_head_after: str | None = None
        self.revoke_before_next_review: dict | None = None
        self.inject_review_on_completion = False
        self.inject_review_on_labels = False
        self.inject_review_on_remove = False
        self.inject_review_on_dismiss = False
        self.invalidate_keep_on_labels = False
        self.change_target_on_labels: ReviewTarget | None = None
        self.change_target_after_remove: ReviewTarget | None = None
        self.change_target_on_history_read: ReviewTarget | None = None
        self.change_target_on_history_read_at: int | None = None
        self.mutate_exact_review_before_envelope = False
        self.arm_stale_on_canonical = False
        self.arm_keep_invalidation_on_canonical = False
        self.arm_stale_on_mutate_canonical = False
        self.arm_keep_invalidation_on_mutate_canonical = False
        self.history_reads = 0
        self.inject_stale_on_history_read: int | None = None
        self.invalidate_keep_on_history_read: int | None = None
        self.transient_stale_on_history_read: int | None = None
        self.transient_keep_on_history_read: int | None = None
        self.transient_records: dict[int, dict] = {}
        self.transient_review_states: dict[int, str] = {}
        self.deleted_comment_ids: set[int] = set()
        self.edited_comment_ids: set[int] = set()

    def _now(self) -> str:
        self.clock += 1
        return f"2026-01-01T00:{self.clock:02d}:00Z"

    @staticmethod
    def _pull(target: ReviewTarget) -> dict:
        return {
            "id": 1,
            "node_id": "PR_1",
            "number": target.number,
            "head": {"repo": {"full_name": target.head_repository}, "sha": target.head_sha},
            "base": {"repo": {"full_name": target.repository}, "ref": target.base_ref, "sha": target.base_sha},
            "merge_base_sha": target.merge_base_sha,
        }

    def get_pull_request(self, repository: str, number: int) -> GitHubResponse:
        self.calls.append(("get_target", self.pull["head"]["sha"]))
        return GitHubResponse(self.pull, {}, 200)

    def get_review_target(self, repository: str, number: int) -> ReviewTarget:
        data = self.get_pull_request(repository, number).data
        return ReviewTarget(
            data["base"]["repo"]["full_name"], data["number"], data["head"]["repo"]["full_name"],
            data["head"]["sha"], data["base"]["ref"], data["base"]["sha"], data["merge_base_sha"],
        )

    def revalidate_config_source(self, source) -> None:
        self.calls.append(("config", source.commit_sha))

    def list_issue_comments(self, repository: str, number: int) -> GitHubResponse:
        self.calls.append(("list_comments", None))
        return GitHubResponse([comment for comment in self.comments if comment["id"] not in self.deleted_comment_ids], {}, 200)

    def list_pull_reviews(self, repository: str, number: int) -> GitHubResponse:
        self.calls.append(("list_reviews", None))
        self.history_reads += 1
        if self.change_target_on_history_read is not None and self.change_target_on_history_read_at == self.history_reads:
            self.pull = self._pull(self.change_target_on_history_read)
            self.change_target_on_history_read = None
            self.change_target_on_history_read_at = None
        if self.transient_stale_on_history_read == self.history_reads - 1:
            self.transient_records.pop(905, None)
            self.transient_stale_on_history_read = None
        if self.transient_keep_on_history_read == self.history_reads - 1:
            self.transient_review_states.clear()
            self.transient_keep_on_history_read = None
        reviews = list(self.reviews)
        if self.inject_stale_on_history_read == self.history_reads and self.reviews:
            stale = deepcopy(self.reviews[0])
            stale["id"] = 905
            stale["node_id"] = "stale-in-canonical-history"
            stale_payload = json.loads(self.payload_from_body(stale["body"]))
            stale_payload["record_id"] = "stale-in-canonical-history"
            stale_payload["metadata_digest"] = metadata_digest(stale_payload)
            stale["body"] = json.dumps(stale_payload)
            self.reviews.append(stale)
            self.inject_stale_on_history_read = None
        if self.invalidate_keep_on_history_read == self.history_reads and self.reviews:
            self.reviews[0]["state"] = "DISMISSED"
            self.invalidate_keep_on_history_read = None
        if self.transient_stale_on_history_read == self.history_reads:
            stale = deepcopy(self.reviews[0])
            stale["id"] = 905
            stale["node_id"] = "stale-in-canonical-history"
            stale_payload = json.loads(self.payload_from_body(stale["body"]))
            stale_payload["record_id"] = "stale-in-canonical-history"
            stale_payload["metadata_digest"] = metadata_digest(stale_payload)
            stale["body"] = json.dumps(stale_payload)
            self.transient_records[905] = stale
            reviews.append(stale)
        if self.transient_keep_on_history_read == self.history_reads and reviews:
            reviews[0] = {**reviews[0], "state": "DISMISSED"}
            self.transient_review_states[reviews[0]["id"]] = "DISMISSED"
        return GitHubResponse(reviews, {}, 200)

    def list_issue_labels(self, repository: str, number: int) -> GitHubResponse:
        self.calls.append(("list_labels", None))
        if self.change_target_on_labels is not None:
            self.pull = self._pull(self.change_target_on_labels)
            self.change_target_on_labels = None
        if self.invalidate_keep_on_labels and self.reviews:
            self.reviews[0]["state"] = "DISMISSED"
            self.invalidate_keep_on_labels = False
        if self.inject_review_on_labels and self.reviews:
            stale = deepcopy(self.reviews[0])
            stale["id"] = 902
            stale["node_id"] = "stale-before-label"
            stale_payload = json.loads(self.payload_from_body(stale["body"]))
            stale_payload["record_id"] = "stale-before-label"
            stale_payload["metadata_digest"] = metadata_digest(stale_payload)
            stale["body"] = json.dumps(stale_payload)
            self.reviews.append(stale)
            self.inject_review_on_labels = False
        if self.label_pages is None:
            return GitHubResponse([{"name": label} for label in sorted(self.labels)], {}, 200)
        self.calls.extend(("list_labels", None) for _ in self.label_pages[1:])
        return GitHubResponse([label for page in self.label_pages for label in page], {}, 200)

    @staticmethod
    def payload_from_body(body: str) -> str:
        if body.lstrip().startswith("{"):
            return body
        marker = "<!-- agentic-review/v1"
        start = body.index(marker) + len(marker)
        return body[start:].split("-->", 1)[0].strip()

    def _envelope(self, record: dict, kind: str) -> GitHubEnvelope:
        if record["id"] in self.deleted_comment_ids:
            raise RuntimeError("record deleted")
        updated = record["updated_at"] if "updated_at" in record else record["submitted_at"]
        if record["id"] in self.edited_comment_ids:
            updated = "2026-01-01T00:09:00Z"
        published = record["created_at"] if "created_at" in record else record["submitted_at"]
        user = record.get("user", {})
        return GitHubEnvelope(
            json.loads(self.payload_from_body(record["body"])), record["node_id"],
            user.get("login", TRUSTED), published, updated, user.get("type", "Bot"),
            record.get("app_id"), record.get("installation_id"), record.get("repository_id"),
            record.get("credential_attestation_digest"),
        )

    def comment_envelope(self, repository: str, comment_id: int) -> GitHubEnvelope:
        return self._envelope(next(item for item in self.comments if item["id"] == comment_id), "comment")

    def review_envelope(self, repository: str, number: int, review_id: int) -> GitHubEnvelope:
        records = [*self.reviews, *self.transient_records.values()]
        return self._envelope(next(item for item in records if item["id"] == review_id), "review")

    def get_pull_review(self, repository: str, number: int, review_id: int) -> GitHubResponse:
        return GitHubResponse(next(item for item in self.reviews if item["id"] == review_id), {}, 200)

    def get_pull_review_record(self, repository: str, number: int, review_id: int):
        records = [*self.reviews, *self.transient_records.values()]
        record = next(item for item in records if item["id"] == review_id)
        if review_id in self.transient_review_states:
            record = {**record, "state": self.transient_review_states[review_id]}
        if self.mutate_exact_review_before_envelope:
            record["state"] = "DISMISSED"
            self.mutate_exact_review_before_envelope = False
        return SimpleNamespace(
            envelope=self._envelope(record, "review"),
            state=record["state"],
            commit_id=record["commit_id"],
            server_id=record["id"],
        )

    def create_issue_comment(self, repository: str, number: int, body: str) -> GitHubResponse:
        record_type = json.loads(self.payload_from_body(body))["record_type"]
        self.calls.append(("create_comment", record_type))
        if "comment" in self.fail or record_type in self.fail:
            raise RuntimeError("comment creation failed")
        now = self._now()
        record = {
            "id": self.next_id, "node_id": f"C_{self.next_id}", "user": {"login": TRUSTED, "type": "Bot"},
            "created_at": now, "updated_at": now, "body": body,
        }
        self.next_id += 1
        self.comments.append(record)
        if record_type == "completion" and self.arm_stale_on_canonical:
            self.transient_stale_on_history_read = self.history_reads + 2
            self.arm_stale_on_canonical = False
        if record_type == "completion" and self.arm_keep_invalidation_on_canonical:
            self.transient_keep_on_history_read = self.history_reads + 2
            self.arm_keep_invalidation_on_canonical = False
        if record_type == "completion" and self.arm_stale_on_mutate_canonical:
            self.inject_stale_on_history_read = self.history_reads + 5
            self.arm_stale_on_mutate_canonical = False
        if record_type == "completion" and self.arm_keep_invalidation_on_mutate_canonical:
            self.invalidate_keep_on_history_read = self.history_reads + 5
            self.arm_keep_invalidation_on_mutate_canonical = False
        if record_type == "completion" and self.inject_review_on_completion and self.reviews:
            stale = deepcopy(self.reviews[0])
            stale["id"] = 901
            stale["node_id"] = "stale-after-completion"
            stale_payload = json.loads(self.payload_from_body(stale["body"]))
            stale_payload["record_id"] = "stale-after-completion"
            stale_payload["metadata_digest"] = metadata_digest(stale_payload)
            stale["body"] = json.dumps(stale_payload)
            self.reviews.append(stale)
            self.inject_review_on_completion = False
        return GitHubResponse(record, {}, 201)

    def create_pull_request_review(self, repository: str, number: int, *, body: str, event: str, commit_id: str) -> GitHubResponse:
        self.calls.append(("create_review", (event, commit_id)))
        if "review" in self.fail:
            raise RuntimeError("review creation failed")
        if self.revoke_before_next_review is not None:
            now = "2026-01-01T00:10:00Z"
            self.comments.append({"id": 900, "node_id": "race-revoke", "user": {"login": TRUSTED, "type": "Bot"},
                                  "created_at": now, "updated_at": now,
                                  "body": json.dumps(self.revoke_before_next_review)})
            self.revoke_before_next_review = None
        now = self._now()
        record = {
            "id": self.next_id, "node_id": f"R_{self.next_id}", "user": {"login": TRUSTED, "type": "Bot"},
            "submitted_at": now, "body": body, "state": "CHANGES_REQUESTED", "commit_id": commit_id,
        }
        self.next_id += 1
        self.reviews.append(record)
        return GitHubResponse(record, {}, 201)

    def add_labels(self, repository: str, number: int, labels) -> GitHubResponse:
        self.calls.append(("add_label", tuple(labels)))
        if "add_label" in self.fail:
            raise RuntimeError("label add failed")
        self.labels.update(labels)
        return GitHubResponse([], {}, 200)

    def remove_label(self, repository: str, number: int, label: str) -> GitHubResponse:
        self.calls.append(("remove_label", label))
        if "remove_label" in self.fail:
            raise RuntimeError("label removal failed")
        self.removed_labels.append(label)
        self.labels.discard(label)
        if self.change_target_after_remove is not None:
            self.change_target_on_history_read = self.change_target_after_remove
            self.change_target_on_history_read_at = self.history_reads + 2
            self.change_target_after_remove = None
        if self.inject_review_on_remove and self.reviews:
            stale = deepcopy(self.reviews[0])
            stale["id"] = 903
            stale["node_id"] = "stale-during-remove"
            stale_payload = json.loads(self.payload_from_body(stale["body"]))
            stale_payload["record_id"] = "stale-during-remove"
            stale_payload["metadata_digest"] = metadata_digest(stale_payload)
            stale["body"] = json.dumps(stale_payload)
            self.reviews.append(stale)
            self.inject_review_on_remove = False
        return GitHubResponse({}, {}, 204)

    def dismiss_workflow_review(self, repository: str, number: int, review_id: int, *, message: str) -> GitHubResponse:
        self.calls.append(("dismiss", review_id))
        if "dismiss" in self.fail:
            raise RuntimeError("dismissal failed")
        for review in self.reviews:
            if review["id"] == review_id:
                review["state"] = "DISMISSED"
        self.reviews = [review for review in self.reviews if review["id"] != review_id]
        self.transient_records.pop(review_id, None)
        if self.inject_review_on_dismiss and self.reviews:
            stale = deepcopy(self.reviews[0])
            stale["id"] = 904
            stale["node_id"] = "stale-during-dismiss"
            stale_payload = json.loads(self.payload_from_body(stale["body"]))
            stale_payload["record_id"] = "stale-during-dismiss"
            stale_payload["metadata_digest"] = metadata_digest(stale_payload)
            stale["body"] = json.dumps(stale_payload)
            self.reviews.append(stale)
            self.inject_review_on_dismiss = False
        return GitHubResponse({"id": review_id, "node_id": f"D_{review_id}"}, {}, 200)


def _publisher(client: FakeGitHub) -> ReviewPublisher:
    return ReviewPublisher(client, configuration=_configuration(), operator_credential=OPERATOR)


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
    assert len(client.calls) == before + 4  # config, target, and the two bounded history reads


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
