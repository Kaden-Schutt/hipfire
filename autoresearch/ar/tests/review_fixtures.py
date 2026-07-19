# Copyright (c) Kaden Schutt
"""Shared review test fixtures; not owned by an individual test module."""


from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


from autoresearch.ar.review.canonical import canonical_digest, metadata_digest
from autoresearch.ar.review.config import AuthenticatedConfigSource, ReviewConfiguration
from autoresearch.ar.review.github import GitHubResponse
from autoresearch.ar.review.capsule import build_review_capsule, capsule_coverage
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

_HEAD_SOURCE = "def main():\n    return 'head'\n"
_HEAD_BLOB = hashlib.sha1(
    b"blob " + str(len(_HEAD_SOURCE.encode())).encode() + b"\0" + _HEAD_SOURCE.encode()
).hexdigest()
_BASE_TREE = "base-tree"
_HEAD_TREE = "head-tree"


def _fixture_capsule():
    return build_review_capsule(FakeGitHub(), TARGET)


def _configuration() -> ReviewConfiguration:
    source = AuthenticatedConfigSource._from_authenticated_boundary(
        __import__("autoresearch.ar.review.config", fromlist=["_SOURCE_PROOF"])._SOURCE_PROOF,
        REPO,
        "main",
        "config-sha",
        "sha256:" + "b" * 64,
        ".",
    )
    capabilities = json.loads(
        (Path(__file__).parents[3] / ".github/agentic-review/capabilities-v1.json").read_text()
    )
    configuration = ReviewConfiguration(
        {},
        capabilities,
        {"schema": "hipfire.agentic-review.trusted-publishers", "version": 1, "apps": [{
            "app_id": 1, "login": TRUSTED, "installation_id": 2, "repository_id": 8,
            "credential_attestation_digest": OPERATOR["credential_attestation_digest"],
        }]},
        source,
    )
    object.__setattr__(configuration, "_loaded_from_protected_paths", True)
    object.__setattr__(configuration, "_loaded_source_digest", source.config_digest)
    object.__setattr__(configuration, "_loaded_root_identity", source.root_identity)
    return configuration


def _proposal(verdict: str = "clean", response_digest: str = "sha256:" + "c" * 64,
              message: str = "Use **the checked value** <instead>.", *, capsule=None) -> ReviewProposal:
    findings = () if verdict == "clean" else (
        Finding("src/main.py", (3, 4), "error", message),
    )
    configuration = _configuration()
    capsule = capsule or _fixture_capsule()
    profile = ValidationProfile.from_mapping(configuration.capabilities["profiles"][0])
    capability = next(item for item in configuration.capabilities["capabilities"] if item["id"] == profile.capability_id)
    row = ValidationLedgerRow(profile, capability_contract_digest(capability), "representative")
    scope = ReviewScope((profile.model_architecture,), profile.covered_hardware)
    values = {
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "capsule_digest": capsule.digest,
        "adapter_id": "adapter",
        "adapter_version": "1",
        "model": "model",
        "response_digest": response_digest,
        "verdict": verdict,
        "findings": findings,
        "validation_ledger": (row.to_mapping(),),
        "configuration_source_digest": configuration.source.config_digest,
        "scope": scope.to_mapping(),
        "coverage": capsule_coverage(capsule),
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
        values["coverage"]["retrieved_file_count"], values["coverage"]["expected_file_count"],
        values["coverage"]["retrieved_blob_count"], values["coverage"]["expected_blob_count"],
        values["coverage"]["retrieved_content_count"], values["coverage"]["expected_content_count"],
        values["coverage"]["coverage_complete"],
        validation_ledger=(row,), configuration_source_digest=configuration.source.config_digest, scope=scope,
    )


def _exemption_configuration() -> ReviewConfiguration:
    policy = json.loads(
        (Path(__file__).parents[3] / ".github/agentic-review/capabilities-v1.json").read_text()
    )
    policy["exemptions"] = [{"id": "docs", "path_globs": ["docs/**"]}]
    base = _configuration()
    result = ReviewConfiguration(base.providers, policy, base.trusted_publishers, base.source)
    object.__setattr__(result, "_loaded_from_protected_paths", True)
    object.__setattr__(result, "_loaded_source_digest", result.source.config_digest)
    object.__setattr__(result, "_loaded_root_identity", result.source.root_identity)
    return result


def _exempt_proposal(capsule_digest: str | None = None, *, capsule=None,
                     exemption_paths: tuple[str, ...] = ("docs/review.md",)) -> ReviewProposal:
    capsule_digest = capsule.digest if capsule is not None and capsule_digest is None else (
        capsule_digest or "sha256:" + "a" * 64
    )
    coverage = capsule_coverage(capsule) if capsule is not None else {
        "retrieved_file_count": 0, "expected_file_count": 0,
        "retrieved_blob_count": 0, "expected_blob_count": 0,
        "retrieved_content_count": 0, "expected_content_count": 0,
        "coverage_complete": True,
    }
    values = {
        "target": TARGET, "target_key": TARGET.target_key(), "capsule_digest": capsule_digest,
        "adapter_id": "adapter", "adapter_version": "1", "model": "model",
        "response_digest": "sha256:" + "c" * 64, "verdict": "clean", "findings": (),
        "coverage": coverage,
        "validation_ledger": (), "configuration_source_digest": "sha256:" + "b" * 64,
        "exemption_ids": ("docs",), "exemption_paths": exemption_paths,
        "scope": ReviewScope((), ()).to_mapping(),
    }
    return ReviewProposal(
        TARGET, capsule_digest, "sha256:" + canonical_digest(values), "clean", (),
        "adapter", "1", "model", values["response_digest"],
        coverage["retrieved_file_count"], coverage["expected_file_count"],
        coverage["retrieved_blob_count"], coverage["expected_blob_count"],
        coverage["retrieved_content_count"], coverage["expected_content_count"],
        coverage["coverage_complete"],
        configuration_source_digest=values["configuration_source_digest"],
        exemption_ids=values["exemption_ids"], exemption_paths=values["exemption_paths"],
        scope=ReviewScope((), ()),
    )


def _ledger_configuration() -> ReviewConfiguration:
    return _exemption_configuration()


def _ledger_proposal(configuration: ReviewConfiguration, *, findings=()) -> tuple[ReviewProposal, ValidationLedgerRow]:
    profile = ValidationProfile.from_mapping(configuration.capabilities["profiles"][0])
    capability = next(item for item in configuration.capabilities["capabilities"] if item["id"] == profile.capability_id)
    row = ValidationLedgerRow(profile, capability_contract_digest(capability), "representative")
    capsule = _fixture_capsule()
    values = {
        "target": TARGET, "target_key": TARGET.target_key(), "capsule_digest": capsule.digest,
        "adapter_id": "adapter", "adapter_version": "1", "model": "model",
        "response_digest": "sha256:" + "c" * 64, "verdict": "clean", "findings": findings,
        "coverage": capsule_coverage(capsule),
        "validation_ledger": (row.to_mapping(),),
        "configuration_source_digest": configuration.source.config_digest,
        "scope": ReviewScope((row.model_architecture,), row.covered_hardware).to_mapping(),
    }
    return ReviewProposal(
        TARGET, values["capsule_digest"], "sha256:" + canonical_digest(values), "clean", findings,
        "adapter", "1", "model", values["response_digest"],
        values["coverage"]["retrieved_file_count"], values["coverage"]["expected_file_count"],
        values["coverage"]["retrieved_blob_count"], values["coverage"]["expected_blob_count"],
        values["coverage"]["retrieved_content_count"], values["coverage"]["expected_content_count"],
        values["coverage"]["coverage_complete"],
        validation_ledger=(row,), configuration_source_digest=values["configuration_source_digest"],
        scope=ReviewScope((row.model_architecture,), row.covered_hardware),
    ), row


class FakeGitHub:
    def __init__(self, *, empty_diff: bool = False, changed_path: str = "src/main.py") -> None:
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
        self.empty_diff = empty_diff
        self.changed_path = changed_path
        self.commits = {
            (TARGET.repository, TARGET.merge_base_sha): {"sha": TARGET.merge_base_sha, "tree": {"sha": _BASE_TREE}},
            (TARGET.head_repository, TARGET.head_sha): {"sha": TARGET.head_sha, "tree": {"sha": _HEAD_TREE}},
        }
        self.trees = {
            (TARGET.repository, _BASE_TREE): {"sha": _BASE_TREE, "tree": [], "truncated": False},
            (TARGET.head_repository, _HEAD_TREE): {
                "sha": _HEAD_TREE,
                "tree": [{"path": self.changed_path, "mode": "100644", "type": "blob", "sha": _HEAD_BLOB}],
                "truncated": False,
            },
        }
        self.blobs = {
            (TARGET.head_repository, _HEAD_BLOB): {
                "sha": _HEAD_BLOB,
                "size": len(_HEAD_SOURCE.encode()),
                "encoding": "base64",
                "content": base64.b64encode(_HEAD_SOURCE.encode()).decode(),
            },
        }

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

    def get_commit(self, repository: str, sha: str) -> GitHubResponse:
        self.calls.append(("get_commit", (repository, sha)))
        return GitHubResponse(self.commits[(repository, sha)], {}, 200)

    def get_tree(self, repository: str, sha: str, *, recursive: bool = False) -> GitHubResponse:
        self.calls.append(("get_tree", (repository, sha, recursive)))
        tree = self.trees[(repository, sha)]
        if self.empty_diff and sha == _HEAD_TREE:
            tree = {**tree, "tree": []}
        return GitHubResponse(tree, {}, 200)

    def get_blob(self, repository: str, sha: str) -> GitHubResponse:
        self.calls.append(("get_blob", (repository, sha)))
        return GitHubResponse(self.blobs[(repository, sha)], {}, 200)

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
            user.get("login", TRUSTED), published, updated, user.get("type", "Bot")
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
