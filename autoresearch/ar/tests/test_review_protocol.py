# Copyright (c) Kaden Schutt
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from autoresearch.ar.review.canonical import canonical_json, canonical_loads, metadata_digest
from autoresearch.ar.review.models import ReviewTarget
from autoresearch.ar.review.protocol import (
    elect_canonical_attempt,
    validate_completion,
    validate_intent,
    validate_report,
    validate_review_metadata,
    validate_revocation,
)


ROOT = Path(__file__).parents[3]
VECTORS = json.loads((Path(__file__).parent / "fixtures" / "review_protocol_vectors.json").read_text())
TARGET = ReviewTarget("owner/repo", 42, "owner/repo", "head-sha", "main", "base-sha", "merge-sha")
TRUSTED = {"review-bot"}


def _intent(*, node="intent-a", attempt="attempt-a", created="2026-01-01T00:00:00Z"):
    return {
        "record_type": "intent",
        "record_id": node,
        "intent_node_id": node,
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "attempt_id": attempt,
        "author": "review-bot",
        "created_at": created,
        "canonical_digest": "pending",
    }


def _report(intent, body="report body"):
    return {
        "record_type": "report",
        "record_id": "report-" + intent["attempt_id"],
        "target": intent["target"],
        "target_key": intent["target_key"],
        "attempt_id": intent["attempt_id"],
        "intent_node_id": intent["intent_node_id"],
        "head_sha": TARGET.head_sha,
        "author": "review-bot",
        "created_at": "2026-01-01T00:01:00Z",
        "report_body": body,
        "report_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }


def _completion(intent):
    return {
        "record_type": "completion",
        "record_id": "completion-" + intent["attempt_id"],
        "target": intent["target"],
        "target_key": intent["target_key"],
        "attempt_id": intent["attempt_id"],
        "intent_node_id": intent["intent_node_id"],
        "head_sha": TARGET.head_sha,
        "author": "review-bot",
        "created_at": "2026-01-01T00:02:00Z",
        "canonical_intent_digest": intent["canonical_digest"],
        "report_id": "report-" + intent["attempt_id"],
    }


def test_canonical_vectors_and_reordered_keys():
    for vector in VECTORS["canonical"]:
        assert canonical_json(vector["value"]) == vector["canonical_utf8"].encode()
        assert hashlib.sha256(canonical_json(vector["value"])).hexdigest() == vector["sha256"]

    value = {"b": 2, "a": 1}
    assert canonical_json(value) == canonical_json({"a": 1, "b": 2})


def test_duplicate_json_keys_and_unsupported_values_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        canonical_loads('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="finite"):
        canonical_json(float("inf"))
    with pytest.raises(ValueError, match="unsupported"):
        canonical_json({1: "not a string key"})


def test_metadata_digest_excludes_its_own_field_and_uses_report_digest():
    vector = VECTORS["metadata"][0]
    metadata = deepcopy(vector["value"])
    assert metadata_digest(metadata) == vector["digest"]
    metadata["metadata_digest"] = "sha256:" + "f" * 64
    assert metadata_digest(metadata) == vector["digest"]


def test_valid_intent_report_and_completion_bind_full_target():
    intent = _intent()
    intent["canonical_digest"] = validate_intent(intent, trusted_authors=TRUSTED)
    report = _report(intent)
    validate_report(report, intent, trusted_authors=TRUSTED)
    validate_completion(_completion(intent), intent, report=report, canonical_intent=intent)


@pytest.mark.parametrize("field", ["target_key", "attempt_id", "intent_node_id"])
def test_altered_intent_ids_are_rejected(field):
    intent = _intent()
    intent["canonical_digest"] = validate_intent(intent, trusted_authors=TRUSTED)
    altered = _report(intent)
    altered[field] = "altered"
    with pytest.raises(ValueError):
        validate_report(altered, intent, trusted_authors=TRUSTED)


def test_altered_body_and_deleted_report_reference_are_rejected():
    intent = _intent()
    intent["canonical_digest"] = validate_intent(intent, trusted_authors=TRUSTED)
    report = _report(intent)
    validate_report(report, intent, trusted_authors=TRUSTED)
    altered = deepcopy(report)
    altered["report_body"] = "tampered"
    with pytest.raises(ValueError, match="body"):
        validate_report(altered, intent, trusted_authors=TRUSTED)
    with pytest.raises(ValueError, match="report"):
        validate_completion(_completion(intent), intent, report=None, canonical_intent=intent)


def test_review_metadata_rejects_untrusted_author_and_mismatching_sha():
    intent = _intent()
    intent["canonical_digest"] = validate_intent(intent, trusted_authors=TRUSTED)
    report = _report(intent)
    validate_report(report, intent, trusted_authors=TRUSTED)
    metadata = {
        "record_type": "review-metadata",
        "record_id": "metadata-1",
        "target": intent["target"],
        "target_key": intent["target_key"],
        "attempt_id": intent["attempt_id"],
        "intent_node_id": intent["intent_node_id"],
        "head_sha": TARGET.head_sha,
        "author": "review-bot",
        "created_at": "2026-01-01T00:03:00Z",
        "report_id": report["record_id"],
        "report_body_sha256": report["report_body_sha256"],
        "metadata_digest": "pending",
    }
    metadata["metadata_digest"] = metadata_digest(metadata)
    validate_review_metadata(metadata, intent, report, trusted_authors=TRUSTED)
    metadata["author"] = "untrusted"
    with pytest.raises(ValueError, match="trusted"):
        validate_review_metadata(metadata, intent, report, trusted_authors=TRUSTED)
    metadata["author"] = "review-bot"
    metadata["head_sha"] = "other-sha"
    with pytest.raises(ValueError, match="SHA|sha"):
        validate_review_metadata(metadata, intent, report, trusted_authors=TRUSTED)


def test_election_tie_breaks_by_node_id_and_rejects_noncanonical_completion():
    first = _intent(node="z-node", attempt="z", created="2026-01-01T00:00:00Z")
    second = _intent(node="a-node", attempt="a", created="2026-01-01T00:00:00Z")
    first["canonical_digest"] = validate_intent(first, trusted_authors=TRUSTED)
    second["canonical_digest"] = validate_intent(second, trusted_authors=TRUSTED)
    selected = elect_canonical_attempt([first, second], [], trusted_authors=TRUSTED)
    assert selected["intent_node_id"] == "a-node"
    with pytest.raises(ValueError, match="canonical"):
        validate_completion(_completion(first), first, report=_report(first), canonical_intent=second)


def test_invalid_revocation_cannot_replace_an_attempt():
    intent = _intent()
    intent["canonical_digest"] = validate_intent(intent, trusted_authors=TRUSTED)
    bad = {
        "record_type": "revocation",
        "record_id": "revoke-1",
        "target_key": intent["target_key"],
        "attempt_id": intent["attempt_id"],
        "canonical_intent_digest": "wrong",
        "author": "review-bot",
        "reason": "replacement",
        "authenticated": True,
        "created_at": "2026-01-01T00:04:00Z",
    }
    with pytest.raises(ValueError, match="digest"):
        validate_revocation(bad, intent, trusted_authors=TRUSTED)
