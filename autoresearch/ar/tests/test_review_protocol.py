# Copyright (c) Kaden Schutt
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from autoresearch.ar.review.canonical import canonical_digest, canonical_json, canonical_loads, metadata_digest
from autoresearch.ar.review.models import ReviewTarget
from autoresearch.ar.review.protocol import (
    elect_canonical_attempt,
    validate_completion,
    validate_intent,
    validate_protocol,
    validate_report,
    validate_review_metadata,
    validate_revocation,
)


ROOT = Path(__file__).parents[3]
VECTORS = json.loads((Path(__file__).parent / "fixtures" / "review_protocol_vectors.json").read_text())
TARGET = ReviewTarget("owner/repo", 42, "owner/repo", "head-sha", "main", "base-sha", "merge-sha")
TRUSTED = {"review-bot"}


def _intent(*, node="intent-a", attempt="attempt-a", created="2026-01-01T00:00:00Z", target=TARGET):
    intent = {
        "record_type": "intent",
        "record_id": node,
        "intent_node_id": node,
        "target": target,
        "target_key": target.target_key(),
        "attempt_id": attempt,
        "author": "review-bot",
        "created_at": created,
        "canonical_digest": "",
    }
    intent["canonical_digest"] = canonical_digest(
        {key: value for key, value in intent.items() if key != "canonical_digest"}
    )
    return intent


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


def test_jcs_integer_boundaries_and_limits_are_fixed_regressions():
    assert canonical_json(VECTORS["regressions"]["safe_integer_min"]) == b"-9007199254740991"
    assert canonical_json(VECTORS["regressions"]["safe_integer_max"]) == b"9007199254740991"
    for value in VECTORS["regressions"]["unsafe_integers"]:
        with pytest.raises(ValueError, match="IEEE-754|safe range"):
            canonical_json(value)
    with pytest.raises(ValueError, match="byte limit"):
        canonical_json(
            VECTORS["regressions"]["overflow_value"],
            max_bytes=VECTORS["regressions"]["overflow_limit"],
        )


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
    validate_completion(_completion(intent), intent, report=report, canonical_intent=intent, trusted_authors=TRUSTED)


def test_pending_intent_digest_is_rejected():
    intent = _intent()
    intent["canonical_digest"] = "pending"
    with pytest.raises(ValueError, match="digest"):
        validate_intent(intent, trusted_authors=TRUSTED)


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
        validate_completion(
            _completion(intent), intent, report=None, canonical_intent=intent, trusted_authors=TRUSTED
        )


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
        "canonical_intent_digest": intent["canonical_digest"],
        "metadata_digest": "pending",
    }
    metadata["metadata_digest"] = metadata_digest(metadata)
    validate_review_metadata(metadata, intent, report, canonical_intent=intent, trusted_authors=TRUSTED)
    metadata["author"] = "untrusted"
    with pytest.raises(ValueError, match="trusted"):
        validate_review_metadata(metadata, intent, report, canonical_intent=intent, trusted_authors=TRUSTED)
    metadata["author"] = "review-bot"
    metadata["head_sha"] = "other-sha"
    with pytest.raises(ValueError, match="SHA|sha"):
        validate_review_metadata(metadata, intent, report, canonical_intent=intent, trusted_authors=TRUSTED)


def test_completion_propagates_trust_and_binds_complete_canonical_intent():
    intent = _intent()
    intent["canonical_digest"] = validate_intent(intent, trusted_authors=TRUSTED)
    report = _report(intent)
    completion = _completion(intent)
    report["author"] = "untrusted"
    with pytest.raises(ValueError, match="trusted"):
        validate_completion(completion, intent, report=report, canonical_intent=intent, trusted_authors=TRUSTED)

    report["author"] = "review-bot"
    altered_target = ReviewTarget(
        "owner/repo", 42, "owner/repo", "different-head", "main", "base-sha", "merge-sha"
    )
    canonical = dict(intent, target=altered_target, target_key=altered_target.target_key())
    canonical["canonical_digest"] = canonical_digest(
        {key: value for key, value in canonical.items() if key != "canonical_digest"}
    )
    with pytest.raises(ValueError, match="canonical|target"):
        validate_completion(completion, intent, report=report, canonical_intent=canonical, trusted_authors=TRUSTED)


@pytest.mark.parametrize(
    "field, value",
    [
        ("target_key", "other-target"),
        ("attempt_id", "other-attempt"),
        ("intent_node_id", "other-node"),
        ("head_sha", "other-head"),
        ("canonical_intent_digest", "other-digest"),
    ],
)
def test_completion_rejects_each_canonical_binding_mismatch(field, value):
    intent = _intent()
    report = _report(intent)
    completion = _completion(intent)
    completion[field] = value
    with pytest.raises(ValueError):
        validate_completion(completion, intent, report=report, canonical_intent=intent, trusted_authors=TRUSTED)


def test_election_tie_breaks_by_node_id_and_rejects_noncanonical_completion():
    first = _intent(node="z-node", attempt="z", created="2026-01-01T00:00:00Z")
    second = _intent(node="a-node", attempt="a", created="2026-01-01T00:00:00Z")
    first["canonical_digest"] = validate_intent(first, trusted_authors=TRUSTED)
    second["canonical_digest"] = validate_intent(second, trusted_authors=TRUSTED)
    selected = elect_canonical_attempt([first, second], [], expected_target=TARGET, trusted_authors=TRUSTED)
    assert selected["intent_node_id"] == "a-node"
    with pytest.raises(ValueError, match="canonical"):
        validate_completion(
            _completion(first), first, report=_report(first), canonical_intent=second, trusted_authors=TRUSTED
        )


def test_election_normalizes_aware_timestamps_before_node_tie_breaking():
    earlier_utc = _intent(
        node="z-node", attempt="z", created=VECTORS["regressions"]["aware_timestamp_early"]
    )
    later_utc = _intent(node="a-node", attempt="a", created=VECTORS["regressions"]["aware_timestamp_late"])
    earlier_utc["canonical_digest"] = validate_intent(earlier_utc, trusted_authors=TRUSTED)
    later_utc["canonical_digest"] = validate_intent(later_utc, trusted_authors=TRUSTED)
    assert elect_canonical_attempt(
        [earlier_utc, later_utc], [], expected_target=TARGET, trusted_authors=TRUSTED
    ) is earlier_utc


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError, match="timezone"):
        validate_intent(_intent(created="2026-01-01T00:00:00"), trusted_authors=TRUSTED)


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


def test_revocation_must_target_current_canonical_intent_in_event_order():
    canonical = _intent(node="canonical", attempt="canonical")
    replacement = _intent(node="replacement", attempt="replacement")
    canonical["canonical_digest"] = validate_intent(canonical, trusted_authors=TRUSTED)
    replacement["canonical_digest"] = validate_intent(replacement, trusted_authors=TRUSTED)
    noncanonical_revocation = {
        "record_type": "revocation",
        "record_id": "revoke-replacement",
        "target_key": replacement["target_key"],
        "attempt_id": replacement["attempt_id"],
        "canonical_intent_digest": replacement["canonical_digest"],
        "author": "review-bot",
        "reason": "replacement",
        "authenticated": True,
        "created_at": "2026-01-01T00:04:00Z",
    }
    with pytest.raises(ValueError, match="canonical"):
        elect_canonical_attempt(
            [canonical, replacement], [], expected_target=TARGET,
            revocations=[noncanonical_revocation], trusted_authors=TRUSTED
        )


def test_election_requires_one_expected_target_and_rejects_mixed_inputs():
    other_target = ReviewTarget("other/repo", 7, "other/repo", "other-head", "main", "base", "merge")
    with pytest.raises(ValueError, match="target"):
        elect_canonical_attempt(
            [_intent(), _intent(node="other", attempt="other", target=other_target)],
            [],
            expected_target=TARGET,
            trusted_authors=TRUSTED,
        )


def test_event_order_sorts_out_of_order_inputs_and_rejects_timestamp_before_intent():
    first = _intent(node="first", attempt="first", created="2026-01-01T00:00:00Z")
    second = _intent(node="second", attempt="second", created="2026-01-01T00:02:00Z")
    first_revocation = {
        "record_type": "revocation", "record_id": "revoke-first", "target_key": first["target_key"],
        "attempt_id": first["attempt_id"], "canonical_intent_digest": first["canonical_digest"],
        "author": "review-bot", "reason": "replace", "authenticated": True,
        "created_at": "2026-01-01T00:01:00Z",
    }
    assert elect_canonical_attempt(
        [second, first], [], expected_target=TARGET, revocations=[first_revocation], trusted_authors=TRUSTED
    ) is second
    first_revocation["created_at"] = "2025-12-31T23:00:00Z"
    with pytest.raises(ValueError, match="canonical"):
        elect_canonical_attempt(
            [first, second], [], expected_target=TARGET,
            revocations=[first_revocation], trusted_authors=TRUSTED
        )


def test_later_revocation_does_not_invalidate_historical_completion():
    first = _intent(node="first", attempt="first")
    second = _intent(node="second", attempt="second", created="2026-01-01T00:03:00Z")
    first_report = _report(first)
    completion = _completion(first)
    first_revocation = {
        "record_type": "revocation", "record_id": "revoke-first", "target_key": first["target_key"],
        "attempt_id": first["attempt_id"], "canonical_intent_digest": first["canonical_digest"],
        "author": "review-bot", "reason": "replace", "authenticated": True,
        "created_at": "2026-01-01T00:04:00Z",
    }
    selected = elect_canonical_attempt(
        [second, first], [completion], expected_target=TARGET, reports=[first_report],
        revocations=[first_revocation], trusted_authors=TRUSTED
    )
    assert selected is second


def test_protocol_requires_trust_and_validates_metadata_against_elected_intent():
    with pytest.raises(ValueError, match="trusted"):
        elect_canonical_attempt([_intent()], [], expected_target=TARGET)
    with pytest.raises(ValueError, match="trusted"):
        validate_protocol([], expected_target=TARGET, trusted_authors=[])


def test_protocol_rejects_metadata_for_noncanonical_intent():
    canonical = _intent(node="canonical", attempt="canonical")
    noncanonical = _intent(node="noncanonical", attempt="noncanonical", created="2026-01-01T00:01:00Z")
    report = _report(noncanonical)
    metadata = {
        "record_type": "review-metadata",
        "record_id": "metadata-noncanonical",
        "target": noncanonical["target"],
        "target_key": noncanonical["target_key"],
        "attempt_id": noncanonical["attempt_id"],
        "intent_node_id": noncanonical["intent_node_id"],
        "head_sha": TARGET.head_sha,
        "author": "review-bot",
        "created_at": "2026-01-01T00:02:00Z",
        "report_id": report["record_id"],
        "report_body_sha256": report["report_body_sha256"],
        "canonical_intent_digest": noncanonical["canonical_digest"],
        "metadata_digest": "",
    }
    metadata["metadata_digest"] = metadata_digest(metadata)
    with pytest.raises(ValueError, match="canonical"):
        validate_protocol(
            [canonical, noncanonical, report, metadata], expected_target=TARGET, trusted_authors=TRUSTED
        )
