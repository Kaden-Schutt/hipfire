# Copyright (c) Kaden Schutt
import hashlib
import json
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
)


VECTORS = json.loads((Path(__file__).parent / "fixtures" / "review_protocol_vectors.json").read_text())
TARGET = ReviewTarget("owner/repo", 42, "owner/repo", "head-sha", "main", "base-sha", "merge-sha")
TRUSTED = {"review-bot"}


def _self_digest(payload, field):
    payload[field] = canonical_digest({key: value for key, value in payload.items() if key != field})
    return payload


def _envelope(payload, node_id, *, author="review-bot", created_at="2026-01-01T00:00:00Z"):
    envelope = {
        "payload": payload,
        "payload_digest": canonical_digest(payload),
        "node_id": node_id,
        "author": author,
        "created_at": created_at,
    }
    envelope["envelope_digest"] = canonical_digest(
        {key: envelope[key] for key in ("author", "created_at", "node_id", "payload_digest")}
    )
    return envelope


def _refresh_envelope(envelope):
    envelope["payload_digest"] = canonical_digest(envelope["payload"])
    envelope["envelope_digest"] = canonical_digest(
        {key: envelope[key] for key in ("author", "created_at", "node_id", "payload_digest")}
    )
    return envelope


def _intent(record_id="intent-a", node_id="gh-intent-a", *, created_at="2026-01-01T00:00:00Z"):
    payload = {
        "record_type": "intent",
        "record_id": record_id,
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "attempt_id": "attempt-" + record_id,
        "canonical_digest": "",
    }
    return _envelope(_self_digest(payload, "canonical_digest"), node_id, created_at=created_at)


def _report(intent, record_id=None, node_id="gh-report-a", *, created_at="2026-01-01T00:01:00Z", body="report body"):
    payload = {
        "record_type": "report",
        "record_id": record_id or "report-" + intent["payload"]["record_id"],
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "attempt_id": intent["payload"]["attempt_id"],
        "intent_record_id": intent["payload"]["record_id"],
        "canonical_intent_node_id": intent["node_id"],
        "canonical_intent_digest": intent["payload"]["canonical_digest"],
        "head_sha": TARGET.head_sha,
        "report_body": body,
        "report_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    return _envelope(payload, node_id, created_at=created_at)


def _metadata(intent, report, node_id="gh-metadata-a", *, created_at="2026-01-01T00:02:00Z", record_id="metadata-a"):
    report_payload = report["payload"]
    payload = {
        "record_type": "review-metadata",
        "record_id": record_id,
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "attempt_id": intent["payload"]["attempt_id"],
        "intent_record_id": intent["payload"]["record_id"],
        "head_sha": TARGET.head_sha,
        "report_record_id": report_payload["record_id"],
        "report_node_id": report["node_id"],
        "report_digest": report["payload_digest"],
        "report_body_sha256": report_payload["report_body_sha256"],
        "canonical_intent_digest": intent["payload"]["canonical_digest"],
        "canonical_intent_node_id": intent["node_id"],
        "metadata_digest": "",
    }
    return _envelope(_self_digest(payload, "metadata_digest"), node_id, created_at=created_at)


def _completion(intent, report, metadata, node_id="gh-completion-a", *, created_at="2026-01-01T00:03:00Z"):
    payload = {
        "record_type": "completion",
        "record_id": "completion-" + intent["payload"]["record_id"],
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "attempt_id": intent["payload"]["attempt_id"],
        "intent_record_id": intent["payload"]["record_id"],
        "head_sha": TARGET.head_sha,
        "canonical_intent_digest": intent["payload"]["canonical_digest"],
        "canonical_intent_node_id": intent["node_id"],
        "report_record_id": report["payload"]["record_id"],
        "report_node_id": report["node_id"],
        "report_digest": report["payload_digest"],
        "metadata_record_id": metadata["payload"]["record_id"],
        "metadata_digest": metadata["payload"]["metadata_digest"],
    }
    return _envelope(payload, node_id, created_at=created_at)


def _revocation(intent, node_id="gh-revoke-a", *, created_at="2026-01-01T00:04:00Z"):
    payload = {
        "record_type": "revocation",
        "record_id": "revocation-" + intent["payload"]["record_id"],
        "target_key": TARGET.target_key(),
        "attempt_id": intent["payload"]["attempt_id"],
        "canonical_intent_digest": intent["payload"]["canonical_digest"],
        "reason": "replacement",
    }
    return _envelope(payload, node_id, created_at=created_at)


def test_jcs_vectors_cover_reordered_keys_controls_utf16_and_safe_numbers():
    for vector in VECTORS["canonical"]:
        encoded = canonical_json(vector["value"])
        assert encoded == vector["canonical_utf8"].encode()
        assert hashlib.sha256(encoded).hexdigest() == vector["sha256"]
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert canonical_json(VECTORS["regressions"]["safe_integer_max"]) == b"9007199254740991"
    assert canonical_json(VECTORS["regressions"]["safe_integer_min"]) == b"-9007199254740991"
    for value in VECTORS["regressions"]["unsafe_integers"]:
        with pytest.raises(ValueError, match="safe range"):
            canonical_json(value)
    metadata_vector = VECTORS["metadata"][0]
    assert metadata_digest(metadata_vector["value"]) == metadata_vector["digest"]


def test_canonical_json_rejects_duplicate_keys_nonfinite_and_limits():
    with pytest.raises(ValueError, match="duplicate"):
        canonical_loads('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="finite"):
        canonical_json(float("inf"))
    with pytest.raises(ValueError, match="byte limit"):
        canonical_json("abcd", max_bytes=3)


def test_envelopes_bind_payload_and_do_not_accept_spoofed_server_facts():
    intent = _intent()
    validate_intent(intent, trusted_authors=TRUSTED)
    tampered = dict(intent, node_id="spoofed")
    with pytest.raises(ValueError, match="envelope"):
        validate_intent(tampered, trusted_authors=TRUSTED)
    tampered = _intent()
    tampered["payload"]["author"] = "attacker"
    tampered["payload_digest"] = canonical_digest(tampered["payload"])
    with pytest.raises(ValueError, match="server|payload"):
        validate_intent(tampered, trusted_authors=TRUSTED)
    tampered = _intent()
    tampered["payload_digest"] = "0" * 64
    with pytest.raises(ValueError, match="payload digest"):
        validate_intent(tampered, trusted_authors=TRUSTED)


def test_post_publication_envelope_facts_are_authenticated_and_trusted():
    intent = _intent()
    with pytest.raises(ValueError, match="trusted"):
        validate_intent(dict(intent, author="untrusted"), trusted_authors=TRUSTED)
    with pytest.raises(ValueError, match="envelope"):
        validate_intent(dict(intent, created_at="2025-01-01T00:00:00Z"), trusted_authors=TRUSTED)
    with pytest.raises(ValueError, match="envelope"):
        validate_intent(dict(intent, node_id="different-node"), trusted_authors=TRUSTED)


def test_valid_history_binds_report_metadata_and_completion_to_envelopes():
    intent = _intent()
    report = _report(intent)
    metadata = _metadata(intent, report)
    completion = _completion(intent, report, metadata)
    validate_report(report, intent, canonical_intent=intent, trusted_authors=TRUSTED)
    validate_review_metadata(metadata, intent, report, canonical_intent=intent, trusted_authors=TRUSTED)
    validate_completion(
        completion,
        intent,
        report,
        metadata,
        canonical_intent=intent,
        trusted_authors=TRUSTED,
    )


def test_completion_requires_canonical_intent_earlier_report_and_metadata():
    intent = _intent()
    report = _report(intent)
    metadata = _metadata(intent, report)
    completion = _completion(intent, report, metadata)
    with pytest.raises(TypeError, match="canonical_intent"):
        validate_completion(completion, intent, report, metadata, trusted_authors=TRUSTED)
    with pytest.raises(ValueError, match="metadata"):
        validate_completion(completion, intent, report, None, canonical_intent=intent, trusted_authors=TRUSTED)
    with pytest.raises(ValueError, match="report"):
        validate_completion(completion, intent, None, metadata, canonical_intent=intent, trusted_authors=TRUSTED)


def test_metadata_digest_and_completion_references_are_verified():
    intent = _intent()
    report = _report(intent)
    metadata = _metadata(intent, report)
    bad_metadata = dict(metadata, payload=dict(metadata["payload"], metadata_digest="0" * 64))
    bad_metadata["payload_digest"] = canonical_digest(bad_metadata["payload"])
    bad_metadata["envelope_digest"] = canonical_digest(
        {key: bad_metadata[key] for key in ("author", "created_at", "node_id", "payload_digest")}
    )
    with pytest.raises(ValueError, match="metadata digest"):
        validate_review_metadata(bad_metadata, intent, report, canonical_intent=intent, trusted_authors=TRUSTED)
    completion = _completion(intent, report, metadata)
    bad_completion = dict(completion, payload=dict(completion["payload"], metadata_digest="wrong"))
    bad_completion["payload_digest"] = canonical_digest(bad_completion["payload"])
    bad_completion["envelope_digest"] = canonical_digest(
        {key: bad_completion[key] for key in ("author", "created_at", "node_id", "payload_digest")}
    )
    with pytest.raises(ValueError, match="metadata"):
        validate_completion(
            bad_completion, intent, report, metadata, canonical_intent=intent, trusted_authors=TRUSTED
        )


def test_protocol_rejects_pre_intent_records_and_noncanonical_publication():
    intent = _intent(created_at="2026-01-01T00:02:00Z")
    report = _report(intent, created_at="2026-01-01T00:01:00Z")
    with pytest.raises(ValueError, match="before|intent"):
        validate_protocol([report, intent], expected_target=TARGET, trusted_authors=TRUSTED)
    later_report = _report(intent, node_id="gh-report-later", created_at="2026-01-01T00:03:00Z")
    early_metadata = _metadata(intent, later_report, created_at="2026-01-01T00:01:00Z")
    with pytest.raises(ValueError, match="before|intent"):
        validate_protocol([early_metadata, later_report, intent], expected_target=TARGET, trusted_authors=TRUSTED)

    first = _intent(record_id="intent-first", node_id="node-first")
    second = _intent(record_id="intent-second", node_id="node-second", created_at="2026-01-01T00:01:00Z")
    noncanonical_report = _report(second, created_at="2026-01-01T00:02:00Z")
    with pytest.raises(ValueError, match="canonical"):
        validate_protocol([first, second, noncanonical_report], expected_target=TARGET, trusted_authors=TRUSTED)


def test_historical_report_metadata_and_completion_survive_replacement():
    first = _intent(record_id="intent-first", node_id="node-first")
    report = _report(first)
    metadata = _metadata(first, report)
    completion = _completion(first, report, metadata)
    second = _intent(
        record_id="intent-second", node_id="node-second", created_at="2026-01-01T00:04:00Z"
    )
    revocation = _revocation(first, created_at="2026-01-01T00:05:00Z")
    selected = validate_protocol(
        [revocation, metadata, second, completion, report, first],
        expected_target=TARGET,
        trusted_authors=TRUSTED,
    )
    assert selected["payload"]["record_id"] == "intent-second"


def test_duplicate_logical_ids_and_node_ids_rejected_before_lookup():
    first = _intent(record_id="same", node_id="node-first")
    second = _intent(record_id="same", node_id="node-second")
    with pytest.raises(ValueError, match="logical|record ID|duplicate"):
        validate_protocol([first, second], expected_target=TARGET, trusted_authors=TRUSTED)
    duplicate_attempt = _intent(record_id="different", node_id="node-different")
    duplicate_attempt["payload"]["attempt_id"] = first["payload"]["attempt_id"]
    duplicate_attempt["payload"]["canonical_digest"] = canonical_digest(
        {key: value for key, value in duplicate_attempt["payload"].items() if key != "canonical_digest"}
    )
    duplicate_attempt["payload_digest"] = canonical_digest(duplicate_attempt["payload"])
    duplicate_attempt["envelope_digest"] = canonical_digest(
        {key: duplicate_attempt[key] for key in ("author", "created_at", "node_id", "payload_digest")}
    )
    with pytest.raises(ValueError, match="attempt"):
        elect_canonical_attempt(
            [first, duplicate_attempt], [], expected_target=TARGET, trusted_authors=TRUSTED
        )
    duplicate_node = _intent(record_id="other", node_id="node-first")
    with pytest.raises(ValueError, match="node"):
        elect_canonical_attempt(
            [first, duplicate_node], [], expected_target=TARGET, trusted_authors=TRUSTED
        )


def test_equal_timestamp_total_order_uses_envelope_node_id_and_is_input_order_independent():
    first = _intent(record_id="intent-a", node_id="a-node")
    second = _intent(record_id="intent-z", node_id="z-node")
    selected = elect_canonical_attempt(
        [first, second], [], expected_target=TARGET, trusted_authors=TRUSTED
    )
    reordered = elect_canonical_attempt(
        [second, first], [], expected_target=TARGET, trusted_authors=TRUSTED
    )
    assert selected["node_id"] == reordered["node_id"] == "a-node"


def test_payload_logical_id_is_distinct_from_authenticated_node_id():
    intent = _intent(record_id="logical-intent", node_id="github-node-123")
    assert intent["payload"]["record_id"] != intent["node_id"]
    validate_intent(intent, trusted_authors=TRUSTED)


def test_exact_and_invalid_intent_payload_digests_are_checked():
    intent = _intent()
    assert validate_intent(intent, trusted_authors=TRUSTED) == intent["payload"]["canonical_digest"]
    invalid = _refresh_envelope(dict(intent, payload=dict(intent["payload"], canonical_digest="wrong")))
    with pytest.raises(ValueError, match="intent canonical digest"):
        validate_intent(invalid, trusted_authors=TRUSTED)


def test_report_body_ids_and_head_sha_are_bound_to_canonical_intent():
    intent = _intent()
    report = _report(intent)
    validate_report(report, intent, canonical_intent=intent, trusted_authors=TRUSTED)
    altered_body = _refresh_envelope(dict(report, payload=dict(report["payload"], report_body="altered")))
    with pytest.raises(ValueError, match="body"):
        validate_report(altered_body, intent, canonical_intent=intent, trusted_authors=TRUSTED)
    for field, value in (
        ("intent_record_id", "other-intent"),
        ("attempt_id", "other-attempt"),
        ("target_key", "other-target"),
        ("head_sha", "other-head"),
        ("canonical_intent_node_id", "other-node"),
        ("canonical_intent_digest", "other-digest"),
    ):
        altered = _refresh_envelope(dict(report, payload=dict(report["payload"], **{field: value})))
        with pytest.raises(ValueError):
            validate_report(altered, intent, canonical_intent=intent, trusted_authors=TRUSTED)


def test_completion_canonical_target_attempt_and_intent_bindings_are_field_exact():
    intent = _intent()
    report = _report(intent)
    metadata = _metadata(intent, report)
    completion = _completion(intent, report, metadata)
    for field, value in (
        ("target_key", "other-target"),
        ("attempt_id", "other-attempt"),
        ("intent_record_id", "other-intent"),
        ("canonical_intent_node_id", "other-node"),
        ("canonical_intent_digest", "other-digest"),
        ("head_sha", "other-head"),
    ):
        altered = _refresh_envelope(dict(completion, payload=dict(completion["payload"], **{field: value})))
        with pytest.raises(ValueError):
            validate_completion(
                altered, intent, report, metadata, canonical_intent=intent, trusted_authors=TRUSTED
            )


def test_aware_offset_ordering_and_naive_timestamps_are_checked():
    earlier_utc = _intent(record_id="z", node_id="z-node", created_at="2026-01-01T00:30:00+02:00")
    later_utc = _intent(record_id="a", node_id="a-node", created_at="2025-12-31T23:00:00Z")
    assert elect_canonical_attempt(
        [later_utc, earlier_utc], [], expected_target=TARGET, trusted_authors=TRUSTED
    ) is earlier_utc
    naive = _intent(created_at="2026-01-01T00:00:00")
    with pytest.raises(ValueError, match="timezone"):
        validate_intent(naive, trusted_authors=TRUSTED)


def test_invalid_and_noncanonical_revocations_are_rejected():
    first = _intent(record_id="first", node_id="first-node")
    second = _intent(record_id="second", node_id="second-node", created_at="2026-01-01T00:01:00Z")
    invalid = _revocation(first)
    invalid = _refresh_envelope(dict(invalid, payload=dict(invalid["payload"], canonical_intent_digest="wrong")))
    with pytest.raises(ValueError, match="canonical|digest"):
        elect_canonical_attempt(
            [first, second], [], expected_target=TARGET, revocations=[invalid], trusted_authors=TRUSTED
        )
    noncanonical = _revocation(second)
    with pytest.raises(ValueError, match="canonical"):
        elect_canonical_attempt(
            [first, second], [], expected_target=TARGET, revocations=[noncanonical], trusted_authors=TRUSTED
        )


def test_report_metadata_completion_all_propagate_envelope_trust():
    intent = _intent()
    report = _report(intent)
    metadata = _metadata(intent, report)
    completion = _completion(intent, report, metadata)
    for record, validator in (
        (dict(report, author="untrusted"), lambda item: validate_report(item, intent, canonical_intent=intent, trusted_authors=TRUSTED)),
        (dict(metadata, author="untrusted"), lambda item: validate_review_metadata(item, intent, report, canonical_intent=intent, trusted_authors=TRUSTED)),
        (dict(completion, author="untrusted"), lambda item: validate_completion(item, intent, report, metadata, canonical_intent=intent, trusted_authors=TRUSTED)),
    ):
        with pytest.raises(ValueError, match="trusted"):
            validator(record)


def test_all_record_types_use_one_timestamp_then_node_id_event_order():
    timestamp = "2026-01-01T00:00:00Z"
    first = _intent(record_id="intent-first", node_id="a-node", created_at=timestamp)
    report = _report(first, node_id="b-node", created_at=timestamp)
    metadata = _metadata(first, report, node_id="c-node", created_at=timestamp)
    completion = _completion(first, report, metadata, node_id="d-node", created_at=timestamp)
    revocation = _revocation(first, node_id="e-node", created_at=timestamp)
    replacement = _intent(record_id="intent-replacement", node_id="f-node", created_at=timestamp)
    selected = validate_protocol(
        [completion, replacement, revocation, metadata, report, first],
        expected_target=TARGET,
        trusted_authors=TRUSTED,
    )
    assert selected["node_id"] == "f-node"
