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
    validate_report(report, intent, trusted_authors=TRUSTED)
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
