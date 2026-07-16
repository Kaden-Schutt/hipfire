# Copyright (c) Kaden Schutt
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.ar.review.canonical import canonical_digest, canonical_json, canonical_loads, metadata_digest
from autoresearch.ar.review.models import ReviewTarget
from autoresearch.ar.review.models import GitHubEnvelope
from autoresearch.ar.review.protocol import (
    elect_canonical_attempt,
    validate_append_only,
    validate_completion,
    validate_intent,
    validate_protocol,
    validate_report,
    validate_review_metadata,
    validate_revocation,
)


VECTORS = json.loads((Path(__file__).parent / "fixtures" / "review_protocol_vectors.json").read_text())
TARGET = ReviewTarget("owner/repo", 42, "owner/repo", "head-sha", "main", "base-sha", "merge-sha")
TRUSTED = {"review-bot"}


def _self_digest(payload, field):
    payload[field] = canonical_digest({key: value for key, value in payload.items() if key != field})
    return payload


def _envelope(payload, node_id, *, author="review-bot", created_at="2026-01-01T00:00:00Z"):
    return GitHubEnvelope(payload=payload, node_id=node_id, author=author, created_at=created_at)


def _refresh_envelope(envelope):
    if isinstance(envelope, GitHubEnvelope):
        return envelope
    return GitHubEnvelope(
        payload=envelope["payload"],
        node_id=envelope["node_id"],
        author=envelope["author"],
        created_at=envelope["created_at"],
    )


def _payload_digest(envelope):
    return canonical_digest(envelope.payload)


def _intent(
    record_id="intent-a",
    node_id="gh-intent-a",
    *,
    created_at="2026-01-01T00:00:00Z",
    target=TARGET,
):
    payload = {
        "record_type": "intent",
        "record_id": record_id,
        "target": target,
        "target_key": target.target_key(),
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
        "report_digest": _payload_digest(report),
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
        "report_digest": _payload_digest(report),
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
    for vector in VECTORS["regressions"]["floats"]:
        encoded = canonical_json(vector["value"])
        assert encoded == vector["canonical_utf8"].encode()
        assert hashlib.sha256(encoded).hexdigest() == vector["sha256"]
    metadata_vector = VECTORS["metadata"][0]
    assert metadata_digest(metadata_vector["value"]) == metadata_vector["digest"]


def test_canonical_json_rejects_duplicate_keys_nonfinite_and_limits():
    with pytest.raises(ValueError, match="duplicate"):
        canonical_loads('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="finite"):
        canonical_json(float("inf"))
    with pytest.raises(ValueError, match="byte limit"):
        canonical_json("abcd", max_bytes=3)
    with pytest.raises(ValueError, match="surrogate|Unicode"):
        canonical_json("\ud800")
    with pytest.raises(ValueError, match="malformed|surrogate|Unicode"):
        canonical_loads('"\\ud800"')
    with pytest.raises(ValueError, match="malformed|Unicode"):
        canonical_loads(b'"\xed\xa0\x80"')


def test_trusted_authors_requires_a_collection_of_complete_identities():
    with pytest.raises(ValueError, match="trusted_authors"):
        validate_intent(_intent(), trusted_authors="review-bot")
    with pytest.raises(ValueError, match="trusted_authors"):
        validate_intent(_intent(), trusted_authors=b"review-bot")
    with pytest.raises(ValueError, match="trusted_authors"):
        validate_intent(_intent(), trusted_authors=["", "review-bot"])


def test_direct_intent_and_revocation_validators_require_nonempty_fields():
    intent = _intent()
    intent_payload = dict(intent.payload, attempt_id="")
    intent_payload["canonical_digest"] = canonical_digest(
        {key: value for key, value in intent_payload.items() if key != "canonical_digest"}
    )
    intent = replace(intent, payload=intent_payload)
    with pytest.raises(ValueError, match="attempt_id"):
        validate_intent(intent, trusted_authors=TRUSTED)
    valid_intent = _intent()
    revocation = _revocation(valid_intent)
    revocation = replace(revocation, payload=dict(revocation.payload, reason=""))
    with pytest.raises(ValueError, match="reason"):
        validate_revocation(revocation, valid_intent, trusted_authors=TRUSTED)


def test_envelopes_bind_payload_and_do_not_accept_spoofed_server_facts():
    intent = _intent()
    validate_intent(intent, trusted_authors=TRUSTED)
    tampered = dict(intent, node_id="spoofed")
    with pytest.raises(ValueError, match="typed GitHubEnvelope"):
        validate_intent(tampered, trusted_authors=TRUSTED)
    tampered = _intent()
    tampered = _refresh_envelope(dict(tampered, payload=dict(tampered["payload"], author="attacker")))
    with pytest.raises(ValueError, match="server|payload"):
        validate_intent(tampered, trusted_authors=TRUSTED)
    tampered = _intent()
    with pytest.raises(ValueError, match="typed GitHubEnvelope"):
        validate_intent(dict(tampered), trusted_authors=TRUSTED)


def test_post_publication_envelope_facts_are_authenticated_and_trusted():
    intent = _intent()
    with pytest.raises(ValueError, match="trusted"):
        validate_intent(replace(intent, author="untrusted"), trusted_authors=TRUSTED)
    with pytest.raises(ValueError, match="timezone"):
        validate_intent(replace(intent, created_at="2025-01-01T00:00:00"), trusted_authors=TRUSTED)
    # The protocol consumes the typed envelope; provenance is authenticated by
    # the future fixed-endpoint client, not by this validator.
    assert validate_intent(replace(intent, node_id="different-node"), trusted_authors=TRUSTED)


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
    bad_metadata = _refresh_envelope(dict(metadata, payload=dict(metadata["payload"], metadata_digest="0" * 64)))
    with pytest.raises(ValueError, match="metadata digest"):
        validate_review_metadata(bad_metadata, intent, report, canonical_intent=intent, trusted_authors=TRUSTED)
    completion = _completion(intent, report, metadata)
    bad_completion = _refresh_envelope(dict(completion, payload=dict(completion["payload"], metadata_digest="wrong")))
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
        (replace(report, author="untrusted"), lambda item: validate_report(item, intent, canonical_intent=intent, trusted_authors=TRUSTED)),
        (replace(metadata, author="untrusted"), lambda item: validate_review_metadata(item, intent, report, canonical_intent=intent, trusted_authors=TRUSTED)),
        (replace(completion, author="untrusted"), lambda item: validate_completion(item, intent, report, metadata, canonical_intent=intent, trusted_authors=TRUSTED)),
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


def test_mixed_expected_targets_are_rejected_before_election():
    other_target = ReviewTarget("other/repo", 7, "other/repo", "other-head", "main", "base", "merge")
    with pytest.raises(ValueError, match="target"):
        elect_canonical_attempt(
            [_intent(), _intent(record_id="other", node_id="other-node", target=other_target)],
            [],
            expected_target=TARGET,
            trusted_authors=TRUSTED,
        )


def test_noncanonical_metadata_after_revocation_is_rejected():
    first = _intent(record_id="first", node_id="first-node")
    report = _report(first)
    second = _intent(record_id="second", node_id="second-node", created_at="2026-01-01T00:04:00Z")
    revocation = _revocation(first, created_at="2026-01-01T00:05:00Z")
    metadata = _metadata(first, report, created_at="2026-01-01T00:06:00Z")
    with pytest.raises(ValueError, match="canonical"):
        validate_protocol(
            [first, report, second, revocation, metadata], expected_target=TARGET, trusted_authors=TRUSTED
        )


def test_untrusted_revocations_are_rejected_from_the_authenticated_envelope():
    intent = _intent()
    revocation = replace(_revocation(intent), author="untrusted")
    with pytest.raises(ValueError, match="trusted"):
        validate_revocation(revocation, intent, trusted_authors=TRUSTED)


def test_altered_report_logical_id_node_and_digest_references_are_rejected():
    intent = _intent()
    report = _report(intent)
    metadata = _metadata(intent, report)

    altered_id = _refresh_envelope(dict(report, payload=dict(report["payload"], record_id="other-report")))
    with pytest.raises(ValueError, match="report|reference"):
        validate_review_metadata(metadata, intent, altered_id, canonical_intent=intent, trusted_authors=TRUSTED)

    altered_node = _refresh_envelope(dict(report, node_id="other-report-node"))
    with pytest.raises(ValueError, match="node"):
        validate_review_metadata(metadata, intent, altered_node, canonical_intent=intent, trusted_authors=TRUSTED)

    altered_digest = _refresh_envelope(
        dict(metadata, payload=dict(metadata["payload"], report_digest="other-report-digest"))
    )
    with pytest.raises(ValueError, match="digest"):
        validate_review_metadata(altered_digest, intent, report, canonical_intent=intent, trusted_authors=TRUSTED)


@pytest.mark.parametrize(
    "field, value",
    [
        ("report_record_id", "other-report"),
        ("report_node_id", "other-report-node"),
        ("report_digest", "other-report-digest"),
        ("metadata_record_id", "other-metadata"),
    ],
)
def test_completion_rejects_report_and_metadata_reference_mismatches(field, value):
    intent = _intent()
    report = _report(intent)
    metadata = _metadata(intent, report)
    completion = _refresh_envelope(dict(_completion(intent, report, metadata), payload={}))
    completion["payload"].update(_completion(intent, report, metadata)["payload"])
    completion["payload"][field] = value
    _refresh_envelope(completion)
    with pytest.raises(ValueError, match="report|metadata"):
        validate_completion(
            completion,
            intent,
            report,
            metadata,
            canonical_intent=intent,
            trusted_authors=TRUSTED,
        )


def test_append_only_snapshot_rejects_alteration_and_deletion():
    intent = _intent()
    report = _report(intent)
    snapshot = [intent, report]
    validate_append_only(snapshot, previous=snapshot)
    altered = _refresh_envelope(dict(intent, payload=dict(intent["payload"], attempt_id="altered")))
    with pytest.raises(ValueError, match="altered"):
        validate_append_only([altered, report], previous=snapshot)
    with pytest.raises(ValueError, match="deleted"):
        validate_append_only([intent], previous=snapshot)


def test_append_only_rejects_duplicate_ids_in_previous_snapshot_before_lookup():
    intent = _intent()
    duplicate_logical = _intent(record_id=intent.payload["record_id"], node_id="other-node")
    with pytest.raises(ValueError, match="duplicate logical"):
        validate_append_only([], previous=[intent, duplicate_logical])
    duplicate_node = _intent(record_id="other-record", node_id=intent.node_id)
    with pytest.raises(ValueError, match="duplicate authenticated"):
        validate_append_only([], previous=[intent, duplicate_node])
