# Copyright (c) Kaden Schutt
import json
import hashlib
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from autoresearch.ar.review.models import (
    AttemptIntentConfig,
    Finding,
    GitHubEnvelope,
    IntentPayload,
    ProviderPolicy,
    ReviewProposal,
    ReviewTarget,
    TrustedApp,
    TrustedPublisher,
    ValidationRequest,
    capability_contract_digest,
    load_capability_policy,
    load_provider_policy,
    load_trusted_publishers_policy,
    validate_capability_policy,
    validate_provider_policy,
    validate_trusted_publishers_policy,
)
from autoresearch.ar.review.canonical import canonical_digest, canonical_json


ROOT = Path(__file__).parents[3]
POLICY_DIR = ROOT / ".github" / "agentic-review"
TARGET = ReviewTarget("owner/repo", 42, "owner/repo", "head", "main", "base", "merge")


def test_review_target_key_is_stable_and_base_sha_sensitive():
    target = ReviewTarget(
        repository="Kaden-Schutt/hipfire",
        number=42,
        head_repository="Kaden-Schutt/hipfire",
        head_sha="head-sha",
        base_ref="main",
        base_sha="base-sha",
        merge_base_sha="merge-base-sha",
    )

    assert target.target_key() == target.target_key()
    assert target.target_key() != ReviewTarget(
        repository=target.repository,
        number=target.number,
        head_repository=target.head_repository,
        head_sha=target.head_sha,
        base_ref=target.base_ref,
        base_sha="different-base-sha",
        merge_base_sha=target.merge_base_sha,
    ).target_key()


def test_contracts_are_frozen():
    target = ReviewTarget("repo", 1, "repo", "head", "main", "base", "merge")
    with pytest.raises(FrozenInstanceError):
        target.base_sha = "changed"

    assert all(
        getattr(cls, "__dataclass_params__").frozen
        for cls in (
            AttemptIntentConfig,
            IntentPayload,
            Finding,
            ReviewProposal,
            ValidationRequest,
            ProviderPolicy,
            TrustedApp,
            TrustedPublisher,
        )
    )


def test_empty_capability_policy_is_rejected():
    with pytest.raises(ValueError, match="capabilit"):
        validate_capability_policy(
            {"schema": "hipfire.agentic-review.capabilities", "version": 1, "capabilities": []}
        )


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:" + "a" * 63,
        "sha256:" + "a" * 65,
        "sha256:" + "A" * 64,
        "sha256:" + "g" * 64,
    ],
)
def test_capability_policy_rejects_invalid_contract_digests(digest):
    policy = json.loads((POLICY_DIR / "capabilities-v1.json").read_text())
    policy["capabilities"][0]["contract_digest"] = digest

    with pytest.raises(ValueError, match="digest"):
        validate_capability_policy(policy)


@pytest.mark.parametrize(
    "field, value",
    [
        ("id", "hipfire/changed@1"),
        ("allowed_suite_revisions", ["changed-suite-v1"]),
        ("required_checks", ["changed-check"]),
        ("artifacts", ["changed-artifact.json"]),
        ("eligible_hardware", ["changed-hardware"]),
        ("pass_criteria", {"all_required_checks_pass": False}),
    ],
)
def test_capability_digest_covers_complete_capability(field, value):
    policy = load_capability_policy(POLICY_DIR / "capabilities-v1.json")
    mutated = deepcopy(policy)
    capability = mutated["capabilities"][0]
    original_digest = capability["contract_digest"]
    capability[field] = value

    changed_digest = capability_contract_digest(capability)
    assert changed_digest != original_digest

    with pytest.raises(ValueError, match="digest|capability ID|pass_criteria"):
        validate_capability_policy(mutated)

    capability["contract_digest"] = changed_digest
    if field not in ("id", "pass_criteria"):
        validate_capability_policy(mutated)


def test_capability_digest_uses_documented_canonical_json():
    policy = load_capability_policy(POLICY_DIR / "capabilities-v1.json")
    capability = policy["capabilities"][0]
    without_digest = {key: value for key, value in capability.items() if key != "contract_digest"}
    expected = "sha256:" + hashlib.sha256(canonical_json(without_digest)).hexdigest()

    assert capability_contract_digest(capability) == expected


def test_capability_policy_shape_and_loader():
    policy = load_capability_policy(POLICY_DIR / "capabilities-v1.json")

    assert policy["schema"] == "hipfire.agentic-review.capabilities"
    assert policy["version"] == 1
    capabilities = policy["capabilities"]
    assert {capability["id"] for capability in capabilities} == {
        "hipfire/rdna3-smoke@1",
        "hipfire/gfx1151-kernel-validation@1",
        "hipfire/dflash-coherence@1",
    }
    for capability in capabilities:
        assert capability["parameters"] == {}
        assert capability["eligible_hardware"]
        for field in (
            "contract_digest",
            "allowed_suite_revisions",
            "required_checks",
            "artifacts",
            "pass_criteria",
        ):
            assert field in capability
        assert capability["pass_criteria"] == {"all_required_checks_pass": True}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda policy: policy.pop("version"),
        lambda policy: policy["capabilities"][0].pop("artifacts"),
        lambda policy: policy["capabilities"][0].update(extra=True),
        lambda policy: policy["capabilities"][0]["required_checks"].append(3),
        lambda policy: policy["capabilities"][0]["required_checks"].append("build"),
        lambda policy: policy["capabilities"][0].update(eligible_hardware=[]),
        lambda policy: policy["capabilities"][0].update(pass_criteria={"other": True}),
    ],
)
def test_capability_loader_rejects_malformed_policy(mutation):
    policy = json.loads((POLICY_DIR / "capabilities-v1.json").read_text())
    mutation(policy)

    with pytest.raises(ValueError):
        validate_capability_policy(policy)


def test_provider_policy_shape_has_bounded_env_based_configuration():
    policy = json.loads((POLICY_DIR / "providers.json").read_text())

    assert policy["schema"] == "hipfire.agentic-review.providers"
    assert policy["version"] == 1
    assert policy["providers"] == []
    validate_provider_policy(policy)


def test_provider_loader_fails_closed_for_unspecified_provider():
    with pytest.raises(ValueError, match="provider"):
        load_provider_policy(POLICY_DIR / "providers.json", "missing")


VALID_PROVIDER = {
    "id": "review-adapter",
    "adapter_id": "neutral-review",
    "adapter_version": "1",
    "endpoint": "https://review.example.invalid/v1",
    "model": "review-model-v1",
    "api_key_env": "HIPFIRE_REVIEW_API_KEY",
    "max_requests": 1,
    "request_deadline_seconds": 30,
    "max_capsule_bytes": 1048576,
    "max_response_bytes": 1048576,
    "max_tokens": 16384,
    "max_cost_usd": 5.0,
}


def provider_policy(provider=None):
    return {
        "schema": "hipfire.agentic-review.providers",
        "version": 1,
        "providers": [provider or VALID_PROVIDER],
    }


@pytest.mark.parametrize(
    "field, value",
    [
        ("endpoint_env", "HIPFIRE_ENDPOINT"),
        ("model_env", "HIPFIRE_MODEL"),
        ("endpoint", "http://review.example.invalid"),
        ("max_requests", 2),
    ],
)
def test_provider_policy_rejects_unprotected_selection_or_budget(field, value):
    provider = deepcopy(VALID_PROVIDER)
    provider[field] = value

    with pytest.raises(ValueError):
        validate_provider_policy(provider_policy(provider))


@pytest.mark.parametrize(
    "field",
    [
        "adapter_id",
        "adapter_version",
        "endpoint",
        "model",
        "api_key_env",
        "request_deadline_seconds",
        "max_capsule_bytes",
        "max_response_bytes",
        "max_tokens",
        "max_cost_usd",
    ],
)
def test_provider_policy_requires_fixed_fields_and_finite_bounds(field):
    provider = deepcopy(VALID_PROVIDER)
    provider.pop(field)

    with pytest.raises(ValueError):
        validate_provider_policy(provider_policy(provider))


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), float("-inf")])
def test_provider_policy_rejects_nonfinite_cost(cost):
    with pytest.raises(ValueError, match="max_cost_usd"):
        ProviderPolicy(
            "review-adapter",
            "neutral-review",
            "1",
            "https://review.example.invalid/v1",
            "review-model-v1",
            "HIPFIRE_REVIEW_API_KEY",
            1,
            30,
            1,
            1,
            1,
            cost,
        )


def test_trusted_publisher_policy_shape():
    policy = load_trusted_publishers_policy(POLICY_DIR / "trusted-publishers.json")

    assert policy["schema"] == "hipfire.agentic-review.trusted-publishers"
    assert policy["version"] == 1
    assert set(policy) == {"schema", "version", "apps"}
    assert policy["apps"] == []


def test_trusted_publishers_rejects_static_users_key():
    policy = {
        "schema": "hipfire.agentic-review.trusted-publishers",
        "version": 1,
        "users": ["Kaden-Schutt"],
        "apps": [],
    }

    with pytest.raises(ValueError, match="unexpected|users"):
        validate_trusted_publishers_policy(policy)


def test_trusted_publishers_accepts_structured_app():
    policy = {
        "schema": "hipfire.agentic-review.trusted-publishers",
        "version": 1,
        "apps": [
            {
                "app_id": 123,
                "login": "review-app[bot]",
                "installation_id": 456,
                "repository_id": 789,
                "credential_attestation_digest": "sha256:" + "a" * 64,
            }
        ],
    }
    validate_trusted_publishers_policy(policy)


@pytest.mark.parametrize(
    "missing",
    ["app_id", "login", "installation_id", "repository_id", "credential_attestation_digest"],
)
def test_trusted_publishers_rejects_incomplete_app(missing):
    app = {
        "app_id": 123,
        "login": "review-app[bot]",
        "installation_id": 456,
        "repository_id": 789,
        "credential_attestation_digest": "sha256:" + "a" * 64,
    }
    app.pop(missing)
    policy = {
        "schema": "hipfire.agentic-review.trusted-publishers",
        "version": 1,
        "apps": [app],
    }

    with pytest.raises(ValueError):
        validate_trusted_publishers_policy(policy)


def test_trusted_publishers_rejects_generic_app_entry():
    policy = {
        "schema": "hipfire.agentic-review.trusted-publishers",
        "version": 1,
        "apps": ["github-actions"],
    }

    with pytest.raises(ValueError):
        validate_trusted_publishers_policy(policy)


def test_review_contracts_bind_required_identity_and_target_fields():
    intent = AttemptIntentConfig(TARGET, "attempt-1", "capability", "suite-v1")
    assert intent.target == TARGET
    assert set(intent.__dataclass_fields__) == {
        "target", "attempt_id", "capability_id", "suite_revision", "provider_id"
    }
    envelope = GitHubEnvelope(
        {"record_id": "logical-intent"}, "gh-node", "review-bot", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    assert envelope.node_id == "gh-node"
    finding = Finding("src/main.py", (1, 2), "warning", "nonblocking")
    proposal = ReviewProposal(TARGET, "sha256:" + "a" * 64, "sha256:" + "b" * 64, "clean", (finding,))
    assert proposal.findings == (finding,)
    request = ValidationRequest(TARGET, "request-1", "capability", "sha256:" + "a" * 64, "sha256:" + "b" * 64)
    assert request.target == TARGET


def test_intent_payload_model_matches_protocol_shape():
    values = {
        "schema": "agentic-review/v1",
        "record_type": "intent",
        "record_id": "logical-intent",
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "attempt_id": "attempt-1",
    }
    values["canonical_digest"] = canonical_digest(values)
    payload = IntentPayload(**values)
    assert payload.to_mapping()["record_id"] == "logical-intent"


def test_intent_payload_json_round_trip_normalizes_target_mapping():
    values = {
        "schema": "agentic-review/v1",
        "record_type": "intent",
        "record_id": "logical-intent",
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "attempt_id": "attempt-1",
    }
    values["target"] = {
        "repository": TARGET.repository,
        "number": TARGET.number,
        "head_repository": TARGET.head_repository,
        "head_sha": TARGET.head_sha,
        "base_ref": TARGET.base_ref,
        "base_sha": TARGET.base_sha,
        "merge_base_sha": TARGET.merge_base_sha,
    }
    values["canonical_digest"] = canonical_digest(values)
    decoded = json.loads(canonical_json(values).decode())
    model = IntentPayload.from_mapping(decoded)
    assert model.target == TARGET
    assert canonical_json(model.to_mapping()) == canonical_json(decoded)
    decoded["target"]["extra"] = "reject"
    with pytest.raises(ValueError, match="target|shape"):
        IntentPayload.from_mapping(decoded)



@pytest.mark.parametrize("severity", ["critical", "blocker", "unknown"])
def test_finding_rejects_arbitrary_severity(severity):
    with pytest.raises(ValueError, match="severity"):
        Finding("src/main.py", (1, 2), severity, "message")


@pytest.mark.parametrize("source_range", [(2, 1), (0, 1), (-1, 1), (1, 0)])
def test_finding_rejects_invalid_source_range(source_range):
    with pytest.raises(ValueError, match="range"):
        Finding("src/main.py", source_range, "error", "message")


def test_clean_proposal_rejects_actionable_finding():
    finding = Finding("src/main.py", (1, 2), "error", "must fix")

    with pytest.raises(ValueError, match="clean|actionable"):
        ReviewProposal(TARGET, "sha256:" + "a" * 64, "sha256:" + "b" * 64, "clean", (finding,))


def test_changes_requested_requires_actionable_finding():
    finding = Finding("src/main.py", (1, 2), "warning", "consider this")

    with pytest.raises(ValueError, match="actionable"):
        ReviewProposal(TARGET, "sha256:" + "a" * 64, "sha256:" + "b" * 64, "changes-requested", (finding,))


def test_changes_requested_accepts_error_finding_and_incomplete_is_explicit():
    finding = Finding("src/main.py", (1, 2), "error", "must fix")
    proposal = ReviewProposal(
        TARGET, "sha256:" + "a" * 64, "sha256:" + "b" * 64, "changes-requested", (finding,)
    )
    incomplete = ReviewProposal(TARGET, "sha256:" + "a" * 64, "sha256:" + "b" * 64, "incomplete", ())

    assert proposal.verdict == "changes-requested"
    assert incomplete.verdict == "incomplete"


@pytest.mark.parametrize("verdict", ["approved", "reject", "unknown"])
def test_review_proposal_rejects_arbitrary_verdict(verdict):
    with pytest.raises(ValueError, match="verdict"):
        ReviewProposal(TARGET, "sha256:" + "a" * 64, "sha256:" + "b" * 64, verdict, ())
