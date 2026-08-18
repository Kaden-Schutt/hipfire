# Copyright (c) Kaden Schutt
import json
import hashlib
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from autoresearch.ar.review.models import (
    AttemptIntentConfig,
    ValidationLedgerRow,
    ValidationProfile,
    Finding,
    GitHubEnvelope,
    IntentPayload,
    ProviderPolicy,
    ReviewProposal,
    ReviewTarget,
    TrustedApp,
    TrustedPublisher,
    ValidationRequest,
    ProposedValidationObligation,
    capability_contract_digest,
    fixture_descriptor_digest,
    profile_digest,
    protected_exemption_evidence,
    load_capability_policy,
    load_provider_policy,
    load_trusted_publishers_policy,
    validate_capability_policy,
    validate_provider_policy,
    validate_trusted_publishers_policy,
)
from autoresearch.ar.review.canonical import canonical_digest, canonical_json, canonical_loads
from autoresearch.ar.review.validation import MAX_VALIDATION_LEDGER_BYTES


ROOT = Path(__file__).parents[3]
POLICY_DIR = ROOT / ".github" / "agentic-review"
TARGET = ReviewTarget("owner/repo", 42, "owner/repo", "head", "main", "base", "merge")


def make_proposal(verdict, findings=(), *, capsule_digest="sha256:" + "a" * 64, response_digest="sha256:" + "c" * 64):
    values = {
        "target": TARGET,
        "target_key": TARGET.target_key(),
        "capsule_digest": capsule_digest,
        "adapter_id": "openai-compatible",
        "adapter_version": "1",
        "model": "review-model-v1",
        "response_digest": response_digest,
        "verdict": verdict,
        "findings": tuple(findings),
    }
    digest = "sha256:" + canonical_digest(values)
    return ReviewProposal(
        TARGET, capsule_digest, digest, verdict, tuple(findings),
        "openai-compatible", "1", "review-model-v1", response_digest,
    )


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
            ValidationProfile,
            ProposedValidationObligation,
            ValidationLedgerRow,
        )
    )


def test_empty_capability_policy_is_rejected():
    policy = json.loads((POLICY_DIR / "capabilities-v1.json").read_text())
    policy["capabilities"] = []
    with pytest.raises(ValueError, match="capabilit"):
        validate_capability_policy(policy)


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


def test_capability_policy_rejects_stale_contract_digest():
    policy = json.loads((POLICY_DIR / "capabilities-v1.json").read_text())
    policy["capabilities"][0]["required_checks"] = ["changed-check"]

    with pytest.raises(ValueError, match="^capability contract digest does not match capability$"):
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


@pytest.mark.parametrize(
    "field, value",
    [
        ("allowed_suite_revisions", ["changed-suite-v1"]),
        ("artifacts", ["changed-artifact.json"]),
        ("eligible_hardware", ["changed-hardware"]),
    ],
)
def test_rehashed_capability_rejects_incoherent_dependent_records(field, value):
    policy = load_capability_policy(POLICY_DIR / "capabilities-v1.json")
    capability_id = "hipfire/rdna3-smoke@1"
    capability = next(item for item in policy["capabilities"] if item["id"] == capability_id)
    capability[field] = value
    capability["contract_digest"] = capability_contract_digest(capability)

    with pytest.raises(ValueError):
        validate_capability_policy(policy)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("id", "hipfire/changed@1", "wrong capability IDs"),
        ("pass_criteria", {"all_required_checks_pass": False}, "pass_criteria"),
    ],
)
def test_rehashed_capability_rejects_invalid_capability_contract(field, value, message):
    policy = load_capability_policy(POLICY_DIR / "capabilities-v1.json")
    capability = policy["capabilities"][0]
    capability[field] = value
    capability["contract_digest"] = capability_contract_digest(capability)

    with pytest.raises(ValueError, match=message):
        validate_capability_policy(policy)


@pytest.mark.parametrize(
    "field, value",
    [
        ("allowed_suite_revisions", ["changed-suite-v1"]),
        ("required_checks", ["changed-check"]),
        ("artifacts", ["changed-artifact.json"]),
        ("eligible_hardware", ["changed-hardware"]),
    ],
)
def test_rehashed_capability_accepts_coherent_dependent_records(field, value):
    policy = load_capability_policy(POLICY_DIR / "capabilities-v1.json")
    capability_id = "hipfire/rdna3-smoke@1"
    capability = next(item for item in policy["capabilities"] if item["id"] == capability_id)
    capability[field] = value
    capability["contract_digest"] = capability_contract_digest(capability)

    profiles = [profile for profile in policy["profiles"] if profile["capability_id"] == capability_id]
    fixture_ids = {profile["fixture_id"] for profile in profiles}
    fixtures = [fixture for fixture in policy["fixtures"] if fixture["fixture_id"] in fixture_ids]
    if field == "allowed_suite_revisions":
        for fixture in fixtures:
            fixture["suite_revision"] = value[0]
    elif field == "artifacts":
        for fixture in fixtures:
            fixture["artifact_identity"] = value[0]
    elif field == "eligible_hardware":
        for profile in profiles:
            profile["representative_hardware"] = value[0]
            profile["covered_hardware"] = value

    if field in ("allowed_suite_revisions", "artifacts"):
        for fixture in fixtures:
            fixture["fixture_digest"] = fixture_descriptor_digest(fixture)
            for profile in policy["profiles"]:
                if profile["fixture_id"] == fixture["fixture_id"]:
                    profile["fixture_digest"] = fixture["fixture_digest"]

    validate_capability_policy(policy)


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
    assert policy["fixtures"]
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


def test_provider_digest_limits_do_not_exceed_model_canonical_ceiling():
    provider = deepcopy(VALID_PROVIDER)
    provider["max_response_bytes"] = (1 << 20) + 1
    with pytest.raises(ValueError, match="canonical|response"):
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
    proposal = make_proposal("clean", (finding,))
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
        make_proposal("clean", (finding,))


def test_changes_requested_requires_actionable_finding():
    finding = Finding("src/main.py", (1, 2), "warning", "consider this")

    with pytest.raises(ValueError, match="actionable"):
        make_proposal("changes-requested", (finding,))


def test_changes_requested_accepts_error_finding_and_incomplete_is_explicit():
    finding = Finding("src/main.py", (1, 2), "error", "must fix")
    proposal = make_proposal("changes-requested", (finding,))
    incomplete = make_proposal("incomplete")

    assert proposal.verdict == "changes-requested"
    assert incomplete.verdict == "incomplete"


def test_review_proposal_requires_provider_audit_fields():
    with pytest.raises(TypeError):
        ReviewProposal(TARGET, "sha256:" + "a" * 64, "sha256:" + "b" * 64, "clean", ())


@pytest.mark.parametrize("verdict", ["approved", "reject", "unknown"])
def test_review_proposal_rejects_arbitrary_verdict(verdict):
    with pytest.raises(ValueError, match="verdict"):
        ReviewProposal(TARGET, "sha256:" + "a" * 64, "sha256:" + "b" * 64, verdict, (),
                       "openai-compatible", "1", "review-model-v1", "sha256:" + "c" * 64)


PROFILE = ValidationProfile(
    id="rdna3-smoke",
    capability_id="hipfire/rdna3-smoke@1",
    model_architecture="qwen3.6-27b",
    fixture_id="qwen3.6-27b-rdna3-smoke-v1",
    fixture_digest="sha256:" + "f" * 64,
    representative_hardware="gfx1100",
    covered_hardware=("gfx1100", "gfx1101"),
)


def test_exemption_evidence_derives_sorted_ids_across_separate_entries():
    exemptions = [
        {"id": "docs", "path_globs": ["docs/**"]},
        {"id": "src", "path_globs": ["src/**"]},
    ]
    assert protected_exemption_evidence(exemptions, ["src/main.py", "docs/review.md"]) == (
        ("docs", "src"), ("docs/review.md", "src/main.py"),
    )


def test_profile_identifier_128_bytes_is_allowed_but_129_is_rejected():
    valid = ValidationProfile("p" * 128, PROFILE.capability_id, PROFILE.model_architecture,
                              PROFILE.fixture_id, PROFILE.fixture_digest,
                              PROFILE.representative_hardware, PROFILE.covered_hardware)
    row = ValidationLedgerRow(valid, "sha256:" + "2" * 64, "representative")
    values = {
        "target": TARGET, "target_key": TARGET.target_key(), "capsule_digest": "sha256:" + "a" * 64,
        "adapter_id": "adapter", "adapter_version": "1", "model": "model",
        "response_digest": "sha256:" + "c" * 64, "verdict": "clean", "findings": (),
        "validation_ledger": (row.to_mapping(),), "configuration_source_digest": "sha256:" + "d" * 64,
    }
    ReviewProposal(TARGET, values["capsule_digest"], "sha256:" + canonical_digest(values), "clean", (),
                   "adapter", "1", "model", values["response_digest"], validation_ledger=(row,),
                   configuration_source_digest=values["configuration_source_digest"])
    invalid_profile = ValidationProfile("p" * 129, PROFILE.capability_id, PROFILE.model_architecture,
                                        PROFILE.fixture_id, PROFILE.fixture_digest,
                                        PROFILE.representative_hardware, PROFILE.covered_hardware)
    invalid_row = ValidationLedgerRow(invalid_profile, "sha256:" + "2" * 64, "representative")
    invalid_values = {**values, "validation_ledger": (invalid_row.to_mapping(),)}
    with pytest.raises(ValueError, match=r"profile_snapshot\.id exceeds its maximum UTF-8 length"):
        ReviewProposal(TARGET, invalid_values["capsule_digest"], "sha256:" + canonical_digest(invalid_values), "clean", (),
                       "adapter", "1", "model", invalid_values["response_digest"],
                       validation_ledger=(invalid_row,), configuration_source_digest=invalid_values["configuration_source_digest"])


def test_serialized_ledger_over_64_kib_is_rejected():
    def rows(first_rationale_length):
        return tuple(sorted((
            ValidationLedgerRow(
                ValidationProfile(f"profile-{index}", "capability", "arch", f"fixture-{index}",
                                   "sha256:" + "f" * 64, "gfx1100", ("gfx1100",)),
                "sha256:" + "2" * 64, "representative",
                ProposedValidationObligation(f"profile-{index}", "x" * (
                    first_rationale_length if index == 0 else 1024
                )),
            ) for index in range(35)), key=lambda row: row.request_id))
    measured_base = len(canonical_json(tuple(row.to_mapping() for row in rows(1))))
    exact_first_rationale_length = MAX_VALIDATION_LEDGER_BYTES - measured_base + 1
    exact_rows = rows(exact_first_rationale_length)
    assert len(canonical_json(tuple(row.to_mapping() for row in exact_rows))) == MAX_VALIDATION_LEDGER_BYTES
    values = {
        "target": TARGET, "target_key": TARGET.target_key(), "capsule_digest": "sha256:" + "a" * 64,
        "adapter_id": "adapter", "adapter_version": "1", "model": "model",
        "response_digest": "sha256:" + "c" * 64, "verdict": "clean", "findings": (),
        "validation_ledger": tuple(row.to_mapping() for row in exact_rows),
        "configuration_source_digest": "sha256:" + "d" * 64,
    }
    ReviewProposal(TARGET, values["capsule_digest"], "sha256:" + canonical_digest(values), "clean", (),
                   "adapter", "1", "model", values["response_digest"], validation_ledger=exact_rows,
                   configuration_source_digest=values["configuration_source_digest"])
    over_rows = rows(exact_first_rationale_length + 1)
    assert len(canonical_json(tuple(row.to_mapping() for row in over_rows))) == MAX_VALIDATION_LEDGER_BYTES + 1
    over_values = {**values, "validation_ledger": tuple(row.to_mapping() for row in over_rows)}
    with pytest.raises(ValueError, match="64 KiB"):
        ReviewProposal(TARGET, over_values["capsule_digest"], "sha256:" + canonical_digest(over_values), "clean", (),
                       "adapter", "1", "model", over_values["response_digest"], validation_ledger=over_rows,
                       configuration_source_digest=over_values["configuration_source_digest"])


def test_validation_profile_and_obligation_are_immutable_and_exact():
    obligation = ProposedValidationObligation("rdna3-smoke", "  run the smoke suite\n  once ")

    assert obligation.rationale == "run the smoke suite once"
    assert PROFILE.fixture_digest != "sha256:" + hashlib.sha256(PROFILE.fixture_id.encode()).hexdigest()
    assert set(ValidationProfile.__dataclass_fields__) == {
        "id", "capability_id", "model_architecture", "fixture_id", "fixture_digest",
        "representative_hardware", "covered_hardware",
    }
    assert set(ProposedValidationObligation.__dataclass_fields__) == {"profile_id", "rationale"}
    with pytest.raises(FrozenInstanceError):
        obligation.profile_id = "changed"


def test_validation_ledger_row_derives_request_id_and_serializes_typed_snapshot():
    obligation = ProposedValidationObligation("rdna3-smoke", "run it")
    row = ValidationLedgerRow(PROFILE, "sha256:" + "2" * 64, "representative", (obligation,))
    serialized = row.to_mapping()

    assert row.request_id == "vr-" + hashlib.sha256(PROFILE.id.encode()).hexdigest()[:16]
    assert len(row.request_id) == 19
    assert serialized["profile_snapshot"] == PROFILE.to_mapping()
    assert serialized["profile_digest"] == profile_digest(PROFILE.to_mapping())
    assert isinstance(serialized["profile_snapshot"]["covered_hardware"], list)
    assert serialized["status"] == "pending"
    assert serialized["validator_snapshot"] == {}
    assert serialized["result_snapshot"] == {}
    decoded = canonical_loads(canonical_json(serialized))
    assert ValidationLedgerRow.from_mapping(decoded).to_mapping() == serialized
    with pytest.raises(TypeError):
        ValidationLedgerRow(PROFILE, "sha256:" + "2" * 64, "representative", (), request_id="provider-id")


@pytest.mark.parametrize("field", ["request_id", "status", "validator_snapshot", "result_snapshot", "capability_id"])
def test_validation_ledger_row_rejects_provider_or_caller_fields(field):
    with pytest.raises(TypeError):
        ValidationLedgerRow(
            PROFILE, "sha256:" + "2" * 64, "representative",
            (ProposedValidationObligation("rdna3-smoke", "required"),), **{field: "caller-value"},
        )


def test_review_proposal_digest_binds_enriched_rows_and_config_source():
    obligation = ProposedValidationObligation("rdna3-smoke", "required")
    row = ValidationLedgerRow(PROFILE, "sha256:" + "2" * 64, "representative", (obligation,))
    config_digest = "sha256:" + "d" * 64
    values = {
        "target": TARGET, "target_key": TARGET.target_key(),
        "capsule_digest": "sha256:" + "a" * 64,
        "adapter_id": "adapter", "adapter_version": "1", "model": "model",
        "response_digest": "sha256:" + "c" * 64, "verdict": "clean", "findings": (),
        "validation_ledger": (row.to_mapping(),), "configuration_source_digest": config_digest,
    }
    proposal = ReviewProposal(
        TARGET, values["capsule_digest"], "sha256:" + canonical_digest(values), "clean", (),
        "adapter", "1", "model", values["response_digest"],
        validation_ledger=(row,), configuration_source_digest=config_digest,
    )

    assert proposal.proposal_digest == "sha256:" + canonical_digest(values)
    with pytest.raises(ValueError, match="proposal digest"):
        ReviewProposal(
            TARGET, values["capsule_digest"], "sha256:" + canonical_digest({**values, "validation_ledger": ()}),
            "clean", (), "adapter", "1", "model", values["response_digest"],
            validation_ledger=(row,), configuration_source_digest=config_digest,
        )


def test_review_proposal_rejects_duplicate_and_noncanonical_ledger_order():
    duplicate = ValidationLedgerRow(PROFILE, "sha256:" + "2" * 64, "representative", ())
    duplicate_values = {
        "target": TARGET, "target_key": TARGET.target_key(), "capsule_digest": "sha256:" + "a" * 64,
        "adapter_id": "adapter", "adapter_version": "1", "model": "model",
        "response_digest": "sha256:" + "c" * 64, "verdict": "clean", "findings": (),
        "validation_ledger": (duplicate.to_mapping(), duplicate.to_mapping()),
        "configuration_source_digest": "sha256:" + "d" * 64,
    }
    with pytest.raises(ValueError, match="unique"):
        ReviewProposal(
            TARGET, duplicate_values["capsule_digest"], "sha256:" + canonical_digest(duplicate_values), "clean", (),
            "adapter", "1", "model", "sha256:" + "c" * 64,
            validation_ledger=(duplicate, duplicate), configuration_source_digest="sha256:" + "d" * 64,
        )

    other_profile = ValidationProfile(
        "another-profile", PROFILE.capability_id, PROFILE.model_architecture, "another-fixture",
        "sha256:" + hashlib.sha256(b"another-fixture").hexdigest(),
        PROFILE.representative_hardware, PROFILE.covered_hardware,
    )
    first = ValidationLedgerRow(PROFILE, "sha256:" + "2" * 64, "representative", ())
    second = ValidationLedgerRow(other_profile, "sha256:" + "2" * 64, "representative", ())
    rows = (first, second)
    if tuple(row.request_id for row in rows) == tuple(sorted(row.request_id for row in rows)):
        rows = (second, first)
    order_values = {
        "target": TARGET, "target_key": TARGET.target_key(), "capsule_digest": "sha256:" + "a" * 64,
        "adapter_id": "adapter", "adapter_version": "1", "model": "model",
        "response_digest": "sha256:" + "c" * 64, "verdict": "clean", "findings": (),
        "validation_ledger": tuple(row.to_mapping() for row in rows),
        "configuration_source_digest": "sha256:" + "d" * 64,
    }
    with pytest.raises(ValueError, match=r"validation ledger request IDs must be sorted and unique"):
        ReviewProposal(
            TARGET, order_values["capsule_digest"], "sha256:" + canonical_digest(order_values), "clean", (),
            "adapter", "1", "model", "sha256:" + "c" * 64,
            validation_ledger=rows, configuration_source_digest="sha256:" + "d" * 64,
        )
