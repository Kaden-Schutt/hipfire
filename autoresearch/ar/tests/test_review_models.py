# Copyright (c) Kaden Schutt
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from autoresearch.ar.review.models import (
    AttemptIntent,
    ProviderPolicy,
    ReviewProposal,
    ReviewTarget,
    TrustedPublisher,
    ValidationRequest,
    validate_capability_policy,
)


ROOT = Path(__file__).parents[3]
POLICY_DIR = ROOT / ".github" / "agentic-review"


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
            AttemptIntent,
            ReviewProposal,
            ValidationRequest,
            ProviderPolicy,
            TrustedPublisher,
        )
    )


def test_empty_capability_policy_is_rejected():
    with pytest.raises(ValueError, match="capabilit"):
        validate_capability_policy(
            {"schema": "hipfire.agentic-review.capabilities", "version": 1, "capabilities": []}
        )


def test_capability_policy_shape():
    policy = json.loads((POLICY_DIR / "capabilities-v1.json").read_text())

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
        for field in (
            "contract_digest",
            "allowed_suite_revisions",
            "required_checks",
            "artifacts",
            "pass_criteria",
        ):
            assert field in capability
        validate_capability_policy(policy)


def test_provider_policy_shape_has_bounded_env_based_configuration():
    policy = json.loads((POLICY_DIR / "providers.json").read_text())

    assert policy["schema"] == "hipfire.agentic-review.providers"
    assert policy["version"] == 1
    assert policy["providers"]
    for provider in policy["providers"]:
        assert provider["endpoint_env"].isidentifier() or provider["endpoint_env"].startswith("HIPFIRE_")
        assert provider["api_key_env"].isidentifier() or provider["api_key_env"].startswith("HIPFIRE_")
        assert provider["model_env"].isidentifier() or provider["model_env"].startswith("HIPFIRE_")
        limits = provider["limits"]
        assert all(isinstance(limits[key], (int, float)) and limits[key] > 0 for key in limits)
        assert {"max_requests", "max_response_bytes", "max_tokens", "max_cost_usd"} <= limits.keys()


def test_trusted_publisher_policy_shape():
    policy = json.loads((POLICY_DIR / "trusted-publishers.json").read_text())

    assert policy["schema"] == "hipfire.agentic-review.trusted-publishers"
    assert policy["version"] == 1
    assert isinstance(policy["users"], list)
    assert isinstance(policy["apps"], list)
