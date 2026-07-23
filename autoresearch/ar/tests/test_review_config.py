# Copyright (c) Kaden Schutt
"""Focused tests for the protected review policy."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.ar.review.config import (
    AuthenticatedConfigSource,
    _SOURCE_PROOF,
    configuration_source_digest,
    load_review_configuration,
)
from autoresearch.ar.review.models import (
    capsule_paths_are_exempt,
    derive_protected_review_scope,
    profile_digest,
    validate_capability_policy,
)


ROOT = Path(__file__).parents[3]
POLICY = ROOT / ".github" / "agentic-review" / "capabilities-v1.json"
CONFIG_PATHS = (
    ".github/agentic-review/providers.json",
    ".github/agentic-review/capabilities-v1.json",
    ".github/agentic-review/trusted-publishers.json",
)


def test_protected_policy_has_exact_profile_and_exemption_schema():
    policy = json.loads(POLICY.read_text())

    assert set(policy) == {"schema", "version", "capabilities", "profiles", "fixtures", "exemptions"}
    assert len(policy["profiles"]) >= len(policy["capabilities"])
    assert policy["fixtures"]
    assert policy["exemptions"] == []
    validate_capability_policy(policy)


def test_profile_digest_covers_profile_content():
    policy = json.loads(POLICY.read_text())
    profile = policy["profiles"][0]
    mutated = deepcopy(profile)
    mutated["model_architecture"] = "qwen3.6-27b-mutated"

    assert profile_digest(mutated) != profile_digest(profile)


def test_protected_exemptions_match_normalized_repository_posix_globs():
    shallow = [{"id": "docs-shallow", "path_globs": ["docs/*"]}]
    nested = [{"id": "docs-nested", "path_globs": ["docs/**"]}]

    assert not capsule_paths_are_exempt(shallow, ["./docs/review.md"])
    assert capsule_paths_are_exempt(shallow, ["docs/review.md"])
    assert not capsule_paths_are_exempt(shallow, ["docs/deep/file.py"])
    assert capsule_paths_are_exempt(nested, ["docs/deep/file.py"])
    assert not capsule_paths_are_exempt(nested, [])
    assert not capsule_paths_are_exempt(nested, ["docs/review.md", "src/main.py"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda policy: policy["profiles"][0].update(fixture_digest="not-a-protected-digest"),
        lambda policy: policy["profiles"][0].update(capability_id="unknown@1"),
        lambda policy: policy["profiles"].append(policy["profiles"][0].copy()),
        lambda policy: policy["profiles"][0].update(covered_hardware=["gfx1151", "gfx1100"]),
        lambda policy: policy["profiles"][0].update(covered_hardware=["gfx1100", "not-eligible"]),
    ],
)
def test_profile_validation_rejects_spec_violations(mutation):
    policy = json.loads(POLICY.read_text())
    mutation(policy)

    with pytest.raises(ValueError):
        validate_capability_policy(policy)


def test_exemption_schema_is_exact_and_paths_are_all_covered():
    policy = json.loads(POLICY.read_text())
    policy["exemptions"] = [{"id": "docs", "path": "docs/**"}]
    with pytest.raises(ValueError):
        validate_capability_policy(policy)


def test_deep_multi_globstar_matching_is_iterative_and_bounded():
    path = "/".join(["prefix"] * 550 + ["segment"] + ["middle"] * 550 + ["target"])
    exemptions = [{"id": "deep", "path_globs": ["**/segment/**/target/**"]}]

    assert capsule_paths_are_exempt(exemptions, [path])


def test_authenticated_source_digest_changes_when_complete_capabilities_bytes_change(tmp_path):
    for relative in CONFIG_PATHS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())

    contents = tuple((tmp_path / path).read_bytes() for path in CONFIG_PATHS)
    source = AuthenticatedConfigSource._from_authenticated_boundary(
        _SOURCE_PROOF, "owner/repo", "main", "a" * 40,
        configuration_source_digest(*contents), tmp_path,
    )
    assert load_review_configuration(tmp_path, source=source).is_protected

    capabilities = tmp_path / CONFIG_PATHS[1]
    capabilities.write_bytes(capabilities.read_bytes() + b"\n")
    assert not load_review_configuration(tmp_path, source=source).is_protected


def test_path_matching_preserves_backslashes_and_whitespace_exactly():
    exemptions = [{"id": "docs", "path_globs": ["docs\\file.md", " docs/trim.md "]}]
    assert capsule_paths_are_exempt(exemptions, ["docs\\file.md"])
    assert capsule_paths_are_exempt(exemptions, [" docs/trim.md "])
    assert not capsule_paths_are_exempt(exemptions, ["docs/file.md"])


def test_scope_derivation_is_complete_for_non_exempt_capsule_and_empty_for_exempt():
    policy = json.loads(POLICY.read_text())
    capsule = SimpleNamespace(manifest=(SimpleNamespace(path="src/main.py"),))
    scope = derive_protected_review_scope(capsule, policy)
    assert scope.model_architectures == ("qwen3.6-27b",)
    assert scope.hardware_architectures == ("gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151")
    policy["exemptions"] = [{"id": "docs", "path_globs": ["docs/**"]}]
    exempt_capsule = SimpleNamespace(manifest=(SimpleNamespace(path="docs/readme.md"),))
    assert derive_protected_review_scope(exempt_capsule, policy).to_mapping() == {
        "model_architectures": [], "hardware_architectures": [],
    }


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda policy: policy["profiles"][0].update(fixture_id="unknown-fixture"), "fixture"),
        (lambda policy: policy["profiles"][0].update(fixture_digest="sha256:" + "0" * 64), "digest"),
        (lambda policy: policy["profiles"][0].update(model_architecture="other-model"), "model"),
        (lambda policy: policy["fixtures"][0].update(artifact_identity="other-report.json"), "descriptor"),
        (lambda policy: policy["fixtures"][0].update(suite_revision="wrong-suite"), "descriptor"),
    ],
)
def test_fixture_manifest_is_authoritative(mutation, message):
    policy = json.loads(POLICY.read_text())
    mutation(policy)
    with pytest.raises(ValueError, match=message):
        validate_capability_policy(policy)


def test_multiple_profiles_per_capability_are_allowed():
    policy = json.loads(POLICY.read_text())
    extra = deepcopy(policy["profiles"][0])
    extra["id"] = "rdna3-smoke-secondary"
    policy["profiles"].append(extra)
    validate_capability_policy(policy)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("suite_revision", "not-allowed", "suite"),
        ("artifact_identity", "not-allowed.json", "artifact"),
    ],
)
def test_fixture_must_match_referenced_capability(field, value, message):
    policy = json.loads(POLICY.read_text())
    fixture = policy["fixtures"][0]
    fixture[field] = value
    from autoresearch.ar.review.models import fixture_descriptor_digest
    fixture["fixture_digest"] = fixture_descriptor_digest(fixture)
    for profile in policy["profiles"]:
        if profile["fixture_id"] == fixture["fixture_id"]:
            profile["fixture_digest"] = fixture["fixture_digest"]
    with pytest.raises(ValueError, match=message):
        validate_capability_policy(policy)
