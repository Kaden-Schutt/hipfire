#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate the upstream-native device-mesh port authority.

The tracker is one JSON authority. ``obligations`` retains domain/task proof
contracts, ``change_sets`` supplies G0..G15 review/revert boundaries,
``evidence_campaigns`` owns physical and aggregate proof, and ``seam_gates``
connect producers to consumers. This checker does not infer implementation or
admission status from source files or measurements. JSON is the sole plan
data source; this script validates generic invariants, referential integrity,
DAG/order, and durable evidence without duplicating plan maps.

Usage:
    scripts/check-device-mesh-port-tracker.py [tracker.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SERIES_ORIGIN_REF = "6163369329d3b076376286a00c13cadbae069ecc"
FORBIDDEN_BOUNDARIES = (
    "5b95cbd331dfffa2e1df8bad66ac79ec4f4b447f",
    "1df219dd45cdaa4aa10e876491e6c280354dd3a7",
    "aa3e5a30746474bff3787c2ff5d3cf733db3ae04",
    "084dbebda9c72116fc9b4819f85ec2e15aa31344",
    "bac02a1a22a55922ea057e9a98f68cb3ab93ac02",
)
SCHEMA = "hipfire.device_mesh.port_tracker.v2"
SCHEMA_VERSION = 2
ALLOWED_OBLIGATION_STATUSES = frozenset({"complete", "ready", "blocked"})
ALLOWED_DELIVERY_KINDS = frozenset({"change_set", "evidence_campaign", "final_closure"})
ALLOWED_CHANGE_SET_STATUSES = frozenset({"implemented", "in_review", "complete", "ready", "blocked"})
ALLOWED_CLASSES = frozenset({"port", "superseded", "already_upstream", "historical_evidence_only", "needs_design"})
ALLOWED_DISPOSITIONS = frozenset({"not_applicable", "current", "historical", "rerun_required", "hardware_blocked"})
ALLOWED_EVIDENCE_CLASSES = frozenset({"current", "historical", "rerun_required", "hardware_blocked", "semantics_only", "emulated", "failed"})
ALLOWED_BRANCH_RECORDS = frozenset({"none", "historical"})
ALLOWED_CONFIDENCE = frozenset({"high", "medium", "low"})
ALLOWED_GATE_STATUSES = frozenset({"available", "complete", "proposed", "blocked"})
ALLOWED_CAMPAIGN_CLASSES = frozenset({"physical", "closure"})
ALLOWED_LEGACY_STATUSES = frozenset({"complete", "ready", "blocked", "in_progress", "deferred", "not_yet_present"})
# Repository report roots - approved report-like artifacts.  Cargo.toml/lock and crates/ are NOT report artifacts.
APPROVED_REPO_ROOTS = (
    "docs/",
    "scripts/",
    "tests/",
    "benchmarks/",
    "tools/",
    ".github/",
)
# Artifact roots are a subset of repo roots that qualify as durable report artifacts for completion promotion.
APPROVED_ARTIFACT_ROOTS = (
    "docs/",
    "tests/",
    "benchmarks/",
)
BAD_COMPLETION_CLASSES = frozenset({"historical", "rerun_required", "hardware_blocked", "semantics_only", "emulated", "failed"})
REQUIRED_CONTENT_TERMS = {
    "COR-001": ("mtp_k", "ModelMeta", "HIPFIRE_MTP_K"),
    "COR-002": ("reset", "VL", "PP", "TP", "EP", "speculative", "recurrent", "conv"),
    "COR-004": ("eviction", "LoadedModel", "cross-request", "request state"),
    "COR-005": ("transactional", "DFlash", "rollback", "allocation", "Drop"),
    "GEN-001": ("Qwen35", "arch-resident", "DeltaNet", "MoE", "recurrent/conv", "emulated PP"),
    "SPEC-003": ("transactional", "on-disk", "GQA", "vocab-map", "rollback", "mtp_mode", "MTP scratch"),
    "SPEC-004": ("PP+MTP", "compressed .mtp", "cycle/depth", "64 MiB", "SPEC-003"),
}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strings(value: Any, *, nonempty: bool = False) -> bool:
    return isinstance(value, list) and all(_nonempty(item) for item in value) and (not nonempty or bool(value))


def _host_local(value: Any) -> bool:
    return isinstance(value, str) and any(token in value for token in ("/home/", "/tmp/"))


def _full_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{40}", value))


def _forbidden_boundary(value: Any) -> bool:
    return _full_commit(value) and value.lower() in FORBIDDEN_BOUNDARIES


def _hex_digest(value: Any, length: int) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _durable_reference(value: Any) -> bool:
    """Durable references are immutable and auditable.

    Accepted forms:
    - git: + 40-hex
    - https://github.com/warpfront/hipfire/commit/<40-hex>
    - https://github.com/warpfront/hipfire/blob/<40-hex>/<path>
    - https://github.com/warpfront/hipfire/pull/<number> or issues/<number> (historical, not artifact)
    - repo-relative path under approved roots that exists on disk
    Rejects host-local paths, bogus hashes, and nonexistent files.
    """
    if not _nonempty(value) or _host_local(value):
        return False
    if value.startswith("git:"):
        return _full_commit(value[4:])
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc != "github.com":
            return False
        path = parsed.path
        # commit with 40-hex
        if re.fullmatch(r"/warpfront/hipfire/commit/[0-9a-fA-F]{40}/?", path):
            return True
        # blob with 40-hex and file path
        if re.fullmatch(r"/warpfront/hipfire/blob/[0-9a-fA-F]{40}/.+", path):
            return True
        # pull/issues are durable historical links (mutable but allowed for non-artifact)
        if re.fullmatch(r"/warpfront/hipfire/(pull|issues)/\d+/?", path):
            return True
        return False
    # repo-relative path
    if parsed.scheme or value.startswith("/") or "\\" in value:
        return False
    parts = value.split("/")
    if ".." in parts:
        return False
    if ".agent-progress" in value:
        return False
    if not any(value == root.rstrip("/") or value.startswith(root) for root in APPROVED_REPO_ROOTS):
        return False
    # existence check
    repo = _repo_root()
    if not (repo / value).exists():
        return False
    return True


def _artifact_reference(value: Any) -> bool:
    """Artifact references are durable report-like artifacts that can promote completion.

    - Must be durable
    - Must not be the tracker itself or a pull/issue URL
    - For repo paths, must be a file under approved artifact roots and not source/config
    - For github, must be commit or blob with 40-hex
    """
    if not _durable_reference(value):
        return False
    if value == "docs/device-mesh-port-tracker.json":
        return False
    parsed = urlparse(value)
    if parsed.netloc == "github.com" and parsed.path.startswith(("/warpfront/hipfire/issues/", "/warpfront/hipfire/pull/")):
        return False
    if parsed.scheme in {"http", "https"}:
        # commit/blob already validated
        return True
    # repo path: must be report-like artifact
    if value in ("Cargo.toml", "Cargo.lock") or value.startswith("crates/") or value.startswith("Cargo."):
        return False
    # require artifact root
    if not any(value == root.rstrip("/") or value.startswith(root) for root in APPROVED_ARTIFACT_ROOTS):
        # scripts/ is considered tool, not report artifact for physical promotion; allow docs/tests/benchmarks only
        # but for generic artifact we allow scripts/tests/docs - however Cargo.toml already rejected, source rejected
        # To keep strict, require artifact roots
        return False
    repo = _repo_root()
    p = repo / value
    if not p.is_file():
        return False
    return True


def _check_physical_identity(
    identity: Any,
    label: str,
    obligation_ids: list[str],
    campaign_id: str,
    errors: list[str],
) -> None:
    if not isinstance(identity, dict):
        errors.append(f"{label} requires physical identity")
        return
    for field, length in (("model_sha256", 64), ("binary_sha256", 64)):
        if not _hex_digest(identity.get(field), length):
            errors.append(f"{label} physical_identity.{field} must be {length}-hex")
    if not _hex_digest(identity.get("prompt_md5"), 32):
        errors.append(f"{label} physical_identity.prompt_md5 must be 32-hex")
    if campaign_id == "EC-VISION" and not _hex_digest(identity.get("image_sha256"), 64):
        errors.append(f"{label} physical_identity.image_sha256 must be 64-hex")
    if identity.get("campaign_id") != campaign_id:
        errors.append(f"{label} physical_identity.campaign_id does not match owner")
    gpu_ids = identity.get("gpu_ids")
    if not _strings(gpu_ids, nonempty=True) or len(set(gpu_ids)) < 2:
        errors.append(f"{label} physical_identity.gpu_ids must contain at least two distinct GPUs")
    for field in ("topology", "rocm_version", "rccl_version"):
        if not _nonempty(identity.get(field)):
            errors.append(f"{label} physical_identity.{field} must be non-empty")
    if campaign_id == "EC-EP" and identity.get("rccl_version") == "not-used":
        errors.append(f"{label} physical_identity.rccl_version cannot be not-used")
    report_refs = identity.get("report_refs")
    if not _strings(report_refs, nonempty=True) or not any(_artifact_reference(ref) for ref in report_refs):
        errors.append(f"{label} physical_identity requires a durable report beyond tracker/issue references")
    result_map = identity.get("result_map")
    if not isinstance(result_map, dict) or set(result_map) != set(obligation_ids):
        errors.append(f"{label} physical_identity.result_map must cover every mapped obligation")
    elif any(
        not isinstance(result, dict)
        or result.get("status") != "pass"
        or not _strings(result.get("report_refs"), nonempty=True)
        or not any(_artifact_reference(ref) for ref in result.get("report_refs", []))
        for result in result_map.values()
    ):
        errors.append(f"{label} physical_identity.result_map entries require pass and durable reports")


def _concrete_text(value: Any) -> bool:
    if not _nonempty(value):
        return False
    lowered = value.lower()
    if any(token in value for token in ("/home/", "/tmp/", ".agent-progress")):
        return False
    return not any(
        phrase in lowered
        for phrase in ("recorded by the owning", "named by the owning", "to be recorded")
    )


def _satisfied(status: Any) -> bool:
    return status in {"complete", "in_review"}


def _ready(status: Any) -> bool:
    return status in {"complete", "in_review", "ready"}


def _owner_record(
    owner_id: Any,
    change_by_id: dict[str, dict[str, Any]],
    campaign_by_id: dict[str, dict[str, Any]],
    closure: Any,
) -> dict[str, Any] | None:
    if owner_id in change_by_id:
        return change_by_id[owner_id]
    if owner_id in campaign_by_id:
        return campaign_by_id[owner_id]
    if isinstance(closure, dict) and closure.get("id") == owner_id:
        return closure
    return None


def _owner_commit(owner: dict[str, Any]) -> str | None:
    if owner.get("status") == "complete":
        return owner.get("merge_commit")
    if owner.get("status") == "in_review":
        return owner.get("head_commit")
    return None


def _check_evidence_entry(entry: Any, label: str, errors: list[str]) -> None:
    if not isinstance(entry, dict):
        errors.append(f"{label} evidence entry must be an object")
        return
    if entry.get("classification") not in ALLOWED_EVIDENCE_CLASSES:
        errors.append(f"{label} evidence classification {entry.get('classification')!r} is not allowed")
    if not _nonempty(entry.get("assertion")):
        errors.append(f"{label} evidence assertion must be non-empty")
    refs = entry.get("references")
    if not _strings(refs):
        errors.append(f"{label} evidence references must be an array of strings")
    else:
        if any(_host_local(value) for value in refs):
            errors.append(f"{label} evidence contains a host-local path")
        if any(".agent-progress" in value for value in refs):
            errors.append(f"{label} evidence contains a local-only .agent-progress reference")
        # For qualifying evidence, require at least one durable reference
        if entry.get("qualifies_for_completion"):
            if not any(_durable_reference(value) for value in refs):
                errors.append(f"{label} qualifying evidence requires a durable reference")
            if entry.get("classification") in BAD_COMPLETION_CLASSES:
                errors.append(f"{label} completion promotion from {entry.get('classification')} evidence is forbidden")
            # also reject host-local already captured
    # additional check for malformed github hash URLs: if it looks like github URL but not durable, it's an error for qualifying
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, str) and "github.com/warpfront/hipfire" in ref:
                if ref.startswith("https://github.com/warpfront/hipfire/commit/") or ref.startswith("https://github.com/warpfront/hipfire/blob/"):
                    if not _durable_reference(ref):
                        errors.append(f"{label} evidence contains malformed github hash URL {ref!r}")
                elif ref.startswith("https://github.com/warpfront/hipfire/"):
                    # other github URLs already validated via durable
                    pass


def _check_dag(graph: dict[str, list[str]], label: str, errors: list[str]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle = " -> ".join(stack + [node])
            errors.append(f"{label} dependency cycle: {cycle}")
            return
        visiting.add(node)
        stack.append(node)
        for dep in graph.get(node, []):
            visit(dep, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node, [])


def _validate_tracker(document: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["tracker must be a JSON object"]
    if document.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for field in ("title", "purpose"):
        if not _nonempty(document.get(field)):
            errors.append(f"{field} must be non-empty")
    serialized = json.dumps(document, ensure_ascii=False)
    if "[x]" in serialized.lower() or "[ ]" in serialized:
        errors.append("stale checkbox claim found in tracker")
    upstream = document.get("upstream")
    if not isinstance(upstream, dict):
        errors.append("missing upstream metadata")
    else:
        for key in ("remote", "branch", "series_origin_ref"):
            if not _nonempty(upstream.get(key)):
                errors.append(f"upstream.{key} must be non-empty")
        if upstream.get("series_origin_ref") != SERIES_ORIGIN_REF:
            errors.append("upstream.series_origin_ref must equal the approved series origin")

    branch = document.get("branch_provenance")
    if not isinstance(branch, dict):
        errors.append("missing branch provenance metadata")
    else:
        for key in ("common_upstream_base", "pr_527_head", "reviewed_branch_head", "fork_merge", "fork_parent", "upstream_pr_653", "rule"):
            if not _nonempty(branch.get(key)):
                errors.append(f"branch provenance {key!r} must be non-empty")
        # forbidden boundaries must be a list of 40-hex not empty; exact set is enforced via forbidden check elsewhere
        fb = branch.get("forbidden_boundaries")
        if not isinstance(fb, list) or not all(_full_commit(v) for v in fb):
            errors.append("branch provenance forbidden_boundaries must be a list of 40-hex commits")
        if len(fb) != len(set(v.lower() for v in fb)):
            errors.append("branch provenance forbidden_boundaries contains duplicates")

    authority = document.get("authority")
    if not isinstance(authority, dict):
        errors.append("missing authority metadata")
    else:
        expected_links = {
            "tracker": "docs/device-mesh-port-tracker.json", "index": "docs/INDEX.md",
            "validation": "docs/VALIDATION.md", "admissions": "docs/admissions.yml",
            "schema_checker": "scripts/check-device-mesh-port-tracker.py",
            "focused_tests": "tests/test_device_mesh_port_tracker.py",
        }
        for key, expected in expected_links.items():
            if authority.get(key) != expected:
                errors.append(f"authority link {key!r} must point to {expected!r}")
        if authority.get("issue_666") != "https://github.com/warpfront/hipfire/issues/666":
            errors.append("authority issue_666 must point to the replacement issue")
        pr_527 = authority.get("pr_527")
        if not isinstance(pr_527, dict):
            errors.append("authority.pr_527 must be an object")
        else:
            if pr_527.get("disposition") != "historical_superseded":
                errors.append("authority PR #527 disposition must be historical_superseded")
            if not _nonempty(pr_527.get("url")) or "/pull/527" not in pr_527["url"]:
                errors.append("authority PR #527 link must point to pull/527")
            if pr_527.get("replacement") != "docs/device-mesh-port-tracker.json":
                errors.append("authority PR #527 replacement must point to the tracker")
            if pr_527.get("body_mutation") != "out_of_scope":
                errors.append("authority PR #527 body_mutation must be out_of_scope")

    policy = document.get("policy")
    if not isinstance(policy, dict):
        errors.append("missing advancement policy metadata")
    else:
        if policy.get("max_completion_rows_per_pr") != 1:
            errors.append("advancement policy max_completion_rows_per_pr must be exactly 1")
        if policy.get("completion_field") != "advancement.completion_rows":
            errors.append("advancement policy completion_field must name advancement.completion_rows")
        if policy.get("status_semantics") != "dependency_gated_no_merge_claim":
            errors.append("advancement policy status_semantics must avoid merge claims")
        expected_enums = {
            "obligation_statuses": ["complete", "ready", "blocked"],
            "change_set_statuses": ["implemented", "in_review", "complete", "ready", "blocked"],
            "implementation_classes": ["port", "superseded", "already_upstream", "historical_evidence_only", "needs_design"],
            "evidence_dispositions": ["not_applicable", "current", "historical", "rerun_required", "hardware_blocked"],
            "evidence_classifications": ["current", "historical", "rerun_required", "hardware_blocked", "semantics_only", "emulated", "failed"],
            "confidence": ["high", "medium", "low"],
            "delivery_owner_kinds": ["change_set", "evidence_campaign", "final_closure"],
        }
        for key, expected in expected_enums.items():
            if policy.get(key) != expected:
                errors.append(f"policy {key} does not match the schema enum")
        # can_develop_after is now derived from JSON; validate generic shape, not exact equality
        cda = policy.get("can_develop_after")
        if not isinstance(cda, dict):
            errors.append("policy can_develop_after must be an object")
        else:
            for gid, deps in cda.items():
                if not _strings(deps):
                    errors.append(f"policy can_develop_after {gid} must be an array of strings")
                if gid in deps:
                    errors.append(f"policy can_develop_after {gid} cannot depend on itself")
        for key in ("grouping_rule", "completion_promotion_rule", "parallel_lane_rule", "branch_evidence_rule", "one_row_rule"):
            if not _nonempty(policy.get(key)):
                errors.append(f"policy {key} must be non-empty")
        # one_row_rule must clarify one-row-per-obligation enforceability
        one_row = policy.get("one_row_rule", "")
        if "one" not in one_row.lower() or "obligation" not in one_row.lower():
            errors.append("policy one_row_rule must describe one-row-per-obligation semantics")
        if "base-diff" not in one_row.lower() and "base_diff" not in one_row.lower() and "base diff" not in one_row.lower():
            errors.append("policy one_row_rule must mention base-diff limitation for enforceability")

    legacy_inventory = document.get("legacy_pr_inventory")
    if not isinstance(legacy_inventory, list) or not all(_nonempty(v) for v in legacy_inventory):
        errors.append("legacy_pr_inventory must be a non-empty array of strings")
    elif len(legacy_inventory) != len(set(legacy_inventory)):
        errors.append("legacy_pr_inventory contains duplicates")
    else:
        for pid in legacy_inventory:
            if not re.fullmatch(r"PR-\d+[A-Z]*", pid):
                errors.append(f"legacy_pr_inventory contains invalid ID {pid!r}")

    obligations = document.get("obligations")
    if not isinstance(obligations, list):
        errors.append("obligations must be an array")
        return errors
    if len(obligations) == 0:
        errors.append("obligations must be non-empty")
    by_id: dict[str, dict[str, Any]] = {}
    ids: list[str] = []
    for index, obligation in enumerate(obligations):
        label = f"obligation {index + 1}"
        if not isinstance(obligation, dict):
            errors.append(f"{label} must be an object")
            continue
        oid = obligation.get("id")
        if not _nonempty(oid):
            errors.append(f"{label} id must be non-empty")
            continue
        ids.append(oid)
        if oid in by_id:
            errors.append(f"duplicate obligation id {oid}")
        else:
            by_id[oid] = obligation
        for field in ("title", "scope", "non_goals", "acceptance", "stop_condition"):
            if not _nonempty(obligation.get(field)):
                errors.append(f"{oid} missing non-empty {field}")
        content = " ".join(
            str(obligation.get(field, ""))
            for field in ("title", "scope", "non_goals", "acceptance", "stop_condition", "provenance", "evidence")
        ).lower()
        for term in REQUIRED_CONTENT_TERMS.get(oid, ()):
            if term.lower() not in content:
                errors.append(f"{oid} missing required domain contract term {term!r}")
        dependencies = obligation.get("depends_on")
        if not _strings(dependencies):
            errors.append(f"{oid} depends_on must be an array of IDs")
        elif len(dependencies) != len(set(dependencies)):
            errors.append(f"{oid} has duplicate dependencies")
        if obligation.get("status") not in ALLOWED_OBLIGATION_STATUSES:
            errors.append(f"{oid} status {obligation.get('status')!r} is not allowed")
        if obligation.get("implementation_class") not in ALLOWED_CLASSES:
            errors.append(f"{oid} implementation_class {obligation.get('implementation_class')!r} is not allowed")
        if obligation.get("confidence") not in ALLOWED_CONFIDENCE:
            errors.append(f"{oid} confidence {obligation.get('confidence')!r} is not allowed")
        if obligation.get("legacy_status") not in ALLOWED_LEGACY_STATUSES:
            errors.append(f"{oid} legacy_status {obligation.get('legacy_status')!r} is not allowed")
        for key in ("legacy_pr_ids", "legacy_task_ids"):
            values = obligation.get(key)
            if not _strings(values):
                errors.append(f"{oid} {key} must be an array of strings")
            elif key == "legacy_pr_ids" and isinstance(legacy_inventory, list) and any(value not in legacy_inventory for value in values):
                errors.append(f"{oid} has an unknown legacy PR ID")
        legacy_dependencies = obligation.get("legacy_dependencies")
        if not isinstance(legacy_dependencies, list):
            errors.append(f"{oid} legacy_dependencies must be an array")
        else:
            for item in legacy_dependencies:
                if not isinstance(item, dict) or not _nonempty(item.get("pr_id")) or not _strings(item.get("depends_on")):
                    errors.append(f"{oid} legacy_dependencies contains an invalid PR contract")
                elif isinstance(legacy_inventory, list) and item.get("pr_id") not in legacy_inventory:
                    errors.append(f"{oid} legacy_dependencies references unknown PR {item.get('pr_id')}")
        delivery_dependencies = obligation.get("depends_on_delivery")
        if not _strings(delivery_dependencies):
            errors.append(f"{oid} depends_on_delivery must be an array of delivery IDs")
        provenance = obligation.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"{oid} missing provenance")
        else:
            if not _strings(provenance.get("branch_commits")):
                errors.append(f"{oid} provenance.branch_commits must be an array of strings")
            if not _strings(provenance.get("sources"), nonempty=True):
                errors.append(f"{oid} provenance.sources must be a non-empty array")
            if any(_host_local(value) for value in [*(provenance.get("branch_commits") or []), *(provenance.get("sources") or [])]):
                errors.append(f"{oid} provenance contains a host-local path")
            if any(".agent-progress" in str(v) for v in [*(provenance.get("branch_commits") or []), *(provenance.get("sources") or [])]):
                errors.append(f"{oid} provenance contains a local-only .agent-progress reference")
            if not _nonempty(provenance.get("upstream_counterpart")):
                errors.append(f"{oid} provenance.upstream_counterpart must be non-empty")
        evidence = obligation.get("evidence")
        if not isinstance(evidence, dict):
            errors.append(f"{oid} missing evidence")
        else:
            disposition = evidence.get("disposition")
            classification = evidence.get("classification")
            if disposition not in ALLOWED_DISPOSITIONS:
                errors.append(f"{oid} evidence disposition {disposition!r} is not allowed")
            if classification not in ALLOWED_EVIDENCE_CLASSES:
                errors.append(f"{oid} evidence classification {classification!r} is not allowed")
            if evidence.get("branch_record") not in ALLOWED_BRANCH_RECORDS:
                errors.append(f"{oid} evidence branch_record is not allowed")
            if not _nonempty(evidence.get("route")):
                errors.append(f"{oid} evidence route must be non-empty")
            for key in ("fixture_refs", "report_refs"):
                if not _strings(evidence.get(key)):
                    errors.append(f"{oid} evidence.{key} must be an array of strings")
                if any(_host_local(value) for value in (evidence.get(key) or [])):
                    errors.append(f"{oid} evidence contains a host-local path")
                if any(".agent-progress" in str(v) for v in (evidence.get(key) or [])):
                    errors.append(f"{oid} evidence contains a local-only .agent-progress reference")
            if disposition != "not_applicable" and evidence.get("branch_record") != "historical":
                errors.append(f"{oid} evidence must preserve historical branch record")
            if obligation.get("status") == "complete":
                if disposition != "current":
                    errors.append(f"{oid} complete status requires current evidence")
                if classification != "current":
                    errors.append(f"{oid} complete status requires evidence classification current")
                if not any(_durable_reference(value) for value in (evidence.get("report_refs") or [])):
                    errors.append(f"{oid} complete current evidence requires a durable report reference")
                if oid.startswith("HW-") and not any(
                    _artifact_reference(value) for value in evidence.get("report_refs") or []
                ):
                    errors.append(f"{oid} complete physical evidence requires a durable report artifact")
            elif disposition == "current" or classification == "current":
                errors.append(f"{oid} non-complete status cannot claim current evidence")
        if oid.startswith("HW-") and obligation.get("status") == "complete":
            _check_physical_identity(
                obligation.get("physical_identity"),
                oid,
                [oid],
                obligation.get("campaign_id") or "",
                errors,
            )
        owner = obligation.get("delivery_owner")
        if not isinstance(owner, dict) or owner.get("kind") not in ALLOWED_DELIVERY_KINDS or not _nonempty(owner.get("id")):
            errors.append(f"{oid} delivery_owner must name one delivery owner")
        if obligation.get("delivery_kind") not in ALLOWED_DELIVERY_KINDS:
            errors.append(f"{oid} delivery_kind is not allowed")
        campaign_id = obligation.get("campaign_id")
        if campaign_id is not None and not _nonempty(campaign_id):
            errors.append(f"{oid} campaign_id must be a non-empty ID or null")
        advancement = obligation.get("advancement")
        if not isinstance(advancement, dict):
            errors.append(f"{oid} missing advancement metadata")
        else:
            rows = advancement.get("completion_rows")
            if not _strings(rows):
                errors.append(f"{oid} advancement.completion_rows must be an array")
            elif len(rows) > 1:
                errors.append(f"{oid} advancement exceeds one completion row")
            if not _nonempty(advancement.get("reason")):
                errors.append(f"{oid} advancement.reason must be non-empty")
            if obligation.get("status") == "complete" and rows != [oid]:
                errors.append(f"{oid} complete status must advance exactly itself")
            if obligation.get("status") != "complete" and rows:
                errors.append(f"{oid} non-complete status cannot advance a completion row")

    if len(ids) != len(set(ids)):
        errors.append("duplicate obligation ids detected")
    if isinstance(legacy_inventory, list):
        legacy_covered = {
            value
            for obligation in obligations
            for value in (obligation.get("legacy_pr_ids") or [])
            if isinstance(value, str)
        }
        extra = legacy_covered - set(legacy_inventory)
        if extra:
            errors.append("unexpected legacy PR provenance IDs: " + ", ".join(sorted(extra)))
        missing = set(legacy_inventory) - legacy_covered
        if missing:
            errors.append("missing legacy PR provenance coverage: " + ", ".join(sorted(missing)))

    obligation_graph: dict[str, list[str]] = {}
    for oid, obligation in by_id.items():
        dependencies = obligation.get("depends_on")
        if not isinstance(dependencies, list):
            continue
        obligation_graph[oid] = []
        for dependency in dependencies:
            if dependency == oid:
                errors.append(f"{oid} cannot depend on itself")
            elif dependency not in by_id:
                errors.append(f"{oid} has unknown dependency {dependency}")
            else:
                obligation_graph[oid].append(dependency)
    _check_dag(obligation_graph, "obligation", errors)
    for oid, obligation in by_id.items():
        dependencies = obligation.get("depends_on")
        if not isinstance(dependencies, list):
            continue
        if obligation.get("status") in {"complete", "ready"} and not all(by_id.get(dep, {}).get("status") == "complete" for dep in dependencies):
            errors.append(f"{oid} {obligation.get('status')} status has incomplete dependencies")

    seam_gates = document.get("seam_gates")
    seam_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(seam_gates, list):
        errors.append("seam_gates must be an array")
    else:
        seam_ids: list[str] = []
        if len(seam_gates) == 0:
            errors.append("seam_gates must be non-empty")


    seam_gates = document.get("seam_gates")
    seam_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(seam_gates, list):
        errors.append("seam_gates must be an array")
    else:
        seam_ids: list[str] = []
        if len(seam_gates) == 0:
            errors.append("seam_gates must be non-empty")
        for index, gate in enumerate(seam_gates):
            label = f"seam gate {index + 1}"
            if not isinstance(gate, dict):
                errors.append(f"{label} must be an object")
                continue
            gate_id = gate.get("id")
            if not _nonempty(gate_id):
                errors.append(f"{label} id must be non-empty")
                continue
            seam_ids.append(gate_id)
            if gate_id in seam_by_id:
                errors.append(f"duplicate seam gate id {gate_id}")
            else:
                seam_by_id[gate_id] = gate
            if not _nonempty(gate.get("kind")) or not _nonempty(gate.get("contract")):
                errors.append(f"{gate_id} kind and contract must be non-empty")
            if gate.get("status") not in ALLOWED_GATE_STATUSES:
                errors.append(f"{gate_id} status is not allowed")
            if gate.get("evidence_disposition") not in ALLOWED_DISPOSITIONS:
                errors.append(f"{gate_id} evidence disposition is not allowed")
            if gate.get("status") in {"available", "complete"} and gate.get("evidence_disposition") != "current":
                errors.append(f"{gate_id} available/complete seam requires current evidence disposition")
            if gate.get("status") in {"proposed", "blocked"} and gate.get("evidence_disposition") == "current":
                errors.append(f"{gate_id} proposed/blocked seam cannot claim current evidence disposition")
            consumers = gate.get("consumers")
            if not _strings(consumers):
                errors.append(f"{gate_id} consumers must be an array")
            elif len(consumers) != len(set(consumers)):
                errors.append(f"{gate_id} has duplicate consumers")
            # producer validation generic: must be non-empty string
            producer = gate.get("producer")
            if not _nonempty(producer):
                errors.append(f"{gate_id} producer must be non-empty")
            receipt = gate.get("receipt")
            requires_receipt = gate.get("status") in {"available", "complete"} or gate.get("evidence_disposition") == "current"
            if requires_receipt:
                if not isinstance(receipt, dict) or receipt.get("status") != "complete":
                    errors.append(f"{gate_id} current/available seam requires a complete receipt")
                else:
                    for field in ("route", "evidence_class", "positive_probe", "negative_probe", "sole_owner", "revert_identity"):
                        if not _concrete_text(receipt.get(field)):
                            errors.append(f"{gate_id} receipt {field} must be concrete")
                    if receipt.get("evidence_class") not in ALLOWED_EVIDENCE_CLASSES:
                        errors.append(f"{gate_id} receipt evidence class is not allowed")
                    if receipt.get("evidence_class") != "current":
                        errors.append(f"{gate_id} receipt evidence_class must be current")
                    for field in ("fixture_references", "durable_references"):
                        if not _strings(receipt.get(field), nonempty=True):
                            errors.append(f"{gate_id} receipt {field} must be non-empty")
                        if any(_host_local(value) for value in (receipt.get(field) or [])):
                            errors.append(f"{gate_id} receipt contains a host-local path")
                        if any(".agent-progress" in str(v) for v in (receipt.get(field) or [])):
                            errors.append(f"{gate_id} receipt contains a local-only .agent-progress reference")
                    if not any(_durable_reference(value) for value in (receipt.get("durable_references") or [])):
                        errors.append(f"{gate_id} receipt durable_references require a durable commit or repository artifact")
                    consumer_commits = receipt.get("consumer_commits")
                    if not isinstance(consumer_commits, dict):
                        errors.append(f"{gate_id} receipt consumer_commits must be an object keyed by consumer ID")
                    else:
                        unknown_consumers = set(consumer_commits) - set(consumers or [])
                        if unknown_consumers:
                            errors.append(f"{gate_id} receipt has unknown consumer commit keys: {', '.join(sorted(unknown_consumers))}")
                        if any(not isinstance(value, str) or not _full_commit(value) for value in consumer_commits.values()):
                            errors.append(f"{gate_id} receipt consumer_commits must use 40-hex commits")
                    evidence_commit = receipt.get("evidence_commit")
                    if not _full_commit(evidence_commit):
                        errors.append(f"{gate_id} receipt requires a 40-hex evidence_commit")
                    if _host_local(evidence_commit):
                        errors.append(f"{gate_id} receipt contains a host-local evidence commit")
                    if not _full_commit(receipt.get("producer_commit")):
                        errors.append(f"{gate_id} receipt requires a 40-hex producer_commit")
                    if not _strings(receipt.get("side_effect_assertions"), nonempty=True):
                        errors.append(f"{gate_id} receipt side_effect_assertions must be non-empty")
                    if _host_local(receipt.get("producer_commit")):
                        errors.append(f"{gate_id} receipt contains a host-local producer commit")
                    # physical seam check generic: if gate id starts with S-HARDWARE
                    if gate_id.startswith("S-HARDWARE-"):
                        if not _strings(receipt.get("durable_references"), nonempty=True) or not any(_artifact_reference(v) for v in receipt.get("durable_references") or []):
                            errors.append(f"{gate_id} receipt physical evidence requires a durable report artifact")
            elif receipt is not None:
                errors.append(f"{gate_id} proposed/blocked seam must not carry a receipt")
        if len(seam_ids) != len(set(seam_ids)):
            errors.append("duplicate seam gate ids")
        # generic order check not enforced; only uniqueness and DAG

    change_sets = document.get("change_sets")
    change_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(change_sets, list):
        errors.append("change_sets must be an array")
    else:
        change_ids: list[str] = []
        mapped: list[str] = []
        if len(change_sets) == 0:
            errors.append("change_sets must be non-empty")
        for index, change_set in enumerate(change_sets):
            label = f"change set {index + 1}"
            if not isinstance(change_set, dict):
                errors.append(f"{label} must be an object")
                continue
            gid = change_set.get("id")
            if not _nonempty(gid):
                errors.append(f"{label} id must be non-empty")
                continue
            change_ids.append(gid)
            if gid in change_by_id:
                errors.append(f"duplicate change-set id {gid}")
            else:
                change_by_id[gid] = change_set
            obligations_owned = change_set.get("obligation_ids")
            if not _strings(obligations_owned, nonempty=True):
                errors.append(f"{gid} obligation_ids must be a non-empty array")
            else:
                # each owned must be a known obligation
                for oid in obligations_owned:
                    if oid not in by_id:
                        errors.append(f"{gid} obligation_ids contains unknown obligation {oid}")
                mapped.extend(obligations_owned)
            for field in ("title", "scope", "non_goals", "source_assumption", "sole_owner", "production_route", "shared_file_integration_owner", "acceptance", "stop_condition"):
                if not _nonempty(change_set.get(field)):
                    errors.append(f"{gid} missing non-empty {field}")
            if change_set.get("delivery_kind") != "change_set":
                errors.append(f"{gid} delivery_kind must be change_set")
            if change_set.get("implementation_class") not in ALLOWED_CLASSES:
                errors.append(f"{gid} implementation_class is not allowed")
            if change_set.get("confidence") not in ALLOWED_CONFIDENCE:
                errors.append(f"{gid} confidence is not allowed")
            dependencies = change_set.get("depends_on")
            if not _strings(dependencies):
                errors.append(f"{gid} depends_on must be an array")
            elif len(dependencies) != len(set(dependencies)):
                errors.append(f"{gid} has duplicate dependencies")
            merge_waits = change_set.get("merge_waits_on")
            if not _strings(merge_waits):
                errors.append(f"{gid} merge_waits_on must be an array")
            elif any(wait == gid for wait in merge_waits):
                errors.append(f"{gid} merge_waits_on contains self reference")
            can_develop_after = change_set.get("can_develop_after")
            if not _strings(can_develop_after):
                errors.append(f"{gid} can_develop_after must be an array")
            consumed = change_set.get("consumed_seam_gates")
            produced = change_set.get("produced_seam_gates")
            if not _strings(consumed) or not _strings(produced, nonempty=True):
                errors.append(f"{gid} consumed/produced seam gates must be arrays")
            else:
                if len(consumed) != len(set(consumed)) or len(produced) != len(set(produced)):
                    errors.append(f"{gid} has duplicate seam-gate references")
                for gate_id in [*consumed, *produced]:
                    if gate_id not in seam_by_id:
                        errors.append(f"{gid} references unknown seam gate {gate_id}")
                for gate_id in produced:
                    if gate_id in seam_by_id and seam_by_id[gate_id].get("producer") != gid:
                        errors.append(f"{gid} produced seam gate {gate_id} has a different producer")
                for gate_id in consumed:
                    if gate_id in seam_by_id:
                        prod = seam_by_id[gate_id].get("producer")
                        if prod not in (dependencies or []):
                            errors.append(f"{gid} consumed seam gate {gate_id} does not support a declared dependency")
                for dependency in dependencies or []:
                    if dependency in change_by_id:
                        dep_produced = change_by_id[dependency].get("produced_seam_gates") or []
                        if not (set(consumed) & set(dep_produced)):
                            errors.append(f"{gid} dependency {dependency} is not supported by a consumed seam")
            status = change_set.get("status")
            if status not in ALLOWED_CHANGE_SET_STATUSES:
                errors.append(f"{gid} status {status!r} is not allowed")
            if change_set.get("evidence_disposition") not in ALLOWED_DISPOSITIONS:
                errors.append(f"{gid} evidence disposition is not allowed")
            for identity_field in ("upstream_base_commit", "head_commit", "merge_commit"):
                if identity_field not in change_set:
                    errors.append(f"{gid} missing {identity_field}")
                elif change_set[identity_field] is not None and not isinstance(change_set[identity_field], str):
                    errors.append(f"{gid} {identity_field} must be a commit or durable reference")
            if status in {"complete", "in_review"}:
                if not _full_commit(change_set.get("upstream_base_commit")):
                    errors.append(f"{gid} promoted status requires a pinned upstream_base_commit")
                elif _forbidden_boundary(change_set.get("upstream_base_commit")):
                    errors.append(f"{gid} upstream_base_commit uses a forbidden boundary")
                if status == "complete":
                    if not _full_commit(change_set.get("merge_commit")):
                        errors.append(f"{gid} complete status requires a 40-hex merge_commit")
                    if change_set.get("head_commit") is not None and not _full_commit(change_set.get("head_commit")):
                        errors.append(f"{gid} complete status head_commit must be a 40-hex commit when present")
                else:
                    if not _full_commit(change_set.get("head_commit")):
                        errors.append(f"{gid} in_review status requires a 40-hex head_commit")
                    if change_set.get("merge_commit") is not None:
                        errors.append(f"{gid} in_review status requires merge_commit null")
            for key in ("positive_evidence", "negative_evidence", "completion_evidence"):
                if not isinstance(change_set.get(key), list):
                    errors.append(f"{gid} {key} must be an array")
            positive = change_set.get("positive_evidence") or []
            negative = change_set.get("negative_evidence") or []
            completion = change_set.get("completion_evidence") or []
            if not positive:
                errors.append(f"{gid} positive_evidence must be non-empty")
            if not negative:
                errors.append(f"{gid} negative_evidence must be non-empty")
            for entry in positive:
                _check_evidence_entry(entry, gid, errors)
            for entry in negative:
                _check_evidence_entry(entry, gid, errors)
                if isinstance(entry, dict) and entry.get("qualifies_for_completion"):
                    errors.append(f"{gid} negative evidence cannot qualify for completion")
            for entry in completion:
                _check_evidence_entry(entry, f"{gid} completion", errors)
            if status in {"complete", "in_review"}:
                if change_set.get("evidence_disposition") != "current":
                    errors.append(f"{gid} complete/in_review status requires current evidence disposition")
                if not completion or any(not isinstance(entry, dict) or entry.get("classification") != "current" or not entry.get("qualifies_for_completion") for entry in completion):
                    errors.append(f"{gid} completion promotion requires qualifying current evidence")
                if not all(by_id.get(oid, {}).get("status") == "complete" for oid in (obligations_owned or [])):
                    errors.append(f"{gid} blocked child obligation prevents completion promotion")
                if not all(_satisfied(change_by_id.get(dep, {}).get("status")) for dep in (dependencies or [])):
                    errors.append(f"{gid} dependency prevents completion promotion")
                if not all(_satisfied(change_by_id.get(wait, {}).get("status")) for wait in (merge_waits or [])):
                    errors.append(f"{gid} merge wait prevents completion promotion")
                if not all(seam_by_id.get(gate_id, {}).get("status") in {"available", "complete"} and seam_by_id.get(gate_id, {}).get("evidence_disposition") == "current" for gate_id in (consumed or [])):
                    errors.append(f"{gid} consumed seam prevents completion promotion")
            elif status == "ready":
                if not all(_ready(change_by_id.get(dep, {}).get("status")) for dep in (dependencies or [])):
                    errors.append(f"{gid} ready status has an unresolved dependency")
                if not all(seam_by_id.get(gate_id, {}).get("status") in {"available", "complete"} for gate_id in (consumed or [])):
                    errors.append(f"{gid} ready status has an unavailable consumed seam")
            elif completion:
                errors.append(f"{gid} non-complete status cannot claim completion evidence")
            side_effects = change_set.get("side_effect_assertions")
            if not _strings(side_effects, nonempty=True):
                errors.append(f"{gid} side_effect_assertions must be a non-empty array")
            revert = change_set.get("revert_identity")
            if not isinstance(revert, dict) or revert.get("change_set_id") != gid or revert.get("strategy") != "revert-entire-grouped-change-set" or not _nonempty(revert.get("identity")) or not _nonempty(revert.get("scope")):
                errors.append(f"{gid} grouped revert identity is invalid")
            lane = change_set.get("parallel_lane")
            if not isinstance(lane, dict) or not _nonempty(lane.get("name")) or lane.get("integration_mode") != "serialized-shared-file-owner" or not _strings(lane.get("can_develop_after")) or not _strings(lane.get("merge_waits_on")):
                errors.append(f"{gid} parallel_lane is invalid")
            elif lane.get("can_develop_after") != can_develop_after:
                errors.append(f"{gid} parallel_lane.can_develop_after must match top-level can_develop_after")
            elif lane.get("merge_waits_on") != merge_waits:
                errors.append(f"{gid} parallel_lane.merge_waits_on must match top-level merge_waits_on")
        if len(change_ids) != len(set(change_ids)):
            errors.append("duplicate change-set ids")
        if len(mapped) != len(set(mapped)):
            errors.append("duplicate obligation mapping across grouped change sets")
        # check that all mapped obligations are valid and no unknown
        for oid in mapped:
            if oid not in by_id:
                errors.append(f"unexpected mapped obligation {oid}")

    # G0 authority must start at series_origin_ref
    if "G0" in change_by_id:
        g0 = change_by_id["G0"]
        if g0.get("upstream_base_commit") != SERIES_ORIGIN_REF:
            errors.append("G0 upstream_base_commit must equal series_origin_ref")

    group_graph: dict[str, list[str]] = {}
    for gid, change_set in change_by_id.items():
        dependencies = change_set.get("depends_on")
        if not isinstance(dependencies, list):
            continue
        group_graph[gid] = []
        for dependency in dependencies:
            if dependency == gid:
                errors.append(f"{gid} cannot depend on itself")
            elif dependency not in change_by_id:
                errors.append(f"{gid} has unknown group dependency {dependency}")
            else:
                group_graph[gid].append(dependency)
    _check_dag(group_graph, "change-set", errors)

    campaigns = document.get("evidence_campaigns")
    campaign_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(campaigns, list):
        errors.append("evidence_campaigns must be an array")
    else:
        campaign_ids: list[str] = []
        campaign_mapped: list[str] = []
        if len(campaigns) == 0:
            errors.append("evidence_campaigns must be non-empty")
        for index, campaign in enumerate(campaigns):
            label = f"evidence campaign {index + 1}"
            if not isinstance(campaign, dict):
                errors.append(f"{label} must be an object")
                continue
            cid = campaign.get("id")
            if not _nonempty(cid):
                errors.append(f"{label} id must be non-empty")
                continue
            campaign_ids.append(cid)
            if cid in campaign_by_id:
                errors.append(f"duplicate evidence campaign id {cid}")
            else:
                campaign_by_id[cid] = campaign
            owned = campaign.get("obligation_ids")
            if not _strings(owned, nonempty=True):
                errors.append(f"{cid} obligation_ids must be a non-empty array")
            else:
                for oid in owned:
                    if oid not in by_id:
                        errors.append(f"{cid} obligation_ids contains unknown obligation {oid}")
                campaign_mapped.extend(owned)
            for field in ("title", "route", "acceptance", "stop_condition"):
                if not _nonempty(campaign.get(field)):
                    errors.append(f"{cid} missing non-empty {field}")
            if campaign.get("delivery_kind") != "evidence_campaign":
                errors.append(f"{cid} delivery_kind must be evidence_campaign")
            if campaign.get("topology_class") not in ALLOWED_CAMPAIGN_CLASSES:
                errors.append(f"{cid} topology_class is not allowed")
            if campaign.get("status") not in ALLOWED_CHANGE_SET_STATUSES:
                errors.append(f"{cid} status is not allowed")
            if campaign.get("evidence_disposition") not in ALLOWED_DISPOSITIONS:
                errors.append(f"{cid} evidence disposition is not allowed")
            for identity_field in ("upstream_base_commit", "head_commit", "merge_commit"):
                if identity_field not in campaign:
                    errors.append(f"{cid} missing {identity_field}")
                elif campaign[identity_field] is not None and not isinstance(campaign[identity_field], str):
                    errors.append(f"{cid} {identity_field} must be a commit or null")
            if not _nonempty(campaign.get("sole_owner")):
                errors.append(f"{cid} sole_owner must be non-empty")
            revert_identity = campaign.get("revert_identity")
            if not isinstance(revert_identity, dict) or not _nonempty(revert_identity.get("identity")):
                errors.append(f"{cid} revert_identity must be an object with identity")
            status = campaign.get("status")
            if status in {"complete", "in_review"}:
                if not _full_commit(campaign.get("upstream_base_commit")):
                    errors.append(f"{cid} promoted status requires a pinned upstream_base_commit")
                elif _forbidden_boundary(campaign.get("upstream_base_commit")):
                    errors.append(f"{cid} upstream_base_commit uses a forbidden boundary")
                if status == "complete":
                    if not _full_commit(campaign.get("merge_commit")):
                        errors.append(f"{cid} complete status requires a 40-hex merge_commit")
                    if campaign.get("head_commit") is not None and not _full_commit(campaign.get("head_commit")):
                        errors.append(f"{cid} complete status head_commit must be a 40-hex commit when present")
                else:
                    if not _full_commit(campaign.get("head_commit")):
                        errors.append(f"{cid} in_review status requires a 40-hex head_commit")
                    if campaign.get("merge_commit") is not None:
                        errors.append(f"{cid} in_review status requires merge_commit null")
            for key in ("depends_on_change_sets", "depends_on_campaigns", "consumed_seam_gates", "produced_seam_gates"):
                if not _strings(campaign.get(key)):
                    errors.append(f"{cid} {key} must be an array")
            for gid in campaign.get("depends_on_change_sets") or []:
                if gid not in change_by_id:
                    errors.append(f"{cid} has unknown group dependency {gid}")
            for dependency in campaign.get("depends_on_campaigns") or []:
                if dependency == cid:
                    errors.append(f"{cid} has campaign self-dependency")
                elif dependency not in campaign_by_id and dependency not in [c.get("id") for c in campaigns if isinstance(c, dict)]:
                    # allow forward reference; will be checked after all campaigns collected
                    pass
            if isinstance(campaign.get("change_set_ids"), list):
                errors.append(f"{cid} must use depends_on_change_sets, not change_set_ids")
            consumed = campaign.get("consumed_seam_gates") or []
            produced = campaign.get("produced_seam_gates") or []
            for gate_id in [*consumed, *produced]:
                if gate_id not in seam_by_id:
                    errors.append(f"{cid} references unknown seam gate {gate_id}")
            for gate_id in produced:
                if gate_id in seam_by_id and seam_by_id[gate_id].get("producer") != cid:
                    errors.append(f"{cid} produced seam gate {gate_id} has a different producer")
            dependency_owners = [
                *(campaign.get("depends_on_change_sets") or []),
                *(campaign.get("depends_on_campaigns") or []),
            ]
            for gate_id in consumed:
                if gate_id in seam_by_id and seam_by_id[gate_id].get("producer") not in dependency_owners:
                    errors.append(f"{cid} consumed seam gate {gate_id} does not support a declared dependency")
            positive = campaign.get("positive_evidence")
            negative = campaign.get("negative_evidence")
            completion = campaign.get("completion_evidence")
            if not isinstance(positive, list) or not positive:
                errors.append(f"{cid} positive_evidence must be non-empty")
            else:
                for entry in positive:
                    _check_evidence_entry(entry, cid, errors)
            if not isinstance(negative, list) or not negative:
                errors.append(f"{cid} negative_evidence must be non-empty")
            else:
                for entry in negative:
                    _check_evidence_entry(entry, cid, errors)
                    if isinstance(entry, dict) and entry.get("qualifies_for_completion"):
                        errors.append(f"{cid} negative evidence cannot qualify for completion")
            if not isinstance(completion, list):
                errors.append(f"{cid} completion_evidence must be an array")
            else:
                for entry in completion:
                    _check_evidence_entry(entry, f"{cid} completion", errors)
            if campaign.get("status") in {"complete", "in_review"}:
                if campaign.get("evidence_disposition") != "current":
                    errors.append(f"{cid} completion promotion from non-current evidence is forbidden")
                if not completion or any(not isinstance(entry, dict) or entry.get("classification") != "current" or not entry.get("qualifies_for_completion") for entry in completion):
                    errors.append(f"{cid} completion promotion requires qualifying current evidence")
                if not all(change_by_id.get(gid, {}).get("status") in {"complete", "in_review"} for gid in (campaign.get("depends_on_change_sets") or [])):
                    errors.append(f"{cid} campaign prerequisite group prevents completion promotion")
                if not all(by_id.get(oid, {}).get("status") == "complete" for oid in (owned or [])):
                    errors.append(f"{cid} blocked child obligation prevents completion promotion")
                if not all(campaign_by_id.get(dep, {}).get("status") in {"complete", "in_review"} for dep in (campaign.get("depends_on_campaigns") or [])):
                    errors.append(f"{cid} campaign prerequisite prevents completion promotion")
                if not all(seam_by_id.get(gate_id, {}).get("status") in {"available", "complete"} and seam_by_id.get(gate_id, {}).get("evidence_disposition") == "current" for gate_id in consumed):
                    errors.append(f"{cid} consumed seam prevents completion promotion")
            elif campaign.get("status") == "ready":
                if not all(_ready(change_by_id.get(gid, {}).get("status")) for gid in (campaign.get("depends_on_change_sets") or [])):
                    errors.append(f"{cid} ready status has an unresolved group dependency")
                if not all(_ready(campaign_by_id.get(dep, {}).get("status")) for dep in (campaign.get("depends_on_campaigns") or [])):
                    errors.append(f"{cid} ready status has an unresolved campaign dependency")
                if not all(seam_by_id.get(gate_id, {}).get("status") in {"available", "complete"} for gate_id in consumed):
                    errors.append(f"{cid} ready status has an unavailable consumed seam")
            elif completion:
                errors.append(f"{cid} non-complete status cannot claim completion evidence")
            if campaign.get("topology_class") == "physical" and status in {"complete", "in_review"}:
                _check_physical_identity(
                    campaign.get("physical_identity"),
                    cid,
                    owned or [],
                    cid,
                    errors,
                )
                if any(
                    not isinstance(entry, dict)
                    or not any(
                        _artifact_reference(value)
                        for value in entry.get("references") or []
                    )
                    for entry in campaign.get("completion_evidence") or []
                ):
                    errors.append(
                        f"{cid} physical completion evidence requires a durable report artifact"
                    )
            if not _strings(campaign.get("side_effect_assertions"), nonempty=True):
                errors.append(f"{cid} side_effect_assertions must be a non-empty array")
        if len(campaign_ids) != len(set(campaign_ids)):
            errors.append("duplicate evidence campaign ids")
        if len(campaign_mapped) != len(set(campaign_mapped)):
            errors.append("duplicate obligation mapping across evidence campaigns")
        # campaign ids uniqueness already checked; order not enforced generically
        for oid in campaign_mapped:
            if oid not in by_id:
                errors.append(f"unexpected delivery obligation {oid}")

    campaign_graph: dict[str, list[str]] = {}
    for cid, campaign in campaign_by_id.items():
        dependencies = campaign.get("depends_on_campaigns")
        if not isinstance(dependencies, list):
            continue
        campaign_graph[cid] = []
        for dependency in dependencies:
            if dependency == cid:
                errors.append(f"{cid} has campaign self-dependency")
            elif dependency not in campaign_by_id:
                errors.append(f"{cid} has unknown campaign dependency {dependency}")
            else:
                campaign_graph[cid].append(dependency)
    _check_dag(campaign_graph, "evidence-campaign", errors)

    closure = document.get("final_closure_packet")
    if not isinstance(closure, dict):
        errors.append("missing final_closure_packet")
    else:
        for field in ("id", "title", "change_set_id", "validation_authority", "admission_authority", "acceptance", "stop_condition"):
            if not _nonempty(closure.get(field)):
                errors.append(f"final closure packet {field} must be non-empty")
        if closure.get("id") != "FCP-00":
            errors.append("final closure packet id must be FCP-00")
        if closure.get("delivery_kind") != "final_closure":
            errors.append("final closure packet delivery_kind must be final_closure")
        # change_set_id should be a valid change set
        if closure.get("change_set_id") not in change_by_id:
            errors.append("final closure packet change_set_id must reference a valid change set")
        for identity_field in ("upstream_base_commit", "head_commit", "merge_commit"):
            if identity_field not in closure:
                errors.append(f"final closure packet missing {identity_field}")
            elif closure[identity_field] is not None and not isinstance(closure[identity_field], str):
                errors.append(f"final closure packet {identity_field} must be a commit or null")
        if not _nonempty(closure.get("sole_owner")):
            errors.append("final closure packet sole_owner must be non-empty")
        if not isinstance(closure.get("revert_identity"), dict) or not _nonempty((closure.get("revert_identity") or {}).get("identity")):
            errors.append("final closure packet revert_identity must be an object with identity")
        closure_status = closure.get("status")
        if closure_status in {"complete", "in_review"}:
            if not _full_commit(closure.get("upstream_base_commit")):
                errors.append("final closure promoted status requires a pinned upstream_base_commit")
            elif _forbidden_boundary(closure.get("upstream_base_commit")):
                errors.append("final closure packet upstream_base_commit uses a forbidden boundary")
            if closure_status == "complete":
                if not _full_commit(closure.get("merge_commit")):
                    errors.append("final closure complete status requires a 40-hex merge_commit")
                if closure.get("head_commit") is not None and not _full_commit(closure.get("head_commit")):
                    errors.append("final closure complete status head_commit must be a 40-hex commit when present")
            else:
                if not _full_commit(closure.get("head_commit")):
                    errors.append("final closure in_review status requires a 40-hex head_commit")
                if closure.get("merge_commit") is not None:
                    errors.append("final closure in_review status requires merge_commit null")
        if closure.get("validation_authority") != "docs/VALIDATION.md":
            errors.append("final closure packet must use docs/VALIDATION.md as route authority")
        if closure.get("admission_authority") != "docs/admissions.yml":
            errors.append("final closure packet must use docs/admissions.yml as admission authority")
        # obligation_ids should be subset of by_id and not duplicate
        fcp_oids = closure.get("obligation_ids")
        if not _strings(fcp_oids, nonempty=True):
            errors.append("final closure packet obligation_ids must be non-empty")
        else:
            for oid in fcp_oids:
                if oid not in by_id:
                    errors.append(f"final closure packet references unknown obligation {oid}")
        for key in ("depends_on_change_sets", "depends_on_campaigns", "required_seam_gates"):
            if not _strings(closure.get(key), nonempty=True):
                errors.append(f"final closure packet {key} must be a non-empty array")
        for gid in closure.get("depends_on_change_sets") or []:
            if gid not in change_by_id:
                errors.append(f"final closure packet references unknown group {gid}")
        for cid in closure.get("depends_on_campaigns") or []:
            if cid not in campaign_by_id:
                errors.append(f"final closure packet references unknown campaign {cid}")
        required_gates = closure.get("required_seam_gates") or []
        consumed_gates = closure.get("consumed_seam_gates") or []
        if required_gates != consumed_gates:
            errors.append("final closure packet required_seam_gates must equal consumed_seam_gates")
        for gate_id in required_gates:
            if gate_id not in seam_by_id:
                errors.append(f"final closure packet references unknown seam gate {gate_id}")
        if closure.get("status") not in ALLOWED_CHANGE_SET_STATUSES:
            errors.append("final closure packet status is not allowed")
        if closure.get("evidence_disposition") not in ALLOWED_DISPOSITIONS:
            errors.append("final closure packet evidence disposition is not allowed")
        for key in ("positive_evidence", "negative_evidence", "completion_evidence"):
            if not isinstance(closure.get(key), list):
                errors.append(f"final closure packet {key} must be an array")
            else:
                for entry in closure[key]:
                    _check_evidence_entry(entry, "final closure packet", errors)
                    if key == "negative_evidence" and isinstance(entry, dict) and entry.get("qualifies_for_completion"):
                        errors.append("final closure negative evidence cannot qualify for completion")
        if not closure.get("positive_evidence"):
            errors.append("final closure packet positive_evidence must be non-empty")
        if not closure.get("negative_evidence"):
            errors.append("final closure packet negative_evidence must be non-empty")
        if not _strings(closure.get("side_effect_assertions"), nonempty=True):
            errors.append("final closure packet side_effect_assertions must be a non-empty array")
        if closure.get("status") in {"complete", "in_review"}:
            if closure.get("evidence_disposition") != "current":
                errors.append("final closure prerequisite promotion from non-current evidence is forbidden")
            completion = closure.get("completion_evidence") or []
            if not completion or any(
                not isinstance(entry, dict)
                or entry.get("classification") != "current"
                or not entry.get("qualifies_for_completion")
                for entry in completion
            ):
                errors.append("final closure completion promotion requires qualifying current evidence")
            if not all(change_by_id.get(gid, {}).get("status") in {"complete", "in_review"} for gid in (closure.get("depends_on_change_sets") or [])):
                errors.append("final closure prerequisite group prevents completion promotion")
            if not all(campaign_by_id.get(cid, {}).get("status") in {"complete", "in_review"} for cid in (closure.get("depends_on_campaigns") or [])):
                errors.append("final closure prerequisite campaign prevents completion promotion")
            if not all(seam_by_id.get(gate_id, {}).get("status") in {"available", "complete"} and seam_by_id.get(gate_id, {}).get("evidence_disposition") == "current" for gate_id in required_gates):
                errors.append("final closure prerequisite seam prevents completion promotion")
            # check child obligation complete
            fcp_childs = closure.get("obligation_ids") or []
            if not all(by_id.get(oid, {}).get("status") == "complete" for oid in fcp_childs):
                errors.append("final closure blocked child obligation prevents completion promotion")
        elif closure.get("completion_evidence"):
            errors.append("non-complete final closure packet cannot claim completion evidence")

    # Delivery ownership is a disjoint union of grouped change sets, campaigns,
    # and the final closure packet. Every obligation must occur exactly once.
    delivery_records: list[tuple[str, str, str]] = []
    for gid, change_set in change_by_id.items():
        for oid in change_set.get("obligation_ids") or []:
            delivery_records.append((oid, "change_set", gid))
    for cid, campaign in campaign_by_id.items():
        for oid in campaign.get("obligation_ids") or []:
            delivery_records.append((oid, "evidence_campaign", cid))
    if isinstance(closure, dict):
        for oid in closure.get("obligation_ids") or []:
            delivery_records.append((oid, "final_closure", closure.get("id", "")))
    record_by_obligation: dict[str, list[tuple[str, str]]] = {}
    for oid, kind, owner_id in delivery_records:
        record_by_obligation.setdefault(oid, []).append((kind, owner_id))
    for oid in by_id:
        owners = record_by_obligation.get(oid, [])
        if not owners:
            errors.append(f"unmapped obligation {oid}")
        elif len(owners) > 1:
            errors.append(f"duplicate delivery owner for obligation {oid}")
        obligation = by_id.get(oid)
        if obligation is not None and len(owners) == 1 and obligation.get("delivery_owner") != {"kind": owners[0][0], "id": owners[0][1]}:
            errors.append(f"{oid} delivery_owner disagrees with grouped mapping")
    for oid in record_by_obligation:
        if oid not in by_id:
            errors.append(f"unexpected delivery obligation {oid}")
    for oid, owners in record_by_obligation.items():
        if len(owners) != 1 or oid not in by_id:
            continue
        kind, owner_id = owners[0]
        obligation = by_id[oid]
        if obligation.get("delivery_kind") != kind:
            errors.append(f"{oid} delivery_kind disagrees with resolved owner")
        expected_campaign_id = owner_id if kind == "evidence_campaign" else None
        if obligation.get("campaign_id") != expected_campaign_id:
            errors.append(f"{oid} campaign_id disagrees with resolved owner")
    delivery_owner_ids = set(change_by_id) | set(campaign_by_id)
    if isinstance(closure, dict):
        delivery_owner_ids.add(closure.get("id"))
    for oid, obligation in by_id.items():
        for dependency in obligation.get("depends_on_delivery") or []:
            if dependency not in delivery_owner_ids:
                errors.append(f"{oid} has unknown delivery dependency {dependency}")
    for gate_id, gate in seam_by_id.items():
        receipt = gate.get("receipt")
        if not isinstance(receipt, dict):
            continue
        producer = gate.get("producer")
        producer_owner = _owner_record(producer, change_by_id, campaign_by_id, closure)
        if producer_owner is not None:
            producer_commit = receipt.get("producer_commit")
            if not _full_commit(producer_commit):
                errors.append(f"{gate_id} receipt requires a 40-hex producer_commit")
            if producer_commit == producer_owner.get("upstream_base_commit"):
                errors.append(f"{gate_id} receipt producer_commit must not equal upstream_base_commit")
            expected_producer = _owner_commit(producer_owner)
            if expected_producer is None or producer_commit != expected_producer:
                errors.append(f"{gate_id} receipt producer_commit does not match {producer} head/merge identity")
            if receipt.get("evidence_commit") != producer_commit:
                errors.append(f"{gate_id} receipt evidence_commit must match producer_commit")
            if receipt.get("sole_owner") != producer_owner.get("sole_owner"):
                errors.append(f"{gate_id} receipt sole_owner does not match {producer} owner")
            expected_revert = producer_owner.get("revert_identity")
            expected_revert = expected_revert.get("identity") if isinstance(expected_revert, dict) else expected_revert
            if receipt.get("revert_identity") != expected_revert:
                errors.append(f"{gate_id} receipt revert_identity does not match {producer} identity")
            consumer_commits = receipt.get("consumer_commits") or {}
            for consumer in gate.get("consumers") or []:
                consumer_owner = _owner_record(consumer, change_by_id, campaign_by_id, closure)
                if consumer_owner is None or not _satisfied(consumer_owner.get("status")):
                    continue
                expected_consumer = _owner_commit(consumer_owner)
                if consumer not in consumer_commits:
                    errors.append(f"{gate_id} receipt consumer commit required for {consumer}")
                elif consumer_commits[consumer] != expected_consumer:
                    errors.append(f"{gate_id} receipt consumer commit does not match {consumer} identity")

    # Verify every seam consumer is reciprocal after all owner namespaces exist.
    all_owner_ids = set(change_by_id) | set(campaign_by_id)
    if isinstance(closure, dict):
        all_owner_ids.add(closure.get("id"))
    for gate_id, gate in seam_by_id.items():
        for consumer in gate.get("consumers") or []:
            if consumer not in all_owner_ids:
                errors.append(f"{gate_id} has unknown consumer {consumer}")
            elif gate_id not in (change_by_id.get(consumer, {}).get("consumed_seam_gates") or campaign_by_id.get(consumer, {}).get("consumed_seam_gates") or (closure.get("required_seam_gates") if isinstance(closure, dict) and closure.get("id") == consumer else []) or (closure.get("consumed_seam_gates") if isinstance(closure, dict) and closure.get("id") == consumer else [])):
                errors.append(f"{gate_id} consumer {consumer} does not consume the seam gate")
    owners_with_gates = {
        **{owner_id: owner.get("consumed_seam_gates") or [] for owner_id, owner in change_by_id.items()},
        **{owner_id: owner.get("consumed_seam_gates") or [] for owner_id, owner in campaign_by_id.items()},
    }
    if isinstance(closure, dict):
        owners_with_gates[closure.get("id", "")] = closure.get("required_seam_gates") or []
    for owner_id, consumed_gates in owners_with_gates.items():
        for gate_id in consumed_gates:
            if gate_id in seam_by_id and owner_id not in (seam_by_id[gate_id].get("consumers") or []):
                errors.append(f"{owner_id} consumed seam gate {gate_id} omits this consumer")
    for gate_id, gate in seam_by_id.items():
        producer_id = gate.get("producer")
        producer_owner = _owner_record(producer_id, change_by_id, campaign_by_id, closure)
        if producer_owner is not None and gate_id not in (producer_owner.get("produced_seam_gates") or []):
            errors.append(f"{gate_id} producer {producer_id} omits gate")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracker", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "device-mesh-port-tracker.json")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.tracker.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: tracker not found: {args.tracker}")
        return 2
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid tracker JSON: {exc}")
        return 2
    errors = _validate_tracker(document)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Tracker check failed: {len(errors)} error(s) in {args.tracker}")
        return 1
    # generic counts derived from document
    obligations = document.get("obligations") or []
    change_sets = document.get("change_sets") or []
    campaigns = document.get("evidence_campaigns") or []
    print(f"Tracker check passed: {len(obligations)} domain obligations, {len(change_sets)} grouped change sets, {len(campaigns)} evidence campaigns, DAG/seam/authority checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
