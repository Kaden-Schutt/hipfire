#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Focused contract tests for the upstream device-mesh tracker authority."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "check-device-mesh-port-tracker.py"
TRACKER = REPO / "docs" / "device-mesh-port-tracker.json"
INVALID_FIXTURE = REPO / "tests" / "fixtures" / "device-mesh-port-tracker.invalid.json"
INDEX = REPO / "docs" / "INDEX.md"


def _run_checker(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_canonical_tracker_satisfies_schema_and_dag():
    result = _run_checker(TRACKER)
    assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_fixture_covers_every_tracker_contract():
    result = _run_checker(INVALID_FIXTURE)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    for marker in (
        "local-only",
        "delivery_kind",
        "unknown",
        "cycle",
        "status",
        "implementation_class",
        "evidence disposition",
        "advancement",
        "completion promotion",
        "authority",
        "missing",
    ):
        assert marker in output, f"missing diagnostic marker {marker!r}:\n{output}"


def _write_document(document: dict, path: Path) -> Path:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _materialize_authority_evidence(document: dict) -> None:
    group = next(item for item in document["change_sets"] if item["id"] == "G0")
    group["status"] = "in_review"
    group["evidence_disposition"] = "current"
    group["upstream_base_commit"] = document["upstream"]["series_origin_ref"]
    group["head_commit"] = "a" * 40
    group["merge_commit"] = None
    group["completion_evidence"] = [
        {
            "classification": "current",
            "assertion": "Current G0 authority evidence.",
            "references": ["docs/device-mesh-port-tracker.json"],
            "qualifies_for_completion": True,
        }
    ]
    gate = next(item for item in document["seam_gates"] if item["id"] == "S-AUTHORITY")
    gate["status"] = "available"
    gate["evidence_disposition"] = "current"
    gate["receipt"] = {
        "status": "complete",
        "producer_commit": group["head_commit"],
        "evidence_commit": group["head_commit"],
        "consumer_commits": {},
        "route": "python3 scripts/check-device-mesh-port-tracker.py",
        "evidence_class": "current",
        "fixture_references": [
            "docs/device-mesh-port-tracker.json",
            "tests/fixtures/device-mesh-port-tracker.invalid.json",
        ],
        "positive_probe": "pytest -q tests/test_device_mesh_port_tracker.py",
        "negative_probe": "python3 scripts/check-device-mesh-port-tracker.py tests/fixtures/device-mesh-port-tracker.invalid.json (expected non-zero)",
        "side_effect_assertions": ["No runtime or authority side effect is permitted."],
        "sole_owner": group["sole_owner"],
        "revert_identity": group["revert_identity"]["identity"],
        "durable_references": [
            "https://github.com/warpfront/hipfire/issues/666",
            "git:" + group["head_commit"],
        ],
    }


def test_domain_obligations_and_campaigns_are_canonical():
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    obligations = {row["id"] for row in document["obligations"]}
    assert len(obligations) > 49
    assert not any(identifier.startswith("PR-") for identifier in obligations)
    assert [campaign["id"] for campaign in document["evidence_campaigns"]] == [
        "EC-EP",
        "EC-PP",
        "EC-TP",
        "EC-VISION",
        "EC-CLOSE",
    ]
    available = [gate["id"] for gate in document["seam_gates"] if gate["status"] in {"available", "complete"}]
    if document["change_sets"][0]["status"] == "implemented":
        assert available == []
    else:
        assert available == ["S-AUTHORITY"]
    assert "after authority" in document["policy"]["parallel_lane_rule"]


def test_group_completion_rejects_one_blocked_child(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    document["change_sets"][5]["obligation_ids"] = ["STEP-MOE-SUBSTRATE", "STEP-002", "STEP-002R"]
    child = next(row for row in document["obligations"] if row["id"] == "STEP-002")
    child["status"] = "blocked"
    child["evidence"]["disposition"] = "rerun_required"
    child["advancement"]["completion_rows"] = []
    result = _run_checker(_write_document(document, tmp_path / "blocked-group.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "G5 blocked child obligation prevents completion promotion" in output


def test_campaign_completion_rejects_one_blocked_child(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    child = next(row for row in document["obligations"] if row["id"] == "HW-001")
    child["status"] = "blocked"
    child["evidence"]["disposition"] = "hardware_blocked"
    child["advancement"]["completion_rows"] = []
    result = _run_checker(_write_document(document, tmp_path / "blocked-campaign.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "EC-EP blocked child obligation prevents completion promotion" in output


def test_final_closure_completion_rejects_one_blocked_child(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    child = next(row for row in document["obligations"] if row["id"] == "DOC-002")
    child["status"] = "blocked"
    child["evidence"]["disposition"] = "rerun_required"
    child["advancement"]["completion_rows"] = []
    result = _run_checker(_write_document(document, tmp_path / "blocked-closure.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "final closure blocked child obligation prevents completion promotion" in output


def test_campaign_dependency_namespaces_and_cycles_are_checked(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        ("unknown", {"EC-PP": ["EC-UNKNOWN"]}, "unknown campaign dependency"),
        ("self", {"EC-CLOSE": ["EC-CLOSE"]}, "campaign self-dependency"),
        (
            "cycle",
            {"EC-EP": ["EC-PP"], "EC-PP": ["EC-EP"]},
            "dependency cycle",
        ),
    )
    for name, updates, marker in cases:
        document = json.loads(json.dumps(base))
        for campaign_id, dependencies in updates.items():
            next(c for c in document["evidence_campaigns"] if c["id"] == campaign_id)["depends_on_campaigns"] = dependencies
        result = _run_checker(_write_document(document, tmp_path / f"{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing campaign diagnostic {marker!r}:\n{output}"


def test_qualifying_evidence_requires_durable_reference(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _materialize_authority_evidence(document)
    document["change_sets"][0]["completion_evidence"][0]["references"] = []
    result = _run_checker(_write_document(document, tmp_path / "no-evidence-ref.json"))
    output = result.stdout + result.stderr
    assert "durable reference" in output


def test_bare_references_cannot_promote_group_or_receipt(tmp_path: Path):
    cases = (
        (
            "group",
            lambda document: document["change_sets"][0]["completion_evidence"][0].update(
                references=["x"]
            ),
            "qualifying evidence requires a durable reference",
        ),
        (
            "receipt",
            lambda document: next(
                gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY"
            )["receipt"].update(durable_references=["x"]),
            "receipt durable_references require a durable commit or repository artifact",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(TRACKER.read_text(encoding="utf-8"))
        _materialize_authority_evidence(document)
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"bare-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing durable-reference diagnostic {marker!r}:\n{output}"


def test_series_origin_and_future_base_identity_rules(tmp_path: Path):
    cases = (
        (
            "origin",
            lambda document: document["upstream"].update(series_origin_ref="b" * 40),
            "upstream.series_origin_ref must equal the approved series origin",
        ),
        (
            "g0",
            lambda document: next(
                group for group in document["change_sets"] if group["id"] == "G0"
            ).update(upstream_base_commit="b" * 40),
            "G0 upstream_base_commit must equal series_origin_ref",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(TRACKER.read_text(encoding="utf-8"))
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"base-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing base diagnostic {marker!r}:\n{output}"

    future = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(future)
    next(group for group in future["change_sets"] if group["id"] == "G1")[
        "upstream_base_commit"
    ] = "c" * 40
    result = _run_checker(_write_document(future, tmp_path / "base-future.json"))
    assert result.returncode == 0, result.stdout + result.stderr



def test_exact_seam_maps_reject_bilateral_omission(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "fcp-close",
            lambda document: next(
                gate for gate in document["seam_gates"] if gate["id"] == "S-CLOSE"
            )["consumers"].remove("FCP-00"),
            "omits this consumer",
        ),
        (
            "campaign-hardware",
            lambda document: next(
                campaign
                for campaign in document["evidence_campaigns"]
                if campaign["id"] == "EC-CLOSE"
            )["consumed_seam_gates"].remove("S-HARDWARE-EP"),
            "does not consume the seam gate",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"seam-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing seam-map diagnostic {marker!r}:\n{output}"


def test_ready_or_available_seam_requires_current_disposition(tmp_path: Path):
    for disposition in ("historical", "hardware_blocked"):
        document = json.loads(TRACKER.read_text(encoding="utf-8"))
        group = next(item for item in document["change_sets"] if item["id"] == "G1")
        group["status"] = "ready"
        gate = next(item for item in document["seam_gates"] if item["id"] == "S-AUTHORITY")
        gate["evidence_disposition"] = disposition
        result = _run_checker(_write_document(document, tmp_path / f"ready-{disposition}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "available/complete seam requires current evidence disposition" in output

def _fake_physical_identity(campaign_id: str, obligation_ids: list[str]) -> dict:
    artifact = "tests/fixtures/device-mesh-port-tracker.invalid.json"
    identity = {
        "model_sha256": "1" * 64,
        "prompt_md5": "2" * 32,
        "binary_sha256": "3" * 64,
        "campaign_id": campaign_id,
        "gpu_ids": ["fake-gpu-0", "fake-gpu-1"],
        "topology": "fake-dual-gpu",
        "rocm_version": "fake-rocm",
        "rccl_version": "fake-rccl",
        "report_refs": [artifact],
        "result_map": {
            obligation_id: {"status": "pass", "report_refs": [artifact]}
            for obligation_id in obligation_ids
        },
    }
    if campaign_id == "EC-VISION":
        identity["image_sha256"] = "4" * 64
    return identity


def _satisfy_all_prerequisites(document: dict) -> None:
    base_commit = document["upstream"]["series_origin_ref"]
    artifact = "tests/fixtures/device-mesh-port-tracker.invalid.json"
    for obligation in document["obligations"]:
        obligation["status"] = "complete"
        obligation["evidence"]["disposition"] = "current"
        obligation["evidence"]["classification"] = "current"
        obligation["evidence"]["branch_record"] = "historical"
        obligation["evidence"]["report_refs"] = [artifact]
        obligation["advancement"]["completion_rows"] = [obligation["id"]]
        if obligation["id"].startswith("HW-"):
            obligation["physical_identity"] = _fake_physical_identity(
                obligation["campaign_id"], [obligation["id"]]
            )
    for index, change_set in enumerate(document["change_sets"], start=1):
        change_set["status"] = "in_review" if change_set["id"] == "G0" else "complete"
        change_set["evidence_disposition"] = "current"
        change_set["upstream_base_commit"] = base_commit
        change_set["head_commit"] = "a" * 40 if change_set["id"] == "G0" else f"{index:040x}"
        change_set["merge_commit"] = None if change_set["id"] == "G0" else f"{index + 100:040x}"
        change_set["completion_evidence"] = [
            {
                "classification": "current",
                "assertion": "Current grouped evidence packet.",
                "references": ["docs/device-mesh-port-tracker.json"],
                "qualifies_for_completion": True,
            }
        ]
    for index, campaign in enumerate(document["evidence_campaigns"], start=201):
        campaign["status"] = "complete"
        campaign["evidence_disposition"] = "current"
        campaign["upstream_base_commit"] = base_commit
        campaign["head_commit"] = f"{index:040x}"
        campaign["merge_commit"] = f"{index + 100:040x}"
        campaign["sole_owner"] = f"{campaign['id']} evidence owner"
        campaign["revert_identity"] = {
            "identity": f"{campaign['id']}:campaign-revert",
            "strategy": "revert-entire-evidence-campaign",
            "scope": "Revert this evidence campaign as one unit.",
        }
        campaign["completion_evidence"] = [
            {
                "classification": "current",
                "assertion": "Current campaign evidence packet.",
                "references": [artifact],
                "qualifies_for_completion": True,
            }
        ]
        if campaign["topology_class"] == "physical":
            campaign["physical_identity"] = _fake_physical_identity(
                campaign["id"], campaign["obligation_ids"]
            )
    closure = document["final_closure_packet"]
    closure["status"] = "complete"
    closure["evidence_disposition"] = "current"
    closure["upstream_base_commit"] = base_commit
    closure["head_commit"] = f"{len(document['evidence_campaigns']) + 300:040x}"
    closure["merge_commit"] = f"{len(document['evidence_campaigns']) + 400:040x}"
    closure["sole_owner"] = "FCP-00 final closure owner"
    closure["revert_identity"] = {
        "identity": "FCP-00:single-final-closure-revert",
        "strategy": "revert-entire-final-closure",
        "scope": "Revert FCP-00 as one unit.",
    }
    closure["completion_evidence"] = [
        {
            "classification": "current",
            "assertion": "Current final closure packet.",
            "references": ["docs/device-mesh-port-tracker.json"],
            "qualifies_for_completion": True,
        }
    ]
    groups = {group["id"]: group for group in document["change_sets"]}
    campaigns = {campaign["id"]: campaign for campaign in document["evidence_campaigns"]}
    owners = {**groups, **campaigns, closure["id"]: closure}

    def owner_commit(owner: dict) -> str:
        return owner["merge_commit"] or owner["head_commit"]

    for gate in document["seam_gates"]:
        gate["status"] = "available"
        gate["evidence_disposition"] = "current"
        producer = owners[gate["producer"]]
        consumer_commits = {
            consumer: owner_commit(owners[consumer])
            for consumer in gate["consumers"]
            if owners[consumer]["status"] in {"complete", "in_review"}
        }
        gate["receipt"] = {
            "status": "complete",
            "producer_commit": owner_commit(producer),
            "evidence_commit": owner_commit(producer),
            "consumer_commits": consumer_commits,
            "route": "Current executable seam route with pinned fixture.",
            "evidence_class": "current",
            "fixture_references": ["docs/device-mesh-port-tracker.json", artifact],
            "positive_probe": "Current positive seam probe command.",
            "negative_probe": "Current fail-closed seam probe command.",
            "side_effect_assertions": ["No duplicate owner or hidden side effect is permitted."],
            "sole_owner": producer["sole_owner"],
            "revert_identity": producer["revert_identity"]["identity"],
            "durable_references": [artifact],
        }
        if gate["id"].startswith("S-HARDWARE-"):
            gate["receipt"]["physical_identity"] = _fake_physical_identity(
                producer["id"], producer["obligation_ids"]
            )


def test_domain_row_contracts_are_not_copied_placeholders():
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in document["obligations"]}
    assert "mtp_k" in rows["COR-001"]["scope"]
    assert "ModelMeta" in rows["COR-001"]["acceptance"]
    assert "LoadedModel" in rows["COR-004"]["acceptance"]
    assert "cross-request" in rows["COR-004"]["acceptance"]
    assert "transactional" in rows["COR-005"]["scope"]
    assert "fault injection" in rows["COR-005"]["acceptance"]
    assert "Qwen35" in rows["GEN-001"]["scope"]
    assert "DeltaNet" in rows["GEN-001"]["acceptance"]
    assert "standard-attention" not in rows["GEN-001"]["scope"]
    assert "standard-attention" in rows["AXIS-001"]["scope"]
    assert "on-disk" in rows["SPEC-003"]["acceptance"]
    assert "rollback" in rows["SPEC-003"]["acceptance"]
    assert "PP+MTP" in rows["SPEC-004"]["scope"]
    assert "compressed .mtp" in rows["SPEC-004"]["scope"]
    assert "64 MiB" in rows["SPEC-004"]["acceptance"]
    assert rows["COR-001"]["legacy_status"] == "complete"
    assert rows["SPEC-003"]["legacy_status"] == "deferred"
    assert rows["COR-002"]["depends_on"] == ["COR-004"]
    assert rows["GEN-001"]["depends_on"] == [
        "COR-002",
        "STEP-001",
        "STEP-002",
        "STEP-003",
        "STEP-005-QWEN35",
        "SPEC-001",
    ]
    assert rows["SPEC-003"]["depends_on"] == ["COR-001"]
    assert rows["SPEC-004"]["depends_on"] == ["GEN-001", "SPEC-002", "SPEC-003"]


def test_group_merge_wait_declaration_is_exact(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        ("missing", lambda group: group.update(merge_waits_on=[]), "parallel_lane.merge_waits_on must match top-level merge_waits_on"),
        (
            "disagree",
            lambda group: group["parallel_lane"].update(merge_waits_on=[]),
            "G5 parallel_lane.merge_waits_on must match top-level merge_waits_on",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        group = next(item for item in document["change_sets"] if item["id"] == "G5")
        mutation(group)
        result = _run_checker(_write_document(document, tmp_path / f"merge-wait-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing merge-wait diagnostic {marker!r}:\n{output}"


def test_development_gate_map_is_exact(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    expected = {
        "G0": [],
        "G1": ["G0"],
        "G2": ["G1"],
        "G3": ["G0"],
        "G4": ["G0"],
        "G5": ["G0"],
        "G6": ["G5"],
        "G7": ["G5"],
        "G8": ["G5"],
        "G9": ["G5"],
        "G10": ["G5"],
        "G11": ["G5"],
        "G12": ["G4"],
        "G13": ["G12"],
        "G14": ["G12"],
        "G15": ["G12"],
    }
    groups = {group["id"]: group for group in base["change_sets"]}
    for group_id, can_develop_after in expected.items():
        assert groups[group_id]["parallel_lane"]["can_develop_after"] == can_develop_after
        assert groups[group_id]["can_develop_after"] == can_develop_after
    cases = (
        (
            "top-level-drift",
            lambda group: group.update(can_develop_after=["G2"]),
            "parallel_lane.can_develop_after must match top-level can_develop_after",
        ),
        (
            "parallel-drift",
            lambda group: group["parallel_lane"].update(can_develop_after=["G2"]),
            "G3 parallel_lane.can_develop_after must match top-level can_develop_after",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        group = next(item for item in document["change_sets"] if item["id"] == "G3")
        mutation(group)
        result = _run_checker(_write_document(document, tmp_path / f"development-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing development-gate diagnostic {marker!r}:\n{output}"


def test_available_current_seam_requires_current_receipt_class(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    for evidence_class in ("failed", "historical", "emulated", "rerun_required"):
        document = json.loads(json.dumps(base))
        _materialize_authority_evidence(document)
        receipt = next(
            gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY"
        )["receipt"]
        receipt["evidence_class"] = evidence_class
        result = _run_checker(
            _write_document(document, tmp_path / f"receipt-{evidence_class}.json")
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "S-AUTHORITY receipt evidence_class must be current" in output


def test_change_set_identities_and_current_seam_receipts_are_required(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _materialize_authority_evidence(document)
    group = next(item for item in document["change_sets"] if item["id"] == "G0")
    assert "upstream_base_commit" in group
    assert "head_commit" in group
    assert "merge_commit" in group
    gate = next(item for item in document["seam_gates"] if item["id"] == "S-AUTHORITY")
    gate["receipt"] = None
    result = _run_checker(_write_document(document, tmp_path / "missing-receipt.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "S-AUTHORITY current/available seam requires a complete receipt" in output


def test_current_seam_receipt_fields_and_group_identity_fail_closed(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "missing-group-base",
            lambda document: next(group for group in document["change_sets"] if group["id"] == "G0").pop("upstream_base_commit"),
            "G0 missing upstream_base_commit",
        ),
        (
            "missing-receipt-field",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].pop("durable_references"),
            "S-AUTHORITY receipt durable_references must be non-empty",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        _materialize_authority_evidence(document)
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing identity/receipt diagnostic {marker!r}:\n{output}"


def test_only_authority_seam_is_available_before_port_work():
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    available = [gate["id"] for gate in document["seam_gates"] if gate["status"] in {"available", "complete"}]
    if document["change_sets"][0]["status"] == "implemented":
        assert available == []
    else:
        assert available == ["S-AUTHORITY"]


def test_commit_identity_and_receipt_mutations_fail_closed(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "issue-head",
            lambda document: next(group for group in document["change_sets"] if group["id"] == "G0").update(
                head_commit="https://github.com/warpfront/hipfire/issues/666#g0"
            ),
            "G0 in_review status requires a 40-hex head_commit",
        ),
        (
            "null-producer",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                producer_commit=None
            ),
            "S-AUTHORITY receipt requires a 40-hex producer_commit",
        ),
        (
            "base-producer",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                producer_commit=document["upstream"]["series_origin_ref"]
            ),
            "S-AUTHORITY receipt producer_commit must not equal upstream_base_commit",
        ),
        (
            "placeholder-route",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                route="recorded by the owning change set"
            ),
            "S-AUTHORITY receipt route must be concrete",
        ),
        (
            "owner-mismatch",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                sole_owner="wrong owner"
            ),
            "S-AUTHORITY receipt sole_owner does not match G0 owner",
        ),
        (
            "revert-mismatch",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                revert_identity="wrong revert"
            ),
            "S-AUTHORITY receipt revert_identity does not match G0 identity",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        _materialize_authority_evidence(document)
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"identity-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing identity/receipt diagnostic {marker!r}:\n{output}"


def test_promoted_consumer_receipt_requires_matching_commit(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "missing",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                consumer_commits={}
            ),
            "S-AUTHORITY receipt consumer commit required for G1",
        ),
        (
            "mismatch",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                consumer_commits={"G1": "a" * 40, "G4": "b" * 40}
            ),
            "S-AUTHORITY receipt consumer commit does not match G1 identity",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        _satisfy_all_prerequisites(document)
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"consumer-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing consumer diagnostic {marker!r}:\n{output}"


def test_campaign_and_fcp_identity_schema_and_owner_receipts(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    for owner in [*document["evidence_campaigns"], document["final_closure_packet"]]:
        assert all(field in owner for field in ("upstream_base_commit", "head_commit", "merge_commit"))
        assert "sole_owner" in owner
        assert "revert_identity" in owner
    _satisfy_all_prerequisites(document)
    cases = (
        (
            "campaign-owner",
            "S-HARDWARE-EP",
            lambda receipt: receipt.update(sole_owner="wrong"),
            "S-HARDWARE-EP receipt sole_owner does not match EC-EP owner",
        ),
        (
            "close-revert",
            "S-CLOSE",
            lambda receipt: receipt.update(revert_identity="wrong"),
            "S-CLOSE receipt revert_identity does not match EC-CLOSE identity",
        ),
    )
    for name, gate_id, mutation, marker in cases:
        mutated = json.loads(json.dumps(document))
        gate = next(item for item in mutated["seam_gates"] if item["id"] == gate_id)
        mutation(gate["receipt"])
        result = _run_checker(_write_document(mutated, tmp_path / f"{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing owner receipt diagnostic {marker!r}:\n{output}"


def test_campaign_and_fcp_seam_producer_reciprocity(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "campaign",
            "EC-EP",
            "S-HARDWARE-EP",
            "campaign",
            "S-HARDWARE-EP producer EC-EP omits gate",
        ),
        (
            "closure",
            "EC-CLOSE",
            "S-CLOSE",
            "campaign",
            "S-CLOSE producer EC-CLOSE omits gate",
        ),
    )
    for name, owner_id, gate_id, owner_kind, marker in cases:
        document = json.loads(json.dumps(base))
        owner = next(
            item
            for item in document["evidence_campaigns"]
            if item["id"] == owner_id
        )
        owner["produced_seam_gates"].remove(gate_id)
        result = _run_checker(_write_document(document, tmp_path / f"producer-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing producer diagnostic {marker!r}:\n{output}"


def test_campaign_and_fcp_complete_heads_require_sha(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "campaign",
            lambda document: next(
                item for item in document["evidence_campaigns"] if item["id"] == "EC-EP"
            ).update(head_commit="https://github.com/warpfront/hipfire/issues/666"),
            "EC-EP complete status head_commit must be a 40-hex commit when present",
        ),
        (
            "fcp",
            lambda document: document["final_closure_packet"].update(
                head_commit="https://github.com/warpfront/hipfire/issues/666"
            ),
            "final closure complete status head_commit must be a 40-hex commit when present",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        _satisfy_all_prerequisites(document)
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"head-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing head diagnostic {marker!r}:\n{output}"


def test_ready_consumer_is_omitted_but_complete_consumer_is_keyed(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    groups = {group["id"]: group for group in document["change_sets"]}
    groups["G1"]["status"] = "ready"
    for group in document["change_sets"]:
        if group["id"] not in {"G0", "G1", "G4"}:
            group["status"] = "blocked"
            group["evidence_disposition"] = "rerun_required"
            group["completion_evidence"] = []
    for campaign in document["evidence_campaigns"]:
        campaign["status"] = "blocked"
        campaign["evidence_disposition"] = "hardware_blocked"
        campaign["completion_evidence"] = []
    closure = document["final_closure_packet"]
    closure["status"] = "blocked"
    closure["evidence_disposition"] = "rerun_required"
    closure["completion_evidence"] = []
    for gate in document["seam_gates"]:
        if gate["id"] == "S-AUTHORITY":
            gate["status"] = "available"
            gate["evidence_disposition"] = "current"
            gate["receipt"]["consumer_commits"] = {"G4": groups["G4"]["merge_commit"]}
        else:
            gate["status"] = "proposed"
            gate["evidence_disposition"] = "rerun_required"
            gate["receipt"] = None
    result = _run_checker(_write_document(document, tmp_path / "compact-consumers.json"))
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert document["seam_gates"][0]["receipt"]["consumer_commits"] == {"G4": groups["G4"]["merge_commit"]}


def test_final_closure_requires_current_qualifying_evidence(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        ("empty", lambda closure: closure.update(completion_evidence=[]), "final closure completion promotion requires qualifying current evidence"),
        (
            "nonqualifying",
            lambda closure: closure.update(
                completion_evidence=[
                    {
                        "classification": "current",
                        "assertion": "Not qualifying.",
                        "references": ["docs/device-mesh-port-tracker.json"],
                        "qualifies_for_completion": False,
                    }
                ]
            ),
            "final closure completion promotion requires qualifying current evidence",
        ),
        (
            "negative",
            lambda closure: closure["negative_evidence"][0].update(qualifies_for_completion=True),
            "final closure negative evidence cannot qualify for completion",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        _satisfy_all_prerequisites(document)
        mutation(document["final_closure_packet"])
        result = _run_checker(_write_document(document, tmp_path / f"closure-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing closure diagnostic {marker!r}:\n{output}"


def test_final_closure_requires_positive_and_negative_evidence(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    for field, marker in (
        ("positive_evidence", "final closure packet positive_evidence must be non-empty"),
        ("negative_evidence", "final closure packet negative_evidence must be non-empty"),
    ):
        document = json.loads(json.dumps(base))
        _satisfy_all_prerequisites(document)
        document["final_closure_packet"][field] = []
        result = _run_checker(_write_document(document, tmp_path / f"closure-{field}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing final-closure evidence diagnostic {marker!r}:\n{output}"


def test_final_closure_required_seams_are_reciprocal(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    next(g for g in document["seam_gates"] if g["id"] == "S-DENSE-AXIS")["consumers"].remove("FCP-00")
    result = _run_checker(_write_document(document, tmp_path / "fcp-seam.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "FCP-00 consumed seam gate S-DENSE-AXIS omits this consumer" in output


def test_legacy_pr_provenance_requires_set_coverage(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    for obligation in document["obligations"]:
        obligation["legacy_pr_ids"] = [
            identifier for identifier in obligation["legacy_pr_ids"] if identifier != "PR-34M"
        ]
    result = _run_checker(_write_document(document, tmp_path / "legacy-missing.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "missing legacy PR provenance coverage" in output


def test_obligation_delivery_namespace_matches_resolved_owner(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "domain",
            "DOC-001",
            lambda row: row.update(delivery_kind="evidence_campaign", campaign_id="EC-EP"),
            "DOC-001 delivery_kind disagrees with resolved owner",
        ),
        (
            "physical",
            "HW-001",
            lambda row: row.update(delivery_kind="evidence_campaign", campaign_id="EC-PP"),
            "HW-001 campaign_id disagrees with resolved owner",
        ),
    )
    for name, obligation_id, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        row = next(row for row in document["obligations"] if row["id"] == obligation_id)
        mutation(row)
        result = _run_checker(_write_document(document, tmp_path / f"delivery-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing delivery diagnostic {marker!r}:\n{output}"

def test_g5_merge_wait_blocks_promotion(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    next(group for group in document["change_sets"] if group["id"] == "G3")["status"] = "blocked"
    result = _run_checker(_write_document(document, tmp_path / "g5-merge-wait.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "G5 merge wait prevents completion promotion" in output


def test_grouped_tracker_maps_each_obligation_once():
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    obligations = {row["id"] for row in document["obligations"]}
    change_set_mapped = [
        obligation_id
        for change_set in document["change_sets"]
        for obligation_id in change_set["obligation_ids"]
    ]
    campaign_mapped = [
        obligation_id
        for campaign in document["evidence_campaigns"]
        for obligation_id in campaign["obligation_ids"]
    ]
    final_mapped = document["final_closure_packet"]["obligation_ids"]
    mapped = change_set_mapped + campaign_mapped + final_mapped
    assert len(document["change_sets"]) == 16
    assert len(document["evidence_campaigns"]) == 5
    assert len(mapped) == len(obligations) == 68
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == obligations


def test_docs_index_links_tracker_without_replacing_authorities():
    body = INDEX.read_text(encoding="utf-8")
    assert "[`docs/device-mesh-port-tracker.json`](device-mesh-port-tracker.json)" in body
    assert "[`docs/VALIDATION.md`](VALIDATION.md)" in body
    assert "[`docs/admissions.yml`](admissions.yml)" in body
def test_completed_obligations_require_current_durable_evidence(tmp_path: Path):
    cases = (
        (
            "not-applicable",
            lambda row: row["evidence"].update(
                disposition="not_applicable",
                classification="current",
                branch_record="none",
                report_refs=[],
            ),
            "HW-001 complete status requires current evidence",
        ),
        (
            "bare-report",
            lambda row: row["evidence"].update(
                disposition="current",
                classification="current",
                report_refs=["x"],
            ),
            "HW-001 complete current evidence requires a durable report reference",
        ),
        (
            "tracker-report",
            lambda row: row["evidence"].update(
                disposition="current",
                classification="current",
                report_refs=["docs/device-mesh-port-tracker.json"],
            ),
            "HW-001 complete physical evidence requires a durable report artifact",
        ),

        (
            "wrong-class",
            lambda row: row["evidence"].update(
                disposition="current",
                classification="historical",
                report_refs=["docs/VALIDATION.md"],
            ),
            "HW-001 complete status requires evidence classification current",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(TRACKER.read_text(encoding="utf-8"))
        _satisfy_all_prerequisites(document)
        row = next(row for row in document["obligations"] if row["id"] == "HW-001")
        row["status"] = "complete"
        row["advancement"]["completion_rows"] = ["HW-001"]
        mutation(row)
        result = _run_checker(
            _write_document(document, tmp_path / f"obligation-{name}.json")
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing obligation evidence diagnostic {marker!r}:\n{output}"


def test_physical_identity_contract_is_complete_and_fail_closed(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    for campaign in document["evidence_campaigns"]:
        if campaign["id"] in {"EC-EP", "EC-PP", "EC-TP", "EC-VISION"}:
            assert "physical_identity" in campaign
    cases = (
        (
            "digest",
            lambda identity: identity.update(model_sha256="x"),
            "EC-EP physical_identity.model_sha256 must be 64-hex",
        ),
        (
            "gpu-count",
            lambda identity: identity.update(gpu_ids=["gpu-a"]),
            "EC-EP physical_identity.gpu_ids must contain at least two distinct GPUs",
        ),
        (
            "result-map",
            lambda identity: identity.update(result_map={}),
            "EC-EP physical_identity.result_map must cover every mapped obligation",
        ),
        (
            "rccl",
            lambda identity: identity.update(rccl_version="not-used"),
            "EC-EP physical_identity.rccl_version cannot be not-used",
        ),
        (
            "tracker-only",
            lambda identity: identity.update(report_refs=["docs/device-mesh-port-tracker.json"]),
            "EC-EP physical_identity requires a durable report beyond tracker/issue references",
        ),
    )
    for name, mutation, marker in cases:
        mutated = json.loads(json.dumps(document))
        _satisfy_all_prerequisites(mutated)
        campaign = next(item for item in mutated["evidence_campaigns"] if item["id"] == "EC-EP")
        mutation(campaign["physical_identity"])
        result = _run_checker(_write_document(mutated, tmp_path / f"physical-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing physical identity diagnostic {marker!r}:\n{output}"
    mutated = json.loads(json.dumps(document))
    _satisfy_all_prerequisites(mutated)
    campaign = next(item for item in mutated["evidence_campaigns"] if item["id"] == "EC-EP")
    campaign["completion_evidence"][0]["references"] = [
        "docs/device-mesh-port-tracker.json"
    ]
    result = _run_checker(_write_document(mutated, tmp_path / "physical-campaign-report.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "EC-EP physical completion evidence requires a durable report artifact" in output

    mutated = json.loads(json.dumps(document))
    _satisfy_all_prerequisites(mutated)
    receipt = next(gate for gate in mutated["seam_gates"] if gate["id"] == "S-HARDWARE-EP")["receipt"]
    receipt["durable_references"] = ["docs/device-mesh-port-tracker.json"]
    result = _run_checker(_write_document(mutated, tmp_path / "physical-seam-report.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "S-HARDWARE-EP receipt physical evidence requires a durable report artifact" in output


def test_forbidden_boundary_bases_never_promote(tmp_path: Path):
    forbidden = "bac02a1a22a55922ea057e9a98f68cb3ab93ac02"
    cases = (
        (
            "group",
            lambda document: next(group for group in document["change_sets"] if group["id"] == "G1").update(upstream_base_commit=forbidden),
            "G1 upstream_base_commit uses a forbidden boundary",
        ),
        (
            "campaign",
            lambda document: next(campaign for campaign in document["evidence_campaigns"] if campaign["id"] == "EC-EP").update(upstream_base_commit=forbidden),
            "EC-EP upstream_base_commit uses a forbidden boundary",
        ),
        (
            "fcp",
            lambda document: document["final_closure_packet"].update(upstream_base_commit=forbidden),
            "final closure packet upstream_base_commit uses a forbidden boundary",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(TRACKER.read_text(encoding="utf-8"))
        _satisfy_all_prerequisites(document)
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"forbidden-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing forbidden-boundary diagnostic {marker!r}:\n{output}"

def test_nonexistent_path_cannot_promote(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _materialize_authority_evidence(document)
    # Use a nonexistent report path for completion promotion
    document["change_sets"][0]["completion_evidence"][0]["references"] = ["docs/nonexistent-report-9999.md"]
    result = _run_checker(_write_document(document, tmp_path / "nonexistent.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "durable reference" in output or "durable report" in output


def test_cargo_toml_as_report_cannot_promote(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _materialize_authority_evidence(document)
    document["change_sets"][0]["completion_evidence"][0]["references"] = ["Cargo.toml"]
    result = _run_checker(_write_document(document, tmp_path / "cargo.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    # Cargo.toml is not a report-like artifact, so completion should be rejected
    assert "durable reference" in output or "durable report artifact" in output or "requires a durable" in output


def test_malformed_hash_url_cannot_promote(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _materialize_authority_evidence(document)
    for bad_url in [
        "https://github.com/warpfront/hipfire/commit/abc123",
        "https://github.com/warpfront/hipfire/commit/zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
        "https://github.com/warpfront/hipfire/blob/abc123/docs/VALIDATION.md",
    ]:
        mutated = json.loads(json.dumps(document))
        mutated["change_sets"][0]["completion_evidence"][0]["references"] = [bad_url]
        result = _run_checker(_write_document(mutated, tmp_path / f"badhash-{bad_url[-6:]}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0, f"bad hash {bad_url!r} should be rejected"
        assert "malformed" in output.lower() or "durable reference" in output or "40-hex" in output


def test_stale_graph_rejected(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    document["purpose"] = "This tracker contains stale checkbox [x] claim"
    result = _run_checker(_write_document(document, tmp_path / "stale.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "stale checkbox" in output


def test_duplicated_source_of_truth_drift_rejected():
    # Checker must not duplicate plan maps; JSON is the sole source.
    checker_text = (REPO / "scripts" / "check-device-mesh-port-tracker.py").read_text(encoding="utf-8")
    forbidden_patterns = [
        "EXPECTED_GROUP_OBLIGATIONS",
        "EXPECTED_GROUP_DEPS",
        "EXPECTED_GROUP_CONSUMED",
        "EXPECTED_GROUP_PRODUCED",
        "EXPECTED_CAMPAIGN_OBLIGATIONS",
        "EXPECTED_SEAM_CONSUMERS",
        "EXPECTED_SEAM_PRODUCERS",
        "EXPECTED_CAN_DEVELOP_AFTER",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in checker_text, f"checker still contains duplicated source-of-truth {pattern!r}"
    # Also ensure tracker and checker agree on G1->G3->G5 chain without duplication
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    g1 = next(g for g in document["change_sets"] if g["id"] == "G1")
    g3 = next(g for g in document["change_sets"] if g["id"] == "G3")
    g5 = next(g for g in document["change_sets"] if g["id"] == "G5")
    assert g3["depends_on"] == ["G1"], "G3 must depend on G1 for landing chain"
    assert g5["depends_on"] == ["G3"], "G5 must depend on G3 for landing chain"
    assert "G3" in g5["merge_waits_on"], "G5 must merge-wait on G3"
    assert "S-TOPOLOGY" in g3["consumed_seam_gates"], "G3 must consume S-TOPOLOGY"
    assert "S-MANIFEST" in g5["consumed_seam_gates"], "G5 must consume S-MANIFEST"
