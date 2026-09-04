# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import importlib.util
import json
import pytest
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REVIEW_PATH = Path(__file__).parent.parent / "review.py"
spec = importlib.util.spec_from_file_location("review", REVIEW_PATH)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

FAKE_OMP = Path(__file__).parent / "fake_omp.py"
FAKE_GH = Path(__file__).parent / "fake_gh.py"

# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    assert review.extract_json('{"a":1}') == {"a": 1}

def test_extract_json_fenced():
    text = 'hello\n```json\n{"phase":"prelim","x":1}\n```\nworld'
    assert review.extract_json(text) == {"phase": "prelim", "x": 1}

def test_extract_json_prose_around():
    text = 'Sure, here is the object: {"phase":"verdict","decision":"block"} hope that helps'
    assert review.extract_json(text) == {"phase": "verdict", "decision": "block"}

def test_extract_json_last_wins():
    text = '{"first":1} some text {"second":2}'
    assert review.extract_json(text) == {"second": 2}

def test_extract_json_none():
    assert review.extract_json("no json here") is None

# ---------------------------------------------------------------------------
# hard_floor / soft_floor
# ---------------------------------------------------------------------------

def _base_select(buckets=None, policy_paths=None):
    return {"buckets": buckets or ["load"], "policy_paths": policy_paths or [], "surfaces": {}}

def _base_evidence(verdict="pass", with_attractor=False):
    ev = {"verdict": verdict, "fixtures": [], "kernel": None}
    if with_attractor:
        ev["fixtures"] = [{
            "tag": "qwen3.6:27b",
            "modes": {
                "battery": {
                    "rows": [{"attractor": True, "empty": False}]
                }
            }
        }]
    return ev

def test_hard_floor_hw_run():
    hard = review.hard_floor(_base_evidence(), _base_select(), "failure", [])
    assert any("hw_run" in r for r in hard)

def test_hard_floor_evidence_verdict():
    hard = review.hard_floor({"verdict": "fail"}, _base_select(), "success", [])
    assert any("evidence" in r for r in hard)

def test_hard_floor_attractor():
    hard = review.hard_floor(_base_evidence(with_attractor=True), _base_select(), "success", [])
    assert any("attractor" in r for r in hard)

def test_hard_floor_policy():
    hard = review.hard_floor(_base_evidence(), _base_select(policy_paths=["scripts/hw-gate/review.py"]), "success", [])
    assert any("policy" in r for r in hard)

def test_hard_floor_ratchet():
    hard = review.hard_floor(_base_evidence(), _base_select(), "success", ["RATCHET-RAISE: foo"])
    assert any("RATCHET" in r for r in hard)

def test_soft_floor_coverage():
    soft = review.soft_floor({"coverage": {"gaps": ["load"]}, "confidence": 0.9, "decision": "greenlight"}, "greenlight")
    assert any("coverage" in r for r in soft)

def test_soft_floor_confidence():
    soft = review.soft_floor({"coverage": {"gaps": []}, "confidence": 0.5}, "greenlight")
    assert any("confidence" in r for r in soft)

def test_soft_floor_needs_human():
    soft = review.soft_floor({"coverage": {"gaps": []}, "confidence": 0.9}, "needs-human")
    assert any("needs-human" in r for r in soft)

def test_floor_greenlight_only_when_clear():
    hard = review.hard_floor(_base_evidence(), _base_select(), "success", [])
    soft = review.soft_floor({"coverage": {"gaps": []}, "confidence": 0.9}, "greenlight")
    assert hard == []
    assert soft == []

# Table-driven hard vs soft final mapping via apply_floor
def test_floor_each_hard_blocks_or_holds():
    # evidence fail => block
    d, r = review.apply_floor("greenlight", {"verdict": "fail"}, _base_select(), "success", [], {"coverage": {"gaps": []}, "confidence": 0.9, "decision": "greenlight"})
    assert d == "block"
    # policy => needs-human (hold)
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(policy_paths=["x"]), "success", [], {"coverage": {"gaps": []}, "confidence": 0.9, "decision": "greenlight"})
    assert d == "needs-human"
    # ratchet => needs-human
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", ["RATCHET-RAISE: hi"], {"coverage": {"gaps": []}, "confidence": 0.9, "decision": "greenlight"})
    assert d == "needs-human"

def test_floor_each_soft_needs_human():
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", [], {"coverage": {"gaps": ["serve"]}, "confidence": 0.9, "decision": "greenlight"})
    assert d == "needs-human"
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", [], {"coverage": {"gaps": []}, "confidence": 0.5, "decision": "greenlight"})
    assert d == "needs-human"
    d, r = review.apply_floor("needs-human", _base_evidence(), _base_select(), "success", [], {"coverage": {"gaps": []}, "confidence": 0.9, "decision": "needs-human"})
    assert d == "needs-human"

def test_floor_greenlight_only_when_clear_apply():
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", [], {"coverage": {"gaps": []}, "confidence": 0.9, "decision": "greenlight"})
    assert d == "greenlight"

# ---------------------------------------------------------------------------
# helpers for e2e
# ---------------------------------------------------------------------------

def _make_repo(tmp: Path, files: dict | None = None):
    repo = tmp / "checkout"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "base.txt").write_text("base\n")
    if files:
        for name, content in files.items():
            p = repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "base.txt").write_text("head\n")
    (repo / "feature.txt").write_text("new\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "feature"], cwd=repo, check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    return repo, base, head

def _fixtures_content(tmp_models: Path | None = None):
    return {
        "schema": "hipfire.hw-gate.fixtures",
        "version": 2,
        "models_dir": str(tmp_models) if tmp_models else "~/.hipfire/models",
        "fixtures": [
            {"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": "abc", "size_bytes": 100, "arch_id": 5, "why": "test"},
            {"tag": "ornith-1.5:35b-a3b-mq4r", "file": "ornith.mq4r", "sha256": "def", "size_bytes": 100, "arch_id": 6, "why": "test"},
        ],
        "buckets": {
            "load": {"why": "load","modes": ["battery"]},
            "serve": {"why": "serve","modes": ["battery","chain"]},
            "kernel": {"why": "kernel","modes": ["battery"]},
        }
    }

# ---------------------------------------------------------------------------
# prelim e2e
# ---------------------------------------------------------------------------

def test_prelim_routes_bucket_union_and_sol():
    tmp = Path(tempfile.mkdtemp())
    models_dir = tmp / "models"
    models_dir.mkdir(parents=True)
    # create both fixture files present
    (models_dir / "qwen3.6-27b.mq4").write_text("dummy")
    (models_dir / "ornith.mq4r").write_text("dummy")
    checkout, base, head = _make_repo(tmp / "repo1")
    select = {
        "schema": "hipfire.hw-gate.select",
        "version": 1,
        "needs_hw": True,
        "buckets": ["load"],
        "policy_paths": [],
        "surfaces": {"load": ["crates/hipfire-loader/foo.rs"], "serve": [], "kernel": [], "policy": [], "other": []},
        "request": {"routes": [{"mode": "battery", "tag": "qwen3.6:27b"}], "claim": "test claim"},
        "request_error": None,
    }
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))
    fixtures = _fixtures_content(models_dir)
    fixtures_path = tmp / "fixtures.json"
    fixtures_path.write_text(json.dumps(fixtures))
    sol_response = {
        "phase": "prelim",
        "summary": "summary",
        "surfaces": ["load"],
        "suspected_regressions": [],
        "run_hardware": True,
        "run_hardware_reasons": ["safe"],
        "routes": [
            {"mode": "battery", "tag": "qwen3.6:27b", "source": "sol", "why": "sol why"},
            {"mode": "battery", "tag": "unknown:tag", "source": "sol", "why": "unknown should be dropped"},
        ],
        "unavailable_routes": [],
        "claim_assessment": "claim ok",
        "questions_for_author": ["q1"]
    }
    responses = [{"json": sol_response}]
    resp_path = tmp / "resp.json"
    resp_path.write_text(json.dumps(responses))
    call_count = tmp / "omp_count"
    gh_log = tmp / "gh_log.jsonl"
    gh_comments = tmp / "gh_comments.json"
    gh_comments.write_text("[]")
    omp_log = tmp / "omp_log.jsonl"
    out_path = tmp / "prelim.json"
    routes_path = tmp / "routes.json"
    system_prompt = tmp / "sol.md"
    system_prompt.write_text("prompt")
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_RESPONSES"] = str(resp_path)
    env["FAKE_OMP_CALL_COUNT"] = str(call_count)
    env["FAKE_OMP_LOG"] = str(omp_log)
    env["FAKE_GH_LOG"] = str(gh_log)
    env["FAKE_GH_COMMENTS"] = str(gh_comments)
    env["HIPFIRE_MODELS_DIR"] = str(models_dir)
    cmd = [
        sys.executable, str(REVIEW_PATH),
        "--seat", "sol", "--phase", "prelim",
        "--repo", "o/r", "--pr", "1",
        "--base", base, "--head", head,
        "--checkout", str(checkout),
        "--select", str(select_path),
        "--fixtures", str(fixtures_path),
        "--system-prompt", str(system_prompt),
        "--out", str(out_path),
        "--routes", str(routes_path),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"stderr {result.stderr}"
    # prelim.json
    data = json.loads(out_path.read_text())
    assert data["schema"] == "hipfire.hw-gate.prelim"
    assert data["version"] == 2
    assert data["seat"] == "sol"
    assert data["run_hardware"] is True
    assert data["prelim"]["summary"] == "summary"
    # routes.json bucket (2 fixtures *1 mode=2) union sol (known tag duplicate => still 2)
    routes = json.loads(routes_path.read_text())
    tags = [r["tag"] for r in routes]
    assert "qwen3.6:27b" in tags
    assert "ornith-1.5:35b-a3b-mq4r" in tags
    # unknown dropped
    assert "unknown:tag" not in tags
    # comment contains routes table and unavailable?
    comments = json.loads(gh_comments.read_text())
    bodies = [c["body"] for c in comments]
    assert any("<!-- hw-gate:sol-prelim -->" in b for b in bodies)
    # omp argv check
    omp_calls = [json.loads(l)["args"] for l in omp_log.read_text().splitlines() if l.strip()]
    assert len(omp_calls) >= 1
    oc = omp_calls[0]
    assert "--model" in oc
    model_idx = oc.index("--model") + 1
    assert oc[model_idx] == os.environ.get("HW_GATE_REVIEW_MODEL", "gpt-5.6-sol")
    assert "--system-prompt" in oc
    assert "@" in " ".join(oc) or any(a.startswith("@") for a in oc)

def test_prelim_unavailable_listed():
    tmp = Path(tempfile.mkdtemp())
    models_dir = tmp / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "qwen3.6-27b.mq4").write_text("dummy")
    # ornith missing => unavailable
    checkout, base, head = _make_repo(tmp / "repo2")
    select = {
        "schema": "hipfire.hw-gate.select","version":1,"needs_hw":True,"buckets":["load"],"policy_paths":[],"surfaces":{"load":[],"serve":[],"kernel":[],"policy":[],"other":[]},
        "request": None, "request_error": None,
    }
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))
    fixtures = _fixtures_content(models_dir)
    fixtures_path = tmp / "fixtures.json"
    fixtures_path.write_text(json.dumps(fixtures))
    sol_response = {
        "phase": "prelim","summary":"s","surfaces":["load"],"suspected_regressions":[],
        "run_hardware": True,"run_hardware_reasons":["r"],
        "routes":[{"mode":"battery","tag":"ornith-1.5:35b-a3b-mq4r","source":"sol","why":"needs ornith"}],
        "unavailable_routes":[],"claim_assessment":"","questions_for_author":[]
    }
    resp_path = tmp / "resp.json"
    resp_path.write_text(json.dumps([{"json": sol_response}]))
    call_count = tmp / "omp_count"
    gh_comments = tmp / "gh_comments.json"
    gh_comments.write_text("[]")
    out_path = tmp / "prelim.json"
    routes_path = tmp / "routes.json"
    system_prompt = tmp / "sol.md"
    system_prompt.write_text("prompt")
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_RESPONSES"] = str(resp_path)
    env["FAKE_OMP_CALL_COUNT"] = str(call_count)
    env["FAKE_GH_LOG"] = str(tmp / "gh_log.jsonl")
    env["FAKE_GH_COMMENTS"] = str(gh_comments)
    env["FAKE_OMP_LOG"] = str(tmp / "omp_log.jsonl")
    env["HIPFIRE_MODELS_DIR"] = str(models_dir)
    cmd = [sys.executable, str(REVIEW_PATH),"--seat","sol","--phase","prelim","--repo","o/r","--pr","1","--base",base,"--head",head,"--checkout",str(checkout),"--select",str(select_path),"--fixtures",str(fixtures_path),"--system-prompt",str(system_prompt),"--out",str(out_path),"--routes",str(routes_path)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    comments = json.loads(gh_comments.read_text())
    bodies = [c["body"] for c in comments]
    # unavailable should be listed in comment
    assert any("ornith-1.5:35b-a3b-mq4r" in b for b in bodies)
    # routes still contains known tag (even if unavailable, per spec may still be in routes)
    routes = json.loads(routes_path.read_text())
    assert any(r["tag"]=="ornith-1.5:35b-a3b-mq4r" for r in routes)

def test_prelim_run_hardware_false_on_garbage():
    tmp = Path(tempfile.mkdtemp())
    models_dir = tmp / "models"
    models_dir.mkdir()
    checkout, base, head = _make_repo(tmp / "repo3")
    select = {"schema":"hipfire.hw-gate.select","version":1,"needs_hw":True,"buckets":["load"],"policy_paths":[],"surfaces":{"load":[],"serve":[],"kernel":[],"policy":[],"other":[]},"request":None,"request_error":None}
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))
    fixtures = _fixtures_content(models_dir)
    fixtures_path = tmp / "fixtures.json"
    fixtures_path.write_text(json.dumps(fixtures))
    call_count = tmp / "omp_count"
    gh_comments = tmp / "gh_comments.json"
    gh_comments.write_text("[]")
    out_path = tmp / "prelim.json"
    routes_path = tmp / "routes.json"
    system_prompt = tmp / "sol.md"
    system_prompt.write_text("prompt")
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_GARBAGE"] = "1"
    env["FAKE_OMP_CALL_COUNT"] = str(call_count)
    env["FAKE_GH_LOG"] = str(tmp / "gh_log.jsonl")
    env["FAKE_GH_COMMENTS"] = str(gh_comments)
    env["FAKE_OMP_LOG"] = str(tmp / "omp_log.jsonl")
    env["HIPFIRE_MODELS_DIR"] = str(models_dir)
    cmd = [sys.executable, str(REVIEW_PATH),"--seat","sol","--phase","prelim","--repo","o/r","--pr","1","--base",base,"--head",head,"--checkout",str(checkout),"--select",str(select_path),"--fixtures",str(fixtures_path),"--system-prompt",str(system_prompt),"--out",str(out_path),"--routes",str(routes_path)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(out_path.read_text())
    assert data["run_hardware"] is False
    assert data["prelim"] is None
    # routes still bucket routes
    routes = json.loads(routes_path.read_text())
    assert len(routes) == 2  # 2 fixtures load
    comments = json.loads(gh_comments.read_text())
    assert any("sol prelim unavailable" in b.lower() or "run_hardware" in b.lower() for b in [c["body"] for c in comments])

def test_prelim_relative_checkout_reaches_the_seat():
    # The workflow runs `review.py --checkout pr` from the job workspace. The
    # first live run on #686 died before Sol read anything: omp was launched
    # with cwd=pr and `--cwd pr`, resolved `pr/pr`, and exited 1.
    tmp = Path(tempfile.mkdtemp())
    models_dir = tmp / "models"
    models_dir.mkdir()
    checkout, base, head = _make_repo(tmp / "pr")
    select = {"schema":"hipfire.hw-gate.select","version":1,"needs_hw":True,"buckets":["load"],"policy_paths":[],"surfaces":{"load":[],"serve":[],"kernel":[],"policy":[],"other":[]},"request":None,"request_error":None}
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))
    fixtures_path = tmp / "fixtures.json"
    fixtures_path.write_text(json.dumps(_fixtures_content(models_dir)))
    sol_response = {
        "phase": "prelim","summary":"s","surfaces":["load"],"suspected_regressions":[],
        "run_hardware": True,"run_hardware_reasons":["touches the loader"],
        "routes":[],"unavailable_routes":[],"claim_assessment":"","questions_for_author":[]
    }
    resp_path = tmp / "resp.json"
    resp_path.write_text(json.dumps([{"json": sol_response}]))
    gh_comments = tmp / "gh_comments.json"
    gh_comments.write_text("[]")
    out_path = tmp / "prelim.json"
    routes_path = tmp / "routes.json"
    system_prompt = tmp / "sol.md"
    system_prompt.write_text("prompt")
    omp_log = tmp / "omp_log.jsonl"
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_RESPONSES"] = str(resp_path)
    env["FAKE_OMP_CALL_COUNT"] = str(tmp / "omp_count")
    env["FAKE_GH_LOG"] = str(tmp / "gh_log.jsonl")
    env["FAKE_GH_COMMENTS"] = str(gh_comments)
    env["FAKE_OMP_LOG"] = str(omp_log)
    env["HIPFIRE_MODELS_DIR"] = str(models_dir)
    rel_checkout = os.path.relpath(checkout, tmp)  # "pr/checkout", as the workflow's "pr"
    cmd = [sys.executable, str(REVIEW_PATH),"--seat","sol","--phase","prelim","--repo","o/r","--pr","1","--base",base,"--head",head,"--checkout",rel_checkout,"--select",str(select_path),"--fixtures",str(fixtures_path),"--system-prompt",str(system_prompt),"--out",str(out_path),"--routes",str(routes_path)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=tmp)
    assert result.returncode == 0, result.stderr
    data = json.loads(out_path.read_text())
    assert data["prelim"] is not None, gh_comments.read_text()
    assert data["run_hardware"] is True
    launches = [json.loads(l) for l in omp_log.read_text().splitlines() if l.strip()]
    assert launches, "seat was never launched"
    cwd_arg = launches[0]["args"][launches[0]["args"].index("--cwd") + 1]
    assert os.path.isabs(cwd_arg) and Path(cwd_arg) == checkout.resolve()

# ---------------------------------------------------------------------------
# verdict e2e
# ---------------------------------------------------------------------------

def _run_verdict(tmp: Path, select_extra: dict | None = None, evidence_extra: dict | None = None, verdict_response: dict | None = None, hw_run_result="success", evidence_md_exists=True):
    checkout, base, head = _make_repo(tmp / "checkout_v")
    select = {"schema":"hipfire.hw-gate.select","version":1,"needs_hw":True,"buckets":["load"],"policy_paths":[],"surfaces":{"load":["foo.rs"],"serve":[],"kernel":[],"policy":[],"other":[]},"request":None,"request_error":None}
    if select_extra:
        select.update(select_extra)
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))
    # prelim
    prelim_json = {
        "schema":"hipfire.hw-gate.prelim","version":2,"seat":"sol","model":"gpt-5.6-sol",
        "prelim":{"phase":"prelim","summary":"s","surfaces":["load"],"suspected_regressions":[],"run_hardware":True,"run_hardware_reasons":["r"],"routes":[],"unavailable_routes":[],"claim_assessment":"","questions_for_author":[]},
        "run_hardware": True, "posted": {"prelim_comment": "http://x"}
    }
    prelim_path = tmp / "prelim.json"
    prelim_path.write_text(json.dumps(prelim_json))
    evidence = {"schema":"hipfire.hw-gate.evidence","version":1,"verdict":"pass","base":base,"head":head,"buckets":["load"],"fixtures":[],"kernel":None}
    if evidence_extra:
        evidence.update(evidence_extra)
    evidence_path = tmp / "hw-gate.json"
    evidence_path.write_text(json.dumps(evidence))
    if evidence_md_exists:
        (tmp / "hw-gate.md").write_text("# Evidence\n\nPer-fixture table\n")
        # need to be alongside evidence_path.parent? In our impl we look at evidence_path.parent/hw-gate.md - which is tmp/hw-gate.md, correct as we wrote.
    else:
        p = tmp / "hw-gate.md"
        if p.exists():
            p.unlink()
    system_prompt = tmp / "sol.md"
    system_prompt.write_text("prompt")
    if verdict_response is None:
        verdict_response = {"phase":"verdict","decision":"greenlight","confidence":0.9,"regressions":[],"coverage":{"surfaces_touched":["load"],"surfaces_evidenced":["load"],"gaps":[]},"claim_verdict":"no-claim","eyeball":[],"rationale":"ok"}
    resp_path = tmp / "resp.json"
    resp_path.write_text(json.dumps([{"json": verdict_response}]))
    call_count = tmp / "omp_count"
    gh_log = tmp / "gh_log.jsonl"
    gh_comments = tmp / "gh_comments.json"
    gh_comments.write_text("[]")
    omp_log = tmp / "omp_log.jsonl"
    out_path = tmp / "verdict.json"
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_RESPONSES"] = str(resp_path)
    env["FAKE_OMP_CALL_COUNT"] = str(call_count)
    env["FAKE_OMP_LOG"] = str(omp_log)
    env["FAKE_GH_LOG"] = str(gh_log)
    env["FAKE_GH_COMMENTS"] = str(gh_comments)
    cmd = [sys.executable, str(REVIEW_PATH),"--seat","sol","--phase","verdict","--repo","o/r","--pr","1","--base",base,"--head",head,"--checkout",str(checkout),"--select",str(select_path),"--prelim",str(prelim_path),"--evidence",str(evidence_path),"--hw-run-result",hw_run_result,"--system-prompt",str(system_prompt),"--out",str(out_path)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return result, out_path, gh_log, gh_comments, omp_log, base, head, checkout

def test_verdict_hard_evidence_block():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, gh_log, gh_comments, omp_log, base, head, checkout = _run_verdict(tmp, evidence_extra={"verdict":"fail"}, verdict_response={"phase":"verdict","decision":"greenlight","confidence":0.9,"regressions":[],"coverage":{"surfaces_touched":["load"],"surfaces_evidenced":["load"],"gaps":[]},"claim_verdict":"no-claim","eyeball":[],"rationale":"ok"})
    assert result.returncode == 0  # hard evidence block still exit 0 (model greenlight but floor blocks) but final is block
    data = json.loads(out_path.read_text())
    assert data["floor"]["final_decision"] == "block"
    assert any("evidence" in r for r in data["floor"]["hard"])
    # review should be --comment only never approve
    gh_calls = [json.loads(l)["args"] for l in gh_log.read_text().splitlines() if l.strip()]
    review_calls = [a for a in gh_calls if a[:2]==["pr","review"]]
    assert review_calls
    assert "--comment" in review_calls[-1]
    assert "--approve" not in review_calls[-1]

def test_verdict_hard_policy_hold():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, *_ = _run_verdict(tmp, select_extra={"policy_paths":["scripts/hw-gate/review.py"]}, verdict_response={"phase":"verdict","decision":"greenlight","confidence":0.9,"regressions":[],"coverage":{"surfaces_touched":["load"],"surfaces_evidenced":["load"],"gaps":[]},"claim_verdict":"no-claim","eyeball":[],"rationale":"ok"})
    data = json.loads(out_path.read_text())
    assert data["floor"]["final_decision"] == "needs-human"
    assert any("policy" in r for r in data["floor"]["hard"])

def test_verdict_soft_coverage_needs_human():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, *_ = _run_verdict(tmp, verdict_response={"phase":"verdict","decision":"greenlight","confidence":0.9,"regressions":[],"coverage":{"surfaces_touched":["load"],"surfaces_evidenced":[],"gaps":["serve"]},"claim_verdict":"no-claim","eyeball":[],"rationale":"ok"})
    data = json.loads(out_path.read_text())
    assert data["floor"]["final_decision"] == "needs-human"
    assert any("coverage" in r for r in data["floor"]["soft"])

def test_verdict_soft_confidence_needs_human():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, *_ = _run_verdict(tmp, verdict_response={"phase":"verdict","decision":"greenlight","confidence":0.5,"regressions":[],"coverage":{"surfaces_touched":["load"],"surfaces_evidenced":["load"],"gaps":[]},"claim_verdict":"no-claim","eyeball":[],"rationale":"ok"})
    data = json.loads(out_path.read_text())
    assert data["floor"]["final_decision"] == "needs-human"
    assert any("confidence" in r for r in data["floor"]["soft"])

def test_verdict_greenlight_only_when_clear():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, gh_log, gh_comments, *_ = _run_verdict(tmp)
    assert result.returncode == 0
    data = json.loads(out_path.read_text())
    assert data["floor"]["final_decision"] == "greenlight"
    # comment marker
    comments = json.loads(gh_comments.read_text())
    assert any("<!-- hw-gate:sol-verdict -->" in c["body"] for c in comments)
    # evidence comment posted
    assert any("<!-- hw-gate:evidence -->" in c["body"] for c in comments)

def test_verdict_unparseable_needs_human_exit1():
    tmp = Path(tempfile.mkdtemp())
    checkout, base, head = _make_repo(tmp / "repo")
    select = {"schema":"hipfire.hw-gate.select","version":1,"needs_hw":True,"buckets":["load"],"policy_paths":[],"surfaces":{"load":[],"serve":[],"kernel":[],"policy":[],"other":[]},"request":None,"request_error":None}
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))
    prelim_json = {"schema":"hipfire.hw-gate.prelim","version":2,"seat":"sol","model":"gpt-5.6-sol","prelim":{"phase":"prelim","summary":"s","surfaces":["load"],"suspected_regressions":[],"run_hardware":True,"run_hardware_reasons":[],"routes":[],"unavailable_routes":[],"claim_assessment":"","questions_for_author":[]},"run_hardware":True,"posted":{}}
    prelim_path = tmp / "prelim.json"
    prelim_path.write_text(json.dumps(prelim_json))
    evidence = {"schema":"hipfire.hw-gate.evidence","version":1,"verdict":"pass","base":base,"head":head,"buckets":["load"],"fixtures":[],"kernel":None}
    evidence_path = tmp / "hw-gate.json"
    evidence_path.write_text(json.dumps(evidence))
    (tmp / "hw-gate.md").write_text("# Ev\n")
    system_prompt = tmp / "sol.md"
    system_prompt.write_text("prompt")
    gh_comments = tmp / "gh_comments.json"
    gh_comments.write_text("[]")
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_GARBAGE"] = "1"
    env["FAKE_OMP_CALL_COUNT"] = str(tmp / "omp_count")
    env["FAKE_GH_LOG"] = str(tmp / "gh_log.jsonl")
    env["FAKE_GH_COMMENTS"] = str(gh_comments)
    env["FAKE_OMP_LOG"] = str(tmp / "omp_log.jsonl")
    out_path = tmp / "verdict.json"
    cmd = [sys.executable, str(REVIEW_PATH),"--seat","sol","--phase","verdict","--repo","o/r","--pr","1","--base",base,"--head",head,"--checkout",str(checkout),"--select",str(select_path),"--prelim",str(prelim_path),"--evidence",str(evidence_path),"--hw-run-result","success","--system-prompt",str(system_prompt),"--out",str(out_path)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    data = json.loads(out_path.read_text())
    assert data["floor"]["final_decision"] == "needs-human"
    assert data["verdict"] is None

# ---------------------------------------------------------------------------
# decide e2e
# ---------------------------------------------------------------------------

def _run_decide(tmp: Path, select_extra: dict | None = None, evidence_extra: dict | None = None, sol_final="greenlight", fable_response: dict | None = None, hw_run_result="success", merge_409=False, checkout_extra_files: dict | None = None):
    checkout, base, head = _make_repo(tmp / "checkout_d", files=checkout_extra_files)
    select = {"schema":"hipfire.hw-gate.select","version":1,"needs_hw":True,"buckets":["load"],"policy_paths":[],"surfaces":{"load":["foo.rs"],"serve":[],"kernel":[],"policy":[],"other":[]},"request":None,"request_error":None}
    if select_extra:
        select.update(select_extra)
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))
    prelim_json = {"schema":"hipfire.hw-gate.prelim","version":2,"seat":"sol","model":"gpt-5.6-sol","prelim":{"phase":"prelim","summary":"s","surfaces":["load"],"suspected_regressions":[],"run_hardware":True,"run_hardware_reasons":[],"routes":[],"unavailable_routes":[],"claim_assessment":"","questions_for_author":[]},"run_hardware":True,"posted":{}}
    prelim_path = tmp / "prelim.json"
    prelim_path.write_text(json.dumps(prelim_json))
    evidence = {"schema":"hipfire.hw-gate.evidence","version":1,"verdict":"pass","base":base,"head":head,"buckets":["load"],"fixtures":[],"kernel":None}
    if evidence_extra:
        evidence.update(evidence_extra)
    evidence_path = tmp / "hw-gate.json"
    evidence_path.write_text(json.dumps(evidence))
    (tmp / "hw-gate.md").write_text("# Evidence\n")
    # sol verdict file with desired final
    sol_verdict_data = {"phase":"verdict","decision": sol_final, "confidence": 0.9 if sol_final=="greenlight" else (0.5 if sol_final=="needs-human" and False else 0.9), "regressions":[],"coverage":{"surfaces_touched":["load"],"surfaces_evidenced":["load"],"gaps": [] if sol_final!="needs-human" else ["serve"]},"claim_verdict":"no-claim","eyeball":[],"rationale":"sol ok" }
    # For needs-human due to coverage, set gaps
    if sol_final == "needs-human":
        sol_verdict_data = {"phase":"verdict","decision":"needs-human","confidence":0.9,"regressions":[],"coverage":{"surfaces_touched":["load"],"surfaces_evidenced":[],"gaps":["serve"]},"claim_verdict":"no-claim","eyeball":[],"rationale":"sol needs human"}
    sol_floor_final = sol_final if sol_final in ("greenlight","needs-human","block") else "needs-human"
    sol_verdict_file = {"schema":"hipfire.hw-gate.verdict","version":2,"seat":"sol","model":"gpt-5.6-sol","verdict": sol_verdict_data,"floor":{"hard":[],"soft":[] if sol_final=="greenlight" else ["coverage"],"model_decision":sol_final,"final_decision":sol_floor_final},"posted":{}}
    verdict_path = tmp / "verdict.json"
    verdict_path.write_text(json.dumps(sol_verdict_file))
    system_prompt = tmp / "fable.md"
    system_prompt.write_text("fable prompt")
    if fable_response is None:
        fable_response = {"phase":"decide","decision":"merge-staging","agrees_with_sol":True,"override":None,"regressions":[],"further_evidence_wanted":[],"rationale":"fable rationale","announcement":"Fable merges."}
    resp_path = tmp / "resp.json"
    resp_path.write_text(json.dumps([{"json": fable_response}]))
    call_count = tmp / "omp_count"
    gh_log = tmp / "gh_log.jsonl"
    gh_comments = tmp / "gh_comments.json"
    gh_comments.write_text("[]")
    omp_log = tmp / "omp_log.jsonl"
    out_path = tmp / "decision.json"
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_RESPONSES"] = str(resp_path)
    env["FAKE_OMP_CALL_COUNT"] = str(call_count)
    env["FAKE_OMP_LOG"] = str(omp_log)
    env["FAKE_GH_LOG"] = str(gh_log)
    env["FAKE_GH_COMMENTS"] = str(gh_comments)
    if merge_409:
        env["FAKE_GH_MERGE_409"] = "1"
    env["HW_GATE_DECIDE_MODEL"] = "claude-fable-5-1"
    cmd = [sys.executable, str(REVIEW_PATH),"--seat","fable","--phase","decide","--repo","o/r","--pr","1","--base",base,"--head",head,"--checkout",str(checkout),"--select",str(select_path),"--prelim",str(prelim_path),"--evidence",str(evidence_path),"--verdict",str(verdict_path),"--hw-run-result",hw_run_result,"--staging","beta","--system-prompt",str(system_prompt),"--out",str(out_path)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return result, out_path, gh_log, gh_comments, omp_log, base, head, checkout

def test_decide_hard_floor_beats_merge():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, gh_log, gh_comments, omp_log, base, head, checkout = _run_decide(tmp, evidence_extra={"verdict":"fail"}, sol_final="greenlight", fable_response={"phase":"decide","decision":"merge-staging","agrees_with_sol":True,"override":None,"regressions":[],"further_evidence_wanted":[],"rationale":"fable wants merge","announcement":"Fable merges."})
    data = json.loads(out_path.read_text())
    assert data["decision_final"] == "block"
    assert data["floor"]["hard"] != []
    # should not have called merges
    gh_calls = [json.loads(l)["args"] for l in gh_log.read_text().splitlines() if l.strip()]
    merges_calls = [a for a in gh_calls if ("merges" in a[1] if len(a)>1 else False)]
    assert len(merges_calls) == 0

def test_decide_override_needs_human_to_merge():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, gh_log, gh_comments, omp_log, base, head, checkout = _run_decide(tmp, sol_final="needs-human", fable_response={"phase":"decide","decision":"merge-staging","agrees_with_sol":False,"override":{"of":"needs-human","why":"evidence closes gap"},"regressions":[],"further_evidence_wanted":[],"rationale":"override rationale","announcement":"Fable merges after review."})
    assert result.returncode == 0, result.stderr
    data = json.loads(out_path.read_text())
    assert data["decision_final"] == "merge-staging"
    assert data["override"] is not None
    assert data["override"]["of"] == "needs-human"
    assert data["merged"] is not None
    assert data["merged"]["base"] == "beta"
    assert data["merged"]["head"] == head
    # merges api called with base=beta
    gh_calls = [json.loads(l)["args"] for l in gh_log.read_text().splitlines() if l.strip()]
    merges_calls = [a for a in gh_calls if ("merges" in a[1] if len(a)>1 else False)]
    assert len(merges_calls) == 1
    arg_str = " ".join(merges_calls[0])
    assert "base=beta" in arg_str
    assert f"head={head}" in arg_str
    assert "hw-gate: merge PR #1" in arg_str
    # labels: merged-staging added, hw-run removed
    assert "merged-staging" in data["posted"]["labels_added"]
    assert "hw-run" in data["posted"]["labels_removed"]
    # review should be approve
    assert any("--approve" in " ".join(a) for a in gh_calls if a[:2]==["pr","review"])
    # omp model should be fable
    omp_calls = [json.loads(l)["args"] for l in omp_log.read_text().splitlines() if l.strip()]
    assert any("claude-fable-5-1" in " ".join(a) for a in omp_calls)

def test_decide_veto_greenlight_to_hold():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, gh_log, gh_comments, omp_log, *_ = _run_decide(tmp, sol_final="greenlight", fable_response={"phase":"decide","decision":"hold","agrees_with_sol":False,"override":{"of":"greenlight","why":"veto, see decoded text"},"regressions":[],"further_evidence_wanted":[],"rationale":"veto rationale","announcement":"Fable holds."})
    data = json.loads(out_path.read_text())
    assert data["decision_final"] == "hold"
    assert data["override"] is not None
    assert data["override"]["of"] == "greenlight"
    gh_calls = [json.loads(l)["args"] for l in gh_log.read_text().splitlines() if l.strip()]
    merges_calls = [a for a in gh_calls if ("merges" in a[1] if len(a)>1 else False)]
    assert len(merges_calls) == 0
    # label needs-human
    assert "needs-human" in data["posted"]["labels_added"]
    assert any("--comment" in " ".join(a) for a in gh_calls if a[:2]==["pr","review"])
    assert not any("--approve" in " ".join(a) for a in gh_calls if a[:2]==["pr","review"])

def test_decide_409_conflict():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, gh_log, gh_comments, omp_log, *_ = _run_decide(tmp, sol_final="needs-human", fable_response={"phase":"decide","decision":"merge-staging","agrees_with_sol":False,"override":{"of":"needs-human","why":"override"},"regressions":[],"further_evidence_wanted":[],"rationale":"r","announcement":"merge"}, merge_409=True)
    data = json.loads(out_path.read_text())
    # nothing merged => the decision cannot stay merge-staging (the status would
    # go green with the head un-staged); it becomes a hold with the reason recorded
    assert data["decision_final"] == "hold"
    assert "staging_merge_conflict" in data["floor"]["hard"]
    assert data["merged"] is not None
    assert "error" in data["merged"]
    assert "409" in data["merged"]["error"] or "409" in str(data["merged"])
    # label should be needs-human instead of merged-staging
    assert "needs-human" in data["posted"]["labels_added"]
    # comment should note conflict
    comments = json.loads(gh_comments.read_text())
    bodies = [c["body"] for c in comments]
    assert any("409" in b or "conflict" in b.lower() for b in bodies)

def test_decide_labels_and_hw_run_removed():
    for dec, label, flag in [("merge-staging","merged-staging","--approve"),("hold","needs-human","--comment"),("block","hw-gate-blocked","--request-changes")]:
        tmp = Path(tempfile.mkdtemp())
        fable_resp = {"phase":"decide","decision":dec,"agrees_with_sol":True,"override":None,"regressions":[],"further_evidence_wanted":[],"rationale":"r","announcement":"a"}
        # For block, need sol_final block to avoid override confusion; but we test direct mapping when no hard floor.
        sol_final = "greenlight" if dec=="merge-staging" else ("needs-human" if dec=="hold" else "block")
        # Ensure sol_final matches so no override needed except hold case will be veto but ok.
        # For this table, we will set sol_final to match fable for clean.
        if dec == "merge-staging":
            sol_final_test = "needs-human"  # to allow merge override, but we cover separately; for this loop test hold/block.
            # skip merge case already covered, test hold and block
            if dec == "merge-staging":
                continue
        result, out_path, gh_log, gh_comments, *_ = _run_decide(tmp, sol_final=sol_final, fable_response=fable_resp)
        data = json.loads(out_path.read_text())
        assert data["decision_final"] == dec
        assert label in data["posted"]["labels_added"]
        assert "hw-run" in data["posted"]["labels_removed"]
        gh_calls = [json.loads(l)["args"] for l in gh_log.read_text().splitlines() if l.strip()]
        review_calls = [a for a in gh_calls if a[:2]==["pr","review"]]
        assert any(flag in " ".join(rc) for rc in review_calls)

def test_omp_argv_per_seat_and_help():
    # check help
    result = subprocess.run([sys.executable, str(REVIEW_PATH), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    help_text = result.stdout + result.stderr
    for flag in ["--seat","--phase","--select","--fixtures","--prelim","--evidence","--verdict","--hw-run-result","--staging","--routes","--system-prompt","--out"]:
        assert flag in help_text, f"missing {flag} in help"

def test_merges_api_call_exact():
    tmp = Path(tempfile.mkdtemp())
    result, out_path, gh_log, gh_comments, omp_log, base, head, checkout = _run_decide(tmp, sol_final="needs-human", fable_response={"phase":"decide","decision":"merge-staging","agrees_with_sol":False,"override":{"of":"needs-human","why":"why"},"regressions":[],"further_evidence_wanted":[],"rationale":"r","announcement":"a"})
    gh_calls = [json.loads(l)["args"] for l in gh_log.read_text().splitlines() if l.strip()]
    merges_calls = [a for a in gh_calls if ("merges" in a[1] if len(a)>1 else False)]
    assert merges_calls
    call = merges_calls[0]
    # expected: gh api repos/o/r/merges -f base=beta -f head=<sha> -f commit_message=...
    assert call[0] == "api"
    assert "repos/o/r/merges" in call[1]
    arg_str = " ".join(call)
    assert "base=beta" in arg_str
    assert f"head={head}" in arg_str
    assert "hw-gate: merge PR #1" in arg_str

def test_fable_unavailable_hold_exit1():
    tmp = Path(tempfile.mkdtemp())
    checkout, base, head = _make_repo(tmp / "checkout_f")
    select = {"schema":"hipfire.hw-gate.select","version":1,"needs_hw":True,"buckets":["load"],"policy_paths":[],"surfaces":{"load":[],"serve":[],"kernel":[],"policy":[],"other":[]},"request":None,"request_error":None}
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))
    prelim_json = {"schema":"hipfire.hw-gate.prelim","version":2,"seat":"sol","model":"gpt-5.6-sol","prelim":{"phase":"prelim","summary":"s","surfaces":["load"],"suspected_regressions":[],"run_hardware":True,"run_hardware_reasons":[],"routes":[],"unavailable_routes":[],"claim_assessment":"","questions_for_author":[]},"run_hardware":True,"posted":{}}
    prelim_path = tmp / "prelim.json"
    prelim_path.write_text(json.dumps(prelim_json))
    evidence = {"schema":"hipfire.hw-gate.evidence","version":1,"verdict":"pass","base":base,"head":head,"buckets":["load"],"fixtures":[],"kernel":None}
    evidence_path = tmp / "hw-gate.json"
    evidence_path.write_text(json.dumps(evidence))
    (tmp / "hw-gate.md").write_text("# Ev\n")
    sol_verdict_file = {"schema":"hipfire.hw-gate.verdict","version":2,"seat":"sol","model":"gpt-5.6-sol","verdict":{"phase":"verdict","decision":"needs-human","confidence":0.9,"regressions":[],"coverage":{"surfaces_touched":["load"],"surfaces_evidenced":[],"gaps":["serve"]},"claim_verdict":"no-claim","eyeball":[],"rationale":"sol"},"floor":{"hard":[],"soft":["coverage"],"model_decision":"needs-human","final_decision":"needs-human"},"posted":{}}
    verdict_path = tmp / "verdict.json"
    verdict_path.write_text(json.dumps(sol_verdict_file))
    system_prompt = tmp / "fable.md"
    system_prompt.write_text("prompt")
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_GARBAGE"] = "1"
    env["FAKE_OMP_CALL_COUNT"] = str(tmp / "omp_count")
    env["FAKE_GH_LOG"] = str(tmp / "gh_log.jsonl")
    env["FAKE_GH_COMMENTS"] = str(tmp / "gh_comments.json")
    env["FAKE_OMP_LOG"] = str(tmp / "omp_log.jsonl")
    out_path = tmp / "decision.json"
    cmd = [sys.executable, str(REVIEW_PATH),"--seat","fable","--phase","decide","--repo","o/r","--pr","1","--base",base,"--head",head,"--checkout",str(checkout),"--select",str(select_path),"--prelim",str(prelim_path),"--evidence",str(evidence_path),"--verdict",str(verdict_path),"--hw-run-result","success","--staging","beta","--system-prompt",str(system_prompt),"--out",str(out_path)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    data = json.loads(out_path.read_text())
    assert data["decision_final"] == "hold"
    assert data["decision"] is None


# ---------------------------------------------------------------------------
# investigate mode: the sandbox env is the security boundary
# ---------------------------------------------------------------------------

def _investigate_args(tmp, **over):
    class A: pass
    a = A()
    a.devices = over.get("devices", "0,1,2,3,4")
    a.home = str(tmp / "home")
    a.evidence_dir = str(tmp / "ev")
    a.bin = str(tmp / "bin")
    a.base_bin = over.get("base_bin", str(tmp / "base-bin"))
    a.round = over.get("round", 1)
    a.base = "basesha"
    return a


def test_investigate_env_is_an_allow_list_and_strips_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("GH_TOKEN", "ghs_secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_secret2")
    monkeypatch.setenv("HW_GATE_SOL_PRIVATE_KEY", "-----BEGIN")
    monkeypatch.setenv("HW_GATE_FABLE_APP_ID", "123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("SOME_SECRET", "x")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "x")
    monkeypatch.setenv("RANDOM_UNRELATED_VAR", "x")
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "12.0.1")
    monkeypatch.setenv("HW_GATE_MAX_MINUTES", "7")
    env, minutes = review._build_investigate_env(_investigate_args(tmp_path))
    for forbidden in ("GH_TOKEN", "GITHUB_TOKEN", "HW_GATE_SOL_PRIVATE_KEY", "HW_GATE_FABLE_APP_ID",
                      "OPENAI_API_KEY", "SOME_SECRET", "AWS_SESSION_TOKEN", "RANDOM_UNRELATED_VAR"):
        assert forbidden not in env, forbidden
    assert env["HW_GATE_DEVICES"] == "0,1,2,3,4" and env["HIP_VISIBLE_DEVICES"] == "0,1,2,3,4"
    assert env["HIPFIRE_MODELS_DIR"] == str(tmp_path / "models")
    assert env["HIPFIRE_HOME"] == str(tmp_path / "home") and (tmp_path / "home").is_dir()
    assert env["HW_GATE_EVIDENCE"] == str(tmp_path / "ev") and (tmp_path / "ev").is_dir()
    assert env["HW_GATE_BIN"] == str(tmp_path / "bin") and env["HW_GATE_BASE_BIN"] == str(tmp_path / "base-bin")
    assert env["HW_GATE_BASE_SHA"] == "basesha" and env["HW_GATE_ROUND"] == "1" and env["HW_GATE_MAX_MINUTES"] == "7"
    assert env["HSA_OVERRIDE_GFX_VERSION"] == "12.0.1"
    assert minutes == "7"


def test_investigate_env_requires_models_dir_and_devices(monkeypatch, tmp_path):
    monkeypatch.delenv("HIPFIRE_MODELS_DIR", raising=False)
    with pytest.raises(review.ReviewError, match="HIPFIRE_MODELS_DIR"):
        review._build_investigate_env(_investigate_args(tmp_path))
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(tmp_path))
    with pytest.raises(review.ReviewError, match="devices"):
        review._build_investigate_env(_investigate_args(tmp_path, devices=""))
    env, _ = review._build_investigate_env(_investigate_args(tmp_path, base_bin=None))
    assert "HW_GATE_BASE_BIN" not in env


def test_investigate_omp_argv_has_full_tools_and_xhigh(monkeypatch, tmp_path):
    """The investigate invocation must not restrict tools and must run at xhigh with the budget."""
    captured = {}
    def fake_run(cmd, **kw):
        captured["cmd"] = cmd; captured["env"] = kw.get("env")
        class R: returncode, stdout, stderr = 0, "", ""
        return R()
    monkeypatch.setattr(review.subprocess, "run", fake_run)
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("GH_TOKEN", "ghs_secret")
    env, minutes = review._build_investigate_env(_investigate_args(tmp_path))
    try:
        review.omp_investigate("prompt text", str(tmp_path / "fable.md"), str(tmp_path), "claude-fable-5-1", env, minutes)
    except review.ReviewError:
        pass  # empty stdout => no decision; we only inspect the invocation here
    cmd = captured["cmd"]
    assert not any(a.startswith("--tools") for a in cmd), cmd
    assert "--thinking" in cmd and cmd[cmd.index("--thinking") + 1] == "xhigh"
    assert cmd[cmd.index("--max-time") + 1] == f"{minutes}m"
    assert "GH_TOKEN" not in captured["env"] and captured["env"]["HW_GATE_DEVICES"] == "0,1,2,3,4"
