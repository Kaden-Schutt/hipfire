import hashlib
import json
import sys
import os
from pathlib import Path
import subprocess

import pytest

import importlib.util

RUN_PATH = Path(__file__).resolve().parents[1] / "run.py"
spec = importlib.util.spec_from_file_location("hw_gate_run", str(RUN_PATH))
run_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_mod)


def test_verify_fixture_cache_hit_miss_mismatch(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    cache_path = tmp_path / "home" / "hw-gate-sha.json"
    content = b"hello world " * 1000
    file_name = "test.mq4"
    fpath = models_dir / file_name
    fpath.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    size = len(content)
    fixture = {"tag": "test:tag", "file": file_name, "sha256": sha, "size_bytes": size}
    res1 = run_mod.verify_fixture(str(models_dir), fixture, str(cache_path))
    assert res1["sha256_ok"] is True
    assert res1["size_ok"] is True
    assert res1["exists"] is True
    assert cache_path.is_file()
    cache1 = json.loads(cache_path.read_text())
    assert len(cache1) == 1
    res2 = run_mod.verify_fixture(str(models_dir), fixture, str(cache_path))
    assert res2["sha256_ok"] is True
    assert res2["actual_sha256"] == sha
    cache2 = json.loads(cache_path.read_text())
    assert cache1 == cache2
    fixture_bad_sha = {"tag": "test:tag", "file": file_name, "sha256": "0"*64, "size_bytes": size}
    res3 = run_mod.verify_fixture(str(models_dir), fixture_bad_sha, str(cache_path))
    assert res3["sha256_ok"] is False
    assert res3["size_ok"] is True
    assert "sha256 mismatch" in res3["reason"]
    fixture_bad_size = {"tag": "test:tag", "file": file_name, "sha256": sha, "size_bytes": size+1}
    res4 = run_mod.verify_fixture(str(models_dir), fixture_bad_size, str(cache_path))
    assert res4["size_ok"] is False
    assert "size mismatch" in res4["reason"]
    fixture_missing = {"tag": "missing:tag", "file": "nope.mq4", "sha256": sha, "size_bytes": size}
    res5 = run_mod.verify_fixture(str(models_dir), fixture_missing, str(cache_path))
    assert res5["exists"] is False
    assert res5["sha256_ok"] is False
    new_content = b"x" * len(content)
    fpath.write_bytes(new_content)
    new_sha = hashlib.sha256(new_content).hexdigest()
    fixture_new = {"tag": "test:tag", "file": file_name, "sha256": new_sha, "size_bytes": size}
    res6 = run_mod.verify_fixture(str(models_dir), fixture_new, str(cache_path))
    assert res6["sha256_ok"] is True
    assert res6["actual_sha256"] == new_sha
    cache3 = json.loads(cache_path.read_text())
    assert len(cache3) >= 2


def test_harness_argv_construction(tmp_path, monkeypatch):
    """Harness command must contain all required flags, HIP_VISIBLE_DEVICES=device, no --devices, per-fixture home."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "serve_harness.py").write_text("# dummy")
    (repo / "target" / "release").mkdir(parents=True)
    for b in ("daemon", "hipfire"):
        (repo / "target" / "release" / b).write_text("bin")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    fixture = {"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": "abc", "size_bytes": 123}
    harness_cfg = {"battery_prompts": "benchmarks/prompts/hw-gate/serve-battery.json", "max_tokens": 256}
    # create prompt file for repo/battery_prompts
    prompt_path = repo / "benchmarks" / "prompts" / "hw-gate" / "serve-battery.json"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("[]")
    env_base = {"HIPFIRE_HOME": str(home), "HIPFIRE_MODELS_DIR": str(models_dir), "OTHER": "x"}
    captured = {}
    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env", {})
        # create dummy out JSON to satisfy _evaluate_mode
        # find --out
        if "--out" in argv:
            idx = argv.index("--out")
            out_file = Path(argv[idx+1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            # write a clean row
            row = {"assistant_content": "hello world", "expected_substrings": [], "attractor": False, "empty": False, "finish": "stop", "genre": "code", "ctx": 100, "cached": 0, "gen": 10, "ans_words": 2, "prefill_tok_s": 1.0, "decode_tok_s": 2.0}
            out_file.write_text(json.dumps([row]))
        if "--serve-log" in argv:
            idx = argv.index("--serve-log")
            Path(argv[idx+1]).parent.mkdir(parents=True, exist_ok=True)
            Path(argv[idx+1]).write_text("serve log")
        return subprocess.CompletedProcess(argv, 0, stdout="harness stdout line", stderr="")
    monkeypatch.setattr(run_mod, "run_cmd", fake_run)
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    res = run_mod._run_harness_mode(str(repo), fixture, env_base, str(logs_dir), "3", "battery", harness_cfg, str(models_dir))
    argv = captured["argv"]
    env = captured["env"]
    argv_s = " ".join(argv)
    # all flags present
    assert "--model" in argv
    assert "--mode" in argv and "battery" in argv
    assert "--max-tokens" in argv and "256" in argv
    assert "--thinking" in argv and "off" in argv
    assert "--thinking-effort" in argv and "none" in argv
    assert "--max-think-tokens" in argv and "0" in argv
    assert "--home" in argv
    assert "--out" in argv
    assert "--serve-log" in argv
    assert "--prompts-file" in argv
    # no --devices
    assert "--devices" not in argv
    assert "--device" not in argv_s or "--devices" not in argv_s
    # HIP_VISIBLE_DEVICES
    assert env.get("HIP_VISIBLE_DEVICES") == "3"
    assert "HIPFIRE_DAEMON_BIN" in env
    assert "HIPFIRE_CLI_BIN" in env
    assert "HIPFIRE_MODELS_DIR" in env
    # per-fixture home is subdirectory of gate home
    home_idx = argv.index("--home")
    per_home = Path(argv[home_idx+1])
    assert str(per_home).startswith(str(home))
    assert "qwen3.6-27b" in str(per_home) or "qwen3" in str(per_home)
    # also check chain mode has no --prompts-file
    captured.clear()
    res2 = run_mod._run_harness_mode(str(repo), fixture, env_base, str(logs_dir), "3", "chain", harness_cfg, str(models_dir))
    argv2 = captured["argv"]
    assert "--prompts-file" not in argv2
    # ensure argv contains thinking flags for chain as well
    assert "--thinking" in argv2
    # verify logs .out captured
    assert (logs_dir / "qwen3.6-27b-battery.out").is_file()
    content_out = (logs_dir / "qwen3.6-27b-battery.out").read_text()
    assert "harness stdout" in content_out


def test_row_evaluation(tmp_path):
    # clean row passes
    clean = {"assistant_content": "The answer is 42 and hello", "expected_substrings": ["42", "hello"], "attractor": False, "empty": False, "finish": "stop", "genre": "x"}
    enriched = run_mod._enrich_row(clean)
    assert enriched["attractor"] is False
    assert enriched["empty"] is False
    assert enriched["runaway"] is False
    assert enriched["recall_ok"] is True
    status, reason, rows = run_mod._evaluate_mode(0, [clean])
    assert status == "pass"

    # attractor fails
    attractor_row = {"assistant_content": "ok", "expected_substrings": [], "attractor": True, "empty": False, "finish": "stop"}
    status, reason, rows = run_mod._evaluate_mode(0, [attractor_row])
    assert status == "fail"
    assert "attractor" in reason

    # empty fails
    empty_row = {"assistant_content": "", "expected_substrings": [], "attractor": False, "empty": True, "finish": "stop"}
    status, reason, rows = run_mod._evaluate_mode(0, [empty_row])
    assert status == "fail"
    assert "empty" in reason

    # finish=length -> runaway is recorded but not fatal (a coherent answer cut by the cap)
    runaway_row = {"assistant_content": "some text", "expected_substrings": [], "attractor": False, "empty": False, "finish": "length"}
    assert run_mod._enrich_row(runaway_row)["runaway"] is True
    status, reason, rows = run_mod._evaluate_mode(0, [runaway_row])
    assert status == "pass"
    # ...but a loop-to-cap carries the harness's attractor flag and fails
    loop_row = dict(runaway_row, attractor=True)
    status, reason, rows = run_mod._evaluate_mode(0, [loop_row])
    assert status == "fail" and "attractor" in reason

    # recall miss fails
    recall_row = {"assistant_content": "hello world", "expected_substrings": ["missing_token"], "attractor": False, "empty": False, "finish": "stop"}
    assert run_mod._enrich_row(recall_row)["recall_ok"] is False
    status, reason, rows = run_mod._evaluate_mode(0, [recall_row])
    assert status == "fail"
    assert "recall" in reason.lower()

    # case-insensitive recall should pass
    ci_row = {"assistant_content": "Hello World", "expected_substrings": ["hello"], "attractor": False, "empty": False, "finish": "stop"}
    assert run_mod._enrich_row(ci_row)["recall_ok"] is True
    status, _, _ = run_mod._evaluate_mode(0, [ci_row])
    assert status == "pass"

    # harness exit non-zero fails even if rows clean
    status, reason, _ = run_mod._evaluate_mode(1, [clean])
    assert status == "fail"
    assert "exit" in reason

    # missing out JSON with exit 0 should be evaluated as fail closed in _run_harness_mode, but _evaluate_mode alone returns pass for None rows; higher layer handles.


def test_mode_union():
    manifest = {
        "buckets": {
            "load": {"modes": ["battery"]},
            "serve": {"modes": ["battery", "chain"]},
            "kernel": {"modes": ["battery"], "redline": {"model_tag": "qwen3.6:27b", "harness_args": []}},
        }
    }
    assert run_mod._modes_for_buckets(["load"], manifest) == ["battery"]
    assert run_mod._modes_for_buckets(["serve"], manifest) == ["battery", "chain"]
    assert run_mod._modes_for_buckets(["load", "serve"], manifest) == ["battery", "chain"]
    assert run_mod._modes_for_buckets(["kernel"], manifest) == ["battery"]
    assert run_mod._modes_for_buckets(["serve", "kernel"], manifest) == ["battery", "chain"]
    assert run_mod._modes_for_buckets(["load", "kernel"], manifest) == ["battery"]
    assert run_mod._modes_for_buckets(["load", "serve", "kernel"], manifest) == ["battery", "chain"]


def test_render_md_turn_tables_and_details(tmp_path):
    # Build evidence with two modes and rows containing backticks
    assistant = "```python\nprint(1)\n```"
    row0 = {"genre": "code", "finish": "stop", "ctx": 128, "cached": 10, "gen": 20, "ans_words": 5, "prefill_tok_s": 10.0, "decode_tok_s": 20.0, "attractor": False, "empty": False, "runaway": False, "recall_ok": True, "expected_substrings": ["print"], "assistant_content": assistant, "prompt_md5": "abc"}
    row1 = {"genre": "prose", "finish": "length", "ctx": 256, "cached": 0, "gen": 256, "ans_words": 100, "prefill_tok_s": 5.0, "decode_tok_s": 15.0, "attractor": True, "empty": False, "runaway": True, "recall_ok": False, "expected_substrings": ["missing"], "assistant_content": "hello", "prompt_md5": "def"}
    evidence = {
        "schema": "hipfire.hw-gate.evidence", "version": 1, "verdict": "fail", "base": "abc123", "head": "def456", "buckets": ["load", "serve"],
        "host": {"gfx": "gfx1201", "rocm": "6.2", "device": "3", "runner": "hiptrx"},
        "binaries": {"daemon_md5": "d1", "hipfire_md5": "h1", "build_seconds": 42.5},
        "fixtures": [
            {
                "tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": "abc", "sha256_ok": True, "size_ok": True,
                "modes": {
                    "battery": {"exit": 0, "seconds": 19.2, "rows": [row0], "status": "pass", "reason": ""},
                    "chain": {"exit": 1, "seconds": 30.0, "rows": [row1], "status": "fail", "reason": "attractor"},
                },
                "status": "fail", "reason": "chain: attractor",
            }
        ],
        "kernel": {"report": {"pass": True}, "exit": 0, "status": "pass", "reason": ""},
        "logs_dir": "hw-gate-logs",
    }
    md = run_mod.render_md(evidence)
    # header table
    assert "| base |" in md
    assert "abc123" in md
    # per-fixture turn table headers
    assert "| mode | idx | genre | finish |" in md.lower() or "| mode |" in md
    assert "code" in md
    assert "stop" in md
    assert "length" in md
    # flags
    assert "attractor" in md.lower()
    assert "runaway" in md.lower()
    assert "recall" in md.lower()
    # details blocks verbatim
    assert "<details><summary>qwen3.6:27b battery turn 0" in md
    assert "<details><summary>qwen3.6:27b chain turn 0" in md
    # verbatim assistant_content inside fence
    fence = run_mod._fence(assistant)
    assert f"{fence}\n{assistant}\n{fence}" in md
    # kernel section
    assert "## kernel" in md.lower()
    # ensure no old serve top-level section leaked? It's okay if not present, but we check fixtures header present
    assert "## fixtures" in md.lower()


def test_render_md_fence_survives_backticks_in_decoded():
    decoded = "```python\nprint(1)\n```\nand ````four````"
    # build minimal evidence with one fixture, one row containing decoded
    row = {"genre": "x", "finish": "stop", "ctx": 0, "cached": 0, "gen": 0, "ans_words": 0, "prefill_tok_s": 0, "decode_tok_s": 0, "attractor": False, "empty": False, "runaway": False, "recall_ok": True, "expected_substrings": [], "assistant_content": decoded, "prompt_md5": ""}
    evidence = {"verdict": "pass", "base": "a", "head": "b", "buckets": ["load"], "host": {}, "binaries": {},
                "fixtures": [{"tag": "t", "sha256_ok": True, "size_ok": True, "modes": {"battery": {"exit": 0, "seconds": 1.0, "rows": [row], "status": "pass", "reason": ""}}, "status": "pass", "reason": ""}],
                "kernel": None, "logs_dir": "x"}
    md = run_mod.render_md(evidence)
    fence = run_mod._fence(decoded)
    assert fence == "`````"
    assert f"{fence}\n{decoded}\n{fence}" in md


def _make_repo_with_harness(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    bin_dir = repo / "target" / "release"
    bin_dir.mkdir(parents=True)
    (bin_dir / "daemon").write_text("dummy")
    (bin_dir / "hipfire").write_text("dummy")
    (bin_dir / "daemon").chmod(0o755)
    (bin_dir / "hipfire").chmod(0o755)
    scripts = repo / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "serve_harness.py").write_text("# dummy")
    (scripts / "redline_daemon_harness.py").write_text("# dummy")
    return repo


def test_main_exit_2_missing_fixture_no_build(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    repo = _make_repo_with_harness(tmp_path)
    fixtures_data = {
        "schema": "hipfire.hw-gate.fixtures",
        "version": 2,
        "models_dir": str(models_dir),
        "harness": {"battery_prompts": "benchmarks/prompts/hw-gate/serve-battery.json", "max_tokens": 256},
        "fixtures": [
            {"tag": "qwen3.6:27b", "file": "missing.mq4", "sha256": "a"*64, "size_bytes": 123, "arch_id": 5, "why": "x"}
        ],
        "buckets": {
            "load": {"modes": ["battery"]},
            "serve": {"modes": ["battery", "chain"]},
            "kernel": {"modes": ["battery"], "redline": {"model_tag": "qwen3.6:27b", "harness_args": ["--pm4"]}}
        }
    }
    fixtures_path = tmp_path / "scripts" / "hw-gate" / "fixtures.json"
    fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    _prompts = tmp_path / "benchmarks" / "prompts" / "hw-gate"
    _prompts.mkdir(parents=True, exist_ok=True)
    (_prompts / "serve-battery.json").write_text("[]")
    fixtures_path.write_text(json.dumps(fixtures_data))
    out = tmp_path / "hw-gate.json"
    md = tmp_path / "hw-gate.md"
    monkeypatch.setenv("HIPFIRE_HOME", str(home))
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(models_dir))
    calls = []
    orig = run_mod.run_cmd
    def tracking_run(argv, **kwargs):
        calls.append(argv)
        if argv and argv[0] == "cargo":
            pytest.fail("cargo build should not be attempted on missing fixture")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    run_mod.run_cmd = tracking_run
    try:
        rc = run_mod.main(["--repo", str(repo), "--fixtures", str(fixtures_path), "--base", "abc", "--head", "def", "--buckets", "load", "--device", "3", "--out", str(out), "--md", str(md)])
    finally:
        run_mod.run_cmd = orig
    assert rc == 2
    assert all("cargo" not in " ".join(c) if isinstance(c, list) else "cargo" not in str(c) for c in calls)
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["verdict"] == "fail"


def test_main_harness_success_and_failure(tmp_path, monkeypatch):
    # Test main exit 0 when harness passes, exit 1 when recall miss
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # create fixture file
    content = b"modeldata"
    sha = hashlib.sha256(content).hexdigest()
    size = len(content)
    fpath = models_dir / "qwen3.6-27b.mq4"
    fpath.write_bytes(content)
    home = tmp_path / "home"
    home.mkdir()
    repo = _make_repo_with_harness(tmp_path)
    # create harness battery_prompts file
    prompt_file = repo / "benchmarks" / "prompts" / "hw-gate" / "serve-battery.json"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text("[]")
    fixtures_data = {
        "schema": "hipfire.hw-gate.fixtures",
        "version": 2,
        "models_dir": str(models_dir),
        "harness": {"battery_prompts": "benchmarks/prompts/hw-gate/serve-battery.json", "max_tokens": 256},
        "fixtures": [
            {"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": sha, "size_bytes": size, "arch_id": 5, "why": "x"}
        ],
        "buckets": {
            "load": {"modes": ["battery"]},
            "serve": {"modes": ["battery", "chain"]},
            "kernel": {"modes": ["battery"], "redline": {"model_tag": "qwen3.6:27b", "harness_args": []}}
        }
    }
    fixtures_path = tmp_path / "scripts" / "hw-gate" / "fixtures.json"
    fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    _prompts = tmp_path / "benchmarks" / "prompts" / "hw-gate"
    _prompts.mkdir(parents=True, exist_ok=True)
    (_prompts / "serve-battery.json").write_text("[]")
    fixtures_path.write_text(json.dumps(fixtures_data))
    out = tmp_path / "hw-gate.json"
    md = tmp_path / "hw-gate.md"
    monkeypatch.setenv("HIPFIRE_HOME", str(home))
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(models_dir))

    def fake_harness_pass(argv, **kwargs):
        if argv[0] == "cargo":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if "serve_harness.py" in " ".join(argv):
            # find --out and write passing rows
            if "--out" in argv:
                idx = argv.index("--out")
                out_p = Path(argv[idx+1])
                out_p.parent.mkdir(parents=True, exist_ok=True)
                row = {"assistant_content": "hello contains token", "expected_substrings": ["token"], "attractor": False, "empty": False, "finish": "stop", "genre": "code", "ctx": 10, "cached": 0, "gen": 10, "ans_words": 3, "prefill_tok_s": 1.0, "decode_tok_s": 2.0}
                out_p.write_text(json.dumps([row]))
            if "--serve-log" in argv:
                idx = argv.index("--serve-log")
                Path(argv[idx+1]).parent.mkdir(parents=True, exist_ok=True)
                Path(argv[idx+1]).write_text("log")
            return subprocess.CompletedProcess(argv, 0, stdout="turn 0 ok", stderr="")
        if "redline_daemon_harness.py" in " ".join(argv):
            if "--out" in argv:
                idx = argv.index("--out")
                Path(argv[idx+1]).parent.mkdir(parents=True, exist_ok=True)
                Path(argv[idx+1]).write_text(json.dumps({"pass": True}))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="unknown gfx1201", stderr="")

    orig = run_mod.run_cmd
    run_mod.run_cmd = fake_harness_pass
    try:
        rc = run_mod.main(["--repo", str(repo), "--fixtures", str(fixtures_path), "--base", "a", "--head", "b", "--buckets", "load", "--device", "3", "--out", str(out), "--md", str(md), "--skip-build"])
    finally:
        run_mod.run_cmd = orig
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["verdict"] == "pass"
    # fixtures should have new shape, not old
    assert "modes" in data["fixtures"][0]
    assert "battery" in data["fixtures"][0]["modes"]
    assert data["fixtures"][0]["modes"]["battery"]["status"] == "pass"
    # top-level serve key removed
    assert "serve" not in data or data.get("serve") is None

    # now test recall miss -> exit 1
    def fake_harness_recall_miss(argv, **kwargs):
        if argv[0] == "cargo":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if "serve_harness.py" in " ".join(argv):
            if "--out" in argv:
                idx = argv.index("--out")
                out_p = Path(argv[idx+1])
                out_p.parent.mkdir(parents=True, exist_ok=True)
                row = {"assistant_content": "hello world", "expected_substrings": ["missing_token"], "attractor": False, "empty": False, "finish": "stop", "genre": "code", "ctx": 10, "cached": 0, "gen": 10, "ans_words": 2, "prefill_tok_s": 1.0, "decode_tok_s": 2.0}
                out_p.write_text(json.dumps([row]))
            if "--serve-log" in argv:
                idx = argv.index("--serve-log")
                Path(argv[idx+1]).parent.mkdir(parents=True, exist_ok=True)
                Path(argv[idx+1]).write_text("log")
            return subprocess.CompletedProcess(argv, 0, stdout="turn", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="unknown", stderr="")

    run_mod.run_cmd = fake_harness_recall_miss
    out2 = tmp_path / "hw-gate2.json"
    md2 = tmp_path / "hw-gate2.md"
    try:
        rc2 = run_mod.main(["--repo", str(repo), "--fixtures", str(fixtures_path), "--base", "a", "--head", "b", "--buckets", "load", "--device", "3", "--out", str(out2), "--md", str(md2), "--skip-build"])
    finally:
        run_mod.run_cmd = orig
    assert rc2 == 1
    data2 = json.loads(out2.read_text())
    assert data2["verdict"] == "fail"
    assert data2["fixtures"][0]["status"] == "fail"


def test_kernel_redline_config_source(tmp_path, monkeypatch):
    # Verify run_kernel reads buckets.kernel.redline.harness_args
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    content = b"data"
    sha = hashlib.sha256(content).hexdigest()
    (models_dir / "qwen3.6-27b.mq4").write_bytes(content)
    repo = _make_repo_with_harness(tmp_path)
    kernel_cfg = {"model_tag": "qwen3.6:27b", "harness_args": ["--pm4", "--capture-repeats", "2"]}
    # manifest with fixtures list
    manifest = {"fixtures": [{"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": sha, "size_bytes": len(content)}], "buckets": {"kernel": {"modes": ["battery"], "redline": kernel_cfg}}}
    captured = {}
    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        if "--out" in argv:
            idx = argv.index("--out")
            Path(argv[idx+1]).parent.mkdir(parents=True, exist_ok=True)
            Path(argv[idx+1]).write_text(json.dumps({"pass": True}))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    monkeypatch.setattr(run_mod, "run_cmd", fake_run)
    res = run_mod.run_kernel(str(repo), str(models_dir), kernel_cfg, manifest, {}, str(tmp_path / "logs"))
    assert res["status"] == "pass"
    assert "--pm4" in captured["argv"]

def test_routes_union_and_mode_selection():
    routes = [
        {"mode": "battery", "tag": "qwen3.6:27b", "source": "bucket", "why": "x"},
        {"mode": "chain", "tag": "qwen3.6:27b", "source": "sol", "why": "y"},
        {"mode": "battery", "tag": "qwen3.6:27b", "source": "author", "why": "z"},
        {"mode": "battery", "tag": "ornith-1.5:35b-a3b-mq4r", "source": "author", "why": "a"},
    ]
    order, per_modes, per_source = run_mod._build_routes_map(routes)
    assert order == ["qwen3.6:27b", "ornith-1.5:35b-a3b-mq4r"]
    assert per_modes["qwen3.6:27b"] == ["battery", "chain"]
    assert per_modes["ornith-1.5:35b-a3b-mq4r"] == ["battery"]
    # bucket priority over author/sol
    assert per_source["qwen3.6:27b"] == "bucket"
    assert per_source["ornith-1.5:35b-a3b-mq4r"] == "author"


def test_registry_resolution_alias(tmp_path):
    registry = {
        "models": {
            "qwen3.8:27b": {"file": "qwen3.8-27b.mq4", "sha256": "a"*64, "size_bytes": 123, "arch_id": 5},
            "qwen3.8:27b-mq4-xt": {"file": "qwen3.8-27b.mq4-xt", "sha256": "b"*64, "size_bytes": 456, "arch_id": 5},
        },
        "aliases": {
            "qwen3.8:fast": "qwen3.8:27b-mq4-xt",
            "qwen3.8": "qwen3.8:27b",
        }
    }
    r = run_mod._resolve_registry_entry("qwen3.8:fast", registry)
    assert r is not None and r["file"] == "qwen3.8-27b.mq4-xt" and r["tag"] == "qwen3.8:fast"
    r2 = run_mod._resolve_registry_entry("qwen3.8", registry)
    assert r2 is not None and r2["file"] == "qwen3.8-27b.mq4"
    r3 = run_mod._resolve_registry_entry("unknown:tag", registry)
    assert r3 is None
    # direct without alias
    r4 = run_mod._resolve_registry_entry("qwen3.8:27b", registry)
    assert r4 is not None and r4["file"] == "qwen3.8-27b.mq4"


def test_routes_unknown_tag_unavailable(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    content = b"modeldata"
    sha = hashlib.sha256(content).hexdigest()
    (models_dir / "qwen3.6-27b.mq4").write_bytes(content)
    home = tmp_path / "home"
    home.mkdir()
    repo = _make_repo_with_harness(tmp_path)
    prompt_file = repo / "benchmarks" / "prompts" / "hw-gate" / "serve-battery.json"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text("[]")
    fixtures_data = {
        "schema": "hipfire.hw-gate.fixtures", "version": 2, "models_dir": str(models_dir),
        "harness": {"battery_prompts": "benchmarks/prompts/hw-gate/serve-battery.json", "max_tokens": 256},
        "fixtures": [{"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": sha, "size_bytes": len(content), "arch_id": 5}],
        "buckets": {"load": {"modes": ["battery"]}, "serve": {"modes": ["battery", "chain"]}, "kernel": {"modes": ["battery"], "redline": {"model_tag": "qwen3.6:27b", "harness_args": []}}}
    }
    fixtures_path = tmp_path / "scripts" / "hw-gate" / "fixtures.json"
    fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    _prompts = tmp_path / "benchmarks" / "prompts" / "hw-gate"
    _prompts.mkdir(parents=True, exist_ok=True)
    (_prompts / "serve-battery.json").write_text("[]")
    fixtures_path.write_text(json.dumps(fixtures_data))
    registry_data = {"models": {}, "aliases": {}}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry_data))
    routes = [
        {"mode": "battery", "tag": "qwen3.6:27b", "source": "bucket", "why": "mandatory"},
        {"mode": "battery", "tag": "unknown:tag", "source": "sol", "why": "sol request"},
    ]
    routes_path = tmp_path / "routes.json"
    routes_path.write_text(json.dumps(routes))
    out = tmp_path / "hw-gate.json"
    md = tmp_path / "hw-gate.md"
    monkeypatch.setenv("HIPFIRE_HOME", str(home))
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(models_dir))
    def fake_pass(argv, **kwargs):
        if argv[0] == "cargo":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if "serve_harness.py" in " ".join(argv):
            if "--out" in argv:
                out_p = Path(argv[argv.index("--out")+1])
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps([{"assistant_content": "hello", "expected_substrings": [], "attractor": False, "empty": False, "finish": "stop", "genre": "x", "ctx": 10, "cached": 0, "gen": 10, "ans_words": 1, "prefill_tok_s": 1.0, "decode_tok_s": 1.0}]))
            if "--serve-log" in argv:
                Path(argv[argv.index("--serve-log")+1]).parent.mkdir(parents=True, exist_ok=True)
                Path(argv[argv.index("--serve-log")+1]).write_text("log")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="unknown", stderr="")
    monkeypatch.setattr(run_mod, "run_cmd", fake_pass)
    rc = run_mod.main(["--repo", str(repo), "--fixtures", str(fixtures_path), "--base", "a", "--head", "b", "--buckets", "load", "--device", "3", "--out", str(out), "--md", str(md), "--routes", str(routes_path), "--registry", str(registry_path), "--skip-build"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["verdict"] == "pass"
    # unknown tag should be unavailable
    unknown = next(f for f in data["fixtures"] if f["tag"] == "unknown:tag")
    assert unknown["status"] == "unavailable"
    assert unknown["reason"] == "unknown tag"
    assert unknown["modes"] == {}
    # mandatory should be pass
    mandatory = next(f for f in data["fixtures"] if f["tag"] == "qwen3.6:27b")
    assert mandatory["status"] == "pass"
    assert mandatory["source"] == "bucket"
    assert unknown["source"] == "sol"


def test_routes_absent_file_unavailable_still_pass(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    content = b"presentdata"
    sha_present = hashlib.sha256(content).hexdigest()
    (models_dir / "qwen3.6-27b.mq4").write_bytes(content)
    home = tmp_path / "home"
    home.mkdir()
    repo = _make_repo_with_harness(tmp_path)
    prompt_file = repo / "benchmarks" / "prompts" / "hw-gate" / "serve-battery.json"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text("[]")
    fixtures_data = {
        "schema": "hipfire.hw-gate.fixtures", "version": 2, "models_dir": str(models_dir),
        "harness": {"battery_prompts": "benchmarks/prompts/hw-gate/serve-battery.json", "max_tokens": 256},
        "fixtures": [{"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": sha_present, "size_bytes": len(content), "arch_id": 5}],
        "buckets": {"load": {"modes": ["battery"]}, "serve": {"modes": ["battery", "chain"]}, "kernel": {"modes": ["battery"], "redline": {"model_tag": "qwen3.6:27b", "harness_args": []}}}
    }
    fixtures_path = tmp_path / "scripts" / "hw-gate" / "fixtures.json"
    fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    _prompts = tmp_path / "benchmarks" / "prompts" / "hw-gate"
    _prompts.mkdir(parents=True, exist_ok=True)
    (_prompts / "serve-battery.json").write_text("[]")
    fixtures_path.write_text(json.dumps(fixtures_data))
    # registry with an extra model whose file is NOT present
    reg_content = b"registrydata"
    reg_sha = hashlib.sha256(reg_content).hexdigest()
    registry_data = {
        "models": {
            "qwen3.8:27b": {"file": "qwen3.8-27b.mq4", "sha256": reg_sha, "size_bytes": len(reg_content), "arch_id": 5},
        },
        "aliases": {}
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry_data))
    routes = [
        {"mode": "battery", "tag": "qwen3.6:27b", "source": "bucket", "why": "mandatory"},
        {"mode": "chain", "tag": "qwen3.8:27b", "source": "author", "why": "author request"},
    ]
    routes_path = tmp_path / "routes.json"
    routes_path.write_text(json.dumps(routes))
    out = tmp_path / "hw-gate.json"
    md = tmp_path / "hw-gate.md"
    monkeypatch.setenv("HIPFIRE_HOME", str(home))
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(models_dir))
    def fake_pass(argv, **kwargs):
        if argv[0] == "cargo":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if "serve_harness.py" in " ".join(argv):
            if "--out" in argv:
                out_p = Path(argv[argv.index("--out")+1])
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps([{"assistant_content": "ok", "expected_substrings": [], "attractor": False, "empty": False, "finish": "stop", "genre": "x", "ctx": 10, "cached": 0, "gen": 10, "ans_words": 1, "prefill_tok_s": 1.0, "decode_tok_s": 1.0}]))
            if "--serve-log" in argv:
                Path(argv[argv.index("--serve-log")+1]).parent.mkdir(parents=True, exist_ok=True)
                Path(argv[argv.index("--serve-log")+1]).write_text("log")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="unknown", stderr="")
    monkeypatch.setattr(run_mod, "run_cmd", fake_pass)
    rc = run_mod.main(["--repo", str(repo), "--fixtures", str(fixtures_path), "--base", "a", "--head", "b", "--buckets", "load", "--device", "3", "--out", str(out), "--md", str(md), "--routes", str(routes_path), "--registry", str(registry_path), "--skip-build"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["verdict"] == "pass"
    routed_missing = next(f for f in data["fixtures"] if f["tag"] == "qwen3.8:27b")
    assert routed_missing["status"] == "unavailable"
    assert "fixture not present on runner" in routed_missing["reason"]
    assert "qwen3.8-27b.mq4" in routed_missing["reason"]
    # ensure it did not cause overall fail
    assert routed_missing["source"] == "author"
    # modes should be empty for unavailable
    assert routed_missing["modes"] == {}
    # md should contain unavailable row and not contain turn table for that tag's chain
    md_text = md.read_text()
    assert "qwen3.8:27b" in md_text
    assert "fixture not present on runner" in md_text
    # there should be no turn table rows for the unavailable fixture's mode
    # check that the unavailable fixture section does not contain "| mode | idx |"
    # we look after the header for that tag until next header


def test_routes_mismatched_mandatory_still_exit2(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    content = b"gooddata"
    bad_content = b"baddata12345"
    # write bad file content but fixtures expects good sha
    (models_dir / "qwen3.6-27b.mq4").write_bytes(bad_content)
    sha_good = hashlib.sha256(content).hexdigest()
    home = tmp_path / "home"
    home.mkdir()
    repo = _make_repo_with_harness(tmp_path)
    prompt_file = repo / "benchmarks" / "prompts" / "hw-gate" / "serve-battery.json"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text("[]")
    fixtures_data = {
        "schema": "hipfire.hw-gate.fixtures", "version": 2, "models_dir": str(models_dir),
        "harness": {"battery_prompts": "benchmarks/prompts/hw-gate/serve-battery.json", "max_tokens": 256},
        "fixtures": [{"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": sha_good, "size_bytes": len(content), "arch_id": 5}],
        "buckets": {"load": {"modes": ["battery"]}, "serve": {"modes": ["battery", "chain"]}, "kernel": {"modes": ["battery"], "redline": {"model_tag": "qwen3.6:27b", "harness_args": []}}}
    }
    fixtures_path = tmp_path / "scripts" / "hw-gate" / "fixtures.json"
    fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    _prompts = tmp_path / "benchmarks" / "prompts" / "hw-gate"
    _prompts.mkdir(parents=True, exist_ok=True)
    (_prompts / "serve-battery.json").write_text("[]")
    fixtures_path.write_text(json.dumps(fixtures_data))
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"models": {}, "aliases": {}}))
    routes = [{"mode": "battery", "tag": "qwen3.6:27b", "source": "bucket", "why": "mandatory"}]
    routes_path = tmp_path / "routes.json"
    routes_path.write_text(json.dumps(routes))
    out = tmp_path / "hw-gate.json"
    md = tmp_path / "hw-gate.md"
    monkeypatch.setenv("HIPFIRE_HOME", str(home))
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(models_dir))
    monkeypatch.setattr(run_mod, "run_cmd", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""))
    rc = run_mod.main(["--repo", str(repo), "--fixtures", str(fixtures_path), "--base", "a", "--head", "b", "--buckets", "load", "--device", "3", "--out", str(out), "--md", str(md), "--routes", str(routes_path), "--registry", str(registry_path), "--skip-build"])
    assert rc == 2
    data = json.loads(out.read_text())
    assert data["verdict"] == "fail"
    assert "precondition_error" in data


def test_render_md_unavailable_rows(tmp_path):
    evidence = {
        "schema": "hipfire.hw-gate.evidence", "version": 1, "verdict": "pass", "base": "a", "head": "b", "buckets": ["load"],
        "host": {"gfx": "gfx1201", "rocm": "6.2", "device": "3", "runner": "hiptrx"},
        "binaries": {"daemon_md5": "d1", "hipfire_md5": "h1", "build_seconds": 42.5},
        "fixtures": [
            {"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": "abc", "sha256_ok": True, "size_ok": True, "source": "bucket", "modes": {"battery": {"exit": 0, "seconds": 1.0, "rows": [{"genre": "x", "finish": "stop", "ctx": 10, "cached": 0, "gen": 10, "ans_words": 1, "prefill_tok_s": 1.0, "decode_tok_s": 1.0, "attractor": False, "empty": False, "runaway": False, "recall_ok": True, "expected_substrings": [], "assistant_content": "ok", "prompt_md5": ""}], "status": "pass", "reason": ""}}, "status": "pass", "reason": ""},
            {"tag": "qwen3.8:27b", "file": "qwen3.8-27b.mq4", "sha256": "def", "sha256_ok": False, "size_ok": False, "source": "author", "modes": {}, "status": "unavailable", "reason": "fixture not present on runner: qwen3.8-27b.mq4"},
            {"tag": "unknown:tag", "file": "", "sha256": "", "sha256_ok": False, "size_ok": False, "source": "sol", "modes": {}, "status": "unavailable", "reason": "unknown tag"},
        ],
        "kernel": None, "logs_dir": "hw-gate-logs",
    }
    md = run_mod.render_md(evidence)
    assert "qwen3.8:27b" in md
    assert "fixture not present on runner" in md
    assert "unknown tag" in md
    # unavailable should not have turn tables
    # ensure that the number of "| mode | idx |" tables equals number of run fixtures (1)
    # Count turn tables occurrences
    count = md.lower().count("| mode | idx |")
    # only the first fixture has battery mode with rows, so count ==1
    assert count == 1
    # source should appear
    assert "source:" in md.lower() or "bucket" in md


def test_routes_modes_per_fixture(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # two fixtures both present
    c1 = b"data1"
    c2 = b"data2!!"
    sha1 = hashlib.sha256(c1).hexdigest()
    sha2 = hashlib.sha256(c2).hexdigest()
    (models_dir / "qwen3.6-27b.mq4").write_bytes(c1)
    (models_dir / "qwen3.8-27b.mq4").write_bytes(c2)
    home = tmp_path / "home"
    home.mkdir()
    repo = _make_repo_with_harness(tmp_path)
    prompt_file = repo / "benchmarks" / "prompts" / "hw-gate" / "serve-battery.json"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text("[]")
    fixtures_data = {
        "schema": "hipfire.hw-gate.fixtures", "version": 2, "models_dir": str(models_dir),
        "harness": {"battery_prompts": "benchmarks/prompts/hw-gate/serve-battery.json", "max_tokens": 256},
        "fixtures": [{"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": sha1, "size_bytes": len(c1), "arch_id": 5}],
        "buckets": {"load": {"modes": ["battery"]}, "serve": {"modes": ["battery", "chain"]}, "kernel": {"modes": ["battery"], "redline": {"model_tag": "qwen3.6:27b", "harness_args": []}}}
    }
    fixtures_path = tmp_path / "scripts" / "hw-gate" / "fixtures.json"
    fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    _prompts = tmp_path / "benchmarks" / "prompts" / "hw-gate"
    _prompts.mkdir(parents=True, exist_ok=True)
    (_prompts / "serve-battery.json").write_text("[]")
    fixtures_path.write_text(json.dumps(fixtures_data))
    registry_data = {"models": {"qwen3.8:27b": {"file": "qwen3.8-27b.mq4", "sha256": sha2, "size_bytes": len(c2), "arch_id": 5}}, "aliases": {}}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry_data))
    routes = [
        {"mode": "battery", "tag": "qwen3.6:27b", "source": "bucket", "why": "mandatory"},
        {"mode": "battery", "tag": "qwen3.8:27b", "source": "author", "why": "x"},
        {"mode": "chain", "tag": "qwen3.8:27b", "source": "author", "why": "x"},
    ]
    routes_path = tmp_path / "routes.json"
    routes_path.write_text(json.dumps(routes))
    out = tmp_path / "hw-gate.json"
    md = tmp_path / "hw-gate.md"
    monkeypatch.setenv("HIPFIRE_HOME", str(home))
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(models_dir))
    captured_modes = {}
    def fake_harness(argv, **kwargs):
        if argv[0] == "cargo":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if "serve_harness.py" in " ".join(argv):
            # extract tag via file name in argv
            model_idx = argv.index("--model")
            model_path = argv[model_idx+1]
            tag = "unknown"
            if "qwen3.6" in model_path:
                tag = "qwen3.6:27b"
            elif "qwen3.8" in model_path:
                tag = "qwen3.8:27b"
            mode_idx = argv.index("--mode")
            mode = argv[mode_idx+1]
            captured_modes.setdefault(tag, []).append(mode)
            if "--out" in argv:
                out_p = Path(argv[argv.index("--out")+1])
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps([{"assistant_content": "ok", "expected_substrings": [], "attractor": False, "empty": False, "finish": "stop", "genre": "x", "ctx": 10, "cached": 0, "gen": 10, "ans_words": 1, "prefill_tok_s": 1.0, "decode_tok_s": 1.0}]))
            if "--serve-log" in argv:
                Path(argv[argv.index("--serve-log")+1]).parent.mkdir(parents=True, exist_ok=True)
                Path(argv[argv.index("--serve-log")+1]).write_text("log")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="unknown", stderr="")
    monkeypatch.setattr(run_mod, "run_cmd", fake_harness)
    rc = run_mod.main(["--repo", str(repo), "--fixtures", str(fixtures_path), "--base", "a", "--head", "b", "--buckets", "load", "--device", "3", "--out", str(out), "--md", str(md), "--routes", str(routes_path), "--registry", str(registry_path), "--skip-build"])
    assert rc == 0
    data = json.loads(out.read_text())
    # qwen3.6 should have only battery
    q1 = next(f for f in data["fixtures"] if f["tag"] == "qwen3.6:27b")
    assert set(q1["modes"].keys()) == {"battery"}
    # qwen3.8 should have battery and chain
    q2 = next(f for f in data["fixtures"] if f["tag"] == "qwen3.8:27b")
    assert set(q2["modes"].keys()) == {"battery", "chain"}
    # ensure harness called with correct modes
    assert "battery" in captured_modes["qwen3.6:27b"] and "chain" not in captured_modes["qwen3.6:27b"]
    assert set(captured_modes["qwen3.8:27b"]) == {"battery", "chain"}


def test_battery_prompts_resolve_against_gate_root_not_pr(tmp_path, monkeypatch):
    """Prompts are gate policy: resolved from the fixtures.json tree, never the PR checkout."""
    gate = tmp_path / "gate"; (gate / "scripts" / "hw-gate").mkdir(parents=True)
    (gate / "benchmarks" / "prompts" / "hw-gate").mkdir(parents=True)
    (gate / "benchmarks" / "prompts" / "hw-gate" / "serve-battery.json").write_text("[]")
    pr = tmp_path / "pr"; (pr / "scripts").mkdir(parents=True)
    (pr / "scripts" / "serve_harness.py").write_text("")
    seen = {}
    def fake_run_cmd(argv, **kw):
        seen["argv"] = argv
        class R: returncode, stdout, stderr = 0, "", ""
        return R()
    monkeypatch.setattr(run_mod, "run_cmd", fake_run_cmd)
    (tmp_path / "out.json").write_text("[]")
    cfg = {"battery_prompts": "benchmarks/prompts/hw-gate/serve-battery.json", "max_tokens": 8, "_gate_root": str(gate)}
    env = {"HIPFIRE_HOME": str(tmp_path / "home")}
    run_mod._run_harness_mode(str(pr), {"tag": "t:x", "file": "x.mq4"}, env, str(tmp_path / "logs"), "0", "battery", cfg, str(tmp_path))
    i = seen["argv"].index("--prompts-file")
    assert seen["argv"][i + 1] == str(gate / "benchmarks" / "prompts" / "hw-gate" / "serve-battery.json")
    # and a PR-less gate root without the file fails closed instead of running a partial battery
    cfg["_gate_root"] = str(pr)
    res = run_mod._run_harness_mode(str(pr), {"tag": "t:x", "file": "x.mq4"}, env, str(tmp_path / "logs"), "0", "battery", cfg, str(tmp_path))
    assert res["status"] == "fail" and "gate policy" in res["reason"]
