# Copyright (c) Kaden Schutt
import json
import sys
import time

from autoresearch.ar.gate import agent


def test_probe_stops_process_group_as_soon_as_verdict_is_written(tmp_path, monkeypatch):
    verdict = tmp_path / "verdict.json"
    child = (
        "import json,time; "
        f"json.dump({{'passed': True, 'detail': 'done'}}, open({str(verdict)!r}, 'w')); "
        "time.sleep(60)"
    )
    real_popen = agent.subprocess.Popen

    def popen(_argv, **kwargs):
        return real_popen([sys.executable, "-c", child], **kwargs)

    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    started = time.monotonic()
    rc = agent.run_codex_probe(
        harness="codex", model="test", effort="high", prompt="probe", cwd=str(tmp_path),
        verdict_path=str(verdict), timeout_seconds=5, poll_seconds=0.01,
    )
    assert rc == 0
    assert time.monotonic() - started < 2
    assert json.loads(verdict.read_text())["passed"] is True


def test_probe_routes_agent_output_away_from_machine_stdout(tmp_path, monkeypatch):
    verdict = tmp_path / "verdict.json"
    child = (
        "import json; print('agent chatter'); "
        f"json.dump({{'passed': True, 'detail': 'done'}}, open({str(verdict)!r}, 'w'))"
    )
    real_popen = agent.subprocess.Popen
    seen = {}

    def popen(_argv, **kwargs):
        seen.update(kwargs)
        return real_popen([sys.executable, "-c", child], **kwargs)

    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    rc = agent.run_codex_probe(
        harness="codex", model="test", effort="high", prompt="probe", cwd=str(tmp_path),
        verdict_path=str(verdict), timeout_seconds=5, poll_seconds=0.01,
    )
    assert rc == 0
    assert seen["stdout"] is sys.stderr
    assert seen["stderr"] is sys.stderr


def test_probe_timeout_kills_process_and_returns_124(tmp_path, monkeypatch):
    real_popen = agent.subprocess.Popen

    def popen(_argv, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)

    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    rc = agent.run_codex_probe(
        harness="codex", model="test", effort="high", prompt="probe", cwd=str(tmp_path),
        verdict_path=str(tmp_path / "missing.json"), timeout_seconds=0.05, poll_seconds=0.01,
    )
    assert rc == 124


def test_probe_removes_stale_verdict_before_starting(tmp_path, monkeypatch):
    verdict = tmp_path / "verdict.json"
    verdict.write_text('{"passed": true, "detail": "stale"}')
    real_popen = agent.subprocess.Popen

    def popen(_argv, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)

    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    rc = agent.run_codex_probe(
        harness="codex", model="test", effort="high", prompt="probe", cwd=str(tmp_path),
        verdict_path=str(verdict), timeout_seconds=0.05, poll_seconds=0.01,
    )
    assert rc == 124
    assert not verdict.exists()


def test_probe_rejects_non_codex_harness(tmp_path):
    try:
        agent.run_codex_probe(
            harness="grok", model="test", effort="high", prompt="probe", cwd=str(tmp_path),
            verdict_path=str(tmp_path / "verdict.json"), timeout_seconds=1,
        )
    except ValueError as exc:
        assert "require harness='codex'" in str(exc)
    else:
        raise AssertionError("non-Codex gate probe must fail closed")
