# Copyright (c) Kaden Schutt
"""No-GPU WIRING tests for LiveServeRunner (Task 2.3).

The live GPU path is exercised in Phase 5; here we only assert the adapter WIRES correctly — the
module imports without a GPU/ROCm, satisfies the ServeRunner contract, and perf_measure returns the
(tok_list, dur_list) shape the orchestrator consumes (mocking the rocprof seam). No daemon spawns.
"""
import autoresearch.ar.certify.serve_runner as sr
from autoresearch.ar.certify.orchestrator import ServeRunner


def test_module_imports_no_gpu():
    # importing the adapter must not require ROCm / a GPU / serve_harness at import time
    assert hasattr(sr, "LiveServeRunner")
    assert hasattr(sr, "_run_rocprof")


def test_is_a_serve_runner():
    assert issubclass(sr.LiveServeRunner, ServeRunner)


def test_perf_measure_returns_tok_and_dur(monkeypatch):
    # _run_rocprof returns (dur, tok); perf_measure must hand the orchestrator (tok, dur)
    monkeypatch.setattr(sr, "_run_rocprof", lambda *a, **k: ([9.0, 9.1, 8.9], [160, 161, 159]))
    r = sr.LiveServeRunner(model="m", arch="gfx1201", dev=0)
    tok_list, dur_list = r.perf_measure("some_daemon_bin")
    assert len(tok_list) == 3 and len(dur_list) == 3
    assert tok_list == [160, 161, 159] and dur_list == [9.0, 9.1, 8.9]


def test_perf_measure_empty_when_rocprof_empty(monkeypatch):
    monkeypatch.setattr(sr, "_run_rocprof", lambda *a, **k: ([], []))
    r = sr.LiveServeRunner(model="m", arch="gfx1201", dev=0, kernel="k")
    tok_list, dur_list = r.perf_measure("d")
    assert tok_list == [] and dur_list == []


def test_clocks_pinned_returns_empty():
    # profile_standard pins the clock, so a separate clock-VOID stream is unnecessary
    r = sr.LiveServeRunner(model="m", arch="gfx1201", dev=0)
    assert r.clocks("d") == []


def test_parity_gens_builds_greedy_request(monkeypatch):
    # the parity arm shells the RAW daemon (voodoo allowed here); capture the request it would send
    captured = {}

    def fake_run(cmd, input=None, **kw):
        captured["input"] = input

        class R:
            stdout = ""
        return R()

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    r = sr.LiveServeRunner(model="qwen.mq4r", arch="gfx1201", dev=0, kv="q8")
    gens = r.parity_gens("daemon_bin")
    req = captured["input"]
    assert '"type":"load"' in req and '"type":"generate"' in req and '"temperature":0.0' in req
    assert '"kv_mode":"q8"' in req
    # one gen dict per parity prompt, each carrying prompt_id + token_ids
    assert len(gens) == len(sr.LiveServeRunner.PARITY_PROMPTS)
    assert all("prompt_id" in g and "token_ids" in g for g in gens)
