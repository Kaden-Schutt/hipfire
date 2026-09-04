# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hw_gate_merge", str(HERE.parent / "merge_evidence.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _lane(tmp, name, gfx, verdict, tags, kernel=None):
    d = tmp / name
    d.mkdir()
    ev = {"verdict": verdict, "base": "b", "head": "h", "buckets": ["load"], "host": {"gfx": gfx},
          "binaries": {"daemon_md5": name}, "fixtures": [{"tag": t, "status": "pass"} for t in tags], "kernel": kernel}
    (d / "hw-gate.json").write_text(json.dumps(ev))
    (d / "hw-gate.md").write_text(f"# {name} md\n")
    return d


def test_all_lanes_pass(tmp_path):
    a = _lane(tmp_path, "hiptrx", "gfx1201", "pass", ["x", "y"])
    b = _lane(tmp_path, "hipx", "gfx1100", "pass", ["x"], kernel={"status": "pass"})
    merged, md = mod.merge([("hiptrx", a), ("hipx", b)])
    assert merged["verdict"] == "pass"
    assert merged["lanes_missing"] == []
    assert [f["host_gfx"] for f in merged["fixtures"]] == ["gfx1201", "gfx1201", "gfx1100"]
    assert merged["kernel"] == {"status": "pass"} and "hipx" in merged["kernels"]
    assert "## lane hiptrx (gfx1201)" in md and "## lane hipx (gfx1100)" in md


def test_one_lane_failing_fails(tmp_path):
    a = _lane(tmp_path, "hiptrx", "gfx1201", "pass", ["x"])
    b = _lane(tmp_path, "hipx", "gfx1100", "fail", ["x"])
    merged, _ = mod.merge([("hiptrx", a), ("hipx", b)])
    assert merged["verdict"] == "fail"


def test_missing_lane_is_not_silence(tmp_path):
    a = _lane(tmp_path, "hiptrx", "gfx1201", "pass", ["x"])
    merged, md = mod.merge([("hiptrx", a), ("hipx", tmp_path / "absent")])
    assert merged["verdict"] == "fail"
    assert merged["lanes_missing"] == ["hipx"]
    assert "no evidence" in md


def test_cli_exit_codes(tmp_path):
    a = _lane(tmp_path, "hiptrx", "gfx1201", "pass", ["x"])
    out, md = tmp_path / "m.json", tmp_path / "m.md"
    assert mod.main(["--lane", f"hiptrx={a}", "--out", str(out), "--md", str(md)]) == 0
    assert json.loads(out.read_text())["head"] == "h"
    assert mod.main(["--lane", f"hiptrx={a}", "--lane", f"hipx={tmp_path / 'absent'}", "--out", str(out), "--md", str(md)]) == 1
