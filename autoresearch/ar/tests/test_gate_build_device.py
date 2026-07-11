# Copyright (c) Kaden Schutt
"""Daemon build (sha-cache), arch->device resolution, arch->box deferral — the seams
that turn the live gate from the subprocess.run("base") scaffold into a real A/B."""
import os

import pytest

from autoresearch.ar.gate import build as B
from autoresearch.ar.gate.device import gpu_gfx_order, resolve_device
from autoresearch.ar.gate.run import affected_archs, daemon_touched


# ---- device resolution (ROCR order == HIP dev order) ----

ROCMINFO = """
*** Agent 1 ***
  Name:                    AMD Ryzen 9 7950X
  Device Type:             CPU
*** Agent 2 ***
  Name:                    gfx1100
  Marketing Name:          Radeon RX 7900 XTX
  Device Type:             GPU
*** Agent 3 ***
  Name:                    gfx1151
  Device Type:             GPU
"""


def test_gpu_order_skips_cpu_agents():
    assert gpu_gfx_order(ROCMINFO) == ["gfx1100", "gfx1151"]


def test_resolve_device_matches_arch_to_hip_index():
    assert resolve_device("gfx1100", rocminfo_text=ROCMINFO) == 0
    assert resolve_device("gfx1151", rocminfo_text=ROCMINFO) == 1


def test_resolve_device_absent_arch_falls_back_to_default():
    assert resolve_device("gfx1201", rocminfo_text=ROCMINFO, default=0) == 0
    assert resolve_device("gfx999", rocminfo_text="", default=3) == 3


def test_strict_device_resolution_never_guesses_on_missing_or_empty_rocminfo():
    with pytest.raises(RuntimeError, match="required arch gfx1201 absent"):
        resolve_device("gfx1201", rocminfo_text=ROCMINFO, strict=True)
    with pytest.raises(RuntimeError, match="rocminfo unavailable"):
        resolve_device("gfx1100", rocminfo_text="", strict=True)


# ---- arch->box deferral ----

class _Cfg:
    archs = ["gfx1100", "gfx1151", "gfx1201"]


def test_daemon_touched_only_for_compiled_paths():
    assert daemon_touched(["kernels/src/gemv.hip"])
    assert daemon_touched(["crates/hipfire-runtime/src/x.rs"])
    assert daemon_touched(["Cargo.lock"])
    assert not daemon_touched(["autoresearch/ar/gate/run.py", "docs/x.md", ".github/workflows/y.yml"])


def test_affected_archs_defers_on_non_daemon_change():
    # docs/ar-only (like #507) -> NO arch affected -> every box defers.
    assert affected_archs(["autoresearch/ar/gate/run.py", "docs/spec.md"], _Cfg()) == []
    # a SHARED kernel change (no arch suffix) -> all archs (conservative: #if blocks).
    assert affected_archs(["kernels/src/gemv.hip"], _Cfg()) == ["gfx1100", "gfx1151", "gfx1201"]
    # a Rust/crates change -> all archs (shared daemon code).
    assert affected_archs(["crates/hipfire-runtime/src/x.rs"], _Cfg()) == ["gfx1100", "gfx1151", "gfx1201"]


def test_affected_archs_narrows_to_arch_specific_kernel():
    # An arch-SUFFIXED kernel affects ONLY that arch -> the OTHER box defers it faithfully.
    # gfx1201-only change: hiptrx runs gfx1201, hipx runs NOTHING (defers 1100+1151).
    assert affected_archs(["kernels/src/gemv_mq4g256_lloyd.gfx1201.hip"], _Cfg()) == ["gfx1201"]
    # gfx1100-only change: hipx runs gfx1100, hiptrx defers gfx1201.
    assert affected_archs(["kernels/src/fused_qkvza.gfx1100.hip"], _Cfg()) == ["gfx1100"]
    # a suffix for an arch we DON'T gate (CDNA gfx942) -> affects none of our archs.
    assert affected_archs(["kernels/src/gemm_bf16_mfma.gfx942.hip"], _Cfg()) == []
    # mixed: an arch-specific file + a shared file -> union widens back to all.
    assert affected_archs(["kernels/src/x.gfx1201.hip", "kernels/src/shared.hip"], _Cfg()) \
        == ["gfx1100", "gfx1151", "gfx1201"]


# ---- daemon build: sha-cache + failure mapping (injected git/cmd, no cargo) ----

class _Proc:
    def __init__(self, rc, stderr=""):
        self.returncode, self.stderr = rc, ""
        self.stderr = stderr


def _git_stub(sha, calls):
    def run_git(repo, *args):
        calls.append(args)
        if args[:1] == ("rev-parse",):
            return (0, sha + "\n")
        return (0, "")
    return run_git


def test_build_daemon_uses_sha_cache(tmp_path, monkeypatch):
    sha = "deadbeef1234"
    (tmp_path / f"gate_daemon_{sha}").write_text("cached-binary")  # pre-seed the cache
    calls = []
    built = {"n": 0}

    def run_cmd(cmd, cwd):
        built["n"] += 1
        return _Proc(0)

    out = B.build_daemon("HEAD", "/repo", cache_dir=str(tmp_path),
                         run_git=_git_stub(sha, calls), run_cmd=run_cmd)
    assert out == str(tmp_path / f"gate_daemon_{sha}")
    assert built["n"] == 0                     # cache hit => cargo NEVER invoked
    assert ("worktree", "add", "--force", "--detach") != tuple(calls[-1][:4]) if calls else True


def test_build_daemon_builds_and_caches_on_miss(tmp_path):
    sha = "cafe00011122"
    calls = []

    def run_cmd(cmd, cwd):
        # emulate cargo producing the daemon binary in the worktree
        os.makedirs(os.path.join(cwd, "target/release/examples"), exist_ok=True)
        with open(os.path.join(cwd, B.DAEMON_REL), "w") as fh:
            fh.write("fresh-binary")
        assert cmd == B.CARGO_BUILD
        return _Proc(0)

    out = B.build_daemon("HEAD", "/repo", cache_dir=str(tmp_path),
                         run_git=_git_stub(sha, calls), run_cmd=run_cmd)
    assert out == str(tmp_path / f"gate_daemon_{sha}")
    assert open(out).read() == "fresh-binary"
    assert ("worktree", "add", "--force", "--detach", str(tmp_path / f"gate_wt_{sha}"), sha) in calls
    assert ("worktree", "remove", "--force", str(tmp_path / f"gate_wt_{sha}")) in calls  # cleaned up


def test_build_daemon_raises_on_build_failure(tmp_path):
    sha = "badf00d55667"

    def run_cmd(cmd, cwd):
        return _Proc(1, stderr="error[E0432]: unresolved import")   # no binary produced

    with pytest.raises(RuntimeError, match="daemon build failed"):
        B.build_daemon("HEAD", "/repo", cache_dir=str(tmp_path),
                       run_git=_git_stub(sha, []), run_cmd=run_cmd)


def test_build_daemon_raises_on_unresolved_ref(tmp_path):
    def run_git(repo, *args):
        return (128, "")   # rev-parse fails
    with pytest.raises(RuntimeError, match="cannot resolve ref"):
        B.build_daemon("nope", "/repo", cache_dir=str(tmp_path), run_git=run_git)
