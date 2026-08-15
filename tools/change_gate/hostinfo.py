# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Host identity probes for the change gate (no GPU required to import)."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

# gfx_target_version (kfd topology) → arch id. Mirrors scripts/speed-gate.sh.
_KFD_GFX_MAP: dict[str, str] = {
    "90006": "gfx906",
    "90008": "gfx908",
    "100100": "gfx1010",
    "100300": "gfx1030",
    "100302": "gfx1030",
    "110000": "gfx1100",
    "110001": "gfx1100",
    "110501": "gfx1151",
    "120000": "gfx1200",
    "120001": "gfx1201",
}

# HSA_OVERRIDE_GFX_VERSION → arch. Mirrors scripts/speed-gate.sh.
_HSA_OVERRIDE_MAP: dict[str, str] = {
    "9.0.6": "gfx906",
    "9.0": "gfx906",
    "10.1.0": "gfx1010",
    "10.1": "gfx1010",
    "10.3.0": "gfx1030",
    "10.3": "gfx1030",
    "11.0.0": "gfx1100",
    "11.0": "gfx1100",
}

_GFX_NAME_RE = re.compile(r"^gfx\d+")
_ROCM_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")


def gfx_arch() -> str | None:
    """Detect the host GPU arch without requiring a live GPU workload.

    Ladder (same order as ``scripts/speed-gate.sh``):
      1. ``HIPFIRE_BASELINE_ARCH`` env
      2. arch-probe binaries (``amdgpu-arch`` / ``offload-arch``)
      3. KFD topology ``gfx_target_version`` mapping
      4. ``rocminfo`` Name: gfx* scrape
      5. ``HSA_OVERRIDE_GFX_VERSION`` override (applied last, like the shell)

    Returns ``None`` when undetectable — never guesses.
    """
    arch: str | None = None

    env_arch = os.environ.get("HIPFIRE_BASELINE_ARCH", "").strip()
    if env_arch:
        arch = env_arch
    else:
        for probe in (
            "amdgpu-arch",
            "offload-arch",
            "/opt/rocm/bin/amdgpu-arch",
            "/opt/rocm/bin/offload-arch",
            "/opt/rocm/llvm/bin/amdgpu-arch",
        ):
            path = probe if probe.startswith("/") else shutil.which(probe)
            if not path or (probe.startswith("/") and not os.access(path, os.X_OK)):
                continue
            try:
                out = subprocess.run(
                    [path],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            line = (out.stdout or "").strip().splitlines()
            if line and line[0].strip():
                cand = line[0].strip()
                if _GFX_NAME_RE.match(cand):
                    arch = cand
                    break

        if arch is None:
            kfd_root = Path("/sys/class/kfd/kfd/topology/nodes")
            if kfd_root.is_dir():
                try:
                    nodes = sorted(kfd_root.iterdir())
                except OSError:
                    nodes = []
                for node in nodes:
                    props = node / "properties"
                    if not props.is_file():
                        continue
                    try:
                        text = props.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    ver: str | None = None
                    for line in text.splitlines():
                        if "gfx_target_version" in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                ver = parts[1].strip()
                            break
                    if ver and ver in _KFD_GFX_MAP:
                        arch = _KFD_GFX_MAP[ver]
                        break

        if arch is None and shutil.which("rocminfo"):
            try:
                out = subprocess.run(
                    ["rocminfo"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                out = None
            if out is not None:
                for line in (out.stdout or "").splitlines():
                    # awk '/^  Name:/ && $2 ~ /^gfx/ {print $2; exit}'
                    if line.startswith("  Name:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].startswith("gfx"):
                            arch = parts[1].strip()
                            break

    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "").strip()
    if override in _HSA_OVERRIDE_MAP:
        arch = _HSA_OVERRIDE_MAP[override]

    if not arch:
        return None
    return arch if _GFX_NAME_RE.match(arch) else None


def rocm_version() -> str | None:
    """Best-effort ROCm version string, or ``None`` if undetectable."""
    env = os.environ.get("ROCM_VERSION", "").strip()
    if env:
        return env

    for path in (
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/version"),
        Path("/opt/rocm/.info/version-dev"),
    ):
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    m = _ROCM_VERSION_RE.search(text)
                    return m.group(1) if m else text.split()[0]
        except OSError:
            continue

    rocm_smi = shutil.which("rocm-smi") or (
        "/opt/rocm/bin/rocm-smi" if os.access("/opt/rocm/bin/rocm-smi", os.X_OK) else None
    )
    if rocm_smi:
        try:
            out = subprocess.run(
                [rocm_smi, "--showdriverversion"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            blob = (out.stdout or "") + (out.stderr or "")
            m = _ROCM_VERSION_RE.search(blob)
            if m:
                return m.group(1)
        except (OSError, subprocess.TimeoutExpired):
            pass

    hipcc = shutil.which("hipcc") or (
        "/opt/rocm/bin/hipcc" if os.access("/opt/rocm/bin/hipcc", os.X_OK) else None
    )
    if hipcc:
        try:
            out = subprocess.run(
                [hipcc, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            blob = (out.stdout or "") + (out.stderr or "")
            m = re.search(r"ROC[Mm].*?(\d+\.\d+(?:\.\d+)?)", blob) or _ROCM_VERSION_RE.search(
                blob
            )
            if m:
                return m.group(1)
        except (OSError, subprocess.TimeoutExpired):
            pass

    return None


def models_dir() -> Path:
    """Resolve the models directory.

    Honour ``HIPFIRE_MODELS_DIR``, then ``${HIPFIRE_DIR:-~/.hipfire}/models``
    (same as ``.research/dead-gates/coherence-gate.sh``).
    """
    explicit = os.environ.get("HIPFIRE_MODELS_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    hipfire_dir = os.environ.get("HIPFIRE_DIR", "").strip()
    root = Path(hipfire_dir).expanduser() if hipfire_dir else Path.home() / ".hipfire"
    return root / "models"


def have_model(basename: str, *, models_dir: Path | str | None = None) -> bool:
    """Return True if ``basename`` exists under the models dir.

    A model "exists" if the path is a file, symlink, or directory — the old
    gate symlink-gated rows (``[ -f ]`` follows symlinks; we also accept a
    directory tree for multi-file layouts).
    """
    root = Path(models_dir) if models_dir is not None else globals()["models_dir"]()
    path = root / basename
    try:
        return path.exists()  # True for file, dir, or symlink (broken → False)
    except OSError:
        return False


def binary_md5(path: str | Path) -> str | None:
    """MD5 hex digest of a file, or ``None`` if unreadable/missing."""
    p = Path(path)
    try:
        if not p.is_file():
            return None
        digest = hashlib.md5()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None
