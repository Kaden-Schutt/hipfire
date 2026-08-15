#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Behavioral gate: bench_ocp_fp8_recipes.sh rejects unreviewed family-looking arches.

Family-looking but unreviewed atoms (e.g. gfx1199 / gfx1250) must fail closed
before hipcc is invoked, matching the exact allowlist used by recipes_fp8.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "bench_ocp_fp8_recipes.sh"

# Unreviewed atoms that still match a naive gfx11xx/gfx12xx family pattern.
UNREVIEWED_FAMILY_LOOKING = ("gfx1199", "gfx1250")


def _make_fake_hipcc(bin_dir: Path) -> Path:
    """Executable that records any invocation; must never run for reject cases."""
    marker = bin_dir / "hipcc_invoked"
    hipcc = bin_dir / "hipcc"
    hipcc.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{marker}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    hipcc.chmod(hipcc.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hipcc


def _run_wrapper(*, arch: str, hipcc: Path, build_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["ARCH"] = arch
    env["HIPCC"] = str(hipcc)
    env["BUILD_DIR"] = str(build_dir)
    # Keep ROCm path resolution from probing the host when possible; HIPCC is forced.
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


class BenchOcpFp8RecipesArchGateTest(unittest.TestCase):
    def test_unreviewed_family_looking_arches_rejected_before_hipcc(self) -> None:
        self.assertTrue(WRAPPER.is_file(), f"missing wrapper: {WRAPPER}")

        for arch in UNREVIEWED_FAMILY_LOOKING:
            with self.subTest(arch=arch):
                with tempfile.TemporaryDirectory(prefix="ocp-fp8-arch-gate-") as tmp:
                    tmp_path = Path(tmp)
                    hipcc = _make_fake_hipcc(tmp_path)
                    build_dir = tmp_path / "build"
                    marker = tmp_path / "hipcc_invoked"

                    result = _run_wrapper(arch=arch, hipcc=hipcc, build_dir=build_dir)
                    combined = f"{result.stdout}\n{result.stderr}"

                    self.assertEqual(
                        result.returncode,
                        2,
                        f"expected exit 2 for ARCH={arch}; got {result.returncode}\n{combined}",
                    )
                    self.assertIn(
                        "unsupported",
                        combined.lower(),
                        f"missing unsupported-architecture diagnostic for ARCH={arch}\n{combined}",
                    )
                    self.assertIn(
                        arch,
                        combined,
                        f"diagnostic should name rejected ARCH={arch}\n{combined}",
                    )
                    self.assertFalse(
                        marker.exists(),
                        f"fake HIPCC must not run for ARCH={arch}; marker contents: "
                        f"{marker.read_text(encoding='utf-8') if marker.exists() else ''}",
                    )


if __name__ == "__main__":
    unittest.main()
