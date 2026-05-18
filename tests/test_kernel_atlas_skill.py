#!/usr/bin/env python3
import json
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "hipfire-kernel-atlas"


class KernelAtlasSkillTest(unittest.TestCase):
    def test_skill_metadata_exposes_atlas_fit_workflow(self):
        skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        meta = json.loads((SKILL_DIR / "skill.json").read_text(encoding="utf-8"))

        self.assertIn("name: hipfire-kernel-atlas", skill)
        self.assertIn("render-fit", skill)
        self.assertIn("collect-ar", skill)
        self.assertIn("collect-dflash", skill)
        self.assertIn("ISA Fit View", skill)
        self.assertEqual(meta["name"], "hipfire-kernel-atlas")
        self.assertIn("quant", meta["description"].lower())
        self.assertIn("render-fit", meta["triggers"])

    def test_render_fit_wrapper_points_at_repo_cli(self):
        wrapper = SKILL_DIR / "render-fit.sh"
        text = wrapper.read_text(encoding="utf-8")

        self.assertIn("scripts/kernel_atlas.py", text)
        self.assertIn("render-fit", text)

        proc = subprocess.run(
            ["bash", str(wrapper), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("render an ASCII ISA/quant fit view", proc.stdout)


if __name__ == "__main__":
    unittest.main()
