#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "env-vars.md"
REFERENCE_DOCS = [ROOT / "AGENTS.md", ROOT / "README.md", ROOT / "CONTRIBUTING.md"]


def env_vars(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return set(re.findall(r"\bHIPFIRE_[A-Z0-9_]+\b", text))


def main() -> int:
    canonical = DOC.read_text(encoding="utf-8", errors="ignore")
    missing: list[tuple[str, str]] = []
    for path in REFERENCE_DOCS:
        for name in sorted(env_vars(path)):
            if name not in canonical:
                missing.append((path.relative_to(ROOT).as_posix(), name))

    if missing:
        print("docs/env-vars.md is missing HIPFIRE_* vars referenced by top-level docs:")
        for path, name in missing:
            print(f"  {path}: {name}")
        return 1

    print("env-docs: top-level HIPFIRE_* references are covered by docs/env-vars.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
