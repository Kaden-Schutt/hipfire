# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "serve_harness.py"


def main() -> None:
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
