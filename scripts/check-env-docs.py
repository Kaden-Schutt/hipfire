#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "env-vars.md"
REFERENCE_DOCS = [ROOT / "AGENTS.md", ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
DIRECT_ENV = re.compile(
    r'\b(?:std::)?env::var(?:_os)?\(\s*"(HIPFIRE_[A-Z0-9_]+)"\s*\)',
    re.MULTILINE,
)
BOOTSTRAP_ENV = {
    "HIPFIRE_HOME",
    "HIPFIRE_MODELS_DIR",
    "HIPFIRE_DAEMON_BIN",
    "HIPFIRE_TUI_BIN",
    "HIPFIRE_CLI_BIN",
    "HIPFIRE_HF_BASE",
    "HIPFIRE_REGISTRY_URL",
    "HIPFIRE_NO_REGISTRY_FETCH",
    "HIPFIRE_KERNEL_CACHE",
    "HIPFIRE_SPILL_DIR",
    "HIPFIRE_QUANT_DIAG_PATH",
}
CENTRAL_CONFIG_READERS = {
    "crates/hipfire-config/src/lib.rs",
    "crates/hipfire-runtime/src/config.rs",
    "crates/rdna-compute/src/feature_flags.rs",
}


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

    direct_reads: list[tuple[str, str]] = []
    for path in (ROOT / "crates").rglob("*.rs"):
        relative = path.relative_to(ROOT).as_posix()
        if (
            relative in CENTRAL_CONFIG_READERS
            or path.name == "build.rs"
            or "examples" in path.parts
            or "tests" in path.parts
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in DIRECT_ENV.findall(text):
            if name not in BOOTSTRAP_ENV:
                direct_reads.append((relative, name))

    if direct_reads:
        print("production HIPFIRE_* reads must use ProcessConfig/TOML:")
        for path, name in sorted(direct_reads):
            print(f"  {path}: {name}")
        return 1

    print(
        "env-docs: references covered; production HIPFIRE_* reads are config-owned"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
