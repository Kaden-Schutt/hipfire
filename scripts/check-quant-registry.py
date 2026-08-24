#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Assert the tree against the committed quant-type register.

Emits four metrics for `scripts/leanup-ratchets.sh`:

    qt_unlisted              QuantType variant absent from the register
    qt_stale                 register row naming no QuantType variant
    qt_unparseable           variant `QuantType::from_u8` refuses
    qt_disposition_mismatch  register disposition contradicts RAW_CODECS

The register is `docs/quant-formats/qt-register.txt`.

Why a table rather than a ceiling
---------------------------------
This started as three counts (`qt_unparseable`, `qt_orphan_codec_rows`,
`qt_unregistered <= 10`). A count cannot see substitution: delete one justified
entry, add an unjustified one, and cardinality is unchanged. That is exactly
`f00fa8058`, where landing qt=43 deleted six match arms belonging to other
formats and the build stayed green.

`scripts/check-layering.py` already established the shape — UNLISTED for a crate
in the tree but not the table, STALE for a table row with no crate. This applies
the same bidirectional check to the qt space, and folds the exemption problem in
with it: `qt_unregistered <= 10` was uninterpretable because ten variants were a
mix of deliberate non-passthrough (22 is an index tensor, 28/29 load via
paro.rs, 31-37 via arch loaders) and genuine omissions like `cf061b7ed`, where
qt=40's encoder, GEMV and is_mq arms all landed and the model simply would not
load. Declaring a disposition per row makes those two cases distinguishable.

What this does not cover
------------------------
Whether a registered format's kernels exist, are wired, or are correct. Those
need the dispatch registry and the redline oracle respectively.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = "docs/quant-formats/qt-register.txt"
RAW_CODEC_SOURCE = "crates/hipfire-runtime/src/weight_backend.rs"

# The quantizer is a binary crate whose module tree has moved between branches
# (hfq.rs after the saddle decomposition, main.rs before it). Accept either.
QUANT_TYPE_SOURCES = (
    "crates/hipfire-quantize/src/hfq.rs",
    "crates/hipfire-quantize/src/main.rs",
)

# A disposition either requires a RAW_CODECS row or forbids one. There is no
# "either way" — that ambiguity is what made the old ceiling unreadable.
NEEDS_CODEC_ROW = {"passthrough": True, "host-decode": False,
                   "arch-loaded": False, "non-weight": False, "reserved": False}

# `reserved` claims a number for work that has not landed yet, so it is the one
# disposition exempt from the STALE check. Without it the register can only
# mirror the enum, and a register that can only describe the present cannot stop
# two branches claiming the same number -- which is the case it exists for.
RESERVED = "reserved"


def _read(rel: str) -> str | None:
    path = ROOT / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def register() -> dict[int, tuple[str, str]]:
    """{qt: (name, disposition)} from the committed table."""
    text = _read(REGISTER)
    if text is None:
        return {}
    rows: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip() if not line.lstrip().startswith("#") else ""
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        rows[int(parts[0])] = (parts[1], parts[2])
    return rows


def quant_type_variants() -> dict[int, str]:
    for rel in QUANT_TYPE_SOURCES:
        text = _read(rel)
        if text is None:
            continue
        block = re.search(r"enum QuantType\s*\{(.*?)\n\}", text, re.S)
        if block is None:
            continue
        return {int(n): nm for nm, n in re.findall(
            r"^\s+([A-Z][A-Za-z0-9_]*)\s*=\s*(\d+)", block.group(1), re.M)}
    return {}


def from_u8_arms() -> set[int]:
    for rel in QUANT_TYPE_SOURCES:
        text = _read(rel)
        if text is None:
            continue
        body = re.search(r"fn from_u8.*?\n    \}", text, re.S)
        if body is None:
            continue
        # Several arms share a line in some revisions; do not anchor to line start.
        return {int(n) for n in re.findall(r"(\d+)\s*=>\s*Some", body.group(0))}
    return set()


def raw_codec_rows() -> set[int]:
    text = _read(RAW_CODEC_SOURCE)
    if text is None:
        return set()
    block = re.search(r"RAW_CODECS[^=]*=\s*&\[(.*?)\n\];", text, re.S)
    return {int(n) for n in re.findall(r"quant_type:\s*(\d+)", block.group(1))} if block else set()


def main(argv: list[str]) -> int:
    variants, reg = quant_type_variants(), register()
    if not variants:
        print("check-quant-registry: could not locate the QuantType enum", file=sys.stderr)
        return 2
    if not reg:
        print(f"check-quant-registry: register {REGISTER} is missing or empty", file=sys.stderr)
        return 2

    parsed, codecs = from_u8_arms(), raw_codec_rows()

    unlisted = sorted(set(variants) - set(reg))
    stale = sorted(qt for qt in set(reg) - set(variants)
                   if reg[qt][1] != RESERVED)
    unparseable = sorted(set(variants) - parsed)

    mismatch: list[tuple[int, str]] = []
    for qt, (_name, disp) in sorted(reg.items()):
        if qt not in variants:
            continue                      # already counted as stale
        want = NEEDS_CODEC_ROW.get(disp)
        if want is None:
            mismatch.append((qt, f"unknown disposition '{disp}'"))
        elif want and qt not in codecs:
            mismatch.append((qt, f"declared {disp} but has no RAW_CODECS row"))
        elif not want and qt in codecs:
            mismatch.append((qt, f"declared {disp} but has a RAW_CODECS row"))

    # A name change in one place and not the other is a rename half-done.
    for qt, (name, _d) in sorted(reg.items()):
        if qt in variants and variants[qt] != name:
            mismatch.append((qt, f"register says {name}, enum says {variants[qt]}"))

    print(f"qt_unlisted                {len(unlisted)}")
    print(f"qt_stale                   {len(stale)}")
    print(f"qt_unparseable             {len(unparseable)}")
    print(f"qt_disposition_mismatch    {len(mismatch)}")

    if "--verbose" in argv or unlisted or stale or unparseable or mismatch:
        for qt in unlisted:
            print(f"  UNLISTED  qt={qt} {variants[qt]} is declared but absent from {REGISTER}")
        for qt in stale:
            print(f"  STALE     qt={qt} {reg[qt][0]} is in {REGISTER} but no variant declares it")
        for qt in unparseable:
            print(f"  UNPARSED  qt={qt} {variants[qt]} is declared but QuantType::from_u8 rejects it")
        for qt, why in mismatch:
            print(f"  MISMATCH  qt={qt} {why}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
