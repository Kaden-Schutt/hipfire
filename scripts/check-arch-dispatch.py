#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Count architecture-key DISPATCH outside the arch crates.

Emits one metric for `scripts/leanup-ratchets.sh`:

    arch_key_dispatch    match arms and equality tests on an arch key

Why this exists
---------------
The leanup programme drove `arch_id ==` from 58 sites to 0 and the ratchet
asserts that. It does not catch the same coupling expressed as a *string*:

    match arch { "gemma4" => ThinkMarkers::GEMMA4, _ => default }

That is a hardcoded architecture table wearing different clothes, and it was
written into `hipfire-runtime/src/emit_text.rs` during Phase 3B before being
reverted. Nothing in the gate would have objected. This closes that hole.

What counts, and what does not
------------------------------
Only dispatch shapes: a `"key" =>` match arm, `== "key"`, or `.eq("key")`.

A raw substring count is deliberately NOT used. Measuring it first found 200
occurrences, almost all JSON payloads (`{"arch":"qwen35"}`) and test fixtures —
data, not branching. A metric dominated by false positives gets ignored, then
raised, then deleted, which is how the previous ratchet ended up asserting
nothing at all.

`#[cfg(test)]` blocks are skipped: a fixture naming an architecture is describing
a protocol message, not routing on it.

Arch crates are exempt. An architecture recognising its own name is not coupling.

The threshold is a CEILING, not zero. `hipfire-generate` is the architecture
composition root by design, and the daemon keeps a narrow `arch_key() ==
"deepseek4"` check because that arch's captured decode graph must be invalidated
across reset while gemma4's stays valid. Those are decisions with reasons. The
ceiling stops a thirteenth appearing without one.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARCH_KEYS = [
    "qwen35",
    "qwen2",
    "llama",
    "lfm2moe",
    "minimax",
    "cohere2moe",
    "deepseek4",
    "dots_ocr",
    "gemma4",
    "muse_glimmer",
]

GEMMA4_FORBIDDEN = (
    "Gemma4Bindings",
    "g4_superop",
    "run_layer_program",
    "HIPFIRE_GEMMA4_STEP_ROUTE",
    "HIPFIRE_FORWARD_LOWERED",
)


def scan_gemma4_forbidden() -> list[tuple[str, int, str]]:
    path = ROOT / "crates/hipfire-arch-gemma4/src/lowered.rs"
    hits: list[tuple[str, int, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for symbol in GEMMA4_FORBIDDEN:
            if symbol in line:
                hits.append((symbol, lineno, line.strip()[:96]))
    return hits



_ALT = "|".join(re.escape(k) for k in ARCH_KEYS)
DISPATCH = re.compile(
    r'("(?:' + _ALT + r')"\s*=>)'      # match arm
    r'|(==\s*"(?:' + _ALT + r')")'     # equality
    r'|(\.eq\("(?:' + _ALT + r')"\))'  # .eq()
)


def scan() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in sorted((ROOT / "crates").glob("*/src/**/*.rs")):
        crate = path.relative_to(ROOT / "crates").parts[0]
        if crate.startswith("hipfire-arch-"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        in_test = False
        depth = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#[cfg(test)]"):
                in_test = True
                depth = 0
                continue
            if in_test:
                depth += line.count("{") - line.count("}")
                if depth < 0:
                    in_test = False
                continue
            if stripped.startswith(("//", "///", "//!", "*")):
                continue
            if DISPATCH.search(line):
                rel = path.relative_to(ROOT).as_posix()
                hits.append((rel, lineno, stripped[:96]))
    return hits


def main(argv: list[str]) -> int:
    hits = scan()
    print(f"arch_key_dispatch          {len(hits)}")
    if "--verbose" in argv:
        for rel, lineno, text in hits:
            print(f"  {rel}:{lineno}: {text}")

    forbidden = scan_gemma4_forbidden()
    if forbidden:
        print(f"gemma4_forbidden           {len(forbidden)}")
        for symbol, lineno, text in forbidden:
            print(f"  crates/hipfire-arch-gemma4/src/lowered.rs:{lineno}: {symbol}: {text}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
