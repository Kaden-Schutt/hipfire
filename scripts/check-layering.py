#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Assert the saddle layer order from the real Cargo graph.

Emits two metrics for `scripts/leanup-ratchets.sh`:

    arch_depends_upward   arch crate -> a crate that sits above the arch band
    layer_cycles          cycles in the non-dev dependency graph

Why derived rather than declared
--------------------------------
The Phase 3 scope asserted "forbid `hipfire-arch-* -> saddle-core |
hipfire-engine | hipfire-dispatch`". That was wrong, and only measuring showed
it: **11 such edges already exist** -- nine arch crates depend on
`hipfire-dispatch` and two on `saddle-core`. Those are layers 4 and 5, *below*
the arch band at 6-7, so they are downward edges and entirely correct.

Hardcoding a guessed rule would have failed the build on legitimate structure.
So the band is computed from the graph each run: anything whose longest
dependency chain is deeper than the deepest arch crate is "above", and an arch
crate depending on it is an inversion.

`[dev-dependencies]` are excluded deliberately. `hipfire-runtime` lists eleven
arch crates there for its examples, and cargo's own cycle checker excludes that
edge -- it is why the workspace builds at all despite every arch crate depending
on `hipfire-runtime`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def workspace_members() -> set[str]:
    return {p.parent.name for p in (ROOT / "crates").glob("*/Cargo.toml")}


def graph(members: set[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in sorted(members):
        text = (ROOT / "crates" / name / "Cargo.toml").read_text(encoding="utf-8")
        body = ""
        for m in re.finditer(r"^\[dependencies\](.*?)(?=^\[|\Z)", text, re.S | re.M):
            body += m.group(1)
        deps = set(re.findall(r"^([A-Za-z0-9_-]+)\s*=", body, re.M))
        out[name] = {d for d in deps if d in members}
    return out


def committed_layers() -> dict[str, int]:
    """The frozen layer assignment. NOT recomputed from the graph under test."""
    out: dict[str, int] = {}
    f = ROOT / "scripts" / "layering.txt"
    if not f.exists():
        print("check-layering: FAIL — scripts/layering.txt is missing.")
        print("  Without it the ordering would be derived from the graph being")
        print("  checked, which is how the first version of this check passed an")
        print("  injected inversion.")
        sys.exit(2)
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        layer, name = line.split(None, 1)
        out[name.strip()] = int(layer)
    return out


def main(argv: list[str]) -> int:
    verbose = "--verbose" in argv
    members = workspace_members()
    g = graph(members)
    layer = committed_layers()

    # a crate present in the tree but absent from the committed file is a gap:
    # it would otherwise be silently exempt from every rule below.
    unlisted = sorted(members - set(layer))
    missing = sorted(set(layer) - members)

    inversions: list[tuple[str, str, int, int]] = []
    for c in sorted(g):
        if c not in layer:
            continue
        for d in sorted(g[c]):
            if d not in layer:
                continue
            if layer[d] >= layer[c]:
                inversions.append((c, d, layer[c], layer[d]))

    if verbose:
        by: dict[int, list[str]] = {}
        for c, v in layer.items():
            by.setdefault(v, []).append(c)
        for v in sorted(by):
            print(f"  layer {v:2d}  {', '.join(sorted(by[v]))}")

    print(f"arch_depends_upward        {len(inversions)}")
    print(f"layer_unlisted_crates      {len(unlisted)}")
    for c, d, lc, ld in inversions:
        print(f"  INVERSION {c} (layer {lc}) -> {d} (layer {ld}); a dependency must be strictly lower")
    for c in unlisted:
        print(f"  UNLISTED  {c} is in the tree but not in scripts/layering.txt")
    for c in missing:
        print(f"  STALE     {c} is in scripts/layering.txt but not in the tree")
    return 1 if (inversions or unlisted) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
