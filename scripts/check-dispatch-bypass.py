#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Hold the dispatch-bypass debt table flat or descending.

Emits five metrics for `scripts/leanup-ratchets.sh`:

    bypass_unlisted     an arch crate with bypass calls that the table omits
    bypass_stale        a table row naming a crate that no longer exists
    bypass_regressions  a crate whose bypass count rose above its recorded number
    bypass_slack        debt paid in the tree but not yet banked in the table
    bypass_total        current sum; the holster objective's progress number

The table is `docs/governance/debt-dispatch-bypass.txt`.

The point is not purity
-----------------------
222 call sites reach `Gpu::gemv_*` / `gemm_*` / `fused_*` directly instead of
resolving through `hipfire_dispatch::KernelRegistry`, which has existed since
`e822b319e` (2026-05-30) and which one-and-a-half arch crates use. That debt is
not getting fixed in one commit and should not be.

What this gate buys is convergence: existing debt is recorded and permitted,
new debt fails the build, and paying debt down is a one-line edit to a number.
Over enough commits the ledger walks to zero without any single change having to
be large. A crate that regresses is caught even if another improved by the same
amount, which a single `<= 222` ceiling could not see.

`bypass_total` is asserted as a descending ceiling even though the per-crate
rows already make growth impossible. It is redundant by construction and kept
anyway, because it is the one number the objective is stated in, and putting it
in leanup-thresholds.txt means the descent shows up in the diff of every commit
that banks progress rather than only in this table.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = "docs/governance/debt-dispatch-bypass.txt"

# Direct kernel entry points on the Gpu handle -- the thing the registry exists
# to replace. `gpu.` prefixed so a method definition in rdna-compute is not a hit.
BYPASS = re.compile(r"\bgpu\.(?:gemv|gemm|fused)_[a-z0-9_]+")
REGISTRY = re.compile(r"KernelKey|KernelRegistry")


def measure() -> dict[str, tuple[int, int]]:
    """{crate: (bypass_calls, registry_refs)} over the arch crates."""
    out: dict[str, tuple[int, int]] = {}
    for crate_dir in sorted((ROOT / "crates").glob("hipfire-arch-*")):
        src = crate_dir / "src"
        if not src.is_dir():
            continue
        bypass = registry = 0
        for path in src.rglob("*.rs"):
            text = path.read_text(encoding="utf-8", errors="replace")
            bypass += len(BYPASS.findall(text))
            registry += len(REGISTRY.findall(text))
        if bypass or registry:
            out[crate_dir.name] = (bypass, registry)
    return out


def table() -> dict[str, int]:
    """{crate: recorded_bypass_allowance} from the committed ledger."""
    path = ROOT / TABLE
    if not path.is_file():
        return {}
    rows: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3 or not parts[2].isdigit():
            continue
        rows[parts[0]] = int(parts[2])
    return rows


def main(argv: list[str]) -> int:
    measured, recorded = measure(), table()
    if not recorded:
        print(f"check-dispatch-bypass: table {TABLE} is missing or empty", file=sys.stderr)
        return 2

    # A crate with zero bypass calls need not be listed -- that is the goal state,
    # and forcing a row for it would make finishing the work feel like paperwork.
    unlisted = sorted(c for c, (b, _) in measured.items() if b > 0 and c not in recorded)
    stale = sorted(c for c in recorded if c not in measured)
    regressions = [
        (c, recorded[c], measured[c][0])
        for c in sorted(recorded)
        if c in measured and measured[c][0] > recorded[c]
    ]
    total = sum(b for b, _ in measured.values())

    # Debt paid but not recorded. Left alone, a stale allowance is headroom the
    # debt can silently regrow into: migrate deepseek4 19 -> 0 without editing
    # the row and nothing stops it returning to 19 later. Asserting this at 0
    # forces the ledger down whenever the tree improves, which is the same
    # discipline check-crate-maps.py applies to generated maps.
    paid = [(c, recorded[c], measured[c][0]) for c in sorted(recorded)
            if c in measured and measured[c][0] < recorded[c]]
    slack = sum(was - now for _c, was, now in paid)

    print(f"bypass_unlisted            {len(unlisted)}")
    print(f"bypass_stale               {len(stale)}")
    print(f"bypass_regressions         {len(regressions)}")
    print(f"bypass_slack               {slack}")
    print(f"bypass_total               {total}")

    for c in unlisted:
        print(f"  UNLISTED  {c} has {measured[c][0]} bypass call(s) and no row in {TABLE}")
    for c in stale:
        print(f"  STALE     {c} is in {TABLE} but has no arch crate in the tree")
    for c, was, now in regressions:
        print(f"  REGRESSED {c} {was} -> {now} (+{now - was}); migrate them or record the new number")
    for c, was, now in paid:
        print(f"  PAID      {c} {was} -> {now} ({now - was}); lower the row in {TABLE} to bank it")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
