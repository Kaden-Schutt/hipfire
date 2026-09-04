#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Merge per-architecture hw-gate evidence into one hw-gate.json + hw-gate.md.

The hardware job runs as a matrix (gfx1201 on hiptrx, gfx1100 on hipx, ...);
each lane writes its own hw-gate.json/.md. The seats read one evidence file,
so this merges the lanes deterministically:

  * `verdict` is pass only when every lane that ran passed; a lane whose
    artifact is missing (runner offline, job cancelled) is recorded under
    `lanes_missing` and makes the verdict `fail` — silence is not evidence.
  * `fixtures` is the concatenation with each entry stamped `host_gfx`.
  * `kernel` is the first lane's non-null kernel report, others under `kernels`.
  * `hw-gate.md` is the lanes' markdown joined under per-lane headings.

Usage: merge_evidence.py --lane NAME=DIR [--lane NAME=DIR ...] --out hw-gate.json --md hw-gate.md
       DIR contains hw-gate.json and hw-gate.md, or is missing/empty for a lane that did not run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def merge(lanes: list[tuple[str, Path]]) -> tuple[dict, str]:
    runs: list[dict] = []
    missing: list[str] = []
    fixtures: list[dict] = []
    kernels: dict[str, dict] = {}
    md_parts: list[str] = []
    for name, d in lanes:
        j = d / "hw-gate.json"
        if not j.is_file():
            missing.append(name)
            md_parts.append(f"## lane {name}\n\n_no evidence: the lane did not run or its artifact is missing_\n")
            continue
        ev = json.loads(j.read_text(encoding="utf-8"))
        gfx = (ev.get("host") or {}).get("gfx", name)
        runs.append({"lane": name, "gfx": gfx, "verdict": ev.get("verdict"), "host": ev.get("host"),
                     "binaries": ev.get("binaries"), "buckets": ev.get("buckets"), "precondition_error": ev.get("precondition_error")})
        for fx in ev.get("fixtures", []):
            fx = dict(fx)
            fx["host_gfx"] = gfx
            fx["lane"] = name
            fixtures.append(fx)
        if ev.get("kernel") is not None:
            kernels[name] = ev["kernel"]
        m = d / "hw-gate.md"
        md_parts.append(f"## lane {name} ({gfx})\n\n" + (m.read_text(encoding="utf-8") if m.is_file() else "_hw-gate.md missing_\n"))
    verdict = "pass" if runs and not missing and all(r["verdict"] == "pass" for r in runs) else "fail"
    first = runs[0] if runs else {}
    merged = {
        "schema": "hipfire.hw-gate.evidence",
        "version": 2,
        "verdict": verdict,
        "base": first.get("base"),
        "head": first.get("head"),
        "buckets": sorted({b for r in runs for b in (r.get("buckets") or [])}),
        "host": {"lanes": [r["gfx"] for r in runs]},
        "binaries": {r["lane"]: r["binaries"] for r in runs},
        "lanes": runs,
        "lanes_missing": missing,
        "fixtures": fixtures,
        "kernel": next(iter(kernels.values()), None),
        "kernels": kernels,
        "logs_dir": "hw-gate-logs",
    }
    header = f"# hw-gate evidence — {len(runs)} lane(s)" + (f", missing: {', '.join(missing)}" if missing else "") + f" — verdict **{verdict}**\n\n"
    return merged, header + "\n".join(md_parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--lane", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    args = ap.parse_args(argv)
    lanes = []
    for spec in args.lane:
        name, _, d = spec.partition("=")
        if not name or not d:
            ap.error(f"--lane expects NAME=DIR, got {spec!r}")
        lanes.append((name, Path(d)))
    merged, md = merge(lanes)
    # base/head: take from the first available lane file verbatim
    for _, d in lanes:
        j = d / "hw-gate.json"
        if j.is_file():
            ev = json.loads(j.read_text(encoding="utf-8"))
            merged["base"], merged["head"] = ev.get("base"), ev.get("head")
            break
    Path(args.out).write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.md).write_text(md, encoding="utf-8")
    print(f"merged {len(merged['lanes'])} lane(s), missing {merged['lanes_missing']}, verdict {merged['verdict']}")
    return 0 if merged["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
