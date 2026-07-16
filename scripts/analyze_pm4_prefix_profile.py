#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Turn an exact retained-PM4 prefix profile into a dispatch bill of debt."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def signature(launch):
    return (
        launch["kernel"],
        tuple(launch["grid"]),
        tuple(launch["block"]),
    )


def analyze(report):
    profile = report.get("pm4_prefix_profile")
    if not profile:
        raise ValueError("report has no pm4_prefix_profile")
    if profile.get("step") != 1 or profile.get("start") != 1:
        raise ValueError("dispatch attribution requires --profile-prefix-start 1 and step 1")

    rows = profile["rows"]
    launches = report["decode"]["captures"][0]["redline_capture"]["sequence"]
    launch_count = profile["launches"]
    if len(rows) != launch_count or len(launches) != launch_count:
        raise ValueError(
            f"profile/sequence length mismatch: rows={len(rows)} "
            f"sequence={len(launches)} launches={launch_count}"
        )

    dispatches = []
    previous = 0.0
    for index, (row, launch) in enumerate(zip(rows, launches), start=1):
        if row["prefix"] != index:
            raise ValueError(f"expected prefix {index}, got {row['prefix']}")
        if row["last_kernel"] != launch["kernel"]:
            raise ValueError(
                f"prefix {index} kernel mismatch: profile={row['last_kernel']} "
                f"capture={launch['kernel']}"
            )
        cumulative = row["median_gpu_us"]
        marginal = cumulative - previous
        previous = cumulative
        dispatches.append(
            {
                "prefix": index,
                "kernel": launch["kernel"],
                "grid": launch["grid"],
                "block": launch["block"],
                "marginal_gpu_us": marginal,
                "cumulative_gpu_us": cumulative,
            }
        )

    grouped = defaultdict(list)
    for row, launch in zip(dispatches, launches):
        grouped[signature(launch)].append(row["marginal_gpu_us"])

    total = rows[-1]["median_gpu_us"]
    groups = []
    for (kernel, grid, block), values in grouped.items():
        group_total = sum(values)
        groups.append(
            {
                "kernel": kernel,
                "grid": list(grid),
                "block": list(block),
                "count": len(values),
                "total_gpu_us": group_total,
                "share_pct": 100.0 * group_total / total,
                "mean_gpu_us": statistics.fmean(values),
                "median_gpu_us": statistics.median(values),
                "p10_gpu_us": percentile(values, 0.10),
                "p90_gpu_us": percentile(values, 0.90),
                "min_gpu_us": min(values),
                "max_gpu_us": max(values),
                "negative_marginals": sum(value < 0.0 for value in values),
            }
        )
    groups.sort(key=lambda row: row["total_gpu_us"], reverse=True)

    return {
        "sequence_hash": report["decode"]["captures"][0]["redline_capture"][
            "sequence_hash"
        ],
        "launches": launch_count,
        "unique_kernels": report["decode"]["captures"][0]["redline_capture"][
            "unique_kernels"
        ],
        "context_tokens": profile["context_tokens"],
        "repeats": profile["repeats"],
        "steady_state": profile.get("steady_state", False),
        "full_prefix_gpu_us": total,
        "implied_gpu_tok_s": 1_000_000.0 / total,
        "groups": groups,
        "dispatches": dispatches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    result = analyze(json.loads(args.report.read_text()))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"sequence={result['sequence_hash']} launches={result['launches']} "
        f"gpu={result['full_prefix_gpu_us']:.3f}us "
        f"implied={result['implied_gpu_tok_s']:.3f} tok/s"
    )
    print("share%  total_us  count  median_us  grid          kernel")
    for row in result["groups"][: args.top]:
        grid = "x".join(str(value) for value in row["grid"])
        print(
            f"{row['share_pct']:6.2f}  {row['total_gpu_us']:8.3f}  "
            f"{row['count']:5d}  {row['median_gpu_us']:9.3f}  "
            f"{grid:12s}  {row['kernel']}"
        )


if __name__ == "__main__":
    main()
