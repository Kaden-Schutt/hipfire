#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Aggregate rocprofv3 DS4 decode kernel statistics into requested stages."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

STAGES = (
    "indexer_scoring",
    "topk_selection",
    "compressed_kv_read",
    "flash_attention",
    "moe",
    "rest",
)


def classify(name: str) -> str:
    n = name.lower()
    if "indexer_relu_score" in n or "indexer_compressed_k_score" in n:
        return "indexer_scoring"
    if "indexer_top_k" in n:
        return "topk_selection"
    if "topk_kv_gather" in n or "indexer_kv_gather" in n:
        return "compressed_kv_read"
    if "deepseek4_attn_swa" in n or "deepseek4_attn_pos0" in n:
        return "flash_attention"
    # Only call-site-specific routed-MoE symbols are safe to assign here.
    # Generic E8-SoA GEMVs are shared by attention, compressors, FFN and the
    # head; assigning all of them to MoE would manufacture a large false share.
    if any(marker in n for marker in ("moe", "expert", "router")):
        return "moe"
    return "rest"


def number(row: dict[str, str], *names: str) -> float:
    normalized = {key.lower().replace(" ", ""): value for key, value in row.items()}
    for name in names:
        key = name.lower().replace(" ", "")
        if key in normalized and normalized[key] not in ("", None):
            return float(normalized[key])
    raise KeyError(f"none of {names!r} in columns {tuple(row)}")


def find_stats(root: Path) -> Path:
    matches = sorted(root.rglob("*kernel_stats.csv"))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one *kernel_stats.csv below {root}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--decode-steps", type=int, required=True,
                        help="number of decode forwards captured, including warmup")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    stats = find_stats(args.profile_dir)

    totals = {stage: {"duration_ns": 0.0, "calls": 0.0} for stage in STAGES}
    kernels: list[dict[str, object]] = []
    with stats.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("Name") or row.get("Kernel_Name") or row.get("KernelName")
            if not name:
                raise SystemExit(f"kernel name column missing in {stats}: {tuple(row)}")
            calls = number(row, "Calls", "Count")
            duration_ns = number(row, "TotalDurationNs", "TotalDuration (ns)", "TotalDuration")
            stage = classify(name)
            totals[stage]["duration_ns"] += duration_ns
            totals[stage]["calls"] += calls
            kernels.append({"name": name, "calls": calls, "duration_ns": duration_ns, "stage": stage})

    total_ns = sum(value["duration_ns"] for value in totals.values())
    report = {
        "stats_csv": str(stats),
        "decode_steps": args.decode_steps,
        "gpu_kernel_total_ms_per_decode": total_ns / 1e6 / args.decode_steps,
        "classification_note": (
            "moe includes only call-site-specific moe/expert/router symbols; "
            "generic E8 GEMVs shared by multiple stages remain in rest"
        ),
        "stages": {},
    }
    print(f"stats_csv={stats}")
    print(f"classification_note={report['classification_note']}")
    print("stage                    total_ms   ms/decode    share_pct      calls")
    for stage in STAGES:
        duration = totals[stage]["duration_ns"]
        calls = int(totals[stage]["calls"])
        stage_report = {
            "total_ms": duration / 1e6,
            "ms_per_decode": duration / 1e6 / args.decode_steps,
            "share_pct": 100.0 * duration / total_ns if total_ns else 0.0,
            "calls": calls,
        }
        report["stages"][stage] = stage_report
        print(
            f"{stage:24s} {stage_report['total_ms']:10.3f} "
            f"{stage_report['ms_per_decode']:11.3f} "
            f"{stage_report['share_pct']:12.3f} {calls:10d}"
        )
    print(f"gpu_kernel_total_ms_per_decode={report['gpu_kernel_total_ms_per_decode']:.3f}")

    report["kernels"] = sorted(kernels, key=lambda row: float(row["duration_ns"]), reverse=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
