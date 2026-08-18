#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
"""Occurrence-weight the final decode positions in a rocprofv3 kernel trace.

DeepSeek's current product prefill can exercise the same single-position body as
decode, making whole-process rocprof statistics misleading at the target KV
depth.  The embedding launch is a stable once-per-rank position boundary.  This
tool locates the final N position boundaries inside a bounded tail of the CSV,
then aggregates only dispatches at or after that timestamp.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path


def iter_tail_rows(path: Path, tail_bytes: int):
    with path.open("rb") as trace:
        header = next(csv.reader([trace.readline().decode("utf-8")]))
        indexes = {name: i for i, name in enumerate(header)}
        trace.seek(0, 2)
        file_size = trace.tell()
        offset = max(0, file_size - tail_bytes)
        trace.seek(offset)
        # At offset zero this consumes the CSV header; at a tail offset it
        # consumes the one possibly-partial record crossed by the seek.
        trace.readline()
        for raw in trace:
            if not raw.strip():
                continue
            yield indexes, next(csv.reader([raw.decode("utf-8")]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--positions", type=int, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--anchor", default="embedding_q8")
    parser.add_argument("--tail-bytes", type=int, default=384 * 1024 * 1024)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.positions < 1 or args.ranks < 1 or args.tail_bytes < 1:
        parser.error("positions, ranks, and tail-bytes must be positive")

    anchor_needed = args.positions * args.ranks
    anchors: deque[int] = deque(maxlen=anchor_needed)
    tail_rows = 0
    for indexes, row in iter_tail_rows(args.trace, args.tail_bytes):
        tail_rows += 1
        if row[indexes["Kernel_Name"]] == args.anchor:
            anchors.append(int(row[indexes["Start_Timestamp"]]))

    if len(anchors) != anchor_needed:
        raise SystemExit(
            f"trace tail contains {len(anchors)} {args.anchor!r} anchors; "
            f"need {anchor_needed}. Increase --tail-bytes."
        )
    threshold = min(anchors)

    kernels: dict[str, dict[str, object]] = defaultdict(
        lambda: {"calls": 0, "total_ns": 0, "min_ns": None, "max_ns": 0, "agents": set()}
    )
    shapes: dict[tuple[str, ...], dict[str, object]] = defaultdict(
        lambda: {"calls": 0, "total_ns": 0, "min_ns": None, "max_ns": 0}
    )
    agents: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "total_ns": 0})
    selected_rows = 0
    selected_anchors = 0
    min_start = None
    max_end = 0
    for indexes, row in iter_tail_rows(args.trace, args.tail_bytes):
        start = int(row[indexes["Start_Timestamp"]])
        if start < threshold:
            continue
        end = int(row[indexes["End_Timestamp"]])
        duration = end - start
        name = row[indexes["Kernel_Name"]]
        agent = row[indexes["Agent_Id"]]
        selected_rows += 1
        selected_anchors += name == args.anchor
        min_start = start if min_start is None else min(min_start, start)
        max_end = max(max_end, end)

        entry = kernels[name]
        entry["calls"] += 1
        entry["total_ns"] += duration
        entry["min_ns"] = duration if entry["min_ns"] is None else min(entry["min_ns"], duration)
        entry["max_ns"] = max(entry["max_ns"], duration)
        entry["agents"].add(agent)
        agents[agent]["calls"] += 1
        agents[agent]["total_ns"] += duration

        shape_key = (
            name,
            agent,
            row[indexes["Grid_Size_X"]],
            row[indexes["Grid_Size_Y"]],
            row[indexes["Grid_Size_Z"]],
            row[indexes["Workgroup_Size_X"]],
            row[indexes["Workgroup_Size_Y"]],
            row[indexes["Workgroup_Size_Z"]],
            row[indexes["VGPR_Count"]],
            row[indexes["SGPR_Count"]],
            row[indexes["LDS_Block_Size"]],
            row[indexes["Scratch_Size"]],
        )
        shape = shapes[shape_key]
        shape["calls"] += 1
        shape["total_ns"] += duration
        shape["min_ns"] = duration if shape["min_ns"] is None else min(shape["min_ns"], duration)
        shape["max_ns"] = max(shape["max_ns"], duration)

    if selected_anchors != anchor_needed:
        raise SystemExit(
            f"selected region contains {selected_anchors} anchors, expected {anchor_needed}; "
            "trace rows may not be timestamp ordered"
        )

    total_ns = sum(int(entry["total_ns"]) for entry in kernels.values())
    ranked = []
    for name, entry in kernels.items():
        calls = int(entry["calls"])
        kernel_total_ns = int(entry["total_ns"])
        ranked.append(
            {
                "name": name,
                "calls": calls,
                "calls_per_position": calls / args.positions,
                "total_ms": kernel_total_ns / 1e6,
                "ms_per_position_all_ranks": kernel_total_ns / args.positions / 1e6,
                "average_ns": kernel_total_ns / calls,
                "percentage_of_aggregate_gpu_time": 100.0 * kernel_total_ns / total_ns,
                "min_ns": entry["min_ns"],
                "max_ns": entry["max_ns"],
                "agents": sorted(entry["agents"]),
            }
        )
    ranked.sort(key=lambda entry: entry["total_ms"], reverse=True)

    ranked_shapes = []
    for key, entry in shapes.items():
        (
            name,
            agent,
            grid_x,
            grid_y,
            grid_z,
            workgroup_x,
            workgroup_y,
            workgroup_z,
            vgpr,
            sgpr,
            lds_bytes,
            scratch_bytes,
        ) = key
        calls = int(entry["calls"])
        shape_total_ns = int(entry["total_ns"])
        ranked_shapes.append(
            {
                "name": name,
                "agent": agent,
                "grid": [int(grid_x), int(grid_y), int(grid_z)],
                "workgroup": [int(workgroup_x), int(workgroup_y), int(workgroup_z)],
                "vgpr": int(vgpr),
                "sgpr": int(sgpr),
                "lds_bytes": int(lds_bytes),
                "scratch_bytes": int(scratch_bytes),
                "calls": calls,
                "calls_per_position": calls / args.positions,
                "total_ms": shape_total_ns / 1e6,
                "ms_per_position": shape_total_ns / args.positions / 1e6,
                "average_ns": shape_total_ns / calls,
                "percentage_of_aggregate_gpu_time": 100.0 * shape_total_ns / total_ns,
                "min_ns": entry["min_ns"],
                "max_ns": entry["max_ns"],
            }
        )
    ranked_shapes.sort(key=lambda entry: entry["total_ms"], reverse=True)

    result = {
        "trace": str(args.trace),
        "positions": args.positions,
        "ranks": args.ranks,
        "anchor": args.anchor,
        "anchor_count": selected_anchors,
        "tail_bytes": args.tail_bytes,
        "tail_rows": tail_rows,
        "selected_rows": selected_rows,
        "threshold_start_timestamp": threshold,
        "wall_span_ms": (max_end - min_start) / 1e6 if min_start is not None else 0.0,
        "aggregate_gpu_ms": total_ns / 1e6,
        "aggregate_gpu_ms_per_position": total_ns / args.positions / 1e6,
        "mean_gpu_ms_per_rank_position": total_ns / args.positions / args.ranks / 1e6,
        "agents": agents,
        "kernels": ranked,
        "shapes": ranked_shapes,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, default=sorted) + "\n")

    print(
        f"positions={args.positions} ranks={args.ranks} selected_rows={selected_rows} "
        f"wall_span={result['wall_span_ms']:.3f}ms "
        f"aggregate_gpu={result['aggregate_gpu_ms']:.3f}ms "
        f"mean_gpu/rank/position={result['mean_gpu_ms_per_rank_position']:.3f}ms"
    )
    print("%gpu   ms/pos(all ranks)  calls/pos  avg_us   kernel")
    for entry in ranked[: args.top]:
        print(
            f"{entry['percentage_of_aggregate_gpu_time']:5.2f} "
            f"{entry['ms_per_position_all_ranks']:18.4f} "
            f"{entry['calls_per_position']:10.2f} "
            f"{entry['average_ns'] / 1000:7.2f}   {entry['name']}"
        )

    print("\n%gpu   ms/pos  calls/pos  avg_us   agent grid/workgroup vgpr/sgpr lds/scratch kernel")
    for entry in ranked_shapes[: args.top]:
        print(
            f"{entry['percentage_of_aggregate_gpu_time']:5.2f} "
            f"{entry['ms_per_position']:8.4f} "
            f"{entry['calls_per_position']:10.2f} "
            f"{entry['average_ns'] / 1000:7.2f}   {entry['agent']} "
            f"{entry['grid']}/{entry['workgroup']} "
            f"{entry['vgpr']}/{entry['sgpr']} "
            f"{entry['lds_bytes']}/{entry['scratch_bytes']} {entry['name']}"
        )


if __name__ == "__main__":
    main()
