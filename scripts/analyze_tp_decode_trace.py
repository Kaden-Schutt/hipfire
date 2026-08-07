#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

"""Isolate the decode tail of a rocprofv3 kernel/RCCL trace.

The DeepSeek finite profiler currently runs prefill through decode_step, so
whole-run kernel statistics are prefill dominated. The output projection is
one fixed-grid launch per forward. The interval after the (N+1)th-from-last
projection through the final projection therefore contains exactly N decode
forwards without assuming that rocprof emitted rows in timestamp order.
"""

import argparse
import csv
import json
from collections import defaultdict


MARKER_NAME = "gemv_mfp4g32_e8_soa_gfx1201"
MARKER_GRID_X = 4_136_960


def rows(path):
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = {name: index for index, name in enumerate(header)}
        for row in reader:
            yield row, columns


def merge_intervals(intervals):
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_trace")
    parser.add_argument("rccl_trace")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    marker_ends = []
    for row, columns in rows(args.kernel_trace):
        if (
            row[columns["Kernel_Name"]] == MARKER_NAME
            and int(row[columns["Grid_Size_X"]]) == MARKER_GRID_X
        ):
            marker_ends.append(int(row[columns["End_Timestamp"]]))

    marker_ends.sort()
    needed = args.steps + 1
    if len(marker_ends) < needed:
        raise RuntimeError(
            f"need {needed} output markers, found {len(marker_ends)}; "
            "verify the model shape and marker grid"
        )

    window_start = marker_ends[-needed]
    window_end = marker_ends[-1]
    wall_ns = window_end - window_start
    kernels = defaultdict(lambda: [0, 0])
    agent_intervals = defaultdict(list)
    queue_intervals = defaultdict(list)
    all_intervals = []
    dispatches = 0

    for row, columns in rows(args.kernel_trace):
        start = int(row[columns["Start_Timestamp"]])
        end = int(row[columns["End_Timestamp"]])
        if end <= window_start or start > window_end:
            continue
        clipped_start = max(start, window_start)
        clipped_end = min(end, window_end)
        if clipped_end <= clipped_start:
            continue

        dispatches += 1
        duration = clipped_end - clipped_start
        name = row[columns["Kernel_Name"]]
        agent = row[columns["Agent_Id"]]
        queue = f"{agent}/q{row[columns['Queue_Id']]}"
        kernels[name][0] += 1
        kernels[name][1] += duration
        interval = (clipped_start, clipped_end)
        agent_intervals[agent].append(interval)
        queue_intervals[queue].append(interval)
        all_intervals.append(interval)

    rccl = defaultdict(lambda: [0, 0])
    for row, columns in rows(args.rccl_trace):
        start = int(row[columns["Start_Timestamp"]])
        end = int(row[columns["End_Timestamp"]])
        if end <= window_start or start > window_end:
            continue
        name = row[columns["Function"]]
        rccl[name][0] += 1
        rccl[name][1] += min(end, window_end) - max(start, window_start)

    kernel_rows = []
    for name, (calls, total_ns) in kernels.items():
        kernel_rows.append(
            {
                "name": name,
                "calls": calls,
                "calls_per_step": calls / args.steps,
                "total_ms": total_ns / 1e6,
                "avg_us": total_ns / calls / 1e3,
                "share_of_four_gpu_capacity_pct": 100 * total_ns / (wall_ns * 4),
            }
        )
    kernel_rows.sort(key=lambda row: row["total_ms"], reverse=True)

    agent_rows = []
    for agent, intervals in agent_intervals.items():
        busy_ns = merge_intervals(intervals)
        agent_rows.append(
            {
                "agent": agent,
                "busy_ms": busy_ns / 1e6,
                "utilization_pct": 100 * busy_ns / wall_ns,
            }
        )
    agent_rows.sort(key=lambda row: row["agent"])

    queue_rows = []
    for queue, intervals in queue_intervals.items():
        busy_ns = merge_intervals(intervals)
        queue_rows.append(
            {
                "queue": queue,
                "busy_ms": busy_ns / 1e6,
                "utilization_pct": 100 * busy_ns / wall_ns,
            }
        )
    queue_rows.sort(key=lambda row: row["busy_ms"], reverse=True)

    rccl_rows = []
    for name, (calls, total_ns) in rccl.items():
        rccl_rows.append(
            {
                "name": name,
                "calls": calls,
                "calls_per_step": calls / args.steps,
                "total_host_ms": total_ns / 1e6,
                "avg_host_us": total_ns / calls / 1e3,
            }
        )
    rccl_rows.sort(key=lambda row: row["total_host_ms"], reverse=True)

    any_gpu_busy_ns = merge_intervals(all_intervals)
    report = {
        "schema": "hipfire-tp-decode-trace-v1",
        "inputs": {
            "kernel_trace": args.kernel_trace,
            "rccl_trace": args.rccl_trace,
        },
        "boundary": {
            "marker_name": MARKER_NAME,
            "marker_grid_x": MARKER_GRID_X,
            "markers_found": len(marker_ends),
            "decode_steps": args.steps,
            "start_timestamp_ns": window_start,
            "end_timestamp_ns": window_end,
            "wall_ms": wall_ns / 1e6,
            "profiled_tok_s": args.steps * 1e9 / wall_ns,
        },
        "dispatches": dispatches,
        "dispatches_per_step": dispatches / args.steps,
        "any_gpu_busy_ms": any_gpu_busy_ns / 1e6,
        "any_gpu_busy_pct": 100 * any_gpu_busy_ns / wall_ns,
        "agents": agent_rows,
        "queues": queue_rows,
        "kernels": kernel_rows,
        "rccl_api": rccl_rows,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
