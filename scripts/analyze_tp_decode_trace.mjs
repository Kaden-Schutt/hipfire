#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

/**
 * Isolate the decode tail of a rocprofv3 kernel/RCCL trace.
 *
 * The DeepSeek finite profiler currently runs prefill through decode_step, so
 * whole-run kernel statistics are prefill dominated. The output projection is
 * one fixed-grid launch per forward. The interval after the (N+1)th-from-last
 * projection through the final projection therefore contains exactly N decode
 * forwards without assuming that rocprof emitted rows in timestamp order.
 */

import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";

function usage() {
  console.error(
    "usage: analyze_tp_decode_trace.mjs KERNEL_TRACE RCCL_TRACE [STEPS]",
  );
  process.exit(2);
}

const [, , kernelPath, rcclPath, stepsArg = "32"] = process.argv;
if (!kernelPath || !rcclPath) usage();

const steps = Number.parseInt(stepsArg, 10);
if (!Number.isSafeInteger(steps) || steps <= 0) usage();

function parseCsv(line) {
  const fields = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      fields.push(field);
      field = "";
    } else {
      field += char;
    }
  }
  fields.push(field);
  return fields;
}

async function forEachCsv(path, callback) {
  const input = createReadStream(path, { highWaterMark: 4 * 1024 * 1024 });
  const lines = createInterface({ input, crlfDelay: Infinity });
  let header;
  let columns;
  for await (const line of lines) {
    if (!line) continue;
    if (!header) {
      header = parseCsv(line);
      columns = new Map(header.map((name, index) => [name, index]));
      continue;
    }
    await callback(parseCsv(line), columns);
  }
}

function numeric(row, columns, name) {
  return Number(row[columns.get(name)]);
}

function mergeIntervals(intervals) {
  if (intervals.length === 0) return 0;
  intervals.sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  let total = 0;
  let [start, end] = intervals[0];
  for (let index = 1; index < intervals.length; index += 1) {
    const [nextStart, nextEnd] = intervals[index];
    if (nextStart <= end) {
      end = Math.max(end, nextEnd);
    } else {
      total += end - start;
      start = nextStart;
      end = nextEnd;
    }
  }
  return total + end - start;
}

// This fixed grid is the 129,280 x 4,096 E8 LM head in the MQ2R P3 route.
const markerName = "gemv_mfp4g32_e8_soa_gfx1201";
const markerGridX = 4_136_960;
const markerEnds = [];

await forEachCsv(kernelPath, (row, columns) => {
  if (
    row[columns.get("Kernel_Name")] === markerName &&
    numeric(row, columns, "Grid_Size_X") === markerGridX
  ) {
    markerEnds.push(numeric(row, columns, "End_Timestamp"));
  }
});

markerEnds.sort((left, right) => left - right);
if (markerEnds.length < steps + 1) {
  throw new Error(
    `need ${steps + 1} output markers, found ${markerEnds.length}; ` +
      "verify the model shape and marker grid",
  );
}

const windowStart = markerEnds.at(-(steps + 1));
const windowEnd = markerEnds.at(-1);
const wallNs = windowEnd - windowStart;
const kernels = new Map();
const agentIntervals = new Map();
const queueIntervals = new Map();
const allIntervals = [];
let dispatches = 0;

await forEachCsv(kernelPath, (row, columns) => {
  const start = numeric(row, columns, "Start_Timestamp");
  const end = numeric(row, columns, "End_Timestamp");
  if (end <= windowStart || start > windowEnd) return;

  const clippedStart = Math.max(start, windowStart);
  const clippedEnd = Math.min(end, windowEnd);
  if (clippedEnd <= clippedStart) return;

  dispatches += 1;
  const duration = clippedEnd - clippedStart;
  const name = row[columns.get("Kernel_Name")];
  const agent = row[columns.get("Agent_Id")];
  const queue = `${agent}/q${row[columns.get("Queue_Id")]}`;

  const kernel = kernels.get(name) ?? { calls: 0, total_ns: 0 };
  kernel.calls += 1;
  kernel.total_ns += duration;
  kernels.set(name, kernel);

  if (!agentIntervals.has(agent)) agentIntervals.set(agent, []);
  agentIntervals.get(agent).push([clippedStart, clippedEnd]);
  if (!queueIntervals.has(queue)) queueIntervals.set(queue, []);
  queueIntervals.get(queue).push([clippedStart, clippedEnd]);
  allIntervals.push([clippedStart, clippedEnd]);
});

const rccl = new Map();
await forEachCsv(rcclPath, (row, columns) => {
  const start = numeric(row, columns, "Start_Timestamp");
  const end = numeric(row, columns, "End_Timestamp");
  if (end <= windowStart || start > windowEnd) return;
  const name = row[columns.get("Function")];
  const call = rccl.get(name) ?? { calls: 0, total_ns: 0 };
  call.calls += 1;
  call.total_ns += Math.min(end, windowEnd) - Math.max(start, windowStart);
  rccl.set(name, call);
});

const kernelRows = [...kernels.entries()]
  .map(([name, data]) => ({
    name,
    calls: data.calls,
    calls_per_step: data.calls / steps,
    total_ms: data.total_ns / 1e6,
    avg_us: data.total_ns / data.calls / 1e3,
    share_of_four_gpu_capacity_pct: (100 * data.total_ns) / (wallNs * 4),
  }))
  .sort((left, right) => right.total_ms - left.total_ms);

const agentRows = [...agentIntervals.entries()]
  .map(([agent, intervals]) => ({
    agent,
    busy_ms: mergeIntervals(intervals) / 1e6,
    utilization_pct: (100 * mergeIntervals(intervals)) / wallNs,
  }))
  .sort((left, right) => left.agent.localeCompare(right.agent));

const queueRows = [...queueIntervals.entries()]
  .map(([queue, intervals]) => ({
    queue,
    busy_ms: mergeIntervals(intervals) / 1e6,
    utilization_pct: (100 * mergeIntervals(intervals)) / wallNs,
  }))
  .sort((left, right) => right.busy_ms - left.busy_ms);

const rcclRows = [...rccl.entries()]
  .map(([name, data]) => ({
    name,
    calls: data.calls,
    calls_per_step: data.calls / steps,
    total_host_ms: data.total_ns / 1e6,
    avg_host_us: data.total_ns / data.calls / 1e3,
  }))
  .sort((left, right) => right.total_host_ms - left.total_host_ms);

const report = {
  schema: "hipfire-tp-decode-trace-v1",
  inputs: { kernel_trace: kernelPath, rccl_trace: rcclPath },
  boundary: {
    marker_name: markerName,
    marker_grid_x: markerGridX,
    markers_found: markerEnds.length,
    decode_steps: steps,
    start_timestamp_ns: windowStart,
    end_timestamp_ns: windowEnd,
    wall_ms: wallNs / 1e6,
    profiled_tok_s: (steps * 1e9) / wallNs,
  },
  dispatches,
  dispatches_per_step: dispatches / steps,
  any_gpu_busy_ms: mergeIntervals(allIntervals) / 1e6,
  any_gpu_busy_pct: (100 * mergeIntervals(allIntervals)) / wallNs,
  agents: agentRows,
  queues: queueRows,
  kernels: kernelRows,
  rccl_api: rcclRows,
};

console.log(JSON.stringify(report, null, 2));
