// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Unit tests for `buildInstrumentSpawnPlan`, the pure arg-forwarding core of
// `hipfire profile --instrument` (T9 CLI sub-task,
// docs/superpowers/plans/2026-07-01-gfx1201-phaseA-perf-instrument.md Task 9
// Step 5). No GPU, no daemon, no subprocess spawn — this only drives the
// argv → { binary, args } / { error } decision, mirroring the
// `parseRestartPort`/`serveTpDangling` pure-helper test pattern in
// cli/serve_ux_parse.test.ts.
//
// Run: cd cli && bun test profile_instrument.test.ts

import { test, expect, describe, beforeAll } from "bun:test";

let mod: typeof import("./index.ts");
beforeAll(async () => { mod = await import("./index.ts"); });

const ARCH = "gfx1201";

describe("buildInstrumentSpawnPlan — default (self-check)", () => {
  test("no flags → kernel_perf_instrument with just --arch <detected>", () => {
    const plan = mod.buildInstrumentSpawnPlan([], ARCH);
    expect(plan).toEqual({ binary: "kernel_perf_instrument", args: ["--arch", "gfx1201"] });
  });

  test("does not mutate the caller's argv array", () => {
    const rest = ["--diff", "--ledger", "/tmp/x.jsonl"];
    const snapshot = [...rest];
    mod.buildInstrumentSpawnPlan(rest, ARCH);
    expect(rest).toEqual(snapshot);
  });
});

describe("buildInstrumentSpawnPlan — --arch override", () => {
  test("explicit --arch wins over the detected arch", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--arch", "gfx1100"], ARCH);
    expect(plan).toEqual({ binary: "kernel_perf_instrument", args: ["--arch", "gfx1100"] });
  });
});

describe("buildInstrumentSpawnPlan — --diff", () => {
  test("--diff forwards --diff after --arch", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--diff"], ARCH);
    expect(plan).toEqual({ binary: "kernel_perf_instrument", args: ["--arch", "gfx1201", "--diff"] });
  });

  test("--diff + --current-dir + --ledger all forward", () => {
    const plan = mod.buildInstrumentSpawnPlan(
      ["--diff", "--current-dir", "/tmp/cur", "--ledger", "/tmp/ledger.jsonl"],
      ARCH,
    );
    expect(plan).toEqual({
      binary: "kernel_perf_instrument",
      args: ["--arch", "gfx1201", "--diff", "--current-dir", "/tmp/cur", "--ledger", "/tmp/ledger.jsonl"],
    });
  });
});

describe("buildInstrumentSpawnPlan — --dynamic", () => {
  test("--dynamic without --rocprof-csv is a usage error (never invokes rocprofv3 itself)", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--dynamic"], ARCH);
    expect("error" in plan).toBe(true);
    if ("error" in plan) expect(plan.error).toContain("--rocprof-csv");
  });

  test("--dynamic --rocprof-csv <path> forwards both, --diff is irrelevant/ignored", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--dynamic", "--rocprof-csv", "/tmp/a3b.csv"], ARCH);
    expect(plan).toEqual({
      binary: "kernel_perf_instrument",
      args: ["--arch", "gfx1201", "--dynamic", "--rocprof-csv", "/tmp/a3b.csv"],
    });
  });

  test("--dynamic + --diff both present: dynamic wins (--diff not appended)", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--dynamic", "--rocprof-csv", "/tmp/a3b.csv", "--diff"], ARCH);
    expect(plan).toEqual({
      binary: "kernel_perf_instrument",
      args: ["--arch", "gfx1201", "--dynamic", "--rocprof-csv", "/tmp/a3b.csv"],
    });
  });

  test("--rocprof-csv=<path> inline form works too", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--dynamic", "--rocprof-csv=/tmp/inline.csv"], ARCH);
    expect(plan).toEqual({
      binary: "kernel_perf_instrument",
      args: ["--arch", "gfx1201", "--dynamic", "--rocprof-csv", "/tmp/inline.csv"],
    });
  });
});

describe("buildInstrumentSpawnPlan — --chip", () => {
  test("--chip spawns dump_chip_profile instead, with only --arch forwarded", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--chip"], ARCH);
    expect(plan).toEqual({ binary: "dump_chip_profile", args: ["--arch", "gfx1201"] });
  });

  test("--chip --arch gfx1100 forwards the override", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--chip", "--arch", "gfx1100"], ARCH);
    expect(plan).toEqual({ binary: "dump_chip_profile", args: ["--arch", "gfx1100"] });
  });

  test("--chip ignores --diff/--dynamic/--fixtures-dir (dump_chip_profile takes none of them)", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--chip", "--diff", "--fixtures-dir", "/tmp/f"], ARCH);
    expect(plan).toEqual({ binary: "dump_chip_profile", args: ["--arch", "gfx1201"] });
  });
});

describe("buildInstrumentSpawnPlan — dangling value flags", () => {
  test("trailing --rocprof-csv with nothing after it is an error, not a crash", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--rocprof-csv"], ARCH);
    expect("error" in plan).toBe(true);
  });

  test("--fixtures-dir followed by another flag (not a value) is an error", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--fixtures-dir", "--diff"], ARCH);
    expect("error" in plan).toBe(true);
  });

  test("dangling --arch is an error", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--arch"], ARCH);
    expect("error" in plan).toBe(true);
  });
});

describe("buildInstrumentSpawnPlan — --fixtures-dir (self-check mode)", () => {
  test("--fixtures-dir forwards under plain self-check (no --diff)", () => {
    const plan = mod.buildInstrumentSpawnPlan(["--fixtures-dir", "/tmp/fx"], ARCH);
    expect(plan).toEqual({
      binary: "kernel_perf_instrument",
      args: ["--arch", "gfx1201", "--fixtures-dir", "/tmp/fx"],
    });
  });
});
