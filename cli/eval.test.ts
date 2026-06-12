import { afterEach, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import {
  EVAL_HELP,
  buildEvalCommand,
  isEvalHelp,
  localSpeedBaselinePath,
  normalizeEvalArgs,
  resolveEvalModel,
  resolveEvalBinary,
  upsertLocalSpeedBaseline,
  writeLocalSpeedBaseline,
} from "./eval";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length) {
    rmSync(tempDirs.pop()!, { recursive: true, force: true });
  }
});

function fakeBin(name = "hipfire-eval"): string {
  const dir = mkdtempSync(join(tmpdir(), "hipfire-eval-test-"));
  tempDirs.push(dir);
  const path = join(dir, name);
  writeFileSync(path, "#!/bin/sh\nexit 0\n");
  return path;
}

describe("hipfire eval CLI delegation", () => {
  test("resolves HIPFIRE_EVAL_BIN first", () => {
    const bin = fakeBin();
    expect(resolveEvalBinary({ HIPFIRE_EVAL_BIN: bin }, "linux", "/missing")).toBe(bin);
  });

  test("returns null when no candidate exists", () => {
    const dir = mkdtempSync(join(tmpdir(), "hipfire-eval-empty-home-"));
    tempDirs.push(dir);
    expect(resolveEvalBinary({ HIPFIRE_DIR: dir }, "linux", "/missing")).toBeNull();
  });

  test("passes args through unchanged", () => {
    const bin = fakeBin();
    const args = ["--model", "m.hfq", "--tier", "medium", "--suite", "gpqa"];
    expect(buildEvalCommand(args, { HIPFIRE_EVAL_BIN: bin }, "linux", "/missing")).toEqual({
      bin,
      args,
    });
    expect(buildEvalCommand(["--version"], { HIPFIRE_EVAL_BIN: bin }, "linux", "/missing")).toEqual(
      {
        bin,
        args: ["--version"],
      },
    );
  });

  test("normalizes human eval forms", () => {
    expect(normalizeEvalArgs(["/models/qwen3.5-4b-mq4.hfq"])).toEqual([
      "--model",
      "/models/qwen3.5-4b-mq4.hfq",
      "--battery",
      "speed",
      "--force",
    ]);
    expect(normalizeEvalArgs(["speed", "m.hfq", "--benchmark"])).toEqual([
      "--model",
      "m.hfq",
      "--battery",
      "speed",
      "--force",
      "--benchmark",
    ]);
    expect(normalizeEvalArgs(["smoke", "m.hfq"])).toEqual([
      "--model",
      "m.hfq",
      "--tier",
      "fast",
    ]);
    expect(normalizeEvalArgs(["runtime", "m.hfq"])).toEqual([
      "--model",
      "m.hfq",
      "--battery",
      "runtime",
    ]);
    expect(normalizeEvalArgs(["dflash", "target.hfq", "--draft", "draft.hfq"])).toEqual([
      "--model",
      "target.hfq",
      "--battery",
      "dflash",
      "--dflash",
      "on",
      "--draft",
      "draft.hfq",
    ]);
    expect(normalizeEvalArgs(["admit", "candidate.hfq", "baseline.hfq"])).toEqual([
      "--model",
      "candidate.hfq",
      "--baseline",
      "baseline.hfq",
      "--battery",
      "quality,speed,barrage",
      "--fail-on-admission",
    ]);
    expect(normalizeEvalArgs(["raw", "--model", "m.hfq", "--battery", "quality"])).toEqual([
      "--model",
      "m.hfq",
      "--battery",
      "quality",
    ]);
  });

  test("writes local speed baseline from speed result rows", () => {
    const dir = mkdtempSync(join(tmpdir(), "hipfire-eval-results-"));
    tempDirs.push(dir);
    writeFileSync(
      join(dir, "results.jsonl"),
      [
        JSON.stringify({
          battery: "speed",
          status: "pass",
          case_id: "pp32_prefill_decode",
          model: "/models/qwen3.5-4b-mq4.hfq",
          metrics: {
            prefill_tokens: 32,
            prefill_tok_s: 600,
            gen_tok_s: 70,
          },
        }),
        JSON.stringify({
          battery: "speed",
          status: "pass",
          case_id: "pp128_prefill_decode",
          model: "/models/qwen3.5-4b-mq4.hfq",
          metrics: {
            prefill_tokens: 128,
            prefill_tok_s: 1300,
          },
        }),
      ].join("\n") + "\n",
    );
    const baseline = join(dir, "baseline.json");
    writeLocalSpeedBaseline("/models/qwen3.5-4b-mq4.hfq", dir, baseline);
    expect(existsSync(baseline)).toBe(true);
    const value = JSON.parse(readFileSync(baseline, "utf8"));
    expect(value.schema).toBe("hipfire.perf_baseline.v1");
    expect(value.model.id).toBe("qwen3.5-4b-mq4");
    expect(value.baselines.speed).toHaveLength(2);
    expect(value.baselines.speed[0].model_id).toBe("qwen3.5-4b-mq4");
    expect(value.baselines.speed[0].model_size).toBe("4b");
    expect(value.baselines.speed[0].prefill_tok_s).toBe(600);
  });

  test("adds missing local speed baseline rows without duplicating existing rows", () => {
    const dir = mkdtempSync(join(tmpdir(), "hipfire-eval-results-"));
    tempDirs.push(dir);
    const baseline = join(dir, "baseline.json");
    writeFileSync(
      baseline,
      JSON.stringify(
        {
          schema: "hipfire.perf_baseline.v1",
          baselines: {
            speed: [
              {
                label: "pp32_prefill_decode",
                model_id: "qwen3.5-4b-mq4",
                model_size: "4b",
                format: "mq4",
                prefill_tokens: 32,
                prefill_tok_s: 600,
                gen_tok_s: 70,
              },
            ],
          },
        },
        null,
        2,
      ) + "\n",
    );
    writeFileSync(
      join(dir, "results.jsonl"),
      [
        JSON.stringify({
          battery: "speed",
          status: "pass",
          case_id: "pp32_prefill_decode",
          model: "/models/qwen3.5-4b-mq4.hfq",
          metrics: {
            prefill_tokens: 32,
            prefill_tok_s: 610,
            gen_tok_s: 71,
          },
        }),
        JSON.stringify({
          battery: "speed",
          status: "pass",
          case_id: "pp128_prefill_decode",
          model: "/models/qwen3.5-4b-mq4.hfq",
          metrics: {
            prefill_tokens: 128,
            prefill_tok_s: 1300,
            gen_tok_s: 69,
          },
        }),
      ].join("\n") + "\n",
    );

    expect(upsertLocalSpeedBaseline("/models/qwen3.5-4b-mq4.hfq", dir, baseline)).toEqual({
      created: false,
      added: 1,
    });
    const value = JSON.parse(readFileSync(baseline, "utf8"));
    expect(value.baselines.speed).toHaveLength(2);
    expect(value.baselines.speed.map((row: any) => row.prefill_tokens)).toEqual([32, 128]);
  });

  test("handles help without requiring a Rust binary", () => {
    expect(isEvalHelp([])).toBe(true);
    expect(isEvalHelp(["--help"])).toBe(true);
    expect(isEvalHelp(["-h"])).toBe(true);
    expect(isEvalHelp(["--model", "m.hfq"])).toBe(false);
    expect(EVAL_HELP).toContain("hipfire eval <model>");
    expect(EVAL_HELP).toContain("hipfire eval admit <candidate>");
    expect(EVAL_HELP).toContain("~/.hipfire/benchmarks/");
    expect(EVAL_HELP).toContain("--version");
    expect(EVAL_HELP).toContain("--executor auto|none|examples|direct|mock");
    expect(EVAL_HELP).toContain("--runs <N>");
    expect(EVAL_HELP).toContain("--warmup-runs <N>");
    expect(EVAL_HELP).toContain("--benchmark");
    expect(EVAL_HELP).toContain("--host-memory-class <s>");
    expect(EVAL_HELP).toContain("--host-memory-width-bits <N>");
    expect(EVAL_HELP).toContain("--host-memory-bandwidth-gbps <N>");
    expect(EVAL_HELP).toContain("--evidence-dir <dir>");
    expect(EVAL_HELP).toContain("--result-cache <dir>");
    expect(EVAL_HELP).toContain("--force");
    expect(EVAL_HELP).toContain("--regenerate");
    expect(EVAL_HELP).toContain("--no-cache");
    expect(EVAL_HELP).toContain("--fail-on-admission");
    expect(EVAL_HELP).toContain("cargo build --release -p hipfire-runtime --bin hipfire-eval");
  });

  test("local speed baseline path uses hfq stem", () => {
    expect(localSpeedBaselinePath("/models/qwen3.5-4b-mq4.hfq")).toContain(
      "qwen3.5-4b-mq4.speed.json",
    );
  });

  test("leaves unresolved model names unchanged for registry tags or missing files", () => {
    expect(resolveEvalModel("qwen3.5:4b")).toBe("qwen3.5:4b");
    expect(resolveEvalModel("missing.hfq")).toBe("missing.hfq");
  });
});
