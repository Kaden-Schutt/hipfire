import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { EVAL_HELP, buildEvalCommand, isEvalHelp, resolveEvalBinary } from "./eval";

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
    expect(resolveEvalBinary({}, "linux", "/missing")).toBeNull();
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

  test("handles help without requiring a Rust binary", () => {
    expect(isEvalHelp([])).toBe(true);
    expect(isEvalHelp(["--help"])).toBe(true);
    expect(isEvalHelp(["-h"])).toBe(true);
    expect(isEvalHelp(["--model", "m.hfq"])).toBe(false);
    expect(EVAL_HELP).toContain("hipfire eval --model <model>");
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
});
