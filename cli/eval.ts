import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { join, resolve, basename } from "path";
import { homedir } from "os";

export const EVAL_HELP = `hipfire eval — quant admission/model evaluation harness

Usage:
  hipfire eval <model>                 Short speed bench + local baseline check
  hipfire eval speed <model>           Same as the default form
  hipfire eval smoke <model>           Fast smoke tier
  hipfire eval admit <candidate> [baseline]
  hipfire eval dflash <target> --draft <draft>
  hipfire eval runtime <model>         Server/runtime admission rows
  hipfire eval quality <model>         Quality battery
  hipfire eval raw --model <model> ... Pass flags directly to hipfire-eval

The default/speed form stores first-run local baselines under
~/.hipfire/benchmarks/ and compares later runs against that file.

Common options:
  --version              Print hipfire-eval version/git metadata
  --tier fast|medium|long|extensive
                          Use a preconfigured battery set
  --battery <a,b>         Low-level battery selection
  --suite <a,b>           Low-level suite selection
  --out <dir>             Write manifest/results/summary/artifacts there
  --baseline <model>      Compare candidate metrics against a quant baseline
  --reference <model>     Compare candidate metrics against a reference model
  --quality-json <path>   Ingest kld_reduce.py result-data.json for quality
  --performance-json <path>
                          Ingest benchmark JSON for speed/perf metrics
  --evidence-json <path>  Ingest profiler/runtime evidence JSON, repeatable
  --evidence-dir <dir>    Ingest standard runtime evidence JSON files from a directory
  --draft <draft>         DFlash draft model for dflash battery
  --profile off|passive   Profiling mode, default off
                          passive captures evidence after scored rows, including rocprofv3 when available
  --runs <N>              Repeat each scored battery N times, default 1
  --warmup-runs <N>       Run and discard N warmup battery passes before scored repeats
  --benchmark             Shorthand for --runs 5 unless --runs is provided; emits aggregate rows
  --host-memory-class <s> Override uncertain host memory class, e.g. gddr6 or lpddr5x
  --host-memory-width-bits <N>
                          Override uncertain memory bus width/channel width
  --host-memory-bandwidth-gbps <N>
                          Override computed peak memory bandwidth
  --executor auto|none|examples|direct|mock
                          Execution backend, default auto uses Hipfire examples when built
  --result-cache <dir>    Result cache root, default ~/.hipfire/eval-results/cache
  --force                 Ignore cache hits for this run, but write new cache entries
  --regenerate            Delete and replace matching cache entries before running
  --no-cache              Disable result cache reads and writes
  --fail-on-admission     Exit non-zero unless admission verdict is promote

Build runner:
  cargo build --release -p hipfire-runtime --bin hipfire-eval`;

export function isEvalHelp(args: string[]): boolean {
  return args.length === 0 || args.some((a) => a === "-h" || a === "--help");
}

export function resolveEvalBinary(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
  dirname: string = __dirname,
): string | null {
  const exe = platform === "win32" ? ".exe" : "";
  const hipfireDir = env.HIPFIRE_DIR ?? join(homedir(), ".hipfire");
  const candidates = [
    env.HIPFIRE_EVAL_BIN,
    resolve(dirname, `../target/release/hipfire-eval${exe}`),
    join(hipfireDir, "bin", `hipfire-eval${exe}`),
  ].filter((p): p is string => !!p);
  return candidates.find((p) => existsSync(p)) ?? null;
}

export function buildEvalCommand(
  args: string[],
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
  dirname: string = __dirname,
): { bin: string; args: string[] } | null {
  const bin = resolveEvalBinary(env, platform, dirname);
  return bin ? { bin, args: normalizeEvalArgs(args) } : null;
}

export async function runEvalCommand(args: string[]): Promise<void> {
  if (isEvalHelp(args)) {
    console.log(EVAL_HELP);
    return;
  }

  let cmd: { bin: string; args: string[] } | null;
  try {
    cmd = buildEvalCommand(args);
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    console.error("Run `hipfire eval --help` for usage.");
    process.exit(2);
  }
  if (!cmd) {
    console.error("hipfire-eval not found.");
    console.error("Build it with: cargo build --release -p hipfire-runtime --bin hipfire-eval");
    process.exit(1);
  }

  const defaultSpeed = defaultSpeedModel(args);
  if (defaultSpeed) {
    const code = await runDefaultSpeedEval(cmd.bin, cmd.args, defaultSpeed);
    process.exit(code);
  }

  const proc = Bun.spawn([cmd.bin, ...cmd.args], {
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  });
  const code = await proc.exited;
  process.exit(code);
}

const HUMAN_SUBCOMMANDS = new Set(["speed", "smoke", "admit", "dflash", "runtime", "quality", "raw"]);

export function normalizeEvalArgs(args: string[]): string[] {
  if (args.length === 0 || args[0]?.startsWith("-")) return [...args];
  const [first, ...rest] = args;
  if (!HUMAN_SUBCOMMANDS.has(first)) {
    return defaultSpeedArgs(first, rest);
  }
  switch (first) {
    case "raw":
      return [...rest];
    case "speed": {
      const { model, rest: tail } = takeModel("speed", rest);
      return defaultSpeedArgs(model, tail);
    }
    case "smoke": {
      const { model, rest: tail } = takeModel("smoke", rest);
      return ["--model", model, "--tier", "fast", ...tail];
    }
    case "runtime": {
      const { model, rest: tail } = takeModel("runtime", rest);
      return ["--model", model, "--battery", "runtime", ...tail];
    }
    case "quality": {
      const { model, rest: tail } = takeModel("quality", rest);
      return ["--model", model, "--battery", "quality", ...tail];
    }
    case "dflash": {
      const { model, rest: tail } = takeModel("dflash", rest);
      return ["--model", model, "--battery", "dflash", "--dflash", "on", ...tail];
    }
    case "admit": {
      const { model, rest: tail } = takeModel("admit", rest);
      const { baseline, rest: admitTail } = takeOptionalPositionalBaseline(tail);
      const baselineArgs = baseline ? ["--baseline", baseline] : [];
      return [
        "--model",
        model,
        ...baselineArgs,
        "--battery",
        "quality,speed,barrage",
        "--fail-on-admission",
        ...admitTail,
      ];
    }
    default:
      return [...args];
  }
}

function takeModel(subcommand: string, args: string[]): { model: string; rest: string[] } {
  const [model, ...rest] = args;
  if (!model || model.startsWith("-")) {
    throw new Error(`hipfire eval ${subcommand} requires a model argument`);
  }
  return { model: resolveEvalModel(model), rest };
}

function takeOptionalPositionalBaseline(args: string[]): { baseline: string | null; rest: string[] } {
  if (args.length === 0 || args[0].startsWith("-") || args.includes("--baseline")) {
    return { baseline: null, rest: args };
  }
  const [baseline, ...rest] = args;
  return { baseline: resolveEvalModel(baseline), rest };
}

function defaultSpeedArgs(model: string, rest: string[]): string[] {
  if (rest.includes("--battery")) return ["--model", model, ...rest];
  const cacheArgs = hasCacheControl(rest) ? [] : ["--force"];
  return ["--model", resolveEvalModel(model), "--battery", "speed", ...cacheArgs, ...rest];
}

function hasCacheControl(args: string[]): boolean {
  return args.some((arg) => arg === "--force" || arg === "--regenerate" || arg === "--no-cache");
}

export function resolveEvalModel(model: string): string {
  if (existsSync(model)) return model;
  if (model.endsWith(".hfq") && !model.includes("/")) {
    const installed = join(homedir(), ".hipfire", "models", model);
    if (existsSync(installed)) return installed;
  }
  return model;
}

function defaultSpeedModel(originalArgs: string[]): string | null {
  if (originalArgs.length === 0 || originalArgs[0]?.startsWith("-")) return null;
  if (originalArgs[0] === "speed") return originalArgs[1] && !originalArgs[1].startsWith("-") ? originalArgs[1] : null;
  if (!HUMAN_SUBCOMMANDS.has(originalArgs[0])) return originalArgs[0];
  return null;
}

function localBenchmarkDir(): string {
  return join(homedir(), ".hipfire", "benchmarks");
}

export function localSpeedBaselinePath(model: string): string {
  const stem = basename(model).replace(/\.hfq$/i, "").replace(/[^a-zA-Z0-9._+-]+/g, "_");
  return join(localBenchmarkDir(), `${stem}.speed.json`);
}

async function runDefaultSpeedEval(bin: string, args: string[], model: string): Promise<number> {
  mkdirSync(localBenchmarkDir(), { recursive: true });
  const runArgs = ensureOutputDir(args, model);
  const baseline = localSpeedBaselinePath(model);
  const env = { ...process.env };
  if (existsSync(baseline)) {
    env.HIPFIRE_PERF_BASELINE = baseline;
    delete env.HIPFIRE_REQUIRE_PERF_BASELINE;
    console.error(`hipfire eval: using local speed baseline ${baseline}`);
  } else {
    env.HIPFIRE_PERF_BASELINE_DIR = localBenchmarkDir();
    delete env.HIPFIRE_REQUIRE_PERF_BASELINE;
    console.error(`hipfire eval: no local speed baseline for ${model}; capturing ${baseline}`);
  }

  const proc = Bun.spawn([bin, ...runArgs], {
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
    env,
  });
  const code = await proc.exited;
  if (code === 0) {
    const out = outputDirFromArgs(runArgs);
    try {
      const update = upsertLocalSpeedBaseline(model, out!, baseline);
      if (update.created) {
        console.error(`hipfire eval: wrote local speed baseline ${baseline}`);
      } else if (update.added > 0) {
        console.error(`hipfire eval: added ${update.added} missing speed baseline row(s) to ${baseline}`);
      }
      if (speedBaselineFailed(out!)) {
        console.error("hipfire eval: speed baseline comparison failed");
        return 1;
      }
    } catch (err) {
      console.error(`hipfire eval: failed to update local speed baseline: ${err}`);
      return 1;
    }
  }
  return code;
}

function ensureOutputDir(args: string[], model: string): string[] {
  if (outputDirFromArgs(args)) return args;
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
  const stem = basename(model).replace(/\.hfq$/i, "").replace(/[^a-zA-Z0-9._+-]+/g, "_");
  const out = join(homedir(), ".hipfire", "eval-results", "runs", `${stamp}-${stem}-speed`);
  return [...args, "--out", out];
}

function outputDirFromArgs(args: string[]): string | null {
  const idx = args.indexOf("--out");
  if (idx >= 0 && args[idx + 1]) return args[idx + 1];
  return null;
}

type SpeedBaselineRow = {
  label: string;
  model_id: string;
  model_size: string | null;
  format: string;
  prefill_tokens: number | null;
  prefill_tok_s?: number;
  gen_tok_s?: number;
};

export function writeLocalSpeedBaseline(model: string, outDir: string, baselinePath: string): void {
  const rows = collectSpeedBaselineRows(model, outDir);
  if (rows.length === 0) {
    throw new Error("speed run did not produce pass rows with benchmark metrics");
  }

  const value = newLocalSpeedBaseline(model, rows);
  writeFileSync(baselinePath, JSON.stringify(value, null, 2) + "\n");
}

export function upsertLocalSpeedBaseline(
  model: string,
  outDir: string,
  baselinePath: string,
): { created: boolean; added: number } {
  if (!existsSync(baselinePath)) {
    writeLocalSpeedBaseline(model, outDir, baselinePath);
    return { created: true, added: collectSpeedBaselineRows(model, outDir).length };
  }

  const rows = collectSpeedBaselineRows(model, outDir);
  if (rows.length === 0) {
    return { created: false, added: 0 };
  }

  const value = JSON.parse(readFileSync(baselinePath, "utf8"));
  value.baselines ??= {};
  value.baselines.speed ??= [];
  const existing = new Set(
    value.baselines.speed.map((row: SpeedBaselineRow) => baselineRowKey(row)),
  );
  const missing = rows.filter((row) => !existing.has(baselineRowKey(row)));
  if (missing.length === 0) {
    return { created: false, added: 0 };
  }
  value.baselines.speed.push(...missing);
  value.updated_on = new Date().toISOString();
  value.model ??= {
    id: inferModelId(model),
    name: basename(model),
  };
  writeFileSync(baselinePath, JSON.stringify(value, null, 2) + "\n");
  return { created: false, added: missing.length };
}

function collectSpeedBaselineRows(model: string, outDir: string): SpeedBaselineRow[] {
  return readFileSync(join(outDir, "results.jsonl"), "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((row) => row.battery === "speed" && row.status === "pass")
    .map((row): SpeedBaselineRow => ({
      label: row.case_id,
      model_id: inferModelId(row.model ?? model),
      model_size: inferModelSize(row.model ?? model),
      format: inferModelFormat(row.model ?? model),
      prefill_tokens: numberMetric(row.metrics?.prefill_tokens),
      prefill_tok_s: numberMetric(row.metrics?.prefill_tok_s) ?? undefined,
      gen_tok_s: numberMetric(row.metrics?.gen_tok_s) ?? undefined,
    }))
    .filter((row) => row.model_size && row.prefill_tokens && (row.prefill_tok_s || row.gen_tok_s));
}

function newLocalSpeedBaseline(model: string, rows: SpeedBaselineRow[]): object {
  const arch = detectArch();
  return {
    schema: "hipfire.perf_baseline.v1",
    arch,
    hardware_profile_hash: "local",
    hardware_profile_hash_source: "local-user-baseline",
    captured_on: new Date().toISOString(),
    captured_via: "hipfire eval <model>",
    tolerance_pct: 5.0,
    model: {
      id: inferModelId(model),
      name: basename(model),
    },
    hardware: {
      gfx: arch,
      notes: "Local user baseline captured by hipfire eval default speed run.",
    },
    baselines: {
      speed: rows,
    },
  };
}

function baselineRowKey(row: SpeedBaselineRow): string {
  return `${row.model_id}\0${row.prefill_tokens}`;
}

function speedBaselineFailed(outDir: string): boolean {
  return readFileSync(join(outDir, "results.jsonl"), "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .some(
      (row) =>
        row.battery === "speed" &&
        row.metrics?.perf_baseline_status === "compared" &&
        row.metrics?.perf_baseline_failed === true,
    );
}

function numberMetric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function inferModelId(model: string): string {
  return basename(model)
    .replace(/\.hfq$/i, "")
    .replace(/[^a-zA-Z0-9._+-]+/g, "_")
    .toLowerCase();
}

function inferModelSize(model: string): string | null {
  const match = basename(model).match(/(?:^|[-.])((?:0\.8|1\.7|2|4|8|9|14|27|30|32|35|72|122)b)(?:[-.])/i);
  return match ? match[1].toLowerCase() : null;
}

function inferModelFormat(model: string): string {
  const stem = basename(model).replace(/\.hfq$/i, "");
  const match = stem.match(/(?:^|-)(bf16|f16|fp16|mq[0-9]|hf[0-9]|mfp[0-9]|hfp[0-9]|q8f16|q8)(?:$|[+.-])/i);
  return match ? match[1].toLowerCase() : "unknown";
}

function detectArch(): string | null {
  const out = Bun.spawnSync(["bash", "-lc", "amdgpu-arch 2>/dev/null | head -1 || rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/ {print $1; exit}'"], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const arch = out.stdout.toString().trim().split(/\s+/)[0];
  return arch || null;
}
