import { existsSync } from "fs";
import { join, resolve } from "path";
import { homedir } from "os";

export const EVAL_HELP = `hipfire eval — quant admission/model evaluation harness

Usage:
  hipfire eval --model <model> [--tier fast|medium|long|extensive]
  hipfire eval --model <model> --battery smoke,quality,speed
  hipfire eval --model <model> --suite gpqa --fetch-datasets

Common options:
  --version              Print hipfire-eval version/git metadata
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
  const candidates = [
    env.HIPFIRE_EVAL_BIN,
    resolve(dirname, `../target/release/hipfire-eval${exe}`),
    join(homedir(), ".hipfire", "bin", `hipfire-eval${exe}`),
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
  return bin ? { bin, args: [...args] } : null;
}

export async function runEvalCommand(args: string[]): Promise<void> {
  if (isEvalHelp(args)) {
    console.log(EVAL_HELP);
    return;
  }

  const cmd = buildEvalCommand(args);
  if (!cmd) {
    console.error("hipfire-eval not found.");
    console.error("Build it with: cargo build --release -p hipfire-runtime --bin hipfire-eval");
    process.exit(1);
  }

  const proc = Bun.spawn([cmd.bin, ...cmd.args], {
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  });
  const code = await proc.exited;
  process.exit(code);
}
