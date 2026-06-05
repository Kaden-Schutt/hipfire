import { existsSync } from "fs";
import { join, resolve } from "path";
import { homedir } from "os";

export const HOST_PROFILE_HELP = `hipfire host-profile — measured host capability report

Usage:
  hipfire host-profile [--out <path>] [--models-dir <dir>] [--runs N]

Common options:
  --out <path>              Write report JSON there
  --models-dir <dir>        Model storage directory to test, default ~/.hipfire/models
  --size-mib <N>            CPU/GPU copy test size in MiB, default 128
  --storage-size-mib <N>    Storage test size in MiB, default 128
  --runs <N>                Samples per test, default 3
  --warmup-runs <N>         Unmeasured warmup samples per test, default 1
  --gpu-max-size-mib <N>    Cap largest GPU read/write sweep payload size
  --gpu-sweep-mib-step <N>  Override default GPU MiB payload spacing
  --skip-gpu                Skip HIP copy tests
  --skip-storage            Skip ~/.hipfire/models storage tests
  --json                    Print report JSON to stdout

Build runner:
  cargo build --release -p hipfire-runtime --bin hipfire-host-profile`;

export function isHostProfileHelp(args: string[]): boolean {
  return args.length === 0 || args.some((a) => a === "-h" || a === "--help");
}

export function resolveHostProfileBinary(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
  dirname: string = __dirname,
): string | null {
  const exe = platform === "win32" ? ".exe" : "";
  const candidates = [
    env.HIPFIRE_HOST_PROFILE_BIN,
    resolve(dirname, `../target/release/hipfire-host-profile${exe}`),
    resolve(dirname, `../target/debug/hipfire-host-profile${exe}`),
    join(homedir(), ".hipfire", "bin", `hipfire-host-profile${exe}`),
  ].filter((p): p is string => !!p);
  return candidates.find((p) => existsSync(p)) ?? null;
}

export function buildHostProfileCommand(
  args: string[],
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
  dirname: string = __dirname,
): { bin: string; args: string[] } | null {
  const bin = resolveHostProfileBinary(env, platform, dirname);
  return bin ? { bin, args: [...args] } : null;
}

export async function runHostProfileCommand(args: string[]): Promise<void> {
  if (isHostProfileHelp(args)) {
    console.log(HOST_PROFILE_HELP);
    return;
  }

  const cmd = buildHostProfileCommand(args);
  if (!cmd) {
    console.error("hipfire-host-profile not found.");
    console.error("Build it with: cargo build --release -p hipfire-runtime --bin hipfire-host-profile");
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
