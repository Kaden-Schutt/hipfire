import {
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "fs";
import { tmpdir, hostname } from "os";
import { join } from "path";

export type ResourceKind = "hip-gpu" | "npu" | "cpu-core";

export type ResourceLockRequest = {
  kind: ResourceKind;
  id: string | number;
};

export type ResourceLockOwner = {
  pid: number;
  host: string;
  command: string;
  started_at: string;
  resource: string;
};

export type ResourceLeaseOptions = {
  rootDir?: string;
  pid?: number;
  command?: string;
  waitMs?: number;
  now?: () => Date;
  isPidAlive?: (pid: number) => boolean;
  sleep?: (ms: number) => Promise<void>;
};

export type ResourceLease = {
  resources: string[];
  release: () => void;
};

export class ResourceLockBusyError extends Error {
  resource: string;
  owner: ResourceLockOwner | null;

  constructor(resource: string, owner: ResourceLockOwner | null) {
    const ownerText = owner
      ? `pid=${owner.pid} host=${owner.host} command=${owner.command}`
      : "unknown owner";
    super(`hipfire resource ${resource} is already locked by ${ownerText}`);
    this.name = "ResourceLockBusyError";
    this.resource = resource;
    this.owner = owner;
  }
}

function defaultIsPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function sanitizeResourceId(id: string | number): string {
  return String(id).replace(/[^A-Za-z0-9_.-]+/g, "_");
}

export function resourceLockName(resource: ResourceLockRequest): string {
  return `${resource.kind}-${sanitizeResourceId(resource.id)}`;
}

function ownerPath(lockDir: string): string {
  return join(lockDir, "owner.json");
}

function readOwner(lockDir: string): ResourceLockOwner | null {
  try {
    return JSON.parse(readFileSync(ownerPath(lockDir), "utf8"));
  } catch {
    return null;
  }
}

function writeOwner(lockDir: string, owner: ResourceLockOwner): void {
  writeFileSync(ownerPath(lockDir), `${JSON.stringify(owner, null, 2)}\n`, { mode: 0o600 });
}

export function parseCpuCoreList(raw: string | undefined | null): number[] {
  if (!raw) return [];
  const out = new Set<number>();
  for (const part of raw.split(",")) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const range = trimmed.match(/^(\d+)-(\d+)$/);
    if (range) {
      const start = Number.parseInt(range[1], 10);
      const end = Number.parseInt(range[2], 10);
      if (end < start) throw new Error(`invalid CPU core range: ${trimmed}`);
      for (let core = start; core <= end; core++) out.add(core);
      continue;
    }
    if (!/^\d+$/.test(trimmed)) throw new Error(`invalid CPU core id: ${trimmed}`);
    out.add(Number.parseInt(trimmed, 10));
  }
  return [...out].sort((a, b) => a - b);
}

async function tryAcquireOne(
  resource: string,
  rootDir: string,
  owner: ResourceLockOwner,
  isPidAlive: (pid: number) => boolean,
): Promise<string> {
  mkdirSync(rootDir, { recursive: true, mode: 0o700 });
  const lockDir = join(rootDir, `${resource}.lock`);
  try {
    mkdirSync(lockDir, { mode: 0o700 });
    writeOwner(lockDir, owner);
    return lockDir;
  } catch (err: any) {
    if (err?.code !== "EEXIST") throw err;
  }

  const existing = readOwner(lockDir);
  if (existing && !isPidAlive(existing.pid)) {
    rmSync(lockDir, { recursive: true, force: true });
    mkdirSync(lockDir, { mode: 0o700 });
    writeOwner(lockDir, owner);
    return lockDir;
  }
  throw new ResourceLockBusyError(resource, existing);
}

export async function acquireResourceLease(
  requests: ResourceLockRequest[],
  options: ResourceLeaseOptions = {},
): Promise<ResourceLease> {
  const uniqueResources = [...new Set(requests.map(resourceLockName))].sort();
  const rootDir = options.rootDir ?? join(tmpdir(), "hipfire-resource-locks");
  const pid = options.pid ?? process.pid;
  const command = options.command ?? process.argv.join(" ");
  const now = options.now ?? (() => new Date());
  const isPidAlive = options.isPidAlive ?? defaultIsPidAlive;
  const sleep = options.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  const waitMs = Math.max(0, options.waitMs ?? 0);
  const deadline = Date.now() + waitMs;
  const acquired: string[] = [];

  const release = () => {
    for (const dir of acquired.splice(0).reverse()) {
      rmSync(dir, { recursive: true, force: true });
    }
  };

  try {
    for (const resource of uniqueResources) {
      while (true) {
        const owner: ResourceLockOwner = {
          pid,
          host: hostname(),
          command,
          started_at: now().toISOString(),
          resource,
        };
        try {
          const dir = await tryAcquireOne(resource, rootDir, owner, isPidAlive);
          acquired.push(dir);
          break;
        } catch (err) {
          if (!(err instanceof ResourceLockBusyError) || Date.now() >= deadline) throw err;
          await sleep(Math.min(250, Math.max(25, deadline - Date.now())));
        }
      }
    }
    return {
      resources: uniqueResources,
      release,
    };
  } catch (err) {
    release();
    throw err;
  }
}

export function buildServeResourceLocks(input: {
  hipGpuIds: Array<string | number>;
  npuIds?: Array<string | number>;
  cpuCores?: number[];
}): ResourceLockRequest[] {
  return [
    ...input.hipGpuIds.map((id) => ({ kind: "hip-gpu" as const, id })),
    ...(input.npuIds ?? []).map((id) => ({ kind: "npu" as const, id })),
    ...(input.cpuCores ?? []).map((id) => ({ kind: "cpu-core" as const, id })),
  ];
}
