import { mkdtempSync, existsSync, readFileSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { describe, expect, test } from "bun:test";
import {
  acquireResourceLease,
  buildServeResourceLocks,
  parseCpuCoreList,
  ResourceLockBusyError,
} from "./resource_lock";

function tempLockRoot(): string {
  return mkdtempSync(join(tmpdir(), "hipfire-resource-lock-test-"));
}

describe("runtime resource locks", () => {
  test("acquires one lock file per scoped resource and releases them", async () => {
    const root = tempLockRoot();
    const lease = await acquireResourceLease(
      buildServeResourceLocks({ hipGpuIds: [0], npuIds: ["accel0"], cpuCores: [2, 3] }),
      { rootDir: root, pid: 1234, command: "hipfire serve", isPidAlive: () => true },
    );

    expect(lease.resources).toEqual(["cpu-core-2", "cpu-core-3", "hip-gpu-0", "npu-accel0"]);
    const owner = JSON.parse(readFileSync(join(root, "hip-gpu-0.lock", "owner.json"), "utf8"));
    expect(owner.pid).toBe(1234);
    expect(owner.command).toBe("hipfire serve");

    lease.release();
    expect(existsSync(join(root, "hip-gpu-0.lock"))).toBe(false);
    rmSync(root, { recursive: true, force: true });
  });

  test("rejects a live conflicting owner", async () => {
    const root = tempLockRoot();
    const first = await acquireResourceLease(
      [{ kind: "hip-gpu", id: 0 }],
      { rootDir: root, pid: 111, isPidAlive: () => true },
    );

    await expect(acquireResourceLease(
      [{ kind: "hip-gpu", id: 0 }],
      { rootDir: root, pid: 222, isPidAlive: () => true },
    )).rejects.toBeInstanceOf(ResourceLockBusyError);

    first.release();
    rmSync(root, { recursive: true, force: true });
  });

  test("reclaims stale pid locks", async () => {
    const root = tempLockRoot();
    await acquireResourceLease(
      [{ kind: "hip-gpu", id: 0 }],
      { rootDir: root, pid: 111, isPidAlive: () => true },
    );
    expect(existsSync(join(root, "hip-gpu-0.lock"))).toBe(true);

    const replacement = await acquireResourceLease(
      [{ kind: "hip-gpu", id: 0 }],
      { rootDir: root, pid: 222, isPidAlive: (pid) => pid !== 111 },
    );
    const owner = JSON.parse(readFileSync(join(root, "hip-gpu-0.lock", "owner.json"), "utf8"));
    expect(owner.pid).toBe(222);

    replacement.release();
    rmSync(root, { recursive: true, force: true });
  });

  test("parses explicit CPU core lists", () => {
    expect(parseCpuCoreList("0,2-4,3")).toEqual([0, 2, 3, 4]);
    expect(parseCpuCoreList("")).toEqual([]);
    expect(() => parseCpuCoreList("4-2")).toThrow("invalid CPU core range");
    expect(() => parseCpuCoreList("gpu0")).toThrow("invalid CPU core id");
  });
});
