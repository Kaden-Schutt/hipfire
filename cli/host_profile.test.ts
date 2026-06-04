import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import {
  HOST_PROFILE_HELP,
  buildHostProfileCommand,
  isHostProfileHelp,
  resolveHostProfileBinary,
} from "./host_profile";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length) {
    rmSync(tempDirs.pop()!, { recursive: true, force: true });
  }
});

function fakeBin(name = "hipfire-host-profile"): string {
  const dir = mkdtempSync(join(tmpdir(), "hipfire-host-profile-test-"));
  tempDirs.push(dir);
  const path = join(dir, name);
  writeFileSync(path, "#!/bin/sh\nexit 0\n");
  return path;
}

describe("hipfire host-profile CLI delegation", () => {
  test("resolves HIPFIRE_HOST_PROFILE_BIN first", () => {
    const bin = fakeBin();
    expect(resolveHostProfileBinary({ HIPFIRE_HOST_PROFILE_BIN: bin }, "linux", "/missing")).toBe(
      bin,
    );
  });

  test("passes args through unchanged", () => {
    const bin = fakeBin();
    const args = [
      "--models-dir",
      "/tmp/models",
      "--runs",
      "2",
      "--warmup-runs",
      "1",
      "--gpu-max-size-mib",
      "64",
      "--gpu-sweep-mib-step",
      "1",
      "--skip-gpu",
    ];
    expect(
      buildHostProfileCommand(args, { HIPFIRE_HOST_PROFILE_BIN: bin }, "linux", "/missing"),
    ).toEqual({ bin, args });
  });

  test("handles help without requiring a Rust binary", () => {
    expect(isHostProfileHelp([])).toBe(true);
    expect(isHostProfileHelp(["--help"])).toBe(true);
    expect(isHostProfileHelp(["--models-dir", "/tmp/models"])).toBe(false);
    expect(HOST_PROFILE_HELP).toContain("hipfire host-profile");
    expect(HOST_PROFILE_HELP).toContain("--models-dir <dir>");
    expect(HOST_PROFILE_HELP).toContain("--storage-size-mib <N>");
    expect(HOST_PROFILE_HELP).toContain("--warmup-runs <N>");
    expect(HOST_PROFILE_HELP).toContain("--gpu-max-size-mib <N>");
    expect(HOST_PROFILE_HELP).toContain("--gpu-sweep-mib-step <N>");
    expect(HOST_PROFILE_HELP).toContain("hipfire-host-profile");
  });
});
