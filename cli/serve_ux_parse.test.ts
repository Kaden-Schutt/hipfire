// Unit tests for the O4 serve/config UX parse fixes (no GPU, no daemon).
//
// These cover the pure, exit-free cores of four CLI bugs found by Codex:
//   HF-CLI-002  restart port-parse must ignore serve flag VALUES (--tp 2)
//   HF-CLI-003  serve --tp with no value must be detectable (dangling)
//   HF-CLI-004  config rejects unknown flags / over-arity per action
//   HF-CLI-005  takeFlagValue gains allowDashValue for free-form string flags
//
// All four are tested through their pure helpers (peekFlagValue,
// parseRestartPort, serveTpDangling, validateConfigActionArgs) so no
// process.exit / daemon spawn is involved.
//
// Run: cd cli && bun test serve_ux_parse.test.ts

import { test, expect, describe, beforeAll } from "bun:test";

let mod: typeof import("./index.ts");
beforeAll(async () => { mod = await import("./index.ts"); });

// ─── HF-CLI-002: restart port-parse ─────────────────────
describe("parseRestartPort (HF-CLI-002)", () => {
  test("`restart -d --tp 2` does NOT treat the --tp VALUE as the port", () => {
    expect(mod.parseRestartPort(["-d", "--tp", "2"], 11435)).toBe(11435);
  });
  test("a real positional port still wins", () => {
    expect(mod.parseRestartPort(["-d", "12345"], 11435)).toBe(12345);
  });
  test("host:port shorthand sets the port, ignoring --kv-mode value", () => {
    expect(mod.parseRestartPort(["--kv-mode", "q8", "0.0.0.0:11500", "-d"], 11435)).toBe(11500);
  });
  test("--idle-timeout numeric value is not mistaken for a port", () => {
    expect(mod.parseRestartPort(["--idle-timeout", "300"], 11435)).toBe(11435);
  });
  test("--tp=N inline form is consumed too", () => {
    expect(mod.parseRestartPort(["--tp=4", "-d"], 11435)).toBe(11435);
  });
  test("no args → default port", () => {
    expect(mod.parseRestartPort([], 11435)).toBe(11435);
  });
});

// ─── HF-CLI-003: serve --tp no-value ────────────────────
describe("serveTpDangling (HF-CLI-003)", () => {
  test("trailing bare --tp is dangling", () => {
    expect(mod.serveTpDangling(["-d", "--tp"])).toBe(true);
  });
  test("--tp with a following value is NOT dangling", () => {
    expect(mod.serveTpDangling(["--tp", "2"])).toBe(false);
  });
  test("--tp=2 inline form is NOT dangling", () => {
    expect(mod.serveTpDangling(["--tp=2"])).toBe(false);
  });
  test("no --tp at all is NOT dangling", () => {
    expect(mod.serveTpDangling(["-d", "11435"])).toBe(false);
  });
  // HF-CLI-003 core fix: `serve --tp -d` must NOT consume the FLAG `-d` as the
  // TP value (that gave rc1 "Invalid --tp value"); a dash-leading next-token is
  // a MISSING value → dangling (USAGE rc2), exactly like a bare trailing --tp.
  test("--tp followed by a flag (-d) is dangling (not a value)", () => {
    expect(mod.serveTpDangling(["--tp", "-d"])).toBe(true);
  });
  test("--tp followed by another long flag (--kv-mode) is dangling", () => {
    expect(mod.serveTpDangling(["--tp", "--kv-mode", "q8"])).toBe(true);
  });
  test("--tp with a value then a trailing flag is NOT dangling", () => {
    expect(mod.serveTpDangling(["--tp", "2", "-d"])).toBe(false);
  });
});

// ─── HF-CLI-004: config unknown-flag / arity ────────────
describe("validateConfigActionArgs (HF-CLI-004)", () => {
  test("`config list --josn` (typo) is rejected", () => {
    const v = mod.validateConfigActionArgs("list", ["--josn"], "--josn");
    expect(v.ok).toBe(false);
    if (!v.ok) expect(v.error).toContain("--josn");
  });
  test("`config list` with no extras is ok", () => {
    expect(mod.validateConfigActionArgs("list", [], undefined).ok).toBe(true);
  });
  test("`config list extrapos` (stray positional) is rejected", () => {
    expect(mod.validateConfigActionArgs("list", ["extrapos"], "extrapos").ok).toBe(false);
  });
  test("`config get --bogus` is rejected", () => {
    expect(mod.validateConfigActionArgs("get", ["temperature", "--bogus"], "temperature").ok).toBe(false);
  });
  test("`config get temperature` is ok", () => {
    expect(mod.validateConfigActionArgs("get", ["temperature"], "temperature").ok).toBe(true);
  });
  // HF-CLI-004 core fix: get/reset/cask-profile must reject extra positional
  // tails (previously silently ignored → exit 0).
  test("`config get temperature extra` (extra positional) is rejected", () => {
    const v = mod.validateConfigActionArgs("get", ["temperature", "extra"], "temperature");
    expect(v.ok).toBe(false);
    if (!v.ok) expect(v.error).toContain("extra");
  });
  test("`config reset temperature extra` (extra positional) is rejected", () => {
    expect(mod.validateConfigActionArgs("reset", ["temperature", "extra"], "temperature").ok).toBe(false);
  });
  test("`config reset temperature` is ok", () => {
    expect(mod.validateConfigActionArgs("reset", ["temperature"], "temperature").ok).toBe(true);
  });
  test("`config cask-profile balanced` (one name) is ok", () => {
    expect(mod.validateConfigActionArgs("cask-profile", ["balanced"], "balanced").ok).toBe(true);
  });
  test("`config cask-profile` (show, no name) is ok", () => {
    expect(mod.validateConfigActionArgs("cask-profile", [], undefined).ok).toBe(true);
  });
  test("`config cask-profile off extra` (extra positional) is rejected", () => {
    const v = mod.validateConfigActionArgs("cask-profile", ["off", "extra"], "off");
    expect(v.ok).toBe(false);
    if (!v.ok) expect(v.error).toContain("extra");
  });
  test("`config set system - terse` keeps a dash VALUE (only key is flag-checked)", () => {
    expect(mod.validateConfigActionArgs("set", ["system", "-", "terse"], "system").ok).toBe(true);
  });
  test("`config set --oops v` rejects a flag in the key slot", () => {
    expect(mod.validateConfigActionArgs("set", ["--oops", "v"], "--oops").ok).toBe(false);
  });
});

// ─── HF-CLI-005: takeFlagValue allowDashValue ───────────
describe("peekFlagValue allowDashValue (HF-CLI-005)", () => {
  test("strict (default): a dash-leading value is treated as MISSING", () => {
    const r = mod.peekFlagValue(["--temp", "--json"], "--temp");
    expect(r.kind).toBe("missing");
  });
  test("strict: a normal value is accepted", () => {
    const r = mod.peekFlagValue(["--temp", "0.7"], "--temp");
    expect(r.kind).toBe("value");
    if (r.kind === "value") expect(r.value).toBe("0.7");
  });
  test("allowDashValue: a dash-leading free-form value is accepted (--system '- terse')", () => {
    const r = mod.peekFlagValue(["--system", "- terse"], "--system", { allowDashValue: true });
    expect(r.kind).toBe("value");
    if (r.kind === "value") expect(r.value).toBe("- terse");
  });
  test("allowDashValue: a dash-leading image path is accepted", () => {
    const r = mod.peekFlagValue(["--image", "-shot.png"], "--image", { allowDashValue: true });
    expect(r.kind).toBe("value");
    if (r.kind === "value") expect(r.value).toBe("-shot.png");
  });
  test("allowDashValue still reports MISSING when the flag is the last token", () => {
    const r = mod.peekFlagValue(["--system"], "--system", { allowDashValue: true });
    expect(r.kind).toBe("missing");
  });
  test("absent flag → absent", () => {
    expect(mod.peekFlagValue(["-d"], "--temp").kind).toBe("absent");
  });
});
