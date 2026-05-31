// Regression test for the `configTui` crash where a CONFIG_DEFAULTS key was
// added without a matching `meta` entry, causing `meta[k].label.length` to
// throw `TypeError: undefined is not an object` on TUI render.
//
// Original crash report: `prefill_drafter_device` was added by PR #319 but
// the TUI's `meta` map wasn't updated. Iterating `Object.keys(CONFIG_DEFAULTS)`
// then dereferencing `meta[k].label` blew up at startup.
//
// We don't import index.ts because of its top-level side effects (see
// parse_tool_calls.test.ts). Instead, parse the source and assert the
// invariant: every CONFIG_DEFAULTS key must also exist as a meta key.
//
// Run: bun test cli/config_meta.test.ts

import { test, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

test("every CONFIG_DEFAULTS key has a matching configTui meta entry", () => {
  const src = readFileSync(join(import.meta.dir, "index.ts"), "utf-8");

  const defaultsMatch = src.match(/const CONFIG_DEFAULTS[^{]+\{([\s\S]*?)\n\};/);
  expect(defaultsMatch).not.toBeNull();
  const cfgKeys = [...defaultsMatch![1].matchAll(/^\s{2}([a-z_]+):/gm)].map(m => m[1]);
  expect(cfgKeys.length).toBeGreaterThan(0);

  const metaMatch = src.match(/const meta: Record<string, FieldMeta> = \{([\s\S]*?)\n  \};/);
  expect(metaMatch).not.toBeNull();
  const metaKeys = [...metaMatch![1].matchAll(/^\s{4}([a-z_]+):\s\{/gm)].map(m => m[1]);

  const missing = cfgKeys.filter(k => !metaKeys.includes(k));
  expect(missing).toEqual([]);
});
