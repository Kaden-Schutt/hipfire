// Regression test: the KV cache default is the universal q8 (reached via the
// "auto" inherit sentinel), NOT a per-arch fwht guess. The old per-arch
// `archDefaults` table — which keyed the default off hardcoded `vram_gb` numbers
// that were wrong for the unified-memory APUs and fed no real fit logic — was
// removed. KV mode is now q8 by default; a model can recommend a compressed mode
// per-model via the registry `default_kv_mode`.
//
// Source-text assertions: we don't import index.ts (it has top-level side
// effects — see config_meta.test.ts), so we parse the source and assert the
// invariants instead.
//
// Run: bun test cli/kv_default.test.ts

import { test, expect } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

const src = readFileSync(join(import.meta.dir, "index.ts"), "utf-8");

test("DEFAULT_KV_MODE is q8", () => {
  expect(src).toMatch(/const DEFAULT_KV_MODE\s*=\s*"q8"/);
});

test("CONFIG_DEFAULTS.kv_cache is the auto sentinel, not a concrete mode", () => {
  // A concrete literal here would bypass the registry default_kv_mode path for a
  // clean config (the latent bug this change also fixes).
  expect(src).toMatch(/CONFIG_DEFAULTS[\s\S]{0,400}kv_cache:\s*"auto"/);
});

test("the per-arch KV/vram table is removed", () => {
  // These identifiers were unique to the removed arch table (the legitimate
  // model-catalog field `min_vram_gb` is unrelated and stays).
  expect(src).not.toContain("function archDefaults");
  expect(src).not.toContain("ARCH_DEFAULTS");
  expect(src).not.toContain("interface ArchDefaults");
});

test("resolveKvMode resolves auto -> registry default_kv_mode, else DEFAULT_KV_MODE", () => {
  const fn = src.slice(src.indexOf("function resolveKvMode"));
  // The auto branch consults the registry default, then falls back to q8.
  expect(fn).toMatch(/regDefault[\s\S]{0,160}DEFAULT_KV_MODE/);
  // And it no longer references the removed arch table.
  expect(fn).not.toContain("ARCH_DEFAULTS");
});
