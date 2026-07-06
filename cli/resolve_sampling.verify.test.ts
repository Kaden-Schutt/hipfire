// W7 P1+P3 verification: config-inheritance resolution proof (NO GPU).
//
// Drives resolveSamplingForSend (the CLI-side explicit-send guard that layers
// registry card recommended_settings < per-model models.json < explicit --flag)
// for every mandated resolution case. A per-model override is injected via a
// temp HOME so loadPerModelConfigs reads a controlled models.json, never the
// real one.
//
// IMPORTANT: HOME must be set BEFORE importing index.ts because the path
// constants (MODELS_CATALOG_PATH etc.) are computed at module-load from
// homedir().

import { test, expect, describe, beforeAll } from "bun:test";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { homedir } from "os";
import { join } from "path";

// NOTE: os.homedir() ignores $HOME on linux (reads the user db directly), so
// the per-model override case drives the REAL ~/.hipfire/models.json: the
// helper below snapshots it, injects `qwen3.6:35b-a3b.temperature=0.7`, runs
// the assertion, and restores the snapshot — so it is deterministic in the
// full suite and leaves no residue. The card / flag / unlisted cases need no
// filesystem setup.

const MODELS_JSON = join(homedir(), ".hipfire", "models.json");

function withPerModelOverride<T>(tag: string, ov: Record<string, unknown>, fn: () => T): T {
  const dir = join(homedir(), ".hipfire");
  mkdirSync(dir, { recursive: true });
  const had = existsSync(MODELS_JSON);
  const snapshot = had ? readFileSync(MODELS_JSON, "utf8") : null;
  try {
    const base = had ? JSON.parse(snapshot as string) : { schema_version: 2, aliases: {}, configs: {}, models: {} };
    base.configs = base.configs ?? {};
    base.configs[tag] = { ...(base.configs[tag] ?? {}), ...ov };
    writeFileSync(MODELS_JSON, JSON.stringify(base, null, 2) + "\n");
    return fn();
  } finally {
    if (snapshot !== null) writeFileSync(MODELS_JSON, snapshot);
  }
}

let resolveSamplingForSend: typeof import("./index.ts").resolveSamplingForSend;

beforeAll(async () => {
  ({ resolveSamplingForSend } = await import("./index.ts"));
});

describe("W7 config-inheritance resolution (no-GPU)", () => {
  test("qwen3.6:35b-a3b inherits card temp=1.0/top_p=0.95/presence_penalty=1.5 (NOT global 0.3/0.8)", () => {
    // Card-only (no per-model override): full card inheritance.
    const card = resolveSamplingForSend("qwen3.6:35b-a3b");
    expect(card.temperature).toBe(1.0);
    expect(card.top_p).toBe(0.95);
    expect(card.presence_penalty).toBe(1.5);
    // Decidedly NOT the global default 0.3/0.8.
    expect(card.temperature).not.toBe(0.3);
    expect(card.top_p).not.toBe(0.8);
  });

  test("per-model models.json override BEATS the card (temp 0.7 wins over card 1.0); top_p/presence stay card", () => {
    withPerModelOverride("qwen3.6:35b-a3b", { temperature: 0.7 }, () => {
      const s = resolveSamplingForSend("qwen3.6:35b-a3b");
      expect(s.temperature).toBe(0.7); // per-model beats card
      expect(s.top_p).toBe(0.95); // card (no per-model top_p)
      expect(s.presence_penalty).toBe(1.5); // card
    });
  });

  test("deepseek-v4-flash → temperature=1.0 / top_p=1.0 (card)", () => {
    const s = resolveSamplingForSend("deepseek-v4-flash");
    expect(s.temperature).toBe(1.0);
    expect(s.top_p).toBe(1.0);
  });

  test("deepseek4 alias resolves to the same card", () => {
    const s = resolveSamplingForSend("deepseek4");
    expect(s.temperature).toBe(1.0);
    expect(s.top_p).toBe(1.0);
  });

  test("minimax-m2.7 → temperature=1.0 / top_p=0.95 + system_prompt", () => {
    const s = resolveSamplingForSend("minimax-m2.7");
    expect(s.temperature).toBe(1.0);
    expect(s.top_p).toBe(0.95);
    expect(typeof s.system_prompt).toBe("string");
    expect(s.system_prompt).toContain("MiniMax-M2.7");
  });

  test("--temp 0.5 explicit flag OVERRIDES the card on qwen3.6:35b-a3b", () => {
    const s = resolveSamplingForSend("qwen3.6:35b-a3b", { temperature: 0.5 });
    expect(s.temperature).toBe(0.5); // explicit flag wins over both card+per-model
    expect(s.top_p).toBe(0.95); // card still applies to the un-flagged knob
  });

  test("unlisted model → CLI omits all sampling (empty view) so daemon arch/global default applies", () => {
    const s = resolveSamplingForSend("some-unlisted-model-xyz:1b");
    expect(s.temperature).toBeUndefined();
    expect(s.top_p).toBeUndefined();
    expect(s.repeat_penalty).toBeUndefined();
    expect(s.presence_penalty).toBeUndefined();
    expect(s.system_prompt).toBeUndefined();
  });
});
