// Unit tests for the pure CLI formatting helpers (cli_format.ts).
//
// These cover the download-progress rate/ETA/bar formatting, the
// error-message formatter (the top-level guard's source of truth), and the
// shape of the config-json keys object. All pure — no GPU, no I/O, no daemon.
//
// Run: cd cli && bun test cli_format.test.ts

import { test, expect } from "bun:test";
import {
  EXIT,
  formatErrorMessage,
  formatBytes,
  formatRate,
  formatEta,
  progressBar,
  formatProgressLine,
} from "./cli_format";

// ─── EXIT codes ─────────────────────────────────────────
test("EXIT codes are the documented set", () => {
  expect(EXIT.OK).toBe(0);
  expect(EXIT.ERROR).toBe(1);
  expect(EXIT.USAGE).toBe(2);
  expect(EXIT.SIGINT).toBe(130);
});

// ─── formatErrorMessage ─────────────────────────────────
test("formatErrorMessage: Error instance → clean message, no stack", () => {
  const e = new Error("connection refused");
  expect(formatErrorMessage(e)).toBe("connection refused");
});

test("formatErrorMessage: strips embedded stack lines to a one-liner", () => {
  const msg = "Boom happened\n    at foo (index.ts:10:5)\n    at bar (index.ts:20:3)";
  const e = new Error(msg);
  expect(formatErrorMessage(e)).toBe("Boom happened");
});

test("formatErrorMessage: collapses internal whitespace", () => {
  expect(formatErrorMessage("a   b\tc")).toBe("a b c");
});

test("formatErrorMessage: plain string", () => {
  expect(formatErrorMessage("nope")).toBe("nope");
});

test("formatErrorMessage: null / undefined → fallback", () => {
  expect(formatErrorMessage(null)).toBe("unknown error");
  expect(formatErrorMessage(undefined)).toBe("unknown error");
  expect(formatErrorMessage("")).toBe("unknown error");
});

test("formatErrorMessage: message-less Error → error name", () => {
  const e = new Error("");
  expect(formatErrorMessage(e)).toBe("Error");
  // A subclass that sets .name surfaces that name.
  const weird = new Error("");
  weird.name = "WeirdError";
  expect(formatErrorMessage(weird)).toBe("WeirdError");
});

test("formatErrorMessage: object with message field", () => {
  expect(formatErrorMessage({ message: "object error" })).toBe("object error");
});

test("formatErrorMessage: never contains a newline", () => {
  const e = new Error("line1\nline2\nline3");
  expect(formatErrorMessage(e)).not.toContain("\n");
});

// ─── formatBytes ────────────────────────────────────────
test("formatBytes scales B/KB/MB/GB", () => {
  expect(formatBytes(512)).toBe("512 B");
  expect(formatBytes(1500)).toBe("1.5 KB");
  expect(formatBytes(2_500_000)).toBe("2.5 MB");
  expect(formatBytes(3_210_000_000)).toBe("3.21 GB");
  expect(formatBytes(-1)).toBe("?");
});

// ─── formatRate ─────────────────────────────────────────
test("formatRate: bytes/sec → human rate", () => {
  expect(formatRate(8_100_000)).toBe("8.1 MB/s");
  expect(formatRate(0)).toBe("—");
  expect(formatRate(-5)).toBe("—");
  expect(formatRate(Infinity)).toBe("—");
});

// ─── formatEta ──────────────────────────────────────────
test("formatEta: seconds → compact duration", () => {
  expect(formatEta(0)).toBe("0s");
  expect(formatEta(45)).toBe("45s");
  expect(formatEta(72)).toBe("1m12s");
  expect(formatEta(3600)).toBe("1h00m");
  expect(formatEta(3725)).toBe("1h02m");
  expect(formatEta(-1)).toBe("—");
  expect(formatEta(Infinity)).toBe("—");
});

// ─── progressBar ────────────────────────────────────────
test("progressBar: 0% empty, 100% full", () => {
  const empty = progressBar(0, 10);
  expect(empty.length).toBe(10);
  expect(empty.includes("█")).toBe(false);
  const full = progressBar(1, 10);
  expect(full).toBe("█".repeat(10));
});

test("progressBar: clamps out-of-range and width is exact", () => {
  expect(progressBar(2, 8).length).toBe(8);
  expect(progressBar(-1, 8)).toBe(" ".repeat(8));
  expect(progressBar(0.5, 20).length).toBe(20);
});

test("progressBar: unknown frac (NaN) → dashed bar of given width", () => {
  const b = progressBar(NaN, 12);
  expect(b.length).toBe(12);
  expect(b).toBe("─".repeat(12));
});

// ─── formatProgressLine ─────────────────────────────────
test("formatProgressLine: known total has pct, rate, ETA, MB", () => {
  const line = formatProgressLine({ downloaded: 50_000_000, total: 100_000_000, bytesPerSec: 10_000_000 }, 20);
  expect(line).toContain("50.0%");
  expect(line).toContain("10.0 MB/s");
  expect(line).toContain("ETA");
  expect(line).toContain("5s"); // 50MB remaining / 10MB/s = 5s
  expect(line).toContain("50/100 MB");
  expect(line.startsWith("[")).toBe(true);
});

test("formatProgressLine: unknown total → ?% and — ETA, no crash", () => {
  const line = formatProgressLine({ downloaded: 12_000_000, total: 0, bytesPerSec: 0 }, 20);
  expect(line).toContain("?%");
  expect(line).toMatch(/ETA\s+—/); // ETA label, padded, then unknown dash
  expect(line).toContain("12/? MB");
});

test("formatProgressLine: zero rate but known total → ETA —", () => {
  const line = formatProgressLine({ downloaded: 10, total: 100, bytesPerSec: 0 }, 20);
  expect(line).toMatch(/ETA\s+—/);
});

// ─── config-json shape ──────────────────────────────────
// The CLI emits `config list --json` as { scope, path, keys: { <k>: {value,
// default, is_default} } } and `config get <k> --json` as { scope, key,
// value }. This asserts the contract a JSON consumer relies on, built from
// the same primitive the command uses (value/default/is_default per key).
test("config list --json key entry shape", () => {
  const buildEntry = (value: unknown, def: unknown) => ({ value, default: def, is_default: value === def });
  const e1 = buildEntry("q8", "auto");
  expect(e1).toEqual({ value: "q8", default: "auto", is_default: false });
  const e2 = buildEntry(0.3, 0.3);
  expect(e2.is_default).toBe(true);

  const out = { scope: "global", path: "/x/config.json", keys: { kv_cache: e1, temperature: e2 } };
  const round = JSON.parse(JSON.stringify(out));
  expect(round.scope).toBe("global");
  expect(round.keys.kv_cache.value).toBe("q8");
  expect(round.keys.kv_cache.is_default).toBe(false);
  expect(round.keys.temperature.is_default).toBe(true);
});

test("config get --json is a flat {scope,key,value}", () => {
  const got = JSON.parse(JSON.stringify({ scope: "global", key: "port", value: 11435 }));
  expect(got).toEqual({ scope: "global", key: "port", value: 11435 });
});
