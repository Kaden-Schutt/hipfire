import { expect, test } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import {
  mtpDraftMaxForSpeculation,
  parseMtpK,
  resolveMtpK,
  validateMtpDraftMaxFlag,
} from "./mtp_k";

const indexSrc = readFileSync(join(import.meta.dir, "index.ts"), "utf-8");

test("MTP K resolves environment before draft and configured values", () => {
  expect(resolveMtpK("8", "7", 6)).toBe(8);
});

test("MTP K resolves draft before configured values", () => {
  expect(resolveMtpK(undefined, "7", 6)).toBe(7);
});

test("MTP K uses per-model or global resolved config before default", () => {
  expect(resolveMtpK(undefined, undefined, 6)).toBe(6);
  expect(resolveMtpK(undefined, undefined, undefined)).toBe(3);
});

test("MTP K ignores environment overrides when MTP is ineligible", () => {
  expect(resolveMtpK("invalid", "invalid", 6, false)).toBe(6);
});

test("MTP K rejects non-decimal and out-of-range values", () => {
  for (const value of ["01", "001", "1.5", "1e0", "+1", "3junk", "0", "9", "10", "11"]) {
    expect(() => parseMtpK(value, "MTP K")).toThrow();
  }
});

test("MTP draft-max flag validates before lowering", () => {
  for (const value of ["01", "1.5", "1e0", "+1", "3junk", "0", "9", "10", "11"]) {
    expect(() => validateMtpDraftMaxFlag(value, "mtp")).toThrow();
  }
  expect(() => validateMtpDraftMaxFlag("12", "ngram")).not.toThrow();
  expect(() => validateMtpDraftMaxFlag("64", "auto")).not.toThrow();
});

test("DeepSeek auto speculation feeds draft max into strict MTP K", () => {
  expect(mtpDraftMaxForSpeculation("7", "auto", "deepseek-v4-flash-mtp")).toBe("7");
  expect(() => validateMtpDraftMaxFlag("9", "auto", "deepseek-v4-flash-mtp")).toThrow();
  expect(mtpDraftMaxForSpeculation("7", "mtp")).toBe("7");
});

test("non-DeepSeek auto speculation leaves every draft window unchanged", () => {
  expect(mtpDraftMaxForSpeculation("64", "auto", "qwen3.5:9b")).toBeUndefined();
});

test("config environment projection preserves an exported MTP K", () => {
  const applyConfigEnv = indexSrc.match(/function applyConfigEnv[\s\S]*?\n}\n/);
  expect(applyConfigEnv).not.toBeNull();
  expect(applyConfigEnv![0]).not.toMatch(/HIPFIRE_MTP_K\s*=/);
  expect(applyConfigEnv![0]).not.toMatch(/HIPFIRE_MTP_MODE\s*=/);
});

test("load message resolves MTP K from environment and model-aware draft max", () => {
  expect(indexSrc).toMatch(
    /const mtpDraftMax = mtpDraftMaxForSpeculation\(process\.env\.HIPFIRE_DRAFT_MAX, speculation, modelIdentity\);[\s\S]*?const ngramDraftMax = draftMaxForMechanism\(process\.env\.HIPFIRE_DRAFT_MAX, speculation, modelIdentity, "ngram"\);[\s\S]*?const dflashDraftMax = draftMaxForMechanism\(process\.env\.HIPFIRE_DRAFT_MAX, speculation, modelIdentity, "dflash"\);[\s\S]*?resolveMtpK\([\s\S]*?process\.env\.HIPFIRE_MTP_K,[\s\S]*?mtpDraftMax,[\s\S]*?resolved\.mtp_k,[\s\S]*?effMtpMode !== "off",[\s\S]*?\);/,
  );
});
