import { describe, expect, test } from "bun:test";
import { resolveServeThinkCap } from "./index.ts";

describe("serve think-cap resolution", () => {
  test("forwards the resolved default medium cap", () => {
    expect(resolveServeThinkCap("on", 2048, undefined, null)).toBe(2048);
  });

  test("preserves uncapped and explicit request precedence", () => {
    expect(resolveServeThinkCap("on", 0, undefined, null)).toBeUndefined();
    expect(resolveServeThinkCap("off", 2048, undefined, null)).toBe(1);
    expect(resolveServeThinkCap("on", 2048, false, null)).toBe(1);
    expect(resolveServeThinkCap("off", 2048, false, 0)).toBeUndefined();
    expect(resolveServeThinkCap("on", 2048, undefined, 4096)).toBe(4096);
  });
});
