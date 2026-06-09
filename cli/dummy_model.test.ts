import { describe, expect, test } from "bun:test";
import {
  buildDummyLoadMessage,
  DUMMY_MODEL_SENTINEL_PATH,
  isDummyModelPath,
  isDummyModelTag,
  resolveDummyModelPath,
} from "./dummy_model";

describe("dummy model sentinel", () => {
  test("resolves only the reserved hipfire:dummy tag", () => {
    expect(resolveDummyModelPath("hipfire:dummy")).toBe(DUMMY_MODEL_SENTINEL_PATH);
    expect(resolveDummyModelPath("dummy")).toBeNull();
    expect(resolveDummyModelPath("qwen3.5:9b")).toBeNull();
    expect(isDummyModelTag("hipfire:dummy")).toBe(true);
    expect(isDummyModelTag("dummy")).toBe(false);
    expect(isDummyModelPath(DUMMY_MODEL_SENTINEL_PATH)).toBe(true);
  });

  test("load message uses dummy params without sidecars or draft", () => {
    const msg = buildDummyLoadMessage();
    expect(msg).toEqual({
      type: "load",
      model: DUMMY_MODEL_SENTINEL_PATH,
      params: { dummy_model: true, max_seq: 4096, physical_cap: 4096 },
    });
    expect(msg.params.draft).toBeUndefined();
    expect(msg.params.cask_sidecar).toBeUndefined();
    expect(msg.params.prefill_drafter).toBeUndefined();
  });
});
