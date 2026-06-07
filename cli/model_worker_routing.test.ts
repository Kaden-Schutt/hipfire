import { describe, expect, test } from "bun:test";
import { pickServingModelWorker } from "./model_worker_routing";

describe("model worker routing", () => {
  const base = {
    requestModelPath: "/models/qwen3.5-9b.mq4",
    currentModelPath: "/models/qwen3.5-9b.mq4",
    currentMaxSeq: 4096,
    requiredMaxSeq: 2048,
    archId: "qwen",
    quantFamily: "mq4",
    stateMode: "q8",
    featureFlags: ["serve", "prefill_batch"],
  };

  test("reuses current worker when model and max-seq are compatible", () => {
    const decision = pickServingModelWorker(base);
    expect(decision.needsReload).toBe(false);
    expect(decision.canReuseCurrentWorker).toBe(true);
    expect(decision.reloadReason).toBe("none");
    expect(decision.workerKey.featureFlags).toEqual(["prefill_batch", "serve"]);
  });

  test("reloads when switching model path", () => {
    const decision = pickServingModelWorker({
      ...base,
      currentModelPath: "/models/qwen3.5-27b.mq4",
    });
    expect(decision.needsReload).toBe(true);
    expect(decision.canReuseCurrentWorker).toBe(false);
    expect(decision.reloadReason).toBe("model_mismatch");
  });

  test("reloads when next request exceeds loaded max_seq", () => {
    const decision = pickServingModelWorker({
      ...base,
      requiredMaxSeq: 8192,
    });
    expect(decision.needsReload).toBe(true);
    expect(decision.canReuseCurrentWorker).toBe(false);
    expect(decision.reloadReason).toBe("max_seq_growth");
  });
});
