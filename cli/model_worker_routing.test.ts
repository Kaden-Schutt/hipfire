import { describe, expect, test } from "bun:test";
import { pickResidentModelWorker, pickServingModelWorker } from "./model_worker_routing";

describe("model worker routing", () => {
  const base = {
    requestModelPath: "/models/qwen3.5-9b-mq4.hfq",
    currentModelPath: "/models/qwen3.5-9b-mq4.hfq",
    currentMaxSeq: 4096,
    requiredMaxSeq: 2048,
    archId: "qwen",
    quantFamily: "mq4",
    stateMode: "q8",
    featureFlags: ["serve", "prefill_batch"],
    acceleratorKind: "hip",
    deviceId: 0,
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
      currentModelPath: "/models/qwen3.5-27b-mq4.hfq",
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

  test("carries accelerator placement into worker identity", () => {
    const dev0 = pickServingModelWorker(base);
    const dev1 = pickServingModelWorker({ ...base, deviceId: 1 });
    expect(dev0.workerKey.deviceId).toBe(0);
    expect(dev1.workerKey.deviceId).toBe(1);
    expect(dev0.workerKey).not.toEqual(dev1.workerKey);
  });

  test("reuses exact resident worker with enough max_seq", () => {
    const decision = pickResidentModelWorker({
      requestModelPath: "/models/a.hfq",
      requiredMaxSeq: 2048,
      maxResidentWorkers: 2,
      workers: [
        { workerKeyId: "a", modelPath: "/models/a.hfq", maxSeq: 4096, lastUsedAtMs: 10 },
        { workerKeyId: "b", modelPath: "/models/b.hfq", maxSeq: 4096, lastUsedAtMs: 20 },
      ],
    });
    expect(decision.action).toBe("reuse");
    expect(decision.routeReason).toBe("worker_reused");
    expect(decision.workerKeyId).toBe("a");
  });

  test("loads second worker under resident cap", () => {
    const decision = pickResidentModelWorker({
      requestModelPath: "/models/b.hfq",
      requiredMaxSeq: 2048,
      maxResidentWorkers: 2,
      workers: [
        { workerKeyId: "a", modelPath: "/models/a.hfq", maxSeq: 4096, lastUsedAtMs: 10 },
      ],
    });
    expect(decision.action).toBe("load");
    expect(decision.routeReason).toBe("worker_loaded");
  });

  test("evicts largest idle worker over cap and uses LRU as tie-breaker", () => {
    const decision = pickResidentModelWorker({
      requestModelPath: "/models/c.hfq",
      requiredMaxSeq: 2048,
      maxResidentWorkers: 2,
      workers: [
        { workerKeyId: "a", modelPath: "/models/a.hfq", maxSeq: 4096, lastUsedAtMs: 10, totalResidentBytes: 100 },
        { workerKeyId: "b", modelPath: "/models/b.hfq", maxSeq: 4096, lastUsedAtMs: 20, totalResidentBytes: 200 },
      ],
    });
    expect(decision.action).toBe("evict_and_load");
    expect(decision.routeReason).toBe("worker_evicted_lru");
    expect(decision.evictWorkerKeyId).toBe("b");
  });

  test("rejects when all resident workers are active", () => {
    const decision = pickResidentModelWorker({
      requestModelPath: "/models/c.hfq",
      requiredMaxSeq: 2048,
      maxResidentWorkers: 2,
      workers: [
        { workerKeyId: "a", modelPath: "/models/a.hfq", maxSeq: 4096, lastUsedAtMs: 10, active: true },
        { workerKeyId: "b", modelPath: "/models/b.hfq", maxSeq: 4096, lastUsedAtMs: 20, active: true },
      ],
    });
    expect(decision.action).toBe("reject");
    expect(decision.routeReason).toBe("worker_cap_exhausted");
  });
});
