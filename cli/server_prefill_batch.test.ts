import { describe, expect, test } from "bun:test";
import {
  buildServerPrefillWorkerKey,
  createServerPrefillSession,
  parseServerPrefillBatchPolicy,
  serverPrefillBatchEligibility,
} from "./server_prefill_batch";
import { parseServerPrefillPolicyControls } from "./scheduler_policy";

const effectiveConfig = {
  prefill_compression: "off" as const,
  prefill_drafter: "",
};

describe("server prefill batching policy", () => {
  test("defaults off with conservative max and wait", () => {
    expect(parseServerPrefillBatchPolicy({})).toEqual({
      enabled: false,
      priority: 64,
      maxBatch: 8,
      waitMs: 5,
      targetPairTokens: 64,
      maxProcessingMs: 100,
    });
  });

  test("legacy env enables and scheduler aliases control policy", () => {
    expect(parseServerPrefillBatchPolicy({
      HIPFIRE_SERVER_PREFILL_BATCH: "1",
      HIPFIRE_SERVER_PREFILL_BATCH_MAX: "128",
      HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS: "9",
      HIPFIRE_SCHED_PRIORITY_DEFAULT: "255",
      HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS: "512",
    })).toEqual({
      enabled: true,
      maxBatch: 64,
      maxProcessingMs: 1000,
      priority: 255,
      targetPairTokens: 512,
      waitMs: 100,
    });
  });

  test("legacy cache-control env enables state cache disk when new and old toggles are parsed", () => {
    expect(parseServerPrefillPolicyControls({
      HIPFIRE_SCHED_STATE_CACHE_DISK: "1",
    })).toEqual({ stateCacheDisk: true });
    expect(parseServerPrefillPolicyControls({
      HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK: "1",
    })).toEqual({ stateCacheDisk: true });
    expect(parseServerPrefillPolicyControls({
      HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK: "1",
      HIPFIRE_SCHED_STATE_CACHE_DISK: "0",
    })).toEqual({ stateCacheDisk: true });
  });
});

describe("server prefill batching eligibility", () => {
  const base = {
    body: { model: "qwen3.6:35b-a3b", messages: [{ role: "user", content: "hello" }] },
    loadedModelPath: "/models/qwen3.6-35b-a3b-mq4.hfq",
    requestModelPath: "/models/qwen3.6-35b-a3b-mq4.hfq",
    loadedMaxSeq: 4096,
    requiredMaxSeq: 2048,
    requestImages: [],
    effectiveConfig,
  };

  test("admits same-model text-only AR requests", () => {
    expect(serverPrefillBatchEligibility(base)).toEqual({
      eligible: true,
      reason: "eligible",
    });
  });

  test("falls back when a model reload would be required", () => {
    expect(serverPrefillBatchEligibility({
      ...base,
      requestModelPath: "/models/other.hfq",
    })).toEqual({
      eligible: false,
      reason: "model_reload",
    });
  });

  test("falls back for tools, images, PFlash, CASK, and max-seq growth", () => {
    expect(serverPrefillBatchEligibility({
      ...base,
      body: { ...base.body, tools: [{ type: "function" }] },
    }).reason).toBe("tools");
    expect(serverPrefillBatchEligibility({
      ...base,
      requestImages: ["base64"],
    }).reason).toBe("image");
    expect(serverPrefillBatchEligibility({
      ...base,
      effectiveConfig: { prefill_compression: "auto", prefill_drafter: "/draft.hfq" },
    }).reason).toBe("pflash");
    expect(serverPrefillBatchEligibility({
      ...base,
      body: { ...base.body, cask_budget: 1024 },
    }).reason).toBe("cask");
    expect(serverPrefillBatchEligibility({
      ...base,
      requiredMaxSeq: 8192,
    }).reason).toBe("max_seq_reload");
  });
});

describe("server prefill session adapter", () => {
  test("builds normalized worker keys and isolated request sessions", () => {
    const workerKey = buildServerPrefillWorkerKey({
      modelPath: "/models/qwen3.6-35b-a3b-mq4.hfq",
      modelDigest: "sha256:a3b",
      archId: 6,
      quantFamily: "mq4",
      stateMode: "q8+deltanet",
      maxSeqBucket: 4096,
      featureFlags: ["qwen35", "prefill_batch"],
    });
    const session = createServerPrefillSession({
      id: "req-1",
      modelPath: workerKey.artifactPath,
      modelDigest: workerKey.artifactDigest,
      archId: workerKey.archId,
      quantFamily: workerKey.quantFamily,
      stateMode: workerKey.stateMode,
      maxSeqBucket: workerKey.maxSeqBucket,
      featureFlags: ["prefill_batch", "qwen35"],
      promptTokens: [1, 2, 3, 4, 5],
      cachedPrefixTokens: 3,
      priority: 128,
      stateKinds: ["attention_kv", "deltanet_recurrent"],
    });

    expect(workerKey.featureFlags).toEqual(["prefill_batch", "qwen35"]);
    expect(session.workerKey).toEqual(workerKey);
    expect(session.suffixTokens).toEqual([4, 5]);
    expect(session.cachedPrefixTokens).toBe(3);
    expect(session.stateHandle.stateKinds).toEqual(["attention_kv", "deltanet_recurrent"]);
    expect(session.priority).toBe(128);
  });
});
