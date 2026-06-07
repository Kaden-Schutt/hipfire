import { describe, expect, test } from "bun:test";
import {
  createRequestSessionDraft,
  modelWorkerKeyId,
  sameModelWorkerKey,
  sessionsCompatibleForPrefill,
  type ModelWorkerKey,
} from "./session_state";

const baseWorker: ModelWorkerKey = {
  artifactPath: "/models/qwen3.6-35b-a3b-mq4.hfq",
  artifactDigest: "sha256:a3b",
  archId: 6,
  quantFamily: "mq4",
  stateMode: "q8+deltanet",
  maxSeqBucket: 4096,
  featureFlags: ["prefill_batch", "qwen35"],
};

describe("model worker keys", () => {
  test("normalizes feature flag order in the stable key id", () => {
    const shuffled = {
      ...baseWorker,
      featureFlags: ["qwen35", "prefill_batch"],
    };
    expect(modelWorkerKeyId(baseWorker)).toBe(modelWorkerKeyId(shuffled));
    expect(sameModelWorkerKey(baseWorker, shuffled)).toBe(true);
  });

  test("separates different model workers", () => {
    expect(sameModelWorkerKey(baseWorker, {
      ...baseWorker,
      artifactDigest: "sha256:dense",
      artifactPath: "/models/qwen3.6-27b-mq4.hfq",
    })).toBe(false);
  });
});

describe("request session drafts", () => {
  test("splits cached prefix from suffix and initializes state position", () => {
    const session = createRequestSessionDraft({
      id: "r1",
      workerKey: baseWorker,
      promptTokens: [10, 11, 12, 13],
      cachedPrefixTokens: 2,
      priority: 64,
      stateKinds: ["attention_kv", "deltanet_recurrent"],
    });

    expect(session.cachedPrefixTokens).toBe(2);
    expect(session.suffixTokens).toEqual([12, 13]);
    expect(session.stateHandle.logicalPosition).toBe(2);
    expect(session.stateHandle.cachedPrefixTokens).toBe(2);
  });

  test("clamps cached prefix and priority", () => {
    const session = createRequestSessionDraft({
      id: "r2",
      workerKey: baseWorker,
      promptTokens: [10, 11],
      cachedPrefixTokens: 20,
      priority: 999,
      stateKinds: ["attention_kv"],
    });

    expect(session.cachedPrefixTokens).toBe(2);
    expect(session.suffixTokens).toEqual([]);
    expect(session.priority).toBe(255);
  });
});

describe("prefill compatibility", () => {
  test("requires same worker and same state kinds", () => {
    const a = createRequestSessionDraft({
      id: "a",
      workerKey: baseWorker,
      promptTokens: [1, 2, 3],
      stateKinds: ["attention_kv", "deltanet_recurrent"],
    });
    const b = createRequestSessionDraft({
      id: "b",
      workerKey: { ...baseWorker, featureFlags: ["qwen35", "prefill_batch"] },
      promptTokens: [4, 5],
      stateKinds: ["deltanet_recurrent", "attention_kv"],
    });
    const c = createRequestSessionDraft({
      id: "c",
      workerKey: { ...baseWorker, artifactDigest: "sha256:other" },
      promptTokens: [6],
      stateKinds: ["attention_kv", "deltanet_recurrent"],
    });
    const d = createRequestSessionDraft({
      id: "d",
      workerKey: baseWorker,
      promptTokens: [7],
      stateKinds: ["attention_kv", "mamba_ssm"],
    });

    expect(sessionsCompatibleForPrefill({ a, b })).toBe(true);
    expect(sessionsCompatibleForPrefill({ a, b: c })).toBe(false);
    expect(sessionsCompatibleForPrefill({ a, b: d })).toBe(false);
  });
});
