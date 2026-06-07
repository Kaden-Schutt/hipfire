import { describe, expect, test } from "bun:test";
import {
  PriorityPrefillScheduler,
  type PrefillBatchSelection,
} from "./worker_scheduler";
import {
  createRequestSessionDraft,
  type ModelWorkerKey,
  type RequestSessionDraft,
  type SessionStateKind,
} from "./session_state";

const qwenWorker: ModelWorkerKey = {
  artifactPath: "/models/qwen3.6-35b-a3b-mq4.hfq",
  artifactDigest: "sha256:qwen-a3b",
  archId: 6,
  quantFamily: "mq4",
  stateMode: "q8+deltanet",
  maxSeqBucket: 4096,
  featureFlags: ["prefill_batch", "qwen35"],
};

const nemotronWorker: ModelWorkerKey = {
  artifactPath: "/models/nemotron-3-ultra-550b-a55b-bf16.hfq",
  artifactDigest: "sha256:nemotron",
  archId: "nemotron3",
  quantFamily: "bf16",
  stateMode: "q8+mamba",
  maxSeqBucket: 8192,
  featureFlags: ["mamba", "prefill_batch"],
};

function session(
  id: string,
  options: {
    priority?: number;
    tokens?: number;
    workerKey?: ModelWorkerKey;
    stateKinds?: readonly SessionStateKind[];
    cachedPrefixTokens?: number;
  } = {},
): RequestSessionDraft {
  const tokens = Array.from({ length: options.tokens ?? 16 }, (_, i) => i + 1);
  return createRequestSessionDraft({
    id,
    workerKey: options.workerKey ?? qwenWorker,
    promptTokens: tokens,
    cachedPrefixTokens: options.cachedPrefixTokens,
    priority: options.priority ?? 64,
    stateKinds: options.stateKinds ?? ["attention_kv", "deltanet_recurrent"],
  });
}

function ids(batch: PrefillBatchSelection | undefined): string[] {
  return batch?.sessions.map((s) => s.id) ?? [];
}

describe("priority prefill scheduler", () => {
  test("dispatches the highest-priority ready request first", () => {
    const scheduler = new PriorityPrefillScheduler({});
    scheduler.enqueue(session("interactive", { priority: 64 }), 0);
    scheduler.enqueue(session("high", { priority: 1 }), 0);

    expect(ids(scheduler.nextPrefillBatch({ nowMs: 5 }))).toEqual(["high"]);
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 5 }))).toEqual(["interactive"]);
  });

  test("realtime requests dispatch immediately as singletons", () => {
    const scheduler = new PriorityPrefillScheduler({
      HIPFIRE_SCHED_PREFILL_BATCH_MAX: "8",
    });
    scheduler.enqueue(session("rt-a", { priority: 0 }), 10);
    scheduler.enqueue(session("rt-b", { priority: 0 }), 10);

    const batch = scheduler.nextPrefillBatch({ nowMs: 10 });
    expect(ids(batch)).toEqual(["rt-a"]);
    expect(batch?.policy.priorityClass).toBe("realtime");
    expect(batch?.policy.maxBatchSize).toBe(1);
  });

  test("interactive requests coalesce until the wait window or max batch", () => {
    const scheduler = new PriorityPrefillScheduler({
      HIPFIRE_SCHED_PREFILL_BATCH_MAX: "3",
      HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE: "5",
    });
    scheduler.enqueue(session("a", { priority: 64 }), 100);
    scheduler.enqueue(session("b", { priority: 64 }), 102);

    expect(scheduler.nextPrefillBatch({ nowMs: 103 })).toBeUndefined();
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 105 }))).toEqual(["a", "b"]);

    scheduler.enqueue(session("c", { priority: 64 }), 200);
    scheduler.enqueue(session("d", { priority: 64 }), 200);
    scheduler.enqueue(session("e", { priority: 64 }), 200);
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 200 }))).toEqual(["c", "d", "e"]);
  });

  test("does not batch incompatible model workers or state kinds", () => {
    const scheduler = new PriorityPrefillScheduler({
      HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE: "0",
    });
    scheduler.enqueue(session("qwen", { priority: 64 }), 0);
    scheduler.enqueue(session("nemotron", {
      priority: 64,
      workerKey: nemotronWorker,
      stateKinds: ["attention_kv", "mamba_ssm", "mamba_conv"],
    }), 0);
    scheduler.enqueue(session("qwen-mamba-state", {
      priority: 64,
      stateKinds: ["attention_kv", "mamba_ssm"],
    }), 0);

    expect(ids(scheduler.nextPrefillBatch({ nowMs: 0 }))).toEqual(["qwen"]);
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 0 }))).toEqual(["nemotron"]);
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 0 }))).toEqual(["qwen-mamba-state"]);
  });

  test("opportunistic requests run unpaired only when the schedule is clear", () => {
    const clearScheduler = new PriorityPrefillScheduler({
      HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS: "32",
    });
    clearScheduler.enqueue(session("op-clear", { priority: 255, tokens: 64, cachedPrefixTokens: 56 }), 0);
    expect(ids(clearScheduler.nextPrefillBatch({ nowMs: 1 }))).toEqual(["op-clear"]);

    const blockedScheduler = new PriorityPrefillScheduler({
      HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE: "5",
      HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS: "32",
    });
    blockedScheduler.enqueue(session("interactive", { priority: 64, tokens: 8 }), 0);
    blockedScheduler.enqueue(session("op-a", { priority: 255, tokens: 64, cachedPrefixTokens: 56 }), 0);
    expect(blockedScheduler.nextPrefillBatch({ nowMs: 1 })).toBeUndefined();

    blockedScheduler.enqueue(session("op-b", { priority: 255, tokens: 64, cachedPrefixTokens: 40 }), 1);
    expect(ids(blockedScheduler.nextPrefillBatch({ nowMs: 5 }))).toEqual(["interactive"]);
    const paired = blockedScheduler.nextPrefillBatch({ nowMs: 5 });
    expect(ids(paired)).toEqual(["op-a", "op-b"]);
    expect(paired?.totalSuffixTokens).toBe(32);
  });

  test("higher-priority queued work prevents opportunistic clear dispatch", () => {
    const scheduler = new PriorityPrefillScheduler({
      HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE: "5",
      HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS: "64",
    });
    scheduler.enqueue(session("interactive", { priority: 64, tokens: 8 }), 0);
    scheduler.enqueue(session("op", { priority: 255, tokens: 8 }), 0);

    expect(scheduler.nextPrefillBatch({ nowMs: 2 })).toBeUndefined();
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 5 }))).toEqual(["interactive"]);
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 5 }))).toEqual(["op"]);
  });

  test("cancels queued requests by id", () => {
    const scheduler = new PriorityPrefillScheduler({
      HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE: "0",
    });
    scheduler.enqueue(session("a"), 0);
    scheduler.enqueue(session("b"), 0);

    expect(scheduler.cancel("a")).toBe(true);
    expect(scheduler.cancel("missing")).toBe(false);
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 0 }))).toEqual(["b"]);
  });

  test("can preview a batch for an incoming session without mutating the queue", () => {
    const scheduler = new PriorityPrefillScheduler({
      HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE: "0",
      HIPFIRE_SCHED_PREFILL_BATCH_MAX: "2",
    });
    scheduler.enqueue(session("a"), 0);

    const p = session("incoming", { priority: 64 });
    const preview = scheduler.previewNextPrefillBatch({ nowMs: 30, incomingSession: p });

    // Existing queue remains unchanged and preview can still reason about it.
    expect(ids(preview)).toEqual(["a", "incoming"]);
    const next = scheduler.nextPrefillBatch({ nowMs: 30 });
    expect(ids(next)).toEqual(["a"]);
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 0 }))).toEqual([]);
    expect(ids(scheduler.nextPrefillBatch({ nowMs: 0 }))).toEqual([]);
  });
});
