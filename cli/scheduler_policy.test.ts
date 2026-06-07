import { describe, expect, test } from "bun:test";
import {
  clampSchedulerPriority,
  parseDefaultSchedulerPriority,
  parseSchedulerPriority,
  schedulerPolicyForPriority,
  schedulerPriorityClass,
  shouldDispatchOpportunistic,
} from "./scheduler_policy";

describe("scheduler priority parsing", () => {
  test("clamps to 256 priority levels", () => {
    expect(clampSchedulerPriority(-1)).toBe(0);
    expect(clampSchedulerPriority(0)).toBe(0);
    expect(clampSchedulerPriority(64.9)).toBe(64);
    expect(clampSchedulerPriority(255)).toBe(255);
    expect(clampSchedulerPriority(999)).toBe(255);
  });

  test("uses interactive default priority", () => {
    expect(parseSchedulerPriority(undefined)).toBe(64);
    expect(parseSchedulerPriority(null)).toBe(64);
    expect(parseDefaultSchedulerPriority({})).toBe(64);
    expect(parseDefaultSchedulerPriority({ HIPFIRE_SCHED_PRIORITY_DEFAULT: "192" })).toBe(192);
    expect(parseSchedulerPriority("not-a-number")).toBe(64);
  });
});

describe("scheduler priority classes", () => {
  test("maps all priority bands deterministically", () => {
    expect(schedulerPriorityClass(0)).toBe("realtime");
    expect(schedulerPriorityClass(1)).toBe("high");
    expect(schedulerPriorityClass(63)).toBe("high");
    expect(schedulerPriorityClass(64)).toBe("interactive");
    expect(schedulerPriorityClass(127)).toBe("interactive");
    expect(schedulerPriorityClass(128)).toBe("background");
    expect(schedulerPriorityClass(191)).toBe("background");
    expect(schedulerPriorityClass(192)).toBe("bulk");
    expect(schedulerPriorityClass(254)).toBe("bulk");
    expect(schedulerPriorityClass(255)).toBe("opportunistic");
  });
});

describe("scheduler policy", () => {
  test("realtime dispatches without coalescing and with a small quantum", () => {
    const policy = schedulerPolicyForPriority(0, {});
    expect(policy.priorityClass).toBe("realtime");
    expect(policy.coalesceWaitMs).toBe(0);
    expect(policy.maxBatchSize).toBe(1);
    expect(policy.maxProcessingMs).toBeLessThan(schedulerPolicyForPriority(64, {}).maxProcessingMs);
  });

  test("interactive default uses configured batch and wait", () => {
    const policy = schedulerPolicyForPriority(64, {
      HIPFIRE_SCHED_PREFILL_BATCH_MAX: "16",
      HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE: "7",
    });
    expect(policy.priorityClass).toBe("interactive");
    expect(policy.maxBatchSize).toBe(16);
    expect(policy.coalesceWaitMs).toBe(7);
    expect(policy.targetPairTokens).toBe(64);
  });

  test("legacy batch-wait env maps to interactive and backgrounds to a stable default", () => {
    const legacyOnly = schedulerPolicyForPriority(64, {
      HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS: "9",
    });
    expect(legacyOnly.coalesceWaitMs).toBe(9);

    const legacyBackground = schedulerPolicyForPriority(128, {
      HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS: "9",
    });
    expect(legacyBackground.coalesceWaitMs).toBe(18);
  });

  test("opportunistic uses configured pairing target and largest wait", () => {
    const policy = schedulerPolicyForPriority(255, {
      HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND: "20",
      HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS: "512",
    });
    expect(policy.priorityClass).toBe("opportunistic");
    expect(policy.coalesceWaitMs).toBe(80);
    expect(policy.targetPairTokens).toBe(512);
    expect(policy.maxProcessingMs).toBeGreaterThan(schedulerPolicyForPriority(128, {}).maxProcessingMs);
  });
});

describe("opportunistic dispatch", () => {
  test("waits for compatible paired work unless schedule is clear", () => {
    expect(shouldDispatchOpportunistic({
      compatibleQueuedTokens: 255,
      scheduleClear: false,
      targetPairTokens: 256,
    })).toBe(false);
    expect(shouldDispatchOpportunistic({
      compatibleQueuedTokens: 256,
      scheduleClear: false,
      targetPairTokens: 256,
    })).toBe(true);
    expect(shouldDispatchOpportunistic({
      compatibleQueuedTokens: 0,
      scheduleClear: true,
      targetPairTokens: 256,
    })).toBe(true);
  });
});
