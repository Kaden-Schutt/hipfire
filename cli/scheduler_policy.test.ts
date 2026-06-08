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

  test("legacy server batch max feeds worker scheduler max", () => {
    const policy = schedulerPolicyForPriority(64, {
      HIPFIRE_SERVER_PREFILL_BATCH_MAX: "2",
    });
    expect(policy.maxBatchSize).toBe(2);
    expect(policy.residentStateMax).toBe(2);
    expect(policy.spillableBatchMax).toBe(2);
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

  test("state residency defaults to the effective class batch limit", () => {
    const realtime = schedulerPolicyForPriority(0, {
      HIPFIRE_SCHED_PREFILL_BATCH_MAX: "16",
    });
    expect(realtime.maxBatchSize).toBe(1);
    expect(realtime.residentStateMax).toBe(1);
    expect(realtime.spillableBatchMax).toBe(1);

    const high = schedulerPolicyForPriority(1, {
      HIPFIRE_SCHED_PREFILL_BATCH_MAX: "16",
    });
    expect(high.maxBatchSize).toBe(4);
    expect(high.residentStateMax).toBe(4);
    expect(high.spillableBatchMax).toBe(4);
  });

  test("disk spill is priority gated when state-cache disk is enabled", () => {
    expect(schedulerPolicyForPriority(64, {
      HIPFIRE_SCHED_STATE_CACHE_DISK: "1",
    }).diskSpillAllowed).toBe(false);

    expect(schedulerPolicyForPriority(128, {
      HIPFIRE_SCHED_STATE_CACHE_DISK: "1",
    }).diskSpillAllowed).toBe(true);

    expect(schedulerPolicyForPriority(255, {
      HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK: "true",
    }).diskSpillAllowed).toBe(true);
  });

  test("state residency and spillable batch limits are configurable and clamped", () => {
    const policy = schedulerPolicyForPriority(64, {
      HIPFIRE_SCHED_PREFILL_BATCH_MAX: "8",
      HIPFIRE_SCHED_RESIDENT_STATE_MAX: "3",
      HIPFIRE_SCHED_SPILLABLE_BATCH_MAX: "12",
      HIPFIRE_SCHED_STATE_CACHE_DISK: "1",
      HIPFIRE_SCHED_STATE_CACHE_DISK_MIN_PRIORITY: "64",
    });
    expect(policy.maxBatchSize).toBe(8);
    expect(policy.residentStateMax).toBe(3);
    expect(policy.spillableBatchMax).toBe(12);
    expect(policy.diskSpillMinPriority).toBe(64);
    expect(policy.diskSpillAllowed).toBe(true);

    const clamped = schedulerPolicyForPriority(64, {
      HIPFIRE_SCHED_RESIDENT_STATE_MAX: "80",
      HIPFIRE_SCHED_SPILLABLE_BATCH_MAX: "2",
    });
    expect(clamped.residentStateMax).toBe(64);
    expect(clamped.spillableBatchMax).toBe(64);
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
