export type SchedulerPriority = number;

export type SchedulerPriorityClass =
  | "realtime"
  | "high"
  | "interactive"
  | "background"
  | "bulk"
  | "opportunistic";

export interface SchedulerPriorityClassLimits {
  realtime: [number, number];
  high: [number, number];
  interactive: [number, number];
  background: [number, number];
  bulk: [number, number];
  opportunistic: [number, number];
}

export const SCHEDULER_PRIORITY_CLASS_LIMITS: SchedulerPriorityClassLimits = {
  realtime: [0, 0],
  high: [1, 63],
  interactive: [64, 127],
  background: [128, 191],
  bulk: [192, 254],
  opportunistic: [255, 255],
};

export interface SchedulerPriorityPolicy {
  priority: SchedulerPriority;
  priorityClass: SchedulerPriorityClass;
  coalesceWaitMs: number;
  maxBatchSize: number;
  targetPairTokens: number;
  maxProcessingMs: number;
}

export interface SchedulerPolicyEnv {
  HIPFIRE_SCHED_PRIORITY_DEFAULT?: string;
  HIPFIRE_SCHED_PREFILL_BATCH_MAX?: string;
  HIPFIRE_SCHED_PREFILL_WAIT_MS_REALTIME?: string;
  HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE?: string;
  HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND?: string;
  HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS?: string;
  HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS?: string;
  HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK?: string;
  [key: string]: string | undefined;
}

export interface OpportunisticDispatchInput {
  compatibleQueuedTokens: number;
  scheduleClear: boolean;
  targetPairTokens: number;
}

export interface ServerPrefillPolicyControls {
  stateCacheDisk: boolean;
}

export const SCHED_PRIORITY_REALTIME = 0;
export const SCHED_PRIORITY_DEFAULT = 64;
export const SCHED_PRIORITY_OPPORTUNISTIC = 255;

function parseInteger(value: string | undefined, fallback: number): number {
  if (value === undefined || value.trim() === "") return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.floor(parsed) : fallback;
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  const v = value.trim().toLowerCase();
  return v === "1" || v === "true" || v === "on" || v === "yes";
}

export function clampSchedulerPriority(value: number): SchedulerPriority {
  if (!Number.isFinite(value)) return SCHED_PRIORITY_DEFAULT;
  return Math.max(0, Math.min(255, Math.floor(value)));
}

export function parseSchedulerPriority(
  value: string | number | undefined,
  fallback = SCHED_PRIORITY_DEFAULT,
): SchedulerPriority {
  if (value === undefined) return clampSchedulerPriority(fallback);
  if (typeof value === "number") return clampSchedulerPriority(value);
  return clampSchedulerPriority(parseInteger(value, fallback));
}

export function parseDefaultSchedulerPriority(
  env: SchedulerPolicyEnv = process.env,
): SchedulerPriority {
  return parseSchedulerPriority(env.HIPFIRE_SCHED_PRIORITY_DEFAULT, SCHED_PRIORITY_DEFAULT);
}

export function schedulerPriorityClass(priority: SchedulerPriority): SchedulerPriorityClass {
  const p = clampSchedulerPriority(priority);
  if (p >= SCHEDULER_PRIORITY_CLASS_LIMITS.realtime[0] && p <= SCHEDULER_PRIORITY_CLASS_LIMITS.realtime[1]) return "realtime";
  if (p >= SCHEDULER_PRIORITY_CLASS_LIMITS.high[0] && p <= SCHEDULER_PRIORITY_CLASS_LIMITS.high[1]) return "high";
  if (p >= SCHEDULER_PRIORITY_CLASS_LIMITS.interactive[0] && p <= SCHEDULER_PRIORITY_CLASS_LIMITS.interactive[1]) return "interactive";
  if (p >= SCHEDULER_PRIORITY_CLASS_LIMITS.background[0] && p <= SCHEDULER_PRIORITY_CLASS_LIMITS.background[1]) return "background";
  if (p >= SCHEDULER_PRIORITY_CLASS_LIMITS.bulk[0] && p <= SCHEDULER_PRIORITY_CLASS_LIMITS.bulk[1]) return "bulk";
  return "opportunistic";
}

export function parseServerPrefillPolicyControls(
  env: SchedulerPolicyEnv = process.env,
): ServerPrefillPolicyControls {
  const stateCacheDisk = parseBoolean(
    env.HIPFIRE_SCHED_STATE_CACHE_DISK,
    parseBoolean(env.HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK, false),
  );
  const legacyStateCacheDisk = parseBoolean(env.HIPFIRE_SERVER_PREFILL_BATCH_STATE_CACHE_DISK, false);
  return { stateCacheDisk: stateCacheDisk || legacyStateCacheDisk };
}

export function schedulerPolicyForPriority(
  priority: SchedulerPriority,
  env: SchedulerPolicyEnv = process.env,
): SchedulerPriorityPolicy {
  const p = clampSchedulerPriority(priority);
  const priorityClass = schedulerPriorityClass(p);
  const maxBatchSize = Math.max(
    1,
    Math.min(64, parseInteger(env.HIPFIRE_SCHED_PREFILL_BATCH_MAX, 8)),
  );
  const legacyInteractiveWait = env.HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS;
  const realtimeWait = Math.max(0, parseInteger(env.HIPFIRE_SCHED_PREFILL_WAIT_MS_REALTIME, 0));
  const interactiveWait = Math.max(
    0,
    parseInteger(env.HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE, legacyInteractiveWait ? parseInteger(legacyInteractiveWait, 5) : 5),
  );
  const legacyBackgroundWait = legacyInteractiveWait === undefined
    ? 25
    : Math.max(0, parseInteger(legacyInteractiveWait, 25) * 2);
  const backgroundWait = Math.max(
    0,
    parseInteger(env.HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND, legacyBackgroundWait),
  );
  const opportunisticBackgroundWait = Math.max(
    0,
    parseInteger(env.HIPFIRE_SCHED_PREFILL_WAIT_MS_BACKGROUND, 25),
  );
  const opportunisticPairTokens = Math.max(
    1,
    parseInteger(env.HIPFIRE_SCHED_OPPORTUNISTIC_MIN_PAIR_TOKENS, 256),
  );

  switch (priorityClass) {
    case "realtime":
      return {
        priority: p,
        priorityClass,
        coalesceWaitMs: realtimeWait,
        maxBatchSize: 1,
        targetPairTokens: 1,
        maxProcessingMs: 25,
      };
    case "high":
      return {
        priority: p,
        priorityClass,
        coalesceWaitMs: Math.min(interactiveWait, 2),
        maxBatchSize: Math.min(maxBatchSize, 4),
        targetPairTokens: 32,
        maxProcessingMs: 50,
      };
    case "interactive":
      return {
        priority: p,
        priorityClass,
        coalesceWaitMs: interactiveWait,
        maxBatchSize,
        targetPairTokens: 64,
        maxProcessingMs: 100,
      };
    case "background":
      return {
        priority: p,
        priorityClass,
        coalesceWaitMs: backgroundWait,
        maxBatchSize,
        targetPairTokens: 128,
        maxProcessingMs: 250,
      };
    case "bulk":
      return {
        priority: p,
        priorityClass,
        coalesceWaitMs: backgroundWait * 2,
        maxBatchSize,
        targetPairTokens: opportunisticPairTokens,
        maxProcessingMs: 500,
      };
    case "opportunistic":
      return {
        priority: p,
        priorityClass,
        coalesceWaitMs: opportunisticBackgroundWait * 4,
        maxBatchSize,
        targetPairTokens: opportunisticPairTokens,
        maxProcessingMs: 1000,
      };
  }
}

export function shouldDispatchOpportunistic(input: OpportunisticDispatchInput): boolean {
  if (input.scheduleClear) return true;
  return input.compatibleQueuedTokens >= Math.max(1, input.targetPairTokens);
}
