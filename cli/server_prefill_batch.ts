import {
  parseDefaultSchedulerPriority,
  schedulerPolicyForPriority,
} from "./scheduler_policy";
import {
  createRequestSessionDraft,
  normalizeModelWorkerKey,
  type ModelWorkerKey,
  type RequestSessionDraft,
  type SessionStateKind,
} from "./session_state";

export interface ServerPrefillBatchPolicy {
  enabled: boolean;
  priority: number;
  maxBatch: number;
  waitMs: number;
  targetPairTokens: number;
  maxProcessingMs: number;
}

export interface ServerPrefillBatchEffectiveConfig {
  prefill_compression: string;
  prefill_drafter: string;
}

export interface ServerPrefillBatchEligibilityInput {
  body: any;
  loadedModelPath: string | null;
  requestModelPath: string | null;
  loadedMaxSeq: number | null;
  requiredMaxSeq: number;
  requestImages: string[];
  effectiveConfig: ServerPrefillBatchEffectiveConfig;
}

export interface ServerPrefillSessionInput {
  id: string;
  modelPath: string;
  modelDigest?: string;
  archId: number | string;
  quantFamily: string;
  stateMode: string;
  maxSeqBucket: number;
  featureFlags?: readonly string[];
  promptTokens: readonly number[];
  cachedPrefixTokens?: number;
  priority?: number;
  stateKinds: readonly SessionStateKind[];
}

export function parseServerPrefillBatchPolicy(
  env: Record<string, string | undefined> = process.env,
): ServerPrefillBatchPolicy {
  const flag = env.HIPFIRE_SERVER_PREFILL_BATCH?.toLowerCase();
  const enabled = flag === "1" || flag === "on" || flag === "true";
  const priority = parseDefaultSchedulerPriority(env);
  const schedulerPolicy = schedulerPolicyForPriority(priority, {
    ...env,
    HIPFIRE_SCHED_PREFILL_BATCH_MAX:
      env.HIPFIRE_SCHED_PREFILL_BATCH_MAX ?? env.HIPFIRE_SERVER_PREFILL_BATCH_MAX,
    HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE:
      env.HIPFIRE_SCHED_PREFILL_WAIT_MS_INTERACTIVE ?? env.HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS,
  });
  return {
    enabled,
    priority,
    maxBatch: schedulerPolicy.maxBatchSize,
    waitMs: schedulerPolicy.coalesceWaitMs,
    targetPairTokens: schedulerPolicy.targetPairTokens,
    maxProcessingMs: schedulerPolicy.maxProcessingMs,
  };
}

export function serverPrefillBatchEligibility(
  input: ServerPrefillBatchEligibilityInput,
): { eligible: boolean; reason: string } {
  const body = input.body ?? {};
  const tools = Array.isArray(body.tools) ? body.tools : [];
  if (tools.length > 0) return { eligible: false, reason: "tools" };
  if (input.requestImages.length > 0) return { eligible: false, reason: "image" };
  if (!input.loadedModelPath || input.loadedModelPath !== input.requestModelPath) {
    return { eligible: false, reason: "model_reload" };
  }
  if (input.loadedMaxSeq !== null && input.requiredMaxSeq > input.loadedMaxSeq) {
    return { eligible: false, reason: "max_seq_reload" };
  }
  if (input.effectiveConfig.prefill_compression !== "off" && input.effectiveConfig.prefill_drafter) {
    return { eligible: false, reason: "pflash" };
  }
  if (body.cask_sidecar || body.cask_budget || body.cask_beta) {
    return { eligible: false, reason: "cask" };
  }
  return { eligible: true, reason: "eligible" };
}

export function buildServerPrefillWorkerKey(
  input: Omit<
    ServerPrefillSessionInput,
    "id" | "promptTokens" | "cachedPrefixTokens" | "priority" | "stateKinds"
  >,
): ModelWorkerKey {
  return normalizeModelWorkerKey({
    artifactPath: input.modelPath,
    artifactDigest: input.modelDigest,
    archId: input.archId,
    quantFamily: input.quantFamily,
    stateMode: input.stateMode,
    maxSeqBucket: input.maxSeqBucket,
    featureFlags: input.featureFlags ?? [],
  });
}

export function createServerPrefillSession(
  input: ServerPrefillSessionInput,
): RequestSessionDraft {
  return createRequestSessionDraft({
    id: input.id,
    workerKey: buildServerPrefillWorkerKey(input),
    promptTokens: input.promptTokens,
    cachedPrefixTokens: input.cachedPrefixTokens,
    priority: input.priority,
    stateKinds: input.stateKinds,
  });
}
