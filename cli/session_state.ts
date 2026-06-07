import {
  SCHED_PRIORITY_DEFAULT,
  type SchedulerPriority,
  clampSchedulerPriority,
} from "./scheduler_policy";

export type SessionStateKind =
  | "attention_kv"
  | "deltanet_recurrent"
  | "mamba_ssm"
  | "mamba_conv"
  | "architecture_specific";

export interface ModelWorkerKey {
  artifactPath: string;
  artifactDigest?: string;
  archId: number | string;
  quantFamily: string;
  stateMode: string;
  maxSeqBucket: number;
  featureFlags: readonly string[];
}

export interface SessionStateHandle {
  workerKey: ModelWorkerKey;
  stateKinds: readonly SessionStateKind[];
  logicalPosition: number;
  cachedPrefixTokens: number;
}

export interface RequestSessionDraft {
  id: string;
  workerKey: ModelWorkerKey;
  priority: SchedulerPriority;
  promptTokens: readonly number[];
  suffixTokens: readonly number[];
  cachedPrefixTokens: number;
  stateHandle: SessionStateHandle;
}

export interface CreateRequestSessionInput {
  id: string;
  workerKey: ModelWorkerKey;
  promptTokens: readonly number[];
  cachedPrefixTokens?: number;
  priority?: number;
  stateKinds: readonly SessionStateKind[];
}

export interface PrefillCompatibilityInput {
  a: RequestSessionDraft;
  b: RequestSessionDraft;
}

export function normalizeFeatureFlags(flags: readonly string[]): readonly string[] {
  return [...new Set(flags)].sort();
}

export function normalizeModelWorkerKey(key: ModelWorkerKey): ModelWorkerKey {
  return {
    ...key,
    featureFlags: normalizeFeatureFlags(key.featureFlags),
  };
}

export function modelWorkerKeyId(key: ModelWorkerKey): string {
  const normalized = normalizeModelWorkerKey(key);
  return [
    normalized.artifactDigest || normalized.artifactPath,
    normalized.archId,
    normalized.quantFamily,
    normalized.stateMode,
    normalized.maxSeqBucket,
    normalized.featureFlags.join("+"),
  ].join("|");
}

export function sameModelWorkerKey(a: ModelWorkerKey, b: ModelWorkerKey): boolean {
  return modelWorkerKeyId(a) === modelWorkerKeyId(b);
}

export function createRequestSessionDraft(
  input: CreateRequestSessionInput,
): RequestSessionDraft {
  const cachedPrefixTokens = Math.max(
    0,
    Math.min(input.promptTokens.length, Math.floor(input.cachedPrefixTokens ?? 0)),
  );
  const workerKey = normalizeModelWorkerKey(input.workerKey);
  const suffixTokens = input.promptTokens.slice(cachedPrefixTokens);
  return {
    id: input.id,
    workerKey,
    priority: clampSchedulerPriority(input.priority ?? SCHED_PRIORITY_DEFAULT),
    promptTokens: input.promptTokens,
    suffixTokens,
    cachedPrefixTokens,
    stateHandle: {
      workerKey,
      stateKinds: input.stateKinds,
      logicalPosition: cachedPrefixTokens,
      cachedPrefixTokens,
    },
  };
}

export function sessionsCompatibleForPrefill(
  input: PrefillCompatibilityInput,
): boolean {
  if (!sameModelWorkerKey(input.a.workerKey, input.b.workerKey)) return false;
  const aKinds = [...input.a.stateHandle.stateKinds].sort().join("|");
  const bKinds = [...input.b.stateHandle.stateKinds].sort().join("|");
  return aKinds === bKinds;
}
