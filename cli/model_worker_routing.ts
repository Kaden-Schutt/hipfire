import {
  normalizeModelWorkerKey,
  type ModelWorkerKey,
} from "./session_state";

export type WorkerReloadReason =
  | "none"
  | "model_mismatch"
  | "max_seq_growth"
  | "worker_reused"
  | "worker_loaded"
  | "worker_evicted_lru"
  | "worker_cap_exhausted";

export interface ServingModelWorkerInput {
  requestModelPath: string;
  currentModelPath: string | null;
  currentMaxSeq: number | null;
  requiredMaxSeq: number;
  archId: number | string;
  quantFamily: string;
  stateMode: string;
  featureFlags: readonly string[];
  artifactDigest?: string;
  maxSeqBucket?: number;
  acceleratorKind?: string;
  deviceId?: number | string;
}

export interface ServingModelWorkerDecision {
  needsReload: boolean;
  reloadReason: WorkerReloadReason;
  workerKey: ModelWorkerKey;
  canReuseCurrentWorker: boolean;
}

export interface ResidentWorkerCandidate {
  workerKeyId: string;
  modelPath: string;
  maxSeq: number;
  lastUsedAtMs: number;
  totalResidentBytes?: number;
  active?: boolean;
}

export interface ResidentWorkerRoutingDecision {
  action: "reuse" | "load" | "evict_and_load" | "reject";
  routeReason: WorkerReloadReason;
  workerKeyId?: string;
  evictWorkerKeyId?: string;
}

export function pickServingModelWorker(input: ServingModelWorkerInput): ServingModelWorkerDecision {
  const modelMismatch = input.currentModelPath !== input.requestModelPath;
  const maxSeqGrowth = !modelMismatch
    && input.currentMaxSeq !== null
    && input.requiredMaxSeq > input.currentMaxSeq;
  const needsReload = modelMismatch || maxSeqGrowth;
  const workerKey = normalizeModelWorkerKey({
    artifactPath: input.requestModelPath,
    artifactDigest: input.artifactDigest,
    archId: input.archId,
    quantFamily: input.quantFamily,
    stateMode: input.stateMode,
    maxSeqBucket: input.maxSeqBucket ?? input.currentMaxSeq ?? input.requiredMaxSeq,
    acceleratorKind: input.acceleratorKind,
    deviceId: input.deviceId,
    featureFlags: input.featureFlags,
  });

  return {
    needsReload,
    reloadReason: needsReload
      ? (modelMismatch ? "model_mismatch" : "max_seq_growth")
      : "none",
    workerKey,
    canReuseCurrentWorker: !needsReload,
  };
}

export function pickResidentModelWorker(input: {
  requestModelPath: string;
  requiredMaxSeq: number;
  maxResidentWorkers: number;
  workers: readonly ResidentWorkerCandidate[];
}): ResidentWorkerRoutingDecision {
  const reusable = input.workers
    .filter((worker) => worker.modelPath === input.requestModelPath && worker.maxSeq >= input.requiredMaxSeq)
    .sort((a, b) => a.maxSeq - b.maxSeq || b.lastUsedAtMs - a.lastUsedAtMs)[0];
  if (reusable) {
    return {
      action: "reuse",
      routeReason: "worker_reused",
      workerKeyId: reusable.workerKeyId,
    };
  }
  if (input.workers.length < Math.max(1, input.maxResidentWorkers)) {
    return {
      action: "load",
      routeReason: "worker_loaded",
    };
  }
  const evictable = input.workers
    .filter((worker) => worker.active !== true)
    .sort((a, b) => (b.totalResidentBytes ?? 0) - (a.totalResidentBytes ?? 0) || a.lastUsedAtMs - b.lastUsedAtMs)[0];
  if (!evictable) {
    return {
      action: "reject",
      routeReason: "worker_cap_exhausted",
    };
  }
  return {
    action: "evict_and_load",
    routeReason: "worker_evicted_lru",
    evictWorkerKeyId: evictable.workerKeyId,
  };
}
