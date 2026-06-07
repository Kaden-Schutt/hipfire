import {
  normalizeModelWorkerKey,
  type ModelWorkerKey,
} from "./session_state";

export type WorkerReloadReason =
  | "none"
  | "model_mismatch"
  | "max_seq_growth";

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
}

export interface ServingModelWorkerDecision {
  needsReload: boolean;
  reloadReason: WorkerReloadReason;
  workerKey: ModelWorkerKey;
  canReuseCurrentWorker: boolean;
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
