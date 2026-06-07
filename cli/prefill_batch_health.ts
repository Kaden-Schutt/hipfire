import type { GenerateBatchPrefillCapability } from "./generate_batch_prefill_protocol";

export type PrefillQueueWaitReason = "selected" | "waiting" | "insufficient_queue" | "not_eligible" | "disabled";
export type BatchExecutionMode = "prefill_batch" | "serial_fallback" | "unsupported" | "disabled";
export type BatchFallbackReason =
  | "idle"
  | "batch_in_progress"
  | "batch_validation_rejected"
  | "validation_failed"
  | "completed_with_prevalidation_rejections"
  | "scaffold_placeholder_completed"
  | "processing_error"
  | "line_fallback_not_enabled"
  | "line_fallback_not_eligible"
  | "line_fallback_disabled"
  | `line_fallback:${string}`
  | `line_rejected:${string}`;

export interface PrefillBatchHealthInputs {
  enabled: boolean;
  queued: number;
  eligible: number;
  selected: number;
  skipped: number;
  cacheHits: number;
  cacheMisses: number;
  queueSize: number;
  generateBatchPrefillCapability: GenerateBatchPrefillCapability;
  generateBatchPrefillCapabilityReason: string;
  queueWaitReason: PrefillQueueWaitReason;
  fallbackReason: string;
  runtimeDispatchSkippedReason: string;
  selectedBatchSize: number;
}

export interface PrefillBatchHealthPayload {
  enabled: boolean;
  queued?: number;
  eligible?: number;
  selected?: number;
  skipped?: number;
  cache_hits?: number;
  cache_misses?: number;
  queue_size?: number;
  generate_batch_prefill_capability?: GenerateBatchPrefillCapability;
  generate_batch_prefill_capability_reason?: string;
  queue_wait_reason?: PrefillQueueWaitReason;
  fallback_reason?: string;
  runtime_dispatch_skipped_reason?: string;
  selected_batch_size?: number;
}

export interface BatchHealthInputs {
  enabled: boolean;
  queued: number;
  selected: number;
  total: number;
  failed: number;
  cancelled: number;
  completed: number;
  completion_window_supported: boolean;
  supported_endpoints: readonly string[];
  execution_mode: BatchExecutionMode;
  last_fallback_reason: string;
  batch_capability: string;
  batch_capability_reason: string;
  selected_batch_execution_mode: BatchExecutionMode;
  fallback_reason: string;
  runtime_dispatch_skipped_reason: string;
  unsupported_mode_hits_total: number;
  validation_errors_total: number;
  streaming_rejections_total: number;
}

export interface BatchHealthPayload {
  enabled: boolean;
  queued?: number;
  selected?: number;
  total?: number;
  failed?: number;
  cancelled?: number;
  completed?: number;
  completion_window_supported?: boolean;
  supported_endpoints?: readonly string[];
  execution_mode?: BatchExecutionMode;
  last_fallback_reason?: string;
  batch_capability?: string;
  batch_capability_reason?: string;
  selected_batch_execution_mode?: BatchExecutionMode;
  fallback_reason?: string;
  runtime_dispatch_skipped_reason?: string;
  unsupported_mode_hits_total?: number;
  validation_errors_total?: number;
  streaming_rejections_total?: number;
}

export function buildPrefillBatchHealthPayload(
  input: PrefillBatchHealthInputs,
): PrefillBatchHealthPayload {
  if (!input.enabled) {
    return { enabled: false };
  }

  return {
    enabled: true,
    queued: input.queued,
    eligible: input.eligible,
    selected: input.selected,
    skipped: input.skipped,
    cache_hits: input.cacheHits,
    cache_misses: input.cacheMisses,
    queue_size: input.queueSize,
    generate_batch_prefill_capability: input.generateBatchPrefillCapability,
    generate_batch_prefill_capability_reason: input.generateBatchPrefillCapabilityReason,
    queue_wait_reason: input.queueWaitReason,
    fallback_reason: input.fallbackReason,
    runtime_dispatch_skipped_reason: input.runtimeDispatchSkippedReason,
    selected_batch_size: input.selectedBatchSize,
  };
}

export function buildBatchHealthPayload(input: BatchHealthInputs): BatchHealthPayload {
  if (!input.enabled) {
    return { enabled: false };
  }

  return {
    enabled: true,
    queued: input.queued,
    selected: input.selected,
    total: input.total,
    failed: input.failed,
    cancelled: input.cancelled,
    completed: input.completed,
    completion_window_supported: input.completion_window_supported,
    supported_endpoints: input.supported_endpoints,
    execution_mode: input.execution_mode,
    last_fallback_reason: input.last_fallback_reason,
    batch_capability: input.batch_capability,
    batch_capability_reason: input.batch_capability_reason,
    selected_batch_execution_mode: input.selected_batch_execution_mode,
    fallback_reason: input.fallback_reason,
    runtime_dispatch_skipped_reason: input.runtime_dispatch_skipped_reason,
    unsupported_mode_hits_total: input.unsupported_mode_hits_total,
    validation_errors_total: input.validation_errors_total,
    streaming_rejections_total: input.streaming_rejections_total,
  };
}
