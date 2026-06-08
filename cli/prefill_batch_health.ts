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
  totalBatches?: number;
  fusedBatches?: number;
  fallbackBatches?: number;
  batchSizeHistogram?: Record<string, number>;
  cacheHits: number;
  cacheMisses: number;
  metadataCacheHits?: number;
  runtimeCacheHits?: number;
  queueSize: number;
  pendingRequests?: number;
  residentRuntimeSessions?: number;
  residentStateLimit?: number;
  spillableBatchMax?: number;
  spillableSessions?: number;
  stateCacheDisk?: boolean;
  stateCacheDiskMinPriority?: number;
  diskSpillAllowed?: boolean;
  generateBatchPrefillCapability: GenerateBatchPrefillCapability;
  generateBatchPrefillCapabilityReason: string;
  queueWaitReason: PrefillQueueWaitReason;
  fallbackReason: string;
  runtimeDispatchSkippedReason: string;
  selectedBatchSize: number;
  lastPrefillTokens?: number;
  lastPrefillMs?: number;
  lastPrefillTokS?: number;
  daemonPrefillPlan?: string;
  daemonPrefillBackend?: string;
}

export interface PrefillBatchHealthPayload {
  enabled: boolean;
  queued?: number;
  eligible?: number;
  selected?: number;
  skipped?: number;
  total_batches?: number;
  fused_batches?: number;
  fallback_batches?: number;
  batch_size_histogram?: Record<string, number>;
  cache_hits?: number;
  cache_misses?: number;
  metadata_cache_hits?: number;
  runtime_cache_hits?: number;
  queue_size?: number;
  pending_requests?: number;
  resident_runtime_sessions?: number;
  resident_state_limit?: number;
  spillable_batch_max?: number;
  spillable_sessions?: number;
  state_cache_disk?: boolean;
  state_cache_disk_min_priority?: number;
  disk_spill_allowed?: boolean;
  generate_batch_prefill_capability?: GenerateBatchPrefillCapability;
  generate_batch_prefill_capability_reason?: string;
  queue_wait_reason?: PrefillQueueWaitReason;
  fallback_reason?: string;
  runtime_dispatch_skipped_reason?: string;
  selected_batch_size?: number;
  last_prefill_tokens?: number;
  last_prefill_ms?: number;
  last_prefill_tok_s?: number;
  daemon_prefill_plan?: string;
  daemon_prefill_backend?: string;
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
    total_batches: input.totalBatches ?? 0,
    fused_batches: input.fusedBatches ?? 0,
    fallback_batches: input.fallbackBatches ?? 0,
    batch_size_histogram: input.batchSizeHistogram ?? {},
    cache_hits: input.cacheHits,
    cache_misses: input.cacheMisses,
    metadata_cache_hits: input.metadataCacheHits ?? 0,
    runtime_cache_hits: input.runtimeCacheHits ?? input.cacheHits,
    queue_size: input.queueSize,
    pending_requests: input.pendingRequests ?? 0,
    resident_runtime_sessions: input.residentRuntimeSessions ?? 0,
    resident_state_limit: input.residentStateLimit ?? 0,
    spillable_batch_max: input.spillableBatchMax ?? input.residentStateLimit ?? 0,
    spillable_sessions: input.spillableSessions ?? 0,
    state_cache_disk: input.stateCacheDisk ?? false,
    state_cache_disk_min_priority: input.stateCacheDiskMinPriority ?? 128,
    disk_spill_allowed: input.diskSpillAllowed ?? false,
    generate_batch_prefill_capability: input.generateBatchPrefillCapability,
    generate_batch_prefill_capability_reason: input.generateBatchPrefillCapabilityReason,
    queue_wait_reason: input.queueWaitReason,
    fallback_reason: input.fallbackReason,
    runtime_dispatch_skipped_reason: input.runtimeDispatchSkippedReason,
    selected_batch_size: input.selectedBatchSize,
    last_prefill_tokens: input.lastPrefillTokens ?? 0,
    last_prefill_ms: input.lastPrefillMs ?? 0,
    last_prefill_tok_s: input.lastPrefillTokS ?? 0,
    daemon_prefill_plan: input.daemonPrefillPlan,
    daemon_prefill_backend: input.daemonPrefillBackend,
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
