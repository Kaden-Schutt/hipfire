import { describe, expect, test } from "bun:test";
import { buildBatchHealthPayload, buildPrefillBatchHealthPayload, type BatchExecutionMode } from "./prefill_batch_health";

describe("prefill batch health payload", () => {
  test("returns disabled payload when batching disabled", () => {
    expect(buildPrefillBatchHealthPayload({
      enabled: false,
      queued: 0,
      eligible: 0,
      selected: 0,
      skipped: 0,
      cacheHits: 0,
      cacheMisses: 0,
      queueSize: 0,
      generateBatchPrefillCapability: "unknown",
      generateBatchPrefillCapabilityReason: "not_probed",
      queueWaitReason: "disabled",
      fallbackReason: "batching_disabled",
      runtimeDispatchSkippedReason: "not_enabled",
      selectedBatchSize: 0,
    })).toEqual({
      enabled: false,
    });
  });

  test("includes serialized batching metadata when enabled", () => {
    expect(buildPrefillBatchHealthPayload({
      enabled: true,
      queued: 2,
      eligible: 1,
      selected: 0,
      skipped: 1,
      cacheHits: 3,
      cacheMisses: 4,
      queueSize: 5,
      generateBatchPrefillCapability: "unsupported",
      generateBatchPrefillCapabilityReason: "per_session_runtime_state_unavailable",
      queueWaitReason: "waiting",
      fallbackReason: "queue_wait:waiting",
      runtimeDispatchSkippedReason: "daemon_generate_batch_prefill_unsupported",
      selectedBatchSize: 4,
    })).toEqual({
      enabled: true,
      queued: 2,
      eligible: 1,
      selected: 0,
      skipped: 1,
      cache_hits: 3,
      cache_misses: 4,
      queue_size: 5,
      generate_batch_prefill_capability: "unsupported",
      generate_batch_prefill_capability_reason: "per_session_runtime_state_unavailable",
      queue_wait_reason: "waiting",
      fallback_reason: "queue_wait:waiting",
      runtime_dispatch_skipped_reason: "daemon_generate_batch_prefill_unsupported",
      selected_batch_size: 4,
    });
  });
});

describe("batch control-plane health payload", () => {
  test("includes machine-readable batch compatibility fields", () => {
    const mode: BatchExecutionMode = "serial_fallback";
    expect(buildBatchHealthPayload({
      enabled: true,
      queued: 1,
      selected: 0,
      total: 3,
      failed: 1,
      cancelled: 0,
      completed: 1,
      completion_window_supported: true,
      supported_endpoints: ["/v1/chat/completions", "/v1/responses"],
      execution_mode: mode,
      last_fallback_reason: "line_fallback:tools",
      batch_capability: "unsupported",
      batch_capability_reason: "per_session_runtime_state_unavailable",
      selected_batch_execution_mode: mode,
      fallback_reason: "line_fallback:tools",
      runtime_dispatch_skipped_reason: "daemon_generate_batch_prefill_unsupported",
      unsupported_mode_hits_total: 2,
      validation_errors_total: 1,
      streaming_rejections_total: 0,
    })).toEqual({
      enabled: true,
      queued: 1,
      selected: 0,
      total: 3,
      failed: 1,
      cancelled: 0,
      completed: 1,
      completion_window_supported: true,
      supported_endpoints: ["/v1/chat/completions", "/v1/responses"],
      execution_mode: mode,
      last_fallback_reason: "line_fallback:tools",
      batch_capability: "unsupported",
      batch_capability_reason: "per_session_runtime_state_unavailable",
      selected_batch_execution_mode: mode,
      fallback_reason: "line_fallback:tools",
      runtime_dispatch_skipped_reason: "daemon_generate_batch_prefill_unsupported",
      unsupported_mode_hits_total: 2,
      validation_errors_total: 1,
      streaming_rejections_total: 0,
    });
  });
});
