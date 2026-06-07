import { describe, expect, test } from "bun:test";
import {
  buildGenerateBatchPrefillProbeMessage,
  interpretGenerateBatchPrefillProbeResponse,
  prefillBatchRuntimeDispatchStatus,
} from "./generate_batch_prefill_protocol";

describe("generate_batch_prefill protocol probe", () => {
  test("builds a valid minimal probe envelope", () => {
    expect(buildGenerateBatchPrefillProbeMessage()).toEqual({
      type: "generate_batch_prefill",
      id: "prefill-batch-probe",
      batch_id: "prefill-batch-probe",
      worker_key_id: "probe-worker",
      sessions: [{
        id: "probe-session",
        suffix_tokens: [1],
        state_handle: {
          state_kinds: ["attention_kv"],
          logical_position: 0,
          cached_prefix_tokens: 0,
        },
        params: {
          max_tokens: 1,
          temperature: 0,
        },
      }],
    });
  });

  test("detects old daemon scaffold as known but unsupported", () => {
    expect(interpretGenerateBatchPrefillProbeResponse({
      type: "generate_batch_prefill_unsupported",
      id: "prefill-batch-probe",
      batch_id: "prefill-batch-probe",
      supported: false,
      reason: "per_session_runtime_state_unavailable",
    })).toEqual({
      capability: "unsupported",
      reason: "per_session_runtime_state_unavailable",
    });
  });

  test("detects older daemons as unknown protocol", () => {
    expect(interpretGenerateBatchPrefillProbeResponse({
      type: "error",
      message: "unknown type: generate_batch_prefill",
    })).toEqual({
      capability: "unknown",
      reason: "daemon_unknown_message",
    });
  });

  test("recognizes daemon serial-prefill support", () => {
    expect(interpretGenerateBatchPrefillProbeResponse({
      type: "generate_batch_prefill_ready",
      reason: "qwen35_serial_exact_token_prefill_available",
    })).toEqual({
      capability: "supported",
      reason: "qwen35_serial_exact_token_prefill_available",
    });
  });
});

describe("generate_batch_prefill serialized fallback metadata", () => {
  test("keeps ineligible requests outside runtime dispatch", () => {
    expect(prefillBatchRuntimeDispatchStatus(false, "unsupported")).toEqual({
      runtimeDispatch: "not_selected",
      runtimeDispatchReason: "not_eligible",
    });
  });

  test("marks old daemons as missing protocol", () => {
    expect(prefillBatchRuntimeDispatchStatus(true, "unknown")).toEqual({
      runtimeDispatch: "skipped_missing_generate_batch_prefill",
      runtimeDispatchReason: "missing_generate_batch_prefill",
    });
  });

  test("marks scaffolded daemons as known unsupported", () => {
    expect(prefillBatchRuntimeDispatchStatus(true, "unsupported")).toEqual({
      runtimeDispatch: "skipped_generate_batch_prefill_unsupported",
      runtimeDispatchReason: "daemon_generate_batch_prefill_unsupported",
    });
  });

  test("reports daemon serial-prefill support for server dispatch", () => {
    expect(prefillBatchRuntimeDispatchStatus(true, "supported")).toEqual({
      runtimeDispatch: "daemon_serial_prefill_available",
      runtimeDispatchReason: "server_dispatch_enabled",
    });
  });
});
