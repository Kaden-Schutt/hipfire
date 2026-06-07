export type GenerateBatchPrefillCapability = "unknown" | "unsupported" | "supported";

export interface GenerateBatchPrefillProbeResult {
  capability: GenerateBatchPrefillCapability;
  reason: string;
}

export function buildGenerateBatchPrefillProbeMessage() {
  return {
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
  };
}

export function interpretGenerateBatchPrefillProbeResponse(
  msg: any,
): GenerateBatchPrefillProbeResult {
  if (msg?.type === "generate_batch_prefill_unsupported") {
    return {
      capability: "unsupported",
      reason: typeof msg.reason === "string" ? msg.reason : "unsupported",
    };
  }
  if (msg?.type === "generate_batch_prefill_ready" || msg?.type === "generate_batch_prefill_supported") {
    return {
      capability: "supported",
      reason: typeof msg.reason === "string" ? msg.reason : "supported",
    };
  }
  if (msg?.type === "error" && typeof msg.message === "string" && msg.message.includes("unknown type")) {
    return {
      capability: "unknown",
      reason: "daemon_unknown_message",
    };
  }
  if (msg?.type === "error") {
    return {
      capability: "unknown",
      reason: typeof msg.message === "string" ? msg.message : "daemon_error",
    };
  }
  return {
    capability: "unknown",
    reason: `unexpected_response:${msg?.type ?? "missing"}`,
  };
}

export function prefillBatchRuntimeDispatchStatus(
  eligible: boolean,
  capability: GenerateBatchPrefillCapability,
): { runtimeDispatch: string; runtimeDispatchReason: string } {
  if (!eligible) {
    return {
      runtimeDispatch: "not_selected",
      runtimeDispatchReason: "not_eligible",
    };
  }
  if (capability === "unsupported") {
    return {
      runtimeDispatch: "skipped_generate_batch_prefill_unsupported",
      runtimeDispatchReason: "daemon_generate_batch_prefill_unsupported",
    };
  }
  if (capability === "supported") {
    return {
      runtimeDispatch: "daemon_serial_prefill_available",
      runtimeDispatchReason: "server_dispatch_enabled",
    };
  }
  return {
    runtimeDispatch: "skipped_missing_generate_batch_prefill",
    runtimeDispatchReason: "missing_generate_batch_prefill",
  };
}
