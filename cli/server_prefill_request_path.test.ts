import { describe, expect, test } from "bun:test";
import { shouldQueueServerPrefillPending } from "./server_prefill_request_path";

describe("server request-path prefill batching guard", () => {
  const base = {
    eligible: true,
    hasScheduler: true,
    queuePreviewReason: "ready",
  };

  test("allows compatible non-streaming chat requests to wait for coalescing", () => {
    expect(shouldQueueServerPrefillPending({
      ...base,
      stream: false,
    })).toBe(true);
  });

  test("never queues streaming chat requests for generate_batch_prefill", () => {
    expect(shouldQueueServerPrefillPending({
      ...base,
      stream: true,
    })).toBe(false);
  });

  test("allows compatible non-streaming responses requests to wait for coalescing", () => {
    expect(shouldQueueServerPrefillPending({
      ...base,
      stream: false,
    })).toBe(true);
  });

  test("does not create pending waiters when the scheduler already selected this request", () => {
    expect(shouldQueueServerPrefillPending({
      ...base,
      stream: false,
      queuePreviewReason: "selected",
    })).toBe(false);
  });

  test("does not create pending waiters for explicitly ineligible preview results", () => {
    expect(shouldQueueServerPrefillPending({
      ...base,
      stream: false,
      queuePreviewReason: "not_eligible",
    })).toBe(false);
  });

  test("does not create pending waiters without an eligible scheduler path", () => {
    expect(shouldQueueServerPrefillPending({
      ...base,
      eligible: false,
      hasScheduler: true,
      stream: false,
    })).toBe(false);
    expect(shouldQueueServerPrefillPending({
      ...base,
      eligible: true,
      hasScheduler: false,
      stream: false,
    })).toBe(false);
  });
});
