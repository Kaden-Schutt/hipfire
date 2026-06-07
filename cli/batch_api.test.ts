import { describe, expect, test } from "bun:test";
import {
  validateBatchInputForBatch,
  validateBatchInputLines,
  buildBatchInputErrorArtifact,
  countUnsupportedModeErrors,
  buildBatchOutputArtifact,
  parseBatchErrorLinesToJsonl,
} from "./batch_api";

describe("batch input JSONL validation", () => {
  test("accepts valid chat completion lines", () => {
    const raw = [
      JSON.stringify({
        custom_id: "a",
        method: "POST",
        url: "/v1/chat/completions",
        body: { model: "qwen3.5:9b", messages: [] },
      }),
      JSON.stringify({
        custom_id: "b",
        method: "POST",
        url: "/v1/chat/completions",
        body: { model: "qwen3.5:9b", messages: [] },
      }),
    ].join("\n");
    const parsed = validateBatchInputForBatch(raw, "/v1/chat/completions");
    expect(parsed.ok).toBe(true);
    expect(parsed.entries).toHaveLength(2);
    expect(parsed.errors).toEqual([]);
    expect(parsed.endpoint).toBe("/v1/chat/completions");
  });

  test("rejects duplicate custom_id and streaming lines", () => {
    const raw = [
      JSON.stringify({
        custom_id: "a",
        method: "POST",
        url: "/v1/chat/completions",
        body: { model: "qwen3.5:9b", messages: [], stream: true },
      }),
      JSON.stringify({
        custom_id: "a",
        method: "POST",
        url: "/v1/chat/completions",
        body: { model: "qwen3.5:9b", messages: [] },
      }),
    ].join("\n");
    const parsed = validateBatchInputForBatch(raw, "/v1/chat/completions");
    expect(parsed.ok).toBe(false);
    expect(parsed.errors.map((e) => e.code).sort()).toEqual(["duplicate_custom_id", "streaming_unsupported"]);
  });

  test("normalizes responses input and validates role/message shape", () => {
    const raw = JSON.stringify({
      custom_id: "r3",
      method: "POST",
      url: "/v1/responses",
      body: {
        model: "qwen3.5:9b",
        input: [
          { role: "user", content: "hello" },
          { role: 123, content: "bad role" },
        ],
        max_output_tokens: 32,
      },
    });
    const parsed = validateBatchInputForBatch(raw, "/v1/responses");
    expect(parsed.ok).toBe(false);
    expect(parsed.errors.some((e) => e.code === "invalid_responses_input")).toBe(true);
  });

  test("rejects image content in responses message list", () => {
    const raw = JSON.stringify({
      custom_id: "r4",
      method: "POST",
      url: "/v1/responses",
      body: {
        model: "qwen3.5:9b",
        input: [
          { role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }] },
        ],
      },
    });
    const parsed = validateBatchInputForBatch(raw, "/v1/responses");
    expect(parsed.ok).toBe(false);
    expect(parsed.errors.some((e) => e.code === "unsupported_content")).toBe(true);
  });

  test("rejects model mismatch across entries", () => {
    const raw = [
      JSON.stringify({
        custom_id: "a",
        method: "POST",
        url: "/v1/chat/completions",
        body: { model: "qwen3.5:9b", messages: [] },
      }),
      JSON.stringify({
        custom_id: "b",
        method: "POST",
        url: "/v1/chat/completions",
        body: { model: "qwen3.5:27b", messages: [] },
      }),
    ].join("\n");
    const parsed = validateBatchInputForBatch(raw, "/v1/chat/completions");
    expect(parsed.ok).toBe(false);
    expect(parsed.errors.some((e) => e.code === "model_mismatch")).toBe(true);
  });

  test("endpoint mismatch produces errors", () => {
    const raw = [
      JSON.stringify({
        custom_id: "a",
        method: "POST",
        url: "/v1/responses",
        body: { model: "qwen3.5:9b", input: "hello" },
      }),
    ].join("\n");
    const parsed = validateBatchInputForBatch(raw, "/v1/chat/completions");
    expect(parsed.ok).toBe(false);
    expect(parsed.errors.some((e) => e.code === "endpoint_mismatch")).toBe(true);
  });

  test("normalizes responses endpoint body into chat-compatible shape", () => {
    const raw = [
      JSON.stringify({
        custom_id: "r1",
        method: "POST",
        url: "/v1/responses",
        body: {
          model: "qwen3.5:9b",
          input: [
            { role: "user", content: "hello" },
          ],
          max_output_tokens: 256,
          stream: false,
        },
      }),
    ].join("\n");
    const parsed = validateBatchInputForBatch(raw, "/v1/responses");
    expect(parsed.ok).toBe(true);
    expect(parsed.entries).toHaveLength(1);
    expect(parsed.entries[0]?.normalized_body?.messages).toEqual([{ role: "user", content: "hello" }]);
    expect(parsed.entries[0]?.normalized_body?.max_tokens).toBe(256);
    expect(parsed.entries[0]?.normalized_body?.stream).toBe(false);
  });

  test("preserves stable custom_id in responses normalization output", () => {
    const raw = JSON.stringify({
      custom_id: "responses-alpha",
      method: "POST",
      url: "/v1/responses",
      body: {
        model: "qwen3.5:9b",
        input: [{ role: "user", content: "hello" }],
      },
    });
    const parsed = validateBatchInputForBatch(raw, "/v1/responses");
    expect(parsed.ok).toBe(true);
    expect(parsed.entries).toHaveLength(1);
    expect(parsed.entries[0]?.custom_id).toBe("responses-alpha");
    expect(parsed.entries[0]?.normalized_body?.messages).toEqual([{ role: "user", content: "hello" }]);
  });

  test("rejects malformed responses input for responses batching", () => {
    const raw = [
      JSON.stringify({
        custom_id: "r2",
        method: "POST",
        url: "/v1/responses",
        body: { model: "qwen3.5:9b", input: 123 },
      }),
    ].join("\n");
    const parsed = validateBatchInputForBatch(raw, "/v1/responses");
    expect(parsed.ok).toBe(false);
    expect(parsed.errors.some((e) => e.code === "invalid_responses_input")).toBe(true);
  });

  test("requires chat/completions message shape", () => {
    const raw = JSON.stringify({
      custom_id: "chat-shape",
      method: "POST",
      url: "/v1/chat/completions",
      body: { model: "qwen3.5:9b", messages: [{ role: 0, content: "bad" }] },
    });
    const parsed = validateBatchInputForBatch(raw, "/v1/chat/completions");
    expect(parsed.ok).toBe(false);
    expect(parsed.errors.some((e) => e.code === "invalid_messages")).toBe(true);
  });

  test("rejects /v1/chat/completions tools mode for batch scaffold", () => {
    const raw = JSON.stringify({
      custom_id: "chat-tools",
      method: "POST",
      url: "/v1/chat/completions",
      body: { model: "qwen3.5:9b", messages: [{ role: "user", content: "hello" }], tools: [{}] },
    });
    const parsed = validateBatchInputForBatch(raw, "/v1/chat/completions");
    expect(parsed.ok).toBe(false);
    expect(parsed.errors.some((e) => e.code === "tools_unsupported")).toBe(true);
  });

  test("counts unsupported-mode-only batch errors", () => {
    const raw = [
      JSON.stringify({
        custom_id: "a",
        method: "POST",
        url: "/v1/chat/completions",
        body: { model: "qwen3.5:9b", messages: [] },
      }),
      JSON.stringify({
        custom_id: "b",
        method: "POST",
        url: "/v1/chat/completions",
        body: { model: "qwen3.5:9b", messages: [], stream: true },
      }),
      JSON.stringify({
        custom_id: "c",
        method: "POST",
        url: "/v1/chat/completions",
        body: { model: "qwen3.5:9b", messages: [{ role: "user", content: 1 }] },
      }),
    ].join("\n");
    const parsed = validateBatchInputForBatch(raw, "/v1/chat/completions");
    expect(parsed.ok).toBe(false);
    expect(parsed.errors.some((e) => e.code === "streaming_unsupported")).toBe(true);
    expect(parsed.errors.some((e) => e.code === "invalid_messages")).toBe(true);
    expect(countUnsupportedModeErrors(parsed.errors)).toBe(2);
  });

  test("builds error artifacts from validation", () => {
    const { entries, errors } = validateBatchInputLines([
      JSON.stringify({ custom_id: "a", method: "GET", url: "/v1/chat/completions", body: {} }),
      "not-json",
    ].join("\n"));
    const errs = [
      ...errors,
      { line: 1, custom_id: "a", code: "invalid_method", message: "method" },
    ];
    const artifact = buildBatchInputErrorArtifact(errs, entries);
    expect(Array.isArray(artifact)).toBe(true);
    expect(artifact[0]).toMatchObject({ custom_id: expect.any(String), error: expect.any(Object) });
  });

  test("returns total non-empty line count", () => {
    const { totalLineCount, errors } = validateBatchInputLines([
      "", 
      JSON.stringify({ custom_id: "a", method: "POST", url: "/v1/chat/completions", body: {} }),
      "not-json",
      JSON.stringify({ custom_id: "a", method: "GET", url: "/v1/chat/completions", body: {} }),
      " ",
    ].join("\n"));

    expect(totalLineCount).toBe(3);
    expect(errors.length).toBeGreaterThan(0);
  });

  test("tags rejected line artifacts for telemetry", () => {
    const { errors } = validateBatchInputLines([
      "not-json",
      JSON.stringify({ custom_id: "a", method: "GET", url: "/v1/chat/completions", body: {} }),
    ].join("\n"));
    const artifact = buildBatchInputErrorArtifact(errors, []);
    expect(artifact.every((line) => line.processing_status === "rejected")).toBe(true);
    expect(artifact[0].error?.type).toBe("invalid_request_error");
  });

  test("builds output artifact with stable custom_id preservation", () => {
    const artifact = buildBatchOutputArtifact([
      {
        custom_id: "stable-id-1",
        response: { id: "resp_1", status: "ok" },
      },
      {
        custom_id: "stable-id-2",
        response: { id: "resp_2", status: "ok" },
      },
    ]);
    expect(artifact).toBe(
      `${JSON.stringify({ custom_id: "stable-id-1", response: { id: "resp_1", status: "ok" } })}\n` +
      `${JSON.stringify({ custom_id: "stable-id-2", response: { id: "resp_2", status: "ok" } })}\n`,
    );
    const parsed = artifact.trim().split("\n").map((line) => JSON.parse(line));
    expect(parsed[0]).toMatchObject({ custom_id: "stable-id-1" });
    expect(parsed[1]).toMatchObject({ custom_id: "stable-id-2" });
  });

  test("serializes parse errors into jsonl with line-scoped custom ids", () => {
    const raw = [
      JSON.stringify({ custom_id: "a", method: "GET", url: "/v1/chat/completions", body: {} }),
      JSON.stringify({ custom_id: "b", method: "POST", url: "/v1/chat/completions", body: { model: "qwen3.5:9b", messages: [] } }),
    ].join("\n");
    const parsed = validateBatchInputLines(raw);
    const lines = parseBatchErrorLinesToJsonl(parsed.errors, parsed.entries);
    const rows = lines.trim().split("\n").map((row) => JSON.parse(row));
    expect(rows.some((row) => row.custom_id === "a")).toBe(true);
    expect(rows.every((row: any) => row.processing_status === "rejected")).toBe(true);
  });
});
