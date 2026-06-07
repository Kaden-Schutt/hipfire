export type BatchEndpoint = "/v1/chat/completions" | "/v1/responses";

export interface BatchInputRecord {
  custom_id: string;
  method: "POST";
  url: string;
  body: Record<string, any>;
  normalized_body?: Record<string, any>;
  headers?: Record<string, string>;
}

export interface BatchInputValidationError {
  line: number;
  custom_id?: string;
  code: string;
  message: string;
}

export type UnsupportedModeCode =
  | "streaming_unsupported"
  | "invalid_messages"
  | "responses_input_unsupported"
  | "tools_unsupported"
  | "unsupported_content"
  | "endpoint_mismatch"
  | "model_missing";

function isUnsupportedModeError(error: BatchInputValidationError): boolean {
  return (
    error.code === "streaming_unsupported"
    || error.code === "tools_unsupported"
    || error.code === "invalid_messages"
    || error.code === "responses_input_unsupported"
    || error.code === "unsupported_content"
    || error.code === "model_missing"
    || error.code === "endpoint_mismatch"
  );
}

export interface ParsedBatchInput {
  entries: BatchInputRecord[];
  errors: BatchInputValidationError[];
  modelHint?: string;
  endpoint: string;
  totalLineCount: number;
}

export interface BatchResponsesNormalizedBody {
  model?: string;
  temperature: number;
  top_p: number;
  max_tokens: number;
  stream: false;
  messages: unknown[];
  input: unknown;
  prompt?: string;
}

export interface BatchResponsesNormalizeResult {
  ok: boolean;
  body: BatchResponsesNormalizedBody | null;
  errors: BatchInputValidationError[];
}

export interface BatchFileRecord {
  id: string;
  object: "file";
  filename: string;
  bytes: number;
  purpose: "batch";
  created_at: number;
}

export type BatchEndpointParseResult =
  | { ok: true; entries: BatchInputRecord[]; endpoint: string; errors: []; totalLineCount: number }
  | { ok: false; entries: BatchInputRecord[]; endpoint: string; errors: BatchInputValidationError[]; totalLineCount: number };

export interface BatchLineArtifact {
  custom_id: string;
  error?: {
    code: string;
    message: string;
    type: string;
  };
  response?: any;
  processing_status?: "succeeded" | "rejected";
}

export type BatchStatus =
  | "validating"
  | "failed"
  | "in_progress"
  | "finalizing"
  | "completed"
  | "expired"
  | "cancelling"
  | "cancelled";

export interface BatchRecord {
  id: string;
  object: "batch";
  status: BatchStatus;
  endpoint: string;
  completion_window: string;
  input_file_id: string;
  output_file_id: string | null;
  error_file_id: string | null;
  request_count: number;
  created_at: number;
  completed_requests?: number;
  in_progress_at?: number;
  completed_at?: number;
  failed_reason?: string;
}

const SUPPORTED_ENDPOINTS: ReadonlySet<string> = new Set(["/v1/chat/completions", "/v1/responses"]);

function isResponsesInput(value: unknown): value is string | Record<string, any> | any[] {
  if (typeof value === "string") return true;
  if (Array.isArray(value)) return true;
  return isJSONObject(value);
}

export function isSupportedBatchEndpoint(endpoint: string): boolean {
  return SUPPORTED_ENDPOINTS.has(endpoint);
}

function isJSONObject(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isUnsupportedContent(value: unknown): boolean {
  if (typeof value === "string") return false;
  if (Array.isArray(value)) {
    if (value.length === 0) return false;
    return value.some((part) => {
      if (typeof part === "string") return false;
      if (!isJSONObject(part)) return true;
      if (part.type === "text" && typeof part.text === "string") return false;
      return true;
    });
  }
  return true;
}

function validateChatCompletionsMessages(messages: unknown, customId: string): BatchInputValidationError[] {
  const errors: BatchInputValidationError[] = [];
  if (!Array.isArray(messages)) {
    errors.push({
      line: 0,
      custom_id: customId,
      code: "invalid_messages",
      message: `custom_id ${customId} requires messages to be an array`,
    });
    return errors;
  }
  const hasInvalidMessage = messages.some((m) => {
    if (!isJSONObject(m)) return true;
    const role = m.role;
    const content = m.content;
    return (
      typeof role !== "string"
      || role.length === 0
      || (typeof content !== "string" && !Array.isArray(content))
      || isUnsupportedContent(content)
    );
  });
  if (hasInvalidMessage) {
    errors.push({
      line: 0,
      custom_id: customId,
      code: "invalid_messages",
      message: `custom_id ${customId} contains invalid chat message entries`,
    });
  }
  return errors;
}

export function normalizeResponsesBatchInputBody(
  body: Record<string, any>,
  customId?: string,
): BatchResponsesNormalizeResult {
  const errors: BatchInputValidationError[] = [];
  const input = body.input;
  if (!isResponsesInput(input)) {
    errors.push({
      line: 0,
      custom_id: customId,
      code: "invalid_responses_input",
      message: `custom_id ${customId ?? "<unknown>"} missing valid responses input`,
    });
    return { ok: false, body: null, errors };
  }

  const inputMessages = input && typeof input === "object" && !Array.isArray(input) && "messages" in input
    ? input.messages
    : undefined;
  const messages: unknown[] = Array.isArray(input)
    ? input
    : Array.isArray(inputMessages)
      ? inputMessages
      : [];

  const unsupportedMessages = messages.filter((entry) => {
    if (!isJSONObject(entry) || typeof entry.role !== "string" || !(typeof entry.content === "string" || Array.isArray(entry.content))) {
      return true;
    }
    return isUnsupportedContent(entry.content);
  });

  const normalizedMessages = messages
    .filter((entry) => isJSONObject(entry)
      && typeof entry.role === "string"
      && (typeof entry.content === "string" || Array.isArray(entry.content))
      && !isUnsupportedContent(entry.content));

  if (Array.isArray(input) && input.length !== normalizedMessages.length) {
    if (unsupportedMessages.length > 0) {
      errors.push({
        line: 0,
        custom_id: customId,
        code: "unsupported_content",
        message: `custom_id ${customId ?? "<unknown>"} includes unsupported content in input`,
      });
    }
    errors.push({
      line: 0,
      custom_id: customId,
      code: "invalid_responses_input",
      message: `custom_id ${customId ?? "<unknown>"} includes non-message entries in input`,
    });
  }

  if (Array.isArray(inputMessages) && inputMessages.length !== normalizedMessages.length) {
    errors.push({
      line: 0,
      custom_id: customId,
      code: unsupportedMessages.length > 0 ? "unsupported_content" : "invalid_responses_input",
      message: `custom_id ${customId ?? "<unknown>"} includes non-message entries in input.messages`,
    });
  }

  return {
    ok: errors.length === 0,
    body: {
      model: body.model ?? undefined,
      temperature: body.temperature ?? 0.0,
      top_p: body.top_p ?? 1.0,
      max_tokens: body.max_output_tokens ?? body.max_tokens ?? 16,
      stream: false,
      messages: normalizedMessages as Array<any>,
      input,
      prompt: typeof input === "string" ? input : undefined,
    },
    errors,
  };
}

function validateResponsesForBatchBody(
  body: Record<string, any>,
  customId: string,
): BatchInputValidationError[] {
  const errors: BatchInputValidationError[] = [];
  if (typeof body.model !== "string" || body.model.trim() === "") {
    errors.push({
      line: 0,
      custom_id: customId,
      code: "model_missing",
      message: `custom_id ${customId} missing model for /v1/responses`,
    });
  }
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    errors.push({
      line: 0,
      custom_id: customId,
      code: "tools_unsupported",
      message: `custom_id ${customId} includes tools, which are unsupported in batch mode`,
    });
  }
  return errors;
}

function validateChatCompletionBody(
  body: Record<string, any>,
  customId: string,
): BatchInputValidationError[] {
  const errors: BatchInputValidationError[] = [];
  if (typeof body.model !== "string" || body.model.trim() === "") {
    errors.push({
      line: 0,
      custom_id: customId,
      code: "model_missing",
      message: `custom_id ${customId} missing model for ${body?.metadata?.endpoint ?? "/v1/chat/completions"}`,
    });
  }
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    errors.push({
      line: 0,
      custom_id: customId,
      code: "tools_unsupported",
      message: `custom_id ${customId} includes tools, which are unsupported in batch mode`,
    });
  }
  errors.push(...validateChatCompletionsMessages(body.messages, customId));
  return errors;
}

function toLineIndex(lineNo: number): number {
  return Math.max(1, lineNo);
}

export function validateBatchInputLines(raw: string): {
  entries: BatchInputRecord[];
  errors: BatchInputValidationError[];
  totalLineCount: number;
} {
  if (raw.trim().length === 0) {
    return {
      entries: [],
      errors: [{ line: 1, code: "empty_batch", message: "batch file is empty" }],
      totalLineCount: 0,
    };
  }

  const entries: BatchInputRecord[] = [];
  const errors: BatchInputValidationError[] = [];
  let totalLineCount = 0;
  const seenIds = new Set<string>();
  const lines = raw.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const ln = toLineIndex(i + 1);
    const rawLine = lines[i].trim();
    if (!rawLine) continue;
    totalLineCount += 1;
    let lineValid = true;

    let parsed: unknown;
    try {
      parsed = JSON.parse(rawLine);
    } catch (err: any) {
      errors.push({
        line: ln,
        code: "invalid_json",
        message: `line ${ln} is not valid JSON`,
      });
      continue;
    }

    if (!isJSONObject(parsed)) {
      errors.push({
        line: ln,
        code: "invalid_object",
        message: `line ${ln} must be a JSON object`,
      });
      continue;
    }

    const customId = parsed.custom_id;
    if (typeof customId !== "string" || customId.trim() === "") {
      errors.push({
        line: ln,
        code: "invalid_custom_id",
        message: `line ${ln} missing or non-string custom_id`,
      });
      continue;
    }

    if (seenIds.has(customId)) {
      errors.push({
        line: ln,
        custom_id: customId,
        code: "duplicate_custom_id",
        message: `duplicate custom_id: ${customId}`,
      });
      lineValid = false;
    } else {
      seenIds.add(customId);
    }

    const method = parsed.method;
    if (method !== "POST") {
      errors.push({
        line: ln,
        custom_id: customId,
        code: "invalid_method",
        message: `line ${ln} method must be POST`,
      });
      lineValid = false;
    }

    const url = parsed.url;
    if (typeof url !== "string" || !url.startsWith("/") || !isSupportedBatchEndpoint(url)) {
      errors.push({
        line: ln,
        custom_id: customId,
        code: "invalid_url",
        message: `line ${ln} url must be one of /v1/chat/completions or /v1/responses`,
      });
      lineValid = false;
    }

    const body = parsed.body;
    if (!isJSONObject(body)) {
      errors.push({
        line: ln,
        custom_id: customId,
        code: "invalid_body",
        message: `line ${ln} body must be a JSON object`,
      });
      lineValid = false;
    }

    if (body.stream === true) {
      errors.push({
        line: ln,
        custom_id: customId,
        code: "streaming_unsupported",
        message: `line ${ln} sets stream=true, which is unsupported in batch mode`,
      });
      lineValid = false;
    }

    if (!lineValid) {
      continue;
    }

    entries.push({
      custom_id: customId,
      method,
      url,
      body,
      headers: parsed.headers && isJSONObject(parsed.headers)
        ? Object.fromEntries(
            Object.entries(parsed.headers).filter(([, value]) => typeof value === "string") as [string, string][],
          )
        : undefined,
    });
  }

  return { entries, errors, totalLineCount };
}

export function validateBatchInputForBatch(
  raw: string,
  expectedEndpoint: string,
): BatchEndpointParseResult {
  const parsed = validateBatchInputLines(raw);

  const endpoint = expectedEndpoint;
  const entries: BatchInputRecord[] = [];
  const filteredErrors = [...parsed.errors];

  if (!isSupportedBatchEndpoint(expectedEndpoint)) {
    return {
      ok: false,
      entries,
      endpoint,
      errors: [
        {
          line: 1,
          code: "invalid_endpoint",
          message: `Unsupported batch endpoint ${expectedEndpoint}`,
        },
        ...filteredErrors,
      ],
      totalLineCount: parsed.totalLineCount,
    };
  }

  let expectedModel: string | undefined;
  for (const entry of parsed.entries) {
    if (entry.url === "/v1/responses") {
      const normalized = normalizeResponsesBatchInputBody(entry.body, entry.custom_id);
      filteredErrors.push(...validateResponsesForBatchBody(entry.body, entry.custom_id));
      if (!normalized.ok || !normalized.body) {
        filteredErrors.push(...normalized.errors);
        continue;
      }
      entry.normalized_body = normalized.body;
    }

    if (entry.url === "/v1/chat/completions") {
      filteredErrors.push(...validateChatCompletionBody(entry.body, entry.custom_id));
      if (filteredErrors.some((err) => err.custom_id === entry.custom_id)) {
        continue;
      }
    }

    if (entry.url !== expectedEndpoint) {
      filteredErrors.push({
        line: 0,
        custom_id: entry.custom_id,
        code: "endpoint_mismatch",
        message: `custom_id ${entry.custom_id} uses ${entry.url} but batch endpoint is ${expectedEndpoint}`,
      });
      continue;
    }

    const model = typeof entry.body.model === "string" ? entry.body.model : undefined;
    if (expectedModel === undefined) {
      expectedModel = model;
    } else if (model !== undefined && model !== expectedModel) {
      filteredErrors.push({
        line: 0,
        custom_id: entry.custom_id,
        code: "model_mismatch",
        message: `custom_id ${entry.custom_id} model family differs from earlier entries`,
      });
      continue;
    }

    if (entry.normalized_body) {
      entries.push(entry);
      continue;
    }

    entries.push({
      custom_id: entry.custom_id,
      method: entry.method,
      url: entry.url,
      body: {
        ...entry.body,
      },
      headers: entry.headers,
    });
  }

  if (filteredErrors.length > 0) {
    return {
      ok: false,
      entries,
      endpoint,
      errors: filteredErrors,
      totalLineCount: parsed.totalLineCount,
    };
  }

  return {
    ok: true,
    entries,
    endpoint,
    errors: [],
    totalLineCount: parsed.totalLineCount,
  };
}

export function buildBatchInputErrorArtifact(errors: BatchInputValidationError[], entries: BatchInputRecord[]): BatchLineArtifact[] {
  const byCustomId = new Map<string, string>();
  for (const err of errors) {
    if (err.custom_id) {
      byCustomId.set(err.custom_id, `${err.code}: ${err.message}`);
    }
  }

  const artifacts: BatchLineArtifact[] = [...entries].map((entry) => ({
    custom_id: entry.custom_id,
      processing_status: "rejected",
    error: {
      code: "batch_entry_error",
      message: byCustomId.get(entry.custom_id) ?? "validation failed for this line",
      type: "invalid_request_error",
    },
  }));

  for (const err of errors) {
    if (err.custom_id && !entries.some((entry) => entry.custom_id === err.custom_id)) {
      artifacts.push({
        custom_id: err.custom_id,
        processing_status: "rejected",
        error: {
          code: err.code,
          message: `${err.message}`,
          type: "invalid_request_error",
        },
      });
    }
    if (!err.custom_id) {
      artifacts.push({
        custom_id: `line-${err.line}`,
        processing_status: "rejected",
        error: {
          code: err.code,
          message: `${err.message}`,
          type: "invalid_request_error",
        },
      });
    }
  }

  return artifacts;
}

export function countUnsupportedModeErrors(errors: BatchInputValidationError[]): number {
  return errors.filter((error) => isUnsupportedModeError(error)).length;
}

export function parseBatchErrorLinesToJsonl(errors: BatchInputValidationError[], entries: BatchInputRecord[]): string {
  return buildBatchInputErrorArtifact(errors, entries)
    .map((line) => JSON.stringify(line))
    .join("\n");
}

export function buildBatchOutputArtifact(lines: Array<{ custom_id: string; response: any }>): string {
  return lines.map((line) => JSON.stringify(line)).join("\n") + (lines.length > 0 ? "\n" : "");
}
