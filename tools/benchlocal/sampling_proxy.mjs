#!/usr/bin/env node
/**
 * Dev-only campaign tooling (not the Rust control plane).
 *
 * Unified OpenAI JSON sampling proxy for BenchLocal nine-pack quality campaigns.
 * Forces the model-recommended sampled request policy on intercepted
 * POST .../v1/chat/completions bodies, attests each rewrite, and
 * otherwise passes traffic through unchanged.
 *
 * Unifies the two 2026-07-31 archived proxies behind THINKING_MODE:
 *   disabled (default) → baseline sha256
 *     578f2be2b530ad4fb7ed76459ff76ba81c538376ee296af26369cf2af72099a2
 *     forces enable_thinking: false; injects no reasoning fields
 *   medium → medium sha256
 *     8a221d7c3ad11abc66302876d5c0d0ebf441d36c4d71d1b055009e4bc1af4c3d
 *     forces enable_thinking: true, reasoning_effort: "medium";
 *     max_think_tokens: 1024 is attestation-only (expected_max_think_tokens),
 *     never written on the wire body
 *
 * Pre-thinkfix baseline variant (reference only, not reproduced):
 *   b87fb8658860d6fe971ae2c9a5b750e304278411c1edc100bc510906ab1b918f
 *
 * Env:
 *   LISTEN_HOST       default 127.0.0.1
 *   LISTEN_PORT       required
 *   TARGET_ORIGIN     required, e.g. http://127.0.0.1:11435
 *   MODEL_SLUG        required, attestation label
 *   ATTESTATION_LOG   required, JSONL path
 *   PRESENCE_PENALTY  default 1.5
 *   THINKING_MODE     disabled (default) | medium
 */

import http from "node:http";
import https from "node:https";
import fs from "node:fs";
import path from "node:path";
import { URL } from "node:url";

const FORCED_SAMPLING = Object.freeze({
  temperature: 1.0,
  top_p: 0.95,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: Number(process.env.PRESENCE_PENALTY ?? "1.5"),
  repetition_penalty: 1.0,
});

const THINKING_MODE = (process.env.THINKING_MODE || "disabled").toLowerCase();
if (THINKING_MODE !== "disabled" && THINKING_MODE !== "medium") {
  console.error(
    `sampling-proxy: THINKING_MODE must be "disabled" or "medium", got ${JSON.stringify(process.env.THINKING_MODE)}`,
  );
  process.exit(1);
}

const FORCED_REASONING_MEDIUM = Object.freeze({
  reasoning_effort: "medium",
  enable_thinking: true,
  max_think_tokens: 1024,
});

function requireEnv(name) {
  const v = process.env[name];
  if (v === undefined || v === "") {
    console.error(`sampling-proxy: missing required env ${name}`);
    process.exit(1);
  }
  return v;
}

const LISTEN_HOST = process.env.LISTEN_HOST || "127.0.0.1";
const LISTEN_PORT = Number(requireEnv("LISTEN_PORT"));
const TARGET_ORIGIN = requireEnv("TARGET_ORIGIN").replace(/\/+$/, "");
const MODEL_SLUG = requireEnv("MODEL_SLUG");
const ATTESTATION_LOG = requireEnv("ATTESTATION_LOG");

if (!Number.isFinite(LISTEN_PORT) || LISTEN_PORT <= 0) {
  console.error("sampling-proxy: LISTEN_PORT must be a positive number");
  process.exit(1);
}

try {
  // Validate origin early.
  // eslint-disable-next-line no-new
  new URL(TARGET_ORIGIN);
} catch {
  console.error(`sampling-proxy: invalid TARGET_ORIGIN: ${TARGET_ORIGIN}`);
  process.exit(1);
}

fs.mkdirSync(path.dirname(path.resolve(ATTESTATION_LOG)), { recursive: true });

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailers",
  "transfer-encoding",
  "upgrade",
  "host",
  "content-length",
]);

function isChatCompletions(method, urlPath) {
  return method === "POST" && /\/v1\/chat\/completions\/?$/.test(urlPath);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function filterRequestHeaders(headers) {
  const out = {};
  for (const [k, v] of Object.entries(headers)) {
    if (v === undefined) continue;
    if (HOP_BY_HOP.has(k.toLowerCase())) continue;
    out[k] = v;
  }
  return out;
}

function filterResponseHeaders(headers) {
  const out = {};
  for (const [k, v] of Object.entries(headers)) {
    if (v === undefined) continue;
    const lk = k.toLowerCase();
    if (lk === "transfer-encoding" || lk === "connection" || lk === "keep-alive") {
      continue;
    }
    out[k] = v;
  }
  return out;
}

function appendAttestation(record) {
  fs.appendFileSync(ATTESTATION_LOG, `${JSON.stringify(record)}\n`, "utf8");
}

function sendJson(res, status, obj) {
  const body = Buffer.from(JSON.stringify(obj), "utf8");
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": body.length,
  });
  res.end(body);
}

function forward(req, res, bodyBuf) {
  const target = new URL(TARGET_ORIGIN);
  const incoming = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  const pathAndQuery = incoming.pathname + incoming.search;
  const isTls = target.protocol === "https:";
  const lib = isTls ? https : http;

  const headers = filterRequestHeaders(req.headers);
  if (bodyBuf !== undefined) {
    headers["content-length"] = String(bodyBuf.length);
  }

  const opts = {
    protocol: target.protocol,
    hostname: target.hostname,
    port: target.port || (isTls ? 443 : 80),
    method: req.method,
    path: pathAndQuery,
    headers,
  };

  const upstream = lib.request(opts, (upRes) => {
    const outHeaders = filterResponseHeaders(upRes.headers);
    res.writeHead(upRes.statusCode || 502, outHeaders);
    upRes.pipe(res);
  });

  upstream.on("error", (err) => {
    if (!res.headersSent) {
      sendJson(res, 502, {
        error: {
          message: `upstream error: ${err.message}`,
          type: "proxy_error",
        },
      });
    } else {
      res.destroy(err);
    }
  });

  if (bodyBuf !== undefined) {
    upstream.end(bodyBuf);
  } else {
    req.pipe(upstream);
  }
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

    if (req.method === "GET" && url.pathname === "/__sampling_proxy/health") {
      sendJson(res, 200, {
        ok: true,
        target: TARGET_ORIGIN,
        modelSlug: MODEL_SLUG,
        forcedSampling: { ...FORCED_SAMPLING },
        thinkingMode: THINKING_MODE,
      });
      return;
    }

    if (isChatCompletions(req.method, url.pathname)) {
      let raw;
      try {
        raw = await readBody(req);
      } catch (err) {
        sendJson(res, 400, {
          error: { message: `failed to read body: ${err.message}`, type: "invalid_request" },
        });
        return;
      }

      let parsed;
      try {
        const text = raw.length ? raw.toString("utf8") : "{}";
        parsed = JSON.parse(text);
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("body must be a JSON object");
        }
      } catch (err) {
        sendJson(res, 400, {
          error: {
            message: `malformed chat completions JSON: ${err.message}`,
            type: "invalid_request",
          },
        });
        return;
      }

      parsed.temperature = FORCED_SAMPLING.temperature;
      parsed.top_p = FORCED_SAMPLING.top_p;
      parsed.top_k = FORCED_SAMPLING.top_k;
      parsed.min_p = FORCED_SAMPLING.min_p;
      parsed.presence_penalty = FORCED_SAMPLING.presence_penalty;
      parsed.repetition_penalty = FORCED_SAMPLING.repetition_penalty;

      // Thinking dimension: disabled reproduces baseline thinkfix;
      // medium reproduces the medium campaign reasoning force.
      // Archive: medium never puts max_think_tokens on the request body —
      // that constant only feeds attestation expected_max_think_tokens.
      const ctk =
        parsed.chat_template_kwargs &&
        typeof parsed.chat_template_kwargs === "object" &&
        !Array.isArray(parsed.chat_template_kwargs)
          ? { ...parsed.chat_template_kwargs }
          : {};
      if (THINKING_MODE === "medium") {
        parsed.reasoning_effort = FORCED_REASONING_MEDIUM.reasoning_effort;
        ctk.enable_thinking = FORCED_REASONING_MEDIUM.enable_thinking;
      } else {
        // Stabilize Qwen think-span validation for agentic multi-turn runs.
        // Does not alter the six forced sampling fields above.
        ctk.enable_thinking = false;
        // Inject no reasoning fields (do not set reasoning_effort / max_think_tokens).
      }
      parsed.chat_template_kwargs = ctk;

      const rewritten = Buffer.from(JSON.stringify(parsed), "utf8");

      // Attestation key set is mode-dependent, matching each archive exactly.
      const attestation = {
        timestamp: new Date().toISOString(),
        modelSlug: MODEL_SLUG,
        path: url.pathname,
        requestModel: parsed.model ?? null,
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        top_k: parsed.top_k,
        min_p: parsed.min_p,
        presence_penalty: parsed.presence_penalty,
        repetition_penalty: parsed.repetition_penalty,
      };
      if (THINKING_MODE === "medium") {
        attestation.reasoning_effort = parsed.reasoning_effort;
        attestation.enable_thinking = parsed.chat_template_kwargs.enable_thinking;
        attestation.expected_max_think_tokens = FORCED_REASONING_MEDIUM.max_think_tokens;
      }
      appendAttestation(attestation);

      forward(req, res, rewritten);
      return;
    }

    // Pass-through: stream body as-is.
    forward(req, res, undefined);
  } catch (err) {
    if (!res.headersSent) {
      sendJson(res, 502, {
        error: { message: `proxy failure: ${err.message}`, type: "proxy_error" },
      });
    } else {
      res.destroy(err);
    }
  }
});

function shutdown(signal) {
  console.error(`sampling-proxy: ${signal}, shutting down`);
  server.close(() => process.exit(0));
  // Force-exit if hang.
  setTimeout(() => process.exit(0), 5000).unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

server.listen(LISTEN_PORT, LISTEN_HOST, () => {
  console.error(
    `sampling-proxy: listening on http://${LISTEN_HOST}:${LISTEN_PORT} -> ${TARGET_ORIGIN} (${MODEL_SLUG})`,
  );
});
