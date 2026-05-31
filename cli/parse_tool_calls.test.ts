// Bun-native test for the defensive tool-call parser (#111).
//
// Focuses on shapes captured from MQ4 quantization drift: the "flat" form
// (sibling args instead of nested `arguments`) and the "XML-tag" form
// (`<plain>NAME</param>`). Strict OpenAI-spec input is the control case.
//
// Run: bun test cli/parse_tool_calls.test.ts
//
// We don't import from index.ts to keep the test runnable without spinning up
// the full CLI module graph (it has top-level side effects). Instead the
// parser is duplicated here. Keep in sync with cli/index.ts:parseToolCalls.

import { test, expect } from "bun:test";

function parseOneToolCall(raw: string): { name: string; arguments: any; repaired: boolean } | null {
  try {
    const tc = JSON.parse(raw);
    if (tc && typeof tc === "object" && typeof tc.name === "string") {
      if (tc.arguments !== undefined) {
        return { name: tc.name, arguments: tc.arguments, repaired: false };
      }
      const drop = new Set(["name", "type", "id", "function"]);
      const args: Record<string, any> = {};
      let coerced = false;
      for (const [k, v] of Object.entries(tc)) {
        if (drop.has(k)) continue;
        args[k] = v;
        coerced = true;
      }
      if (coerced) return { name: tc.name, arguments: args, repaired: true };
      return { name: tc.name, arguments: {}, repaired: false };
    }
  } catch {}

  const xmlPatterns = [
    /^<\s*plain\s*>\s*([A-Za-z_][\w.]*)\s*<\s*\/\s*param\s*>/,
    /^<\s*function\s*=\s*([A-Za-z_][\w.]*)\s*>/,
    /^<\s*tool\s*name\s*=\s*"?([A-Za-z_][\w.]*)"?\s*>/,
  ];
  // Probe (1): Qwen3.5/3.6 native `<function=NAME>...<parameter=K>V</parameter>...</function>`.
  const fnMatch = raw.match(/^<\s*function\s*=\s*([A-Za-z_][\w.]*)\s*>([\s\S]*?)(?:<\s*\/\s*function\s*>|$)/);
  if (fnMatch) {
    const fname = fnMatch[1];
    const body = fnMatch[2];
    const params: Record<string, any> = {};
    const paramRe = /<\s*parameter\s*=\s*([A-Za-z_][\w.]*)\s*>([\s\S]*?)<\s*\/\s*parameter\s*>/g;
    let anyParam = false;
    for (const pm of body.matchAll(paramRe)) {
      params[pm[1]] = coerceParamValue(pm[2].trim());
      anyParam = true;
    }
    if (anyParam) return { name: fname, arguments: params, repaired: true };
  }
  for (const pat of xmlPatterns) {
    const nm = raw.match(pat);
    if (!nm) continue;
    const after = raw.slice(nm[0].length).trim();
    const args = extractFirstJsonObject(after);
    if (args !== null) return { name: nm[1], arguments: args, repaired: true };
    return { name: nm[1], arguments: {}, repaired: true };
  }
  // Form 4: last-resort field-level extraction (broken/truncated JSON).
  // Keep in sync with cli/index.ts:parseOneToolCall Form 4.
  const nameMatch = raw.match(/(?<![A-Za-z_])["']?name["']?\s*:\s*["']([A-Za-z_][\w.-]*)["']/);
  if (nameMatch) {
    const fname = nameMatch[1];
    const argsLeader = raw.match(/["']arguments["']\s*:\s*/);
    let args: any = null;
    if (argsLeader && argsLeader.index !== undefined) {
      args = extractFirstJsonObject(raw.slice(argsLeader.index + argsLeader[0].length));
    }
    if (args === null) args = extractFirstJsonObject(raw);
    if (args === null) {
      // Balanced-but-off-spec object (trailing comma, …) → keep call, empty
      // args. Truncated (no balanced object) → drop so it surfaces as content
      // + finish_reason, not a phantom write({}).
      if (jsonObjectIsComplete(raw)) return { name: fname, arguments: {}, repaired: true };
      return null;
    }
    return { name: fname, arguments: args, repaired: true };
  }
  return null;
}

function coerceParamValue(s: string): any {
  if (s === "") return "";
  if (s === "true" || s === "false" || s === "null") return JSON.parse(s);
  if (/^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(s)) {
    const n = Number(s);
    if (Number.isFinite(n)) return n;
  }
  if ((s.startsWith("{") && s.endsWith("}")) || (s.startsWith("[") && s.endsWith("]"))) {
    try { return JSON.parse(s); } catch {}
  }
  return s;
}

function extractFirstJsonObject(s: string): any | null {
  const start = s.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let escape = false;
  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (escape) { escape = false; continue; }
      if (ch === "\\") { escape = true; continue; }
      if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') { inStr = true; continue; }
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        try { return JSON.parse(s.slice(start, i + 1)); }
        catch { return null; }
      }
    }
  }
  return null;
}

function jsonObjectIsComplete(s: string): boolean {
  const start = s.indexOf("{");
  if (start < 0) return false;
  let depth = 0, inStr = false, escape = false;
  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (escape) { escape = false; continue; }
      if (ch === "\\") { escape = true; continue; }
      if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') { inStr = true; continue; }
    if (ch === "{") depth++;
    else if (ch === "}") { depth--; if (depth === 0) return true; }
  }
  return false;
}

test("strict OpenAI form parses without repair flag", () => {
  const r = parseOneToolCall('{"name": "write", "arguments": {"path": "/tmp/x", "content": "y"}}');
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({ path: "/tmp/x", content: "y" });
  expect(r!.repaired).toBe(false);
});

test("zero-arg tool call passes through with empty args", () => {
  const r = parseOneToolCall('{"name": "list_files"}');
  expect(r).not.toBeNull();
  expect(r!.name).toBe("list_files");
  expect(r!.arguments).toEqual({});
  expect(r!.repaired).toBe(false);
});

test("flat form (sibling args, no `arguments` wrapper) is repaired", () => {
  // Captured from qwen3.6:27b MQ4 multi-tool stream on 2026-05-01 (#111).
  const raw = '{"name": "write", "path": "/tmp/rate_limiter.py", "content": "print(1)"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({ path: "/tmp/rate_limiter.py", content: "print(1)" });
  expect(r!.repaired).toBe(true);
});

test("flat form with extra metadata keys (id, type) drops them", () => {
  const raw = '{"id": "abc", "type": "function", "name": "bash", "command": "ls -la"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("bash");
  expect(r!.arguments).toEqual({ command: "ls -la" });
  expect(r!.repaired).toBe(true);
});

test("XML-corruption form <plain>NAME</param> {ARGS} is repaired", () => {
  // Captured from reporter Fluorax (#111 issue body).
  const raw = '<plain>write</param> {"path": "/home/mike/rate_limiter.py", "content": "y"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({ path: "/home/mike/rate_limiter.py", content: "y" });
  expect(r!.repaired).toBe(true);
});

test("XML <function=NAME> variant is repaired", () => {
  const raw = '<function=read>{"path": "/etc/passwd"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("read");
  expect(r!.arguments).toEqual({ path: "/etc/passwd" });
  expect(r!.repaired).toBe(true);
});

test("XML form with unparseable JSON tail emits empty args, preserves name", () => {
  const raw = '<plain>write</param> {"path": "/tmp/x", "content": "broken';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({});
  expect(r!.repaired).toBe(true);
});

test("totally unparseable garbage returns null", () => {
  const r = parseOneToolCall("totally not a tool call");
  expect(r).toBeNull();
});

test("nested arguments object survives extraction with strings containing braces", () => {
  // Common content shape: code with `{` inside; balanced-brace walker must
  // not be tricked by braces inside JSON strings.
  const raw = '<plain>write</param> {"path": "/tmp/x.py", "content": "def f(): return {1: 2}"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({ path: "/tmp/x.py", content: "def f(): return {1: 2}" });
});

test("escaped quotes inside JSON strings are handled", () => {
  const raw = '<plain>write</param> {"path": "/tmp/x", "content": "say \\"hi\\""}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.arguments).toEqual({ path: "/tmp/x", content: 'say "hi"' });
});

// Mirror of the nested-stripping logic in parseToolCalls (cli/index.ts).
// Keep in sync.
function parseToolCallsBlock(raw: string) {
  let r = raw.trim();
  let stripped = 0;
  while (r.startsWith("<tool_call>")) {
    r = r.slice("<tool_call>".length).trimStart();
    stripped++;
  }
  if (!r) return null;
  const parsed = parseOneToolCall(r);
  return parsed ? { ...parsed, nestedStripped: stripped } : null;
}

test("nested <tool_call> opener is stripped, payload still parses (#111)", () => {
  // The shape that broke v0.1.9-alpha: model emits two stacked openers
  // before the JSON body lands. Outer regex captures content starting
  // with another `<tool_call>`; the parser must strip that prefix.
  const raw = '<tool_call>\n{"name": "write", "arguments": {"path": "/tmp/x", "content": "y"}}';
  const r = parseToolCallsBlock(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("write");
  expect(r!.arguments).toEqual({ path: "/tmp/x", content: "y" });
  expect(r!.nestedStripped).toBe(1);
});

test("multiple nested <tool_call> openers all get stripped (#111)", () => {
  const raw = '<tool_call>\n<tool_call>\n<tool_call>\n{"name": "read", "arguments": {"path": "/etc"}}';
  const r = parseToolCallsBlock(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("read");
  expect(r!.nestedStripped).toBe(3);
});

test("no nested opener leaves payload unmodified (#111)", () => {
  const raw = '{"name": "bash", "arguments": {"command": "ls"}}';
  const r = parseToolCallsBlock(raw);
  expect(r).not.toBeNull();
  expect(r!.nestedStripped).toBe(0);
  expect(r!.repaired).toBe(false);
});

test("nested openers with empty body returns null (no false-positive call)", () => {
  // Pure attractor with no JSON behind the openers — must NOT emit a
  // bogus tool call. The handler should let this fall through to the
  // "no tool calls" path.
  const raw = '<tool_call>\n<tool_call>\n<tool_call>';
  const r = parseToolCallsBlock(raw);
  expect(r).toBeNull();
});

// --- Qwen3.5/3.6 native XML output (Phase 2, Jinja path) ---

test("Qwen3.6 native <function=NAME>...<parameter=K>V</parameter>...</function> shape", () => {
  // Bytes captured from the daemon smoke (qwen3.6-27b + DFlash drafter
  // + HIPFIRE_JINJA_CHAT=1) — the template emits parameter siblings
  // separated by newlines, with one leading + one trailing newline
  // inside each <parameter> body.
  const raw = '<function=get_weather>\n<parameter=city>\nSan Francisco\n</parameter>\n<parameter=unit>\nf\n</parameter>\n</function>';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("get_weather");
  expect(r!.arguments).toEqual({ city: "San Francisco", unit: "f" });
  expect(r!.repaired).toBe(true);
});

test("Qwen3.6 XML coerces typed parameter values (numbers / bool / null / json)", () => {
  // Tool runners downstream expect typed args — coerce string bodies
  // that look like JSON primitives so `{"count": 42}` doesn't arrive
  // as `{"count": "42"}`. Strings that don't match a JSON shape stay
  // strings (Qwen often emits e.g. paths or free-text).
  const raw = '<function=execute>'
    + '<parameter=count>42</parameter>'
    + '<parameter=ratio>3.14</parameter>'
    + '<parameter=enabled>true</parameter>'
    + '<parameter=missing>null</parameter>'
    + '<parameter=cfg>{"k":1}</parameter>'
    + '<parameter=tags>["a","b"]</parameter>'
    + '<parameter=path>/tmp/x.txt</parameter>'
    + '</function>';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.arguments).toEqual({
    count: 42,
    ratio: 3.14,
    enabled: true,
    missing: null,
    cfg: { k: 1 },
    tags: ["a", "b"],
    path: "/tmp/x.txt",
  });
});

test("Qwen3.6 XML with zero parameters (no-arg call)", () => {
  // `<function=NAME></function>` with no parameter siblings — the
  // probe-(1) anyParam guard must NOT fire (else `arguments` would be
  // {} but `repaired` set spuriously). Should fall through to the
  // legacy probe-(3) shape which emits empty args + repaired=true.
  const raw = '<function=list_dirs></function>';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("list_dirs");
  expect(r!.arguments).toEqual({});
});

test("<function=NAME>{JSON} (MQ4 corruption shape) still parses via probe-(2)", () => {
  // Regression: the existing MQ4 XML-corruption repair must still work
  // — probe-(1)'s parameter-sibling logic falls through cleanly when
  // there are no <parameter=...> blocks but a JSON-object args body
  // is present.
  const raw = '<function=read>{"path": "/etc/passwd"}';
  const r = parseOneToolCall(raw);
  expect(r).not.toBeNull();
  expect(r!.name).toBe("read");
  expect(r!.arguments).toEqual({ path: "/etc/passwd" });
  expect(r!.repaired).toBe(true);
});

test("truncated tool call (unterminated arguments) returns null — no fabricated empty args", () => {
  // A `write` cut off mid-`content` (by max_tokens or a grammar force-close).
  // The args object never closes, so no balanced object is recoverable. Must
  // NOT fabricate `{}` and present write({}) to the client as executable —
  // that fails schema validation (the write-tool empty-args incident). The
  // call is dropped so the emission surfaces as content + finish_reason.
  const r = parseOneToolCall('{"name": "write", "arguments": {"path": "/tmp/big.zig", "content": "const std = @im');
  expect(r).toBeNull();
});

test("broken outer JSON but a COMPLETE arguments object still parses via Form 4", () => {
  // Special-token leakage corrupted the outer structure (leading `{` lost),
  // but the args object is balanced — Form 4 must still recover name + args,
  // distinguishing real recovery from the truncation case above.
  const r = parseOneToolCall('name": "read", "arguments": {"path": "/tmp/x"}');
  expect(r).not.toBeNull();
  expect(r!.name).toBe("read");
  expect(r!.arguments).toEqual({ path: "/tmp/x" });
});

test("complete-but-off-spec args (trailing comma) keeps the call with empty args", () => {
  // Brace-balanced but invalid JSON (trailing comma) — a model formatting
  // glitch, NOT truncation. The call is preserved by name (mirrors the
  // daemon's form4_handles_trailing_comma); only genuinely truncated calls
  // are dropped. Keeps daemon/CLI parsers in agreement so the asst-turn
  // cache fingerprint doesn't diverge.
  const r = parseOneToolCall('{"name": "read", "arguments": {"path": "/x",},}');
  expect(r).not.toBeNull();
  expect(r!.name).toBe("read");
  expect(r!.arguments).toEqual({});
});
