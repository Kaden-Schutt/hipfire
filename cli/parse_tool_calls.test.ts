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

// ── Non-Qwen tool-call parsers (LFM2.5 arch 11, MiniMax-M2 arch 10) ──────────
// Duplicated from cli/index.ts:parseToolCalls — keep in sync.
function makeCallId(): string {
  return `call_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}
function splitTopLevel(s: string, sep: string): string[] {
  const out: string[] = [];
  let depth = 0, q: string | null = null, esc = false, cur = "";
  for (const ch of s) {
    if (esc) { cur += ch; esc = false; continue; }
    if (q) { cur += ch; if (ch === "\\") esc = true; else if (ch === q) q = null; continue; }
    if (ch === "'" || ch === '"') { q = ch; cur += ch; continue; }
    if (ch === "(" || ch === "[" || ch === "{") { depth++; cur += ch; continue; }
    if (ch === ")" || ch === "]" || ch === "}") { depth--; cur += ch; continue; }
    if (ch === sep && depth === 0) { out.push(cur); cur = ""; continue; }
    cur += ch;
  }
  if (cur.length) out.push(cur);
  return out;
}
function topLevelEq(s: string): number {
  let depth = 0, q: string | null = null, esc = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (esc) { esc = false; continue; }
    if (q) { if (ch === "\\") esc = true; else if (ch === q) q = null; continue; }
    if (ch === "'" || ch === '"') { q = ch; continue; }
    if (ch === "(" || ch === "[" || ch === "{") depth++;
    else if (ch === ")" || ch === "]" || ch === "}") depth--;
    else if (ch === "=" && depth === 0) return i;
  }
  return -1;
}
function parsePyValue(v: string): any {
  v = v.trim();
  if (v === "True") return true;
  if (v === "False") return false;
  if (v === "None") return null;
  if (v.length >= 2 && v[0] === "'" && v[v.length - 1] === "'") {
    const body = v.slice(1, -1).replace(/\\'/g, "'").replace(/(?<!\\)"/g, '\\"');
    try { return JSON.parse('"' + body + '"'); } catch { return v.slice(1, -1); }
  }
  try { return JSON.parse(v); } catch { return v; }
}
function parseBracketCallToolCalls(text: string): { content: string | null; tool_calls: any[] | null } {
  const startTag = "<|tool_call_start|>", endTag = "<|tool_call_end|>";
  const si = text.indexOf(startTag);
  const before = text.slice(0, si).trim();
  let inner = text.slice(si + startTag.length);
  const ei = inner.indexOf(endTag);
  if (ei >= 0) inner = inner.slice(0, ei);
  inner = inner.trim();
  if (inner.startsWith("[")) inner = inner.slice(1);
  if (inner.endsWith("]")) inner = inner.slice(0, -1);
  const tool_calls: any[] = [];
  for (const callStr of splitTopLevel(inner, ",").map((s) => s.trim()).filter(Boolean)) {
    const m = callStr.match(/^([A-Za-z_][\w.]*)\s*\(([\s\S]*)\)\s*$/);
    if (!m) continue;
    const name = m[1];
    const argsStr = m[2].trim();
    const args: Record<string, any> = {};
    let pos = 0;
    if (argsStr) {
      for (const partRaw of splitTopLevel(argsStr, ",")) {
        const part = partRaw.trim();
        if (!part) continue;
        const eq = topLevelEq(part);
        if (eq < 0) { args[String(pos++)] = parsePyValue(part); continue; }
        args[part.slice(0, eq).trim()] = parsePyValue(part.slice(eq + 1));
      }
    }
    tool_calls.push({ id: makeCallId(), type: "function", function: { name, arguments: JSON.stringify(args) } });
  }
  if (!tool_calls.length) return { content: text, tool_calls: null };
  return { content: before || null, tool_calls };
}
function parseXmlInvokeToolCalls(text: string): { content: string | null; tool_calls: any[] | null } {
  const startTag = "<minimax:tool_call>";
  const si = text.indexOf(startTag);
  const before = text.slice(0, si).trim();
  let region = text.slice(si + startTag.length);
  const ei = region.indexOf("</minimax:tool_call>");
  if (ei >= 0) region = region.slice(0, ei);
  const tool_calls: any[] = [];
  const invokeRe = /<invoke\s+name="([^"]+)"\s*>([\s\S]*?)(?:<\/invoke>|$)/g;
  for (const im of region.matchAll(invokeRe)) {
    const name = im[1];
    const args: Record<string, any> = {};
    const paramRe = /<parameter\s+name="([^"]+)"\s*>([\s\S]*?)<\/parameter>/g;
    for (const pm of im[2].matchAll(paramRe)) {
      const raw = pm[2].replace(/^\s*\n/, "").replace(/\n\s*$/, "").trim();
      let val: any;
      try { val = JSON.parse(raw); } catch { val = raw; }
      args[pm[1]] = val;
    }
    tool_calls.push({ id: makeCallId(), type: "function", function: { name, arguments: JSON.stringify(args) } });
  }
  if (!tool_calls.length) return { content: text, tool_calls: null };
  return { content: before || null, tool_calls };
}

// LFM2.5 — the exact bytes the model emitted in the e2e verification.
test("LFM2.5 <|tool_call_start|>[fn(k=\"v\")] parses to OpenAI tool_calls", () => {
  const r = parseBracketCallToolCalls('<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>');
  expect(r.tool_calls).not.toBeNull();
  expect(r.tool_calls!.length).toBe(1);
  expect(r.tool_calls![0].function.name).toBe("get_weather");
  expect(JSON.parse(r.tool_calls![0].function.arguments)).toEqual({ location: "Paris" });
  expect(r.content).toBeNull();
});

test("LFM2.5 single-quoted strings + typed args (int/bool) coerce", () => {
  const r = parseBracketCallToolCalls("<|tool_call_start|>[set(name='Bob', age=42, active=True, note=None)]<|tool_call_end|>");
  expect(JSON.parse(r.tool_calls![0].function.arguments)).toEqual({ name: "Bob", age: 42, active: true, note: null });
});

test("LFM2.5 multiple calls + string arg containing a comma", () => {
  const r = parseBracketCallToolCalls('<|tool_call_start|>[a(x=1), b(msg="hello, world", arr=[1,2])]<|tool_call_end|>');
  expect(r.tool_calls!.length).toBe(2);
  expect(r.tool_calls![0].function.name).toBe("a");
  expect(JSON.parse(r.tool_calls![1].function.arguments)).toEqual({ msg: "hello, world", arr: [1, 2] });
});

test("LFM2.5 zero-arg call", () => {
  const r = parseBracketCallToolCalls("<|tool_call_start|>[now()]<|tool_call_end|>");
  expect(r.tool_calls![0].function.name).toBe("now");
  expect(JSON.parse(r.tool_calls![0].function.arguments)).toEqual({});
});

test("LFM2.5 unclosed end tag still parses (defensive)", () => {
  const r = parseBracketCallToolCalls('<|tool_call_start|>[get_weather(location="Paris")]');
  expect(r.tool_calls![0].function.name).toBe("get_weather");
});

// MiniMax-M2 — <minimax:tool_call><invoke name><parameter name> XML.
test("MiniMax-M2 <minimax:tool_call><invoke> parses to OpenAI tool_calls", () => {
  const out = '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">Paris</parameter>\n</invoke>\n</minimax:tool_call>';
  const r = parseXmlInvokeToolCalls(out);
  expect(r.tool_calls!.length).toBe(1);
  expect(r.tool_calls![0].function.name).toBe("get_weather");
  expect(JSON.parse(r.tool_calls![0].function.arguments)).toEqual({ location: "Paris" });
});

test("MiniMax-M2 typed (json) parameter values coerce; multiple invokes", () => {
  const out = '<minimax:tool_call>\n<invoke name="search"><parameter name="q">cats</parameter><parameter name="limit">5</parameter></invoke>\n<invoke name="ping"></invoke>\n</minimax:tool_call>';
  const r = parseXmlInvokeToolCalls(out);
  expect(r.tool_calls!.length).toBe(2);
  expect(JSON.parse(r.tool_calls![0].function.arguments)).toEqual({ q: "cats", limit: 5 });
  expect(JSON.parse(r.tool_calls![1].function.arguments)).toEqual({});
});

test("MiniMax-M2 preserves assistant content before the tool call", () => {
  const out = 'Let me check.\n<minimax:tool_call>\n<invoke name="f"><parameter name="x">1</parameter></invoke>\n</minimax:tool_call>';
  const r = parseXmlInvokeToolCalls(out);
  expect(r.content).toBe("Let me check.");
  expect(r.tool_calls![0].function.name).toBe("f");
});

// Exact bytes the real MiniMax-M2 *embedded* chat_template renders for an
// assistant tool call (note the leading indentation before <parameter> that
// the template emits) — captured via jinja2 render of the model's own template.
test("MiniMax-M2 real embedded-template emission (indented) parses", () => {
  const out = '<minimax:tool_call>\n<invoke name="get_weather">\n                <parameter name="location">Paris</parameter>\n                <parameter name="units">metric</parameter>\n                </invoke>\n</minimax:tool_call>';
  const r = parseXmlInvokeToolCalls(out);
  expect(r.tool_calls!.length).toBe(1);
  expect(r.tool_calls![0].function.name).toBe("get_weather");
  expect(JSON.parse(r.tool_calls![0].function.arguments)).toEqual({ location: "Paris", units: "metric" });
});
