#!/usr/bin/env bun
// Deterministic byte-identical check for DFlash prompt-cache reuse.
//
// Turn 1 = [system, user] (cold on any daemon). Capture the model's verbatim
// assistant emission E1. Turn 2 = [system, user, E1, FIXED tool result] —
// a pure extension, so on a cache-enabled daemon it's an incremental HIT, and
// on a cache-disabled daemon (HIPFIRE_QWEN_PROMPT_CACHE=0) it's a full prefill.
// The model is temp=0 (greedy) and the tool result is FIXED, so the ONLY
// variable is incremental-vs-full prefill. If reuse is byte-correct, turn-2's
// output signature is identical across the two daemon configs.
//
// Run against the cache-ON daemon, then the cache-OFF daemon; compare the
// printed `turn2_sig`.
import { createHash } from "node:crypto";

const PORT = parseInt(process.argv[2] || "11435", 10);
const MODEL = "qwen3.6-27b.mq4";
const SYSTEM = "You are Pi, a coding agent with tools: bash. Call one tool at a time.";
const USER = "Run `echo hello` with bash, then tell me what it printed.";
const FIXED_TOOL = "hello\n";
const TOOLS = [{ type: "function", function: { name: "bash", description: "Run a bash command", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } } }];

function sig(msg: any): string {
  const tcs = msg.tool_calls ?? [];
  const h = createHash("sha1");
  h.update(JSON.stringify({ c: msg.content ?? "", t: tcs.map((x: any) => ({ n: x.function?.name, a: x.function?.arguments })) }));
  return h.digest("hex").slice(0, 12);
}
async function post(messages: any[]) {
  const body = { model: MODEL, messages, tools: TOOLS, tool_choice: "auto", max_tokens: 96, temperature: 0, stream: false, chat_template_kwargs: { enable_thinking: false } };
  const r = await fetch(`http://127.0.0.1:${PORT}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j: any = await r.json();
  return { msg: j.choices?.[0]?.message ?? {}, cached: j.usage?.prompt_tokens_details?.cached_tokens ?? 0, prompt: j.usage?.prompt_tokens ?? 0 };
}

// Turn 1 (cold).
const t1 = await post([{ role: "system", content: SYSTEM }, { role: "user", content: USER }]);
console.log(`turn1: sig=${sig(t1.msg)} cached=${t1.cached} prompt=${t1.prompt} tool=${(t1.msg.tool_calls?.[0]?.function?.name) ?? "(none)"}`);
const e1 = t1.msg;
const tcs = e1.tool_calls ?? [];
// Turn 2 = pure extension with the model's verbatim E1 + fixed tool result.
const msgs2: any[] = [{ role: "system", content: SYSTEM }, { role: "user", content: USER }, { role: "assistant", content: e1.content ?? "", ...(tcs.length ? { tool_calls: tcs } : {}) }];
if (tcs.length) msgs2.push({ role: "tool", tool_call_id: tcs[0].id, content: FIXED_TOOL });
const t2 = await post(msgs2);
console.log(`turn2: sig=${sig(t2.msg)} cached=${t2.cached} prompt=${t2.prompt}  (HIT if cached>0)`);
console.log(`turn2_sig=${sig(t2.msg)}`);
