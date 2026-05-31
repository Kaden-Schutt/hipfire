#!/usr/bin/env bun
// Pi-workload prompt-cache harness — REAL agentic loop.
//
// Emulates Pi driving the daemon on the Blink-hash task: each turn the model
// GENERATES an assistant turn (tool_call), we execute the tool exactly as Pi
// would (bash/read/write), then feed the model's VERBATIM assistant message +
// the tool result back as history for the next turn. This is the faithful
// reproduction of the shared transcript's chat/tool-call chain — and it's what
// the prompt cache needs: the history's assistant turns must be byte-identical
// to what the model emitted, so `asst_turn_cache` replays them and each turn is
// a pure extension (LCP == prior conversation).
//
// Reports per-turn usage.prompt_tokens_details.cached_tokens. A working cache
// means cached_tokens(turn k) ≈ prompt_tokens(turn k-1).
//
// Usage: bun scripts/pi_cache_replay.ts [--port 11435] [--model qwen3.6-27b.mq4]
//        [--turns 7] [--gen 128]
import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const opt = (n: string, d: string) => { const i = args.indexOf(n); return i >= 0 && args[i + 1] ? args[i + 1] : d; };
const PORT = parseInt(opt("--port", "11435"), 10);
const MODEL = opt("--model", "qwen3.6-27b.mq4");
const MAX_TURNS = parseInt(opt("--turns", "7"), 10);
const GEN = parseInt(opt("--gen", "128"), 10);
const TRUNC = 50000; // Pi truncates large tool outputs (~50KB)

const SYSTEM =
  "You are Pi, a coding agent with access to tools: bash, read, write. " +
  "Accomplish the user's task by calling one tool at a time. After reading the " +
  "paper, begin implementing.";
const USER =
  "Implement a Blink-hash tree in Zig 0.16.0. Download and read the paper to get full context. " +
  "https://www.vldb.org/pvldb/vol16/p1235-cha.pdf. The rest of this repo is not relevant. " +
  "Make sure to use pdftotext to read it";

const TOOLS = [
  { type: "function", function: { name: "bash", description: "Run a bash command", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } } },
  { type: "function", function: { name: "read", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } } },
  { type: "function", function: { name: "write", description: "Write a file", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } } },
];

function execTool(name: string, a: any): string {
  try {
    if (name === "bash") {
      const r = spawnSync("bash", ["-c", a.command ?? ""], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024, timeout: 120000 });
      const out = (r.stdout ?? "") + (r.stderr ?? "");
      return out.length > TRUNC ? out.slice(0, TRUNC) + "\n[truncated]" : (out || "(no output)");
    }
    if (name === "read") {
      const out = readFileSync(a.path, "utf8");
      return out.length > TRUNC ? out.slice(0, TRUNC) + "\n[truncated]" : out;
    }
    if (name === "write") { writeFileSync(a.path, a.content ?? ""); return `wrote ${a.path}`; }
  } catch (e: any) { return `error: ${e?.message || e}`; }
  return `unknown tool ${name}`;
}

const THINK = args.includes("--think");
async function callModel(messages: any[]) {
  const body: any = { model: MODEL, messages, tools: TOOLS, tool_choice: "auto", max_tokens: GEN, temperature: 0, stream: false };
  if (!THINK) body.chat_template_kwargs = { enable_thinking: false };
  else body.reasoning = { effort: "medium" };
  const t0 = performance.now();
  const res = await fetch(`http://127.0.0.1:${PORT}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const wall = (performance.now() - t0) / 1000;
  const j: any = await res.json();
  return { j, wall };
}

const messages: any[] = [{ role: "system", content: SYSTEM }, { role: "user", content: USER }];
console.log(`Pi agentic cache loop → model=${MODEL} turns=${MAX_TURNS} gen=${GEN}`);
console.log("turn  prompt_tok  cached_tok  reuse%   out_sig     tool                    wall_s");
console.log("--------------------------------------------------------------------------");
let prevPrompt = 0;
for (let k = 0; k < MAX_TURNS; k++) {
  const { j, wall } = await callModel(messages);
  const u = j.usage ?? {};
  const cached = u.prompt_tokens_details?.cached_tokens ?? 0;
  const prompt = u.prompt_tokens ?? 0;
  const choice = j.choices?.[0];
  const msg = choice?.message ?? {};
  const tcs = msg.tool_calls ?? [];
  const reusePct = prevPrompt > 0 ? (100 * cached / prevPrompt).toFixed(1) : "—";
  let toolDesc = "(no tool)";
  if (tcs.length > 0) {
    const tc = tcs[0];
    let a: any = {}; try { a = JSON.parse(tc.function.arguments || "{}"); } catch {}
    toolDesc = `${tc.function.name}:${(a.command || a.path || "").slice(0, 18)}`;
  }
  // Deterministic per-turn output signature (content + tool_calls) for
  // byte-identical comparison across cache-on vs forced-full runs.
  const sig = (() => {
    const h = require("node:crypto").createHash("sha1");
    h.update(JSON.stringify({ c: msg.content ?? "", t: tcs.map((x: any) => ({ n: x.function?.name, a: x.function?.arguments })) }));
    return h.digest("hex").slice(0, 10);
  })();
  console.log(`${String(k).padEnd(5)} ${String(prompt).padEnd(11)} ${String(cached).padEnd(11)} ${String(reusePct).padEnd(8)} ${String(sig).padEnd(11)} ${toolDesc.padEnd(23)} ${wall.toFixed(1)}`);
  prevPrompt = prompt;
  // Feed the model's VERBATIM assistant message back, then execute tools.
  messages.push({ role: "assistant", content: msg.content ?? "", ...(tcs.length ? { tool_calls: tcs } : {}) });
  if (tcs.length === 0) { console.log("  (model stopped calling tools)"); break; }
  for (const tc of tcs) {
    let a: any = {}; try { a = JSON.parse(tc.function.arguments || "{}"); } catch {}
    const result = execTool(tc.function.name, a);
    messages.push({ role: "tool", tool_call_id: tc.id, content: result });
  }
}
