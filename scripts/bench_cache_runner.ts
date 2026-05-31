#!/usr/bin/env bun
// Prompt-cache + DFlash bench RUNNER — one (dflash, thinking) cell.
//
// Hits the daemon's OpenAI streaming endpoint and records, per request:
//   - perf:  ttft_ms, prefill_tok_s, decode_tok_s, tau/cycles (DFlash only), wall_s
//   - cache: prompt_tokens, cached_tokens (reuse%), completion_tokens
//   - sanity: unique-word ratio of the output (attractor smell test)
//   - tools:  expected tool called with parseable args (toolcalls scenario)
//
// Scenarios (each uses a UNIQUE system nonce so they don't cache-pollute):
//   short      — N distinct single-turn prompts (cold-ish; baseline prefill+decode)
//   multiturn  — a growing agentic conversation (verbatim assistant replay +
//                FIXED tool results) → demonstrates prefix-cache reuse per turn
//   divergent  — prime a long context, then send a PREFIX render (dropped tail)
//                → demonstrates checkpoint-RESUME (cached>0 on a non-extension)
//   toolcalls  — distinct tools (bash/read/write/grep) → tool-call correctness + perf
//
// Emits one JSON line per measured aggregate to --out (append); prints a human
// summary to stdout. Thinking is per-request: --think off ⇒ enable_thinking:false,
// --think on ⇒ reasoning.effort=medium (routes to AR).
//
// Usage: bun scripts/bench_cache_runner.ts --port 11435 --model qwen3.6-27b.mq4 \
//          --think off --label "dflash=on think=off" --trials 3 --out /tmp/bench.jsonl
import { appendFileSync } from "node:fs";
import { createHash } from "node:crypto";

const A = process.argv.slice(2);
const opt = (n: string, d: string) => { const i = A.indexOf(n); return i >= 0 && A[i + 1] ? A[i + 1] : d; };
const PORT = parseInt(opt("--port", "11435"), 10);
const MODEL = opt("--model", "qwen3.6-27b.mq4");
const THINK = opt("--think", "off") === "on";
const LABEL = opt("--label", "cell");
const TRIALS = parseInt(opt("--trials", "3"), 10);
const OUT = opt("--out", "");
// Size knobs (27B prefill is ~90 t/s, so keep contexts moderate). --fast shrinks
// everything for harness validation.
const FAST = A.includes("--fast");
// Sizes tuned so each cell is ~3-4 min on 27B while still demonstrating the
// effects: a ~9K-tok divergent prime crosses the 2048 checkpoint interval (so
// resume engages) and fits any per-request timeout; multiturn grows to ~7K.
const DOC_FNS = parseInt(opt("--doc-fns", FAST ? "20" : "48"), 10);
const MT_TURNS = parseInt(opt("--mt-turns", FAST ? "3" : "4"), 10);
const DIV_CHUNKS = parseInt(opt("--div-chunks", FAST ? "3" : "5"), 10);
const URL = `http://127.0.0.1:${PORT}/v1/chat/completions`;

type Stream = {
  text: string; reasoning: string;
  toolCalls: { name: string; args: string }[];
  usage: { prompt: number; cached: number; completion: number };
  timings: { ttft_ms: number; prefill_tok_s: number; decode_tok_s: number; tau: number; cycles: number; dflash: boolean };
  wall_s: number; ok: boolean;
};

async function streamChat(messages: any[], tools: any[] | null, maxTok: number, timeoutMs = 240_000): Promise<Stream> {
  const body: any = {
    model: MODEL, messages, max_tokens: maxTok, temperature: 0, stream: true,
    stream_options: { include_usage: true },
  };
  if (tools) { body.tools = tools; body.tool_choice = "auto"; }
  if (THINK) body.reasoning = { effort: "medium" };
  else body.chat_template_kwargs = { enable_thinking: false };

  const out: Stream = {
    text: "", reasoning: "", toolCalls: [],
    usage: { prompt: 0, cached: 0, completion: 0 },
    timings: { ttft_ms: 0, prefill_tok_s: 0, decode_tok_s: 0, tau: 0, cycles: 0, dflash: false },
    wall_s: 0, ok: false,
  };
  const t0 = performance.now();
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs); // per-request ceiling
  let res: Response;
  try { res = await fetch(URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctl.signal }); }
  catch (e: any) { clearTimeout(timer); out.wall_s = (performance.now() - t0) / 1000; return out; }
  if (!res.body) { clearTimeout(timer); out.wall_s = (performance.now() - t0) / 1000; return out; }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const tcAcc: Record<number, { name: string; args: string }> = {};
  try {
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      const s = line.trim();
      if (!s.startsWith("data:")) continue;
      const payload = s.slice(5).trim();
      if (payload === "[DONE]") continue;
      let j: any; try { j = JSON.parse(payload); } catch { continue; }
      const d = j.choices?.[0]?.delta;
      if (d?.content) out.text += d.content;
      if (d?.reasoning_content) out.reasoning += d.reasoning_content;
      if (Array.isArray(d?.tool_calls)) {
        for (const tc of d.tool_calls) {
          const i = tc.index ?? 0;
          tcAcc[i] = tcAcc[i] || { name: "", args: "" };
          if (tc.function?.name) tcAcc[i].name += tc.function.name;
          if (tc.function?.arguments) tcAcc[i].args += tc.function.arguments;
        }
      }
      if (j.usage) {
        out.usage.prompt = j.usage.prompt_tokens ?? out.usage.prompt;
        out.usage.completion = j.usage.completion_tokens ?? out.usage.completion;
        out.usage.cached = j.usage.prompt_tokens_details?.cached_tokens ?? out.usage.cached;
      }
      if (j.timings) {
        const t = j.timings;
        out.timings.ttft_ms = t.ttft_ms ?? out.timings.ttft_ms;
        out.timings.prefill_tok_s = t.prefill_tok_s ?? out.timings.prefill_tok_s;
        out.timings.decode_tok_s = t.decode_tok_s ?? out.timings.decode_tok_s;
        out.timings.tau = t.tau ?? out.timings.tau;
        out.timings.cycles = t.cycles ?? out.timings.cycles;
        out.timings.dflash = t.dflash ?? out.timings.dflash;
      }
    }
  }
  } catch (e: any) { /* stream aborted/errored — return what we have */ }
  finally { clearTimeout(timer); }
  out.wall_s = (performance.now() - t0) / 1000;
  out.toolCalls = Object.keys(tcAcc).sort((a, b) => +a - +b).map(k => tcAcc[+k]);
  out.ok = out.usage.completion > 0 || out.toolCalls.length > 0;
  return out;
}

const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : 0; };
function uniqRatio(txt: string): number { const w = txt.split(/\s+/).filter(Boolean); return w.length ? new Set(w).size / w.length : 1; }
function record(scenario: string, metrics: any) {
  const row = { label: LABEL, think: THINK ? "on" : "off", scenario, ...metrics };
  if (OUT) appendFileSync(OUT, JSON.stringify(row) + "\n");
  return row;
}

const NONCE = LABEL.replace(/\s+/g, "_");
const sysFor = (s: string) => `You are a precise coding assistant. [bench:${NONCE}:${s}]`;

const TOOLS = [
  { type: "function", function: { name: "bash", description: "Run a bash command", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } } },
  { type: "function", function: { name: "read", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } } },
  { type: "function", function: { name: "write", description: "Write a file", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } } },
  { type: "function", function: { name: "grep", description: "Search files for a pattern", parameters: { type: "object", properties: { pattern: { type: "string" }, path: { type: "string" } }, required: ["pattern"] } } },
];

// A fixed, sizeable document chunk used as deterministic tool output so the
// multiturn context grows reproducibly (~1.6k tokens/chunk).
function docChunk(n: number): string {
  let s = `# Module ${n}\n`;
  for (let i = 0; i < DOC_FNS; i++) s += `fn op_${n}_${i}(x: i64) -> i64 { return x * ${n} + ${i}; } // step ${i} of module ${n}\n`;
  return s;
}

// ---------------- scenarios ----------------

async function scShort() {
  const prompts = [
    "Write a one-line Python list comprehension that squares 0..9.",
    "What does the Rust `?` operator do? One sentence.",
    "Give the bash command to count lines in all .rs files under src/.",
  ];
  const ttft: number[] = [], pre: number[] = [], dec: number[] = [], tau: number[] = [], wall: number[] = [];
  let coh = 1, ok = true;
  for (let t = 0; t < TRIALS; t++) {
    for (let pi = 0; pi < prompts.length; pi++) {
      const r = await streamChat([{ role: "system", content: sysFor(`short${pi}`) }, { role: "user", content: prompts[pi] }], null, 120);
      ttft.push(r.timings.ttft_ms); pre.push(r.timings.prefill_tok_s); dec.push(r.timings.decode_tok_s);
      if (r.timings.dflash) tau.push(r.timings.tau); wall.push(r.wall_s);
      coh = Math.min(coh, uniqRatio(r.text)); ok = ok && r.ok;
    }
  }
  return record("short", { ok, ttft_ms: Math.round(median(ttft)), prefill_tok_s: +median(pre).toFixed(1), decode_tok_s: +median(dec).toFixed(1), tau: tau.length ? +median(tau).toFixed(2) : null, wall_s: +median(wall).toFixed(2), min_uniq_ratio: +coh.toFixed(3), n: ttft.length });
}

async function scMultiturn() {
  // Scripted GROWING chat (append-only): each turn the user appends a new
  // document and the model acks; we feed its VERBATIM reply back (so the
  // asst_turn_cache replays it ⇒ the prior turns stay a byte-exact prefix ⇒
  // cache HIT). Measures prefix-cache reuse as the context grows — independent
  // of whether the model chooses to call tools (covered separately).
  const messages: any[] = [{ role: "system", content: sysFor("multiturn") }];
  const turns: any[] = [];
  let firstTurnPrompt = 0;
  for (let k = 0; k < MT_TURNS; k++) {
    messages.push({ role: "user", content: `Document ${k} follows; reply with exactly "ack ${k}".\n${docChunk(k)}` });
    const r = await streamChat(messages, null, 24);
    const reuse = firstTurnPrompt > 0 ? r.usage.cached / Math.max(1, r.usage.prompt) : 0;
    turns.push({ turn: k, prompt: r.usage.prompt, cached: r.usage.cached, reuse: +reuse.toFixed(3), prefill_tok_s: +r.timings.prefill_tok_s.toFixed(1), decode_tok_s: +r.timings.decode_tok_s.toFixed(1), tau: r.timings.dflash ? +r.timings.tau.toFixed(2) : null, wall_s: +r.wall_s.toFixed(2) });
    firstTurnPrompt = r.usage.prompt;
    messages.push({ role: "assistant", content: r.text }); // verbatim → byte-exact prefix next turn
  }
  // Reuse on turns >=1 (turn 0 is cold). Shows prefill cost staying flat as
  // context grows because only the new document is prefilled.
  const cacheTurns = turns.filter(t => t.turn >= 1);
  const avgReuse = cacheTurns.length ? cacheTurns.reduce((a, t) => a + t.reuse, 0) / cacheTurns.length : 0;
  return record("multiturn", { ok: turns.length > 1 && avgReuse > 0.3, turns, avg_reuse_t1plus: +avgReuse.toFixed(3), final_ctx_tokens: turns[turns.length - 1]?.prompt ?? 0, total_wall_s: +turns.reduce((a, t) => a + t.wall_s, 0).toFixed(2) });
}

async function scDivergent() {
  // Prime a long context, then send a PREFIX render (drops the tail) → a
  // non-extension divergence that exercises checkpoint-RESUME. cached>0 here
  // means the resume reused the prefix instead of cold-prefilling.
  let long = "Here are configuration sections.\n";
  for (let i = 0; i < DIV_CHUNKS; i++) long += docChunk(i);
  const longUser = long + "\nQuestion: how many modules are described above?";
  const shortUser = long.slice(0, Math.floor(long.length * 0.6)) + "\nQuestion: explain what op_0_0 returns, in one sentence.";
  const sys = sysFor("divergent");
  // Prime (cold) — also captures the cold prefill RATE so we can DERIVE the cold
  // cost of the divergent render instead of running a second full cold prefill
  // (saves the biggest request in the scenario). Generous timeout: a one-time
  // cold prime must not be aborted (that would leave the resume nothing to reuse
  // and mis-measure as 0%).
  const p = await streamChat([{ role: "system", content: sys }, { role: "user", content: longUser }], null, 8, 900_000);
  // Divergent prefix render (dropped tail, lcp < prior_len) → checkpoint RESUME.
  const r = await streamChat([{ role: "system", content: sys }, { role: "user", content: shortUser }], null, 80, 900_000);
  const replay = Math.max(0, r.usage.prompt - r.usage.cached);
  const coldRate = p.timings.prefill_tok_s > 0 ? p.timings.prefill_tok_s : r.timings.prefill_tok_s; // tok/s, cold
  const coldEstS = coldRate > 0 ? r.usage.prompt / coldRate : 0;           // cost to cold-prefill all of SHORT
  const resumePrefillS = (r.timings.ttft_ms || 0) / 1000;                  // ≈ resume prefill (replay + first tok)
  return record("divergent", {
    ok: r.ok && r.usage.cached > 0,
    resume_prompt: r.usage.prompt, resume_cached: r.usage.cached,
    resume_reuse: +(r.usage.cached / Math.max(1, r.usage.prompt)).toFixed(3),
    replay_tokens: replay,
    resume_prefill_s: +resumePrefillS.toFixed(2),
    cold_est_s: +coldEstS.toFixed(2),
    prefill_speedup: resumePrefillS > 0 ? +(coldEstS / resumePrefillS).toFixed(1) : null,
    resume_uniq_ratio: +uniqRatio(r.text).toFixed(3),
  });
}

async function scToolcalls() {
  const cases = [
    { user: "List the files in the current directory.", want: "bash" },
    { user: "Read the file config.json.", want: "read" },
    { user: "Create a file hello.py containing a print statement.", want: "write" },
    { user: "Find every occurrence of the word TODO under the src directory.", want: "grep" },
  ];
  const rows: any[] = []; let correct = 0;
  const ttft: number[] = [], dec: number[] = [], tau: number[] = [];
  for (const c of cases) {
    const r = await streamChat([{ role: "system", content: sysFor("tools") }, { role: "user", content: c.user }], TOOLS, 160);
    const tc = r.toolCalls[0];
    let argsOk = false; try { if (tc) { JSON.parse(tc.args || "{}"); argsOk = true; } } catch {}
    const nameOk = tc?.name === c.want;
    if (nameOk && argsOk) correct++;
    rows.push({ want: c.want, got: tc?.name ?? "(none)", args_parse: argsOk, ok: nameOk && argsOk });
    ttft.push(r.timings.ttft_ms); dec.push(r.timings.decode_tok_s); if (r.timings.dflash) tau.push(r.timings.tau);
  }
  return record("toolcalls", { ok: correct === cases.length, correct, total: cases.length, cases: rows, ttft_ms: Math.round(median(ttft)), decode_tok_s: +median(dec).toFixed(1), tau: tau.length ? +median(tau).toFixed(2) : null });
}

// ---------------- main ----------------
(async () => {
  console.error(`[bench] ${LABEL} (think=${THINK ? "on" : "off"}) model=${MODEL} trials=${TRIALS}`);
  // Warmup (kernel cache + DPM) — discarded.
  await streamChat([{ role: "system", content: sysFor("warmup") }, { role: "user", content: "Say hi." }], null, 16);
  const guard = async (name: string, fn: () => Promise<any>) => {
    try { return await fn(); }
    catch (e: any) { console.error(`[bench] scenario ${name} errored: ${e?.message || e}`); return record(name, { ok: false, error: String(e?.message || e) }); }
  };
  const short = await guard("short", scShort);
  const mt = await guard("multiturn", scMultiturn);
  const dv = await guard("divergent", scDivergent);
  const tools = await guard("toolcalls", scToolcalls);
  const pct = (x: number) => `${((x ?? 0) * 100).toFixed(0)}%`;
  console.log(`\n=== ${LABEL} ===`);
  if (short.error) console.log(`short:     ERROR ${short.error}`);
  else console.log(`short:     ttft=${short.ttft_ms}ms prefill=${short.prefill_tok_s}t/s decode=${short.decode_tok_s}t/s tau=${short.tau ?? "-"} uniq=${short.min_uniq_ratio}`);
  if (mt.error) console.log(`multiturn: ERROR ${mt.error}`);
  else {
    console.log(`multiturn: avg_reuse(t1+)=${pct(mt.avg_reuse_t1plus)} final_ctx=${mt.final_ctx_tokens}tok total_wall=${mt.total_wall_s}s`);
    for (const t of mt.turns ?? []) console.log(`   turn ${t.turn}: prompt=${t.prompt} cached=${t.cached} reuse=${pct(t.reuse)} prefill=${t.prefill_tok_s}t/s decode=${t.decode_tok_s}t/s tau=${t.tau ?? "-"} wall=${t.wall_s}s`);
  }
  if (dv.error) console.log(`divergent: ERROR ${dv.error}`);
  else console.log(`divergent: resume reuse ${pct(dv.resume_reuse)} (replay ${dv.replay_tokens}/${dv.resume_prompt}) prefill ${dv.resume_prefill_s}s vs cold-est ${dv.cold_est_s}s = ${dv.prefill_speedup ?? "-"}x uniq=${dv.resume_uniq_ratio}`);
  if (tools.error) console.log(`toolcalls: ERROR ${tools.error}`);
  else console.log(`toolcalls: ${tools.correct}/${tools.total} correct  ttft=${tools.ttft_ms}ms decode=${tools.decode_tok_s}t/s tau=${tools.tau ?? "-"}`);
})();
