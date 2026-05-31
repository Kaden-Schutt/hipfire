#!/usr/bin/env bun
// Eyeball the TEXT a resumed DFlash turn emits (CLAUDE.md requires decoded-text
// inspection for spec-decode changes). Primes a long context, then sends a
// divergent prefix render (→ resume) with a prose-eliciting question and decodes
// a longer window. Prints the text + a crude repetition check.
const PORT = parseInt(process.argv[2] || "11435", 10);
const MODEL = process.argv[3] || "qwen3.6-27b.mq4";
const SYS = "You are a helpful assistant.";
// A coherent, distinctive passage repeated with numbering so it crosses several
// checkpoint intervals and SHORT is a clean textual prefix of LONG.
const para = (n: number) =>
  `Section ${n}: The cache stores key-value tensors per layer; rotary positions index the slot. `;
let body = "";
for (let i = 0; i < 220; i++) body += para(i);
const LONG = body + " Given all sections, summarize the caching scheme in one paragraph.";
const SHORT = body.slice(0, Math.floor(body.length * 0.55)) + " Based on the sections so far, explain how rotary positions relate to KV cache slots, in detail.";

async function post(user: string, maxTok: number) {
  const body = { model: MODEL, messages: [{ role: "system", content: SYS }, { role: "user", content: user }], max_tokens: maxTok, temperature: 0, stream: false, chat_template_kwargs: { enable_thinking: false } };
  const r = await fetch(`http://127.0.0.1:${PORT}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j: any = await r.json();
  return { txt: j.choices?.[0]?.message?.content ?? "", cached: j.usage?.prompt_tokens_details?.cached_tokens ?? 0 };
}
await post(LONG, 8); // prime (checkpoints captured)
const r = await post(SHORT, 220); // divergent prefix → resume, longer decode
console.log(`cached=${r.cached} (resume reused prefix)`);
console.log("---- decoded text ----");
console.log(r.txt);
console.log("---- repetition check ----");
const toks = r.txt.split(/\s+/).filter(Boolean);
const uniq = new Set(toks).size;
console.log(`words=${toks.length} unique=${uniq} ratio=${(uniq / Math.max(1, toks.length)).toFixed(3)} ${uniq / Math.max(1, toks.length) < 0.3 ? "← LOW (possible attractor)" : "(healthy)"}`);
