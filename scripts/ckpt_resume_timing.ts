#!/usr/bin/env bun
// Timing proof for divergent-render checkpoint resume on the real model/path.
// Primes a large context, then sends a render that diverges (a prefix of it) —
// the shape that used to force a ~290s cold prefill (→ client "terminated").
// With checkpoint-resume the divergent turn replays only a tail. Reports the
// resume turn's wall time + cached_tokens; grep serve.log for "[qwen-cache
// resume] ... replaying N tokens" to see the replay count.
const PORT = parseInt(process.argv[2] || "11435", 10);
const MODEL = process.argv[3] || "qwen3.6-27b.mq4";
const SYSTEM = "You are a precise assistant. Answer in one short sentence.";
const words: string[] = [];
for (let i = 0; i < 3500; i++) words.push(`item${i}`);
const LONG = "Inventory follows. " + words.join(" ") + ". Question: how many items?";
const SHORT = "Inventory follows. " + words.slice(0, 2400).join(" ") + ". Question: name the first item.";

async function post(user: string, maxTok: number) {
  const body = { model: MODEL, messages: [{ role: "system", content: SYSTEM }, { role: "user", content: user }], max_tokens: maxTok, temperature: 0, stream: false, chat_template_kwargs: { enable_thinking: false } };
  const t0 = performance.now();
  const r = await fetch(`http://127.0.0.1:${PORT}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j: any = await r.json();
  return { wall: (performance.now() - t0) / 1000, cached: j.usage?.prompt_tokens_details?.cached_tokens ?? 0, prompt: j.usage?.prompt_tokens ?? 0 };
}
console.log(`resume timing → model=${MODEL}`);
const prime = await post(LONG, 8);
console.log(`prime(LONG):     prompt=${prime.prompt} cached=${prime.cached} wall=${prime.wall.toFixed(1)}s (cold prime)`);
const resume = await post(SHORT, 16);
console.log(`resume(SHORT):   prompt=${resume.prompt} cached=${resume.cached} wall=${resume.wall.toFixed(1)}s`);
console.log(resume.cached > 0
  ? `RESUME OK — reused ${resume.cached} cached tokens; divergent turn replayed only ${resume.prompt - resume.cached} (would have cold-prefilled ${resume.prompt}).`
  : `NO RESUME — cached=0 (cold). Check routing / checkpoints.`);
