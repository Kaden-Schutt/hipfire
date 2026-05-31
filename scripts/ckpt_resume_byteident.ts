#!/usr/bin/env bun
// Byte-identical correctness check for divergent-render CHECKPOINT RESUME.
//
// Reproduces the real failure shape: the client's new render is a *prefix* of
// the daemon's stored conversation (it dropped trailing history), so lcp <
// prior_len → the daemon must resume from the latest prefill checkpoint ≤ lcp
// and re-prefill only the tail, instead of cold-prefilling from zero.
//
// Plan (single daemon, temp=0 greedy):
//   1. PRIME: post a LONG prompt (max_tokens small). Daemon prefills it +
//      captures checkpoints; conversation_tokens is now long.
//   2. RESUME: post a SHORT prompt whose text is a strict prefix of LONG. Its
//      render diverges from the stored conversation partway → lcp < prior_len
//      → resume fires. Capture out_sig + cached_tokens.  [grep serve.log for
//      "[qwen-cache resume]"]
//   3. RESET, then post the SHORT prompt again COLD (lcp=0, full prefill).
//   4. Compare: resume output sig MUST equal cold output sig (byte-identical).
import { createHash } from "node:crypto";

const PORT = parseInt(process.argv[2] || "11435", 10);
const MODEL = process.argv[3] || "qwen3.6-27b.mq4";
const SYSTEM = "You are a precise assistant. Answer in one short sentence.";
// LONG = many distinct words so it tokenizes to several thousand tokens and
// crosses multiple checkpoint intervals; SHORT is a strict textual prefix.
const NWORDS = parseInt(process.argv[4] || "3000", 10);
const NSLICE = parseInt(process.argv[5] || "1800", 10);
const words: string[] = [];
for (let i = 0; i < NWORDS; i++) words.push(`item${i}`);
const LONG = "Here is a numbered inventory, then a question. " + words.join(" ") + ". Now: how many distinct items did I list, approximately?";
const SHORT = "Here is a numbered inventory, then a question. " + words.slice(0, NSLICE).join(" ") + ". Now: name the very first item.";

function sig(msg: any): string {
  const h = createHash("sha1");
  h.update(JSON.stringify({ c: msg.content ?? "", t: (msg.tool_calls ?? []).map((x: any) => ({ n: x.function?.name, a: x.function?.arguments })) }));
  return h.digest("hex").slice(0, 12);
}
const THINK = process.argv.includes("--think");
async function post(user: string, maxTok: number) {
  const body: any = { model: MODEL, messages: [{ role: "system", content: SYSTEM }, { role: "user", content: user }], max_tokens: maxTok, temperature: 0, stream: false };
  if (THINK) body.reasoning = { effort: "medium" }; else body.chat_template_kwargs = { enable_thinking: false };
  const r = await fetch(`http://127.0.0.1:${PORT}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j: any = await r.json();
  return { msg: j.choices?.[0]?.message ?? {}, cached: j.usage?.prompt_tokens_details?.cached_tokens ?? 0, prompt: j.usage?.prompt_tokens ?? 0 };
}
async function reset() {
  // The CLI owns the reset protocol; emulate a fresh conversation by posting a
  // throwaway 1-token request after the daemon-side reset. Simplest portable
  // path: rely on the daemon's per-request LCP — a cold SHORT after a LONG
  // primes differently, so we force a true reset via the dedicated endpoint if
  // present, else fall back to a divergent tiny prompt to clear the prefix.
  await fetch(`http://127.0.0.1:${PORT}/v1/internal/reset`, { method: "POST" }).catch(() => {});
}

console.log(`ckpt-resume byte-ident → model=${MODEL} port=${PORT}`);
// 1. PRIME with LONG.
const p = await post(LONG, 8);
console.log(`prime(LONG): prompt=${p.prompt} cached=${p.cached}`);
// 2. RESUME: SHORT is a prefix of LONG → divergence → resume.
const rsm = await post(SHORT, 64);
console.log(`resume(SHORT): prompt=${rsm.prompt} cached=${rsm.cached} sig=${sig(rsm.msg)}  (cached>0 ⇒ prefix reused)`);
// 3. Clear, then COLD baseline. A reset endpoint may not exist; in that case we
//    prime with a DIFFERENT long prompt so SHORT diverges at the system prompt
//    (lcp tiny, no checkpoint) → effectively cold. Then post SHORT cold.
await reset();
const primeOther = await post("Completely unrelated short preamble with different words entirely.", 8);
const cold = await post(SHORT, 64);
console.log(`cold(SHORT):   prompt=${cold.prompt} cached=${cold.cached} sig=${sig(cold.msg)}`);
console.log("");
console.log(`resume_sig=${sig(rsm.msg)}`);
console.log(`cold_sig  =${sig(cold.msg)}`);
console.log(sig(rsm.msg) === sig(cold.msg) ? "PASS: byte-identical (resume == cold)" : "FAIL: resume diverged from cold");
