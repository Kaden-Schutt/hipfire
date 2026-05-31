#!/usr/bin/env bun
// Perf A/B for the checkpoint-taking overhead. Sends a fixed ~8K-token prompt
// (crosses several 2048 checkpoint intervals so checkpoints ARE captured) and
// reports median prefill/decode tok/s + ttft over N trials. Run once against a
// resume-ON daemon and once against a resume-OFF daemon (HIPFIRE_CACHE_CKPT_RESUME=0)
// and compare — checkpoint capture should be in the noise.
const PORT = parseInt(process.argv[2] || "11435", 10);
const MODEL = process.argv[3] || "qwen3.6-27b.mq4";
const TRIALS = parseInt(process.argv[4] || "3", 10);
let doc = "Here is reference material.\n";
for (let n = 0; n < 5; n++) { for (let i = 0; i < 80; i++) doc += `fn op_${n}_${i}(x: i64) -> i64 { return x * ${n} + ${i}; } // step ${i}\n`; }
const PROMPT = doc + "\nSummarize the above in one sentence.";

async function once(nonce: number) {
  const body = { model: MODEL, messages: [{ role: "system", content: `perf ${nonce}` }, { role: "user", content: PROMPT }], max_tokens: 64, temperature: 0, stream: true, stream_options: { include_usage: true }, chat_template_kwargs: { enable_thinking: false } };
  const r = await fetch(`http://127.0.0.1:${PORT}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const reader = r.body!.getReader(); const dec = new TextDecoder(); let buf = ""; let t: any = {};
  while (true) { const { done, value } = await reader.read(); if (done) break; buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n"); buf = lines.pop() ?? "";
    for (const l of lines) { const s = l.trim(); if (!s.startsWith("data:")) continue; const p = s.slice(5).trim(); if (p === "[DONE]") continue; try { const j = JSON.parse(p); if (j.timings) t = j.timings; } catch {} } }
  return t;
}
const med = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
await once(999); // warm
const pre: number[] = [], dec: number[] = [], ttft: number[] = [];
for (let i = 0; i < TRIALS; i++) { const t = await once(i); pre.push(t.prefill_tok_s || 0); dec.push(t.decode_tok_s || 0); ttft.push(t.ttft_ms || 0); }
console.log(`prefill_tok_s=${med(pre).toFixed(1)} decode_tok_s=${med(dec).toFixed(1)} ttft_ms=${med(ttft).toFixed(0)}  (n=${TRIALS}, ~${doc.length} chars)`);
