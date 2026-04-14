// Side-by-side bench of hipfire (asym3 MQ4) vs ollama (default Q4_K_M)
// on matched prompt + generation length. Reports prefill + decode tok/s
// for each (model, backend) pair using the backend's own reported
// timings.
//
// hipfire numbers come from `hipfire bench <tag>` (prefill pp128/pp512,
// decode tg128, TTFT). Ollama numbers come from /api/generate with
// temperature=0 and num_predict=128 — the response JSON carries
// prompt_eval_count, prompt_eval_duration, eval_count, eval_duration.
//
// Usage:
//   bun cli/bench_vs_ollama.ts                   # all common models
//   bun cli/bench_vs_ollama.ts qwen3.5:9b        # single model
import { spawn } from "bun";

type OllamaBench = { prefill_tok_s: number; decode_tok_s: number; prefill_tok: number; decode_tok: number; };

async function benchOllama(model: string, prompt: string, num_predict = 128): Promise<OllamaBench | null> {
  try {
    // Warm-up so the model is resident + kernels JIT'd.
    await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      body: JSON.stringify({ model, prompt: "hi", stream: false,
        options: { num_predict: 4, temperature: 0 } }),
    });
    const res = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      body: JSON.stringify({ model, prompt, stream: false,
        options: { num_predict, temperature: 0 } }),
    });
    const d: any = await res.json();
    if (!d.eval_count) return null;
    return {
      prefill_tok: d.prompt_eval_count,
      prefill_tok_s: d.prompt_eval_count / (d.prompt_eval_duration / 1e9),
      decode_tok: d.eval_count,
      decode_tok_s: d.eval_count / (d.eval_duration / 1e9),
    };
  } catch (e) {
    console.error(`ollama bench failed for ${model}:`, e);
    return null;
  }
}

type HipBench = { pp128: number; pp512: number; decode_tok_s: number; ttft_ms: number; };

async function benchHipfire(tag: string): Promise<HipBench | null> {
  // Call hipfire's existing bench command and parse its table output.
  const proc = spawn({
    cmd: ["bun", `${import.meta.dir}/index.ts`, "bench", tag],
    stdout: "pipe", stderr: "inherit",
  });
  const out = await new Response(proc.stdout).text();
  await proc.exited;
  // Parse the prefill pp128/pp512 row and the Decode mean row.
  const pp = out.match(/pp128\s+([\d.]+)/);
  const pp2 = out.match(/pp512\s+([\d.]+)/);
  const dec = out.match(/Decode\s+tok\/s\s+([\d.]+)/);
  const ttft = out.match(/TTFT\s+ms\s+([\d.]+)/);
  if (!pp || !dec) { console.error("failed to parse hipfire bench"); return null; }
  return {
    pp128: parseFloat(pp[1]),
    pp512: pp2 ? parseFloat(pp2[1]) : NaN,
    decode_tok_s: parseFloat(dec[1]),
    ttft_ms: ttft ? parseFloat(ttft[1]) : NaN,
  };
}

// Match hipfire bench's default 20-token prompt.
const PROMPT = "Explain the theory of general relativity in simple terms.";

const PAIRS: { name: string; hipfire: string; ollama: string }[] = [
  { name: "0.8b",    hipfire: "qwen3.5:0.8b",    ollama: "qwen3.5:0.8b" },
  { name: "4b",      hipfire: "qwen3.5:4b",      ollama: "qwen3.5:4b" },
  { name: "9b",      hipfire: "qwen3.5:9b",      ollama: "qwen3.5:9b" },
  { name: "35b-a3b", hipfire: "qwen3.5:35b-a3b", ollama: "qwen3.5:35b-a3b" }, // may not exist yet
];

const args = process.argv.slice(2);
const filter = args.length ? args[0] : null;

const rows: any[] = [];
for (const p of PAIRS) {
  if (filter && p.hipfire !== filter) continue;
  console.log(`\n=== ${p.name} ===`);

  console.log(`→ hipfire asym3 ...`);
  const h = await benchHipfire(p.hipfire);
  if (h) console.log(`  pp128=${h.pp128.toFixed(0)} pp512=${h.pp512.toFixed(0)} decode=${h.decode_tok_s.toFixed(1)} ttft=${h.ttft_ms.toFixed(1)}ms`);

  console.log(`→ ollama Q4_K_M ...`);
  const o = await benchOllama(p.ollama, PROMPT);
  if (o) console.log(`  prefill=${o.prefill_tok_s.toFixed(0)} (${o.prefill_tok} tok) decode=${o.decode_tok_s.toFixed(1)} (${o.decode_tok} tok)`);
  else console.log(`  (ollama model '${p.ollama}' not available — skip)`);

  rows.push({ model: p.name, hipfire: h, ollama: o });
}

// Summary table.
console.log(`\n\n╔═════════════ hipfire asym3 vs ollama Q4_K_M ═════════════╗`);
console.log(`║                                                             ║`);
const fmt = (n: number | undefined) => n === undefined || Number.isNaN(n) ? "   —  " : n.toFixed(0).padStart(6);
const fmtF = (n: number | undefined) => n === undefined || Number.isNaN(n) ? "  —   " : n.toFixed(1).padStart(6);
console.log(`Model    | hf pp128 | hf pp512 | hf dec | oll pp | oll dec | dec Δ`);
console.log(`---------|----------|----------|--------|--------|---------|-------`);
for (const r of rows) {
  const h = r.hipfire || {};
  const o = r.ollama || {};
  const deltaDec = h.decode_tok_s && o.decode_tok_s
    ? `${(((h.decode_tok_s / o.decode_tok_s) - 1) * 100).toFixed(0).padStart(4)}%`
    : "   — ";
  console.log(`${r.model.padEnd(8)} | ${fmt(h.pp128)}   | ${fmt(h.pp512)}   | ${fmtF(h.decode_tok_s)} | ${fmt(o.prefill_tok_s)} | ${fmtF(o.decode_tok_s)}  | ${deltaDec}`);
}
