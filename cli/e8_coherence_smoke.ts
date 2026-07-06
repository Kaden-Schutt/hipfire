// mfp4-E8 batched-prefill coherence smoke through the daemon.
// Gate ON (HIPFIRE_E8_GFX12=1 from the parent env) forces the batched prefill
// path under test. Greedy decode; eyeball for fluency / on-topic / no loops.
// Run: HIPFIRE_E8_GFX12=1 HIPFIRE_DEBUG_BATCH=1 bun cli/e8_coherence_smoke.ts
import { spawn } from "bun";
import { homedir } from "os";

const MODEL = `${homedir()}/.hipfire/models/q36a3b.mfp4e8-gptq-v2.hfq`;
const DAEMON = `${import.meta.dir}/../target/release/examples/daemon`;

const env: any = { ...process.env, HIPFIRE_KV_MODE: "q8" };
delete env.HIPFIRE_GRAPH;
const proc = spawn({ cmd: [DAEMON], stdin: "pipe", stdout: "pipe", stderr: "inherit", env });
const stdin = proc.stdin;
const stdout = proc.stdout.getReader();
const decoder = new TextDecoder();
const encoder = new TextEncoder();
let buf = "";
async function send(o: any) { await stdin.write(encoder.encode(JSON.stringify(o) + "\n")); }
async function readUntil(p: (m: any) => boolean): Promise<any[]> {
  const out: any[] = [];
  while (true) {
    if (buf.includes("\n")) {
      const i = buf.indexOf("\n"); const line = buf.slice(0, i); buf = buf.slice(i + 1);
      if (line.trim()) { const m = JSON.parse(line); out.push(m); if (p(m)) return out; }
    } else {
      const { value, done } = await stdout.read();
      if (done) return out;
      buf += decoder.decode(value);
    }
  }
}

await send({ type: "load", model: MODEL, params: { max_seq: 4096 } });
const loaded = await readUntil((m) => m.type === "loaded" || m.type === "error");
if (loaded.some((m) => m.type === "error")) { console.log("LOAD ERROR", loaded); process.exit(1); }

// A longer paragraph → a multi-hundred-token prefill that actually exercises
// the batched path at scale, plus a bare factual prompt (catches lobotomy that
// rich prompts mask).
const CTX =
  "The transformer architecture, introduced in 2017, replaced recurrent " +
  "networks for most sequence tasks. Its core idea is self-attention: every " +
  "token computes a weighted sum over all other tokens, where the weights come " +
  "from learned query and key projections. This lets the model relate distant " +
  "tokens in a single step, and because the attention over all positions can be " +
  "computed in parallel, transformers train far faster on modern accelerators " +
  "than sequential RNNs. Stacks of attention plus feed-forward layers, with " +
  "residual connections and normalization, make up the encoder and decoder. ";

const prompts: [string, string][] = [
  ["factual", "The capital of France is"],
  ["reasoning", "Explain in two sentences why the sky appears blue during the day."],
  ["long-ctx", CTX + "\n\nIn one sentence, what is the core idea of the transformer?"],
];

for (const [id, prompt] of prompts) {
  await send({ type: "generate", id, prompt, temperature: 0.0, max_tokens: 140, repeat_penalty: 1.0 });
  let text = "";
  const msgs = await readUntil((m) => m.type === "done" && m.id === id);
  for (const m of msgs) if (m.type === "token" && m.id === id) text += m.text;
  console.log(`\n===== [${id}] prompt(${prompt.length} chars) =====`);
  console.log(text.trim());
}

await send({ type: "unload" });
await readUntil((m) => m.type === "unloaded");
stdin.end();
await proc.exited;
