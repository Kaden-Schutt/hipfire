// Multi-turn smoke test for the A3B daemon path.
// Spawns the daemon, loads the local A3B mq4, and runs a 3-turn chat to
// verify the per-turn KV-cache + DeltaNet state continuity. Each `generate`
// call only prefills the new turn's tokens; the daemon should remember the
// earlier user/assistant exchange.
//
// Usage:
//   bun cli/test_a3b_multiturn.ts
import { spawn } from "bun";
import { homedir } from "os";

const MODEL = `${homedir()}/.hipfire/models/qwen3.5-35b-a3b-mq4.hfq`;
const DAEMON = `${import.meta.dir}/../target/release/hipfire-daemon`;

const proc = spawn({
  cmd: [DAEMON],
  stdin: "pipe",
  stdout: "pipe",
  stderr: "inherit",
  env: { ...process.env },
});

const stdin = proc.stdin;
const stdout = proc.stdout.getReader();
const decoder = new TextDecoder();
const encoder = new TextEncoder();
let buf = "";

async function send(obj: any): Promise<void> {
  await stdin.write(encoder.encode(JSON.stringify(obj) + "\n"));
}

async function readUntil(predicate: (msg: any) => boolean): Promise<any[]> {
  const out: any[] = [];
  while (true) {
    if (buf.includes("\n")) {
      const idx = buf.indexOf("\n");
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 1);
      if (line.trim()) {
        const msg = JSON.parse(line);
        out.push(msg);
        if (predicate(msg)) return out;
      }
    } else {
      const { value, done } = await stdout.read();
      if (done) return out;
      buf += decoder.decode(value);
    }
  }
}

async function runTurn(turn: number, prompt: string): Promise<{ text: string; tok_s: number }> {
  console.log(`\n──── Turn ${turn} ────`);
  console.log(`> ${prompt}`);
  const id = `turn-${turn}`;
  await send({
    type: "generate", id, prompt,
    temperature: 0.0, max_tokens: 80, repeat_penalty: 1.0,
  });
  let text = "";
  let tok_s = 0;
  process.stdout.write("< ");
  const msgs = await readUntil((m) => m.type === "done" && m.id === id);
  for (const m of msgs) {
    if (m.type === "token" && m.id === id) {
      text += m.text;
      process.stdout.write(m.text);
    } else if (m.type === "done" && m.id === id) {
      tok_s = m.tok_s ?? 0;
    } else if (m.type === "error") {
      console.error(`\nERROR: ${m.message}`);
      throw new Error(m.message);
    }
  }
  console.log(`\n  [${msgs.filter(m => m.type === "token").length} tokens, ${tok_s.toFixed(1)} tok/s]`);
  return { text, tok_s };
}

console.log(`Loading ${MODEL} ...`);
await send({ type: "load", model: MODEL, params: { max_seq: 4096 } });
const loadResp = (await readUntil((m) => m.type === "loaded" || m.type === "error"))[0];
if (loadResp.type === "error") {
  console.error(`Load failed: ${loadResp.message}`);
  process.exit(1);
}
console.log(`Loaded: arch=${loadResp.arch}, layers=${loadResp.layers}, vocab=${loadResp.vocab}`);

const t1 = await runTurn(1, "What is 2 + 2?");
const t2 = await runTurn(2, "Now multiply that by 5.");
const t3 = await runTurn(3, "And tell me what month it is mentioned in 'Cloudy with a Chance of Meatballs 2'.");

await send({ type: "unload" });
await readUntil((m) => m.type === "unloaded");
stdin.end();
await proc.exited;

console.log(`\n── Summary ──`);
console.log(`  Turn 1: ${t1.tok_s.toFixed(1)} tok/s`);
console.log(`  Turn 2: ${t2.tok_s.toFixed(1)} tok/s`);
console.log(`  Turn 3: ${t3.tok_s.toFixed(1)} tok/s`);
