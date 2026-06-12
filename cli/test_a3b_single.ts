// Single-turn smoke test through the daemon. Goal: compare output to
// the standalone a3b_smoke_forward smoke test. If they differ, the
// daemon's prefill / sample path is doing something wrong.
import { spawn } from "bun";
import { homedir } from "os";

const MODEL = `${homedir()}/.hipfire/models/qwen3.5-35b-a3b-mq4.hfq`;
const DAEMON = `${import.meta.dir}/../target/release/hipfire-daemon`;

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
await readUntil((m) => m.type === "loaded" || m.type === "error");

// Greedy decode (temp=0, RP=1.0) should match the smoke test exactly.
await send({
  type: "generate", id: "single", prompt: "What is 2 + 2?",
  temperature: 0.0, max_tokens: 50, repeat_penalty: 1.0,
});
let text = "";
console.log("Daemon output:");
const msgs = await readUntil((m) => m.type === "done" && m.id === "single");
for (const m of msgs) {
  if (m.type === "token" && m.id === "single") {
    text += m.text;
    process.stdout.write(m.text);
  }
}
console.log("\n\nFull text:", JSON.stringify(text));

await send({ type: "unload" });
await readUntil((m) => m.type === "unloaded");
stdin.end();
await proc.exited;
