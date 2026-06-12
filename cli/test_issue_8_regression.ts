// Regression check against the prompts from issue #8 (v0.1.2-alpha
// report). Runs the deterministic test prompts and prints outputs so
// we can compare against the historical failure modes.
//
// Usage: bun cli/test_issue_8_regression.ts [model_path]
import { spawn } from "bun";
import { homedir } from "os";

const MODEL = process.argv[2] || `${homedir()}/.hipfire/models/qwen3.5-35b-a3b-mq4.hfq`;
const DAEMON = `${import.meta.dir}/../target/release/examples/daemon`;

console.log(`Regression test against issue #8 prompts.\nModel: ${MODEL}\n`);

async function runPrompts(label: string, prompts: { name: string; prompt: string; max_tokens: number }[]) {
  const proc = spawn({
    cmd: [DAEMON], stdin: "pipe", stdout: "pipe", stderr: "inherit",
    env: { ...process.env },
  });
  const stdin = proc.stdin;
  const stdout = proc.stdout.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buf = "";

  const send = (o: any) => stdin.write(encoder.encode(JSON.stringify(o) + "\n"));
  const readUntil = async (p: (m: any) => boolean): Promise<any[]> => {
    const out: any[] = [];
    while (true) {
      if (buf.includes("\n")) {
        const i = buf.indexOf("\n");
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (line.trim()) { const m = JSON.parse(line); out.push(m); if (p(m)) return out; }
      } else {
        const { value, done } = await stdout.read();
        if (done) return out;
        buf += decoder.decode(value);
      }
    }
  };

  await send({ type: "load", model: MODEL, params: { max_seq: 2048 } });
  const loaded = (await readUntil((m) => m.type === "loaded" || m.type === "error"))[0];
  if (loaded.type === "error") {
    console.error(`[${label}] load failed: ${loaded.message}`);
    stdin.end(); return;
  }
  console.log(`=== ${label} (arch=${loaded.arch}) ===`);

  for (const p of prompts) {
    await send({
      type: "generate", id: p.name, prompt: p.prompt,
      temperature: 0.0, max_tokens: p.max_tokens, repeat_penalty: 1.0,
    });
    let text = "";
    const msgs = await readUntil((m) => m.type === "done" && m.id === p.name);
    for (const m of msgs) {
      if (m.type === "token" && m.id === p.name) text += m.text;
    }
    // Strip the chat <think></think> block (if present) to focus on output
    const contentOnly = text.replace(/<think>[\s\S]*?<\/think>\s*/, "").trim();
    console.log(`\n── ${p.name}\n  prompt: ${JSON.stringify(p.prompt)}\n  output: ${JSON.stringify(contentOnly)}`);
  }

  await send({ type: "unload" });
  await readUntil((m) => m.type === "unloaded");
  stdin.end();
  await proc.exited;
}

const prompts = [
  {
    name: "exact-copy",
    prompt: "Repeat exactly: alpha beta\\ngamma\\tdelta",
    max_tokens: 60,
  },
  {
    name: "primes",
    prompt: "Return the first 5 prime numbers, comma-separated. Just the numbers, no explanation.",
    max_tokens: 40,
  },
  {
    name: "numbers-1-10",
    prompt: "Return numbers 1 to 10, one per line.",
    max_tokens: 60,
  },
  {
    name: "json",
    prompt: 'Return exactly this JSON: {"a":1,"b":2}',
    max_tokens: 40,
  },
  {
    name: "single-token-100",
    prompt: "Return exactly: 100",
    max_tokens: 20,
  },
  {
    name: "single-token-42",
    prompt: "Return exactly: 42",
    max_tokens: 20,
  },
  {
    name: "capital-france",
    prompt: "What is the capital of France? Answer with just the city name.",
    max_tokens: 20,
  },
];

// Determinism check: run "primes" 3 times, compare.
const detPrompt = {
  name: "primes-det",
  prompt: "Return the first 5 prime numbers, comma-separated. Just the numbers, no explanation.",
  max_tokens: 40,
};

await runPrompts("A3B", prompts);

console.log(`\n\n=== DETERMINISM (3× primes) ===`);
for (let i = 0; i < 3; i++) {
  const proc = spawn({
    cmd: [DAEMON], stdin: "pipe", stdout: "pipe", stderr: "inherit",
    env: { ...process.env },
  });
  const stdin = proc.stdin;
  const stdout = proc.stdout.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buf = "";
  const send = (o: any) => stdin.write(encoder.encode(JSON.stringify(o) + "\n"));
  const readUntil = async (p: (m: any) => boolean): Promise<any[]> => {
    const out: any[] = [];
    while (true) {
      if (buf.includes("\n")) {
        const i = buf.indexOf("\n");
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (line.trim()) { const m = JSON.parse(line); out.push(m); if (p(m)) return out; }
      } else {
        const { value, done } = await stdout.read();
        if (done) return out;
        buf += decoder.decode(value);
      }
    }
  };
  await send({ type: "load", model: MODEL, params: { max_seq: 1024 } });
  await readUntil((m) => m.type === "loaded" || m.type === "error");
  await send({
    type: "generate", id: `det-${i}`, prompt: detPrompt.prompt,
    temperature: 0.0, max_tokens: detPrompt.max_tokens, repeat_penalty: 1.0,
  });
  let text = "";
  const msgs = await readUntil((m) => m.type === "done" && m.id === `det-${i}`);
  for (const m of msgs) if (m.type === "token" && m.id === `det-${i}`) text += m.text;
  const c = text.replace(/<think>[\s\S]*?<\/think>\s*/, "").trim();
  console.log(`  run ${i + 1}: ${JSON.stringify(c)}`);
  await send({ type: "unload" });
  await readUntil((m) => m.type === "unloaded");
  stdin.end();
  await proc.exited;
}
