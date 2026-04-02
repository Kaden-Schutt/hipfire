#!/usr/bin/env bun
// hipfire CLI — ollama-style UX for AMD GPU inference
// Usage:
//   hipfire pull qwen3.5:9b          → download model
//   hipfire run qwen3.5:9b [prompt]  → generate (auto-pulls if needed)
//   hipfire serve                     → start daemon + HTTP server
//   hipfire list                      → show local + available models

import { spawn } from "bun";
import { existsSync, readdirSync, statSync, unlinkSync, mkdirSync } from "fs";
import { join, resolve, basename } from "path";
import { homedir } from "os";

const HIPFIRE_DIR = join(homedir(), ".hipfire");
const MODELS_DIR = join(HIPFIRE_DIR, "models");
const DEFAULT_PORT = 11435;
const TEMP_CORRECTION = 0.82;

mkdirSync(MODELS_DIR, { recursive: true });

// ─── Model Registry ─────────────────────────────────────
// Maps "name:tag" → { repo, file, size_gb, min_vram_gb }
// Default tag (no quant suffix) = HFQ4

const HF_BASE = "https://huggingface.co";

// Per-model HuggingFace repos: schuttdev/hipfire-{family}-{size}
function hfRepo(family: string, size: string) { return `schuttdev/hipfire-${family}-${size}`; }

interface ModelEntry {
  repo: string;
  file: string;
  size_gb: number;
  min_vram_gb: number;
  desc: string;
}

const REGISTRY: Record<string, ModelEntry> = {
  // Qwen3.5 HFQ4 (default)
  "qwen3.5:0.8b":  { repo: hfRepo("qwen3.5","0.8b"), file: "qwen3.5-0.8b.q4.hfq",  size_gb: 0.5,  min_vram_gb: 1,  desc: "222 tok/s, tiny & fast" },
  "qwen3.5:2b":    { repo: hfRepo("qwen3.5","2b"),   file: "qwen3.5-2b.q4.hfq",    size_gb: 1.2,  min_vram_gb: 2,  desc: "141 tok/s" },
  "qwen3.5:4b":    { repo: hfRepo("qwen3.5","4b"),   file: "qwen3.5-4b.q4.hfq",    size_gb: 2.1,  min_vram_gb: 4,  desc: "63 tok/s, best balance" },
  "qwen3.5:9b":    { repo: hfRepo("qwen3.5","9b"),   file: "qwen3.5-9b.q4.hfq",    size_gb: 4.5,  min_vram_gb: 6,  desc: "45 tok/s, best quality 8GB" },
  "qwen3.5:27b":   { repo: hfRepo("qwen3.5","27b"),  file: "qwen3.5-27b.q4.hfq",   size_gb: 14.3, min_vram_gb: 16, desc: "best quality, needs 16GB+" },

  // Qwen3.5 HFQ6
  "qwen3.5:0.8b-hfq6": { repo: hfRepo("qwen3.5","0.8b"), file: "qwen3.5-0.8b.hfq6.hfq", size_gb: 0.6,  min_vram_gb: 1,  desc: "210 tok/s, higher quality" },
  "qwen3.5:2b-hfq6":   { repo: hfRepo("qwen3.5","2b"),   file: "qwen3.5-2b.hfq6.hfq",   size_gb: 1.6,  min_vram_gb: 3,  desc: "127 tok/s" },
  "qwen3.5:4b-hfq6":   { repo: hfRepo("qwen3.5","4b"),   file: "qwen3.5-4b.hfq6.hfq",   size_gb: 3.3,  min_vram_gb: 5,  desc: "53 tok/s" },
  "qwen3.5:9b-hfq6":   { repo: hfRepo("qwen3.5","9b"),   file: "qwen3.5-9b.hfq6.hfq",   size_gb: 6.8,  min_vram_gb: 8,  desc: "37 tok/s, near-FP16" },
  "qwen3.5:27b-hfq6":  { repo: hfRepo("qwen3.5","27b"),  file: "qwen3.5-27b.hfq6.hfq",  size_gb: 21.4, min_vram_gb: 24, desc: "needs 24GB (7900 XTX)" },

  // Qwen3 HFQ4
  "qwen3:0.6b":    { repo: hfRepo("qwen3","0.6b"),   file: "qwen3-0.6b-hfq4.hfq",  size_gb: 0.4,  min_vram_gb: 1,  desc: "standard attention" },
  "qwen3:8b":      { repo: hfRepo("qwen3","8b"),     file: "qwen3-8b.q4.hfq",       size_gb: 4.1,  min_vram_gb: 6,  desc: "59.9 tok/s, standard attention" },
};

// Aliases
const ALIASES: Record<string, string> = {
  "qwen3.5": "qwen3.5:4b",
  "qwen3.5:latest": "qwen3.5:9b",
  "qwen3.5:small": "qwen3.5:0.8b",
  "qwen3": "qwen3:8b",
  "qwen3.5:large": "qwen3.5:27b",
};

function resolveModelTag(input: string): string {
  // Direct registry match
  if (REGISTRY[input]) return input;
  // Alias
  if (ALIASES[input]) return ALIASES[input];
  // Try adding "qwen3.5:" prefix
  if (REGISTRY[`qwen3.5:${input}`]) return `qwen3.5:${input}`;
  return input;
}

function downloadUrl(entry: ModelEntry): string {
  return `${HF_BASE}/${entry.repo}/resolve/main/${entry.file}`;
}

// ─── Daemon IPC ─────────────────────────────────────────

class Engine {
  private proc: ReturnType<typeof spawn> | null = null;
  private lines: string[] = [];
  private buffer = "";

  async start() {
    const bins = [
      resolve(__dirname, "../target/release/examples/daemon"),
      join(HIPFIRE_DIR, "bin", "daemon"),
    ];
    const bin = bins.find(p => existsSync(p));
    if (!bin) throw new Error("daemon not found. cargo build --release --features deltanet --example daemon -p engine");

    this.proc = spawn([bin], { stdin: "pipe", stdout: "pipe", stderr: "inherit" });
  }

  async send(msg: object) {
    if (!this.proc?.stdin) throw new Error("not running");
    this.proc.stdin.write(JSON.stringify(msg) + "\n");
    await this.proc.stdin.flush();
  }

  async recv(): Promise<any> {
    if (!this.proc?.stdout) throw new Error("not running");
    const reader = this.proc.stdout.getReader();
    while (true) {
      if (this.lines.length > 0) {
        reader.releaseLock();
        return JSON.parse(this.lines.shift()!);
      }
      const { value, done } = await reader.read();
      if (done) throw new Error("daemon closed");
      this.buffer += new TextDecoder().decode(value);
      const parts = this.buffer.split("\n");
      this.buffer = parts.pop() || "";
      this.lines.push(...parts.filter(l => l.trim()));
      reader.releaseLock();
    }
  }

  async *generate(msg: object): AsyncGenerator<any> {
    await this.send(msg);
    while (true) {
      const r = await this.recv();
      yield r;
      if (r.type === "done" || r.type === "error") break;
    }
  }

  async stop() {
    try { await this.send({ type: "unload" }); } catch {}
    this.proc?.kill();
  }
}

// ─── Pull (Download) ────────────────────────────────────

async function pull(tag: string): Promise<string> {
  const resolved = resolveModelTag(tag);
  const entry = REGISTRY[resolved];
  if (!entry) {
    console.error(`Unknown model: ${tag}`);
    console.error(`Available: ${Object.keys(REGISTRY).join(", ")}`);
    process.exit(1);
  }

  const dest = join(MODELS_DIR, entry.file);
  if (existsSync(dest)) {
    const sz = (statSync(dest).size / 1e9).toFixed(1);
    console.error(`Already downloaded: ${entry.file} (${sz}GB)`);
    return dest;
  }

  const url = downloadUrl(entry);
  console.error(`Pulling ${resolved} (${entry.size_gb}GB)...`);
  console.error(`  ${url}`);

  const res = await fetch(url);
  if (!res.ok) {
    console.error(`Download failed: ${res.status} ${res.statusText}`);
    console.error(`URL: ${url}`);
    process.exit(1);
  }

  const total = parseInt(res.headers.get("content-length") || "0");
  const tmpDest = dest + ".tmp";
  const writer = Bun.file(tmpDest).writer();
  let downloaded = 0;
  let lastPrint = 0;

  for await (const chunk of res.body as AsyncIterable<Uint8Array>) {
    writer.write(chunk);
    downloaded += chunk.length;
    const now = Date.now();
    if (now - lastPrint > 500 || downloaded === total) {
      const pct = total > 0 ? ((downloaded / total) * 100).toFixed(1) : "?";
      const mb = (downloaded / 1e6).toFixed(0);
      const totalMb = total > 0 ? (total / 1e6).toFixed(0) : "?";
      process.stderr.write(`\r  ${mb}/${totalMb} MB (${pct}%)`);
      lastPrint = now;
    }
  }
  await writer.end();
  console.error("");

  // Rename tmp → final (atomic-ish)
  const { renameSync } = await import("fs");
  renameSync(tmpDest, dest);

  const sz = (statSync(dest).size / 1e9).toFixed(1);
  console.error(`  Saved: ${dest} (${sz}GB)`);
  return dest;
}

// ─── Commands ───────────────────────────────────────────

async function run(model: string, prompt: string, image?: string, temp = 0.3, maxTokens = 512) {
  let path = findModel(model);

  // Auto-pull if model tag is recognized but not downloaded
  if (!path) {
    const resolved = resolveModelTag(model);
    if (REGISTRY[resolved]) {
      console.error(`Model not found locally. Pulling ${resolved}...`);
      path = await pull(model);
    } else {
      console.error(`Model not found: ${model}`);
      console.error(`Run: hipfire pull <model>  (e.g. hipfire pull qwen3.5:9b)`);
      console.error(`See: hipfire list --remote`);
      process.exit(1);
    }
  }

  if (image && !existsSync(image)) { console.error(`Image not found: ${image}`); process.exit(1); }

  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();
  await e.send({ type: "load", model: path });
  const loaded = await e.recv();
  if (loaded.type === "error") { console.error(loaded.message); process.exit(1); }
  const vlTag = loaded.vl ? " VL" : "";
  console.error(`[${loaded.arch}${vlTag}] ${loaded.dim}d ${loaded.layers}L ${loaded.vocab} vocab`);

  if (image && !loaded.vl) {
    console.error(`WARNING: --image passed but model does not have a vision encoder. Ignoring image.`);
    image = undefined;
  }

  const genMsg: any = {
    type: "generate", id: "run", prompt,
    temperature: temp * TEMP_CORRECTION, max_tokens: maxTokens,
  };
  if (image) {
    genMsg.image = resolve(image);
    console.error(`[VL: ${image}]`);
  }

  for await (const msg of e.generate(genMsg)) {
    if (msg.type === "token") process.stdout.write(msg.text);
    else if (msg.type === "done") console.error(`\n[${msg.tokens} tok, ${msg.tok_s} tok/s]`);
  }
  await e.stop();
}

async function serve(port: number) {
  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();
  let current: string | null = null;

  let busy = false;
  const queue: Array<{ resolve: () => void }> = [];
  async function acquireLock() {
    if (!busy) { busy = true; return; }
    await new Promise<void>(resolve => queue.push({ resolve }));
  }
  function releaseLock() {
    const next = queue.shift();
    if (next) next.resolve();
    else busy = false;
  }

  console.error(`[hipfire] http://localhost:${port}/v1/chat/completions`);

  Bun.serve({
    port,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === "/health") return Response.json({ status: "ok", model: current });
      if (url.pathname === "/v1/models") return Response.json({ data: listLocal().map(m => ({ id: m.name })) });

      if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
        await acquireLock();
        try {
        const body = await req.json();
        const prompt = (body.messages || []).map((m: any) => m.content).join("\n");
        const path = findModel(body.model || "default");
        if (!path) { releaseLock(); return Response.json({ error: "model not found" }, { status: 404 }); }

        if (current !== path) {
          if (current) { await e.send({ type: "unload" }); await e.recv(); }
          await e.send({ type: "load", model: path }); await e.recv();
          current = path;
        }

        if (body.stream) {
          const enc = new TextEncoder();
          return new Response(new ReadableStream({
            async start(ctrl) {
              for await (const msg of e.generate({
                type: "generate", id: "api", prompt,
                temperature: (body.temperature ?? 0.3) * TEMP_CORRECTION,
                max_tokens: body.max_tokens ?? 512,
              })) {
                if (msg.type === "token") {
                  ctrl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: msg.text } }] })}\n\n`));
                } else if (msg.type === "done") {
                  ctrl.enqueue(enc.encode(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`));
                  ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
                  ctrl.close();
                }
              }
            }
          }), { headers: { "Content-Type": "text/event-stream" } });
        }

        let content = "";
        for await (const msg of e.generate({
          type: "generate", id: "api", prompt,
          temperature: (body.temperature ?? 0.3) * TEMP_CORRECTION,
          max_tokens: body.max_tokens ?? 512,
        })) { if (msg.type === "token") content += msg.text; }
        releaseLock();
        return Response.json({ choices: [{ message: { role: "assistant", content } }] });
        } finally { releaseLock(); }
      }
      return Response.json({ error: "not found" }, { status: 404 });
    }
  });
}

// ─── Helpers ────────────────────────────────────────────

function findModel(name: string): string | null {
  // Direct file path
  if (existsSync(name)) return resolve(name);

  // Resolve tag → filename
  const resolved = resolveModelTag(name);
  const entry = REGISTRY[resolved];
  if (entry) {
    const p = join(MODELS_DIR, entry.file);
    if (existsSync(p)) return p;
  }

  // Fuzzy search local dirs
  const dirs = [resolve(__dirname, "../models"), MODELS_DIR];
  for (const dir of dirs) {
    try { for (const f of readdirSync(dir)) {
      if (f.endsWith(".hfq") && (f.includes(name) || f.includes(name.replace(":", "-")))) return join(dir, f);
    }} catch {}
  }
  return null;
}

function listLocal() {
  const models: { name: string; tag: string; size: string }[] = [];
  const seen = new Set<string>();
  for (const dir of [MODELS_DIR, resolve(__dirname, "../models")]) {
    try { for (const f of readdirSync(dir)) {
      if (f.endsWith(".hfq") && !seen.has(f)) {
        seen.add(f);
        const sz = (statSync(join(dir, f)).size / 1e9).toFixed(1);
        // Find matching registry tag
        const tag = Object.entries(REGISTRY).find(([_, e]) => e.file === f)?.[0] || "";
        models.push({ name: f, tag, size: `${sz}GB` });
      }
    }} catch {}
  }
  return models;
}

// ─── Main ───────────────────────────────────────────────

const [cmd, ...rest] = process.argv.slice(2);
switch (cmd) {
  case "serve": await serve(parseInt(rest[0]) || DEFAULT_PORT); break;
  case "run": {
    const model = rest[0];
    if (!model) { console.error("Usage: hipfire run <model> [--image img.png] [prompt]\n\nExamples:\n  hipfire run qwen3.5:9b \"Hello\"\n  hipfire run qwen3.5:4b --image photo.png \"Describe this\""); process.exit(1); }
    const imgIdx = rest.indexOf("--image");
    const image = imgIdx >= 0 ? rest[imgIdx + 1] : undefined;
    const filtered = rest.slice(1).filter((_, i) => i !== imgIdx - 1 && i !== imgIdx);
    const prompt = filtered.join(" ") || (image ? "Describe this image." : "Hello");
    await run(model, prompt, image);
    break;
  }
  case "pull": {
    const tag = rest[0];
    if (!tag) { console.error("Usage: hipfire pull <model>\n\nExamples:\n  hipfire pull qwen3.5:9b\n  hipfire pull qwen3.5:4b-hfq6\n  hipfire pull qwen3.5:27b\n\nAvailable:\n" + Object.entries(REGISTRY).map(([t, e]) => `  ${t.padEnd(22)} ${e.size_gb.toString().padStart(5)}GB  ${e.desc}`).join("\n")); process.exit(1); }
    await pull(tag);
    break;
  }
  case "list": {
    const showRemote = rest.includes("--remote") || rest.includes("-r");
    const local = listLocal();
    if (local.length > 0) {
      console.log("Local models:\n");
      for (const m of local) {
        const tag = m.tag ? ` (${m.tag})` : "";
        console.log(`  ${m.name.padEnd(35)} ${m.size.padStart(6)}${tag}`);
      }
    } else {
      console.log("No local models. Pull one:\n  hipfire pull qwen3.5:9b\n");
    }
    if (showRemote || local.length === 0) {
      console.log("\nAvailable models:\n");
      const localFiles = new Set(local.map(m => m.name));
      for (const [tag, entry] of Object.entries(REGISTRY)) {
        const status = localFiles.has(entry.file) ? " [downloaded]" : "";
        console.log(`  ${tag.padEnd(22)} ${entry.size_gb.toString().padStart(5)}GB  ${entry.desc}${status}`);
      }
      console.log("\nPull: hipfire pull <model>  (e.g. hipfire pull qwen3.5:9b)");
    }
    break;
  }
  case "rm": {
    const tag = rest[0] || "";
    const resolved = resolveModelTag(tag);
    const entry = REGISTRY[resolved];
    const path = entry ? join(MODELS_DIR, entry.file) : findModel(tag);
    if (path && existsSync(path)) {
      unlinkSync(path);
      console.log(`Removed ${path}`);
    } else {
      console.error(`Model not found: ${tag}`);
    }
    break;
  }
  default:
    console.log(`hipfire — LLM inference for AMD GPUs

  pull <model>          Download model from HuggingFace
  run <model> [prompt]  Generate text (auto-pulls if needed)
  serve [port]          Start OpenAI-compatible server (default: ${DEFAULT_PORT})
  list [-r]             Show local models (-r: show available too)
  rm <model>            Delete model

Models:
  hipfire pull qwen3.5:9b            # 4.5GB, best quality for 8GB cards
  hipfire pull qwen3.5:4b            # 2.1GB, best speed/quality balance
  hipfire pull qwen3.5:27b           # 14.3GB, needs 16GB+ VRAM
  hipfire pull qwen3.5:9b-hfq6      # 6.8GB, higher quality (6-bit)

Quick start:
  hipfire pull qwen3.5:4b
  hipfire run qwen3.5:4b "What is the capital of France?"
  hipfire serve`);
}
