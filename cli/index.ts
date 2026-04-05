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
  "qwen3.5:0.8b":  { repo: hfRepo("qwen3.5","0.8b"), file: "qwen3.5-0.8b.hf4",     size_gb: 0.5,  min_vram_gb: 1,  desc: "190 tok/s, tiny & fast" },
  "qwen3.5:2b":    { repo: hfRepo("qwen3.5","2b"),   file: "qwen3.5-2b.hf4",       size_gb: 1.2,  min_vram_gb: 2,  desc: "141 tok/s" },
  "qwen3.5:4b":    { repo: hfRepo("qwen3.5","4b"),   file: "qwen3.5-4b.hf4",       size_gb: 2.1,  min_vram_gb: 4,  desc: "61 tok/s, best balance" },
  "qwen3.5:9b":    { repo: hfRepo("qwen3.5","9b"),   file: "qwen3.5-9b.hf4",       size_gb: 4.5,  min_vram_gb: 6,  desc: "43 tok/s, best quality 8GB" },
  "qwen3.5:27b":   { repo: hfRepo("qwen3.5","27b"),  file: "qwen3.5-27b.hf4",      size_gb: 14.3, min_vram_gb: 16, desc: "16GB+, use -hf6 for coding" },

  // Qwen3.5 HFQ6
  "qwen3.5:0.8b-hf6":  { repo: hfRepo("qwen3.5","0.8b"), file: "qwen3.5-0.8b.hf6",     size_gb: 0.6,  min_vram_gb: 1,  desc: "180 tok/s, higher quality" },
  "qwen3.5:2b-hf6":    { repo: hfRepo("qwen3.5","2b"),   file: "qwen3.5-2b.hf6",       size_gb: 1.6,  min_vram_gb: 3,  desc: "127 tok/s" },
  "qwen3.5:4b-hf6":    { repo: hfRepo("qwen3.5","4b"),   file: "qwen3.5-4b.hf6",       size_gb: 3.3,  min_vram_gb: 5,  desc: "53 tok/s" },
  "qwen3.5:9b-hf6":    { repo: hfRepo("qwen3.5","9b"),   file: "qwen3.5-9b.hf6",       size_gb: 6.8,  min_vram_gb: 8,  desc: "34 tok/s, near-FP16" },
  "qwen3.5:27b-hf6":   { repo: hfRepo("qwen3.5","27b"),  file: "qwen3.5-27b.hf6",      size_gb: 21.4, min_vram_gb: 24, desc: "needs 24GB (7900 XTX)" },

  // Qwen3 HFQ4 (original quantizer filenames — see docs/MODELS.md for naming notes)
  "qwen3:0.6b":    { repo: hfRepo("qwen3","0.6b"),   file: "qwen3-0.6b.hf4",          size_gb: 0.4,  min_vram_gb: 1,  desc: "standard attention" },
  "qwen3:8b":      { repo: hfRepo("qwen3","8b"),     file: "qwen3-8b.hf4",            size_gb: 4.1,  min_vram_gb: 6,  desc: "60 tok/s, standard attention" },
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
  // Backward compat: old hfq4/hfq6 tags → hf4/hf6
  const normalized = input.replace(/-hfq(\d)/, "-hf$1").replace(/\.hfq$/, ".hf4");
  // Direct registry match
  if (REGISTRY[normalized]) return normalized;
  // Alias
  if (ALIASES[normalized]) return ALIASES[normalized];
  // Try adding "qwen3.5:" prefix
  if (REGISTRY[`qwen3.5:${normalized}`]) return `qwen3.5:${normalized}`;
  return normalized;
}

function downloadUrl(entry: ModelEntry): string {
  return `${HF_BASE}/${entry.repo}/resolve/main/${entry.file}`;
}

// ─── Daemon IPC ─────────────────────────────────────────

class Engine {
  private proc: ReturnType<typeof spawn> | null = null;
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  private lines: string[] = [];
  private buffer = "";

  async start() {
    const exe = process.platform === "win32" ? ".exe" : "";
    const bins = [
      resolve(__dirname, `../target/release/examples/daemon${exe}`),
      join(HIPFIRE_DIR, "bin", `daemon${exe}`),
    ];
    const bin = bins.find(p => existsSync(p));
    if (!bin) throw new Error("daemon not found. cargo build --release --features deltanet --example daemon -p engine");

    this.proc = spawn([bin], { stdin: "pipe", stdout: "pipe", stderr: "inherit" });
    this.reader = this.proc.stdout!.getReader();
  }

  async send(msg: object) {
    if (!this.proc?.stdin) throw new Error("not running");
    this.proc.stdin.write(JSON.stringify(msg) + "\n");
    await this.proc.stdin.flush();
  }

  async recv(): Promise<any> {
    if (!this.reader) throw new Error("not running");
    while (true) {
      if (this.lines.length > 0) {
        return JSON.parse(this.lines.shift()!);
      }
      const { value, done } = await this.reader.read();
      if (done) throw new Error("daemon closed");
      this.buffer += new TextDecoder().decode(value);
      const parts = this.buffer.split("\n");
      this.buffer = parts.pop() || "";
      this.lines.push(...parts.filter(l => l.trim()));
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
    this.reader?.releaseLock();
    this.reader = null;
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

  // Hint for 27B HFQ4: recommend HFQ6 for complex tasks
  if (resolved === "qwen3.5:27b") {
    console.error(`TIP: For coding/complex tasks, use: hipfire pull qwen3.5:27b-hf6 (needs 24GB VRAM)`);
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

async function run(model: string, prompt: string, image?: string, temp = 0.3, maxTokens = 512, repeatPenalty = 1.3, topP = 0.8) {
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
  await e.send({ type: "load", model: path, turbo: 4 });
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
    repeat_penalty: repeatPenalty, top_p: topP,
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
    idleTimeout: 255, // max allowed — model loading can take 30s+
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
          await e.send({ type: "load", model: path, turbo: 4 }); await e.recv();
          current = path;
        }

        const reqId = `chatcmpl-${Date.now().toString(36)}`;
        const modelName = body.model || "hipfire";
        const genParams = {
          type: "generate", id: "api", prompt,
          temperature: (body.temperature ?? 0.3) * TEMP_CORRECTION,
          max_tokens: body.max_tokens ?? 512,
          repeat_penalty: body.repeat_penalty ?? body.frequency_penalty ? 1.0 + (body.frequency_penalty ?? 0) : 1.3,
          top_p: body.top_p ?? 0.8,
        };

        if (body.stream) {
          const enc = new TextEncoder();
          return new Response(new ReadableStream({
            async start(ctrl) {
              try {
                let tokens = 0;
                for await (const msg of e.generate(genParams)) {
                  if (msg.type === "token") {
                    tokens++;
                    ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                      id: reqId, object: "chat.completion.chunk", created: Math.floor(Date.now()/1000), model: modelName,
                      choices: [{ index: 0, delta: { content: msg.text }, finish_reason: null }]
                    })}\n\n`));
                  } else if (msg.type === "done") {
                    ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                      id: reqId, object: "chat.completion.chunk", created: Math.floor(Date.now()/1000), model: modelName,
                      choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
                    })}\n\n`));
                    ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
                    ctrl.close();
                  }
                }
              } finally { releaseLock(); }
            }
          }), { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
        }

        let content = "";
        let promptTokens = 0;
        let completionTokens = 0;
        for await (const msg of e.generate(genParams)) {
          if (msg.type === "token") { content += msg.text; completionTokens++; }
          else if (msg.type === "done") { promptTokens = msg.prompt_tokens ?? 0; }
        }
        return Response.json({
          id: reqId, object: "chat.completion", created: Math.floor(Date.now()/1000), model: modelName,
          choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
          usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens }
        });
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
    // Backward compat: try old .hfq naming for the SAME quant level only
    const base = entry.file.replace(/\.(hf4|hf6)$/, "");
    const isHf6 = entry.file.endsWith(".hf6");
    const oldNames = isHf6
      ? [base + ".hfq6.hfq"]                              // HF6 → only try old hfq6
      : [base + ".q4.hfq", base + "-hfq4.hfq", base + ".hfq"];  // HF4 → only try old q4/hfq4
    for (const old of oldNames) {
      const op = join(MODELS_DIR, old);
      if (existsSync(op)) return op;
    }
  }

  // Fuzzy search local dirs
  const dirs = [resolve(__dirname, "../models"), MODELS_DIR];
  for (const dir of dirs) {
    try { for (const f of readdirSync(dir)) {
      if ((f.endsWith(".hf4") || f.endsWith(".hf6") || f.endsWith(".hfq")) && (f.includes(name) || f.includes(name.replace(":", "-")))) return join(dir, f);
    }} catch {}
  }
  return null;
}

function listLocal() {
  const models: { name: string; tag: string; size: string }[] = [];
  const seen = new Set<string>();
  for (const dir of [MODELS_DIR, resolve(__dirname, "../models")]) {
    try { for (const f of readdirSync(dir)) {
      if ((f.endsWith(".hf4") || f.endsWith(".hf6") || f.endsWith(".hfq")) && !seen.has(f)) {
        seen.add(f);
        const sz = (statSync(join(dir, f)).size / 1e9).toFixed(1);
        // Find matching registry tag (check new and old naming)
        const fNorm = f.replace(/\.q4\.hfq$/, ".hf4").replace(/\.hfq6\.hfq$/, ".hf6").replace(/-hfq4\.hfq$/, ".hf4").replace(/\.hfq$/, ".hf4");
        const tag = Object.entries(REGISTRY).find(([_, e]) => e.file === f || e.file === fNorm)?.[0] || "";
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
    if (!model) { console.error("Usage: hipfire run <model> [flags] [prompt]\n\nFlags:\n  --temp <float>           Temperature (default 0.3)\n  --top-p <float>          Top-p sampling (default 0.8)\n  --repeat-penalty <float> Repeat penalty (default 1.3)\n  --max-tokens <int>       Max tokens to generate (default 512)\n  --image <path>           Image for VL models\n\nExamples:\n  hipfire run qwen3.5:9b \"Hello\"\n  hipfire run qwen3.5:9b --temp 0.7 --max-tokens 256 \"Write a poem\"\n  hipfire run qwen3.5:4b --image photo.png \"Describe this\""); process.exit(1); }
    // Parse --key value flags
    const flagDefs: Record<string, { default: number | string | undefined }> = {
      "--image": { default: undefined }, "--temp": { default: 0.3 },
      "--top-p": { default: 0.8 }, "--repeat-penalty": { default: 1.3 },
      "--max-tokens": { default: 512 },
    };
    const flags: Record<string, string> = {};
    const flagIndices = new Set<number>();
    for (const key of Object.keys(flagDefs)) {
      const idx = rest.indexOf(key);
      if (idx >= 0 && idx + 1 < rest.length) {
        const val = rest[idx + 1];
        // Reject flag values that look like other flags
        if (val.startsWith("--")) { console.error(`Error: ${key} requires a value, got '${val}'`); process.exit(1); }
        // Validate numeric flags
        if (key !== "--image" && isNaN(Number(val))) { console.error(`Error: ${key} requires a number, got '${val}'`); process.exit(1); }
        flags[key] = val;
        flagIndices.add(idx); flagIndices.add(idx + 1);
      } else if (idx >= 0) {
        console.error(`Error: ${key} requires a value`); process.exit(1);
      }
    }
    const image = flags["--image"];
    const temp = Number(flags["--temp"] ?? 0.3);
    const topP = Number(flags["--top-p"] ?? 0.8);
    const repeatPenalty = Number(flags["--repeat-penalty"] ?? 1.3);
    const maxTokens = Math.floor(Number(flags["--max-tokens"] ?? 512));
    if (temp < 0) { console.error("Error: --temp must be >= 0 (0 = greedy)"); process.exit(1); }
    if (topP <= 0 || topP > 1) { console.error("Error: --top-p must be in (0, 1]"); process.exit(1); }
    if (repeatPenalty < 1) { console.error("Error: --repeat-penalty must be >= 1.0"); process.exit(1); }
    if (maxTokens < 1) { console.error("Error: --max-tokens must be >= 1"); process.exit(1); }
    const filtered = rest.slice(1).filter((_, i) => !flagIndices.has(i + 1));
    const prompt = filtered.join(" ") || (image ? "Describe this image." : "Hello");
    await run(model, prompt, image, temp, maxTokens, repeatPenalty, topP);
    break;
  }
  case "pull": {
    const tag = rest[0];
    if (!tag) { console.error("Usage: hipfire pull <model>\n\nExamples:\n  hipfire pull qwen3.5:9b\n  hipfire pull qwen3.5:4b-hf6\n  hipfire pull qwen3.5:27b\n\nAvailable:\n" + Object.entries(REGISTRY).map(([t, e]) => `  ${t.padEnd(22)} ${e.size_gb.toString().padStart(5)}GB  ${e.desc}`).join("\n")); process.exit(1); }
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
  case "update": {
    console.error("Updating hipfire...");
    const srcDir = join(HIPFIRE_DIR, "src");
    const repoDir = existsSync(join(srcDir, "Cargo.toml")) ? srcDir : resolve(__dirname, "..");
    const git = (args: string[]) => Bun.spawnSync(["git", ...args], { cwd: repoDir, stdio: ["inherit", "inherit", "inherit"] });
    git(["pull", "origin", "master"]);
    // Rebuild
    console.error("Rebuilding...");
    const build = Bun.spawnSync(
      ["cargo", "build", "--release", "--features", "deltanet", "--example", "daemon", "--example", "infer", "--example", "run", "-p", "engine"],
      { cwd: repoDir, stdio: ["inherit", "inherit", "inherit"] }
    );
    if (build.exitCode !== 0) { console.error("Build failed."); process.exit(1); }
    // Recopy binaries
    const binDir = join(HIPFIRE_DIR, "bin");
    const { copyFileSync } = await import("fs");
    const exe = process.platform === "win32" ? ".exe" : "";
    for (const bin of ["daemon", "infer", "run"]) {
      const src = join(repoDir, `target/release/examples/${bin}${exe}`);
      const dst = join(binDir, `${bin}${exe}`);
      if (existsSync(src)) { copyFileSync(src, dst); }
    }
    // Recopy CLI
    copyFileSync(join(repoDir, "cli/index.ts"), join(HIPFIRE_DIR, "cli/index.ts"));
    // Detect GPU arch from sysfs (cross-platform, no external commands)
    let archOut = "";
    try { archOut = await Bun.file("/sys/class/kfd/kfd/topology/nodes/1/properties").text(); } catch {}
    if (!archOut) try { archOut = await Bun.file("/sys/class/kfd/kfd/topology/nodes/0/properties").text(); } catch {}
    const verMatch = archOut.match(/gfx_target_version\s+(\d+)/);
    let gpuArch = "unknown";
    if (verMatch) {
      // Derive gfx arch from version number: e.g. 100100→gfx1010, 110001→gfx1100, 115100→gfx1151
      const ver = parseInt(verMatch[1]);
      const major = Math.floor(ver / 10000);
      const minor = Math.floor((ver % 10000) / 100);
      const step = ver % 100;
      gpuArch = `gfx${major}${minor.toString().padStart(2, '0')}${step || '0'}`;
      // Normalize: gfx10010 → gfx1010, gfx110000 stays gfx1100
      gpuArch = gpuArch.replace(/^(gfx\d{4})0$/, '$1');
    }
    if (gpuArch !== "unknown") {
      const kernelSrc = join(repoDir, "kernels/compiled", gpuArch);
      const kernelDst = join(binDir, "kernels/compiled", gpuArch);
      mkdirSync(kernelDst, { recursive: true });
      if (existsSync(kernelSrc)) {
        for (const f of readdirSync(kernelSrc)) {
          if (f.endsWith(".hsaco")) copyFileSync(join(kernelSrc, f), join(kernelDst, f));
        }
        console.error(`  Updated ${gpuArch} kernels ✓`);
      }
    }
    // Rename legacy .hfq model files to .hf4/.hf6
    const { renameSync } = await import("fs");
    try {
      for (const f of readdirSync(MODELS_DIR)) {
        if (!f.endsWith(".hfq")) continue;
        let newName = "";
        if (f.endsWith(".q4.hfq")) newName = f.replace(/\.q4\.hfq$/, ".hf4");
        else if (f.endsWith(".hfq6.hfq")) newName = f.replace(/\.hfq6\.hfq$/, ".hf6");
        else if (f.match(/-hfq4\.hfq$/)) newName = f.replace(/-hfq4\.hfq$/, ".hf4");
        else if (f.match(/-hfq4g\d+\.hfq$/)) continue; // skip experimental variants
        else newName = f.replace(/\.hfq$/, ".hf4"); // bare .hfq → assume hf4
        if (newName && newName !== f && !existsSync(join(MODELS_DIR, newName))) {
          renameSync(join(MODELS_DIR, f), join(MODELS_DIR, newName));
          console.error(`  Renamed ${f} → ${newName}`);
        }
      }
    } catch {}
    console.error("hipfire updated ✓");
    break;
  }
  case "diag": {
    console.log("hipfire diagnostics\n");

    // 1. Check daemon binary
    const exe2 = process.platform === "win32" ? ".exe" : "";
    const bins = [
      resolve(__dirname, `../target/release/examples/daemon${exe2}`),
      join(HIPFIRE_DIR, "bin", `daemon${exe2}`),
    ];
    const daemonBin = bins.find(p => existsSync(p));
    console.log(`daemon binary: ${daemonBin ? "found" : "NOT FOUND"}`);
    if (!daemonBin) { console.log("  Install: hipfire update\n"); process.exit(1); }

    // 2. Check local models
    const models = listLocal();
    console.log(`local models:  ${models.length}`);
    for (const m of models) console.log(`  ${m.name.padEnd(35)} ${m.size.padStart(6)}`);

    // 3. Check pre-compiled kernels
    const binDir = join(HIPFIRE_DIR, "bin");
    const kernelBase = join(binDir, "kernels", "compiled");
    const cwdKernelBase = resolve(__dirname, "../kernels/compiled");
    const kBase = existsSync(kernelBase) ? kernelBase : existsSync(cwdKernelBase) ? cwdKernelBase : null;
    if (kBase) {
      const arches = readdirSync(kBase).filter(d => d.startsWith("gfx"));
      for (const arch of arches) {
        const dir = join(kBase, arch);
        const hsaco = readdirSync(dir).filter(f => f.endsWith(".hsaco")).length;
        const hashes = readdirSync(dir).filter(f => f.endsWith(".hash")).length;
        console.log(`kernels/${arch}: ${hsaco} blobs, ${hashes} hashes${hashes < hsaco ? " (INCOMPLETE)" : ""}`);
      }
    } else {
      console.log("kernels:       NOT FOUND");
    }

    // 4. Probe daemon for GPU info
    console.log("\nProbing GPU via daemon...");
    try {
      const e = new Engine();
      await e.start();
      await e.send({ type: "ping" }); await e.recv();
      await e.send({ type: "diag" });
      const diag = await e.recv();
      if (diag.type === "diag") {
        console.log(`  GPU arch:    ${diag.arch}`);
        console.log(`  HIP version: ${diag.hip_version}`);
        console.log(`  VRAM free:   ${diag.vram_free_mb} MB`);
        console.log(`  VRAM total:  ${diag.vram_total_mb} MB`);
        console.log(`  Model:       ${diag.model_loaded ? diag.model_arch : "none loaded"}`);
        console.log(`  Kernels:     ${diag.kernels} blobs, ${diag.kernel_hashes} hashes`);

        // Recommendations based on total VRAM
        const vram = diag.vram_total_mb;
        console.log("");
        if (vram < 4000) {
          console.log("TIP: <4GB VRAM — use qwen3.5:0.8b (430MB)");
        } else if (vram < 6000) {
          console.log("TIP: 4-6GB VRAM — use qwen3.5:4b (2.1GB)");
        } else if (vram < 16000) {
          console.log("TIP: 6-16GB VRAM — qwen3.5:9b (4.5GB) is your best option");
        } else if (vram < 24000) {
          console.log("TIP: 16-24GB VRAM — qwen3.5:27b HFQ4 (14.3GB). Note: HFQ4 degrades on complex tasks");
        } else {
          console.log("TIP: 24GB+ VRAM — qwen3.5:27b-hf6 (21.4GB) for best quality");
        }
        if (models.length === 0) {
          const rec = vram < 4000 ? "qwen3.5:0.8b" : vram < 6000 ? "qwen3.5:4b" : vram < 16000 ? "qwen3.5:9b" : vram < 24000 ? "qwen3.5:27b" : "qwen3.5:27b-hf6";
          console.log(`TIP: No models downloaded. Run: hipfire pull ${rec}`);
        }
      } else {
        console.log(`  Error: ${diag.message || "unexpected response"}`);
      }
      await e.stop();
    } catch (err: any) {
      console.log(`  Failed to start daemon: ${err.message}`);
      console.log("  Check: ROCm/HIP installed? AMD GPU visible? /dev/kfd accessible?");
    }

    console.log("\nDone.");
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
  diag                  Diagnostics — GPU, VRAM, HIP version, kernels, models
  rm <model>            Delete model
  update                Pull latest code, rebuild, update kernels

Models:
  hipfire pull qwen3.5:9b            # 4.5GB, best quality for 8GB cards
  hipfire pull qwen3.5:4b            # 2.1GB, best speed/quality balance
  hipfire pull qwen3.5:27b           # 14.3GB, needs 16GB+ VRAM
  hipfire pull qwen3.5:9b-hf6      # 6.8GB, higher quality (6-bit)

Quick start:
  hipfire pull qwen3.5:4b
  hipfire run qwen3.5:4b "What is the capital of France?"
  hipfire serve`);
}
