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
const CONFIG_PATH = join(HIPFIRE_DIR, "config.json");
const DEFAULT_PORT = 11435;
const TEMP_CORRECTION = 0.82;

mkdirSync(MODELS_DIR, { recursive: true });

// ─── Persistent config ─────────────────────────────────
interface HipfireConfig {
  kv_cache: string;       // "q8" (default) or "turbo4", "turbo3", "turbo2"
  default_model: string;  // model tag for serve pre-warm, e.g. "qwen3.5:9b"
  temperature: number;    // default temperature for run
  top_p: number;
  repeat_penalty: number;
  max_tokens: number;
  port: number;           // default serve port
}

const CONFIG_DEFAULTS: HipfireConfig = {
  kv_cache: "q8",
  default_model: "qwen3.5:9b",
  temperature: 0.3,
  top_p: 0.8,
  repeat_penalty: 1.3,
  max_tokens: 512,
  port: DEFAULT_PORT,
};

function validateConfigValue(key: string, value: any): boolean {
  switch (key) {
    case "kv_cache": return ["q8", "turbo2", "turbo3", "turbo4"].includes(value);
    case "temperature": return typeof value === "number" && value >= 0 && value <= 2;
    case "top_p": return typeof value === "number" && value > 0 && value <= 1;
    case "repeat_penalty": return typeof value === "number" && value >= 1 && value <= 3;
    case "max_tokens": return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 32768;
    case "port": return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 65535;
    case "default_model": return typeof value === "string" && value.trim().length > 0;
    default: return false;
  }
}

function loadConfig(): HipfireConfig {
  try {
    const raw = JSON.parse(require("fs").readFileSync(CONFIG_PATH, "utf-8"));
    const result = { ...CONFIG_DEFAULTS };
    for (const key of Object.keys(CONFIG_DEFAULTS)) {
      if (key in raw && validateConfigValue(key, raw[key])) {
        (result as any)[key] = raw[key];
      }
    }
    return result;
  } catch { return { ...CONFIG_DEFAULTS }; }
}

function saveConfig(cfg: HipfireConfig) {
  // Only write keys that differ from defaults
  const out: Record<string, any> = {};
  for (const [k, v] of Object.entries(cfg)) {
    if (v !== (CONFIG_DEFAULTS as any)[k]) out[k] = v;
  }
  require("fs").writeFileSync(CONFIG_PATH, JSON.stringify(out, null, 2) + "\n");
}

const cfg = loadConfig();

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

  // Qwen3.5 MagnumQuant MQ4 — FWHT-rotated 4-bit. Q8-grade quality at Q4 bandwidth,
  // backed by a mandatory 9-test byte-exact greedy quality gate (tests/quality-baselines/).
  // Perf numbers are RX 7900 XTX forward-only (200-iter bench_qwen35_forward).
  "qwen3.5:0.8b-mq4":  { repo: hfRepo("qwen3.5","0.8b"), file: "qwen3.5-0.8b.mq4",     size_gb: 0.55, min_vram_gb: 1,  desc: "447 tok/s, quality-gated" },
  "qwen3.5:4b-mq4":    { repo: hfRepo("qwen3.5","4b"),   file: "qwen3.5-4b.mq4",       size_gb: 2.6,  min_vram_gb: 4,  desc: "187 tok/s, quality-gated" },
  "qwen3.5:9b-mq4":    { repo: hfRepo("qwen3.5","9b"),   file: "qwen3.5-9b.mq4",       size_gb: 5.3,  min_vram_gb: 6,  desc: "135 tok/s, quality-gated" },
  "qwen3.5:27b-mq4":   { repo: hfRepo("qwen3.5","27b"),  file: "qwen3.5-27b.mq4",      size_gb: 15.0, min_vram_gb: 16, desc: "46 tok/s, quality-gated 16GB+" },

  // Qwen3 (standard attention)
  "qwen3:0.6b":    { repo: hfRepo("qwen3","0.6b"),   file: "qwen3-0.6b.hf4",          size_gb: 0.4,  min_vram_gb: 1,  desc: "standard attention" },
  "qwen3:8b":      { repo: hfRepo("qwen3","8b"),     file: "qwen3-8b.hf4",            size_gb: 4.1,  min_vram_gb: 6,  desc: "60 tok/s, standard attention" },

  // Community finetunes (Qwen3.5 architecture, same engine)
  "carnice:9b":      { repo: "schuttdev/hipfire-carnice-9b",   file: "carnice-9b.hf4",     size_gb: 4.5, min_vram_gb: 6, desc: "Hermes tool-use finetune" },
  "carnice:9b-hf6":  { repo: "schuttdev/hipfire-carnice-9b",   file: "carnice-9b.hf6",     size_gb: 6.8, min_vram_gb: 8, desc: "Hermes tool-use, higher quality" },
  "qwopus:9b":       { repo: "schuttdev/hipfire-qwopus-9b",    file: "qwopus-9b.hf4",      size_gb: 4.5, min_vram_gb: 6, desc: "Qwopus3.5 v3 finetune" },
  "qwopus:9b-hf6":   { repo: "schuttdev/hipfire-qwopus-9b",    file: "qwopus-9b.hf6",      size_gb: 6.8, min_vram_gb: 8, desc: "Qwopus3.5 v3, higher quality" },
  "qwopus:4b":       { repo: "schuttdev/hipfire-qwopus-4b",    file: "qwopus-4b.hf4",      size_gb: 2.1, min_vram_gb: 4, desc: "Qwopus3.5 v3, 4B" },
  "qwopus:27b":      { repo: "schuttdev/hipfire-qwopus-27b",   file: "qwopus-27b.hf4",     size_gb: 14.3, min_vram_gb: 16, desc: "Qwopus3.5 v3, 27B" },
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
    this.buffer = "";
    this.lines = [];
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

  /// Drain any in-flight generation until "done" or "error". Call this after
  /// a generate stream is interrupted (e.g., client disconnect) to resync
  /// the daemon's stdout before sending the next command.
  /// If drain times out, kills and restarts the daemon — a dangling recv()
  /// on a killed process resolves with "daemon closed" harmlessly.
  async drain() {
    let drained = false;
    try {
      // Use a single timeout for the entire drain operation
      const result = await Promise.race([
        (async () => {
          while (true) {
            const r = await this.recv();
            if (r.type === "done" || r.type === "error") return true;
          }
        })(),
        new Promise<false>((res) => setTimeout(() => res(false), 10_000)),
      ]);
      drained = result;
    } catch { /* daemon closed — already clean */ drained = true; }

    if (!drained) {
      // Timed out — dangling recv() still holds the reader.
      // Kill the daemon to cancel it, then restart fresh.
      console.error("[hipfire] drain timed out — restarting daemon");
      await this.stop();
      await this.start();
      await this.send({ type: "ping" }); await this.recv();
    }
  }

  generating = false;

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

  const turboMode = process.env.TURBO ? Number(process.env.TURBO) : (cfg.kv_cache === "q8" ? 0 : Number(cfg.kv_cache.replace("turbo", "")));
  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();
  await e.send({ type: "load", model: path, turbo: turboMode });
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

  let inThink = false;
  let stripNextLeadingNl = false;
  for await (const msg of e.generate(genMsg)) {
    if (msg.type === "token") {
      let text = msg.text as string;
      if (!inThink && text.includes("<think>")) { inThink = true; text = text.replace(/<think>/g, ""); }
      if (inThink) {
        if (text.includes("</think>")) {
          text = text.split("</think>").slice(1).join("</think>");
          inThink = false;
          stripNextLeadingNl = true; // strip newline between </think> and content
        } else { continue; }
      }
      text = text.replace(/<\|im_end\|>/g, "");
      if (!text) continue;
      if (stripNextLeadingNl) { text = text.replace(/^\n+/, ""); stripNextLeadingNl = false; if (!text) continue; }
      process.stdout.write(text);
    }
    else if (msg.type === "done") console.error(`\n[${msg.tokens} tok, ${msg.tok_s} tok/s]`);
  }
  await e.stop();
}

async function serve(port: number) {
  const turboMode = process.env.TURBO ? Number(process.env.TURBO) : (cfg.kv_cache === "q8" ? 0 : Number(cfg.kv_cache.replace("turbo", "")));
  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();
  let current: string | null = null;

  // Pre-warm: load default model and compile kernels before accepting requests
  const defaultModel = process.env.HIPFIRE_MODEL || cfg.default_model;
  const rawWarmPath = findModel(defaultModel);
  const warmPath = rawWarmPath ? resolve(rawWarmPath) : null;
  if (warmPath) {
    try {
      console.error(`[hipfire] pre-warming ${defaultModel}...`);
      await e.send({ type: "load", model: warmPath, turbo: turboMode });
      const loadResult = await e.recv();
      if (loadResult.type === "error") {
        console.error(`[hipfire] pre-warm load failed: ${loadResult.message} (will load on first request)`);
      } else {
        for await (const msg of e.generate({ type: "generate", id: "warmup", prompt: "Hi", temperature: 0, max_tokens: 1 })) {
          if (msg.type === "done") break;
        }
        await e.send({ type: "reset" }); await e.recv();
        current = warmPath;
        console.error(`[hipfire] warm-up complete`);
      }
    } catch (err: any) {
      console.error(`[hipfire] pre-warm failed: ${err?.message} — restarting daemon`);
      current = null;
      try { await e.stop(); } catch {}
      await e.start();
      await e.send({ type: "ping" }); await e.recv();
    }
  }

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

      if (url.pathname !== "/v1/chat/completions" || req.method !== "POST")
        return Response.json({ error: "not found" }, { status: 404 });

      await acquireLock();
      let lockReleased = false;
      const safeRelease = () => { if (!lockReleased) { lockReleased = true; releaseLock(); } };

      // If a previous generation was interrupted (client disconnect), drain
      // remaining daemon output before sending new commands.
      // If drain restarts the daemon, clear current so model reloads.
      if (e.generating) {
        await e.drain();
        e.generating = false;
        current = null; // daemon may have restarted — force model reload
      }

      try {
        const body = await req.json();
        const messages: any[] = body.messages || [];
        const tools: any[] = body.tools || [];

        // OpenAI API is stateless: each request has the full conversation.
        // Reset daemon state so prior requests don't bleed into this one.
        await e.send({ type: "reset" }); await e.recv();

        // Build prompt from messages with proper role handling
        let systemPrompt = "";
        let userPrompt = "";

        // Extract system message
        const sysMsg = messages.find((m: any) => m.role === "system");
        if (sysMsg) systemPrompt = sysMsg.content;

        // Format tools into system prompt (Hermes format)
        if (tools.length > 0) {
          const toolsBlock = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
            + tools.map((t: any) => JSON.stringify(t)).join("\n")
            + "\n</tools>\n\n"
            + 'If you choose to call a function ONLY reply in the following format with NO suffix:\n\n'
            + '<tool_call>\n{"name": "example_function", "arguments": {"param": "value"}}\n</tool_call>';
          systemPrompt = systemPrompt ? systemPrompt + "\n\n" + toolsBlock : toolsBlock;
        }

        // Build conversation as multi-turn ChatML prompt.
        // The daemon wraps the prompt as: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        // We embed ChatML turn boundaries inside the prompt so multi-turn conversations
        // (especially tool-calling flows) have proper role structure instead of being
        // collapsed into a single user turn.
        const nonSystem = messages.filter((m: any) => m.role !== "system");
        const convParts: string[] = [];
        for (let i = 0; i < nonSystem.length; i++) {
          const m = nonSystem[i];
          const role = m.role;
          let text = "";

          if (role === "tool") {
            text = `<tool_response>\n${m.content}\n</tool_response>`;
          } else if (role === "assistant" && m.tool_calls) {
            text = m.content || "";
            for (const tc of m.tool_calls) {
              const fn = tc.function || tc;
              text += `\n<tool_call>\n${JSON.stringify({ name: fn.name, arguments: JSON.parse(fn.arguments || "{}") })}\n</tool_call>`;
            }
          } else {
            text = m.content || "";
          }

          if (i === 0) {
            // First message: daemon provides <|im_start|>user\n wrapper,
            // but if it's not a user message, close the user turn and start the right role
            if (role === "user") {
              convParts.push(text);
            } else {
              convParts.push(`<|im_end|>\n<|im_start|>${role}\n${text}`);
            }
          } else {
            // Subsequent messages: close previous turn, start new one
            convParts.push(`<|im_end|>\n<|im_start|>${role}\n${text}`);
          }
        }
        userPrompt = convParts.join("");

        const rawPath = findModel(body.model || "default");
        if (!rawPath) { safeRelease(); return Response.json({ error: "model not found" }, { status: 404 }); }
        // Normalize to avoid spurious reloads when registry vs fuzzy search give different paths
        const path = resolve(rawPath);

        if (current !== path) {
          if (current) { await e.send({ type: "unload" }); await e.recv(); }
          await e.send({ type: "load", model: path, turbo: turboMode });
          const loadResult = await e.recv();
          if (loadResult.type === "error") {
            current = null;
            safeRelease();
            return Response.json({ error: `model load failed: ${loadResult.message}` }, { status: 500 });
          }
          current = path;
        }

        const reqId = `chatcmpl-${Date.now().toString(36)}`;
        const created = Math.floor(Date.now() / 1000);
        const modelName = body.model || "hipfire";
        const genParams: any = {
          type: "generate", id: reqId, prompt: userPrompt,
          temperature: (body.temperature ?? 0.3) * TEMP_CORRECTION,
          max_tokens: body.max_tokens ?? 512,
          repeat_penalty: body.repeat_penalty ?? body.frequency_penalty ? 1.0 + (body.frequency_penalty ?? 0) : 1.3,
          top_p: body.top_p ?? 0.8,
        };
        if (systemPrompt) genParams.system = systemPrompt;

        // Parse tool calls from model output: <tool_call>{"name":..., "arguments":...}</tool_call>
        function parseToolCalls(text: string): { content: string | null; tool_calls: any[] | null } {
          if (!text.includes("<tool_call>")) return { content: text, tool_calls: null };
          const pattern = /<tool_call>\s*(.*?)\s*<\/tool_call>|<tool_call>\s*(.*)/gs;
          const matches = [...text.matchAll(pattern)];
          if (!matches.length) return { content: text, tool_calls: null };
          const tool_calls: any[] = [];
          for (const m of matches) {
            const raw = (m[1] || m[2] || "").trim();
            if (!raw) continue;
            try {
              const tc = JSON.parse(raw);
              tool_calls.push({
                id: `call_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
                type: "function",
                function: { name: tc.name, arguments: JSON.stringify(tc.arguments || {}) }
              });
            } catch {}
          }
          if (!tool_calls.length) return { content: text, tool_calls: null };
          const before = text.slice(0, text.indexOf("<tool_call>")).trim();
          return { content: before || null, tool_calls };
        }

        if (body.stream) {
          const enc = new TextEncoder();
          let streamCancelled = false;
          e.generating = true;
          const hasTool = tools.length > 0;
          return new Response(new ReadableStream({
            async start(ctrl) {
              try {
                let inThink = false;
                let stripNextLeadingNl = false;
                // When tools are present, accumulate full output for tool-call parsing
                let accumulated = hasTool ? "" : null;
                for await (const msg of e.generate(genParams)) {
                  if (streamCancelled) continue; // drain remaining tokens, don't enqueue
                  if (msg.type === "token") {
                    let text = msg.text as string;
                    if (!inThink && text.includes("<think>")) { inThink = true; text = text.replace(/<think>/g, ""); }
                    if (inThink) {
                      if (text.includes("</think>")) {
                        text = text.split("</think>").slice(1).join("</think>");
                        inThink = false;
                        stripNextLeadingNl = true;
                      } else { continue; }
                    }
                    text = text.replace(/<\|im_end\|>/g, "");
                    if (!text) continue;
                    if (stripNextLeadingNl) { text = text.replace(/^\n+/, ""); stripNextLeadingNl = false; if (!text) continue; }
                    if (accumulated !== null) {
                      accumulated += text; // buffer for tool-call parsing at end
                    } else {
                      ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                        id: reqId, object: "chat.completion.chunk", created, model: modelName,
                        choices: [{ index: 0, delta: { content: text }, finish_reason: null }]
                      })}\n\n`));
                    }
                  } else if (msg.type === "done") {
                    // When tools are present, parse accumulated text for tool calls
                    if (accumulated !== null) {
                      const parsed = parseToolCalls(accumulated);
                      if (parsed.tool_calls) {
                        if (parsed.content) {
                          ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                            id: reqId, object: "chat.completion.chunk", created, model: modelName,
                            choices: [{ index: 0, delta: { content: parsed.content }, finish_reason: null }]
                          })}\n\n`));
                        }
                        for (let ti = 0; ti < parsed.tool_calls.length; ti++) {
                          ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                            id: reqId, object: "chat.completion.chunk", created, model: modelName,
                            choices: [{ index: 0, delta: { tool_calls: [{ index: ti, ...parsed.tool_calls[ti] }] }, finish_reason: null }]
                          })}\n\n`));
                        }
                        ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                          id: reqId, object: "chat.completion.chunk", created, model: modelName,
                          choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }]
                        })}\n\n`));
                      } else {
                        // No tool calls — flush accumulated content
                        if (accumulated) {
                          ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                            id: reqId, object: "chat.completion.chunk", created, model: modelName,
                            choices: [{ index: 0, delta: { content: accumulated }, finish_reason: null }]
                          })}\n\n`));
                        }
                        ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                          id: reqId, object: "chat.completion.chunk", created, model: modelName,
                          choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
                        })}\n\n`));
                      }
                    } else {
                      ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                        id: reqId, object: "chat.completion.chunk", created, model: modelName,
                        choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
                      })}\n\n`));
                    }
                    ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
                    ctrl.close();
                    return;
                  } else if (msg.type === "error") {
                    ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                      id: reqId, object: "chat.completion.chunk", created, model: modelName,
                      choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
                    })}\n\n`));
                    ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
                    ctrl.close();
                    return;
                  }
                }
                // Safety: if loop exits without done/error (shouldn't happen), close stream
                try { ctrl.close(); } catch {}
              } finally {
                e.generating = false;
                safeRelease();
              }
            },
            cancel() { streamCancelled = true; } // lock released in finally after generation drains
          }), { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
        }

        let content = "";
        let completionTokens = 0;
        e.generating = true;
        for await (const msg of e.generate(genParams)) {
          if (msg.type === "token") { content += msg.text; completionTokens++; }
        }
        e.generating = false;

        // Strip think tags and special tokens.
        // Greedy match: strip everything from first <think> to last </think>.
        // If <think> is unclosed, strip from <think> to end of content.
        content = content.replace(/<think>[\s\S]*?<\/think>\s*/g, "")
          .replace(/<think>[\s\S]*$/, "") // unclosed think block
          .replace(/<\|im_end\|>/g, "").trim();

        // Check for tool calls in response
        const parsed = parseToolCalls(content);
        const choice: any = { index: 0, finish_reason: parsed.tool_calls ? "tool_calls" : "stop" };
        if (parsed.tool_calls) {
          choice.message = { role: "assistant", content: parsed.content, tool_calls: parsed.tool_calls };
        } else {
          choice.message = { role: "assistant", content };
        }

        safeRelease();
        return Response.json({
          id: reqId, object: "chat.completion", created, model: modelName,
          choices: [choice],
          usage: { prompt_tokens: 0, completion_tokens: completionTokens, total_tokens: completionTokens }
        });
      } catch (err: any) {
        safeRelease();
        return Response.json({ error: err?.message || "internal error" }, { status: 500 });
      }
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
    // (only applies to .hf4 / .hf6 — .mq4 has no legacy alias)
    if (entry.file.endsWith(".hf4") || entry.file.endsWith(".hf6")) {
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
  }

  // Fuzzy search local dirs (top-level + one level of subdirectories)
  // If the name includes a quant hint (hf4/hf6/mq4), match exactly. Otherwise prefer .hf4 (default quant).
  const searchName = name.replace(":", "-");
  const hasQuantHint = /\.(hf[46]|mq4)$|-(hf[46]|mq4)$/.test(name);
  const isModel = (f: string) => {
    if (!(f.endsWith(".hf4") || f.endsWith(".hf6") || f.endsWith(".hfq") || f.endsWith(".mq4"))) return false;
    if (!(f.includes(name) || f.includes(searchName))) return false;
    // Exact filename match — user typed the specific file they want
    if (f === name || f === searchName) return true;
    // If user didn't specify a quant, only match the default (4-bit HF) variant.
    // HF4 is still the default because it's the fastest; MQ4 is explicit opt-in
    // for quality-gated workloads. .hf6 and .mq4 both require an explicit tag.
    if (!hasQuantHint) {
      if (f.endsWith(".hf6")) return false;
      if (f.endsWith(".mq4")) return false;
      if (f.endsWith(".hfq")) {
        const stem = f.slice(0, -4); // strip .hfq
        const isDefaultQ4 = stem.endsWith(".q4") || stem.endsWith("-hfq4")
          || stem === searchName || stem === name;
        if (!isDefaultQ4) return false;
      }
    }
    return true;
  };
  const dirs = [resolve(__dirname, "../models"), MODELS_DIR];
  for (const dir of dirs) {
    try {
      for (const f of readdirSync(dir)) {
        const full = join(dir, f);
        if (isModel(f)) return full;
        // Check one level of subdirectories (e.g., models/community/)
        try {
          if (statSync(full).isDirectory()) {
            for (const sf of readdirSync(full)) {
              if (isModel(sf)) return join(full, sf);
            }
          }
        } catch {}
      }
    } catch {}
  }
  return null;
}

function listLocal() {
  const models: { name: string; tag: string; size: string }[] = [];
  const seen = new Set<string>();
  for (const dir of [MODELS_DIR, resolve(__dirname, "../models")]) {
    let entries: string[];
    try { entries = readdirSync(dir); } catch { continue; }
    for (const f of entries) {
      if ((f.endsWith(".hf4") || f.endsWith(".hf6") || f.endsWith(".hfq") || f.endsWith(".mq4")) && !seen.has(f)) {
        seen.add(f);
        // statSync may throw on dangling symlinks or files removed mid-scan;
        // skip those individually instead of aborting the rest of the loop
        // (a previous try/catch wrapping the entire iteration ate everything
        // after the first stale symlink — see commit log for the bug story).
        try {
          const sz = (statSync(join(dir, f)).size / 1e9).toFixed(1);
          // Find matching registry tag (check new and old naming)
          const fNorm = f.replace(/\.q4\.hfq$/, ".hf4").replace(/\.hfq6\.hfq$/, ".hf6").replace(/-hfq4\.hfq$/, ".hf4").replace(/\.hfq$/, ".hf4");
          const tag = Object.entries(REGISTRY).find(([_, e]) => e.file === f || e.file === fNorm)?.[0] || "";
          models.push({ name: f, tag, size: `${sz}GB` });
        } catch {}
      }
    }
  }
  return models;
}

// ─── Bench ──────────────────────────────────────────────

interface BenchResult {
  label: string;
  decode: number[];
  prefill: number[];
}

function stats(arr: number[]): { mean: number; min: number; max: number; stdev: number } {
  if (arr.length === 0) return { mean: 0, min: 0, max: 0, stdev: 0 };
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const variance = arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / arr.length;
  return { mean, min, max, stdev: Math.sqrt(variance) };
}

function fmtNum(n: number, w = 7): string {
  return n.toFixed(1).padStart(w);
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  return Promise.race([
    promise.finally(() => clearTimeout(timer)),
    new Promise<T>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms / 1000}s`)), ms);
    }),
  ]);
}

// benchRun result + flag indicating the engine is poisoned (timed out mid-stream)
interface BenchRunResult { decode: number; prefill: number; tokens: number; ok: boolean; poisoned: boolean }

async function benchRun(e: Engine, prompt: string, maxTokens: number, timeoutMs = 120_000): Promise<BenchRunResult> {
  const fail = { decode: 0, prefill: 0, tokens: 0, ok: false, poisoned: false };
  try {
    await withTimeout(e.send({ type: "reset" }).then(() => e.recv()), 10_000, "reset");
  } catch { return { ...fail, poisoned: true }; }
  const genMsg = {
    type: "generate", id: "bench", prompt,
    temperature: 0, max_tokens: maxTokens,
    repeat_penalty: 1.1, top_p: 1.0,
  };
  let decode = 0, prefill = 0, tokens = 0;
  try {
    const run = async () => {
      for await (const msg of e.generate(genMsg)) {
        if (msg.type === "done") {
          decode = msg.tok_s || 0;
          tokens = msg.tokens || 0;
        }
      }
    };
    await withTimeout(run(), timeoutMs, "generate");
  } catch {
    // Timed out mid-stream — daemon is reading/writing stale data, must be killed
    return { ...fail, poisoned: true };
  }
  return { decode, prefill, tokens, ok: decode > 0, poisoned: false };
}

async function bench(model: string, runs: number, experimental: boolean, prompt: string) {
  let modelPath = findModel(model);
  if (!modelPath) {
    const resolved = resolveModelTag(model);
    if (REGISTRY[resolved]) {
      console.error(`Model not found locally. Pulling ${resolved}...`);
      modelPath = await pull(model);
    } else {
      console.error(`Model not found: ${model}`);
      process.exit(1);
    }
  }

  const turboMode = process.env.TURBO ? Number(process.env.TURBO) : (cfg.kv_cache === "q8" ? 0 : Number(cfg.kv_cache.replace("turbo", "")));

  // Start daemon
  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();
  await e.send({ type: "load", model: modelPath, turbo: turboMode });
  const loaded = await e.recv();
  if (loaded.type === "error") { console.error(loaded.message); process.exit(1); }

  // Get real GPU arch from diag (loaded.arch is the model arch, not GPU)
  await e.send({ type: "diag" });
  const diag = await e.recv();
  const gpuArch = diag.arch || "unknown";
  const isRdna2 = gpuArch === "gfx1030" || gpuArch === "gfx1031";

  console.error(`hipfire bench`);
  console.error(`  model:  ${basename(modelPath!)}  [${loaded.arch}]`);
  console.error(`  gpu:    ${gpuArch}`);
  console.error(`  kv:     ${cfg.kv_cache}`);
  console.error(`  runs:   ${runs}`);
  console.error(`  prompt: "${prompt.length > 60 ? prompt.slice(0, 57) + "..." : prompt}"`);

  if (experimental && !isRdna2) {
    console.error(`\n--exp requires RDNA2 (gfx1030/gfx1031), detected ${gpuArch}. Running standard bench.`);
  }

  const doExp = experimental && isRdna2;

  if (doExp) {
    // ── Experimental: RDNA2 variant comparison ──
    // Each variant requires a daemon restart (env var read at kernel compile time)
    const variants = [
      { n: 1, name: "baseline-rdna2",   desc: "(32,16) 2x-unroll" },
      { n: 2, name: "high-occupancy",   desc: "(32,20) 2x-unroll" },
      { n: 3, name: "wide-unroll",      desc: "(32,12) 4x-unroll" },
      { n: 4, name: "dp4a-packed",      desc: "(32,16) dp4a+factored" },
      { n: 5, name: "cache-aggressive", desc: "(32,16) packed+factored" },
    ];

    console.error(`  mode:   experimental (5 RDNA2 kernel variants x ${runs} runs)\n`);
    await e.stop();

    const results: BenchResult[] = [];

    const LOAD_TIMEOUT = 120_000;  // 2min for kernel compile + model load
    const RUN_TIMEOUT = 60_000;   // 1min per generation run

    for (const v of variants) {
      // Clear kernel cache so variant recompiles
      try { const { execSync } = require("child_process"); execSync("rm -rf /tmp/hipfire_kernels/"); } catch {}

      // Restart daemon with variant env var
      process.env.HIPFIRE_RDNA2_VARIANT = String(v.n);
      const ve = new Engine();
      let variantOk = false;
      try {
        await ve.start();
        await withTimeout(ve.send({ type: "ping" }).then(() => ve.recv()), 10_000, "ping");
        await ve.send({ type: "load", model: modelPath, turbo: turboMode });
        const vloaded = await withTimeout(ve.recv(), LOAD_TIMEOUT, `v${v.n} load`);
        if (vloaded.type === "error") {
          console.error(`  v${v.n} ${v.name}: LOAD FAIL — ${vloaded.message}`);
        } else {
          variantOk = true;
        }
      } catch (err: any) {
        console.error(`  v${v.n} ${v.name}: ${err.message || "startup failed"}`);
      }

      if (!variantOk) {
        results.push({ label: `v${v.n} ${v.name}`, decode: [], prefill: [] });
        await ve.stop();
        continue;
      }

      // Warmup
      const warmup = await benchRun(ve, "Hello", 16, 30_000);
      if (warmup.poisoned) {
        console.error(`  v${v.n} ${v.name}: warmup timed out`);
        results.push({ label: `v${v.n} ${v.name}`, decode: [], prefill: [] });
        await ve.stop();
        continue;
      }

      process.stderr.write(`  v${v.n} ${v.name.padEnd(18)} `);
      const decodes: number[] = [];
      const prefills: number[] = [];
      let abandoned = false;

      for (let r = 0; r < runs; r++) {
        const res = await benchRun(ve, prompt, 128, RUN_TIMEOUT);
        if (res.poisoned) {
          // Daemon stream is corrupt — kill it and abort this variant
          process.stderr.write("TIMEOUT ");
          await ve.stop();
          abandoned = true;
          break;
        }
        if (!res.ok) {
          process.stderr.write("FAIL ");
          continue;
        }
        decodes.push(res.decode);
        process.stderr.write(".");
      }
      console.error("");
      results.push({ label: `v${v.n} ${v.name}`, decode: decodes, prefill: prefills });
      if (!abandoned) await ve.stop();
    }
    delete process.env.HIPFIRE_RDNA2_VARIANT;

    // Results table
    console.log("");
    console.log("  V  Name                       Decode tok/s");
    console.log("     launch_bounds               mean   min   max   stdev");
    console.log("  " + "─".repeat(60));

    let bestMean = 0, bestLabel = "";
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      const v = variants[i];
      const d = stats(r.decode);
      if (d.mean > bestMean) { bestMean = d.mean; bestLabel = r.label; }
      if (r.decode.length === 0) {
        console.log(`  ${v.n}  ${v.name.padEnd(18)} ${v.desc.padEnd(22)} FAIL`);
      } else {
        console.log(
          `  ${v.n}  ${v.name.padEnd(18)} ${v.desc.padEnd(9)}` +
          `${fmtNum(d.mean)}${fmtNum(d.min)}${fmtNum(d.max)}${fmtNum(d.stdev)}`
        );
      }
    }

    if (bestLabel) {
      console.log(`\n  Best: ${bestLabel} at ${bestMean.toFixed(1)} tok/s`);
      const bestV = bestLabel.match(/v(\d)/)?.[1] || "1";
      console.log(`  Set default: export HIPFIRE_RDNA2_VARIANT=${bestV}`);
    }

  } else {
    // ── Standard bench ──
    console.error(`  mode:   standard\n`);

    // Warmup
    process.stderr.write("  warming up...");
    const warmup = await benchRun(e, "Hello", 16);
    if (warmup.poisoned) {
      console.error(" TIMEOUT — daemon unresponsive");
      await e.stop();
      process.exit(1);
    }
    console.error(" done\n");

    const decodes: number[] = [];
    const tokenCounts: number[] = [];

    for (let r = 0; r < runs; r++) {
      process.stderr.write(`  run ${r + 1}/${runs} `);
      const res = await benchRun(e, prompt, 128);
      if (res.poisoned) {
        console.error("TIMEOUT — daemon killed");
        await e.stop();
        break;
      }
      if (!res.ok) {
        console.error("FAIL");
        continue;
      }
      decodes.push(res.decode);
      tokenCounts.push(res.tokens);
      console.error(`${res.decode.toFixed(1)} tok/s (${res.tokens} tok)`);
    }

    const d = stats(decodes);

    console.log("");
    console.log("  Decode  tok/s");
    console.log(`    mean:  ${d.mean.toFixed(1)}`);
    console.log(`    min:   ${d.min.toFixed(1)}`);
    console.log(`    max:   ${d.max.toFixed(1)}`);
    console.log(`    stdev: ${d.stdev.toFixed(1)}`);

    if (d.mean > 0) {
      console.log(`    ms/tok: ${(1000 / d.mean).toFixed(1)}`);
    }

    if (isRdna2) {
      console.log(`\n  Tip: Run 'hipfire bench --exp ${model}' to test RDNA2 kernel variants`);
    }

    await e.stop();
  }
}

// ─── Profile ────────────────────────────────────────────

async function profile(modelTag: string | undefined, jsonOutput: boolean, kernelFilter: string | undefined) {
  // Start daemon — we need kernels compiled to profile them
  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();

  // Load a model if specified (triggers kernel compilation for that model's quant type)
  if (modelTag) {
    let modelPath = findModel(modelTag);
    if (!modelPath) {
      const resolved = resolveModelTag(modelTag);
      if (REGISTRY[resolved]) {
        console.error(`Model not found locally. Pulling ${resolved}...`);
        modelPath = await pull(modelTag);
      }
    }
    if (modelPath) {
      const turboMode = process.env.TURBO ? Number(process.env.TURBO) : (cfg.kv_cache === "q8" ? 0 : Number(cfg.kv_cache.replace("turbo", "")));
      await e.send({ type: "load", model: modelPath, turbo: turboMode });
      const loaded = await e.recv();
      if (loaded.type === "error") {
        console.error(`Load failed: ${loaded.message}`);
        await e.stop();
        process.exit(1);
      }
    }
  }

  // Request profile data
  await e.send({ type: "profile" });
  const data = await e.recv();
  await e.stop();

  if (data.type !== "profile") {
    console.error(data.message || "profile failed");
    process.exit(1);
  }

  const gpu = data.gpu;
  const kernels: any[] = data.kernels || [];

  // Apply kernel filter
  const filtered = kernelFilter
    ? kernels.filter((k: any) => k.name.includes(kernelFilter))
    : kernels;

  if (jsonOutput) {
    console.log(JSON.stringify(data, null, 2));
    return;
  }

  // Pretty-print hardware summary
  const icStr = gpu.infinity_cache_mb > 0 ? ` | IC: ${gpu.infinity_cache_mb}MB` : "";
  console.log(`GPU: ${gpu.arch} (${gpu.generation})`);
  console.log(`${gpu.cu_count} CUs | ${gpu.cu_count * gpu.simds_per_cu} SIMDs | Peak BW: ${gpu.peak_bw_gbs.toFixed(0)} GB/s | Boost: ${gpu.boost_clock_mhz} MHz`);
  console.log(`VGPRs/SIMD: ${gpu.vgprs_per_simd} | LDS/CU: ${(gpu.lds_per_cu / 1024)}KB | L2: ${gpu.l2_cache_mb}MB${icStr} | VRAM: ${(gpu.vram_mb / 1024).toFixed(1)}GB`);
  console.log(`Roofline ridge: ${gpu.ridge_point.toFixed(1)} FLOP/byte`);

  if (filtered.length === 0) {
    console.log("\nNo compiled kernels found. Load a model first: hipfire profile <model>");
    return;
  }

  // Kernel table
  console.log(`\nKernel Report (${filtered.length} kernels):`);
  console.log("┌" + "─".repeat(26) + "┬───────┬───────┬─────────┬────────────┬───────────┐");
  console.log("│ Kernel" + " ".repeat(19) + "│ VGPRs │ SGPRs │ LDS (B) │ Occupancy  │ Limiter   │");
  console.log("├" + "─".repeat(26) + "┼───────┼───────┼─────────┼────────────┼───────────┤");

  const bottlenecks: string[] = [];
  for (const k of filtered) {
    const occ = k.occupancy;
    const occStr = `${String(occ.waves).padStart(2)}/${occ.max} ${occ.pct.toFixed(0).padStart(3)}%`;
    const name = k.name.length > 24 ? k.name.slice(0, 24) + ".." : k.name.padEnd(24);
    console.log(
      `│ ${name} │ ${String(k.vgprs).padStart(5)} │ ${String(k.sgprs).padStart(5)} │ ${String(k.lds_bytes).padStart(7)} │ ${occStr.padStart(10)} │ ${occ.limiter.padEnd(9)} │`
    );
    if (occ.limiter !== "wave limit") {
      bottlenecks.push(`${k.name}: occupancy limited by ${occ.limiter} (${k.vgprs} VGPRs → ${occ.waves}/${occ.max} waves)`);
    }
  }
  console.log("└" + "─".repeat(26) + "┴───────┴───────┴─────────┴────────────┴───────────┘");

  // Bottleneck analysis
  if (bottlenecks.length > 0) {
    console.log("\nBottleneck Analysis:");
    for (const b of bottlenecks) {
      console.log(`  ${b}`);
    }
  }

  // Occupancy summary
  const fullOcc = filtered.filter((k: any) => k.occupancy.limiter === "wave limit").length;
  console.log(`\n${fullOcc}/${filtered.length} kernels at max occupancy`);
}

// ─── Main ───────────────────────────────────────────────

const [cmd, ...rest] = process.argv.slice(2);
switch (cmd) {
  case "serve": await serve(parseInt(rest[0]) || cfg.port); break;
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
    const temp = Number(flags["--temp"] ?? cfg.temperature);
    const topP = Number(flags["--top-p"] ?? cfg.top_p);
    const repeatPenalty = Number(flags["--repeat-penalty"] ?? cfg.repeat_penalty);
    const maxTokens = Math.floor(Number(flags["--max-tokens"] ?? cfg.max_tokens));
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
  case "profile": {
    const jsonFlag = rest.includes("--json");
    const kernelIdx = rest.indexOf("--kernel");
    const kernelFilter = kernelIdx >= 0 && kernelIdx + 1 < rest.length ? rest[kernelIdx + 1] : undefined;
    const skipSet = new Set<number>();
    if (jsonFlag) skipSet.add(rest.indexOf("--json"));
    if (kernelIdx >= 0) { skipSet.add(kernelIdx); skipSet.add(kernelIdx + 1); }
    const positional = rest.filter((_, i) => !skipSet.has(i));
    const profileModel = positional[0]; // optional: model to load (triggers kernel compile)
    await profile(profileModel, jsonFlag, kernelFilter);
    break;
  }
  case "update": {
    console.error("Updating hipfire...");
    const srcDir = join(HIPFIRE_DIR, "src");
    const repoDir = existsSync(join(srcDir, "Cargo.toml")) ? srcDir : resolve(__dirname, "..");
    const git = (args: string[]) => Bun.spawnSync(["git", ...args], { cwd: repoDir, stdio: ["inherit", "inherit", "inherit"] });
    const gitOut = (args: string[]) => {
      const r = Bun.spawnSync(["git", ...args], { cwd: repoDir, stdout: "pipe", stderr: "pipe" });
      return { code: r.exitCode ?? 1, out: (r.stdout?.toString() ?? "").trim() };
    };
    const must = (code: number | null | undefined, msg: string) => {
      if ((code ?? 1) !== 0) {
        console.error(`  ${msg}`);
        console.error(`  Repo: ${repoDir}`);
        process.exit(1);
      }
    };
    // Refuse to auto-reset when on a feature branch: `hipfire update` is for
    // end-users syncing master, not for developers working off a dev branch.
    const branch = gitOut(["rev-parse", "--abbrev-ref", "HEAD"]);
    if (branch.code === 0 && branch.out && branch.out !== "master" && branch.out !== "HEAD") {
      console.error(`  Current branch is '${branch.out}', not master.`);
      console.error(`  'hipfire update' only updates master. Run 'git pull' manually for other branches.`);
      process.exit(1);
    }
    // Fetch upstream master. Works on shallow clones (extends depth as needed).
    must(git(["fetch", "origin", "master"]).exitCode, "git fetch origin master failed (check network / remote access)");
    // Refuse to silently drop unpushed local commits on master. Developers
    // working directly on master need to push (or rebase) before updating.
    const ahead = gitOut(["rev-list", "--count", "origin/master..HEAD"]);
    if (ahead.code === 0 && parseInt(ahead.out || "0", 10) > 0) {
      console.error(`  Local master has ${ahead.out} unpushed commit(s) — refusing to reset.`);
      console.error(`  Push or rebase your commits, then re-run 'hipfire update'.`);
      process.exit(1);
    }
    // If the working tree is dirty (e.g. Cargo.lock rewritten by a different
    // cargo version, line-ending drift on Windows, or genuine edits), stash
    // everything under a named entry so the user can recover via `git stash pop`.
    // This replaces the old `git pull` which aborted with
    //   "Your local changes to the following files would be overwritten by merge"
    // whenever any tracked file was modified.
    const status = gitOut(["status", "--porcelain"]);
    if (status.code === 0 && status.out.length > 0) {
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const stashMsg = `hipfire-update-${stamp}`;
      console.error(`  Local modifications detected — stashing as '${stashMsg}'`);
      must(
        git(["stash", "push", "--include-untracked", "-m", stashMsg]).exitCode,
        "git stash failed — aborting so your changes aren't lost",
      );
      console.error(`  Recover later with: git -C ${repoDir} stash pop`);
    }
    // Hard-reset to upstream. After the stash (or on a clean tree) this is a
    // guaranteed fast-forward-or-force to origin/master — no merge to abort.
    must(
      git(["reset", "--hard", "origin/master"]).exitCode,
      "git reset --hard origin/master failed — repo may be in an inconsistent state",
    );
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
    // Pre-compile GPU kernels so `hipfire serve` starts instantly
    const daemonForPrecompile = join(binDir, `daemon${exe}`) ;
    if (existsSync(daemonForPrecompile)) {
      console.error("Pre-compiling GPU kernels...");
      const pc = Bun.spawnSync([daemonForPrecompile, "--precompile"], { stdio: ["inherit", "inherit", "inherit"] });
      if (pc.exitCode !== 0) console.error("  Warning: kernel precompilation failed (serve will compile on first run)");
    }
    console.error("hipfire updated ✓");
    break;
  }
  case "diag": {
    console.log("hipfire diagnostics\n");
    const sh = (cmd: string) => {
      try { const r = Bun.spawnSync(["bash", "-c", cmd], { stdout: "pipe", stderr: "pipe" }); return r.stdout?.toString().trim() || ""; }
      catch { return ""; }
    };

    // ── 1. Platform detection ──────────────────────────────
    const platform = process.platform;
    const isWsl = existsSync("/proc/version") && (sh("cat /proc/version") || "").toLowerCase().includes("microsoft");
    const isNativeLinux = platform === "linux" && !isWsl;
    const isWindows = platform === "win32";
    const platformLabel = isWsl ? "WSL2 (Windows Subsystem for Linux)" : isWindows ? "Windows (native)" : isNativeLinux ? "Linux (native)" : platform;
    console.log(`platform:      ${platformLabel}`);
    if (isWsl) {
      const wslVer = sh("cat /proc/version");
      const kernelMatch = wslVer.match(/(\d+\.\d+\.\d+)/);
      if (kernelMatch) console.log(`  WSL kernel:  ${kernelMatch[1]}`);
    }

    // ── 2. GPU hardware detection (platform-independent) ──
    console.log("");
    let gpuDetected = false;

    // 2a. PCIe — works on native Linux and WSL2
    const lspci = sh("lspci 2>/dev/null | grep -i 'vga\\|display\\|3d'");
    if (lspci) {
      console.log("PCI GPUs:");
      for (const line of lspci.split("\n")) console.log(`  ${line.trim()}`);
      gpuDetected = lspci.toLowerCase().includes("amd") || lspci.toLowerCase().includes("radeon");
    } else {
      console.log("PCI GPUs:      (lspci not available)");
    }

    // 2b. DRM render nodes + /dev/dxg
    const driNodes = sh("ls /dev/dri/ 2>/dev/null");
    const hasRenderNode = driNodes.includes("renderD");
    const hasDxg = existsSync("/dev/dxg");
    console.log(`/dev/dri/:     ${driNodes ? driNodes.replace(/\n/g, ", ") : "NOT FOUND"}`);
    if (hasDxg) console.log(`/dev/dxg:      present (DirectX GPU paravirtualization)`);

    // 2c. Find the AMD GPU card in sysfs (skip iGPUs / non-AMD cards)
    // Prefer card with vendor 0x1002 (AMD); fall back to first card if none match
    const amdCard = sh("for c in /sys/class/drm/card[0-9]; do [ \"$(cat $c/device/vendor 2>/dev/null)\" = '0x1002' ] && echo $c && break; done")
      || sh("for c in /sys/class/drm/card[0-9]; do [ -e $c/device/vendor ] && echo $c && break; done");

    if (hasRenderNode && amdCard) {
      const drmDriver = sh(`basename $(readlink -f ${amdCard}/device/driver) 2>/dev/null`)
        || (hasDxg ? "dxg" : "unknown");
      console.log(`  DRM driver:  ${drmDriver}`);
      if (drmDriver === "amdgpu") {
        console.log(`  Redline:     COMPATIBLE (libdrm_amdgpu path available)`);
      } else if (drmDriver === "dxg" || (isWsl && drmDriver !== "amdgpu")) {
        console.log(`  Redline:     NOT AVAILABLE (GPU-PV, not native amdgpu driver)`);
      }
    }

    // 2e. /dev/kfd (ROCm Kernel Fusion Driver)
    const hasKfd = existsSync("/dev/kfd");
    const kfdReadable = hasKfd && sh("test -r /dev/kfd && echo yes") === "yes";
    console.log(`/dev/kfd:      ${hasKfd ? (kfdReadable ? "present, readable" : "present, NOT READABLE (permission denied)") : "NOT FOUND"}`);

    // 2f. sysfs GPU info (from the AMD card we found, not just the first)
    const vendor = amdCard ? sh(`cat ${amdCard}/device/vendor 2>/dev/null`) : "";
    const device = amdCard ? sh(`cat ${amdCard}/device/device 2>/dev/null`) : "";
    if (vendor) console.log(`  vendor:      ${vendor}${vendor === "0x1002" ? " (AMD)" : vendor === "0x10de" ? " (NVIDIA — not supported)" : ""}`);
    if (device) console.log(`  device:      ${device}`);

    // 2g. amdgpu kernel module
    const amdgpuLoaded = sh("lsmod 2>/dev/null | grep amdgpu | head -1");
    console.log(`amdgpu module: ${amdgpuLoaded ? "loaded" : "NOT LOADED"}`);

    // ── 3. ROCm / HIP runtime ──────────────────────────────
    console.log("");
    const hipccVer = sh("hipcc --version 2>&1 | head -3");
    const rocminfoGpu = sh("rocminfo 2>/dev/null | grep -E 'Name:.*gfx|Marketing'");
    const hipConfig = sh("hipconfig --full 2>/dev/null | head -5");
    console.log(`hipcc:         ${hipccVer ? hipccVer.split("\n")[0] : "NOT FOUND"}`);
    if (rocminfoGpu) {
      console.log("rocminfo GPUs:");
      for (const line of rocminfoGpu.split("\n").slice(0, 4)) console.log(`  ${line.trim()}`);
    } else {
      console.log(`rocminfo:      ${sh("which rocminfo 2>/dev/null") ? "installed but no GPUs detected" : "NOT FOUND"}`);
    }

    // ── 4. Daemon binary + models ──────────────────────────
    console.log("");
    const exe2 = process.platform === "win32" ? ".exe" : "";
    const daemonBins = [
      resolve(__dirname, `../target/release/examples/daemon${exe2}`),
      join(HIPFIRE_DIR, "bin", `daemon${exe2}`),
    ];
    const daemonBin = daemonBins.find(p => existsSync(p));
    console.log(`daemon:        ${daemonBin ? "found" : "NOT FOUND — run: hipfire update"}`);

    const models = listLocal();
    console.log(`local models:  ${models.length}`);
    for (const m of models) console.log(`  ${m.name.padEnd(35)} ${m.size.padStart(6)}`);

    // 5. Pre-compiled kernels
    const binDir2 = join(HIPFIRE_DIR, "bin");
    const kernelBase = join(binDir2, "kernels", "compiled");
    const cwdKernelBase = resolve(__dirname, "../kernels/compiled");
    const kBase = existsSync(kernelBase) ? kernelBase : existsSync(cwdKernelBase) ? cwdKernelBase : null;
    if (kBase) {
      const arches = readdirSync(kBase).filter(d => d.startsWith("gfx"));
      for (const arch of arches) {
        const dir = join(kBase, arch);
        const hsaco = readdirSync(dir).filter(f => f.endsWith(".hsaco")).length;
        const hashes = readdirSync(dir).filter(f => f.endsWith(".hash")).length;
        console.log(`kernels/${arch}: ${hsaco} blobs, ${hashes} hashes${hashes < hsaco ? " (run: hipfire update)" : ""}`);
      }
    } else {
      console.log("kernels:       NOT FOUND");
    }

    // ── 6. Live GPU probe via daemon ───────────────────────
    if (daemonBin) {
      console.log("\nProbing GPU via HIP runtime...");
      try {
        const de = new Engine();
        await de.start();
        await de.send({ type: "ping" }); await de.recv();
        await de.send({ type: "diag" });
        const diag = await de.recv();
        if (diag.type === "diag") {
          console.log(`  GPU arch:    ${diag.arch}`);
          console.log(`  HIP version: ${diag.hip_version}`);
          console.log(`  VRAM free:   ${diag.vram_free_mb} MB`);
          console.log(`  VRAM total:  ${diag.vram_total_mb} MB`);

          const vram = diag.vram_total_mb;
          if (models.length === 0 && vram > 0) {
            const rec = vram < 4000 ? "qwen3.5:0.8b" : vram < 6000 ? "qwen3.5:4b" : vram < 16000 ? "qwen3.5:9b" : vram < 24000 ? "qwen3.5:27b" : "qwen3.5:27b-hf6";
            console.log(`\nTIP: No models downloaded. Run: hipfire pull ${rec}`);
          }
        } else {
          console.log(`  Error: ${diag.message || "unexpected response"}`);
        }
        await de.stop();
      } catch (err: any) {
        console.log(`  HIP probe failed: ${err.message}`);
        // Give actionable guidance based on what we found above
        if (isWindows) {
          console.log("\n  hipfire requires Linux. On Windows, use WSL2:");
          console.log("    1. Install WSL2: wsl --install -d Ubuntu");
          console.log("    2. Install ROCm in WSL2: https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html");
          console.log("    3. Install hipfire inside WSL2");
        } else if (isWsl) {
          if (!hasKfd && !hasRenderNode) {
            console.log("\n  No GPU device nodes found in WSL2.");
            console.log("  Install the AMD GPU driver for WSL2:");
            console.log("    sudo amdgpu-install --usecase=wsl");
            console.log("  If amdgpu-install is not available, install ROCm:");
            console.log("    https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html");
            console.log("  Note: ROCm WSL2 support requires a compatible AMD GPU and recent Windows drivers.");
          } else if (hasRenderNode && !hasKfd) {
            console.log("\n  /dev/dri found but /dev/kfd missing. ROCm may not be installed:");
            console.log("    sudo amdgpu-install --usecase=wsl");
          } else if (hasKfd) {
            console.log("\n  /dev/kfd found but HIP can't see GPU. Try:");
            console.log("    1. Verify ROCm version matches your GPU: apt list --installed | grep rocm");
            console.log("    2. Check permissions: ls -la /dev/kfd /dev/dri/renderD*");
            console.log("    3. Add user to render group: sudo usermod -aG render $USER");
          }
        } else {
          if (!amdgpuLoaded) {
            console.log("\n  amdgpu kernel module not loaded. Check:");
            console.log("    1. dmesg | grep -i amdgpu");
            console.log("    2. Is this an AMD GPU? (NVIDIA GPUs are not supported)");
          } else if (!hasKfd) {
            console.log("\n  amdgpu loaded but /dev/kfd missing. Install ROCm:");
            console.log("    https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html");
          } else if (!kfdReadable) {
            console.log("\n  /dev/kfd not readable. Fix permissions:");
            console.log("    sudo usermod -aG render $USER && newgrp render");
          }
        }
      }
    }

    // ── 7. Config ──────────────────────────────────────────
    console.log(`\nconfig:        ${CONFIG_PATH}`);
    for (const k of Object.keys(CONFIG_DEFAULTS) as (keyof HipfireConfig)[]) {
      const v = cfg[k];
      if (v !== CONFIG_DEFAULTS[k]) console.log(`  ${k} = ${v}`);
    }

    console.log("\nDone.");
    break;
  }
  case "bench": {
    const exp = rest.includes("--exp");
    const runsIdx = rest.indexOf("--runs");
    const runs = runsIdx >= 0 && runsIdx + 1 < rest.length ? parseInt(rest[runsIdx + 1]) : 5;
    if (isNaN(runs) || runs < 1) { console.error("Error: --runs must be a positive integer"); process.exit(1); }
    // Filter out flags to find model and prompt
    const skipSet = new Set<number>();
    if (exp) skipSet.add(rest.indexOf("--exp"));
    if (runsIdx >= 0) { skipSet.add(runsIdx); skipSet.add(runsIdx + 1); }
    const positional = rest.filter((_, i) => !skipSet.has(i));
    const benchModel = positional[0];
    if (!benchModel) {
      console.error(`Usage: hipfire bench <model> [--exp] [--runs N] [prompt]

  Standard benchmark: measure decode + prefill tok/s over N runs.
  --exp    RDNA2 only: test all 5 kernel variants (occupancy/unroll/cache tradeoffs)
  --runs   Number of runs per variant (default: 5)

Examples:
  hipfire bench qwen3.5:4b
  hipfire bench qwen3.5:9b --runs 3
  hipfire bench --exp qwen3.5:4b --runs 5`);
      process.exit(1);
    }
    const benchPrompt = positional.slice(1).join(" ") || "Explain the theory of general relativity in simple terms.";
    await bench(benchModel, runs, exp, benchPrompt);
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
  case "config": {
    const [action, key, value] = rest;
    const validKeys = Object.keys(CONFIG_DEFAULTS) as (keyof HipfireConfig)[];

    if (!action || action === "list") {
      // Show all config values, marking non-default ones
      console.log(`Config: ${CONFIG_PATH}\n`);
      for (const k of validKeys) {
        const v = cfg[k];
        const isDefault = v === CONFIG_DEFAULTS[k];
        console.log(`  ${k.padEnd(18)} ${String(v).padEnd(14)}${isDefault ? "(default)" : ""}`);
      }
      console.log(`\nSet:   hipfire config set <key> <value>`);
      console.log(`Reset: hipfire config reset [key]`);
    } else if (action === "get") {
      if (!key) { console.error("Usage: hipfire config get <key>"); process.exit(1); }
      if (!validKeys.includes(key as any)) { console.error(`Unknown key: ${key}\nValid keys: ${validKeys.join(", ")}`); process.exit(1); }
      console.log(cfg[key as keyof HipfireConfig]);
    } else if (action === "set") {
      if (!key || value === undefined) { console.error("Usage: hipfire config set <key> <value>\n\nKeys:\n" + validKeys.map(k => `  ${k.padEnd(18)} (default: ${CONFIG_DEFAULTS[k]})`).join("\n")); process.exit(1); }
      if (!validKeys.includes(key as any)) { console.error(`Unknown key: ${key}\nValid keys: ${validKeys.join(", ")}`); process.exit(1); }
      const defaultVal = CONFIG_DEFAULTS[key as keyof HipfireConfig];
      const parsed = typeof defaultVal === "number" ? Number(value) : value;
      if (typeof defaultVal === "number" && isNaN(parsed as number)) { console.error(`${key} requires a number`); process.exit(1); }
      if (!validateConfigValue(key, parsed)) {
        const hints: Record<string, string> = {
          kv_cache: "one of: q8, turbo2, turbo3, turbo4",
          temperature: "number between 0 and 2",
          top_p: "number in (0, 1]",
          repeat_penalty: "number between 1.0 and 3.0",
          max_tokens: "integer between 1 and 32768",
          port: "integer between 1 and 65535",
          default_model: "non-empty model tag",
        };
        console.error(`${key} must be ${hints[key] || "valid"}`); process.exit(1);
      }
      (cfg as any)[key] = parsed;
      saveConfig(cfg);
      console.log(`${key} = ${parsed}`);
    } else if (action === "reset") {
      if (key) {
        if (!validKeys.includes(key as any)) { console.error(`Unknown key: ${key}`); process.exit(1); }
        (cfg as any)[key] = CONFIG_DEFAULTS[key as keyof HipfireConfig];
        saveConfig(cfg);
        console.log(`${key} reset to ${CONFIG_DEFAULTS[key as keyof HipfireConfig]}`);
      } else {
        saveConfig({ ...CONFIG_DEFAULTS });
        console.log("All config reset to defaults");
      }
    } else {
      console.error("Usage: hipfire config [list|get|set|reset]");
    }
    break;
  }
  default:
    console.log(`hipfire — LLM inference for AMD GPUs

  pull <model>          Download model from HuggingFace
  run <model> [prompt]  Generate text (auto-pulls if needed)
  serve [port]          Start OpenAI-compatible server (default: ${cfg.port})
  bench <model> [opts]  Benchmark tok/s (--exp for RDNA2 variant sweep, --runs N)
  profile [model]       Kernel efficiency profiler (--json, --kernel <name>)
  list [-r]             Show local models (-r: show available too)
  config [list|set|get|reset]  Persistent settings (kv_cache, temperature, etc.)
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
