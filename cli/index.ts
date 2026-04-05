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
  const warmPath = findModel(defaultModel);
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

        // Build conversation as a single prompt preserving message order.
        // Skip the system message (handled separately), render everything else in order.
        const convParts: string[] = [];
        for (const m of messages) {
          if (m.role === "system") continue;
          if (m.role === "tool") {
            convParts.push(`<tool_response>\n${m.content}\n</tool_response>`);
          } else if (m.role === "assistant" && m.tool_calls) {
            let text = m.content || "";
            for (const tc of m.tool_calls) {
              const fn = tc.function || tc;
              text += `\n<tool_call>\n${JSON.stringify({ name: fn.name, arguments: JSON.parse(fn.arguments || "{}") })}\n</tool_call>`;
            }
            convParts.push(text);
          } else {
            convParts.push(m.content || "");
          }
        }
        userPrompt = convParts.join("\n");

        const path = findModel(body.model || "default");
        if (!path) { safeRelease(); return Response.json({ error: "model not found" }, { status: 404 }); }

        if (current !== path) {
          if (current) { await e.send({ type: "unload" }); await e.recv(); }
          await e.send({ type: "load", model: path, turbo: turboMode }); await e.recv();
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
          return new Response(new ReadableStream({
            async start(ctrl) {
              try {
                let inThink = false;
                let stripNextLeadingNl = false;
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
                    ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                      id: reqId, object: "chat.completion.chunk", created, model: modelName,
                      choices: [{ index: 0, delta: { content: text }, finish_reason: null }]
                    })}\n\n`));
                  } else if (msg.type === "done") {
                    ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                      id: reqId, object: "chat.completion.chunk", created, model: modelName,
                      choices: [{ index: 0, delta: {}, finish_reason: "stop" }]
                    })}\n\n`));
                    ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
                    ctrl.close();
                  }
                }
              } finally { safeRelease(); }
            },
            cancel() { streamCancelled = true; } // lock released in finally after generation drains
          }), { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
        }

        let content = "";
        let completionTokens = 0;
        for await (const msg of e.generate(genParams)) {
          if (msg.type === "token") { content += msg.text; completionTokens++; }
        }

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
