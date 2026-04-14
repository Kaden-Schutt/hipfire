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
  kv_cache: string;       // "auto" (per-arch default), "q8", "asym4", "asym3", "asym2"
  flash_mode: string;     // "auto" (ctx-gated), "always", "never" — only affects Q8 path
  default_model: string;  // model tag for serve pre-warm, e.g. "qwen3.5:9b"
  temperature: number;    // default temperature for run
  top_p: number;
  repeat_penalty: number;
  max_tokens: number;     // per-turn generation cap
  max_seq: number;        // KV cache capacity allocated at model load (shared across turns)
  thinking: string;       // "on" (model reasons in <think>, stripped from display) | "off" (suppress thinking)
  max_think_tokens: number; // per-turn budget for <think>...</think> reasoning (0 = unlimited)
  port: number;           // default serve port
  idle_timeout: number;   // serve: seconds of inactivity before unloading the model (0 = never)
  // ── Experimental / research knobs (OFF by default, no stable contract) ──
  // Gates the daemon's `budget_alert_at_tok` + `budget_alert_text` generate
  // params. When false (default), the daemon ignores those params entirely.
  // Research-only feature: in-band nudges to the model's own think stream,
  // which CAN leak into visible output if the client doesn't also constrain
  // when the alert fires (e.g. injecting past </think>). Only enable if you
  // understand the knob. See docs/MULTI_MODEL_QUEUE.md path for where this
  // lives in the roadmap.
  experimental_budget_alert: boolean;
}

// Detect GPU at import time for smart defaults
const DETECTED_ARCH = detectGpuArch();
const ARCH_DEFAULTS = archDefaults(DETECTED_ARCH);

const CONFIG_DEFAULTS: HipfireConfig = {
  kv_cache: ARCH_DEFAULTS.kv_cache,
  flash_mode: "auto",
  default_model: "qwen3.5:9b",
  temperature: 0.3,
  top_p: 0.8,
  // 1.05 is the minimum penalty that prevents short-range loops without
  // pushing greedy/low-temperature outputs off-manifold. 1.3 (Ollama-ish)
  // causes MQ4/MQ6 models to emit gibberish at temp=0 because the penalty
  // applies uniformly even in greedy mode. 1.05 is user-validated.
  repeat_penalty: 1.05,
  max_tokens: 512,
  max_seq: 32768,
  thinking: "on",
  max_think_tokens: 0,
  port: DEFAULT_PORT,
  idle_timeout: 300,
  experimental_budget_alert: false,
};

function validateConfigValue(key: string, value: any): boolean {
  switch (key) {
    case "kv_cache": return ["auto", "q8", "asym4", "asym3", "asym2", "turbo", "turbo4", "turbo3", "turbo2"].includes(value);
    case "flash_mode": return ["auto", "always", "never"].includes(value);
    case "temperature": return typeof value === "number" && value >= 0 && value <= 2;
    case "top_p": return typeof value === "number" && value > 0 && value <= 1;
    case "repeat_penalty": return typeof value === "number" && value >= 1 && value <= 3;
    case "max_tokens": return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 131072;
    case "max_seq": return typeof value === "number" && Number.isInteger(value) && value >= 512 && value <= 524288;
    case "thinking": return ["on", "off"].includes(value);
    case "max_think_tokens": return typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 32768;
    case "port": return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 65535;
    case "idle_timeout": return typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 86400;
    case "default_model": return typeof value === "string" && value.trim().length > 0;
    case "experimental_budget_alert": return typeof value === "boolean";
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

// ─── Per-model config overlays ──────────────────────────
// Sparse per-tag overrides. Stored in ~/.hipfire/per_model_config.json.
// Resolution order: --flag > per-model > global > engine fallback.

const PER_MODEL_CONFIG_PATH = join(HIPFIRE_DIR, "per_model_config.json");

// Fields that make sense to override per-model. port + idle_timeout + default_model
// are serve-wide so they stay global-only.
const PER_MODEL_KEYS = [
  "kv_cache", "flash_mode", "temperature", "top_p",
  "repeat_penalty", "max_tokens", "max_seq", "thinking", "max_think_tokens",
] as const;
type PerModelKey = typeof PER_MODEL_KEYS[number];

type PerModelOverride = Partial<Pick<HipfireConfig, PerModelKey>>;
type PerModelConfigs = Record<string, PerModelOverride>;

function loadPerModelConfigs(): PerModelConfigs {
  try {
    const raw = JSON.parse(require("fs").readFileSync(PER_MODEL_CONFIG_PATH, "utf-8"));
    const out: PerModelConfigs = {};
    for (const [tag, ov] of Object.entries(raw ?? {})) {
      const clean: PerModelOverride = {};
      for (const k of PER_MODEL_KEYS) {
        const v = (ov as any)?.[k];
        if (v !== undefined && validateConfigValue(k, v)) (clean as any)[k] = v;
      }
      if (Object.keys(clean).length > 0) out[tag] = clean;
    }
    return out;
  } catch { return {}; }
}

function savePerModelConfigs(all: PerModelConfigs) {
  // Drop empty entries so the file stays minimal
  const clean: PerModelConfigs = {};
  for (const [tag, ov] of Object.entries(all)) {
    if (Object.keys(ov).length > 0) clean[tag] = ov;
  }
  require("fs").writeFileSync(PER_MODEL_CONFIG_PATH, JSON.stringify(clean, null, 2) + "\n");
}

// Return the effective config for a given model tag. Per-model overrides
// win over global. If tag is null/undefined, returns the global config.
// Reads the global config fresh each call so edits via `hipfire config set`
// take effect without restarting a running `hipfire serve`.
function resolveModelConfig(tag: string | null | undefined): HipfireConfig {
  const base = loadConfig();
  if (!tag) return base;
  const resolved = resolveModelTag(tag);
  const overrides = loadPerModelConfigs()[resolved] ?? loadPerModelConfigs()[tag] ?? {};
  return { ...base, ...overrides };
}

// thinking: "off" prepends a directive to the system prompt asking the model
// to skip <think> reasoning. Effective on instruction-tuned models (Qwen 3.5
// in particular honors this); advisory-only — for hard suppression we'd need
// a daemon-side <think></think> bypass injection.
function applyThinkingMode(systemPrompt: string | undefined, thinking: string): string | undefined {
  if (thinking !== "off") return systemPrompt;
  const directive = "Respond directly without using <think>...</think> reasoning blocks. Give the final answer only.";
  return systemPrompt ? `${directive}\n\n${systemPrompt}` : directive;
}

// Build the {type: "load", ...} message for the daemon, carrying per-model
// params (max_seq). The tag is optional — pass it from the caller when known,
// else we fall back to global cfg.
function buildLoadMessage(path: string, tag?: string | null): any {
  const resolved = resolveModelConfig(tag);
  // Guard: the KV cache must be big enough to hold at least one max_tokens
  // response plus a little prompt headroom; otherwise the daemon panics mid-
  // generation. Auto-bump rather than crash.
  const minViable = resolved.max_tokens + 1024;
  const max_seq = Math.max(resolved.max_seq, minViable);
  if (max_seq > resolved.max_seq) {
    console.error(`[hipfire] note: max_seq (${resolved.max_seq}) < max_tokens (${resolved.max_tokens}) + 1024 — bumping to ${max_seq} for this load`);
  }
  return { type: "load", model: path, params: { max_seq } };
}

// ─── Model Registry ─────────────────────────────────────
// Maps "name:tag" → { repo, file, size_gb, min_vram_gb }
// Default tag (no quant suffix) = MQ4 (FWHT-rotated 4-bit, WMMA-accelerated on RDNA3+)

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
  // Qwen3.5 MagnumQuant MQ4 — rotated 4-bit with quality gate.
  // Default format: best quality-per-bit, WMMA-accelerated prefill on RDNA3.
  // Perf: RX 7900 XTX (gfx1100), decode tok/s / prefill tok/s with WMMA.
  "qwen3.5:0.8b":     { repo: hfRepo("qwen3.5","0.8b"), file: "qwen3.5-0.8b.mq4",   size_gb: 0.55, min_vram_gb: 1,  desc: "386 / 5100 tok/s" },
  "qwen3.5:4b":       { repo: hfRepo("qwen3.5","4b"),   file: "qwen3.5-4b.mq4",     size_gb: 2.6,  min_vram_gb: 4,  desc: "169 / 1900 tok/s" },
  "qwen3.5:9b":       { repo: hfRepo("qwen3.5","9b"),   file: "qwen3.5-9b.mq4",     size_gb: 5.3,  min_vram_gb: 6,  desc: "125 / 1720 tok/s" },
  "qwen3.5:27b":      { repo: hfRepo("qwen3.5","27b"),  file: "qwen3.5-27b.mq4",    size_gb: 15.0, min_vram_gb: 16, desc: "45 / 489 tok/s, 16GB+" },

  // Qwen3.5 MQ6 — 6-bit rotated, higher quality / larger file (~1.47× MQ4)
  "qwen3.5:0.8b-mq6": { repo: hfRepo("qwen3.5","0.8b"), file: "qwen3.5-0.8b.mq6",   size_gb: 0.67, min_vram_gb: 2,  desc: "MQ6, higher quality" },
  "qwen3.5:4b-mq6":   { repo: hfRepo("qwen3.5","4b"),   file: "qwen3.5-4b.mq6",     size_gb: 3.5,  min_vram_gb: 5,  desc: "MQ6, higher quality" },
  "qwen3.5:9b-mq6":   { repo: hfRepo("qwen3.5","9b"),   file: "qwen3.5-9b.mq6",     size_gb: 7.3,  min_vram_gb: 8,  desc: "MQ6, higher quality" },
  "qwen3.5:27b-mq6":  { repo: hfRepo("qwen3.5","27b"),  file: "qwen3.5-27b.mq6",    size_gb: 21.4, min_vram_gb: 24, desc: "MQ6, higher quality" },

  // Qwen3 (standard attention, not DeltaNet)
  "qwen3:0.6b":       { repo: hfRepo("qwen3","0.6b"),   file: "qwen3-0.6b.hf4",     size_gb: 0.4,  min_vram_gb: 1,  desc: "standard attention" },
  "qwen3:8b":         { repo: hfRepo("qwen3","8b"),     file: "qwen3-8b.hf4",       size_gb: 4.1,  min_vram_gb: 6,  desc: "60 tok/s, standard attention" },

  // Community finetunes (Qwen3.5 architecture)
  "carnice:9b":        { repo: "schuttdev/hipfire-carnice-9b",  file: "carnice-9b.mq4",  size_gb: 5.0, min_vram_gb: 6,  desc: "Hermes tool-use, MQ4" },
  "carnice:27b":       { repo: "schuttdev/hipfire-carnice-27b", file: "carnice-27b.mq4", size_gb: 15.0, min_vram_gb: 16, desc: "Hermes 27B tool-use, MQ4" },
  "carnice:9b-mq6":    { repo: "schuttdev/hipfire-carnice-9b",  file: "carnice-9b.mq6",  size_gb: 7.3, min_vram_gb: 8,  desc: "Hermes tool-use, MQ6 higher quality" },
  "carnice:27b-mq6":   { repo: "schuttdev/hipfire-carnice-27b", file: "carnice-27b.mq6", size_gb: 21.4, min_vram_gb: 24, desc: "Hermes 27B, MQ6 higher quality" },
  // Qwopus 3.5 v3 — Jackrong fine-tune (reasoning, CoT, competitive programming)
  "qwopus:9b":         { repo: "schuttdev/hipfire-qwopus-9b", file: "qwopus-9b.mq4",  size_gb: 5.3, min_vram_gb: 6,  desc: "Qwopus3.5 v3, MQ4" },
  "qwopus:4b":         { repo: "schuttdev/hipfire-qwopus-4b", file: "qwopus-4b.mq4",  size_gb: 2.6, min_vram_gb: 4,  desc: "Qwopus3.5 v3, 4B MQ4" },
  "qwopus:27b":        { repo: "schuttdev/hipfire-qwopus-27b", file: "qwopus-27b.mq4", size_gb: 15.0, min_vram_gb: 16, desc: "Qwopus3.5 v3, 27B MQ4" },
  "qwopus:9b-mq6":     { repo: "schuttdev/hipfire-qwopus-9b", file: "qwopus-9b.mq6",  size_gb: 7.3, min_vram_gb: 8,  desc: "Qwopus3.5 v3, MQ6 higher quality" },
  "qwopus:4b-mq6":     { repo: "schuttdev/hipfire-qwopus-4b", file: "qwopus-4b.mq6",  size_gb: 3.8, min_vram_gb: 5,  desc: "Qwopus3.5 v3, 4B MQ6" },
  "qwopus:27b-mq6":    { repo: "schuttdev/hipfire-qwopus-27b", file: "qwopus-27b.mq6", size_gb: 21.4, min_vram_gb: 24, desc: "Qwopus3.5 v3, 27B MQ6" },
};

// Aliases (also map retired hf4/hf6/mq4 tags to current names)
const ALIASES: Record<string, string> = {
  "qwen3.5": "qwen3.5:4b",
  "qwen3.5:latest": "qwen3.5:9b",
  "qwen3.5:small": "qwen3.5:0.8b",
  "qwen3.5:large": "qwen3.5:27b",
  "qwen3": "qwen3:8b",
  "carnice": "carnice:9b",
  "qwopus": "qwopus:9b",
  // Retired format tags → current MQ4
  "qwen3.5:0.8b-mq4": "qwen3.5:0.8b", "qwen3.5:4b-mq4": "qwen3.5:4b",
  "qwen3.5:9b-mq4": "qwen3.5:9b", "qwen3.5:27b-mq4": "qwen3.5:27b",
  "qwen3.5:0.8b-hf4": "qwen3.5:0.8b", "qwen3.5:2b-hf4": "qwen3.5:4b",
  "qwen3.5:4b-hf4": "qwen3.5:4b", "qwen3.5:9b-hf4": "qwen3.5:9b",
  "qwen3.5:27b-hf4": "qwen3.5:27b",
  "qwen3.5:0.8b-hf6": "qwen3.5:0.8b-mq6", "qwen3.5:2b-hf6": "qwen3.5:4b-mq6",
  "qwen3.5:4b-hf6": "qwen3.5:4b-mq6", "qwen3.5:9b-hf6": "qwen3.5:9b-mq6",
  "qwen3.5:27b-hf6": "qwen3.5:27b-mq6",
  // Qwopus: old hf4 tags → new mq4
  "qwopus:9b-hf4": "qwopus:9b", "qwopus:4b-hf4": "qwopus:4b", "qwopus:27b-hf4": "qwopus:27b",
  "qwopus:9b-mq4": "qwopus:9b", "qwopus:4b-mq4": "qwopus:4b", "qwopus:27b-mq4": "qwopus:27b",
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

// ─── GPU arch detection + per-arch defaults ──────────────
function detectGpuArch(): string {
  // Read KFD sysfs for GPU arch (same as install command)
  for (const node of ["1", "0"]) {
    try {
      const props = require("fs").readFileSync(`/sys/class/kfd/kfd/topology/nodes/${node}/properties`, "utf8");
      const m = props.match(/gfx_target_version\s+(\d+)/);
      if (m) {
        const ver = parseInt(m[1]);
        const major = Math.floor(ver / 10000);
        const minor = Math.floor((ver % 10000) / 100);
        const step = ver % 100;
        let arch = `gfx${major}${minor.toString().padStart(2, '0')}${step || '0'}`;
        return arch.replace(/^(gfx\d{4})0$/, '$1');
      }
    } catch {}
  }
  return "unknown";
}

interface ArchDefaults {
  kv_cache: string;        // best KV mode for this hardware
  vram_gb: number;         // approximate VRAM
}

function archDefaults(arch: string): ArchDefaults {
  // Default KV cache policy (RotorQuant asymmetric):
  //   asym3 (K 3-bit rotated + V Q8) is the default across arches — 5.5×
  //   compression vs fp32 with verbatim rare-token recall on head_dim=256
  //   models (Qwen 3.5 family). Memory-tight cards get asym2 (6.0×, still
  //   recall-safe for common tokens). Users can override to `q8` for
  //   maximum quality or `asym4` for extra K precision headroom.
  switch (arch) {
    // RDNA3 — asym3 everywhere; 24 GB cards fit full context easily.
    case "gfx1100": return { kv_cache: "asym3", vram_gb: 24 };  // 7900 XTX
    case "gfx1101": return { kv_cache: "asym3", vram_gb: 16 };  // 7900 XT
    case "gfx1102": return { kv_cache: "asym3", vram_gb: 12 };  // 7800 XT
    case "gfx1151": return { kv_cache: "asym2", vram_gb: 16 };  // Strix Halo APU (shared mem — tight)
    // RDNA4
    case "gfx1200": case "gfx1201":
      return { kv_cache: "asym3", vram_gb: 16 };                // 9070 XT
    // RDNA2
    case "gfx1030": return { kv_cache: "asym3", vram_gb: 32 };  // V620 (32 GB — plenty of headroom)
    case "gfx1031": return { kv_cache: "asym3", vram_gb: 12 };  // 6700 XT
    case "gfx1032": return { kv_cache: "asym2", vram_gb: 8 };   // 6600 XT (8 GB — asym2 for headroom)
    // RDNA1
    case "gfx1010": return { kv_cache: "asym2", vram_gb: 8 };   // 5700 XT
    case "gfx1013": return { kv_cache: "asym2", vram_gb: 14 };  // BC-250 APU
    // Fallback — unknown arch, asym3 is the new safe default.
    default: return { kv_cache: "asym3", vram_gb: 8 };
  }
}

// ─── KV cache mode resolver ──────────────────────────────
// Canonical modes: q8, asym4, asym3, asym2.
// Legacy aliases: turbo→asym3, turbo2→asym2, turbo3→asym3, turbo4→asym4
// (plus "auto" → arch default).
function resolveKvMode(cfg: HipfireConfig): string {
  const raw = process.env.HIPFIRE_KV_MODE || cfg.kv_cache;
  if (raw === "auto") return ARCH_DEFAULTS.kv_cache;
  if (raw === "turbo" || raw === "turbo3") return "asym3";
  if (raw === "turbo2") return "asym2";
  if (raw === "turbo4") return "asym4";
  return raw;
}

// Set all config-driven env vars in one place so every daemon-spawning
// codepath picks up the user's current settings consistently.
// Called before `new Engine().start()`.
function applyConfigEnv(cfg: HipfireConfig): void {
  process.env.HIPFIRE_KV_MODE = resolveKvMode(cfg);
  // Only set HIPFIRE_ATTN_FLASH if the user hasn't already set it in their
  // shell (env overrides config). `auto` is the engine default — skip the
  // env var in that case so the engine's own default applies.
  if (!process.env.HIPFIRE_ATTN_FLASH) {
    if (cfg.flash_mode === "always" || cfg.flash_mode === "never") {
      process.env.HIPFIRE_ATTN_FLASH = cfg.flash_mode;
    }
  }
  // Experimental budget-alert gate. The daemon reads this env var on every
  // generate request; if not set to "1", it refuses `budget_alert_at_tok`
  // even if a client passes it. Keeps an unstable research feature from
  // leaking into real responses via misconfigured callers. Setting cleanly
  // (no env → unset) matters because this is the signed gate.
  if (cfg.experimental_budget_alert) {
    process.env.HIPFIRE_EXPERIMENTAL_BUDGET_ALERT = "1";
  } else {
    delete process.env.HIPFIRE_EXPERIMENTAL_BUDGET_ALERT;
  }
}

// ─── Background serve lifecycle ─────────────────────────
// `hipfire serve -d` forks to background; `hipfire stop` kills it.
// `hipfire run` auto-detects and uses a running serve via HTTP.

const SERVE_PID_FILE = join(HIPFIRE_DIR, "serve.pid");
const SERVE_LOG_FILE = join(HIPFIRE_DIR, "serve.log");

function isPidAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

function readServePid(): number | null {
  try {
    const raw = require("fs").readFileSync(SERVE_PID_FILE, "utf-8").trim();
    const pid = parseInt(raw, 10);
    if (!pid || !isPidAlive(pid)) return null;
    return pid;
  } catch { return null; }
}

// Cheap liveness probe: 500ms health check. Used by `run` to decide HTTP vs local spawn.
async function isServeUp(port: number): Promise<boolean> {
  try {
    const ctl = AbortSignal.timeout(500);
    const r = await fetch(`http://127.0.0.1:${port}/health`, { signal: ctl });
    return r.ok;
  } catch { return false; }
}

// Drive `hipfire run` through an existing serve's /v1/chat/completions stream.
// Returns false if it couldn't connect (caller falls back to local spawn).
async function runViaHttp(
  port: number, model: string, prompt: string,
  image: string | undefined,
  temp: number, maxTokens: number, repeatPenalty: number, topP: number,
): Promise<boolean> {
  // VL flows go through the image-base64 path on the daemon which the HTTP
  // wrapper doesn't expose — fall back to local spawn.
  if (image) return false;

  const body: any = {
    model, stream: true,
    messages: [{ role: "user", content: prompt }],
    temperature: temp, max_tokens: maxTokens,
    repeat_penalty: repeatPenalty, top_p: topP,
  };

  let resp: Response;
  try {
    resp = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err: any) {
    console.error(`[hipfire] serve connection failed: ${err?.message ?? err} — falling back to local daemon`);
    return false;
  }
  if (!resp.ok) {
    const txt = await resp.text().catch(() => "");
    console.error(`[hipfire] serve returned HTTP ${resp.status}: ${txt.slice(0, 200)}`);
    return false;
  }
  if (!resp.body) { console.error("[hipfire] serve returned no body"); return false; }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let inThink = false;
  let stripNextLeadingNl = false;
  let tokens = 0;
  const t0 = Date.now();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") { buffer = ""; break; }
      try {
        const chunk = JSON.parse(data);
        // Top-level {"error":{...}} is how the serve surfaces daemon-side
        // rejections (e.g. KV-budget overrun). Print it and set a non-zero
        // exit code so `hipfire run` doesn't silently look successful.
        if (chunk.error) {
          process.stderr.write(`\n[hipfire] ${chunk.error.message || "server error"}\n`);
          process.exitCode = 1;
          continue;
        }
        const delta = chunk.choices?.[0]?.delta ?? {};
        let text: string = delta.content ?? "";
        if (!text) continue;
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
        process.stdout.write(text);
        tokens++;
      } catch {}
    }
  }
  const secs = (Date.now() - t0) / 1000;
  if (tokens > 0) console.error(`\n[${tokens} tok, ${(tokens / secs).toFixed(1)} tok/s via serve]`);
  return true;
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

    this.proc = spawn([bin], { stdin: "pipe", stdout: "pipe", stderr: "inherit", env: { ...process.env } });
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

  // Hint for 27B MQ4: suggest MQ6 for complex reasoning / coding when available
  if (resolved === "qwen3.5:27b" && REGISTRY["qwen3.5:27b-mq6"]) {
    console.error(`TIP: For coding/complex tasks, use: hipfire pull qwen3.5:27b-mq6 (needs 24GB VRAM)`);
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

  // If a serve daemon is already running on this port, proxy through its HTTP
  // API — saves the 2-5s cold-start cost of loading the model every invocation.
  // Local spawn falls through only when no serve is present (or HTTP errors out).
  const useLocal = process.env.HIPFIRE_LOCAL === "1" || image !== undefined;
  if (!useLocal && await isServeUp(cfg.port)) {
    const ok = await runViaHttp(cfg.port, model, prompt, image, temp, maxTokens, repeatPenalty, topP);
    if (ok) return;
    // runViaHttp logged its own failure reason; fall back to local spawn.
  }

  applyConfigEnv(cfg);
  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();
  await e.send(buildLoadMessage(path, model));
  const loaded = await e.recv();
  if (loaded.type === "error") { console.error(loaded.message); process.exit(1); }
  const vlTag = loaded.vl ? " VL" : "";
  console.error(`[${loaded.arch}${vlTag}] ${loaded.dim}d ${loaded.layers}L ${loaded.vocab} vocab`);

  if (image && !loaded.vl) {
    console.error(`WARNING: --image passed but model does not have a vision encoder. Ignoring image.`);
    image = undefined;
  }

  const modelCfg = resolveModelConfig(model);
  const thinkingDirective = applyThinkingMode(undefined, modelCfg.thinking);
  const genMsg: any = {
    type: "generate", id: "run", prompt,
    temperature: temp * TEMP_CORRECTION, max_tokens: maxTokens,
    repeat_penalty: repeatPenalty, top_p: topP,
  };
  if (thinkingDirective) genMsg.system = thinkingDirective;
  if (modelCfg.max_think_tokens > 0) genMsg.max_think_tokens = modelCfg.max_think_tokens;
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
    else if (msg.type === "error") {
      // Surface daemon-side rejections (e.g. KV-budget overrun) instead of
      // exiting 0 with no visible output. Sets exitCode so downstream shell
      // pipelines can detect the failure.
      process.stderr.write(`\n[hipfire] ${msg.message || "generation failed"}\n`);
      process.exitCode = 1;
      break;
    }
  }
  await e.stop();
}

async function serve(port: number) {
  applyConfigEnv(cfg);
  // Write the PID so `hipfire stop` / `hipfire ps` / `hipfire run` can find us.
  // Cleanup on normal exit; stale PID on crash is tolerated (isPidAlive catches it).
  try {
    require("fs").writeFileSync(SERVE_PID_FILE, String(process.pid));
  } catch {}
  const cleanupPid = () => { try { require("fs").unlinkSync(SERVE_PID_FILE); } catch {} };
  process.on("exit", cleanupPid);
  process.on("SIGTERM", () => { cleanupPid(); process.exit(0); });
  process.on("SIGINT", () => { cleanupPid(); process.exit(0); });

  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();
  let current: string | null = null;
  // Track the `max_seq` the currently-loaded model was loaded with, so we can
  // detect when a live `max_tokens` bump (via `hipfire config set max_tokens`
  // or a client-sent body.max_tokens) needs more headroom than the KV cache
  // was allocated for — and reload instead of letting the daemon overrun.
  let currentMaxSeq: number | null = null;

  // Idle eviction: after `idle_timeout` seconds of no requests, unload the
  // model to free VRAM. Next request reloads it (one-shot cost). 0 disables.
  let lastRequestTime = Date.now();
  const idleTimeoutMs = cfg.idle_timeout * 1000;
  const evictionInterval = idleTimeoutMs > 0 ? setInterval(async () => {
    if (!current) return;                              // nothing to unload
    if (Date.now() - lastRequestTime < idleTimeoutMs) return;
    try {
      console.error(`[hipfire] idle for ${cfg.idle_timeout}s — unloading model (VRAM freed; next request will reload)`);
      await e.send({ type: "unload" });
      await e.recv();
      current = null;
      currentMaxSeq = null;
    } catch (err: any) {
      console.error(`[hipfire] eviction failed: ${err?.message ?? err}`);
    }
  }, Math.min(60_000, idleTimeoutMs)) : null;
  // Keep process alive irrespective of the interval; clean up on exit.
  if (evictionInterval) process.on("exit", () => clearInterval(evictionInterval));

  // Pre-warm: load default model and compile kernels before accepting requests
  const defaultModel = process.env.HIPFIRE_MODEL || cfg.default_model;
  const rawWarmPath = findModel(defaultModel);
  const warmPath = rawWarmPath ? resolve(rawWarmPath) : null;
  if (warmPath) {
    try {
      console.error(`[hipfire] pre-warming ${defaultModel}...`);
      const warmLoadMsg = buildLoadMessage(warmPath, defaultModel);
      await e.send(warmLoadMsg);
      const loadResult = await e.recv();
      if (loadResult.type === "error") {
        console.error(`[hipfire] pre-warm load failed: ${loadResult.message} (will load on first request)`);
      } else {
        for await (const msg of e.generate({ type: "generate", id: "warmup", prompt: "Hi", temperature: 0, max_tokens: 1 })) {
          if (msg.type === "done") break;
        }
        await e.send({ type: "reset" }); await e.recv();
        current = warmPath;
        currentMaxSeq = warmLoadMsg.params.max_seq;
        console.error(`[hipfire] warm-up complete`);
      }
    } catch (err: any) {
      console.error(`[hipfire] pre-warm failed: ${err?.message} — restarting daemon`);
      current = null;
      currentMaxSeq = null;
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
      if (url.pathname === "/health") {
        return Response.json({
          status: "ok",
          model: current,
          idle_timeout_sec: cfg.idle_timeout,
          pid: process.pid,
        });
      }
      if (url.pathname === "/v1/models") return Response.json({ data: listLocal().map(m => ({ id: m.name })) });

      if (url.pathname !== "/v1/chat/completions" || req.method !== "POST")
        return Response.json({ error: "not found" }, { status: 404 });

      // Update idle timer on every real request (eviction loop checks against this).
      lastRequestTime = Date.now();

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
        currentMaxSeq = null;
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
        //
        // CRITICAL: the Qwen 3.5 chat template strips <think>...</think> from
        // HISTORICAL assistant messages (anything before the last user query).
        // Passing them through verbatim drags stale reasoning into the KV cache
        // and wrecks recall — the model treats the past thinking as current
        // context and drifts away from the user's actual facts. Strip thinking
        // blocks from every assistant message in the conversation history.
        const stripThinking = (s: string): string =>
          s.replace(/<think>[\s\S]*?<\/think>\s*/g, "")
           .replace(/<think>[\s\S]*$/, "");

        const nonSystem = messages.filter((m: any) => m.role !== "system");
        const convParts: string[] = [];
        for (let i = 0; i < nonSystem.length; i++) {
          const m = nonSystem[i];
          const role = m.role;
          let text = "";

          if (role === "tool") {
            text = `<tool_response>\n${m.content}\n</tool_response>`;
          } else if (role === "assistant") {
            text = stripThinking(m.content || "");
            if (m.tool_calls) {
              for (const tc of m.tool_calls) {
                const fn = tc.function || tc;
                text += `\n<tool_call>\n${JSON.stringify({ name: fn.name, arguments: JSON.parse(fn.arguments || "{}") })}\n</tool_call>`;
              }
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

        // Resolve effective config FIRST so we can size the KV cache against
        // the actual per-request max_tokens (body.max_tokens or config). The
        // daemon's KV buffers are sized at load time — if max_tokens grows
        // beyond currentMaxSeq we MUST reload instead of sending a request
        // the daemon would either reject or, worse, overrun the buffer with.
        const effective = resolveModelConfig(body.model);
        const requestMaxTokens = body.max_tokens ?? effective.max_tokens;
        const requiredMaxSeq = Math.max(effective.max_seq, requestMaxTokens + 1024);

        const needReload = current !== path
          || (currentMaxSeq !== null && requiredMaxSeq > currentMaxSeq);

        if (needReload) {
          if (current) { await e.send({ type: "unload" }); await e.recv(); }
          const loadMsg = buildLoadMessage(path, body.model);
          if (requiredMaxSeq > loadMsg.params.max_seq) {
            console.error(`[hipfire] request max_tokens=${requestMaxTokens} needs max_seq >= ${requiredMaxSeq} — bumping load (was ${loadMsg.params.max_seq})`);
            loadMsg.params.max_seq = requiredMaxSeq;
          }
          await e.send(loadMsg);
          const loadResult = await e.recv();
          if (loadResult.type === "error") {
            current = null;
            currentMaxSeq = null;
            safeRelease();
            return Response.json({ error: `model load failed: ${loadResult.message}` }, { status: 500 });
          }
          current = path;
          currentMaxSeq = loadMsg.params.max_seq;
        }

        const reqId = `chatcmpl-${Date.now().toString(36)}`;
        const created = Math.floor(Date.now() / 1000);
        const modelName = body.model || "hipfire";
        // Fall back to the user's configured defaults (global or per-model) when
        // an OpenAI client doesn't set a field. 512 was a hardcoded surprise
        // that ignored `hipfire config set max_tokens …`.
        const genParams: any = {
          type: "generate", id: reqId, prompt: userPrompt,
          temperature: (body.temperature ?? effective.temperature) * TEMP_CORRECTION,
          max_tokens: requestMaxTokens,
          repeat_penalty: body.repeat_penalty ?? (body.frequency_penalty != null ? 1.0 + body.frequency_penalty : effective.repeat_penalty),
          top_p: body.top_p ?? effective.top_p,
        };
        // Per-model thinking mode: prepend the "no-think" directive when
        // this model's override (or the global config) sets thinking=off.
        const thinkModel = resolveModelConfig(body.model).thinking;
        const resolvedSystem = applyThinkingMode(systemPrompt || undefined, thinkModel);
        if (resolvedSystem) genParams.system = resolvedSystem;

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
                    // Propagate daemon-side errors (e.g. KV-budget rejection on a
                    // giant prompt) to the client instead of masking them as a
                    // normal zero-token "stop" — otherwise clients can't tell a
                    // real failure from a model that just produced no output.
                    const errMsg = msg.message || "generation failed";
                    ctrl.enqueue(enc.encode(`data: ${JSON.stringify({
                      error: { message: errMsg, type: "invalid_request_error" }
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
        let daemonError: string | null = null;
        e.generating = true;
        for await (const msg of e.generate(genParams)) {
          if (msg.type === "token") { content += msg.text; completionTokens++; }
          else if (msg.type === "error") { daemonError = msg.message || "generation failed"; }
        }
        e.generating = false;

        // If the daemon rejected the request mid-generate (e.g. KV-budget
        // overrun on a huge system prompt), surface that as a 400 instead of
        // returning a 200 with empty content — otherwise a client that sent a
        // too-large request can't distinguish failure from a zero-token reply.
        if (daemonError) {
          safeRelease();
          return Response.json(
            { error: { message: daemonError, type: "invalid_request_error" } },
            { status: 400 }
          );
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

// ─── Quantize ───────────────────────────────────────────
// `hipfire quantize <hf-id|local-dir> [--format mq4|mq6|q8] [-o out]`
//
// Wraps the `hipfire-quantize` binary. Accepts either an HF model ID
// (e.g. `Qwen/Qwen3-0.6B`) — downloaded via the `hf` CLI — or a local
// directory of safetensors. Produces a single file readable by the
// engine loader.

function findQuantizeBinary(): string | null {
  const exe = process.platform === "win32" ? ".exe" : "";
  const candidates = [
    resolve(__dirname, `../target/release/hipfire-quantize${exe}`),
    join(HIPFIRE_DIR, "bin", `hipfire-quantize${exe}`),
  ];
  return candidates.find(p => existsSync(p)) || null;
}

interface QuantizeOpts {
  formats: string[];                 // one or more of mq4/mq6/q8
  output?: string;                   // explicit path (only valid with single format)
  outputDir?: string;                // directory for multi-format outputs
  stem?: string;                     // override output basename (default: inferred from input)
  uploadRepo?: string;               // schuttdev/hipfire-... — upload after quantize
  createRepo?: boolean;              // pass --create-repo to `hf upload`
  installLocal?: boolean;            // copy result into ~/.hipfire/models
  register?: string;                 // tag to add to registry (e.g., "qwopus:4b")
}

async function hfDownloadModel(hfId: string): Promise<string> {
  const cacheDir = join(HIPFIRE_DIR, "hf-cache", hfId.replace(/\//g, "_"));
  mkdirSync(cacheDir, { recursive: true });
  console.error(`Downloading ${hfId} from HuggingFace to ${cacheDir} ...`);
  const dl = Bun.spawnSync(
    [
      "hf", "download", hfId, "--local-dir", cacheDir,
      "--include", "*.safetensors",
      "--include", "*.safetensors.index.json",
      "--include", "config.json",
      "--include", "tokenizer.json",
      "--include", "tokenizer_config.json",
      "--include", "special_tokens_map.json",
      "--include", "generation_config.json",
    ],
    { stdio: ["inherit", "inherit", "inherit"] },
  );
  if ((dl.exitCode ?? 1) !== 0) {
    console.error(`hf download failed.`);
    console.error(`  Check: hf auth whoami  (run 'hf auth login' if not authed)`);
    console.error(`  Or install: pip install -U huggingface_hub`);
    process.exit(1);
  }
  return cacheDir;
}

async function quantize(input: string, opts: QuantizeOpts): Promise<void> {
  const bin = findQuantizeBinary();
  if (!bin) {
    console.error("hipfire-quantize binary not found.");
    console.error("  Build: cargo build --release -p hipfire-quantize");
    console.error("  Or:    hipfire update");
    process.exit(1);
  }

  // HF ID = exactly one `/`, HF-valid chars, and no such directory exists.
  const looksLikeHfId = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+$/.test(input)
    && !existsSync(input);
  const inputDir = looksLikeHfId ? await hfDownloadModel(input) : resolve(input);
  if (!looksLikeHfId && !existsSync(inputDir)) {
    console.error(`Input not found: ${inputDir}`);
    process.exit(1);
  }

  const baseName = opts.stem ?? (looksLikeHfId ? input.split("/").pop()! : basename(inputDir));

  // Sanity: --output is only meaningful with a single format
  if (opts.output && opts.formats.length > 1) {
    console.error("--output conflicts with multiple --format values. Use --output-dir instead.");
    process.exit(1);
  }
  const outDir = opts.outputDir ? resolve(opts.outputDir) : resolve(".");
  if (opts.outputDir) mkdirSync(outDir, { recursive: true });

  const produced: { format: string; path: string }[] = [];

  for (const format of opts.formats) {
    const out = opts.output
      ? resolve(opts.output)
      : join(outDir, `${baseName}.${format}`);

    console.error(`\nQuantizing ${inputDir}`);
    console.error(`  → ${out} (${format})`);
    const t0 = Date.now();
    const proc = Bun.spawnSync(
      [bin, "--input", inputDir, "--output", out, "--format", format],
      { stdio: ["inherit", "inherit", "inherit"] },
    );
    if ((proc.exitCode ?? 1) !== 0) {
      console.error(`Quantization failed (exit ${proc.exitCode})`);
      process.exit(1);
    }
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    try {
      const sz = (statSync(out).size / 1e9).toFixed(2);
      console.error(`Done: ${out} (${sz} GB, ${secs}s)`);
    } catch {
      console.error(`Done: ${out} (${secs}s)`);
    }
    produced.push({ format, path: out });
  }

  // Optional: drop the produced artifacts into ~/.hipfire/models so
  // `hipfire list` + `hipfire run` find them without any extra steps.
  if (opts.installLocal) {
    mkdirSync(MODELS_DIR, { recursive: true });
    for (const p of produced) {
      const dest = join(MODELS_DIR, basename(p.path));
      if (resolve(dest) !== resolve(p.path)) {
        require("fs").copyFileSync(p.path, dest);
        console.error(`Installed → ${dest}`);
      }
    }
  }

  // Optional: push the artifacts to a schuttdev-style HF repo. We upload
  // each produced file individually so partial failures don't wipe state.
  if (opts.uploadRepo) {
    // `hf upload` does not create the repo itself — if --create-repo is set,
    // use `hf repos create --exist-ok` which is idempotent.
    if (opts.createRepo) {
      console.error(`Ensuring HF repo ${opts.uploadRepo} exists ...`);
      const mk = Bun.spawnSync(
        ["hf", "repos", "create", opts.uploadRepo, "--type", "model", "--exist-ok"],
        { stdio: ["inherit", "inherit", "inherit"] },
      );
      if ((mk.exitCode ?? 1) !== 0) {
        console.error(`hf repos create failed. Check: hf auth whoami`);
        process.exit(1);
      }
    }
    for (const p of produced) {
      console.error(`\nUploading ${p.path} → ${opts.uploadRepo}:${basename(p.path)} ...`);
      const up = Bun.spawnSync(
        ["hf", "upload", opts.uploadRepo, p.path, basename(p.path)],
        { stdio: ["inherit", "inherit", "inherit"] },
      );
      if ((up.exitCode ?? 1) !== 0) {
        console.error(`Upload failed for ${p.path} (exit ${up.exitCode}).`);
        console.error(`  Check: hf auth whoami   |   If repo missing, pass --create-repo.`);
        process.exit(1);
      }
    }
    console.error(`\nUploaded ${produced.length} file(s) to ${opts.uploadRepo}.`);
  }

  // Optional: append a local user-alias so the custom tag is addressable.
  if (opts.register) {
    const aliasPath = join(HIPFIRE_DIR, "models.json");
    let aliases: Record<string, any> = {};
    try { aliases = JSON.parse(require("fs").readFileSync(aliasPath, "utf-8")); } catch {}
    const primary = produced.find(p => p.format === "mq4") ?? produced[0];
    aliases[opts.register] = {
      repo: opts.uploadRepo ?? "",
      file: basename(primary.path),
      local_path: primary.path,
      registered_at: new Date().toISOString(),
    };
    require("fs").writeFileSync(aliasPath, JSON.stringify(aliases, null, 2) + "\n");
    console.error(`Registered ${opts.register} → ${basename(primary.path)}`);
    console.error(`  Try: hipfire run ${opts.register} "hello"`);
  }
}

// ─── Helpers ────────────────────────────────────────────

interface UserAlias {
  repo?: string;
  file: string;
  local_path?: string;
  registered_at?: string;
}

function loadUserAliases(): Record<string, UserAlias> {
  try {
    return JSON.parse(require("fs").readFileSync(join(HIPFIRE_DIR, "models.json"), "utf-8"));
  } catch { return {}; }
}

function findModel(name: string): string | null {
  // Direct file path
  if (existsSync(name)) return resolve(name);

  // User aliases (from `hipfire quantize ... --register`) take precedence
  // over the built-in REGISTRY so custom tags always resolve.
  const userAliases = loadUserAliases();
  const alias = userAliases[name] || userAliases[resolveModelTag(name)];
  if (alias) {
    if (alias.local_path && existsSync(alias.local_path)) return resolve(alias.local_path);
    const p = join(MODELS_DIR, alias.file);
    if (existsSync(p)) return p;
  }

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
  // If the name includes a quant hint (hf4/hf6/mq4/mq6), match exactly.
  // Otherwise prefer .mq4 (default quant: FWHT-rotated 4-bit, quality-gated,
  // WMMA-accelerated on RDNA3+). Fall back to .hf4 only if no .mq4 is found
  // so Qwen3 (which currently ships only .hf4) still resolves.
  const searchName = name.replace(":", "-");
  const hasQuantHint = /\.(hf[46]|mq[46])$|-(hf[46]|mq[46])$/.test(name);
  const matchesName = (f: string) => f === name || f === searchName
    || f.includes(name) || f.includes(searchName);
  const hasValidExt = (f: string) => f.endsWith(".mq4") || f.endsWith(".mq6")
    || f.endsWith(".hf4") || f.endsWith(".hf6") || f.endsWith(".hfq");

  // Preference order when no quant hint: .mq4 → .hf4 → .hf6 → .mq6 → .hfq
  // (MQ6 only if explicitly asked; HF6 ditto — both are larger files.)
  const extPriority = (f: string): number => {
    if (f.endsWith(".mq4")) return 0;
    if (f.endsWith(".hf4")) return 1;
    if (f.endsWith(".hfq")) return 2; // legacy HF4 naming
    if (f.endsWith(".mq6")) return 3;
    if (f.endsWith(".hf6")) return 4;
    return 99;
  };

  const isModel = (f: string) => {
    if (!hasValidExt(f)) return false;
    if (!matchesName(f)) return false;
    if (f === name || f === searchName) return true;
    // With a quant hint in the name, caller is explicit — any matching file is fine.
    if (hasQuantHint) return true;
    // No hint: accept any valid extension; extPriority picks the best one.
    // Still filter .hfq to default-q4 flavor (.q4.hfq / -hfq4.hfq stems) so
    // we don't return an experimental -hfq4g128.hfq instead of a proper .mq4.
    if (f.endsWith(".hfq")) {
      const stem = f.slice(0, -4);
      const isDefaultQ4 = stem.endsWith(".q4") || stem.endsWith("-hfq4")
        || stem === searchName || stem === name;
      if (!isDefaultQ4) return false;
    }
    return true;
  };

  const dirs = [resolve(__dirname, "../models"), MODELS_DIR];
  const candidates: string[] = [];
  for (const dir of dirs) {
    try {
      for (const f of readdirSync(dir)) {
        const full = join(dir, f);
        if (isModel(f)) candidates.push(full);
        // One level of subdirectories (e.g. models/community/)
        try {
          if (statSync(full).isDirectory()) {
            for (const sf of readdirSync(full)) {
              if (isModel(sf)) candidates.push(join(full, sf));
            }
          }
        } catch {}
      }
    } catch {}
  }
  if (candidates.length === 0) return null;
  // When the user had an explicit hint, any match is fine — return the first
  // (same behavior as before). Otherwise pick by preference order.
  candidates.sort((a, b) => extPriority(basename(a)) - extPriority(basename(b)));
  return candidates[0];
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
  ttft: number[];
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

function fmtBytes(b: number): string {
  if (b >= 1024 * 1024 * 1024) return (b / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  if (b >= 1024 * 1024) return (b / (1024 * 1024)).toFixed(1) + " MB";
  if (b >= 1024) return (b / 1024).toFixed(1) + " KB";
  return b + " B";
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

// benchRun result + flag indicating the engine is poisoned (timed out mid-stream).
// `decode` is pure decode tok/s (post-prefill); `wall` is whole-request tok/s
// (kept for backward-compat / sanity); `prefill` is prompt-processing tok/s.
interface BenchRunResult {
  decode: number;
  prefill: number;
  wall: number;
  ttftMs: number;
  prefillMs: number;
  prefillTokens: number;
  tokens: number;
  ok: boolean;
  poisoned: boolean;
}

async function benchRun(e: Engine, prompt: string, maxTokens: number, timeoutMs = 120_000): Promise<BenchRunResult> {
  const fail = { decode: 0, prefill: 0, wall: 0, ttftMs: 0, prefillMs: 0, prefillTokens: 0, tokens: 0, ok: false, poisoned: false };
  try {
    await withTimeout(e.send({ type: "reset" }).then(() => e.recv()), 10_000, "reset");
  } catch { return { ...fail, poisoned: true }; }
  const genMsg = {
    type: "generate", id: "bench", prompt,
    temperature: 0, max_tokens: maxTokens,
    repeat_penalty: 1.1, top_p: 1.0,
  };
  let decode = 0, prefill = 0, wall = 0, ttftMs = 0, prefillMs = 0, prefillTokens = 0, tokens = 0;
  try {
    const run = async () => {
      for await (const msg of e.generate(genMsg)) {
        if (msg.type === "done") {
          // New daemons emit split metrics; fall back to tok_s if missing.
          wall = msg.tok_s || 0;
          decode = msg.decode_tok_s ?? wall;
          prefill = msg.prefill_tok_s ?? 0;
          ttftMs = msg.ttft_ms ?? 0;
          prefillMs = msg.prefill_ms ?? 0;
          prefillTokens = msg.prefill_tokens ?? 0;
          tokens = msg.tokens || 0;
        }
      }
    };
    await withTimeout(run(), timeoutMs, "generate");
  } catch {
    // Timed out mid-stream — daemon is reading/writing stale data, must be killed
    return { ...fail, poisoned: true };
  }
  return {
    decode, prefill, wall, ttftMs, prefillMs, prefillTokens, tokens,
    ok: decode > 0,
    poisoned: false,
  };
}

// Synthetic prefill measurement: runs `bench_prefill` on the daemon which
// times forward_prefill_batch over N deterministic tokens from a zeroed
// state. Returns tok/s and ms, or null on error (e.g. N > max_seq).
async function benchPrefill(e: Engine, tokens: number, timeoutMs = 60_000): Promise<{ tokS: number; ms: number } | null> {
  try {
    await withTimeout(e.send({ type: "bench_prefill", tokens }), 5_000, "bench_prefill send");
    const res = await withTimeout(e.recv(), timeoutMs, `bench_prefill (${tokens} tok)`);
    if (res.type === "prefill_result") {
      return { tokS: res.tok_s || 0, ms: res.ms || 0 };
    }
    // Surface daemon errors to stderr but don't poison the engine; the
    // state reset on the daemon side is independent of the error path.
    if (res.type === "error" && res.message) {
      console.error(`  pp${tokens}: ${res.message}`);
    }
    return null;
  } catch {
    return null;
  }
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

  applyConfigEnv(cfg);

  // Start daemon
  const e = new Engine();
  await e.start();
  await e.send({ type: "ping" }); await e.recv();

  // Pre-load VRAM snapshot — lets us compute weights+scratch+KV footprint
  // by diffing against the post-load snapshot.
  await e.send({ type: "diag" });
  const preDiag = await e.recv();
  const vramFreePreMb = preDiag.vram_free_mb || 0;
  const vramTotalMb = preDiag.vram_total_mb || 0;
  const gpuArch = preDiag.arch || "unknown";
  const hipVer = preDiag.hip_version || "?";
  const isRdna2 = gpuArch === "gfx1030" || gpuArch === "gfx1031";

  const loadMsg = buildLoadMessage(modelPath, model);
  await e.send(loadMsg);
  const loaded = await e.recv();
  if (loaded.type === "error") { console.error(loaded.message); process.exit(1); }

  // Post-load VRAM snapshot — delta gives model footprint.
  await e.send({ type: "diag" });
  const postDiag = await e.recv();
  const vramFreePostMb = postDiag.vram_free_mb || 0;
  const loadedMb = Math.max(0, vramFreePreMb - vramFreePostMb);

  console.error(`hipfire bench`);
  console.error(`  model:     ${basename(modelPath!)}  [${loaded.arch}]`);
  if (loaded.dim)    console.error(`  arch:      dim=${loaded.dim}, layers=${loaded.layers}, vocab=${loaded.vocab}${loaded.vl ? " (vision)" : ""}`);
  console.error(`  gpu:       ${gpuArch}  (HIP ${hipVer})`);
  console.error(`  kv_cache:  ${cfg.kv_cache}`);
  console.error(`  max_seq:   ${loadMsg.params.max_seq}`);
  if (loadedMb > 0) console.error(`  vram:      ${loadedMb} MB loaded  (${vramFreePostMb}/${vramTotalMb} MB free)`);
  else              console.error(`  vram:      ${vramFreePostMb}/${vramTotalMb} MB free`);
  console.error(`  runs:      ${runs}`);
  console.error(`  prompt:    "${prompt.length > 60 ? prompt.slice(0, 57) + "..." : prompt}"`);

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
        await ve.send(buildLoadMessage(modelPath, model));
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
        results.push({ label: `v${v.n} ${v.name}`, decode: [], prefill: [], ttft: [] });
        await ve.stop();
        continue;
      }

      // Warmup
      const warmup = await benchRun(ve, "Hello", 16, 30_000);
      if (warmup.poisoned) {
        console.error(`  v${v.n} ${v.name}: warmup timed out`);
        results.push({ label: `v${v.n} ${v.name}`, decode: [], prefill: [], ttft: [] });
        await ve.stop();
        continue;
      }

      process.stderr.write(`  v${v.n} ${v.name.padEnd(18)} `);
      const decodes: number[] = [];
      const prefills: number[] = [];
      const ttfts: number[] = [];
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
        if (res.prefill > 0) prefills.push(res.prefill);
        if (res.ttftMs > 0) ttfts.push(res.ttftMs);
        process.stderr.write(".");
      }
      console.error("");
      results.push({ label: `v${v.n} ${v.name}`, decode: decodes, prefill: prefills, ttft: ttfts });
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
    console.error(`  mode:      standard\n`);

    // Warmup
    process.stderr.write("  warming up...");
    const warmup = await benchRun(e, "Hello", 16);
    if (warmup.poisoned) {
      console.error(" TIMEOUT — daemon unresponsive");
      await e.stop();
      process.exit(1);
    }
    console.error(" done\n");

    // Synthetic prefill tests: canonical pp128/pp512/pp1024 numbers that
    // don't depend on prompt tokenization. Older daemons ignore the command
    // and return an error; we silently skip in that case. Each size is run
    // `runs` times so we can report variance.
    const ppSizes = [128, 512, 1024, 2048].filter(n => n + 32 <= loadMsg.params.max_seq);
    const ppResults: { size: number; samples: number[]; ms: number[] }[] = [];
    if (ppSizes.length > 0) {
      process.stderr.write("  prefill: ");
      for (const size of ppSizes) {
        // Discarded warmup: the first prefill at a new size often hits cold
        // kernel-specific caches (scratch buffers sized for this N, memoized
        // launch configs). Throwing it away gives tight variance.
        await benchPrefill(e, size);

        const samples: number[] = [];
        const mss: number[] = [];
        for (let r = 0; r < runs; r++) {
          const res = await benchPrefill(e, size);
          if (!res) break;
          samples.push(res.tokS);
          mss.push(res.ms);
        }
        if (samples.length > 0) {
          ppResults.push({ size, samples, ms: mss });
          const s = stats(samples);
          process.stderr.write(`pp${size}=${s.mean.toFixed(0)} `);
        } else {
          process.stderr.write(`pp${size}=skip `);
        }
      }
      console.error("");
    }

    const decodes: number[] = [];
    const prefills: number[] = [];
    const ttfts: number[] = [];
    const walls: number[] = [];
    const tokenCounts: number[] = [];
    let lastPrefillTokens = 0;

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
      walls.push(res.wall);
      if (res.prefill > 0)  prefills.push(res.prefill);
      if (res.ttftMs > 0)   ttfts.push(res.ttftMs);
      if (res.prefillTokens) lastPrefillTokens = res.prefillTokens;
      tokenCounts.push(res.tokens);
      // One-liner: pp tok/s | TTFT ms | decode tok/s (n tok)
      const pp = res.prefill > 0 ? `pp ${res.prefill.toFixed(0)} tok/s` : `pp --`;
      const tt = res.ttftMs > 0  ? `TTFT ${res.ttftMs.toFixed(0)} ms` : `TTFT --`;
      console.error(`${pp} | ${tt} | decode ${res.decode.toFixed(1)} tok/s (${res.tokens} tok)`);
    }

    const d = stats(decodes);
    const p = stats(prefills);
    const t = stats(ttfts);
    const w = stats(walls);

    console.log("");

    // Synthetic prefill scaling table (pp128, pp512, pp1024, ...): canonical
    // numbers comparable across builds and against other engines.
    if (ppResults.length > 0) {
      console.log(`  Prefill    tok/s      mean      min      max    stdev     ms`);
      console.log("  " + "─".repeat(64));
      for (const pp of ppResults) {
        const s = stats(pp.samples);
        const mMean = pp.ms.reduce((a, b) => a + b, 0) / pp.ms.length;
        console.log(
          `  pp${String(pp.size).padEnd(5)}         ` +
          `${fmtNum(s.mean,9)}${fmtNum(s.min,9)}${fmtNum(s.max,9)}${fmtNum(s.stdev,9)}   ${mMean.toFixed(1)}`
        );
      }
      console.log("");
    }

    console.log(`                       mean      min      max    stdev`);
    console.log("  " + "─".repeat(58));
    if (p.mean > 0) {
      console.log(`  Prefill  tok/s  ${fmtNum(p.mean,9)}${fmtNum(p.min,9)}${fmtNum(p.max,9)}${fmtNum(p.stdev,9)}   (user prompt, ${lastPrefillTokens} tok)`);
    }
    if (t.mean > 0) {
      console.log(`  TTFT     ms     ${fmtNum(t.mean,9)}${fmtNum(t.min,9)}${fmtNum(t.max,9)}${fmtNum(t.stdev,9)}`);
    }
    console.log(`  Decode   tok/s  ${fmtNum(d.mean,9)}${fmtNum(d.min,9)}${fmtNum(d.max,9)}${fmtNum(d.stdev,9)}`);
    if (w.mean > 0 && Math.abs(w.mean - d.mean) > 0.5) {
      // Wall-clock is useful only when prefill meaningfully drags on decode.
      console.log(`  Wall     tok/s  ${fmtNum(w.mean,9)}${fmtNum(w.min,9)}${fmtNum(w.max,9)}${fmtNum(w.stdev,9)}`);
    }

    if (d.mean > 0) {
      console.log(`\n  Decode ms/tok: ${(1000 / d.mean).toFixed(2)}`);
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
      applyConfigEnv(cfg);
      await e.send(buildLoadMessage(modelPath, modelTag));
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

// ─── Config TUI ─────────────────────────────────────────
// Keyboard-driven settings editor. Raw ANSI, no deps.
//   ↑/↓     — move between rows
//   ←/→/sp  — cycle enum values (kv_cache, default_model)
//   -/+     — nudge numeric values by their step
//   enter   — edit a text/number field directly
//   r       — reset selected row to default
//   s       — save (writes ~/.hipfire/config.json, keeps only non-defaults)
//   q / Esc — save+quit
//   Ctrl+C  — abort without saving

interface FieldMeta {
  label: string;
  desc: string;
  options?: string[];           // enum values — shown inline, cycle-able
  range?: [number, number];     // numeric clamp
  step?: number;                // +/- nudge amount
  decimals?: number;            // display precision for floats
}

// TUI exit actions — the case "config" orchestrator uses these to decide
// what screen to show next. "exit" = user is done. "open_picker" = user
// pressed Enter on the "[per-model configs]" virtual row.
type TuiExit = "exit" | "open_picker";

// Scope = null → edit global config. Scope = tag string → edit per-model
// overlay for that tag. Per-model mode shows inherited values dimmed and
// highlights overrides in cyan; `r` removes an override.
function configTui(cfg: HipfireConfig, scope?: string | null): Promise<TuiExit> {
  const isPerModel = !!scope;
  const resolvedTag = scope ? resolveModelTag(scope) : null;

  // Per-model mode: base values come from global cfg; overrides are sparse.
  let overrides: PerModelOverride = isPerModel
    ? { ...(loadPerModelConfigs()[resolvedTag!] ?? {}) }
    : {};

  // In per-model mode only show keys that can actually be overridden.
  const allKeys = Object.keys(CONFIG_DEFAULTS) as (keyof HipfireConfig)[];
  const keys = isPerModel
    ? allKeys.filter(k => (PER_MODEL_KEYS as readonly string[]).includes(k))
    : allKeys;
  // Virtual rows (nav-only, not real config keys). Only in global mode.
  const navKeys = isPerModel ? [] : ["__per_model__"];
  const totalRows = keys.length + navKeys.length;

  // Effective value for a key: override wins in per-model mode, else cfg.
  const effective = (k: keyof HipfireConfig): any =>
    isPerModel && (overrides as any)[k] !== undefined ? (overrides as any)[k] : cfg[k];
  const isOverridden = (k: keyof HipfireConfig): boolean =>
    isPerModel && (overrides as any)[k] !== undefined;

  // Build default_model options from REGISTRY so users can cycle through
  // known tags without typing. "custom" lets them fall back to free text.
  const modelOptions = Object.keys(REGISTRY).sort();

  const meta: Record<string, FieldMeta> = {
    kv_cache: {
      label: "kv_cache",
      desc: "KV cache quantization (more bits = higher quality, more VRAM)",
      options: ["auto", "q8", "asym4", "asym3", "asym2"],
    },
    flash_mode: {
      label: "flash_mode",
      desc: "Flash attention (Q8: auto=ctx≥2048, always=force, never=disable; asym always flash)",
      options: ["auto", "always", "never"],
    },
    default_model: {
      label: "default_model",
      desc: "model pre-warmed when `hipfire serve` starts",
      options: modelOptions,
    },
    temperature: {
      label: "temperature",
      desc: "sampling randomness — 0 = greedy, higher = more diverse",
      range: [0, 2], step: 0.05, decimals: 2,
    },
    top_p: {
      label: "top_p",
      desc: "nucleus sampling — only consider tokens covering this probability mass",
      range: [0, 1], step: 0.05, decimals: 2,
    },
    repeat_penalty: {
      label: "repeat_penalty",
      desc: "discourage repeats — 1.05 is safe for MQ4/MQ6, 1.3 causes gibberish",
      range: [1, 3], step: 0.05, decimals: 2,
    },
    max_tokens: {
      label: "max_tokens",
      desc: "default generation cap per `hipfire run` invocation (per-turn stop)",
      range: [1, 131072], step: 64,
    },
    max_seq: {
      label: "max_seq",
      desc: "KV cache capacity (tokens). Allocated at model load — bigger = longer context",
      range: [512, 524288], step: 4096,
    },
    thinking: {
      label: "thinking",
      desc: "Reasoning mode. on = model uses <think>...</think> (stripped from display); off = suppress thinking, answer directly",
      options: ["on", "off"],
    },
    max_think_tokens: {
      label: "max_think_tokens",
      desc: "Budget for reasoning inside <think>...</think> (0 = unlimited). Truncates if exceeded.",
      range: [0, 32768], step: 128,
    },
    port: {
      label: "port",
      desc: "HTTP port for `hipfire serve` (OpenAI-compatible API)",
      range: [1, 65535], step: 1,
    },
    idle_timeout: {
      label: "idle_timeout",
      desc: "serve: seconds idle before unloading model (frees VRAM; 0 = never unload)",
      range: [0, 86400], step: 30,
    },
  };

  let selected = 0;
  let dirty = false;
  let editing = false;
  let editBuffer = "";
  let flash = "";                  // transient status message

  const stdout = process.stdout;
  const stdin = process.stdin;
  const write = (s: string) => stdout.write(s);

  // Colors
  const C = {
    reset: "\x1b[0m",
    dim: "\x1b[2m",
    bold: "\x1b[1m",
    red: "\x1b[31m",
    green: "\x1b[32m",
    yellow: "\x1b[33m",
    cyan: "\x1b[36m",
    magenta: "\x1b[35m",
    inv: "\x1b[7m",
  };

  const fmtValue = (k: keyof HipfireConfig): string => {
    const v = effective(k);
    const m = meta[k];
    if (typeof v === "number" && m.decimals !== undefined) {
      return v.toFixed(m.decimals);
    }
    return String(v);
  };

  const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));

  const roundStep = (v: number, step: number, decimals?: number) => {
    if (decimals !== undefined) return Number(v.toFixed(decimals));
    if (Number.isInteger(step)) return Math.round(v);
    return v;
  };

  // Write to whichever bag this scope is editing: overrides in per-model
  // mode, the global cfg otherwise. Always marks dirty.
  const setValue = (k: keyof HipfireConfig, v: any) => {
    if (isPerModel) (overrides as any)[k] = v;
    else (cfg as any)[k] = v;
    dirty = true;
  };

  const cycleOption = (k: keyof HipfireConfig, dir: number) => {
    const m = meta[k];
    if (!m.options) return;
    const cur = String(effective(k));
    let idx = m.options.indexOf(cur);
    if (idx < 0) idx = 0;
    const next = m.options[(idx + dir + m.options.length) % m.options.length];
    setValue(k, next);
  };

  const nudge = (k: keyof HipfireConfig, dir: number) => {
    const m = meta[k];
    if (!m.range || m.step === undefined) return;
    const cur = Number(effective(k));
    const raw = cur + dir * m.step;
    const next = clamp(roundStep(raw, m.step, m.decimals), m.range[0], m.range[1]);
    if (validateConfigValue(k as string, next)) {
      setValue(k, next);
    }
  };

  const commitEdit = () => {
    const k = keys[selected];
    const defaultVal = CONFIG_DEFAULTS[k];
    const parsed = typeof defaultVal === "number" ? Number(editBuffer) : editBuffer;
    if (editBuffer.length > 0 && validateConfigValue(k as string, parsed as any)) {
      const m = meta[k];
      const finalVal = typeof parsed === "number" && m.decimals !== undefined
        ? Number((parsed as number).toFixed(m.decimals))
        : parsed;
      setValue(k, finalVal);
      flash = `${C.green}${k} = ${fmtValue(k)}${C.reset}`;
    } else {
      flash = `${C.red}invalid value for ${k}${C.reset}`;
    }
    editing = false;
    editBuffer = "";
  };

  const render = () => {
    // Cursor home + clear screen
    write("\x1b[H\x1b[2J");
    if (isPerModel) {
      write(`${C.bold}hipfire config ${C.cyan}${resolvedTag}${C.reset}  ${C.dim}${PER_MODEL_CONFIG_PATH}${C.reset}\n`);
      write(`${C.dim}per-model overlay — overrides win over global. Use r to remove an override.${C.reset}\n\n`);
    } else {
      write(`${C.bold}hipfire config${C.reset}  ${C.dim}${CONFIG_PATH}${C.reset}\n`);
      write(`${C.dim}GPU: ${DETECTED_ARCH} · auto = ${ARCH_DEFAULTS.kv_cache}${C.reset}\n\n`);
    }

    // Column widths
    const labelW = Math.max(...keys.map(k => meta[k].label.length)) + 2;
    const valueW = 14;

    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      const m = meta[k];
      const v = effective(k);
      const overridden = isOverridden(k);
      const isDefault = !isPerModel && v === CONFIG_DEFAULTS[k];
      const caret = i === selected ? `${C.cyan}▸${C.reset}` : " ";

      // Value (editing takes priority visually)
      let valCell: string;
      if (editing && i === selected) {
        valCell = `${C.yellow}${editBuffer}${C.inv} ${C.reset}`.padEnd(valueW + 20);
      } else {
        let color: string;
        if (isPerModel) {
          color = overridden ? C.cyan : C.dim;  // overridden values pop; inherited dim
        } else {
          color = isDefault ? C.dim : C.green;
        }
        valCell = `${color}${fmtValue(k)}${C.reset}`;
        const pad = Math.max(0, valueW - fmtValue(k).length);
        valCell = valCell + " ".repeat(pad);
      }

      let optHint = "";
      const flashModeIgnored = k === "flash_mode" &&
        typeof effective("kv_cache") === "string" &&
        effective("kv_cache").startsWith("asym");
      if (m.options) {
        if (m.options.length <= 6) {
          optHint = m.options.map(o => {
            if (o === String(v)) {
              return flashModeIgnored ? `${C.dim}${o}${C.reset}` : `${C.cyan}${o}${C.reset}`;
            }
            return `${C.dim}${o}${C.reset}`;
          }).join(" ");
          if (flashModeIgnored) optHint += `  ${C.yellow}(ignored — asym is flash-only)${C.reset}`;
        } else {
          const idx = m.options.indexOf(String(v));
          const pos = idx >= 0 ? `${idx + 1}/${m.options.length}` : `?/${m.options.length}`;
          optHint = `${C.dim}←→ cycle (${pos})${C.reset}`;
        }
      } else if (m.range) {
        optHint = `${C.dim}${m.range[0]}${m.step && !Number.isInteger(m.step) ? ".0" : ""}–${m.range[1]}${C.reset}`;
      }

      // Status chip on the right: "(default)" for global, "(overridden)" or
      // "(inherited)" for per-model mode so the user sees which rows belong
      // to this model vs pulled from global.
      let chip: string;
      if (isPerModel) {
        chip = overridden
          ? `${C.cyan}(overridden)${C.reset}`
          : `${C.dim}(inherited)${C.reset}`;
      } else {
        chip = isDefault ? `${C.dim}(default)${C.reset}` : " ".repeat(9);
      }
      const rowHeader = `${caret} ${m.label.padEnd(labelW)} ${valCell} ${chip}`;
      write(`${rowHeader}  ${optHint}\n`);
      if (i === selected) {
        write(`${" ".repeat(3 + labelW)}${C.dim}${m.desc}${C.reset}\n`);
      }
    }

    // Virtual nav rows (global mode only). Shown as a distinct-looking row
    // the user can Enter into.
    for (let n = 0; n < navKeys.length; n++) {
      const rowIdx = keys.length + n;
      const nk = navKeys[n];
      const caret = rowIdx === selected ? `${C.cyan}▸${C.reset}` : " ";
      if (nk === "__per_model__") {
        const pmAll = loadPerModelConfigs();
        const count = Object.keys(pmAll).length;
        const label = "per-model configs".padEnd(labelW);
        const val = count > 0
          ? `${C.magenta}${count} override set${count === 1 ? "" : "s"}${C.reset}`
          : `${C.dim}no overrides${C.reset}`;
        write(`\n${caret} ${C.bold}${label}${C.reset} ${val}  ${C.dim}→ enter to open model picker${C.reset}\n`);
        if (rowIdx === selected) {
          write(`${" ".repeat(3 + labelW)}${C.dim}Per-model overlays let you customize settings for a specific model (e.g. bigger max_seq for long ctx on 9B).${C.reset}\n`);
        }
      }
    }

    write("\n");
    if (editing) {
      write(`  ${C.dim}enter: save · esc: cancel · backspace: delete${C.reset}\n`);
    } else {
      const saveState = dirty ? `${C.yellow}●${C.reset} unsaved` : `${C.dim}saved${C.reset}`;
      const resetHelp = isPerModel ? "r remove override" : "r reset";
      write(`  ${C.dim}↑↓ nav · ←→/space cycle · -/+ tweak · enter edit · ${resetHelp} · s save · q quit${C.reset}   ${saveState}\n`);
    }
    if (flash) {
      write(`\n  ${flash}\n`);
      flash = "";
    }
  };

  return new Promise<TuiExit>((resolve) => {
    if (!stdout.isTTY || !stdin.isTTY) {
      // Can't run a TUI without a real terminal — fall through to list view
      listConfig(cfg);
      resolve("exit");
      return;
    }

    stdin.setRawMode!(true);
    stdin.resume();
    stdin.setEncoding("utf8");
    write("\x1b[?25l"); // hide cursor

    const cleanup = () => {
      write("\x1b[?25h"); // show cursor
      stdin.setRawMode!(false);
      stdin.pause();
      stdin.removeAllListeners("data");
      write("\n");
    };

    const onData = (data: string) => {
      if (editing) {
        // Text/number edit mode
        if (data === "\r" || data === "\n") {
          commitEdit();
        } else if (data === "\x1b" || data === "\x1b\x1b") {
          editing = false;
          editBuffer = "";
          flash = `${C.dim}edit cancelled${C.reset}`;
        } else if (data === "\x7f" || data === "\b") {
          editBuffer = editBuffer.slice(0, -1);
        } else if (data === "\x03") { // Ctrl+C
          cleanup();
          process.exit(130);
        } else if (data.length === 1 && data.charCodeAt(0) >= 32) {
          editBuffer += data;
        }
        render();
        return;
      }

      // Helpers for virtual-row awareness
      const onNavRow = () => selected >= keys.length;
      const currentNavKey = () => onNavRow() ? navKeys[selected - keys.length] : null;
      const saveAndExit = (action: TuiExit) => {
        if (dirty) {
          if (isPerModel) {
            const all = loadPerModelConfigs();
            if (Object.keys(overrides).length === 0) delete all[resolvedTag!];
            else all[resolvedTag!] = { ...overrides };
            savePerModelConfigs(all);
          } else {
            saveConfig(cfg);
          }
        }
        cleanup();
        resolve(action);
      };

      // Navigation + mutation
      switch (data) {
        case "\x1b[A": // up
          selected = (selected + totalRows - 1) % totalRows;
          break;
        case "\x1b[B": // down
          selected = (selected + 1) % totalRows;
          break;
        case "\x1b[C": // right
        case " ":
          if (onNavRow()) break;
          cycleOption(keys[selected], +1);
          if (!meta[keys[selected]].options) nudge(keys[selected], +1);
          break;
        case "\x1b[D": // left
          if (onNavRow()) break;
          cycleOption(keys[selected], -1);
          if (!meta[keys[selected]].options) nudge(keys[selected], -1);
          break;
        case "+": case "=":
          if (onNavRow()) break;
          nudge(keys[selected], +1);
          break;
        case "-": case "_":
          if (onNavRow()) break;
          nudge(keys[selected], -1);
          break;
        case "\r": case "\n": {
          if (onNavRow()) {
            if (currentNavKey() === "__per_model__") {
              saveAndExit("open_picker");
              return;
            }
            break;
          }
          const k = keys[selected];
          const m = meta[k];
          if (m.options) {
            cycleOption(k, +1);
          } else {
            editing = true;
            editBuffer = "";
          }
          break;
        }
        case "r": case "R":
          if (onNavRow()) break;
          if (isPerModel) {
            const k = keys[selected];
            if (isOverridden(k)) {
              delete (overrides as any)[k];
              dirty = true;
              flash = `${C.dim}${k} override removed (inheriting global)${C.reset}`;
            } else {
              flash = `${C.dim}${keys[selected]} is already inherited${C.reset}`;
            }
          } else {
            (cfg as any)[keys[selected]] = CONFIG_DEFAULTS[keys[selected]];
            dirty = true;
            flash = `${C.dim}${keys[selected]} reset${C.reset}`;
          }
          break;
        case "s": case "S":
          if (isPerModel) {
            const all = loadPerModelConfigs();
            if (Object.keys(overrides).length === 0) delete all[resolvedTag!];
            else all[resolvedTag!] = { ...overrides };
            savePerModelConfigs(all);
          } else {
            saveConfig(cfg);
          }
          dirty = false;
          flash = `${C.green}saved${C.reset}`;
          break;
        case "q": case "Q": case "\x1b":
          saveAndExit("exit");
          return;
        case "\x03": case "\x04": // Ctrl+C / Ctrl+D
          cleanup();
          process.exit(130);
      }
      render();
    };

    stdin.on("data", onData);
    render();
  });
}

// Sub-TUI launched from the global config TUI's "[per-model configs]" row.
// Lists registered models (REGISTRY + any user-registered aliases), shows
// which have overrides, and returns the selected tag or null if user escapes.
function modelPickerTui(): Promise<string | null> {
  const tags = [
    ...Object.keys(REGISTRY),
    ...Object.keys(loadUserAliases()),
  ].filter((t, i, arr) => arr.indexOf(t) === i).sort();

  if (tags.length === 0) {
    console.log("No models registered. Pull one first: hipfire pull qwen3.5:9b");
    return Promise.resolve(null);
  }

  const overlays = loadPerModelConfigs();
  let selected = 0;
  const stdout = process.stdout;
  const stdin = process.stdin;
  const write = (s: string) => stdout.write(s);
  const C = {
    reset: "\x1b[0m", dim: "\x1b[2m", bold: "\x1b[1m",
    cyan: "\x1b[36m", magenta: "\x1b[35m", yellow: "\x1b[33m",
  };

  const render = () => {
    write("\x1b[H\x1b[2J");
    write(`${C.bold}hipfire config — model picker${C.reset}\n`);
    write(`${C.dim}Select a model to edit its per-model overrides. Esc to cancel.${C.reset}\n\n`);
    for (let i = 0; i < tags.length; i++) {
      const tag = tags[i];
      const ov = overlays[tag];
      const cnt = ov ? Object.keys(ov).length : 0;
      const caret = i === selected ? `${C.cyan}▸${C.reset}` : " ";
      const entry = REGISTRY[tag];
      const desc = entry?.desc ?? "(user-registered)";
      const size = entry ? `${entry.size_gb}GB`.padStart(7) : "".padStart(7);
      const marker = cnt > 0
        ? `${C.magenta}● ${cnt} override${cnt === 1 ? "" : "s"}${C.reset}`
        : `${C.dim}(no overrides)${C.reset}`;
      write(` ${caret} ${tag.padEnd(22)} ${size}  ${marker.padEnd(30)} ${C.dim}${desc}${C.reset}\n`);
    }
    write(`\n  ${C.dim}↑↓ nav · enter open · esc/q cancel${C.reset}\n`);
  };

  return new Promise<string | null>((resolve) => {
    if (!stdout.isTTY || !stdin.isTTY) { resolve(null); return; }
    stdin.setRawMode!(true);
    stdin.resume();
    stdin.setEncoding("utf8");
    write("\x1b[?25l");

    const cleanup = () => {
      write("\x1b[?25h");
      stdin.setRawMode!(false);
      stdin.pause();
      stdin.removeAllListeners("data");
      write("\n");
    };

    stdin.on("data", (data: string) => {
      switch (data) {
        case "\x1b[A": selected = (selected + tags.length - 1) % tags.length; render(); return;
        case "\x1b[B": selected = (selected + 1) % tags.length; render(); return;
        case "\r": case "\n":
          cleanup();
          resolve(tags[selected]);
          return;
        case "q": case "Q": case "\x1b":
          cleanup();
          resolve(null);
          return;
        case "\x03": case "\x04":
          cleanup();
          process.exit(130);
      }
    });
    render();
  });
}

function listConfig(cfg: HipfireConfig): void {
  const validKeys = Object.keys(CONFIG_DEFAULTS) as (keyof HipfireConfig)[];
  console.log(`Config: ${CONFIG_PATH}\n`);
  for (const k of validKeys) {
    const v = cfg[k];
    const isDefault = v === CONFIG_DEFAULTS[k];
    console.log(`  ${k.padEnd(18)} ${String(v).padEnd(14)}${isDefault ? "(default)" : ""}`);
  }
  console.log(`\nInteractive: hipfire config`);
  console.log(`Set:         hipfire config set <key> <value>`);
  console.log(`Reset:       hipfire config reset [key]`);
}

// ─── Main ───────────────────────────────────────────────

const [cmd, ...rest] = process.argv.slice(2);
switch (cmd) {
  case "serve": {
    // Parse flags: `hipfire serve [port] [-d|--detach]`. Port can be anywhere.
    let port: number | null = null;
    let detach = false;
    for (const a of rest) {
      if (a === "-d" || a === "--detach" || a === "--background") detach = true;
      else if (/^\d+$/.test(a)) port = parseInt(a, 10);
      else if (a === "-h" || a === "--help") {
        console.error(`Usage: hipfire serve [port] [-d|--detach]\n\n`
          + `  [port]     HTTP port (default: cfg.port = ${cfg.port})\n`
          + `  -d, --detach   Fork to background; log to ${SERVE_LOG_FILE}, PID in ${SERVE_PID_FILE}\n\n`
          + `Background daemon:\n`
          + `  hipfire serve -d           # start in background\n`
          + `  hipfire stop               # kill it\n`
          + `  hipfire ps                 # check if running\n`
          + `  tail -f ${SERVE_LOG_FILE}  # follow log\n`);
        process.exit(0);
      } else { console.error(`Unknown serve arg: ${a}`); process.exit(1); }
    }
    port = port ?? cfg.port;

    if (detach) {
      // Refuse to start a second one.
      const existing = readServePid();
      if (existing) {
        console.error(`hipfire serve already running (PID ${existing}) on port ${cfg.port}.`);
        console.error(`  Stop it: hipfire stop`);
        process.exit(1);
      }
      // Fork a detached child. `setsid` gives it its own session so Ctrl-C
      // in the parent shell doesn't reach it; `nohup` ignores SIGHUP; stdout
      // + stderr go to the log file. HIPFIRE_DETACHED prevents infinite forking.
      const self = process.argv[0];
      const script = process.argv[1];
      const logFd = require("fs").openSync(SERVE_LOG_FILE, "a");
      const child = Bun.spawn(["setsid", "nohup", self, script, "serve", String(port)], {
        stdin: "ignore",
        stdout: logFd,
        stderr: logFd,
        env: { ...process.env, HIPFIRE_DETACHED: "1" },
      });
      child.unref();
      // Poll until /health is reachable. First-run kernel JIT on slower
      // hardware (APUs, gfx1013) can take well over a minute for a 9B model,
      // so give it a generous window. Subsequent starts hit the kernel cache
      // and return in seconds.
      const READINESS_TIMEOUT_MS = 300_000;   // 5 minutes
      const deadline = Date.now() + READINESS_TIMEOUT_MS;
      console.log(`Waiting for serve to become ready (up to ${READINESS_TIMEOUT_MS / 1000}s for first-run kernel JIT)...`);
      while (Date.now() < deadline) {
        await new Promise(r => setTimeout(r, 500));
        if (await isServeUp(port)) break;
        // Show progress every 30s
        const elapsed = Math.floor((Date.now() - (deadline - READINESS_TIMEOUT_MS)) / 1000);
        if (elapsed > 0 && elapsed % 30 === 0) {
          process.stderr.write(`  ...still starting (${elapsed}s — tail ${SERVE_LOG_FILE} to watch)\r`);
        }
      }
      if (await isServeUp(port)) {
        console.log(`hipfire serve started in background (PID ${child.pid}, port ${port})`);
        console.log(`  log:  ${SERVE_LOG_FILE}`);
        console.log(`  stop: hipfire stop`);
      } else {
        console.error(`Serve started (PID ${child.pid}) but /health did not respond within ${READINESS_TIMEOUT_MS / 1000}s.`);
        console.error(`Check the log: tail -f ${SERVE_LOG_FILE}`);
      }
      break;
    }
    await serve(port);
    break;
  }
  case "stop": {
    const pid = readServePid();
    if (!pid) {
      console.log("hipfire serve is not running.");
      break;
    }
    try {
      process.kill(pid, "SIGTERM");
      // Wait up to 5s for graceful shutdown
      for (let i = 0; i < 50; i++) {
        await new Promise(r => setTimeout(r, 100));
        if (!isPidAlive(pid)) break;
      }
      if (isPidAlive(pid)) {
        console.error(`PID ${pid} did not exit within 5s — sending SIGKILL`);
        try { process.kill(pid, "SIGKILL"); } catch {}
      }
      try { require("fs").unlinkSync(SERVE_PID_FILE); } catch {}
      console.log(`hipfire serve stopped (PID ${pid})`);
    } catch (err: any) {
      console.error(`Failed to stop serve (PID ${pid}): ${err?.message ?? err}`);
      process.exit(1);
    }
    break;
  }
  case "run": {
    const model = rest[0];
    if (!model) { console.error("Usage: hipfire run <model> [flags] [prompt]\n\nFlags:\n  --temp <float>           Temperature (default 0.3)\n  --top-p <float>          Top-p sampling (default 0.8)\n  --repeat-penalty <float> Repeat penalty (default 1.05)\n  --max-tokens <int>       Max tokens to generate (default 512)\n  --image <path>           Image for VL models\n\nExamples:\n  hipfire run qwen3.5:9b \"Hello\"\n  hipfire run qwen3.5:9b --temp 0.7 --max-tokens 256 \"Write a poem\"\n  hipfire run qwen3.5:4b --image photo.png \"Describe this\""); process.exit(1); }
    // Parse --key value flags
    const flagDefs: Record<string, { default: number | string | undefined }> = {
      "--image": { default: undefined }, "--temp": { default: 0.3 },
      "--top-p": { default: 0.8 }, "--repeat-penalty": { default: 1.05 },
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
    const userAliases = loadUserAliases();
    if (Object.keys(userAliases).length > 0) {
      console.log("\nUser aliases (hipfire quantize --register):\n");
      for (const [tag, a] of Object.entries(userAliases)) {
        const where = a.local_path ?? (a.repo ? `${a.repo}:${a.file}` : a.file);
        console.log(`  ${tag.padEnd(22)} ${where}`);
      }
    }
    if (showRemote || local.length === 0) {
      console.log("\nAvailable models:\n");
      const localFiles = new Set(local.map(m => m.name));
      for (const [tag, entry] of Object.entries(REGISTRY)) {
        const status = localFiles.has(entry.file) ? " [downloaded]" : "";
        console.log(`  ${tag.padEnd(22)} ${entry.size_gb.toString().padStart(5)}GB  ${entry.desc}${status}`);
      }
      console.log("\nPull:     hipfire pull <model>      (e.g. hipfire pull qwen3.5:9b)");
      console.log("Quantize: hipfire quantize <hf-id>   (registers a local alias)");
    }
    break;
  }
  case "ps": {
    // List running hipfire-related processes: serve daemons, quantize jobs, uploads.
    const sh = (cmd: string) => {
      try { const r = Bun.spawnSync(["bash", "-c", cmd], { stdout: "pipe", stderr: "pipe" }); return r.stdout?.toString().trim() || ""; }
      catch { return ""; }
    };
    const grepPatterns = [
      "hipfire-quantize",        // quantizer binary
      "target/release/examples/daemon",  // inference daemon
      "target/release/examples/serve",   // http serve wrapper (if any)
      "cli/index.ts.*serve",     // bun CLI running serve
      "cli/index.ts.*quantize",  // bun CLI running quantize
      "hf upload schuttdev",     // HF uploads
    ];
    const groups: { label: string; pattern: string; entries: string[] }[] = [
      { label: "Inference daemon", pattern: "daemon", entries: [] },
      { label: "Quantize jobs", pattern: "quantize", entries: [] },
      { label: "HF uploads", pattern: "hf upload", entries: [] },
    ];
    const lines = sh(`ps -eo pid,etime,rss,args | grep -E '${grepPatterns.join("|")}' | grep -v grep`).split("\n").filter(Boolean);
    for (const line of lines) {
      const m = line.match(/^\s*(\d+)\s+(\S+)\s+(\d+)\s+(.+)$/);
      if (!m) continue;
      const [, pid, etime, rss, args] = m;
      const rssMb = (parseInt(rss) / 1024).toFixed(0);
      const shortArgs = args.length > 140 ? args.slice(0, 140) + "…" : args;
      const entry = `  ${pid.padStart(7)}  ${etime.padStart(10)}  ${rssMb.padStart(6)}M  ${shortArgs}`;
      if (/daemon/.test(args)) groups[0].entries.push(entry);
      else if (/quantize/.test(args)) groups[1].entries.push(entry);
      else if (/hf upload/.test(args)) groups[2].entries.push(entry);
    }
    let total = 0;
    for (const g of groups) total += g.entries.length;
    if (total === 0) {
      console.log("No hipfire processes running.");
      console.log("\nStart one:");
      console.log("  hipfire serve                # inference daemon");
      console.log("  hipfire quantize <hf-id>     # quantize a model");
      break;
    }
    console.log(`${total} hipfire process${total === 1 ? "" : "es"} running:\n`);
    console.log("  PID        ETIME       RSS     COMMAND");
    for (const g of groups) {
      if (g.entries.length === 0) continue;
      console.log(`\n[${g.label}]`);
      for (const e of g.entries) console.log(e);
    }
    // Show local serve port availability + detached PID (if any)
    const port = cfg.port;
    const portInUse = sh(`ss -tlnp 2>/dev/null | grep :${port}`);
    const detachedPid = readServePid();
    if (detachedPid) {
      console.log(`\nserve port ${port}: ACTIVE (detached, PID ${detachedPid})`);
      console.log(`  stop: hipfire stop    |    log: tail -f ${SERVE_LOG_FILE}`);
    } else if (portInUse) {
      console.log(`\nserve port ${port}: ACTIVE (foreground)`);
    } else {
      console.log(`\nserve port ${port}: free`);
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
    // ── Dep autodetect ──────────────────────────────────────
    // Tools we spawn during update aren't always in $PATH even when
    // installed — rustup lives at ~/.cargo/bin, bun at ~/.bun/bin, ROCm
    // at /opt/rocm/bin on most distros. Empirically the v620 update run
    // failed because the login shell's PATH is minimal while the user's
    // interactive shell loads those bindirs via profile snippets. We probe
    // well-known locations, augment process.env.PATH with any found dirs,
    // and error fast with an install hint if a required dep is missing.
    const findDep = (binary: string, extraDirs: string[]): string | null => {
      // 1. Already in PATH
      const inPath = Bun.spawnSync(["sh", "-c", `command -v ${binary}`], { stdout: "pipe", stderr: "pipe" });
      const found = (inPath.stdout?.toString() ?? "").trim();
      if (inPath.exitCode === 0 && found) return found;
      // 2. Distro-specific known locations
      for (const dir of extraDirs) {
        const path = join(dir, binary);
        if (existsSync(path)) return path;
      }
      return null;
    };
    const depsNeeded = [
      { name: "git",   dirs: ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"],
        hint: "Install git via your distro's package manager." },
      { name: "cargo", dirs: [join(process.env.HOME || "", ".cargo/bin"), "/usr/bin"],
        hint: "Install rustup: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" },
      { name: "hipcc", dirs: ["/opt/rocm/bin", "/opt/rocm-6.0.0/bin", "/opt/rocm-5.7.0/bin", "/usr/bin"],
        hint: "Install ROCm: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/" },
    ];
    const missing: { name: string; hint: string }[] = [];
    const augmentDirs = new Set<string>();
    const depAbsPath: Record<string, string> = {};
    for (const d of depsNeeded) {
      const p = findDep(d.name, d.dirs);
      if (!p) { missing.push(d); continue; }
      depAbsPath[d.name] = p;
      // Any found tool's directory goes onto PATH so spawned children (e.g.
      // cargo invoking rustc) see the rest of the toolchain.
      const dir = p.substring(0, p.lastIndexOf("/"));
      if (dir) augmentDirs.add(dir);
    }
    if (missing.length) {
      console.error("\nMissing required dependencies:");
      for (const d of missing) console.error(`  • ${d.name} — ${d.hint}`);
      console.error("\nAborting update. Install the above and retry `hipfire update`.");
      process.exit(1);
    }
    // bun dir too — its subtree helpers need to resolve bun during cargo builds.
    const bunPath = findDep("bun", [join(process.env.HOME || "", ".bun/bin"), "/usr/bin"]);
    if (bunPath) augmentDirs.add(bunPath.substring(0, bunPath.lastIndexOf("/")));
    if (augmentDirs.size) {
      const curr = (process.env.PATH || "").split(":").filter(Boolean);
      const fresh = [...augmentDirs].filter(d => !curr.includes(d));
      if (fresh.length) {
        process.env.PATH = [...fresh, ...curr].join(":");
        console.error(`  PATH augmented with: ${fresh.join(", ")}`);
      }
    }
    // Bun.spawnSync's command lookup uses the child's env PATH, which inherits
    // from process.env.PATH — but we've observed cases where a bare-name
    // lookup fails even after a mid-process PATH mutation. Using the absolute
    // path we resolved up-front sidesteps the issue entirely. Child processes
    // (cargo → rustc, rustc → cc, etc.) still need PATH augmented above.
    const GIT_BIN = depAbsPath["git"]!;
    const CARGO_BIN = depAbsPath["cargo"]!;
    const git = (args: string[]) => Bun.spawnSync([GIT_BIN, ...args], { cwd: repoDir, stdio: ["inherit", "inherit", "inherit"] });
    const gitOut = (args: string[]) => {
      const r = Bun.spawnSync([GIT_BIN, ...args], { cwd: repoDir, stdout: "pipe", stderr: "pipe" });
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
    // Sync the CLI FIRST, before the Rust build. The CLI is pure Bun/TS — it
    // doesn't depend on the daemon compiling. If the build fails later (ROCm
    // version mismatch, missing header, WSL quirks), the registry + bug fixes
    // in the CLI are already live so `hipfire pull`, `hipfire list`, and
    // config commands keep working. Previously the copy happened after the
    // cargo build, so a build failure left the CLI frozen at its install-time
    // version — users saw "unknown model" for entries added post-install.
    const { copyFileSync } = await import("fs");
    const exe = process.platform === "win32" ? ".exe" : "";
    const binDir = join(HIPFIRE_DIR, "bin");
    copyFileSync(join(repoDir, "cli/index.ts"), join(HIPFIRE_DIR, "cli/index.ts"));
    console.error("  CLI updated ✓");
    // Rebuild
    console.error("Rebuilding daemon (this may take a few minutes)...");
    const build = Bun.spawnSync(
      [CARGO_BIN, "build", "--release", "--features", "deltanet", "--example", "daemon", "--example", "infer", "--example", "run", "-p", "engine"],
      { cwd: repoDir, stdio: ["inherit", "inherit", "inherit"], env: { ...process.env } }
    );
    if (build.exitCode !== 0) {
      console.error("");
      console.error("  Daemon build failed. CLI is updated (so `hipfire pull`,");
      console.error("  `hipfire list`, `hipfire config` still work), but the");
      console.error("  daemon binary was NOT rebuilt.");
      console.error("");
      console.error("  To diagnose:  hipfire diag");
      console.error("  To retry:     cd ~/.hipfire/src && cargo build --release --features deltanet -p engine --example daemon");
      process.exit(1);
    }
    // Build the CPU quantizer binary too so `hipfire quantize` works out of the box.
    const buildQ = Bun.spawnSync(
      [CARGO_BIN, "build", "--release", "-p", "hipfire-quantize"],
      { cwd: repoDir, stdio: ["inherit", "inherit", "inherit"], env: { ...process.env } }
    );
    if (buildQ.exitCode !== 0) {
      console.error("  hipfire-quantize build failed (quantize subcommand won't work). Continuing.");
    }
    // Recopy binaries
    // Example binaries live under target/release/examples/
    for (const bin of ["daemon", "infer", "run"]) {
      const src = join(repoDir, `target/release/examples/${bin}${exe}`);
      const dst = join(binDir, `${bin}${exe}`);
      if (existsSync(src)) { copyFileSync(src, dst); }
    }
    // Workspace binaries (e.g. hipfire-quantize) live under target/release/
    for (const bin of ["hipfire-quantize"]) {
      const src = join(repoDir, `target/release/${bin}${exe}`);
      const dst = join(binDir, `${bin}${exe}`);
      if (existsSync(src)) { copyFileSync(src, dst); }
    }
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
      // Clear the persistent install cache — stale blobs here outlive a
      // version bump because the .hash sidecars only detect source drift
      // for the kernels that still exist, not orphans. Empirically, one
      // renamed-or-cache-key-changed kernel can linger as a stale blob
      // and get loaded by the new daemon at a fresh lookup key's
      // location, producing subtly wrong math (non-failing hash check
      // because the OLD blob's hash still matches the OLD source we no
      // longer ship). `/tmp/hipfire_kernels` dies at reboot; this one
      // doesn't, so it's the one that actually needs the cleanup.
      const { rmSync } = await import("fs");
      if (existsSync(kernelDst)) {
        try { rmSync(kernelDst, { recursive: true, force: true }); } catch {}
      }
      try { rmSync("/tmp/hipfire_kernels", { recursive: true, force: true }); } catch {}
      mkdirSync(kernelDst, { recursive: true });
      if (existsSync(kernelSrc)) {
        for (const f of readdirSync(kernelSrc)) {
          if (f.endsWith(".hsaco")) copyFileSync(join(kernelSrc, f), join(kernelDst, f));
        }
        console.error(`  Updated ${gpuArch} kernels ✓ (cache cleared)`);
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
      // Explicit env pass-through: Bun.spawnSync's default env inheritance
      // on some platforms (observed on Arch/Cachy) drops mid-run PATH
      // mutations when stdio: "inherit" is used. The daemon's kernel
      // precompile shells out to hipcc, which needs /opt/rocm/bin on PATH.
      const pc = Bun.spawnSync([daemonForPrecompile, "--precompile"], {
        stdio: ["inherit", "inherit", "inherit"],
        env: { ...process.env },
      });
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

          const ad = archDefaults(diag.arch || "unknown");
          console.log(`  kv default:  ${ad.kv_cache} (${ad.vram_gb}GB VRAM)`);
          const hasWmma = (diag.arch || "").startsWith("gfx11") || (diag.arch || "").startsWith("gfx12");
          console.log(`  WMMA:        ${hasWmma ? "yes (4.1x prefill)" : "no (FP16 packed, +15% prefill)"}`);

          const vram = diag.vram_total_mb;
          if (models.length === 0 && vram > 0) {
            const rec = vram < 4000 ? "qwen3.5:0.8b" : vram < 6000 ? "qwen3.5:4b" : "qwen3.5:9b";
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
  case "quantize": {
    const input = rest[0];
    if (!input || input === "-h" || input === "--help") {
      console.error(`Usage: hipfire quantize <hf-model-id | local-dir> [flags]

Flags:
  --format <mq4|mq6|q8>      Quantization format (repeatable — default: mq4)
  --both                     Shorthand for --format mq4 --format mq6
  -o, --output <path>        Output file (single format only)
  --output-dir <dir>         Directory for outputs (multi-format: required)
  --stem <name>              Override the output basename (default: input basename)
  --upload <owner/repo>      Push outputs to HuggingFace after quantize
  --create-repo              Create the HF repo if it doesn't exist
  --install                  Copy outputs into ~/.hipfire/models (so \`hipfire run\` finds them)
  --register <tag>           Add a local alias (e.g. my-finetune:4b) to ~/.hipfire/models.json

Formats:
  mq4   FWHT-rotated 4-bit, quality-gated — recommended for production
  mq6   FWHT-rotated 6-bit — higher quality, ~1.47x file size
  q8    Symmetric Q8 — reference/debugging

Examples:
  # Quantize any Qwen 3.5 model from HF, both formats, upload + install:
  hipfire quantize Jackrong/Qwopus3.5-4B-v3 --both \\
      --upload schuttdev/hipfire-qwopus-4b --create-repo \\
      --install --register qwopus:4b

  # Local fine-tune → MQ4:
  hipfire quantize ./my-finetune --format mq4 -o finetune.mq4

  # One-shot all formats from local dir:
  hipfire quantize ./model --format mq4 --format mq6 --output-dir ./out

The quantizer runs on CPU and takes minutes-to-tens-of-minutes
depending on model size. HF downloads cache at ~/.hipfire/hf-cache/.`);
      process.exit(input ? 0 : 1);
    }
    const formats: string[] = [];
    let output: string | undefined;
    let outputDir: string | undefined;
    let stem: string | undefined;
    let uploadRepo: string | undefined;
    let createRepo = false;
    let installLocal = false;
    let register: string | undefined;
    for (let i = 1; i < rest.length; i++) {
      const a = rest[i];
      if (a === "--format") {
        const f = rest[++i];
        if (!f) { console.error("--format requires a value"); process.exit(1); }
        formats.push(f);
      } else if (a === "--both") {
        formats.push("mq4", "mq6");
      } else if (a === "-o" || a === "--output") {
        output = rest[++i];
        if (!output) { console.error("--output requires a value"); process.exit(1); }
      } else if (a === "--output-dir") {
        outputDir = rest[++i];
        if (!outputDir) { console.error("--output-dir requires a value"); process.exit(1); }
      } else if (a === "--stem") {
        stem = rest[++i];
        if (!stem) { console.error("--stem requires a value"); process.exit(1); }
      } else if (a === "--upload") {
        uploadRepo = rest[++i];
        if (!uploadRepo || !/^[^/]+\/[^/]+$/.test(uploadRepo)) {
          console.error("--upload requires owner/repo (e.g. schuttdev/hipfire-foo)"); process.exit(1);
        }
      } else if (a === "--create-repo") {
        createRepo = true;
      } else if (a === "--install") {
        installLocal = true;
      } else if (a === "--register") {
        register = rest[++i];
        if (!register) { console.error("--register requires a tag (e.g. my-finetune:4b)"); process.exit(1); }
      } else {
        console.error(`Unknown argument: ${a}\nRun 'hipfire quantize --help' for usage.`);
        process.exit(1);
      }
    }
    if (formats.length === 0) formats.push("mq4");
    const validFormats = ["mq4", "mq6", "q8", "q8f16", "hfq4", "hfq4g256", "hfq6", "hfq6g256"];
    for (const f of formats) {
      if (!validFormats.includes(f)) {
        console.error(`Unsupported format: ${f}\nSupported: mq4, mq6, q8`);
        process.exit(1);
      }
    }
    // Dedupe preserving order (e.g. --both --format mq4 shouldn't quantize twice)
    const uniqFormats = Array.from(new Set(formats));
    await quantize(input, {
      formats: uniqFormats,
      output, outputDir, stem,
      uploadRepo, createRepo,
      installLocal,
      register,
    });
    break;
  }
  case "config": {
    // `hipfire config`                                  → global TUI
    // `hipfire config list|get|set|reset [...]`          → global scripting
    // `hipfire config <model:tag>`                       → per-model TUI
    // `hipfire config <model:tag> list|get|set|reset ...` → per-model scripting
    //
    // Disambiguate: first arg is a model tag if it's a known REGISTRY entry
    // (resolved) or matches the `name:tag` shape. Otherwise treat as action.
    let [firstArg, maybeKey, ...valueArgs] = rest;
    let modelScope: string | null = null;
    if (firstArg && !["list", "get", "set", "reset"].includes(firstArg)) {
      // If looks like a tag, scope to that model
      const resolved = resolveModelTag(firstArg);
      if (REGISTRY[resolved] || firstArg.includes(":")) {
        modelScope = resolved;
        [firstArg, maybeKey, ...valueArgs] = rest.slice(1);
      }
    }
    const action = firstArg;
    const key = maybeKey;
    const value = valueArgs.join(" ") || undefined;

    const validKeys = Object.keys(CONFIG_DEFAULTS) as (keyof HipfireConfig)[];

    // Per-model scripting helpers (shared between get/set/reset)
    const writePerModel = (k: PerModelKey, v: any) => {
      const all = loadPerModelConfigs();
      const cur = all[modelScope!] ?? {};
      (cur as any)[k] = v;
      all[modelScope!] = cur;
      savePerModelConfigs(all);
    };
    const unsetPerModel = (k: PerModelKey) => {
      const all = loadPerModelConfigs();
      const cur = all[modelScope!];
      if (cur && k in cur) {
        delete (cur as any)[k];
        if (Object.keys(cur).length === 0) delete all[modelScope!];
        savePerModelConfigs(all);
        return true;
      }
      return false;
    };

    if (!action) {
      // Bare invocation → TUI. The global TUI can signal "open_picker" when
      // the user selects [per-model configs]; we then loop between picker →
      // per-model TUI → picker until the user cancels out.
      if (modelScope) {
        await configTui(cfg, modelScope);
      } else {
        let state: "global" | "picker" = "global";
        let pendingTag: string | null = null;
        while (true) {
          if (state === "global") {
            const act = await configTui(cfg, null);
            if (act === "exit") break;
            state = "picker";
          } else {
            const picked = pendingTag ?? await modelPickerTui();
            pendingTag = null;
            if (!picked) { state = "global"; continue; }
            await configTui(cfg, picked);
            // After the per-model editor exits, return to the picker so the
            // user can tweak another model; Esc in the picker goes back to
            // global.
          }
        }
      }
    } else if (action === "list") {
      if (modelScope) {
        const ov = loadPerModelConfigs()[modelScope] ?? {};
        const merged = resolveModelConfig(modelScope);
        console.log(`Per-model config: ${modelScope}  (${PER_MODEL_CONFIG_PATH})\n`);
        for (const k of validKeys) {
          if (!(PER_MODEL_KEYS as readonly string[]).includes(k)) continue;
          const v = (merged as any)[k];
          const isOverridden = k in ov;
          const marker = isOverridden ? "(overridden)" : "(inherited)";
          console.log(`  ${k.padEnd(18)} ${String(v).padEnd(14)}${marker}`);
        }
        console.log(`\nInteractive: hipfire config ${modelScope}`);
        console.log(`Set:         hipfire config ${modelScope} set <key> <value>`);
        console.log(`Unset:       hipfire config ${modelScope} reset <key>`);
      } else {
        listConfig(cfg);
      }
    } else if (action === "get") {
      if (!key) { console.error(`Usage: hipfire config${modelScope ? ` ${modelScope}` : ""} get <key>`); process.exit(1); }
      if (!validKeys.includes(key as any)) { console.error(`Unknown key: ${key}\nValid keys: ${validKeys.join(", ")}`); process.exit(1); }
      if (modelScope) {
        if (!(PER_MODEL_KEYS as readonly string[]).includes(key)) {
          console.error(`${key} is not a per-model override (use global: hipfire config get ${key})`);
          process.exit(1);
        }
        const v = (resolveModelConfig(modelScope) as any)[key];
        console.log(v);
      } else {
        console.log(cfg[key as keyof HipfireConfig]);
      }
    } else if (action === "set") {
      if (!key || value === undefined) {
        const validForScope = modelScope ? PER_MODEL_KEYS : validKeys;
        console.error(`Usage: hipfire config${modelScope ? ` ${modelScope}` : ""} set <key> <value>\n\nKeys:\n` + (validForScope as readonly string[]).map((k: string) => `  ${k.padEnd(18)} (default: ${(CONFIG_DEFAULTS as any)[k]})`).join("\n"));
        process.exit(1);
      }
      if (!validKeys.includes(key as any)) { console.error(`Unknown key: ${key}\nValid keys: ${validKeys.join(", ")}`); process.exit(1); }
      if (modelScope && !(PER_MODEL_KEYS as readonly string[]).includes(key)) {
        console.error(`${key} is global-only (set via: hipfire config set ${key} <value>)`);
        process.exit(1);
      }
      const defaultVal = CONFIG_DEFAULTS[key as keyof HipfireConfig];
      const parsed = typeof defaultVal === "number" ? Number(value) : value;
      if (typeof defaultVal === "number" && isNaN(parsed as number)) { console.error(`${key} requires a number`); process.exit(1); }
      if (!validateConfigValue(key, parsed)) {
        const hints: Record<string, string> = {
          kv_cache: "one of: auto, q8, asym4, asym3, asym2 (turbo/turbo2/turbo3/turbo4 aliases also accepted)",
          flash_mode: "one of: auto, always, never (applies to Q8 path; asym modes are flash-only)",
          temperature: "number between 0 and 2",
          top_p: "number in (0, 1]",
          repeat_penalty: "number between 1.0 and 3.0",
          max_tokens: "integer between 1 and 131072",
          max_seq: "KV cache capacity (tokens). Integer 512-524288",
          thinking: "one of: on, off. Controls whether the model reasons in <think> blocks.",
          max_think_tokens: "integer 0-32768. Budget for reasoning tokens (0 = unlimited).",
          port: "integer between 1 and 65535",
          idle_timeout: "seconds of inactivity before serve unloads the model (0 = never, max 86400)",
          default_model: "non-empty model tag",
        };
        console.error(`${key} must be ${hints[key] || "valid"}`); process.exit(1);
      }
      if (modelScope) {
        writePerModel(key as PerModelKey, parsed);
        console.log(`${modelScope}: ${key} = ${parsed} (overridden)`);
      } else {
        (cfg as any)[key] = parsed;
        saveConfig(cfg);
        console.log(`${key} = ${parsed}`);
      }
    } else if (action === "reset") {
      if (modelScope) {
        // Per-model reset = remove the override so it falls back to global.
        if (key) {
          if (!validKeys.includes(key as any)) { console.error(`Unknown key: ${key}`); process.exit(1); }
          if (unsetPerModel(key as PerModelKey)) {
            console.log(`${modelScope}: ${key} override removed (inheriting global)`);
          } else {
            console.log(`${modelScope}: ${key} was not overridden`);
          }
        } else {
          const all = loadPerModelConfigs();
          delete all[modelScope];
          savePerModelConfigs(all);
          console.log(`${modelScope}: all overrides cleared`);
        }
      } else if (key) {
        if (!validKeys.includes(key as any)) { console.error(`Unknown key: ${key}`); process.exit(1); }
        (cfg as any)[key] = CONFIG_DEFAULTS[key as keyof HipfireConfig];
        saveConfig(cfg);
        console.log(`${key} reset to ${CONFIG_DEFAULTS[key as keyof HipfireConfig]}`);
      } else {
        saveConfig({ ...CONFIG_DEFAULTS });
        console.log("All config reset to defaults");
      }
    } else {
      console.error(`Usage: hipfire config${modelScope ? ` ${modelScope}` : ""} [list|get|set|reset]`);
    }
    break;
  }
  default: {
    // First-run hint: if no config, no models, show a friendly setup tip.
    // (Only when invoked with no args — still show full help text below.)
    if (!cmd) {
      const hasModels = existsSync(MODELS_DIR) && readdirSync(MODELS_DIR).length > 0;
      const hasConfig = existsSync(CONFIG_PATH);
      const isFirstRun = !hasModels && !hasConfig;
      if (isFirstRun) {
        console.log(`\x1b[1mWelcome to hipfire — LLM inference for AMD GPUs\x1b[0m`);
        console.log(`\nDetected GPU: \x1b[36m${DETECTED_ARCH || "unknown"}\x1b[0m · KV default: \x1b[36m${ARCH_DEFAULTS.kv_cache}\x1b[0m`);
        console.log(`\nFirst-run setup:`);
        console.log(`  1. Sanity-check your GPU:   \x1b[1mhipfire diag\x1b[0m`);
        console.log(`  2. Pull a model:            \x1b[1mhipfire pull qwen3.5:4b\x1b[0m`);
        console.log(`  3. Run your first prompt:   \x1b[1mhipfire run qwen3.5:4b "hello"\x1b[0m`);
        console.log(`  4. Tweak settings:          \x1b[1mhipfire config\x1b[0m  (interactive)`);
        console.log(`\nFull command list:\n`);
      }
    }
    console.log(`hipfire — LLM inference for AMD GPUs

  pull <model>          Download model from HuggingFace
  run <model> [prompt]  Generate text (auto-pulls; uses running serve if any)
  serve [port] [-d]     Start OpenAI-compatible server (-d = background daemon)
  stop                  Stop the background serve daemon
  quantize <hf-id|dir>  Quantize to MQ4/MQ6 (CPU) — with optional HF upload
  bench <model> [opts]  Benchmark tok/s (--exp for RDNA2 variant sweep, --runs N)
  profile [model]       Kernel efficiency profiler (--json, --kernel <name>)
  list [-r]             Show local models (-r: show available too)
  config                Interactive settings editor (TUI); also: config [list|set|get|reset]
  diag                  Diagnostics — GPU, VRAM, HIP version, kernels, models
  ps                    Show running hipfire processes (serve, quantize, uploads)
  rm <model>            Delete model
  update                Pull latest code, rebuild, update kernels

Models (MQ4 default: FWHT-rotated 4-bit, quality-gated):
  hipfire pull qwen3.5:4b            # 2.6GB, best speed/quality balance
  hipfire pull qwen3.5:9b            # 5.3GB, best quality for 8GB cards
  hipfire pull qwen3.5:27b           # 15GB, needs 16GB+ VRAM
  hipfire pull qwen3.5:0.8b          # 0.55GB, tiny footprint

MQ6 tags (higher quality, ~1.47× larger):
  hipfire pull qwen3.5:9b-mq6        # 7.3GB, higher quality 9B
  hipfire pull qwen3.5:27b-mq6       # 21GB, needs 24GB+ VRAM

Quick start:
  hipfire pull qwen3.5:4b
  hipfire run qwen3.5:4b "What is the capital of France?"
  hipfire serve

Quantize any Qwen 3.5 HF model (or local dir) — one-shot download + upload:
  hipfire quantize Qwen/Qwen3.5-4B
  hipfire quantize Jackrong/Qwopus3.5-4B-v3 --both \\
        --upload schuttdev/hipfire-qwopus-4b --create-repo \\
        --install --register qwopus:4b
  hipfire quantize ./my-finetune --format mq6 -o my-finetune.mq6`);
    break;
  }
}
