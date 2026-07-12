// Dynamic registry loader (task #47).
//
// The CLI is compiled to a single binary by `bun build --compile`, which
// inlines cli/registry.json forever — shipped binaries never learn about new
// models. This module fetches registry/v1.json from the repo's raw GitHub
// URL with a 24h on-disk cache, falling back to (in order) fresh cache →
// network → stale cache → the bundled registry.json the binary was built
// with. The CLI must never get WORSE than the bundled data: any fetch /
// parse / validation failure silently keeps the fallback chain going.
//
// Side-effect-free module (no top-level IO) so bun tests can import it
// directly, and hipfire-tui can share both the loader and the cache file at
// ~/.hipfire/registry.cache.json.
//
// registry/v1.json is a strict superset of cli/registry.json (same
// models/aliases shape + schema_version/generated_at/sha256/size_bytes/
// arch_id/quant), so a validated dynamic registry can simply replace the
// bundled REGISTRY/ALIASES maps. See scripts/registry_gen.py.

import { readFileSync, writeFileSync, renameSync, mkdirSync, unlinkSync } from "fs";
import { dirname } from "path";

export const REGISTRY_SCHEMA_VERSION = 1;
export const DEFAULT_REGISTRY_URL =
  "https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/registry/v1.json";
export const REGISTRY_CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24h
export const REGISTRY_FETCH_TIMEOUT_MS = 3500; // never hang the CLI offline

export interface RegistrySidecarV1 {
  file: string;
  sha256?: string | null;
  size_bytes?: number | null;
}

/// Same legacy shape as index.ts's ModelEntry plus the additive v1 fields.
export interface RegistryModelEntryV1 {
  repo: string;
  file: string;
  size_gb: number;
  min_vram_gb: number;
  desc: string;
  triattn?: RegistrySidecarV1;
  mtp?: RegistrySidecarV1;
  /// Tool-call wire format expected by this model card. Missing/null lets the
  /// CLI use its conservative family heuristic.
  default_tool_format?: "hermes" | "qwen_xml" | null;
  dspark?: RegistrySidecarV1;
  sha256?: string | null;
  size_bytes?: number | null;
  arch_id?: number | null;
  quant?: string | null;
  /// Optional per-model KV-cache default. When present it takes precedence
  /// over the per-GPU archDefaults fallback in resolveKvMode (the registry is
  /// the per-model card). Must be one of REGISTRY_KV_MODE_VALUES — an invalid
  /// value rejects the whole entry (fail-closed, same as arch_id/quant).
  default_kv_mode?: string | null;
  /// Optional curated author-recommended INFERENCE settings, inherited from
  /// the model's parent card. Layered UNDER per-model models.json config and
  /// OVER the CLI global default (resolveModelConfig). Each field is optional;
  /// out-of-bounds values reject the whole entry (fail-closed, see validEntry).
  recommended_settings?: RegistryRecommendedSettings | null;
}

/// Curated author-recommended sampling + framing defaults for a registry model
/// (the "card recs"). Richer than the .hfq-baked generation_config: it can also
/// carry a default system prompt. All fields optional. Bounds enforced in
/// validEntry (fail-closed).
export interface RegistryRecommendedSettings {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  repeat_penalty?: number;
  system_prompt?: string;
}

/// Canonical KV-mode allowlist accepted as a per-model `default_kv_mode`.
/// MUST stay in sync with the CLI's resolveKvMode/validateConfigValue set
/// (cli/index.ts) and scripts/registry_gen.py KNOWN_KV_MODES. Includes the
/// legacy turbo* aliases since resolveKvMode still maps them.
export const REGISTRY_KV_MODE_VALUES: readonly string[] = [
  "auto",
  "q8",
  "asym4",
  "asym3",
  "asym2",
  "fwht4",
  "fwht3",
  "fwht2",
  "turbo",
  "turbo4",
  "turbo3",
  "turbo2",
];

export interface RegistryV1 {
  schema_version: number;
  generated_at: string;
  models: Record<string, RegistryModelEntryV1>;
  aliases: Record<string, string>;
}

export interface RegistryCacheFile {
  fetched_at: number; // epoch ms
  url: string;
  registry: RegistryV1;
}

export type RegistrySource = "cache" | "network" | "stale-cache" | "bundled";

export interface LoadResult {
  source: RegistrySource;
  /// null ⇒ caller keeps the bundled registry.
  registry: RegistryV1 | null;
}

// ─── pure validation ─────────────────────────────────────

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function validSidecar(v: unknown): boolean {
  return isRecord(v) && typeof v.file === "string" && v.file.length > 0;
}

function validEntry(v: unknown): v is RegistryModelEntryV1 {
  if (!isRecord(v)) return false;
  if (typeof v.repo !== "string") return false; // "" = local-only, allowed
  if (typeof v.file !== "string" || v.file.length === 0) return false;
  if (typeof v.size_gb !== "number" || !(v.size_gb >= 0)) return false;
  if (typeof v.min_vram_gb !== "number" || !(v.min_vram_gb >= 0)) return false;
  if (typeof v.desc !== "string") return false;
  if (v.triattn !== undefined && !validSidecar(v.triattn)) return false;
  if (v.mtp !== undefined && !validSidecar(v.mtp)) return false;
  if (
    v.default_tool_format !== undefined &&
    v.default_tool_format !== null &&
    v.default_tool_format !== "hermes" &&
    v.default_tool_format !== "qwen_xml"
  ) {
    return false;
  }
  if (v.dspark !== undefined && !validSidecar(v.dspark)) return false;
  // Fail-closed on default_kv_mode: a present value must be a known KV mode.
  // null/undefined means "no per-model recommendation" → arch fallback.
  if (v.default_kv_mode !== undefined && v.default_kv_mode !== null) {
    if (
      typeof v.default_kv_mode !== "string" ||
      !REGISTRY_KV_MODE_VALUES.includes(v.default_kv_mode)
    ) {
      return false;
    }
  }
  // Fail-closed bounds-check on recommended_settings: any present numeric knob
  // out of range, or a non-string system_prompt, rejects the whole entry.
  if (v.recommended_settings !== undefined && v.recommended_settings !== null) {
    if (!validRecommendedSettings(v.recommended_settings)) return false;
  }
  return true;
}

/// Bounds for the curated card recs. Each present field is range-checked so a
/// malformed registry never feeds an absurd sampler value into the daemon.
///   temperature 0..2, top_p 0..1, top_k > 0 (int), min_p 0..1,
///   presence_penalty 0..2, repeat_penalty 0.5..2, system_prompt = string.
function validRecommendedSettings(v: unknown): boolean {
  if (!isRecord(v)) return false;
  const numInRange = (
    x: unknown,
    lo: number,
    hi: number,
    intOnly = false,
  ): boolean => {
    if (typeof x !== "number" || !Number.isFinite(x)) return false;
    if (intOnly && !Number.isInteger(x)) return false;
    return x >= lo && x <= hi;
  };
  if (v.temperature !== undefined && !numInRange(v.temperature, 0, 2)) {
    return false;
  }
  if (v.top_p !== undefined && !numInRange(v.top_p, 0, 1)) return false;
  // top_k > 0 (exclusive) integer — 0 would disable sampling entirely.
  if (
    v.top_k !== undefined &&
    (!numInRange(v.top_k, 1, 100000, true))
  ) {
    return false;
  }
  if (v.min_p !== undefined && !numInRange(v.min_p, 0, 1)) return false;
  if (
    v.presence_penalty !== undefined &&
    !numInRange(v.presence_penalty, 0, 2)
  ) {
    return false;
  }
  if (
    v.repeat_penalty !== undefined &&
    !numInRange(v.repeat_penalty, 0.5, 2)
  ) {
    return false;
  }
  if (v.system_prompt !== undefined && typeof v.system_prompt !== "string") {
    return false;
  }
  return true;
}

/// Fail-closed structural validation. Returns the typed registry or null —
/// a registry that fails ANY check is rejected wholesale (we never serve a
/// half-broken model list when the bundled fallback is known-good).
/// `schema_version` must be exactly 1: a future v2 with breaking shape
/// changes must not be trusted by binaries that only understand v1.
export function validateRegistryV1(data: unknown): RegistryV1 | null {
  if (!isRecord(data)) return null;
  if (data.schema_version !== REGISTRY_SCHEMA_VERSION) return null;
  if (typeof data.generated_at !== "string") return null;
  if (!isRecord(data.models) || Object.keys(data.models).length === 0) return null;
  for (const entry of Object.values(data.models)) {
    if (!validEntry(entry)) return null;
  }
  if (!isRecord(data.aliases)) return null;
  const models = data.models as Record<string, RegistryModelEntryV1>;
  const aliases: Record<string, string> = {};
  for (const [k, v] of Object.entries(data.aliases)) {
    if (typeof v !== "string") return null;
    // Drop (don't fail on) aliases to tags this registry doesn't carry —
    // an alias is a convenience redirect, not load-bearing data.
    if (models[v] !== undefined) aliases[k] = v;
  }
  return {
    schema_version: data.schema_version,
    generated_at: data.generated_at,
    models,
    aliases,
  };
}

export function parseCacheFile(raw: string): RegistryCacheFile | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(data)) return null;
  if (typeof data.fetched_at !== "number" || !Number.isFinite(data.fetched_at)) return null;
  if (typeof data.url !== "string") return null;
  const registry = validateRegistryV1(data.registry);
  if (!registry) return null;
  return { fetched_at: data.fetched_at, url: data.url, registry };
}

export function cacheIsFresh(
  cache: Pick<RegistryCacheFile, "fetched_at">,
  nowMs: number,
  ttlMs: number = REGISTRY_CACHE_TTL_MS,
): boolean {
  // A fetched_at in the future (clock skew, restored backup) is NOT fresh —
  // treat it as stale so we re-fetch rather than trusting it for years.
  return cache.fetched_at <= nowMs && nowMs - cache.fetched_at < ttlMs;
}

// ─── load orchestration ──────────────────────────────────

export interface LoadOptions {
  cachePath: string;
  url?: string;
  ttlMs?: number;
  nowMs?: number;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
  readFile?: (path: string) => string; // throws if missing (fs semantics)
  writeFile?: (path: string, contents: string) => void;
}

function defaultWriteFile(path: string, contents: string): void {
  // Atomic-ish: tmp + rename, so a crash mid-write never leaves a torn cache.
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp.${process.pid}`;
  try {
    writeFileSync(tmp, contents);
    renameSync(tmp, path);
  } catch (err) {
    try {
      unlinkSync(tmp);
    } catch {}
    throw err;
  }
}

/// Fallback chain: fresh cache → network (writes cache) → stale cache →
/// bundled (registry: null). Never throws.
export async function loadDynamicRegistry(opts: LoadOptions): Promise<LoadResult> {
  const url = opts.url ?? DEFAULT_REGISTRY_URL;
  const ttlMs = opts.ttlMs ?? REGISTRY_CACHE_TTL_MS;
  const nowMs = opts.nowMs ?? Date.now();
  const timeoutMs = opts.timeoutMs ?? REGISTRY_FETCH_TIMEOUT_MS;
  const fetchImpl = opts.fetchImpl ?? fetch;
  const readFile = opts.readFile ?? ((p: string) => readFileSync(p, "utf8"));
  const writeFile = opts.writeFile ?? defaultWriteFile;

  let cache: RegistryCacheFile | null = null;
  try {
    cache = parseCacheFile(readFile(opts.cachePath));
  } catch {
    cache = null; // missing/unreadable cache file
  }
  // A cache fetched from a different URL (e.g. HIPFIRE_REGISTRY_URL override
  // changed) must not satisfy freshness for the current URL.
  if (cache && cache.url !== url) cache = null;

  if (cache && cacheIsFresh(cache, nowMs, ttlMs)) {
    return { source: "cache", registry: cache.registry };
  }

  try {
    const resp = await fetchImpl(url, { signal: AbortSignal.timeout(timeoutMs) });
    if (resp.ok) {
      const registry = validateRegistryV1(await resp.json());
      if (registry) {
        try {
          const cacheFile: RegistryCacheFile = { fetched_at: nowMs, url, registry };
          writeFile(opts.cachePath, JSON.stringify(cacheFile));
        } catch {
          // Cache write failure is non-fatal — we still have the registry.
        }
        return { source: "network", registry };
      }
    }
  } catch {
    // Offline / DNS / timeout / non-JSON — fall through.
  }

  if (cache) {
    return { source: "stale-cache", registry: cache.registry };
  }
  return { source: "bundled", registry: null };
}
