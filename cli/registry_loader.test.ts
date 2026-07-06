// Bun-native tests for cli/registry_loader.ts (task #47).
//
// Direct import: registry_loader.ts is side-effect-free at module top level.
// loadDynamicRegistry takes injectable fetch/readFile/writeFile/nowMs, so
// every branch of the fallback chain (fresh cache → network → stale cache →
// bundled) is exercised hermetically — no network, no ~/.hipfire writes.

import { test, expect, describe } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import {
  validateRegistryV1,
  parseCacheFile,
  cacheIsFresh,
  loadDynamicRegistry,
  REGISTRY_CACHE_TTL_MS,
  type RegistryV1,
  type RegistryCacheFile,
} from "./registry_loader.ts";

const URL = "https://example.invalid/registry/v1.json";
const CACHE_PATH = "/nonexistent/registry.cache.json";

function goodRegistry(): RegistryV1 {
  return {
    schema_version: 1,
    generated_at: "2026-06-09T00:00:00Z",
    models: {
      "qwen3.5:9b": {
        repo: "schuttdev/hipfire-qwen3.5-9b",
        file: "qwen3.5-9b.mq4",
        size_gb: 5.3,
        min_vram_gb: 6,
        desc: "125 / 1720 tok/s",
        sha256: "829a84c708ee".padEnd(64, "0"),
        size_bytes: 5311808512,
        arch_id: 5,
        quant: "mq4",
      },
    },
    aliases: { "qwen3.5:latest": "qwen3.5:9b" },
  };
}

function cacheFile(fetchedAt: number, registry = goodRegistry()): string {
  const c: RegistryCacheFile = { fetched_at: fetchedAt, url: URL, registry };
  return JSON.stringify(c);
}

function fetchReturning(body: unknown, ok = true): typeof fetch {
  return (async () =>
    new Response(JSON.stringify(body), { status: ok ? 200 : 500 })) as unknown as typeof fetch;
}

function fetchThrowing(): typeof fetch {
  return (async () => {
    throw new Error("offline");
  }) as unknown as typeof fetch;
}

// ─── validateRegistryV1 ──────────────────────────────────

describe("validateRegistryV1", () => {
  test("accepts a well-formed v1 registry", () => {
    const r = validateRegistryV1(goodRegistry());
    expect(r).not.toBeNull();
    expect(r!.models["qwen3.5:9b"].quant).toBe("mq4");
    expect(r!.aliases["qwen3.5:latest"]).toBe("qwen3.5:9b");
  });

  test("accepts the real committed registry/v1.json", () => {
    const raw = JSON.parse(readFileSync(join(import.meta.dir, "../registry/v1.json"), "utf8"));
    const r = validateRegistryV1(raw);
    expect(r).not.toBeNull();
    expect(Object.keys(r!.models).length).toBeGreaterThan(10);
  });

  test("real v1.json is a strict superset of bundled registry.json", () => {
    const bundled = JSON.parse(readFileSync(join(import.meta.dir, "registry.json"), "utf8"));
    const v1 = JSON.parse(readFileSync(join(import.meta.dir, "../registry/v1.json"), "utf8"));
    const isSuperset = (old: unknown, neu: unknown): boolean => {
      if (old !== null && typeof old === "object" && !Array.isArray(old)) {
        if (neu === null || typeof neu !== "object" || Array.isArray(neu)) return false;
        return Object.entries(old as Record<string, unknown>).every(([k, v]) =>
          isSuperset(v, (neu as Record<string, unknown>)[k]),
        );
      }
      return old === neu;
    };
    expect(isSuperset(bundled.models, v1.models)).toBe(true);
    expect(isSuperset(bundled.aliases, v1.aliases)).toBe(true);
  });

  test("rejects wrong schema_version (fail-closed against future v2)", () => {
    expect(validateRegistryV1({ ...goodRegistry(), schema_version: 2 })).toBeNull();
    const noVersion = goodRegistry() as unknown as Record<string, unknown>;
    delete noVersion.schema_version;
    expect(validateRegistryV1(noVersion)).toBeNull();
  });

  test("rejects non-object / empty / structurally broken registries", () => {
    expect(validateRegistryV1(null)).toBeNull();
    expect(validateRegistryV1("[]")).toBeNull();
    expect(validateRegistryV1({ ...goodRegistry(), models: {} })).toBeNull();
    expect(validateRegistryV1({ ...goodRegistry(), models: [] })).toBeNull();
    expect(validateRegistryV1({ ...goodRegistry(), aliases: "x" })).toBeNull();
  });

  test("rejects wholesale when any single entry is malformed", () => {
    const r = goodRegistry();
    (r.models as Record<string, unknown>)["bad:tag"] = { repo: "x", file: "" };
    expect(validateRegistryV1(r)).toBeNull();

    const r2 = goodRegistry();
    (r2.models["qwen3.5:9b"] as unknown as Record<string, unknown>).size_gb = "5.3";
    expect(validateRegistryV1(r2)).toBeNull();

    const r3 = goodRegistry();
    (r3.models["qwen3.5:9b"] as unknown as Record<string, unknown>).triattn = { file: 42 };
    expect(validateRegistryV1(r3)).toBeNull();
  });

  test("allows empty repo (local-only) and optional v1 fields missing", () => {
    const r = goodRegistry();
    r.models["local:only"] = {
      repo: "",
      file: "local.mq4",
      size_gb: 1,
      min_vram_gb: 2,
      desc: "pre-release",
    };
    expect(validateRegistryV1(r)).not.toBeNull();
  });

  test("accepts a valid per-model default_kv_mode and null/missing", () => {
    const r = goodRegistry();
    (r.models["qwen3.5:9b"] as unknown as Record<string, unknown>).default_kv_mode = "q8";
    expect(validateRegistryV1(r)).not.toBeNull();

    const r2 = goodRegistry();
    (r2.models["qwen3.5:9b"] as unknown as Record<string, unknown>).default_kv_mode = null;
    expect(validateRegistryV1(r2)).not.toBeNull();
    // missing field already covered by goodRegistry() (no field set)
    expect(validateRegistryV1(goodRegistry())).not.toBeNull();
  });

  test("rejects an entry whose default_kv_mode is not in the KV allowlist", () => {
    const r = goodRegistry();
    (r.models["qwen3.5:9b"] as unknown as Record<string, unknown>).default_kv_mode = "lloyd9";
    expect(validateRegistryV1(r)).toBeNull();

    const r2 = goodRegistry();
    (r2.models["qwen3.5:9b"] as unknown as Record<string, unknown>).default_kv_mode = 8;
    expect(validateRegistryV1(r2)).toBeNull();
  });

  test("drops aliases pointing at missing tags instead of failing", () => {
    const r = goodRegistry();
    r.aliases["ghost"] = "not:a:tag";
    const v = validateRegistryV1(r);
    expect(v).not.toBeNull();
    expect(v!.aliases["ghost"]).toBeUndefined();
    expect(v!.aliases["qwen3.5:latest"]).toBe("qwen3.5:9b");
  });
});

// ─── cache parsing + freshness ───────────────────────────

describe("cache", () => {
  test("parseCacheFile round-trips a valid cache", () => {
    const c = parseCacheFile(cacheFile(1000));
    expect(c).not.toBeNull();
    expect(c!.fetched_at).toBe(1000);
    expect(c!.url).toBe(URL);
  });

  test("parseCacheFile rejects garbage, torn JSON, bad registry", () => {
    expect(parseCacheFile("")).toBeNull();
    expect(parseCacheFile("{ torn")).toBeNull();
    expect(parseCacheFile(JSON.stringify({ fetched_at: "soon", url: URL, registry: goodRegistry() }))).toBeNull();
    expect(parseCacheFile(JSON.stringify({ fetched_at: 1, url: URL, registry: { schema_version: 2 } }))).toBeNull();
  });

  test("cacheIsFresh respects TTL and rejects future timestamps", () => {
    const now = 10_000_000;
    expect(cacheIsFresh({ fetched_at: now - 1 }, now)).toBe(true);
    expect(cacheIsFresh({ fetched_at: now - REGISTRY_CACHE_TTL_MS + 1 }, now)).toBe(true);
    expect(cacheIsFresh({ fetched_at: now - REGISTRY_CACHE_TTL_MS }, now)).toBe(false);
    expect(cacheIsFresh({ fetched_at: now + 5000 }, now)).toBe(false); // clock skew
    expect(cacheIsFresh({ fetched_at: now - 10 }, now, 5)).toBe(false); // custom ttl
  });
});

// ─── loadDynamicRegistry fallback chain ──────────────────

describe("loadDynamicRegistry", () => {
  const now = 1_750_000_000_000;

  test("fresh cache short-circuits without fetching", async () => {
    let fetched = false;
    const result = await loadDynamicRegistry({
      url: URL,
      cachePath: CACHE_PATH,
      nowMs: now,
      readFile: () => cacheFile(now - 1000),
      writeFile: () => {},
      fetchImpl: (async () => {
        fetched = true;
        throw new Error("must not fetch");
      }) as unknown as typeof fetch,
    });
    expect(result.source).toBe("cache");
    expect(result.registry).not.toBeNull();
    expect(fetched).toBe(false);
  });

  test("stale cache → network fetch wins and rewrites the cache", async () => {
    const writes: Array<[string, string]> = [];
    const result = await loadDynamicRegistry({
      url: URL,
      cachePath: CACHE_PATH,
      nowMs: now,
      readFile: () => cacheFile(now - REGISTRY_CACHE_TTL_MS - 1),
      writeFile: (p, s) => writes.push([p, s]),
      fetchImpl: fetchReturning(goodRegistry()),
    });
    expect(result.source).toBe("network");
    expect(result.registry).not.toBeNull();
    expect(writes.length).toBe(1);
    expect(writes[0][0]).toBe(CACHE_PATH);
    const written = parseCacheFile(writes[0][1]);
    expect(written).not.toBeNull();
    expect(written!.fetched_at).toBe(now);
  });

  test("no cache → network", async () => {
    const result = await loadDynamicRegistry({
      url: URL,
      cachePath: CACHE_PATH,
      nowMs: now,
      readFile: () => {
        throw new Error("ENOENT");
      },
      writeFile: () => {},
      fetchImpl: fetchReturning(goodRegistry()),
    });
    expect(result.source).toBe("network");
  });

  test("fetch fails → stale cache still serves", async () => {
    const result = await loadDynamicRegistry({
      url: URL,
      cachePath: CACHE_PATH,
      nowMs: now,
      readFile: () => cacheFile(now - REGISTRY_CACHE_TTL_MS * 3),
      writeFile: () => {},
      fetchImpl: fetchThrowing(),
    });
    expect(result.source).toBe("stale-cache");
    expect(result.registry).not.toBeNull();
  });

  test("fetch fails + no cache → bundled (registry null)", async () => {
    const result = await loadDynamicRegistry({
      url: URL,
      cachePath: CACHE_PATH,
      nowMs: now,
      readFile: () => {
        throw new Error("ENOENT");
      },
      writeFile: () => {},
      fetchImpl: fetchThrowing(),
    });
    expect(result.source).toBe("bundled");
    expect(result.registry).toBeNull();
  });

  test("invalid network payload is rejected → stale cache", async () => {
    const result = await loadDynamicRegistry({
      url: URL,
      cachePath: CACHE_PATH,
      nowMs: now,
      readFile: () => cacheFile(now - REGISTRY_CACHE_TTL_MS - 1),
      writeFile: () => {
        throw new Error("must not cache invalid payload");
      },
      fetchImpl: fetchReturning({ schema_version: 2, models: {} }),
    });
    expect(result.source).toBe("stale-cache");
  });

  test("HTTP error status → bundled when no cache", async () => {
    const result = await loadDynamicRegistry({
      url: URL,
      cachePath: CACHE_PATH,
      nowMs: now,
      readFile: () => {
        throw new Error("ENOENT");
      },
      writeFile: () => {},
      fetchImpl: fetchReturning(goodRegistry(), /* ok= */ false),
    });
    expect(result.source).toBe("bundled");
  });

  test("cache from a different URL is ignored (env override changed)", async () => {
    const result = await loadDynamicRegistry({
      url: "https://other.invalid/v1.json",
      cachePath: CACHE_PATH,
      nowMs: now,
      readFile: () => cacheFile(now - 1000), // fresh, but for URL not other.invalid
      writeFile: () => {},
      fetchImpl: fetchThrowing(),
    });
    expect(result.source).toBe("bundled");
  });

  test("cache write failure does not poison the network result", async () => {
    const result = await loadDynamicRegistry({
      url: URL,
      cachePath: CACHE_PATH,
      nowMs: now,
      readFile: () => {
        throw new Error("ENOENT");
      },
      writeFile: () => {
        throw new Error("EROFS");
      },
      fetchImpl: fetchReturning(goodRegistry()),
    });
    expect(result.source).toBe("network");
    expect(result.registry).not.toBeNull();
  });
});
