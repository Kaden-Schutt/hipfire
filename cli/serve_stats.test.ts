// Bun-native unit tests for the O3c-1 serve /stats body builder + a real
// no-model HTTP hit. buildStatsBody is pure (no daemon/GPU), so this runs on
// CPU with no TTY. Run: bun test cli/serve_stats.test.ts

import { test, expect, describe } from "bun:test";
import {
  buildStatsBody,
  BoundedLock,
  type ServeStatsInput,
} from "./serve_admission";

describe("buildStatsBody", () => {
  test("idle / no model loaded → model:null, queue 0, requests 0, no recent_tok_s", () => {
    const b = buildStatsBody({
      model: null,
      startMs: 1000,
      nowMs: 1000,
      queueDepth: 0,
      requestsServed: 0,
    });
    expect(b.model).toBeNull();
    expect(b.uptime_s).toBe(0);
    expect(b.queue_depth).toBe(0);
    expect(b.requests_served).toBe(0);
    expect("recent_tok_s" in b).toBe(false); // honest: omitted until real data
  });

  test("loaded model + uptime + queue + requests", () => {
    const b = buildStatsBody({
      model: "/home/u/.hipfire/models/qwen3.5-9b.mq4",
      startMs: 10_000,
      nowMs: 73_400, // 63.4s later
      queueDepth: 3,
      requestsServed: 42,
    });
    expect(b.model).toBe("/home/u/.hipfire/models/qwen3.5-9b.mq4");
    expect(b.uptime_s).toBe(63);
    expect(b.queue_depth).toBe(3);
    expect(b.requests_served).toBe(42);
  });

  test("recent_tok_s included only when finite + positive, rounded to 2dp", () => {
    const base: ServeStatsInput = {
      model: "m", startMs: 0, nowMs: 0, queueDepth: 0, requestsServed: 1,
    };
    expect(buildStatsBody({ ...base, recentTokS: 151.237 }).recent_tok_s).toBe(151.24);
    expect("recent_tok_s" in buildStatsBody({ ...base, recentTokS: 0 })).toBe(false);
    expect("recent_tok_s" in buildStatsBody({ ...base, recentTokS: -5 })).toBe(false);
    expect("recent_tok_s" in buildStatsBody({ ...base, recentTokS: NaN })).toBe(false);
    expect("recent_tok_s" in buildStatsBody({ ...base, recentTokS: null })).toBe(false);
  });

  test("negative/NaN clock skew + garbage counters clamp to non-negative ints", () => {
    const b = buildStatsBody({
      model: null,
      startMs: 5000,
      nowMs: 1000, // nowMs < startMs (clock skew)
      queueDepth: -2,
      requestsServed: Number.NaN,
    });
    expect(b.uptime_s).toBe(0);
    expect(b.queue_depth).toBe(0);
    expect(b.requests_served).toBe(0);
  });

  test("BoundedLock.inflight feeds queue_depth (held + queued)", () => {
    const lock = new BoundedLock({ maxQueueDepth: 8, maxWaitMs: 0 });
    expect(lock.inflight).toBe(0);
    lock.acquire(); // holds slot, not awaited
    expect(lock.inflight).toBe(1);
    lock.acquire(); // queued
    lock.acquire(); // queued
    expect(lock.inflight).toBe(3);
    const b = buildStatsBody({
      model: "m", startMs: 0, nowMs: 0, queueDepth: lock.inflight, requestsServed: 0,
    });
    expect(b.queue_depth).toBe(3);
  });
});

describe("/stats route — real HTTP hit, no model loaded", () => {
  // Mirrors the index.ts fetch handler's /stats branch exactly, without the
  // daemon/GPU. Proves the route serves valid JSON with model:null + queue 0
  // when serve is idle.
  test("GET /stats returns the idle shape over HTTP", async () => {
    const serveStartMs = Date.now() - 5_000;
    const serveLock = new BoundedLock({ maxQueueDepth: 8, maxWaitMs: 0 });
    const current: string | null = null; // no model loaded
    const requestsServed = 0;
    const recentTokS: number | null = null;

    const server = Bun.serve({
      port: 0, // ephemeral
      fetch(req) {
        const url = new URL(req.url);
        if (url.pathname === "/stats") {
          return Response.json(buildStatsBody({
            model: current,
            startMs: serveStartMs,
            nowMs: Date.now(),
            queueDepth: serveLock.inflight,
            requestsServed,
            recentTokS,
          }));
        }
        return new Response("not found", { status: 404 });
      },
    });
    try {
      const r = await fetch(`http://${server.hostname}:${server.port}/stats`);
      expect(r.status).toBe(200);
      const j = await r.json() as any;
      expect(j.model).toBeNull();
      expect(j.queue_depth).toBe(0);
      expect(j.requests_served).toBe(0);
      expect(j.uptime_s).toBeGreaterThanOrEqual(4);
      expect("recent_tok_s" in j).toBe(false);
    } finally {
      server.stop(true);
    }
  });
});
