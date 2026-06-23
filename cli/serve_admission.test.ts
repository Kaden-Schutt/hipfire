// Bun-native unit tests for cli/serve_admission.ts — the pure serve-layer
// robustness logic (O2a). Direct import: serve_admission.ts is side-effect-free
// (no fs/env/network/daemon), unlike cli/index.ts.
//
// Run: bun test cli/serve_admission.test.ts

import { test, expect, describe } from "bun:test";
import {
  DEFAULT_MAX_REQUEST_BYTES,
  checkContentLength,
  BoundedBodyReader,
  BoundedLock,
  LockSaturatedError,
  parseServePidFile,
  serializeServePidRecord,
  validatePidOwnership,
  reapPlanForPlatform,
  epKvModeWarning,
  sanitizeDaemonName,
  parseListenInodesForPort,
  decideProcfsPortOwnership,
  type ServeBind,
  formatBind,
  bindFetchTarget,
  bindFromPidRecord,
  serveProbeHost,
} from "./serve_admission";

// ─── Task 2: isValidSocketPath ───────────────────────────────────────────────

import { isValidSocketPath } from "./serve_admission";

test("isValidSocketPath: empty string is valid (disabled)", () => {
  expect(isValidSocketPath("")).toBe(true);
});
test("isValidSocketPath: absolute path is valid", () => {
  expect(isValidSocketPath("/run/user/1000/hf.sock")).toBe(true);
});
test("isValidSocketPath: relative path is invalid", () => {
  expect(isValidSocketPath("run/hf.sock")).toBe(false);
});
test("isValidSocketPath: NUL byte is invalid", () => {
  expect(isValidSocketPath("/run/hf\0.sock")).toBe(false);
});
test("isValidSocketPath: over 255 chars is invalid", () => {
  expect(isValidSocketPath("/" + "a".repeat(255))).toBe(false);
});

// ─── Task 4: resolveServeBind ────────────────────────────────────────────────

import { resolveServeBind } from "./serve_admission";

const base = { cliSocketPath: null, cliHost: null, cliPort: null,
               cfgSocketPath: "", cfgHost: "127.0.0.1", cfgPort: 11435 };

test("resolveServeBind: CLI socket + CLI host is mutually exclusive (error)", () => {
  const r = resolveServeBind({ ...base, cliSocketPath: "/run/hf.sock", cliHost: "0.0.0.0" });
  expect("error" in r && r.error).toContain("mutually exclusive");
});

test("resolveServeBind: CLI socket + CLI port is mutually exclusive (error)", () => {
  const r = resolveServeBind({ ...base, cliSocketPath: "/run/hf.sock", cliPort: 9000 });
  expect("error" in r && r.error).toContain("mutually exclusive");
});

test("resolveServeBind: non-absolute CLI socket is an error", () => {
  const r = resolveServeBind({ ...base, cliSocketPath: "run/hf.sock" });
  expect("error" in r && r.error).toContain("absolute");
});

test("resolveServeBind: absolute CLI socket → unix bind", () => {
  const r = resolveServeBind({ ...base, cliSocketPath: "/run/hf.sock" });
  expect(r).toEqual({ bind: { kind: "unix", path: "/run/hf.sock" } });
});

test("resolveServeBind: CLI host/port → tcp bind (CLI wins over cfg)", () => {
  const r = resolveServeBind({ ...base, cliHost: "0.0.0.0", cliPort: 9000, cfgPort: 11435 });
  expect(r).toEqual({ bind: { kind: "tcp", host: "0.0.0.0", port: 9000 } });
});

test("resolveServeBind: cfg socket_path → unix bind when no CLI bind", () => {
  const r = resolveServeBind({ ...base, cfgSocketPath: "/run/cfg.sock" });
  expect(r).toEqual({ bind: { kind: "unix", path: "/run/cfg.sock" } });
});

test("resolveServeBind: nothing set → tcp from cfg defaults", () => {
  const r = resolveServeBind(base);
  expect(r).toEqual({ bind: { kind: "tcp", host: "127.0.0.1", port: 11435 } });
});

test("resolveServeBind: explicit CLI socket wins over cfg host/port", () => {
  const r = resolveServeBind({ ...base, cliSocketPath: "/run/cli.sock", cfgHost: "0.0.0.0", cfgPort: 9000 });
  expect(r).toEqual({ bind: { kind: "unix", path: "/run/cli.sock" } });
});

// ─── Task 1: ServeBind + pure bind helpers ───────────────────────────────────

test("formatBind: tcp renders host:port", () => {
  expect(formatBind({ kind: "tcp", host: "127.0.0.1", port: 11435 })).toBe("127.0.0.1:11435");
});

test("formatBind: tcp brackets IPv6 host", () => {
  expect(formatBind({ kind: "tcp", host: "::1", port: 11435 })).toBe("[::1]:11435");
});

test("formatBind: does not double-bracket an already-bracketed IPv6 host", () => {
  expect(formatBind({ kind: "tcp", host: "[::1]", port: 11435 })).toBe("[::1]:11435");
});

test("formatBind: unix renders unix:path", () => {
  expect(formatBind({ kind: "unix", path: "/run/hf.sock" })).toBe("unix:/run/hf.sock");
});

test("bindFetchTarget: tcp builds a probe URL via serveProbeHost", () => {
  const t = bindFetchTarget({ kind: "tcp", host: "0.0.0.0", port: 11435 }, "/health");
  expect(t.url).toBe("http://127.0.0.1:11435/health");
  expect(t.unix).toBeUndefined();
});

test("bindFetchTarget: tcp brackets an IPv6 host (else the URL is malformed)", () => {
  const t = bindFetchTarget({ kind: "tcp", host: "::1", port: 11435 }, "/health");
  expect(t.url).toBe("http://[::1]:11435/health");
});

test("bindFetchTarget: unix uses localhost placeholder + unix path", () => {
  const t = bindFetchTarget({ kind: "unix", path: "/run/hf.sock" }, "/v1/chat/completions");
  expect(t.url).toBe("http://localhost/v1/chat/completions");
  expect(t.unix).toBe("/run/hf.sock");
});

test("bindFromPidRecord: socketPath present → unix bind", () => {
  const b = bindFromPidRecord({ pid: 1, socketPath: "/run/hf.sock" });
  expect(b).toEqual({ kind: "unix", path: "/run/hf.sock" });
});

test("bindFromPidRecord: legacy/TCP record → tcp bind with defaults", () => {
  expect(bindFromPidRecord({ pid: 1 })).toEqual({ kind: "tcp", host: "127.0.0.1", port: 11435 });
  expect(bindFromPidRecord({ pid: 1, host: "0.0.0.0", port: 9000 }))
    .toEqual({ kind: "tcp", host: "0.0.0.0", port: 9000 });
});

test("serveProbeHost maps wildcard hosts to loopback", () => {
  expect(serveProbeHost("0.0.0.0")).toBe("127.0.0.1");
  expect(serveProbeHost("192.168.1.5")).toBe("192.168.1.5");
});

// ─── req-body-no-cap: Content-Length size check ─────────────────────────────

describe("checkContentLength", () => {
  test("within limit → no reject, length parsed", () => {
    const v = checkContentLength("1024", 64 * 1024);
    expect(v.reject).toBe(false);
    expect(v.length).toBe(1024);
  });

  test("over limit → reject (413) with parsed length", () => {
    const v = checkContentLength(String(100 * 1024 * 1024), 64 * 1024 * 1024);
    expect(v.reject).toBe(true);
    expect(v.length).toBe(100 * 1024 * 1024);
  });

  test("missing header → no reject, length null (stream with cap)", () => {
    const v = checkContentLength(null);
    expect(v.reject).toBe(false);
    expect(v.length).toBeNull();
  });

  test("empty header → no reject, length null", () => {
    expect(checkContentLength("").length).toBeNull();
  });

  test("unparseable header (chunked-ish) → no reject, length null", () => {
    const v = checkContentLength("not-a-number", 64);
    expect(v.reject).toBe(false);
    expect(v.length).toBeNull();
  });

  test("absurd length beyond safe integer → reject", () => {
    const v = checkContentLength("99999999999999999999999999", 64);
    expect(v.reject).toBe(true);
  });

  test("exactly at limit → not rejected", () => {
    const v = checkContentLength("64", 64);
    expect(v.reject).toBe(false);
    expect(v.length).toBe(64);
  });

  test("default cap is 64 MiB", () => {
    expect(DEFAULT_MAX_REQUEST_BYTES).toBe(64 * 1024 * 1024);
  });
});

describe("BoundedBodyReader (chunked cap)", () => {
  test("stays within cap across chunks", () => {
    const r = new BoundedBodyReader(100);
    expect(r.push(40)).toBe(true);
    expect(r.push(40)).toBe(true);
    expect(r.overflowed).toBe(false);
    expect(r.bytesRead).toBe(80);
  });

  test("overflows once cumulative bytes exceed cap", () => {
    const r = new BoundedBodyReader(100);
    expect(r.push(60)).toBe(true);
    expect(r.push(60)).toBe(false); // 120 > 100
    expect(r.overflowed).toBe(true);
  });

  test("latches overflowed (idempotent after breach)", () => {
    const r = new BoundedBodyReader(10);
    r.push(20);
    expect(r.push(1)).toBe(false);
    expect(r.overflowed).toBe(true);
  });
});

// ─── req-body-no-cap: streamed-read-BEFORE-lock ordering invariant ──────────
//
// The fetch handler in index.ts now reads + size-caps the FULL body (via
// BoundedBodyReader for EVERY body shape: header / chunked / absent / malformed
// Content-Length) BEFORE calling serveLock.acquire(). This test models that
// ordering with the pure pieces: a chunked/absent-Content-Length body that
// overflows the cap must 413 from the streamed read alone, with the lock NEVER
// acquired. A BoundedLock spy proves acquire() was not called on the reject path.
describe("streamed body read happens BEFORE the serve lock", () => {
  // Simulate the handler's pre-lock body stage: checkContentLength gives a
  // verdict, then (regardless of declared length) we stream through a cap. The
  // lock is only touched if we get past the body stage with no 413.
  function preLockBodyStage(
    contentLengthHeader: string | null,
    chunkBytes: number[],
    maxBytes: number,
    lock: BoundedLock,
  ): { status: number; lockAcquired: boolean } {
    let lockAcquired = false;
    const acquireSpy = () => { lockAcquired = true; return lock.acquire(); };

    const cl = checkContentLength(contentLengthHeader, maxBytes);
    if (cl.reject) return { status: 413, lockAcquired }; // header-based 413, pre-lock

    // ALWAYS stream through the cap (no req.json shortcut), exactly like index.ts.
    const reader = new BoundedBodyReader(maxBytes);
    let overflowed = false;
    for (const n of chunkBytes) {
      if (!reader.push(n)) { overflowed = true; break; }
    }
    if (overflowed) return { status: 413, lockAcquired }; // streamed 413, still pre-lock

    // Only NOW would the handler acquire the lock.
    acquireSpy();
    return { status: 200, lockAcquired };
  }

  test("chunked (no Content-Length) overflow → 413 with lock NEVER acquired", () => {
    const lock = new BoundedLock({ maxQueueDepth: 4 });
    const r = preLockBodyStage(null, [50, 50, 50], 100, lock); // 150 > 100
    expect(r.status).toBe(413);
    expect(r.lockAcquired).toBe(false);
    expect(lock.isBusy).toBe(false); // lock untouched
  });

  test("malformed Content-Length that lies small → still capped by streamed read pre-lock", () => {
    const lock = new BoundedLock({ maxQueueDepth: 4 });
    // Garbage header → checkContentLength returns length:null (no header trust),
    // and the streamed read catches the real oversize body BEFORE the lock.
    const r = preLockBodyStage("not-a-number", [80, 80], 100, lock);
    expect(r.status).toBe(413);
    expect(r.lockAcquired).toBe(false);
  });

  test("within-cap body → lock acquired AFTER the read completes", () => {
    const lock = new BoundedLock({ maxQueueDepth: 4 });
    const r = preLockBodyStage("60", [60], 100, lock);
    expect(r.status).toBe(200);
    expect(r.lockAcquired).toBe(true);
    expect(lock.isBusy).toBe(true); // acquired exactly once, after the body read
  });

  test("oversized declared Content-Length → header 413 with lock NEVER acquired", () => {
    const lock = new BoundedLock({ maxQueueDepth: 4 });
    const r = preLockBodyStage(String(10 * 1024), [], 1024, lock);
    expect(r.status).toBe(413);
    expect(r.lockAcquired).toBe(false);
  });
});

// ─── lock-no-backpressure: bounded admission queue ──────────────────────────

describe("BoundedLock", () => {
  test("uncontended acquire resolves immediately", async () => {
    const lock = new BoundedLock({ maxQueueDepth: 4 });
    await lock.acquire().promise;
    expect(lock.isBusy).toBe(true);
    expect(lock.queueDepth).toBe(0);
  });

  test("FIFO: queued waiter resolves on release", async () => {
    const lock = new BoundedLock({ maxQueueDepth: 4 });
    await lock.acquire().promise; // holder
    let resolved = false;
    const w = lock.acquire();
    w.promise.then(() => { resolved = true; });
    expect(lock.queueDepth).toBe(1);
    expect(resolved).toBe(false);
    lock.release();
    await Promise.resolve(); // let microtask flush
    await w.promise;
    expect(resolved).toBe(true);
  });

  test("rejects with 503 (LockSaturatedError) at queue cap", async () => {
    const lock = new BoundedLock({ maxQueueDepth: 2, maxWaitMs: 5000 });
    await lock.acquire().promise; // holder
    // Fill the queue to depth 2.
    const q1 = lock.acquire();
    const q2 = lock.acquire();
    q1.promise.catch(() => {});
    q2.promise.catch(() => {});
    expect(lock.queueDepth).toBe(2);
    // Third waiter → rejected.
    const q3 = lock.acquire();
    let err: any = null;
    await q3.promise.catch(e => { err = e; });
    expect(err).toBeInstanceOf(LockSaturatedError);
    expect(err.retryAfterSec).toBeGreaterThanOrEqual(1);
  });

  test("aborted waiter is removed from the queue (no dead resolve)", async () => {
    const lock = new BoundedLock({ maxQueueDepth: 4 });
    await lock.acquire().promise; // holder
    const w1 = lock.acquire();
    const w2 = lock.acquire();
    w1.promise.catch(() => {});
    expect(lock.queueDepth).toBe(2);
    w1.abort(); // client disconnected
    expect(lock.queueDepth).toBe(1);
    // Release hands the lock to w2 (the live waiter), NOT the dead w1.
    let w2Resolved = false;
    w2.promise.then(() => { w2Resolved = true; });
    lock.release();
    await w2.promise;
    expect(w2Resolved).toBe(true);
  });

  test("maxWaitMs times out a long-queued waiter with 503", async () => {
    const lock = new BoundedLock({ maxQueueDepth: 4, maxWaitMs: 20 });
    await lock.acquire().promise; // holder never releases
    const w = lock.acquire();
    let err: any = null;
    await w.promise.catch(e => { err = e; });
    expect(err).toBeInstanceOf(LockSaturatedError);
  });

  test("release with empty queue clears busy", async () => {
    const lock = new BoundedLock({ maxQueueDepth: 4 });
    await lock.acquire().promise;
    lock.release();
    expect(lock.isBusy).toBe(false);
  });
});

// ─── pid-reuse: pidfile record + ownership validation ───────────────────────

describe("parseServePidFile", () => {
  test("new JSON record", () => {
    const rec = parseServePidFile(JSON.stringify({ pid: 4242, startTime: 123, host: "0.0.0.0", port: 11435, token: "abc" }));
    expect(rec).not.toBeNull();
    expect(rec!.pid).toBe(4242);
    expect(rec!.token).toBe("abc");
    expect(rec!.legacy).toBe(false);
  });

  test("legacy bare-numeric pid", () => {
    const rec = parseServePidFile("9001\n");
    expect(rec).not.toBeNull();
    expect(rec!.pid).toBe(9001);
    expect(rec!.legacy).toBe(true);
  });

  test("garbage → null", () => {
    expect(parseServePidFile("not a pid")).toBeNull();
    expect(parseServePidFile("")).toBeNull();
    expect(parseServePidFile(null)).toBeNull();
    expect(parseServePidFile("{bad json")).toBeNull();
  });

  test("JSON without a valid pid → null", () => {
    expect(parseServePidFile(JSON.stringify({ host: "x" }))).toBeNull();
    expect(parseServePidFile(JSON.stringify({ pid: -1 }))).toBeNull();
  });

  test("round-trips through serialize", () => {
    const rec = { pid: 7, startTime: 555, host: "127.0.0.1", port: 11435, token: "tok" };
    const back = parseServePidFile(serializeServePidRecord({ ...rec, legacy: false }));
    expect(back!.pid).toBe(7);
    expect(back!.token).toBe("tok");
    expect(back!.legacy).toBe(false);
  });
});

// UNIFIED PORT-FIRST ownership model (BUG pid-reuse fix3). The validator now
// applies the SAME decision order to BOTH new-format and legacy bare-pid
// records, keyed on the resolved TARGET port (evidence.ownsPort is the probe of
// that port). Order: (1) port verdict is authoritative — owns→OWNED,
// not-own→REFUSE, for both formats; (2) port inconclusive → /health token;
// (3) both inconclusive → cmdline AND startTime fallback (never overrides port).
describe("validatePidOwnership", () => {
  test("dead pid → not owned (stale)", () => {
    const v = validatePidOwnership({ pid: 1, legacy: false }, {}, false);
    expect(v.owned).toBe(false);
  });

  // ── 1. PORT FIRST: authoritative for BOTH new and legacy ──

  test("new record, pid OWNS the target port → owned (definitive, port-first)", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false },
      { ownsPort: true },
      true,
    );
    expect(v.owned).toBe(true);
    expect(v.reason).toContain("port");
  });

  test("legacy record, pid OWNS the target port → owned (same port-first rule)", () => {
    // The prior gap: legacy had no record.port, so port was never probed. Now the
    // caller threads the TARGET port and the legacy pid is probed against it.
    const v = validatePidOwnership(
      { pid: 9001, legacy: true },
      { ownsPort: true },
      true,
    );
    expect(v.owned).toBe(true);
  });

  test("new record, does NOT own the target port → REFUSE (reused pid)", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false },
      { ownsPort: false },
      true,
    );
    expect(v.owned).toBe(false);
  });

  test("legacy alive but does NOT own the target port → REFUSE (reused pid, no longer killed on bare liveness)", () => {
    const v = validatePidOwnership(
      { pid: 9001, legacy: true },
      { ownsPort: false },
      true,
    );
    expect(v.owned).toBe(false);
  });

  test("PORT OVERRIDES cmdline+startTime: foreign-cmdline pid that OWNS the port → owned", () => {
    // A positive port verdict can never be overridden by cmdline/startTime.
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, startTime: 100 },
      { cmdline: "/usr/bin/postgres -D /var/lib/pg", procStartTime: 999, ownsPort: true },
      true,
    );
    expect(v.owned).toBe(true);
  });

  test("PORT OVERRIDES cmdline+startTime: matching daemon cmdline+startTime but does NOT own port → REFUSE", () => {
    // startTime is NEVER the sole signal when the port is probeable — a
    // not-own-port verdict refuses even with a perfect cmdline + startTime match.
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, startTime: 555 },
      { cmdline: "examples/daemon", procStartTime: 555, ownsPort: false },
      true,
    );
    expect(v.owned).toBe(false);
  });

  // ── 2. Port INCONCLUSIVE → /health token ──

  test("port inconclusive + health token confirms → owned", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, token: "secret" },
      { cmdline: "bun cli/index.ts serve", healthToken: "secret" }, // ownsPort undefined
      true,
    );
    expect(v.owned).toBe(true);
  });

  test("port inconclusive + health token MISMATCH → not owned (different daemon)", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, token: "secret" },
      { cmdline: "bun cli/index.ts serve", healthToken: "OTHER" },
      true,
    );
    expect(v.owned).toBe(false);
  });

  test("health token confirms even when cmdline would NOT match (token beats fallback)", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, token: "secret" },
      { cmdline: "/usr/bin/postgres", healthToken: "secret" },
      true,
    );
    expect(v.owned).toBe(true);
  });

  // ── 3. BOTH port AND /health inconclusive → cmdline AND startTime fallback ──

  test("fallback: cmdline + startTime BOTH match (port & health inconclusive) → owned (hung-serve)", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, startTime: 555 },
      { cmdline: "examples/daemon", procStartTime: 555 }, // no ownsPort, no token/health
      true,
    );
    expect(v.owned).toBe(true);
  });

  test("fallback: matching cmdline but NO startTime → refuse (cmdline alone insufficient)", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false },
      { cmdline: "/home/k/hipfire/target/release/examples/daemon" },
      true,
    );
    expect(v.owned).toBe(false);
  });

  test("fallback: startTime match but FOREIGN cmdline → refuse (reused pid)", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, startTime: 100 },
      { cmdline: "/home/k/hipfire/scripts/some-unrelated-tool --foo", procStartTime: 100 },
      true,
    );
    expect(v.owned).toBe(false);
  });

  test("fallback tightened cmdline: `hipfire chat` subcommand is NOT the serve daemon → refuse", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, startTime: 100 },
      { cmdline: "bun /home/k/hipfire/cli/index.ts chat", procStartTime: 100 },
      true,
    );
    expect(v.owned).toBe(false);
  });

  test("fallback tightened cmdline: real `cli/index.ts serve` + startTime IS matched", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, startTime: 100 },
      { cmdline: "bun /home/k/hipfire/cli/index.ts serve 0.0.0.0 11435", procStartTime: 100 },
      true,
    );
    expect(v.owned).toBe(true);
  });

  test("fallback: startTime MISMATCH with matching cmdline → refuse (reused pid)", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false, startTime: 111 },
      { cmdline: "examples/daemon", procStartTime: 999 },
      true,
    );
    expect(v.owned).toBe(false);
  });

  test("fallback: no port, no token, no cmdline/startTime evidence → refuse", () => {
    const v = validatePidOwnership(
      { pid: 1234, legacy: false },
      {},
      true,
    );
    expect(v.owned).toBe(false);
  });

  // ── legacy fallback (port & health both inconclusive) ──

  test("legacy alive, port inconclusive (no probe) → owned (best-effort, keeps old daemons stoppable)", () => {
    const v = validatePidOwnership({ pid: 9001, legacy: true }, {}, true);
    expect(v.owned).toBe(true);
  });
});

// ─── reap-linux-only: platform gate ─────────────────────────────────────────

describe("reapPlanForPlatform", () => {
  test("linux → supported", () => {
    const p = reapPlanForPlatform("linux");
    expect(p.supported).toBe(true);
    expect(p.procfsFallback).toBe(true);
  });

  test("darwin → unsupported, explicit note (no silent free claim)", () => {
    const p = reapPlanForPlatform("darwin");
    expect(p.supported).toBe(false);
    expect(p.note.toLowerCase()).toContain("unsupported");
  });

  test("win32 → unsupported, note says port not freed", () => {
    const p = reapPlanForPlatform("win32");
    expect(p.supported).toBe(false);
    expect(p.note.toLowerCase()).toContain("not freed");
  });
});

// ─── ep-ignores-kvmode: warning gate ────────────────────────────────────────

describe("epKvModeWarning", () => {
  test("tp=1 → no warning regardless of kv-mode", () => {
    expect(epKvModeWarning(1, "q8")).toBeNull();
  });

  test("tp>1 + default/auto/absent kv-mode → no warning", () => {
    expect(epKvModeWarning(4, "auto")).toBeNull();
    expect(epKvModeWarning(4, null)).toBeNull();
    expect(epKvModeWarning(4, "")).toBeNull();
  });

  test("tp>1 + non-default kv-mode → warning naming the mode + tp", () => {
    const w = epKvModeWarning(4, "asym4");
    expect(w).not.toBeNull();
    expect(w!).toContain("asym4");
    expect(w!).toContain("tp=4");
    expect(w!.toLowerCase()).toContain("ignored");
  });
});

// ─── reap-escape: daemon-name shell-injection guard ─────────────────────────

describe("sanitizeDaemonName", () => {
  test("plain allowlisted name passes through", () => {
    expect(sanitizeDaemonName("daemon")).toBe("daemon");
    expect(sanitizeDaemonName("hipfire-daemon")).toBe("hipfire-daemon");
    expect(sanitizeDaemonName("my_daemon.v2")).toBe("my_daemon.v2");
  });

  test("trims surrounding whitespace", () => {
    expect(sanitizeDaemonName("  daemon  ")).toBe("daemon");
  });

  test("absent/empty → default fallback", () => {
    expect(sanitizeDaemonName(undefined)).toBe("daemon");
    expect(sanitizeDaemonName(null)).toBe("daemon");
    expect(sanitizeDaemonName("")).toBe("daemon");
    expect(sanitizeDaemonName("   ")).toBe("daemon");
  });

  test("REJECTS shell metacharacters → falls back to default (no injection)", () => {
    // The classic injection: `daemon; rm -rf ~` would run arbitrary shell if
    // interpolated unescaped into pgrep/pkill/fuser.
    expect(sanitizeDaemonName("daemon; rm -rf ~")).toBe("daemon");
    expect(sanitizeDaemonName("$(touch /tmp/pwned)")).toBe("daemon");
    expect(sanitizeDaemonName("`id`")).toBe("daemon");
    expect(sanitizeDaemonName("daemon && curl evil")).toBe("daemon");
    expect(sanitizeDaemonName("daemon|cat")).toBe("daemon");
    expect(sanitizeDaemonName("a b")).toBe("daemon"); // space is not allowlisted
    expect(sanitizeDaemonName("daemon\nrm")).toBe("daemon");
  });

  test("custom fallback honored", () => {
    expect(sanitizeDaemonName("bad;name", "examples-daemon")).toBe("examples-daemon");
  });

  test("over-length name → fallback", () => {
    expect(sanitizeDaemonName("a".repeat(65))).toBe("daemon");
  });

  test("leading-dash name → fallback (would be parsed as a pgrep/pkill option)", () => {
    // "-" IS in the allowlist charset, so these pass the regex — but a name
    // beginning with "-" would be consumed by pgrep/pkill as an option flag
    // (e.g. -x, --help). Reject it.
    expect(sanitizeDaemonName("-x")).toBe("daemon");
    expect(sanitizeDaemonName("--help")).toBe("daemon");
    expect(sanitizeDaemonName("-daemon")).toBe("daemon");
    expect(sanitizeDaemonName("  -x  ")).toBe("daemon"); // post-trim leading dash
    expect(sanitizeDaemonName("-x", "examples-daemon")).toBe("examples-daemon");
    // A dash in the MIDDLE / END is still fine (real names like hipfire-daemon).
    expect(sanitizeDaemonName("hipfire-daemon")).toBe("hipfire-daemon");
    expect(sanitizeDaemonName("daemon-")).toBe("daemon-");
  });
});

// ─── BUG pid-reuse: authoritative procfs port-ownership ─────────────────────
//
// These exercise the pure helpers that back probePortOwner()'s procfs branch.
// The key regression they pin: when the port IS listening but the candidate pid
// holds NO matching socket inode, the verdict must be FALSE (not "inconclusive"
// → best-effort → recycled-pid mistaken for owner).

// A realistic /proc/net/tcp: header + one LISTEN row on port 11435 (0x2CAB,
// inode 555001) + one ESTABLISHED row on a different port. Columns:
// sl local rem st tx:rx tr:tm retrnsmt uid timeout inode ...
const PROC_NET_TCP_LISTEN_11435 =
  "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n" +
  "   0: 00000000:2CAB 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 555001 1 0000000000000000 100 0 0 10 0\n" +
  "   1: 0100007F:1234 0100007F:9ABC 01 00000000:00000000 00:00000000 00000000  1000        0 777002 1 0000000000000000 20 4 30 10 -1\n";

// Same view but with NO listener on the target port at all.
const PROC_NET_TCP_NO_LISTEN =
  "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n" +
  "   0: 0100007F:1234 0100007F:9ABC 01 00000000:00000000 00:00000000 00000000  1000        0 777002 1 0000000000000000 20 4 30 10 -1\n";

describe("parseListenInodesForPort", () => {
  test("returns the LISTEN socket inode for the target port", () => {
    expect(parseListenInodesForPort(PROC_NET_TCP_LISTEN_11435, 11435)).toEqual(["555001"]);
  });

  test("ignores non-LISTEN (established) rows on the same port", () => {
    // The row on port 0x1234 is st=01 (ESTABLISHED) → not a listener.
    expect(parseListenInodesForPort(PROC_NET_TCP_LISTEN_11435, 0x1234)).toEqual([]);
  });

  test("no listener on the port → empty", () => {
    expect(parseListenInodesForPort(PROC_NET_TCP_NO_LISTEN, 11435)).toEqual([]);
  });

  test("header-only / empty text → empty", () => {
    expect(parseListenInodesForPort("", 11435)).toEqual([]);
    expect(parseListenInodesForPort("  sl  local_address ...\n", 11435)).toEqual([]);
  });
});

describe("decideProcfsPortOwnership (authoritative verdict)", () => {
  test("port listening + candidate HOLDS the inode → owned (true)", () => {
    const listen = parseListenInodesForPort(PROC_NET_TCP_LISTEN_11435, 11435);
    expect(decideProcfsPortOwnership(listen, new Set(["555001"]))).toBe(true);
  });

  test("port listening + candidate does NOT hold the inode → NOT owned (false)", () => {
    // THE REGRESSION: candidate holds some/other sockets (or none) while the port
    // is owned by a different pid → must be a definite false, not inconclusive.
    const listen = parseListenInodesForPort(PROC_NET_TCP_LISTEN_11435, 11435);
    expect(decideProcfsPortOwnership(listen, new Set(["999999"]))).toBe(false);
    expect(decideProcfsPortOwnership(listen, new Set<string>())).toBe(false);
  });

  test("no listener on the port → port free → NOT owned (false)", () => {
    const listen = parseListenInodesForPort(PROC_NET_TCP_NO_LISTEN, 11435);
    expect(listen).toEqual([]);
    expect(decideProcfsPortOwnership(listen, new Set(["555001"]))).toBe(false);
  });
});
