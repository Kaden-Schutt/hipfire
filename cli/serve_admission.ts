// Pure, side-effect-free helpers for cli/index.ts's serve layer.
//
// Lifted out of index.ts so the request-admission / pid-ownership / platform-gate
// logic can be unit-tested directly without loading the full CLI module graph
// (which has top-level side effects: loadConfig(), process.exit on bad config,
// env reads). Same pattern as cli/chat_pure.ts + cli/registry_loader.ts.
//
// Run tests: `bun test cli/serve_admission.test.ts`
//
// NONE of these functions touch fs, env, the network, or the daemon. They take
// already-read values (a header string, a parsed record, process.platform) and
// return decisions. The thin I/O wrappers stay in index.ts.

// ─── BUG req-body-no-cap: request body size cap ─────────────────────────────

// Default max request body for /v1/chat/completions. 64 MiB is generous for an
// agent prompt carrying entire file bodies (Pi's write/edit tools) yet bounds a
// single client from pinning the serve lock with a multi-GB upload. Overridable
// via config key `max_request_bytes` / env HIPFIRE_MAX_REQUEST_BYTES.
export const DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024;

export interface ContentLengthVerdict {
  // true → reject with HTTP 413 BEFORE acquiring the serve lock.
  reject: boolean;
  // Parsed length when a valid Content-Length header was present; null for
  // chunked / absent / unparseable headers (caller must stream with a cap).
  length: number | null;
  // Human reason for logging / the 413 body.
  reason: string;
}

// Decide, from the Content-Length header alone, whether to 413 before touching
// the lock. A missing / chunked / malformed header returns reject:false with
// length:null — the caller must then read the body through a byte-counting cap
// (see BoundedBodyReader) because the declared length can't be trusted.
export function checkContentLength(
  header: string | null | undefined,
  maxBytes: number = DEFAULT_MAX_REQUEST_BYTES,
): ContentLengthVerdict {
  if (header == null || header === "") {
    return { reject: false, length: null, reason: "no content-length (stream with cap)" };
  }
  const trimmed = header.trim();
  // Reject only on a CLEANLY-PARSED over-limit value. A garbage header is not
  // trusted as a size signal — fall through to the streaming cap.
  if (!/^\d+$/.test(trimmed)) {
    return { reject: false, length: null, reason: "unparseable content-length (stream with cap)" };
  }
  const n = parseInt(trimmed, 10);
  if (!Number.isSafeInteger(n)) {
    // Absurdly large declared length — reject outright.
    return { reject: true, length: null, reason: `content-length ${trimmed} exceeds safe integer` };
  }
  if (n > maxBytes) {
    return { reject: true, length: n, reason: `content-length ${n} > max ${maxBytes}` };
  }
  return { reject: false, length: n, reason: "within limit" };
}

// Byte-counting cap for chunked / no-Content-Length bodies. Feed each chunk's
// byte length; once the running total exceeds the cap, `overflowed` latches true
// and the caller aborts the read with a 413 instead of buffering unboundedly.
export class BoundedBodyReader {
  private total = 0;
  overflowed = false;
  constructor(public readonly maxBytes: number = DEFAULT_MAX_REQUEST_BYTES) {}
  // Returns true if STILL within the cap (caller keeps reading); false once the
  // cap is breached (caller stops and 413s). Idempotent once overflowed.
  push(chunkBytes: number): boolean {
    if (this.overflowed) return false;
    this.total += Math.max(0, chunkBytes | 0);
    if (this.total > this.maxBytes) {
      this.overflowed = true;
      return false;
    }
    return true;
  }
  get bytesRead(): number { return this.total; }
}

// ─── BUG lock-no-backpressure: bounded admission queue ──────────────────────

export interface BoundedLockOptions {
  // Max number of waiters allowed to queue behind the in-flight holder. When the
  // queue is full a new acquire is REJECTED (503) instead of enqueued.
  maxQueueDepth: number;
  // Optional max wait (ms) before a queued waiter is failed with 503. 0/absent
  // disables the timeout (depth cap still applies).
  maxWaitMs?: number;
}

export class LockSaturatedError extends Error {
  // Seconds the client should wait before retrying (Retry-After header value).
  constructor(public readonly retryAfterSec: number, msg: string) {
    super(msg);
    this.name = "LockSaturatedError";
  }
}

interface Waiter {
  resolve: () => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout> | null;
  // Set true once the waiter is removed from the queue (resolved/rejected/aborted)
  // so a late settle is a no-op.
  settled: boolean;
}

// FIFO admission lock with a bounded queue. Drop-in for index.ts's
// acquireLock/releaseLock pair, but:
//   - rejects with LockSaturatedError (→ HTTP 503 + Retry-After) at capacity
//     instead of enqueuing unboundedly,
//   - supports a max-wait timeout,
//   - lets an aborted/disconnected client be removed from the queue so no dead
//     {resolve} entry is ever woken (which would deadlock the holder slot).
export class BoundedLock {
  private busy = false;
  private queue: Waiter[] = [];
  constructor(private readonly opts: BoundedLockOptions) {}

  get queueDepth(): number { return this.queue.length; }
  get isBusy(): boolean { return this.busy; }

  // Acquire the lock. Resolves when held. Rejects with LockSaturatedError when
  // the queue is already at maxQueueDepth, or after maxWaitMs elapses.
  // Returns an `abort()` you call if the client disconnects while still queued —
  // it removes the dead waiter so its slot frees for a live client.
  acquire(): { promise: Promise<void>; abort: () => void } {
    if (!this.busy) {
      this.busy = true;
      return { promise: Promise.resolve(), abort: () => {} };
    }
    if (this.queue.length >= this.opts.maxQueueDepth) {
      const retry = this.opts.maxWaitMs && this.opts.maxWaitMs > 0
        ? Math.max(1, Math.ceil(this.opts.maxWaitMs / 1000))
        : 1;
      return {
        promise: Promise.reject(
          new LockSaturatedError(retry, `serve queue full (depth ${this.queue.length}/${this.opts.maxQueueDepth})`),
        ),
        // Swallow the rejection if the caller never attaches a handler before
        // calling abort(); abort is a no-op here (nothing was enqueued).
        abort: () => {},
      };
    }
    let w!: Waiter;
    const promise = new Promise<void>((resolve, reject) => {
      w = { resolve, reject, timer: null, settled: false };
      if (this.opts.maxWaitMs && this.opts.maxWaitMs > 0) {
        w.timer = setTimeout(() => {
          if (w.settled) return;
          this.remove(w);
          w.settled = true;
          reject(new LockSaturatedError(
            Math.max(1, Math.ceil(this.opts.maxWaitMs! / 1000)),
            `serve queue wait exceeded ${this.opts.maxWaitMs}ms`,
          ));
        }, this.opts.maxWaitMs);
      }
      this.queue.push(w);
    });
    const abort = () => {
      if (w.settled) return;
      w.settled = true;
      if (w.timer) clearTimeout(w.timer);
      this.remove(w);
      // Reject so an awaiting caller unwinds; harmless if already returned.
      w.reject(new LockSaturatedError(1, "client disconnected while queued"));
    };
    return { promise, abort };
  }

  // Release the lock; hand it to the next live waiter (skipping any that were
  // aborted/timed-out and somehow lingered).
  release(): void {
    let next = this.queue.shift();
    while (next && next.settled) next = this.queue.shift();
    if (next) {
      next.settled = true;
      if (next.timer) clearTimeout(next.timer);
      next.resolve();
    } else {
      this.busy = false;
    }
  }

  private remove(w: Waiter): void {
    const i = this.queue.indexOf(w);
    if (i >= 0) this.queue.splice(i, 1);
  }

  // Total in-flight load: the held slot (if busy) plus everyone queued behind it.
  // This is the number the Dashboard wants ("how many requests are in the
  // serve right now"), distinct from `queueDepth` (waiters only).
  get inflight(): number { return (this.busy ? 1 : 0) + this.queue.length; }
}

// ─── O3c-1: live serve /stats body (pure, side-effect-free) ─────────────────
// Inputs the serve fetch handler gathers from its own state; output is the
// exact JSON shape returned by GET /stats. Kept here (not in index.ts) so it
// is unit-testable without importing the CLI's side-effectful module.
//
// Contract:
//   - `model` is the loaded-model tag or null (serve idle / no model) — never
//     a fabricated placeholder.
//   - `uptime_s` is whole seconds since the serve process started.
//   - `queue_depth` is the BoundedLock in-flight count (held + queued).
//   - `requests_served` is the cumulative count of admitted chat requests.
//   - `recent_tok_s` is OMITTED unless a finite, positive decode rate was
//     observed on the most recent completed generation (honest: absent until
//     real data exists).
export interface ServeStatsInput {
  model: string | null;
  startMs: number;
  nowMs: number;
  queueDepth: number;
  requestsServed: number;
  recentTokS?: number | null;
}
export interface ServeStatsBody {
  model: string | null;
  uptime_s: number;
  queue_depth: number;
  requests_served: number;
  recent_tok_s?: number;
}
export function buildStatsBody(s: ServeStatsInput): ServeStatsBody {
  const uptimeMs = Math.max(0, s.nowMs - s.startMs);
  const body: ServeStatsBody = {
    model: s.model ?? null,
    uptime_s: Math.floor(uptimeMs / 1000),
    queue_depth: Number.isFinite(s.queueDepth) && s.queueDepth > 0 ? Math.floor(s.queueDepth) : 0,
    requests_served: Number.isFinite(s.requestsServed) && s.requestsServed > 0 ? Math.floor(s.requestsServed) : 0,
  };
  if (typeof s.recentTokS === "number" && Number.isFinite(s.recentTokS) && s.recentTokS > 0) {
    body.recent_tok_s = Math.round(s.recentTokS * 100) / 100;
  }
  return body;
}

// ─── BUG pid-reuse: pidfile record + ownership validation ───────────────────

export interface ServePidRecord {
  pid: number;
  // ms epoch when this serve started (cheap dedup signal; the authoritative
  // ownership check is /proc starttime on Linux).
  startTime?: number;
  host?: string;
  port?: number;
  token?: string;
  // Non-empty when the serve bound a Unix domain socket instead of host:port.
  // Populated in Task 5; parsed in Task 7. Added here (Task 1) so bindFromPidRecord
  // compiles without a forward reference.
  socketPath?: string;
  // true when parsed from a legacy bare-numeric serve.pid (pre-record format).
  // Callers MUST do best-effort validation only (skip the new /proc checks) so
  // an old daemon stays stoppable.
  legacy?: boolean;
}

// Parse a serve.pid file body. Accepts BOTH the new JSON record AND a legacy
// bare-numeric pid (one line, just digits). Returns null on garbage.
export function parseServePidFile(raw: string | null | undefined): ServePidRecord | null {
  if (raw == null) return null;
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  // New format: a JSON object.
  if (trimmed.startsWith("{")) {
    try {
      const o = JSON.parse(trimmed);
      if (o && typeof o === "object" && Number.isInteger(o.pid) && o.pid > 0) {
        return {
          pid: o.pid,
          startTime: typeof o.startTime === "number" ? o.startTime : undefined,
          host: typeof o.host === "string" ? o.host : undefined,
          port: typeof o.port === "number" ? o.port : undefined,
          token: typeof o.token === "string" ? o.token : undefined,
          legacy: false,
        };
      }
      return null;
    } catch { return null; }
  }
  // Legacy format: a bare pid.
  if (/^\d+$/.test(trimmed)) {
    const pid = parseInt(trimmed, 10);
    if (pid > 0) return { pid, legacy: true };
  }
  return null;
}

// Serialize a serve.pid record (new JSON format). Pretty single-line.
export function serializeServePidRecord(rec: ServePidRecord): string {
  const { legacy, ...body } = rec;
  return JSON.stringify(body);
}

// Linux /proc evidence used to confirm a pid is actually OUR serve daemon and
// not a reused pid. All fields optional — a caller that couldn't read /proc
// passes them absent and gets a conservative verdict.
export interface PidEvidence {
  // Field 22 of /proc/<pid>/stat (process start time in clock ticks). When the
  // record carries a procStartTime we require an exact match.
  procStartTime?: number;
  // /proc/<pid>/cmdline (NUL-joined → space-joined is fine for matching).
  cmdline?: string;
  // Port-ownership probe of the RESOLVED target port (lsof/ss/procfs result):
  // true = the pid is the listener on that port, false = the port is owned by a
  // different pid / the pid is demonstrably not the owner, undefined = the probe
  // was inconclusive (no lsof/ss/procfs signal). The target port is threaded in
  // by the caller and applies to legacy records too (which carry no record.port).
  ownsPort?: boolean;
  // /health responded with this token (matches record.token).
  healthToken?: string;
}

export interface OwnershipVerdict {
  // true → safe to SIGTERM/SIGKILL this pid.
  owned: boolean;
  // When false: caller should unlink the stale pidfile and NOT kill the pid.
  reason: string;
}

// Substrings that identify a hipfire serve daemon in its cmdline. Configurable
// match — default covers the spawned `examples/daemon` compute process AND the
// bun CLI `serve` invocation (`… cli/index.ts serve …` or a `hipfire serve`
// wrapper). BUG pid-reuse: TIGHTENED from a bare "hipfire" substring (which
// matched any path containing the word — e.g. a checkout at /home/x/hipfire/…
// running an UNRELATED binary, or the `hipfire chat`/`quantize` subcommands) to
// the actual daemon / serve invocations only. A bare project-path match is no
// longer sufficient to authorize a kill.
export const DEFAULT_DAEMON_CMDLINE_MATCHERS = [
  "examples/daemon",   // the spawned Rust compute daemon
  "release/daemon",    // alt build layout
  "cli/index.ts serve", // bun CLI run-from-source: `bun … cli/index.ts serve`
  "hipfire serve",     // installed wrapper: `hipfire serve`
];

function cmdlineMatches(cmdline: string, matchers: string[]): boolean {
  const c = cmdline.toLowerCase();
  return matchers.some(m => c.includes(m.toLowerCase()));
}

// Decide whether `pid` (named by `record`) may be killed, given whatever
// evidence the caller could gather. UNIFIED PORT-FIRST ownership model (BUG
// pid-reuse fix3): the SAME logic applies to BOTH new-format and legacy
// bare-pid records, keyed on the RESOLVED bind/target port the stop/restart/
// serve-dedup call is operating on (the caller threads it in; evidence.ownsPort
// is the probe of THAT port, not necessarily record.port — legacy records carry
// no port, so the caller probes the target port instead).
//
// Contract — NEVER kill a pid that fails validation (caller unlinks the stale
// pidfile instead). Decision order, port verdict is authoritative:
//   1. PORT FIRST (authoritative, new AND legacy):
//        ownsPort === true   → OWNED (the pid IS the listener on our port).
//        ownsPort === false  → REFUSE (port owned by someone else / pid is
//                               demonstrably not the owner — reused pid).
//      A port verdict can NEVER be overridden by cmdline/startTime below.
//   2. Port INCONCLUSIVE (ownsPort === undefined): /health token match →
//        OWNED on match, REFUSE on mismatch (token present on both sides).
//   3. BOTH port AND /health inconclusive: last-resort fallback —
//        cmdline-match AND startTime-match → OWNED (same process instance;
//        hung-serve case). cmdline/startTime are NEVER the sole signal when the
//        port was probeable; they only speak when the port couldn't.
export function validatePidOwnership(
  record: ServePidRecord,
  evidence: PidEvidence,
  alive: boolean,
  matchers: string[] = DEFAULT_DAEMON_CMDLINE_MATCHERS,
): OwnershipVerdict {
  if (!alive) {
    return { owned: false, reason: "pid not alive — stale pidfile" };
  }

  // ── 1. PORT FIRST (authoritative for both new and legacy records) ──
  // A port verdict is final: it can never be overridden by cmdline/startTime.
  if (evidence.ownsPort === true) {
    return {
      owned: true,
      reason: (record.legacy ? "legacy pid owns the target port" : "pid owns the target port")
        + " (port-ownership, definitive)",
    };
  }
  if (evidence.ownsPort === false) {
    return {
      owned: false,
      reason: (record.legacy ? "legacy pid alive but does not own the target port — reused pid"
                             : "pid does not own the target port — reused pid"),
    };
  }

  // ── 2. Port INCONCLUSIVE: /health token is the next authority ──
  // A daemon that echoes our exact instance token IS our serve; a mismatch is a
  // hard refusal. (Legacy records carry no token, so this never fires for them.)
  if (record.token !== undefined && evidence.healthToken !== undefined) {
    if (evidence.healthToken === record.token) {
      return { owned: true, reason: "port inconclusive; /health token confirmed (definitive)" };
    }
    return { owned: false, reason: "port inconclusive; /health token mismatch — different daemon" };
  }

  // ── 3. BOTH port and /health inconclusive: last-resort fallback ──
  // cmdline AND startTime must BOTH agree (same process instance). This is the
  // hung-serve case where the pid is live but not listening yet / unprobeable.
  if (record.legacy) {
    // Legacy bare-pid records have neither cmdline-authoritative token nor a
    // recorded startTime to cross-check. With the port unprobeable we cannot
    // distinguish a reused pid from the real daemon → best-effort: trust
    // liveness (keeps old daemons stoppable when no port signal exists).
    return { owned: true, reason: "legacy pidfile, port inconclusive — best-effort ownership (alive)" };
  }

  // New record: require BOTH a matching cmdline AND a matching startTime. Either
  // alone is insufficient (a reused pid may share a generic daemon cmdline; a
  // startTime without a daemon cmdline is not our serve). Both present + both
  // agree → same process instance.
  const cmdlineKnown = evidence.cmdline !== undefined;
  const cmdlineOk = cmdlineKnown && cmdlineMatches(evidence.cmdline!, matchers);
  if (cmdlineKnown && !cmdlineOk) {
    return { owned: false, reason: "port inconclusive; cmdline does not match a hipfire daemon — reused pid" };
  }
  const startTimeKnown = record.startTime !== undefined && evidence.procStartTime !== undefined;
  const startTimeMatches = startTimeKnown && record.startTime === evidence.procStartTime;
  if (startTimeKnown && !startTimeMatches) {
    return { owned: false, reason: "port inconclusive; proc start-time mismatch — reused pid" };
  }
  if (cmdlineOk && startTimeMatches) {
    return { owned: true, reason: "port inconclusive; cmdline + proc start-time match (same instance)" };
  }

  // Neither port, nor /health, nor a (cmdline AND startTime) pair could confirm
  // ownership. Refuse — a reused pid is alive too; we will not kill without at
  // least one positive ownership signal. The caller unlinks the stale pidfile.
  return {
    owned: false,
    reason: "no port / token / (cmdline+startTime) confirmation — refusing to kill (possible reused pid)",
  };
}

// ─── BUG reap-escape: daemon-name shell-injection guard ─────────────────────

// HIPFIRE_DAEMON_NAME is interpolated into pgrep/pkill/fuser shell commands. An
// unsanitized value (e.g. `daemon; rm -rf ~`) would execute arbitrary shell.
// Allowlist a conservative process-name charset: letters, digits, dot, dash,
// underscore (covers real binary names like `examples/daemon`'s comm `daemon`,
// `hipfire-daemon`, etc.). Anything else → reject and fall back to the default
// so NO attacker-controlled metacharacter ever reaches the shell.
//
// Returns the sanitized name, or null when the input is empty/invalid (caller
// uses the default "daemon"). We REJECT rather than strip so a partially-stripped
// name can't silently match an unintended process.
const DAEMON_NAME_ALLOWED = /^[A-Za-z0-9._-]+$/;

export function sanitizeDaemonName(
  raw: string | null | undefined,
  fallback = "daemon",
): string {
  if (raw == null) return fallback;
  const trimmed = raw.trim();
  if (trimmed === "") return fallback;
  // pgrep matches comm (TASK_COMM_LEN-1 = 15 chars); a longer name never matches
  // anything, but we still must keep it shell-safe. Bound length defensively.
  if (trimmed.length > 64) return fallback;
  if (!DAEMON_NAME_ALLOWED.test(trimmed)) return fallback;
  // BUG reap-dash: a leading "-" (e.g. "-x", "--help") is in the allowlist set
  // but pgrep/pkill would parse it as an OPTION FLAG, not a process name. Reject
  // it. (The reap commands also use `-x --` so a dash can never be parsed as an
  // option — belt-and-suspenders — but rejecting here keeps the name honest.)
  if (trimmed.startsWith("-")) return fallback;
  return trimmed;
}

// ─── BUG reap-linux-only: platform gate for the reap commands ───────────────

export interface ReapPlan {
  // True when the pgrep/pkill/fuser path is available (Linux).
  supported: boolean;
  // True when a procfs port-owner fallback is worth attempting.
  procfsFallback: boolean;
  // Human note explaining what will / won't happen — surfaced to the operator so
  // a non-Linux `stop --force` does NOT silently claim the port was freed.
  note: string;
}

export function reapPlanForPlatform(platform: string): ReapPlan {
  if (platform === "linux") {
    return { supported: true, procfsFallback: true, note: "linux: pgrep/pkill/fuser available" };
  }
  // macOS has lsof but no /proc and different pkill semantics; Windows neither.
  if (platform === "darwin") {
    return {
      supported: false,
      procfsFallback: false,
      note: "darwin: pgrep -x/pkill -x/fuser are Linux-specific; orphan reap unsupported — use lsof -i to find the port owner manually",
    };
  }
  return {
    supported: false,
    procfsFallback: false,
    note: `${platform}: orphan reap is Linux-only; the port was NOT freed by this command`,
  };
}

// ─── BUG pid-reuse: authoritative procfs port-ownership decision ────────────

// The procfs branch of probePortOwner() used to bail to `undefined`
// ("inconclusive") whenever the candidate pid held no socket fds — even though
// /proc/net/tcp already showed the port owned by some OTHER pid. The validator
// then fell to best-effort, so a RECYCLED foreign pid could be treated as
// owning the port. These two pure helpers make the procfs view authoritative:
// whenever /proc is readable we can always answer true/false.

// Parse /proc/net/tcp{,6} text and collect the socket inode(s) of any LISTEN
// (st == "0A") entry whose local port matches `port`. The local column is
// "<addr-hex>:<port-hex>"; inode is column index 9. Returns the matching inode
// strings (empty = no listener on this port in this text).
export function parseListenInodesForPort(procNetTcpText: string, port: number): string[] {
  const portHex = port.toString(16).toUpperCase().padStart(4, "0");
  const inodes: string[] = [];
  const lines = procNetTcpText.split("\n");
  // Skip the header row (line 0) if present.
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cols = line.split(/\s+/);
    if (cols.length < 10) continue;
    const local = cols[1];   // "0100007F:2CB3"
    const state = cols[3];   // "0A" = LISTEN
    const inode = cols[9];
    if (state !== "0A") continue;
    const colon = local.lastIndexOf(":");
    if (colon < 0) continue;
    if (local.slice(colon + 1).toUpperCase() !== portHex) continue;
    inodes.push(inode);
  }
  return inodes;
}

// Decide port ownership from already-gathered procfs evidence. `listenInodes`
// is parseListenInodesForPort()'s result for the target port (across tcp+tcp6);
// `candidateSocketInodes` is the set of socket inodes the candidate pid holds
// (from /proc/<pid>/fd symlinks "socket:[<inode>]").
//
//   - no LISTEN entry on the port            → port is FREE → false (our serve gone)
//   - LISTEN entry + candidate holds it       → true  (candidate owns the port)
//   - LISTEN entry + candidate does NOT hold  → false (foreign/recycled pid)
//
// Always definite — never undefined — because a readable procfs is a complete
// view of the port's listeners.
export function decideProcfsPortOwnership(
  listenInodes: readonly string[],
  candidateSocketInodes: ReadonlySet<string>,
): boolean {
  if (listenInodes.length === 0) return false; // port not listening → not owned
  for (const ino of listenInodes) {
    if (candidateSocketInodes.has(ino)) return true;
  }
  return false; // port is listening, but not by this candidate
}

// ─── BUG ep-ignores-kvmode: EP + non-default kv-mode warning ────────────────

// Return a warning string when the operator requested both tp>1 (EP/multi-GPU
// load) AND a non-default --kv-mode, because the daemon EP path currently drops
// kv_mode_override. null → no warning (single-GPU, or default/absent kv-mode).
export function epKvModeWarning(tp: number, kvMode: string | null | undefined): string | null {
  if (!Number.isInteger(tp) || tp <= 1) return null;
  if (kvMode == null || kvMode === "" || kvMode === "auto") return null;
  return `--kv-mode ${kvMode} is currently IGNORED on the EP/multi-GPU path (tp=${tp}); `
    + `the daemon falls back to its per-arch default KV mode. This is a known limitation (O2b).`;
}

// ─── Task 2: socket path validation ─────────────────────────────────────────

// "" disables (TCP mode); otherwise an ABSOLUTE filesystem path <=255 bytes,
// no NUL. Absoluteness is required so detached children / later run|chat from
// other cwds resolve the SAME path the serve bound (spec §6.1, M10).
export function isValidSocketPath(value: string): boolean {
  return value.length <= 255
    && !value.includes("\0")
    && (value === "" || value.startsWith("/"));
}

// ─── Task 4: resolveServeBind ────────────────────────────────────────────────

// The SINGLE precedence point for "which transport does this serve bind".
// Explicit CLI intent always wins over config; within one invocation a CLI
// socket and an explicit CLI host/port are mutually exclusive (spec §6.3).
export function resolveServeBind(i: {
  cliSocketPath: string | null;
  cliHost: string | null;
  cliPort: number | null;
  cfgSocketPath: string;
  cfgHost: string;
  cfgPort: number;
}): { bind: ServeBind } | { error: string } {
  if (i.cliSocketPath !== null) {
    if (i.cliHost !== null || i.cliPort !== null) {
      return { error: "--socket-path and an explicit host/port are mutually exclusive" };
    }
    if (!isValidSocketPath(i.cliSocketPath) || i.cliSocketPath === "") {
      return { error: "--socket-path must be an absolute path (<=255 chars, no NUL)" };
    }
    return { bind: { kind: "unix", path: i.cliSocketPath } };
  }
  if (i.cliHost !== null || i.cliPort !== null) {
    return { bind: { kind: "tcp", host: i.cliHost ?? i.cfgHost, port: i.cliPort ?? i.cfgPort } };
  }
  if (i.cfgSocketPath !== "") {
    return { bind: { kind: "unix", path: i.cfgSocketPath } };
  }
  return { bind: { kind: "tcp", host: i.cfgHost, port: i.cfgPort } };
}

// ─── Task 1: ServeBind union + pure bind helpers ─────────────────────────────

// Mirrors index.ts DEFAULT_PORT (index.ts:63). Kept local to avoid a circular
// import (index.ts imports this file).
const DEFAULT_PORT = 11435;

// A serve listens on exactly one transport. The whole tcp-vs-unix decision is
// confined to this union and the helpers below (Approach B).
export type ServeBind =
  | { kind: "tcp"; host: string; port: number }
  | { kind: "unix"; path: string };

// Map a bind/listen host to the host a *client* should connect to: wildcard
// listen addresses are not connectable, so probe loopback instead.
export function serveProbeHost(host: string): string {
  if (host === "0.0.0.0" || host === "" || host === "::") return "127.0.0.1";
  return host;
}

export function formatBind(b: ServeBind): string {
  if (b.kind === "unix") return `unix:${b.path}`;
  // bracket IPv6 literals
  const h = b.host.includes(":") ? `[${b.host}]` : b.host;
  return `${h}:${b.port}`;
}

// How to reach a bind with fetch(). Bun ignores `unix: undefined`, so the tcp
// path is a plain URL with no extra option.
export function bindFetchTarget(b: ServeBind, path: string): { url: string; unix?: string } {
  if (b.kind === "unix") return { url: `http://localhost${path}`, unix: b.path };
  return { url: `http://${serveProbeHost(b.host)}:${b.port}${path}` };
}

// The single place a lifecycle command turns a tracked pid record into the bind
// it must operate on. socketPath wins; otherwise legacy/TCP with defaults.
export function bindFromPidRecord(rec: ServePidRecord): ServeBind {
  if (rec.socketPath) return { kind: "unix", path: rec.socketPath };
  return { kind: "tcp", host: rec.host ?? "127.0.0.1", port: rec.port ?? DEFAULT_PORT };
}
