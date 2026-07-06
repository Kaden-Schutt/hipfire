// Pure formatting helpers for the hipfire CLI.
//
// These live in their own module (NOT index.ts) so they can be unit-tested
// without triggering index.ts's top-level side effects (await
// initDynamicRegistry(), the command dispatch, mkdirSync, etc.). Everything
// here is deterministic and free of I/O.

// ─── Exit codes ─────────────────────────────────────────
// A small documented set so scripts can branch on hipfire's outcome.
export const EXIT = {
  OK: 0,       // success
  ERROR: 1,    // runtime error (download failed, daemon crash, bad model, ...)
  USAGE: 2,    // bad flag / missing required arg / unknown command
  SIGINT: 130, // interrupted (128 + SIGINT(2))
} as const;

// ─── Error-message formatter ────────────────────────────
// Turn any thrown value into a single clean human line. NEVER leaks a raw
// Bun/V8 stack to the user — the top-level guard prints
// `hipfire: <formatErrorMessage(err)>` + a `run hipfire diag` hint.
export function formatErrorMessage(err: unknown): string {
  if (err == null) return "unknown error";
  if (typeof err === "string") return collapseWs(err) || "unknown error";
  if (err instanceof Error) {
    // Prefer .message; fall back to the constructor name for message-less errors.
    const msg = (err.message ?? "").trim();
    if (msg) return collapseWs(stripStack(msg));
    return err.name || "Error";
  }
  if (typeof err === "object") {
    const m = (err as any).message;
    if (typeof m === "string" && m.trim()) return collapseWs(stripStack(m));
    try { return collapseWs(JSON.stringify(err)); } catch { /* fallthrough */ }
  }
  return collapseWs(String(err)) || "unknown error";
}

// If a message itself embedded a stack (e.g. "Boom\n  at foo (x.ts:1:2)"),
// keep only the first non-stack line so the user sees a clean one-liner.
function stripStack(msg: string): string {
  const firstLine = msg.split("\n").find(l => !/^\s*at\s/.test(l)) ?? msg.split("\n")[0];
  return firstLine ?? msg;
}

function collapseWs(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

// ─── Human-readable sizes / rates ───────────────────────
export function formatBytes(bytes: number): string {
  if (!isFinite(bytes) || bytes < 0) return "?";
  if (bytes < 1e3) return `${bytes} B`;
  if (bytes < 1e6) return `${(bytes / 1e3).toFixed(1)} KB`;
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

// Bytes/sec → "12.3 MB/s". Used by the pull progress line.
export function formatRate(bytesPerSec: number): string {
  if (!isFinite(bytesPerSec) || bytesPerSec <= 0) return "—";
  return `${formatBytes(bytesPerSec)}/s`;
}

// Seconds → compact "1h02m", "3m07s", "12s", "—" (unknown).
export function formatEta(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m}m${String(rem).padStart(2, "0")}s`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h${String(m).padStart(2, "0")}m`;
}

// Unicode progress bar. `frac` is clamped to [0,1]; unknown total → empty bar
// dashes. Uses 1/8th block sub-characters for smooth fill.
export function progressBar(frac: number, width = 24): string {
  if (!isFinite(frac)) return "─".repeat(width);
  const f = Math.max(0, Math.min(1, frac));
  const full = Math.floor(f * width);
  const remainder = f * width - full;
  const eighths = "▏▎▍▌▋▊▉";
  let bar = "█".repeat(full);
  if (full < width) {
    const idx = Math.floor(remainder * 8); // 0..7
    if (idx > 0) bar += eighths[idx - 1];
    bar = bar.padEnd(width, " ");
  }
  return bar.slice(0, width);
}

export interface ProgressFields {
  downloaded: number;
  total: number; // 0 = unknown
  bytesPerSec: number; // 0 = unknown
}

// Compose the full progress line (sans \r and trailing pad). Pure → testable.
//   "[████████▏           ]  45.2%   8.1 MB/s   ETA 1m12s   123/272 MB"
export function formatProgressLine(f: ProgressFields, barWidth = 20): string {
  const { downloaded, total, bytesPerSec } = f;
  const known = total > 0;
  const frac = known ? downloaded / total : NaN;
  const pct = known ? `${(frac * 100).toFixed(1)}%` : "?%";
  const bar = progressBar(known ? frac : NaN, barWidth);
  const rate = formatRate(bytesPerSec);
  const remaining = known ? Math.max(0, total - downloaded) : 0;
  const eta = known && bytesPerSec > 0 ? formatEta(remaining / bytesPerSec) : "—";
  const got = (downloaded / 1e6).toFixed(0);
  const tot = known ? (total / 1e6).toFixed(0) : "?";
  return `[${bar}] ${pct.padStart(6)}  ${rate.padStart(10)}  ETA ${eta.padStart(6)}  ${got}/${tot} MB`;
}
