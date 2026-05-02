// Pure helpers for cli/chat.ts. Lifted to a side-effect-free module so they can
// be unit-tested directly without loading the full CLI module graph (which has
// top-level side effects from `cli/index.ts`).
//
// Run tests: `bun test cli/chat_pure.test.ts`

// ─── Grapheme handling ──────────────────────────────────────────────────────

const SEGMENTER: any = (typeof (globalThis as any).Intl !== "undefined" && (Intl as any).Segmenter)
  ? new (Intl as any).Segmenter(undefined, { granularity: "grapheme" })
  : null;

export function graphemes(s: string): string[] {
  if (!s) return [];
  if (SEGMENTER) {
    const out: string[] = [];
    for (const seg of SEGMENTER.segment(s)) out.push(seg.segment);
    return out;
  }
  return Array.from(s);
}

// ─── Paste sanitization ─────────────────────────────────────────────────────

// Strip control bytes < 0x20 except \n (LF) and \t (TAB). Prevents pasted ANSI
// escape sequences from being interpreted by downstream renderers and prevents
// stray BEL/NUL/etc. from corrupting the input buffer.
export function sanitizePaste(s: string): string {
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c >= 32 || c === 0x0a || c === 0x09) out += s[i];
  }
  return out;
}

// ─── Token estimation ───────────────────────────────────────────────────────

// Rough English+code heuristic. Real tokenization happens server-side; this is
// only used for client-side context-fill warnings and tok/s display.
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3.5);
}

// ─── Tok/s sliding-window math ──────────────────────────────────────────────

// Pure sliding-window rate calculator. `times` is a list of token-arrival
// timestamps (ms); the caller is responsible for trimming entries older than
// `windowMs` before calling. Returns 0 for fewer than 2 samples.
export function computeTokPerSec(times: number[]): number {
  if (times.length < 2) return 0;
  const span = ((times[times.length - 1] ?? 0) - (times[0] ?? 0)) / 1000;
  if (span <= 0) return 0;
  return times.length / span;
}

// Trim out-of-window entries from `times` in place. Returns the modified array.
export function trimTokenWindow(times: number[], now: number, windowMs: number = 2000): number[] {
  const cutoff = now - windowMs;
  while (times.length > 0 && (times[0] ?? 0) < cutoff) {
    times.shift();
  }
  return times;
}

// ─── /trim logic ────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface TrimResult {
  kept: ChatMessage[];
  dropped: number;
  remainingTokens: number;
}

// Drop oldest user/assistant turns until the conversation fits under
// `targetPct` of `ctxLimit`. Always preserves a leading system message if
// present at index 0. Always keeps at least one message after the system slot.
export function trimMessages(
  messages: ChatMessage[],
  ctxLimit: number,
  targetPct: number = 0.5,
): TrimResult {
  const out = [...messages];
  const target = ctxLimit * (Number.isFinite(targetPct) && targetPct > 0 ? targetPct : 0.5);
  let used = out.reduce((s, m) => s + estimateTokens(m.content), 0);
  let dropped = 0;
  const firstIdx = (out[0]?.role === "system") ? 1 : 0;
  while (used > target && out.length > firstIdx + 1) {
    const removed = out.splice(firstIdx, 1)[0]!;
    used -= estimateTokens(removed.content);
    dropped++;
  }
  return { kept: out, dropped, remainingTokens: used };
}

// ─── ANSI stripping (NO_COLOR support) ──────────────────────────────────────
//
// Removes SGR (\x1b[...m), OSC 8 hyperlinks (\x1b]8;...\x1b\\text\x1b]8;;\x1b\\),
// and other CSI/private-mode sequences so output remains readable when colors
// are disabled. Honors https://no-color.org — set NO_COLOR=1 in the environment
// or pass --no-color on the command line.
//
// We sanitize at write-time rather than gating each SGR call site, so all
// current and future styling code stays unchanged and the same kill-switch
// applies to everything (including OSC 8 hyperlinks emitted by markdown links).
//
// OSC 8 format is: ESC ] 8 ; params ; URI ST text ESC ] 8 ; ; ST
// where ST (string terminator) is either ESC \ (BEL also accepted in spec).
// We collapse the wrapping markers and keep the visible text.

const SGR_RE = /\x1b\[[0-9;?]*[A-Za-z]/g;          // ESC [ params final-byte (any CSI)
const OSC8_OPEN_RE = /\x1b\]8;[^\x07\x1b]*(?:\x1b\\|\x07)/g; // ESC ] 8 ; ... ST
const OSC8_CLOSE_RE = /\x1b\]8;;(?:\x1b\\|\x07)/g;
const OSC_GENERIC_RE = /\x1b\][^\x07\x1b]*(?:\x1b\\|\x07)/g; // any other OSC

export function stripAnsi(s: string): string {
  if (!s) return s;
  return s
    .replace(OSC8_CLOSE_RE, "")    // close before open (close is more specific)
    .replace(OSC8_OPEN_RE, "")
    .replace(OSC_GENERIC_RE, "")
    .replace(SGR_RE, "");
}

// ─── Markdown rendering ─────────────────────────────────────────────────────

// Phase 1 markdown: fenced code blocks, inline code, bold, italic. ANSI SGR
// codes only; no syntax highlighting. Render at commit-time only — never on
// the streaming tail (partial delimiters cause styling pop-in).
//
// `fenceWidth` controls the horizontal-rule width above/below code fences;
// callers pass `Math.min(60, stdout.columns)` when rendering to a TTY.
export function renderMarkdown(text: string, fenceWidth: number = 60): string {
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_m: string, lang: string, code: string) => {
    const border = "\x1b[2m" + "─".repeat(Math.max(1, fenceWidth)) + "\x1b[0m";
    const label = lang ? `\x1b[2m[${lang}]\x1b[0m` : "\x1b[2m[code]\x1b[0m";
    return `\n${border}\n${label}\n${code}\n${border}`;
  });
  // Headings: # / ## / ### at start of line. Bold + bright cyan for #, bold
  // for ##, dim-bold for ###. Whole line gets the styling so wrap looks ok.
  text = text.replace(/^### +(.+)$/gm, (_m: string, inner: string) => `\x1b[1;2m${inner}\x1b[0m`);
  text = text.replace(/^## +(.+)$/gm, (_m: string, inner: string) => `\x1b[1m${inner}\x1b[0m`);
  text = text.replace(/^# +(.+)$/gm, (_m: string, inner: string) => `\x1b[1;36m${inner}\x1b[0m`);
  // Block quotes: `> body` → dim `>`, italic body. Anchored to start-of-line.
  text = text.replace(/^> +(.+)$/gm, (_m: string, inner: string) => `\x1b[2m>\x1b[0m \x1b[3m${inner}\x1b[0m`);
  // Bullet lists: `- item` or `* item` (with leading whitespace) → `• item`
  // with the bullet dimmed. Indentation preserved. Anchored to line start.
  text = text.replace(/^(\s*)[-*] +(.+)$/gm, (_m: string, indent: string, inner: string) => `${indent}\x1b[2m•\x1b[0m ${inner}`);
  // Numbered lists: `1. foo`, `12. bar` → dim the digits + dot.
  text = text.replace(/^(\s*)(\d+\.) +(.+)$/gm, (_m: string, indent: string, num: string, inner: string) => `${indent}\x1b[2m${num}\x1b[0m ${inner}`);
  // Bare URLs (http/https/file) — underline + OSC 8 hyperlink. Done BEFORE
  // markdown-link replacement so the URL inside [text](url) is protected by
  // the negative lookbehind on `(` and `[`. Stops at whitespace or trailing
  // bracket-style punctuation.
  text = text.replace(/(?<![(\[])\b(https?:\/\/[^\s)\]]+|file:\/\/[^\s)\]]+)/g, (_m: string, url: string) =>
    `\x1b]8;;${url}\x1b\\\x1b[4m${url}\x1b[0m\x1b]8;;\x1b\\`,
  );
  // Markdown links: [text](url) → underline text, dim parens with raw URL.
  // OSC 8 hyperlink: `\x1b]8;;url\x1b\\text\x1b]8;;\x1b\\` makes text
  // clickable in iTerm2/kitty/Wezterm/modern xterm; degrades to underline
  // in non-supporting terminals. Inner URL is wrapped raw (not via OSC 8)
  // to avoid double-wrapping when the visible text is shown in parens.
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m: string, label: string, url: string) =>
    `\x1b]8;;${url}\x1b\\\x1b[4m${label}\x1b[0m\x1b]8;;\x1b\\ \x1b[2m(${url})\x1b[0m`,
  );
  text = text.replace(/`([^`]+)`/g, (_m: string, code: string) => `\x1b[7m${code}\x1b[0m`);
  text = text.replace(/\*\*([^*]+)\*\*/g, (_m: string, inner: string) => `\x1b[1m${inner}\x1b[0m`);
  text = text.replace(/(?<!\*)\*(?!\*)([^*]+)\*(?!\*)/g, (_m: string, inner: string) => `\x1b[3m${inner}\x1b[0m`);
  return text;
}

// ─── Streaming-friendly code-fence detection ────────────────────────────────
//
// Detects the open/close lines of a fenced code block (` ```python `, ` ``` `)
// during line-by-line streaming. The full `renderMarkdown` regex needs the
// whole fence in one pass, which doesn't work when each committed line is
// rendered independently. This helper lets the streaming path style fence
// boundaries per-line without buffering the whole code block.

export interface FenceLineInfo {
  isFenceOpen: boolean;   // line is the opening ```lang
  isFenceClose: boolean;  // line is the closing ```
  lang: string;           // language tag from the open fence (empty if none)
}

export function detectFenceLine(line: string, currentlyInFence: boolean): FenceLineInfo {
  const trimmed = line.trimStart();
  if (!trimmed.startsWith("```")) {
    return { isFenceOpen: false, isFenceClose: false, lang: "" };
  }
  if (currentlyInFence) {
    // Inside a fence, ``` always closes (we don't try to handle ` ```lang `
    // mid-fence as a nested open — markdown doesn't support nesting either)
    return { isFenceOpen: false, isFenceClose: true, lang: "" };
  }
  // Outside a fence: ` ```python ` opens, language is whatever follows the ticks
  const lang = trimmed.slice(3).trim().split(/\s+/)[0] ?? "";
  return { isFenceOpen: true, isFenceClose: false, lang };
}

// Render a fence-open line as a dim rule + language label.
export function renderFenceOpen(lang: string, fenceWidth: number = 60): string {
  const border = "\x1b[2m" + "─".repeat(Math.max(1, fenceWidth)) + "\x1b[0m";
  const label = lang ? `\x1b[2m[${lang}]\x1b[0m` : "\x1b[2m[code]\x1b[0m";
  return `${border}\n${label}`;
}

// Render a fence-close line as a dim rule.
export function renderFenceClose(fenceWidth: number = 60): string {
  return "\x1b[2m" + "─".repeat(Math.max(1, fenceWidth)) + "\x1b[0m";
}

// ─── Bracketed paste state machine ──────────────────────────────────────────
//
// Bracketed paste arrives as: `\x1b[200~ ...content... \x1b[201~`. Both
// markers may be split across stdin chunks. This is a pure transducer: feed
// it stdin chunks, get back either { paste: string } when a complete paste is
// assembled, or { keystroke: string } for normal input. State persists across
// calls via the returned object.

export interface PasteParserState {
  inPaste: boolean;
  buf: string;
}

export interface PasteParseResult {
  state: PasteParserState;
  paste: string | null;       // non-null when a complete paste was assembled
  passthrough: string | null; // non-null for non-paste input to be handled normally
}

const PASTE_START = "\x1b[200~";
const PASTE_END = "\x1b[201~";

export function feedPasteParser(state: PasteParserState, chunk: string): PasteParseResult {
  if (!state.inPaste) {
    if (chunk.startsWith(PASTE_START)) {
      const rest = chunk.slice(PASTE_START.length);
      const endIdx = rest.indexOf(PASTE_END);
      if (endIdx !== -1) {
        // Whole paste arrived in one chunk.
        const paste = sanitizePaste(rest.slice(0, endIdx).replace(/\r\n?/g, "\n"));
        return { state: { inPaste: false, buf: "" }, paste, passthrough: null };
      }
      return {
        state: { inPaste: true, buf: sanitizePaste(rest.replace(/\r\n?/g, "\n")) },
        paste: null,
        passthrough: null,
      };
    }
    return { state, paste: null, passthrough: chunk };
  }

  // In paste mode: keep accumulating until we see PASTE_END.
  const endIdx = chunk.indexOf(PASTE_END);
  if (endIdx !== -1) {
    const paste = state.buf + sanitizePaste(chunk.slice(0, endIdx).replace(/\r\n?/g, "\n"));
    return { state: { inPaste: false, buf: "" }, paste, passthrough: null };
  }
  return {
    state: { inPaste: true, buf: state.buf + sanitizePaste(chunk.replace(/\r\n?/g, "\n")) },
    paste: null,
    passthrough: null,
  };
}

// ─── Input history with draft preservation ─────────────────────────────────
//
// readline-style history navigation. Saves the in-progress draft on first
// up-arrow, restores it when the user navigates back past the most recent
// entry with down-arrow. Pure state machine; caller owns rendering.

export interface HistoryState {
  history: string[];
  index: number;          // points at history entry to restore; == history.length means "draft"
  draft: string | null;   // saved in-progress buffer
}

export function historyUp(state: HistoryState, currentBuffer: string): { state: HistoryState; buffer: string } {
  if (state.index === 0 || state.history.length === 0) {
    return { state, buffer: currentBuffer };
  }
  const draft = (state.index === state.history.length && state.draft === null) ? currentBuffer : state.draft;
  const newIndex = state.index - 1;
  return {
    state: { history: state.history, index: newIndex, draft },
    buffer: state.history[newIndex] ?? "",
  };
}

export function historyDown(state: HistoryState, currentBuffer: string): { state: HistoryState; buffer: string } {
  if (state.index >= state.history.length) {
    return { state, buffer: currentBuffer };
  }
  const newIndex = state.index + 1;
  if (newIndex === state.history.length) {
    return {
      state: { history: state.history, index: newIndex, draft: null },
      buffer: state.draft ?? "",
    };
  }
  return {
    state: { history: state.history, index: newIndex, draft: state.draft },
    buffer: state.history[newIndex] ?? "",
  };
}

export function historySubmit(state: HistoryState, submitted: string): HistoryState {
  return {
    history: [...state.history, submitted],
    index: state.history.length + 1,
    draft: null,
  };
}
