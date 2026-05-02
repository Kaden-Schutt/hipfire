import type { HipfireConfig } from "./index.ts";
import { findModel, resolveModelTag, isServeUp } from "./index.ts";
import {
  graphemes, sanitizePaste, estimateTokens,
  computeTokPerSec, trimTokenWindow,
  trimMessages, renderMarkdown, stripAnsi,
  detectFenceLine, renderFenceOpen, renderFenceClose,
  feedPasteParser, type PasteParserState,
  historyUp, historyDown, historySubmit, type HistoryState,
  type ChatMessage,
} from "./chat_pure.ts";

interface ChatState {
  messages: ChatMessage[];
  inputBuf: string;
  inputCursor: number;            // grapheme index, not code-unit index
  history: HistoryState;          // input history with draft preservation
  streaming: boolean;
  committedLines: string[];
  tokPerSec: number;
  abortController: AbortController | null;
  lastAbortTime: number | null;
  modelTag: string;
  daemonPid: number | null;
  daemonPort: number;
  paste: PasteParserState;        // bracketed-paste accumulator
  escBuf: string;
  tokenTimes: number[];
  totalTokens: number;
  thinking: boolean;
  ctxLimit: number;
  cleanedUp: boolean;
}

export interface ChatTuiOptions {
  noColor?: boolean;  // explicit --no-color flag from CLI
}

export async function chatTui(tag: string, cfg: HipfireConfig, opts: ChatTuiOptions = {}): Promise<void> {
  const stdin = process.stdin;
  const stdout = process.stdout;

  if (!stdout.isTTY || !stdin.isTTY) {
    console.error("Error: hipfire chat requires an interactive terminal (TTY).");
    process.exit(1);
  }

  const modelPath = findModel(tag);
  if (!modelPath) {
    console.error(`Model not found: ${tag}`);
    console.error("Run 'hipfire list' to see available models, or 'hipfire pull <tag>' to download.");
    process.exit(1);
  }

  const modelTag = resolveModelTag(tag);

  // Chat-shaped floor for max_tokens. The global default (cli/index.ts:148) is
  // 512, tuned for one-shot `hipfire run` invocations; in chat that truncates
  // mid-sentence with no user-visible warning. Floor only — a deliberate
  // higher value from `hipfire config set max_tokens` or mid-session
  // `/set max_tokens N` still wins.
  const CHAT_MIN_MAX_TOKENS = 8192;
  if (cfg.max_tokens < CHAT_MIN_MAX_TOKENS) {
    cfg.max_tokens = CHAT_MIN_MAX_TOKENS;
  }

  const state: ChatState = {
    messages: [],
    inputBuf: "",
    inputCursor: 0,
    history: { history: [], index: 0, draft: null },
    streaming: false,
    committedLines: [],
    tokPerSec: 0,
    abortController: null,
    lastAbortTime: null,
    modelTag,
    daemonPid: null,
    daemonPort: cfg.port,
    paste: { inPaste: false, buf: "" },
    escBuf: "",
    tokenTimes: [],
    totalTokens: 0,
    thinking: false,
    // Real context limit comes from cfg.max_seq (registry-resolved per model).
    ctxLimit: cfg.max_seq && cfg.max_seq > 0 ? cfg.max_seq : 32768,
    cleanedUp: false,
  };

  // NO_COLOR support: https://no-color.org. Strips SGR + OSC 8 hyperlinks
  // at write-time so the per-site styling code stays untouched. Also auto-
  // disables when stdout/stderr aren't TTYs (already gated above for stdin
  // but redundancy is cheap). Honors:
  //   - explicit --no-color flag (opts.noColor)
  //   - NO_COLOR env var (any non-empty value, per the spec)
  //   - CLICOLOR=0 (de-facto-standard fallback)
  const colorOff = opts.noColor === true
    || (process.env.NO_COLOR !== undefined && process.env.NO_COLOR !== "")
    || process.env.CLICOLOR === "0";

  const w = colorOff
    ? (text: string) => stdout.write(stripAnsi(text))
    : (text: string) => stdout.write(text);
  const we = colorOff
    ? (text: string) => process.stderr.write(stripAnsi(text))
    : (text: string) => process.stderr.write(text);

  // ─── Daemon management ──────────────────────────────────

  const existingServe = await isServeUp(cfg.port);
  if (existingServe) {
    state.daemonPort = cfg.port;
    we(`[hipfire] Using existing serve on port ${cfg.port}\n`);
  } else {
    state.daemonPort = cfg.port;
    we(`[hipfire] Starting serve on port ${state.daemonPort}...\n`);

    const proc = Bun.spawn(
      [process.argv[0], process.argv[1], "serve", String(state.daemonPort)],
      {
        stdout: "pipe",
        // Pipe instead of inherit so daemon log lines don't bleed into the chat UI.
        // Drained to /dev/null below; tail -f ~/.hipfire/serve.log if you want them.
        stderr: "pipe",
        // Suppress the daemon's PID-file write so it doesn't clobber a long-lived
        // `hipfire serve -d`. The chat session owns this daemon's lifecycle directly
        // via state.daemonPid; `hipfire stop` should not touch it.
        env: { ...process.env, HIPFIRE_NO_PID_FILE: "1" },
      },
    );
    state.daemonPid = proc.pid ?? null;

    // Drain stderr/stdout to avoid buffer backpressure (and prevent UI bleed).
    if (proc.stderr) {
      const reader = (proc.stderr as ReadableStream).getReader();
      (async () => { try { while (true) { const { done } = await reader.read(); if (done) break; } } catch {} })();
    }
    if (proc.stdout) {
      const reader = (proc.stdout as ReadableStream).getReader();
      (async () => { try { while (true) { const { done } = await reader.read(); if (done) break; } } catch {} })();
    }

    const spin = ["|", "/", "-", "\\"];
    const deadline = Date.now() + 120_000;
    let si = 0;
    while (Date.now() < deadline) {
      await new Promise<void>(r => setTimeout(r, 500));
      if (await isServeUp(state.daemonPort)) break;
      si = (si + 1) % 4;
      we(`\r  Waiting for serve... ${spin[si]}       `);
    }
    we("\r\x1b[K");

    if (!await isServeUp(state.daemonPort)) {
      we("Daemon failed to start within 120s. Check logs.\n");
      if (state.daemonPid) {
        try { process.kill(state.daemonPid, "SIGTERM"); } catch {}
      }
      process.exit(1);
    }
    we(`[hipfire] Serve ready on port ${state.daemonPort}\n`);
  }

  // ─── Raw mode setup ─────────────────────────────────────

  stdin.setRawMode!(true);
  stdin.resume();
  stdin.setEncoding("utf8");
  w("\x1b[?2004h"); // Bracketed paste

  // Handler for SIGWINCH; intentionally a no-op since we don't manage scrollback,
  // but registered so we can deregister it cleanly on exit (and so we don't
  // inherit any default behavior from the parent shell).
  const onResize = () => {};
  process.stdout.on("resize", onResize);

  // ─── Cleanup (idempotent, runnable from finally OR signal handler) ─────────

  const cleanup = () => {
    if (state.cleanedUp) return;
    state.cleanedUp = true;
    if (state.abortController) {
      try { state.abortController.abort(); } catch {}
    }
    if (state.daemonPid) {
      try { process.kill(state.daemonPid, "SIGTERM"); } catch {}
    }
    // Restore terminal: leave raw mode FIRST, then disable bracketed paste —
    // matches xterm spec ordering and avoids stuck mode on exotic terminals.
    try { stdin.setRawMode!(false); } catch {}
    try { stdin.pause(); } catch {}
    try { stdin.removeAllListeners("data"); } catch {}
    try { stdin.removeAllListeners("close"); } catch {}
    try { stdout.write("\x1b[?2004l"); } catch {}
    try { stdout.write("\x1b[?25h"); } catch {}  // ensure cursor visible
    try { process.stdout.off("resize", onResize); } catch {}
  };

  const onSignal = (signo: string) => {
    cleanup();
    try { stdout.write(`\n[chat] received ${signo}, exiting.\n`); } catch {}
    process.exit(0);
  };
  process.on("SIGINT", () => onSignal("SIGINT"));
  process.on("SIGTERM", () => onSignal("SIGTERM"));
  process.on("SIGHUP", () => onSignal("SIGHUP"));

  // ─── Markdown rendering ─────────────────────────────────
  // Applied only to COMMITTED lines (full lines that ended with \n) and to the
  // final flush of the response. Never to the streaming tail — partial markdown
  // delimiters cause flicker as styling pops in/out.

  const md = (text: string): string => renderMarkdown(text, Math.min(60, stdout.columns ?? 60));

  // ─── Input line rendering ───────────────────────────────
  // Uses real terminal cursor (\x1b[?25h + position reporting) instead of
  // inverse-video — respects user's cursor-shape preferences and works on
  // light themes.

  function renderInputLine() {
    const lines = state.inputBuf.split("\n");
    const curLineRaw = lines[lines.length - 1] || "";
    const curLineGraphemes = graphemes(curLineRaw);

    // Where in the (multi-line) buffer does the current line start? Find by
    // counting graphemes per logical line.
    let cursorInLine = state.inputCursor;
    for (let i = 0; i < lines.length - 1; i++) {
      cursorInLine -= graphemes(lines[i]!).length + 1; // +1 for the \n
    }
    cursorInLine = Math.max(0, Math.min(cursorInLine, curLineGraphemes.length));

    // Hide cursor while we redraw, then show + position at the end.
    w("\x1b[?25l");
    w("\r\x1b[K");
    w("\x1b[1;36m>\x1b[0m ");

    let prefixGraphemes = 2; // "> "
    if (lines.length > 1) {
      const indicator = `[${lines.length} lines] `;
      w(`\x1b[2m${indicator}\x1b[0m`);
      prefixGraphemes += indicator.length;
    }

    w(curLineGraphemes.join(""));

    // Move terminal cursor to logical position.
    // Note: this assumes 1-column-wide graphemes for cursor positioning.
    // CJK/emoji wide-glyph cursor positioning is handled by the terminal's
    // own width tracking when we use cursor-back from end-of-line.
    const back = curLineGraphemes.length - cursorInLine;
    if (back > 0) w(`\x1b[${back}D`);
    w("\x1b[?25h");
  }

  // ─── Slash commands ─────────────────────────────────────

  function handleSlashCommand() {
    const trimmed = state.inputBuf.trimStart();
    const parts = trimmed.slice(1).split(/\s+/);
    const cmd = parts[0];
    const args = parts.slice(1);

    switch (cmd) {
      case "help":
      case "?":
        w(`\n\x1b[1mAvailable commands:\x1b[0m
  /help, /?          Show this help
  /clear             Clear conversation history
  /stats             Show model stats (tok/s, context usage)
  /trim [pct]        Drop oldest user/assistant turns (default trim to 50% ctx)
  /set <key> <val>   Adjust temperature, top_p, max_tokens for this session
  /exit, /quit       Exit chat

\x1b[1mKeybindings:\x1b[0m
  CTRL+O             Insert newline (multi-line input)
  CTRL+C             Abort stream (press twice from idle to exit)
  CTRL+L             Clear screen
  CTRL+D             Exit (when input empty)
  Up/Down            Navigate input history (drafts preserved)
  Left/Right         Move cursor
  Home/End           Jump to start/end of line
  Backspace/Delete   Delete characters
\n`);
        break;

      case "clear":
        state.messages = [];
        state.committedLines = [];
        state.totalTokens = 0;
        state.tokenTimes = [];
        state.tokPerSec = 0;
        state.lastAbortTime = null;
        w("\x1b[2J\x1b[H");
        w(`Chat cleared. Model: ${state.modelTag}\n\n`);
        break;

      case "stats": {
        const used = state.messages.reduce((s, m) => s + estimateTokens(m.content), 0);
        const pct = state.ctxLimit > 0 ? (used / state.ctxLimit) * 100 : 0;
        w(`\n  Model:     ${state.modelTag}\n`);
        w(`  Messages:  ${state.messages.length}\n`);
        w(`  Tokens:    ~${used} / ${state.ctxLimit} (${pct.toFixed(0)}%)\n`);
        w(`  Tok/s:     ${state.tokPerSec.toFixed(1)} (last turn)\n`);
        w(`  Total tok: ${state.totalTokens} this session\n\n`);
        break;
      }

      case "trim": {
        const targetPct = args[0] ? parseFloat(args[0]) / 100 : 0.5;
        const result = trimMessages(state.messages, state.ctxLimit, targetPct);
        state.messages = result.kept;
        w(`\nTrimmed ${result.dropped} message(s); ${state.messages.length} remain (~${result.remainingTokens} tok).\n\n`);
        break;
      }

      case "set": {
        const key = args[0];
        const val = args[1];
        if (!key || val === undefined) {
          w("\nUsage: /set <temperature|top_p|max_tokens|repeat_penalty> <value>\n\n");
          break;
        }
        const num = Number(val);
        if (!Number.isFinite(num)) {
          w(`\nInvalid value: ${val} (expected a number)\n\n`);
          break;
        }
        switch (key) {
          case "temperature":
          case "temp":
            cfg.temperature = num; w(`\nSet temperature=${num} for this session.\n\n`); break;
          case "top_p":
            cfg.top_p = num; w(`\nSet top_p=${num} for this session.\n\n`); break;
          case "max_tokens":
            cfg.max_tokens = Math.round(num); w(`\nSet max_tokens=${Math.round(num)} for this session.\n\n`); break;
          case "repeat_penalty":
            cfg.repeat_penalty = num; w(`\nSet repeat_penalty=${num} for this session.\n\n`); break;
          default:
            w(`\nUnknown key: ${key}. Try: temperature, top_p, max_tokens, repeat_penalty\n\n`);
        }
        break;
      }

      case "exit":
      case "quit":
        throw new Error("User exit");

      default:
        w(`\nUnknown command: /${cmd}. Type /help for available commands.\n\n`);
    }

    state.inputBuf = "";
    state.inputCursor = 0;
    // Reset history navigation cursor to the bottom; clear any saved draft.
    state.history = { ...state.history, index: state.history.history.length, draft: null };
    renderInputLine();
  }

  // ─── Token tracking ─────────────────────────────────────
  // Character-based heuristic. SSE chunks are NOT one token each — count by
  // estimated tokens (chars / 3.5) instead. Window is per-turn (reset by
  // streamResponse) so the rate reflects the current generation, not session avg.

  function updateTokPerSec(charsAdded: number) {
    const tokAdded = Math.max(1, Math.round(charsAdded / 3.5));
    const now = Date.now();
    for (let i = 0; i < tokAdded; i++) state.tokenTimes.push(now);
    state.totalTokens += tokAdded;
    trimTokenWindow(state.tokenTimes, now);
    state.tokPerSec = computeTokPerSec(state.tokenTimes);
  }

  function checkContextOverflow() {
    const totalTokens = state.messages.reduce((s, m) => s + estimateTokens(m.content), 0);
    const pct = state.ctxLimit > 0 ? (totalTokens / state.ctxLimit) * 100 : 0;
    if (pct > 80) {
      w(`\n\x1b[33m[WARNING: Context ~${pct.toFixed(0)}% full (${totalTokens}/${state.ctxLimit}). Type /trim to drop old messages.]\x1b[0m\n`);
    }
  }

  // ─── SSE stream consumer ────────────────────────────────
  // Append-only on the live tail: we track how much of the current incomplete
  // line has been written and only emit the new bytes. No \r + clear-EOL +
  // re-render every token. Markdown rendering is deferred to commit-time
  // (i.e. when we hit \n) so partial backticks/asterisks don't pop styling.

  async function streamResponse(userMessage: string): Promise<void> {
    state.messages.push({ role: "user", content: userMessage });
    checkContextOverflow();

    state.streaming = true;
    state.abortController = new AbortController();
    state.committedLines = [];
    state.thinking = false;

    // Per-turn rate window: clear stale entries from previous turns.
    state.tokenTimes = [];
    state.tokPerSec = 0;

    const body: Record<string, unknown> = {
      model: state.modelTag,
      stream: true,
      messages: state.messages.map(m => ({ role: m.role, content: m.content })),
      temperature: cfg.temperature,
      max_tokens: cfg.max_tokens,
      repeat_penalty: cfg.repeat_penalty,
      top_p: cfg.top_p,
    };

    w("\n");

    let resp: Response;
    try {
      resp = await fetch(
        `http://127.0.0.1:${state.daemonPort}/v1/chat/completions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: state.abortController.signal,
        },
      );
    } catch (err: any) {
      if (err.name === "AbortError") {
        w("[Stream aborted]\n");
      } else {
        w(`\x1b[31m[ERROR: Daemon disconnected. ${err.message ?? err}]\x1b[0m\n`);
        w("Restart chat to reconnect.\n");
      }
      state.streaming = false;
      state.abortController = null;
      renderInputLine();
      return;
    }

    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      w(`\x1b[31m[ERROR: HTTP ${resp.status}: ${txt.slice(0, 200)}]\x1b[0m\n`);
      state.streaming = false;
      state.abortController = null;
      renderInputLine();
      return;
    }

    if (!resp.body) {
      w("\x1b[31m[ERROR: No response body]\x1b[0m\n");
      state.streaming = false;
      state.abortController = null;
      renderInputLine();
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let incompleteLine = "";
    let incompleteLineWritten = 0;  // chars of incompleteLine already on screen
    let fullResponse = "";
    let firstChunk = true;
    // Tracks whether we're inside a ```fence```. Streaming line-by-line, the
    // full-block fence regex in renderMarkdown can never match — we detect
    // the open/close lines explicitly and style them per-line.
    let inFence = false;
    const fenceWidth = Math.min(60, stdout.columns ?? 60);

    // ASCII spinner — works on every terminal, including older xterm and the
    // plain Linux console (braille pattern chars don't render there).
    const spinChars = ["|", "/", "-", "\\"];
    let spinInterval: ReturnType<typeof setInterval> | null = setInterval(() => {
      if (firstChunk) {
        const label = state.thinking ? "thinking..." : "generating...";
        w(`\r  ${spinChars[Math.floor(Date.now() / 80) % spinChars.length]} ${label}\x1b[K`);
      }
    }, 80);

    try {
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

          let chunk: any;
          try { chunk = JSON.parse(data); }
          catch (e: any) {
            we(`\x1b[2m[SSE parse: ${(e?.message ?? "error").slice(0, 60)}]\x1b[0m`);
            continue;
          }

          if (chunk.error) {
            w(`\n\x1b[31m[hipfire] ${chunk.error.message || "server error"}\x1b[0m\n`);
            continue;
          }

          const delta = chunk.choices?.[0]?.delta ?? {};

          if (delta.reasoning_content && !delta.content) {
            if (!state.thinking) {
              state.thinking = true;
              if (firstChunk) {
                w("\r\x1b[K");
                firstChunk = false;
              }
              w("\x1b[2m[thinking]\x1b[0m");
            }
            continue;
          }

          let text: string = delta.content ?? "";
          if (!text) continue;

          if (state.thinking) {
            state.thinking = false;
            w("\r\x1b[K");
          }

          if (firstChunk) {
            // Dim "Assistant:" role prefix so a copy-pasted transcript
            // shows roles unambiguously (matches the bright "You:" on
            // the user side).
            w("\r\x1b[K\x1b[2mAssistant:\x1b[0m ");
            firstChunk = false;
          }

          fullResponse += text;
          incompleteLine += text;

          if (text.includes("\n")) {
            const parts = incompleteLine.split("\n");
            const cols = stdout.columns ?? 80;
            for (let i = 0; i < parts.length - 1; i++) {
              const p = parts[i]!;
              const fence = detectFenceLine(p, inFence);

              // Decide what string to write for this committed line.
              // - Fence open: dim rule + [lang] label
              // - Fence close: dim rule
              // - Inside fence: raw text (no markdown rendering applied)
              // - Outside fence: markdown-rendered, with the wrap-aware
              //   re-emit guard (don't \r\x1b[K a soft-wrapped line)
              let toEmit: string;
              let needsClearTail: boolean;
              if (fence.isFenceOpen) {
                toEmit = renderFenceOpen(fence.lang, fenceWidth);
                needsClearTail = true;
                inFence = true;
              } else if (fence.isFenceClose) {
                toEmit = renderFenceClose(fenceWidth);
                needsClearTail = true;
                inFence = false;
              } else if (inFence) {
                // Inside a code block: don't apply markdown to body lines.
                toEmit = p;
                needsClearTail = false;
              } else {
                const rendered = md(p);
                const wrapped = p.length >= cols;
                if (rendered === p || wrapped) {
                  toEmit = p;
                  needsClearTail = false;
                } else {
                  toEmit = rendered;
                  needsClearTail = true;
                }
              }

              if (i === 0 && needsClearTail) {
                // First commit and we need to redraw — clear the raw tail
                // first. Safe only when the toEmit fits on one visual row,
                // which we've already gated on above.
                w("\r\x1b[K" + toEmit + "\n");
              } else if (i === 0) {
                // Tail is already on screen as raw text; just terminate it.
                w("\n");
              } else {
                w(toEmit + "\n");
              }
              state.committedLines.push(p);
            }
            incompleteLine = parts[parts.length - 1] ?? "";
            incompleteLineWritten = 0;
            // Print the (unstyled) tail for the next incomplete line.
            if (incompleteLine.length > 0) {
              w(incompleteLine);
              incompleteLineWritten = incompleteLine.length;
            }
          } else {
            // Append-only: write only the new bytes since the last paint.
            // No \r, no clear-EOL, no full re-render. Markdown is unstyled
            // here on purpose — it will be re-rendered when the line commits.
            if (incompleteLine.length > incompleteLineWritten) {
              w(incompleteLine.slice(incompleteLineWritten));
              incompleteLineWritten = incompleteLine.length;
            }
          }

          updateTokPerSec(text.length);
        }
      }
    } catch (err: any) {
      if (err.name === "AbortError") {
        if (!firstChunk) w("\n");
        w("[Stream aborted]\n");
      }
    }

    if (spinInterval) { clearInterval(spinInterval); spinInterval = null; }

    // Flush the trailing incomplete line. Apply the same fence-aware
    // commit logic as the per-chunk path: detect fence open/close, skip
    // markdown inside fences, wrap-aware re-emit otherwise.
    if (incompleteLine && !firstChunk) {
      const cols = stdout.columns ?? 80;
      const fence = detectFenceLine(incompleteLine, inFence);
      let toEmit: string;
      let needsClearTail: boolean;
      if (fence.isFenceOpen) {
        toEmit = renderFenceOpen(fence.lang, fenceWidth);
        needsClearTail = true;
        inFence = true;
      } else if (fence.isFenceClose) {
        toEmit = renderFenceClose(fenceWidth);
        needsClearTail = true;
        inFence = false;
      } else if (inFence) {
        toEmit = incompleteLine;
        needsClearTail = false;
      } else {
        const rendered = md(incompleteLine);
        const wrapped = incompleteLine.length >= cols;
        if (rendered === incompleteLine || wrapped) {
          toEmit = incompleteLine;
          needsClearTail = false;
        } else {
          toEmit = rendered;
          needsClearTail = true;
        }
      }
      if (needsClearTail) {
        w("\r\x1b[K" + toEmit + "\n");
      } else {
        w("\n");
      }
      state.committedLines.push(incompleteLine);
    }

    if (state.tokPerSec > 0 && state.totalTokens > 0) {
      we(`\x1b[2m  ${state.tokPerSec.toFixed(0)} tok/s\x1b[0m`);
    }

    if (fullResponse) {
      state.messages.push({ role: "assistant", content: fullResponse });
    }

    state.streaming = false;
    state.abortController = null;
    w("\n");
    renderInputLine();
  }

  // ─── Submit handler ─────────────────────────────────────

  function handleSubmit() {
    const msg = state.inputBuf.trim();
    if (!msg) return;

    state.history = historySubmit(state.history, state.inputBuf);
    state.inputBuf = "";
    state.inputCursor = 0;

    w("\r\x1b[K");
    w(`\x1b[1;33mYou:\x1b[0m ${msg}\n`);

    streamResponse(msg).catch((err: any) => {
      w(`\x1b[31mError: ${err.message ?? err}\x1b[0m\n`);
      state.streaming = false;
      state.abortController = null;
      renderInputLine();
    });
  }

  // ─── Input mutation helpers (grapheme-cursor aware) ─────────────────────

  function insertText(text: string) {
    if (!text) return;
    const g = graphemes(state.inputBuf);
    const left = g.slice(0, state.inputCursor).join("");
    const right = g.slice(state.inputCursor).join("");
    state.inputBuf = left + text + right;
    state.inputCursor += graphemes(text).length;
  }

  function deleteBackward() {
    if (state.inputCursor === 0) return;
    const g = graphemes(state.inputBuf);
    g.splice(state.inputCursor - 1, 1);
    state.inputBuf = g.join("");
    state.inputCursor--;
  }

  function deleteForward() {
    const g = graphemes(state.inputBuf);
    if (state.inputCursor >= g.length) return;
    g.splice(state.inputCursor, 1);
    state.inputBuf = g.join("");
  }

  function bufferGraphemeLength(): number {
    return graphemes(state.inputBuf).length;
  }

  function isAtNewline(offset: number): boolean {
    const g = graphemes(state.inputBuf);
    return g[offset] === "\n";
  }

  // ─── Raw input handler ──────────────────────────────────

  function handleInput(chunk: string) {
    if (state.escBuf) {
      chunk = state.escBuf + chunk;
      state.escBuf = "";
    }

    // Buffer lone ESC and wait for next chunk. Don't try to be clever about
    // partial CSI sequences — \x1b[? would have been wrongly classified as
    // "complete" by the prior digit-only regex.
    if (chunk === "\x1b") {
      state.escBuf = chunk;
      return;
    }

    // Bracketed paste — pure state-machine in chat_pure.ts handles all the
    // start/end-marker-split-across-chunks and CRLF-normalization edge cases.
    {
      const r = feedPasteParser(state.paste, chunk);
      state.paste = r.state;
      if (r.paste !== null) {
        insertText(r.paste);
        renderInputLine();
        return;
      }
      if (r.passthrough === null) return;          // still mid-paste, swallow
      // Otherwise fall through to keystroke handling with r.passthrough.
      chunk = r.passthrough;
    }

    switch (chunk) {
      case "\x0f": // CTRL+O — explicit newline
        if (!state.streaming) {
          insertText("\n");
          renderInputLine();
        }
        break;

      case "\r":
        if (!state.streaming && state.inputBuf.trim()) {
          if (state.inputBuf.trimStart().startsWith("/")) {
            handleSlashCommand();
          } else {
            handleSubmit();
          }
        }
        break;

      case "\x03": { // CTRL+C
        const now = Date.now();
        if (state.streaming) {
          state.abortController?.abort();
          state.lastAbortTime = now;
        } else if (state.lastAbortTime && now - state.lastAbortTime < 1000) {
          throw new Error("User interrupt");
        } else {
          // First idle Ctrl+C: silently arm the second-hit timer. No nag.
          // Discoverability is via /help, not via inline message.
          state.lastAbortTime = now;
          // If the user had partial input, clear it (bash semantics).
          if (state.inputBuf.length > 0) {
            state.inputBuf = "";
            state.inputCursor = 0;
            state.history = { ...state.history, index: state.history.history.length, draft: null };
            w("^C\n");
            renderInputLine();
          }
        }
        break;
      }

      case "\x04": // CTRL+D
        if (!state.streaming && state.inputBuf === "") {
          throw new Error("User exit");
        }
        break;

      case "\x0c": // CTRL+L
        if (!state.streaming) {
          w("\x1b[2J\x1b[H");
          renderInputLine();
        }
        break;

      case "\x7f":
      case "\b":
        if (!state.streaming && state.inputCursor > 0) {
          deleteBackward();
          renderInputLine();
        }
        break;

      case "\x1b[D": // Left arrow
        if (!state.streaming && state.inputCursor > 0) {
          if (!isAtNewline(state.inputCursor - 1)) {
            state.inputCursor--;
            renderInputLine();
          }
        }
        break;

      case "\x1b[C": // Right arrow
        if (!state.streaming && state.inputCursor < bufferGraphemeLength()) {
          if (!isAtNewline(state.inputCursor)) {
            state.inputCursor++;
            renderInputLine();
          }
        }
        break;

      case "\x1b[A": // Up arrow — history back
        if (!state.streaming) {
          const r = historyUp(state.history, state.inputBuf);
          if (r.buffer !== state.inputBuf || r.state !== state.history) {
            state.history = r.state;
            state.inputBuf = r.buffer;
            state.inputCursor = bufferGraphemeLength();
            renderInputLine();
          }
        }
        break;

      case "\x1b[B": // Down arrow — history forward
        if (!state.streaming) {
          const r = historyDown(state.history, state.inputBuf);
          if (r.buffer !== state.inputBuf || r.state !== state.history) {
            state.history = r.state;
            state.inputBuf = r.buffer;
            state.inputCursor = bufferGraphemeLength();
            renderInputLine();
          }
        }
        break;

      case "\x1b[H": // Home
        if (!state.streaming) {
          // Move to start of current logical line.
          const g = graphemes(state.inputBuf);
          let i = state.inputCursor;
          while (i > 0 && g[i - 1] !== "\n") i--;
          state.inputCursor = i;
          renderInputLine();
        }
        break;

      case "\x1b[F": // End
        if (!state.streaming) {
          const g = graphemes(state.inputBuf);
          let i = state.inputCursor;
          while (i < g.length && g[i] !== "\n") i++;
          state.inputCursor = i;
          renderInputLine();
        }
        break;

      case "\x1b[3~": // Delete
        if (!state.streaming && state.inputCursor < bufferGraphemeLength()) {
          deleteForward();
          renderInputLine();
        }
        break;

      default:
        if (!state.streaming && chunk.length > 0 && chunk.charCodeAt(0) >= 32) {
          if (chunk.startsWith("\x1b")) break;
          insertText(chunk);
          renderInputLine();
        }
        break;
    }
  }

  // ─── Main loop ──────────────────────────────────────────

  try {
    w(`\x1b[1mhipfire chat\x1b[0m — ${state.modelTag}\n`);
    w("Type /help for commands. CTRL+O for multi-line input.\n\n");
    renderInputLine();

    await new Promise<void>((resolve, reject) => {
      stdin.on("data", (data: string) => {
        try {
          handleInput(data);
        } catch (err) {
          reject(err);
        }
      });

      stdin.on("close", () => {
        reject(new Error("User exit"));
      });
    });
  } catch (err: any) {
    if (err.message === "User exit" || err.message === "User interrupt") {
      w("\nExiting...\n");
    } else {
      w(`\n\x1b[31mError: ${err.message ?? err}\x1b[0m\n`);
    }
  } finally {
    cleanup();
    w("\n");
  }
}
