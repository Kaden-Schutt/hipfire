# Phase 2 TODOs for `hipfire chat`

Collected during Phase 1 implementation + adversarial review (`tui_rev_claude.md`).
Phase 1 shipped in PR #129 (closes #63). This document tracks what was
*intentionally* deferred so it doesn't get lost.

Source map for each item: A=adversarial review (numbered finding), I=user
session-testing report, D=design discussion. Keep this when triaging — the
finding numbers point at concrete bug analysis in `tui_rev_claude.md`.

---

## Renderer / styling

### Markdown features (deferred from "what styling makes sense" discussion)

- **Bullet lists** [D] — `- item` / `* item` → `• item` with dim leading
  marker. ~3 LOC regex. High frequency in LLM output.
- **Numbered lists** [D] — `1. foo` → dim the digits. ~3 LOC.
- **Block quotes** [D] — `> quoted` → dim the `>`, italic the body. ~5 LOC.
- **Strikethrough** [D] — `~~text~~` → `\x1b[9m...\x1b[0m`. ~3 LOC. Rare.
- **Horizontal rules** [D] — line of `---` / `***` / `___` → dim full-width
  rule. ~3 LOC.
- **Links** [D] — `[text](url)` → underline text, dim parens; with OSC 8
  hyperlink (`\x1b]8;;url\x1b\\text\x1b]8;;\x1b\\`) for terminals that
  support it (iTerm2, kitty, Wezterm, modern xterm) and raw-link fallback.
  ~10 LOC.
- **`Assistant:` role prefix** [D] — current output has no role marker on
  assistant turns; only `You:` is shown for user input. Adding a dim
  `Assistant:` prefix on the first chunk makes transcript copy-paste
  parseable. ~5 LOC.
- **Auto-detect raw URLs** [D] — `https://...` not in markdown link form →
  underline + OSC 8. ~10 LOC.
- **Auto-detect file:line refs** [D] — `cli/chat.ts:447` → underline (and
  OSC 8 to `file://` if absolute). Useful for code-focused chat where
  models cite source locations. ~15 LOC.

### Markdown features (bigger lifts)

- **Syntax highlighting in code fences** [D] — currently fence body emits
  raw. Lightweight tokenizer for Python/TS/Rust/JSON would cover ~80% of
  LLM output. Hand-rolled per-language is ~80 LOC each; library
  (cli-highlight, prism in pure JS) ~50 KB. **Highest user-perceived
  quality bump.** All competitors (opencode, aider, parllama) do this.
- **Tables** [D] — `| col | col |` parsed and rendered with column
  alignment. Multi-line construct → needs the same buffering architecture
  as fenced blocks (see "Buffer-then-render" below). ~50 LOC + state.
- **Nested SGR fix** [D] — bold-then-inline-code currently breaks because
  inline code's `\x1b[0m` resets the outer bold. Need a style-stack in
  `renderMarkdown` that re-applies outer styles after inner closes.
  ~20-30 LOC, makes `**use \`foo()\` here**` actually bold throughout.

### Renderer architecture changes

- **Wrapped-line styling** [A #6, I] — currently lines longer than terminal
  width skip the markdown re-emit (Phase 1 ships them as raw text). Proper
  fix: count visual rows = `ceil(unicode_width(line) / cols)`, walk up
  with `\x1b[<n>F` per row, clear each, then re-emit styled. Needs
  Unicode width handling (CJK = 2, emoji = 2, ZWJ sequences variable).
  ~50 LOC.
- **Buffer-then-render for multi-line constructs** [D] — fenced blocks
  currently styled per-line via `detectFenceLine`. For tables (and any
  future multi-line construct) we need an accumulator: buffer lines until
  the construct closes, then render the whole block with proper alignment
  / box-drawing. Affects `streamResponse` SSE consumer. Trade-off: no
  output during accumulation, which is bad UX for long blocks.

---

## Input / interaction

- **Type-ahead during streaming** [A #2] — current code locks all input
  while a stream is in flight. Most modern chat tools allow typing the
  next prompt while the model is still answering. After the per-line
  rendering changes from Phase 1, the path is clear: `\x1b[s` save cursor
  before tail repaint, `\x1b[u` restore after, input prompt lives on a
  separate line below. ~20 LOC.
- **Backpressure on stdout.write** [A #25, GLM-5 M1] — `process.stdout.write()`
  returns false when the internal buffer is full; current code ignores
  this. On slow links (SSH, serial) this could drop bytes. Fix: await
  drain event when write returns false. Requires making writes async,
  larger refactor. Defer until field reports.
- **CSI parser state machine** [A #14, GLM-5 H3] — `escBuf` currently
  buffers only lone `\x1b`. Most LSP / private-mode CSI sequences arrive
  in one chunk on Linux, but exotic terminals might split mid-sequence.
  Replace ad-hoc handling with a proper CSI parser (state machine over
  `\x1b [ params; final-byte`). ~40 LOC.
- **Reasoning + content interleaving** [A #24, Gemini #5] — current
  `[thinking]` indicator only fires when `delta.reasoning_content` is
  present *and* `delta.content` is absent. If a future model interleaves
  them, the indicator clobbers the content. Need a state machine that
  tracks transitions and re-emits `[thinking]` when reasoning resumes.

---

## Daemon / lifecycle

- **Daemon ownership ambiguity** [A #20] — when chat-A spawns an
  ephemeral daemon and chat-B (in another terminal) detects + reuses it,
  chat-B doesn't track ownership. If chat-A exits before chat-B, the
  daemon dies underneath chat-B. Fix: lock-file coordination or
  reference-counting in `~/.hipfire/`. ~30 LOC.
- **Spawn engine directly, skip HTTP layer** [GLM-5 C1 alternative] — the
  cleanest fix for the original PID-file collision (Phase 1 used the
  env-var gate workaround). Bypassing `serve()` and talking to the
  Engine via IPC would also save ~5ms per request. Bigger refactor —
  share more code with `runLocal`.

---

## Slash commands / config

- **`/model` switch mid-session** [out-of-scope-Phase-1] — currently
  bound to one model for the session. Adding `/model qwen3.5:27b` would
  request the daemon to swap. Daemon already supports load/unload.
  ~30 LOC + UI for "loading…" feedback.
- **`/save` and `/load` for session persistence** [plan §"Out of Scope"]
  — write `messages` array to `~/.hipfire/sessions/<timestamp>.json`,
  load on demand. Useful for resuming a debugging conversation across
  days. ~50 LOC.
- **`/edit` to edit prior turns** [plan §"Out of Scope"] — opens `$EDITOR`
  on the last user message, re-submits on save. Useful for refining a
  prompt without retyping. ~40 LOC + temp-file dance.
- **`/system <text>`** — set or update the system prompt mid-session. ~5 LOC.
- **`/regen`** — regenerate the last assistant turn (drop it, re-stream
  from the same user message). ~10 LOC.
- **`/copy` and `/copy <n>`** — copy last assistant turn (or turn `n`)
  to clipboard via OSC 52. Cross-platform, no `xclip`/`pbcopy` shell-out
  needed. ~15 LOC.

---

## Telemetry / accuracy

- **Daemon-reported tok/s** [A #3, all-three-reviews] — Phase 1 uses a
  character-based heuristic (`chars / 3.5`). Real fix: have the daemon
  emit `usage.completion_tokens` in the SSE stream (currently only in
  the non-streaming path at `cli/index.ts:1909`). Requires daemon-side
  change to track per-chunk token count and emit it in the final
  `[DONE]`-adjacent chunk. Cross-cutting with `runViaHttp` which has
  the same chunk-as-token bug at `cli/index.ts:856`.
- **Per-token latency display** — `time-to-first-token` and
  `inter-token-latency-p50` for the last turn. Useful for the "feels
  slow" debugging path. ~20 LOC, hook into the existing `tokenTimes`
  buffer.

---

## Test coverage

- **Integration test for `streamResponse` against a fake daemon** [D] —
  Phase 1 tests cover the 8 pure helpers in `chat_pure.ts` (74 tests).
  The `streamResponse` consumer itself isn't tested — needs a mock
  fetch that emits canned SSE chunks. Would catch regressions in
  fence-detection, wrap-aware re-emit, abort handling. ~80 LOC of test
  scaffolding.
- **Snapshot test for `renderInputLine` output** [D] — pure formatter
  function once lifted, but currently tied to `stdout.write` calls.
  Worth lifting if we add more multi-line input features.
- **Fuzz test for `feedPasteParser`** [D] — randomized chunking of
  paste content with start/end markers split at every byte boundary.
  Phase 1 has hand-picked split cases; fuzzing would find off-by-ones.
  ~30 LOC.

---

## Documentation

- **`docs/CHAT.md` examples** — add screenshots / asciinema cast of a
  real chat session showing markdown rendering, slash commands, paste.
- **`docs/CHAT.md` keybinding cheatsheet card** — terminal-printable
  one-pager that users can `cat` while learning.

---

## Already-fixed-but-fragile (revisit if anything regresses)

These are Phase 1 fixes that solved the symptom but left the underlying
mechanism somewhat fragile. Worth a re-look if related bugs surface:

- **PID-file collision** [A #4 / GLM-5 C1] — fixed via
  `HIPFIRE_NO_PID_FILE=1` env gate. Cleaner solution: spawn engine
  directly (see "Daemon" section).
- **Soft-wrap duplication** [A wrap fix, I] — fixed by skipping markdown
  re-emit on wrapped lines. User loses styling on those lines. Proper
  fix is the multi-row clear (see "Wrapped-line styling" above).
- **Fence body unstyled** [I, intentional] — fence body lines emit raw
  to keep code readable. Real fix is syntax highlighting (see
  "Syntax highlighting" above).

---

## Won't-fix / explicit non-goals

For the record, these were considered and rejected for Phase 2 too:

- **Vim keybindings (j/k scroll)** — native terminal scrollback handles
  this; we don't manage our own scroll buffer.
- **Mouse support in chat area** — terminal natively supports mouse
  selection / copy on the rendered output. Adding mouse handlers
  would conflict with that.
- **Cross-vendor compute backend** — out of scope per CLAUDE.md
  (issue #44 closed). hipfire is HIP/ROCm-direct; chat surface
  inherits this constraint.

---

## Effort estimate

Quick-win batch (bullets, numbered lists, block quotes, role prefix,
auto-link with OSC 8): **~30 LOC, 1-2 hours**. Recommended for a
follow-up PR right after #129 lands.

Medium batch (syntax highlighting, type-ahead, daemon-reported tok/s,
nested SGR): **~250 LOC, 1-2 days**.

Large batch (tables, wrapped-line styling with Unicode width,
spawn-engine-directly refactor): **~500 LOC, 3-5 days**. Worth
deferring until the chat surface has soak time and more user input.
