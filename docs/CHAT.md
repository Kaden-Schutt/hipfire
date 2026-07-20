# Chat

Audience: interactive multi-turn use via `hipfire chat` and how that relates to one-shot `run` and OpenAI-compatible `serve`.

## Quick start

```bash
hipfire pull qwen3.5:4b
hipfire chat qwen3.5:4b
```

```text
Usage: hipfire chat <model> [--no-color]
```

- Requires a real TTY (stdin and stdout).
- Model must already resolve locally (`findModel`); if missing: `hipfire pull <tag>` then retry.
- Color off: `--no-color`, or `NO_COLOR` set, or `CLICOLOR=0`.

## One-shot vs interactive vs HTTP

| Path | Command | Session | Daemon |
|---|---|---|---|
| One-shot | `hipfire run <model> [prompt]` | Single prompt; optional `--system` / sampling flags | Reuses healthy `serve` over HTTP, else one-shot daemon. Forces local when `HIPFIRE_LOCAL=1` or any of `--kv-mode`, `--json`, `--no-stream`. |
| Interactive | `hipfire chat <model>` | Multi-turn TUI, history, slash commands | Reuses `serve` on `cfg.host`:`cfg.port` if up; else spawns an ephemeral serve **without** writing `serve.pid` (`HIPFIRE_NO_PID_FILE=1`) and tears it down on exit. |
| API / tools | `hipfire serve ...` | OpenAI clients, agents | Long-lived process; see [SERVE.md](SERVE.md). |

`run` and `chat` share model-path resolution and (when HTTP) the `/v1/chat/completions` stack. **Chat sampling/context currently use the global config** (`cfg` from `~/.hipfire/config.json`), not the per-model overlay ladder — session `/set` adjusts that global snapshot only. Serve-side load settings (KV, speculation, etc.) remain whatever the attached or spawned serve loaded. Prefer a background `hipfire serve -d` when you mix chat, scripts, and API clients so weights stay warm.

## Daemon attach behavior

1. Probe configured bind (`host`/`port` from config, defaults `0.0.0.0` / `11435`). **No authentication and no TLS** on the HTTP API — prefer `hipfire config set host 127.0.0.1` (or a positional serve host) for local-only; expose beyond localhost only on a trusted/firewalled network or behind an **authenticated TLS-terminating reverse proxy**.
2. If healthy → attach and print that bind.
3. Else spawn `hipfire serve <host> <port>` for this session, wait up to **120s** for `/health`, then enter the TUI.
4. On exit / signal: abort in-flight stream; if this session owns the daemon PID, SIGTERM it; restore terminal modes.

Tracked background serves (`serve -d`) keep `~/.hipfire/serve.pid` and are stopped with `hipfire stop`. Chat-spawned daemons intentionally do **not** claim that pidfile so they cannot clobber a long-lived serve. Chat-owned serve stdout/stderr are piped and discarded — they do **not** write `~/.hipfire/serve.log` (that file is **detached serve only**).

If attach fails: `hipfire diag`, `hipfire ps`. For logs, start the same model with a **foreground** `hipfire serve …` in another terminal, or `hipfire serve … -d` first and then `tail -f ~/.hipfire/serve.log`.

## Session behavior

- **Streaming** tokens with live markdown (code fences, bold/italic).
- **Multi-line input:** `CTRL+O` inserts a newline; bracketed paste is enabled.
- **History:** Up/Down recall prior submissions (draft preserved while browsing).
- **Context:** uses global `max_seq` as the context limit (fallback 32768). Warns around **~80%** full; use `/trim`.
- **`max_tokens` floor:** if global `max_tokens` is below **8192**, chat raises it to 8192 for the session so multi-turn answers are less likely to stop mid-sentence. Higher global config or `/set max_tokens` still wins.
- **Sampling for the session:** starts from **global** config (`temperature`, `top_p`, `repeat_penalty`, floored `max_tokens`); adjust with `/set` (not persisted; not the per-model ladder).

### Slash commands

| Command | Action |
|---|---|
| `/help`, `/?` | Help + keybindings |
| `/clear` | Clear conversation and on-screen history |
| `/stats` | Model tag, message count, ~tokens vs limit, last tok/s |
| `/trim [pct]` | Drop oldest turns (default target ~50% of context) |
| `/set <key> <val>` | Session-only: `temperature`/`temp`, `top_p`, `max_tokens`, `repeat_penalty` |
| `/exit`, `/quit` | Leave chat |

### Keybindings

| Key | Action |
|---|---|
| Enter | Send |
| CTRL+O | Newline |
| CTRL+C | Abort active stream; from idle, second press exits |
| CTRL+L | Clear screen |
| CTRL+D | Exit when input empty |
| Up / Down | Input history |
| Left / Right / Home / End | Cursor |
| Backspace / Delete | Edit |

## Thinking and chat framing

Many curated models (Qwen 3.5/3.6 family and relatives) are **reasoning** models: they may emit a `<think>...</think>` block before the visible answer.

**Display contract (CLI / OpenAI layer, not a silent daemon drop):**

- Streamed **answer** text goes to the main transcript.
- Tokens inside an open think span are treated as reasoning (stripped from plain `content`; on the HTTP API they can appear as `reasoning_content` for clients that render it).
- `hipfire run` stdout shows the answer path; thinking still **consumes** `max_tokens` budget.

**Config knobs** (owned by [CONFIG.md](CONFIG.md) / [MODELS.md](MODELS.md) — do not retune defaults here):

| Key | Default | Role |
|---|---|---|
| `thinking` | `on` | `off` hard-caps thinking (effective 1-token / `enable_thinking=false` signals) so the visible answer is preferred; the model may still spend internal work depending on arch. |
| `thinking_budget` | `med` | Named cap → resolved `max_think_tokens` (`low` 512 … `uncapped` 0). |
| `max_think_tokens` | preset-driven | Raw override; wins over the preset when set. |
| `chat_template` | empty | Optional `.j2`/`.jinja` path; empty keeps engine/model default. |
| `default_chatml` | `true` | Fallback ChatML when no template resolves. |

HTTP extras (`chat_template_kwargs.enable_thinking`, `preserve_thinking`, `reasoning_effort`, etc.) are documented under [MODELS.md](MODELS.md) and [SERVE.md](SERVE.md).

**LFM and other families:** chat framing and stop rules are model-specific. Prefer the registry tag’s documented template; do not assume Qwen `<think>` semantics on every artifact. Numerical parity bugs and wrong chat frames are different failures — LFM chat-framing route ownership is [VALIDATION.md](VALIDATION.md); the `lfm_serve_harness.py` helper is **branch-implemented** (see [SERVE.md](SERVE.md) branch-only subsection). Model notes stay in [MODELS.md](MODELS.md).

## Speculation note

Interactive chat uses the same load-time speculation settings as serve/run. **`dflash_mode` defaults to `off`**: a paired draft on disk does not engage DFlash until you opt in (`hipfire config set dflash_mode auto` or per-model). Details: [CONFIG.md](CONFIG.md), [CLI.md](CLI.md).

## Color and terminal

16-color ANSI + optional OSC 8 links. Disable with `--no-color` / `NO_COLOR` / `CLICOLOR=0` (SGR and hyperlinks stripped at write time; markdown still plain-text readable).

## Errors

| Symptom | Fix |
|---|---|
| “requires an interactive terminal” | Run in a real terminal, not a pipe |
| Model not found | `hipfire list` / `hipfire pull <tag>` |
| Daemon failed within 120s | `hipfire diag`; check ROCm/HIP. Chat-owned serves do not write `serve.log` — reproduce with foreground `hipfire serve` or start detached and inspect `~/.hipfire/serve.log` |
| Port conflict with existing serve | Attach is automatic when healthy; or `hipfire stop` and retry |
| Truncation mid-answer | `/set max_tokens …` or raise config; check thinking budget on long reasoners |
| Context warnings | `/trim` or raise `max_seq` in config (reload model) |

## Next reading

- [GETTING_STARTED.md](GETTING_STARTED.md) — install and first pull
- [CLI.md](CLI.md) — full command index
- [SERVE.md](SERVE.md) — HTTP API, idle unload, multi-client
- [CONFIG.md](CONFIG.md) — persistent knobs
- [MODELS.md](MODELS.md) — tags, thinking deep-dive, templates, BYO models
- [VALIDATION.md](VALIDATION.md) — sole route selector (incl. LFM framing harness)
- [INDEX.md](INDEX.md) — docs ownership map
