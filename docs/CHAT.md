# Chat

hipfire provides two native Rust chat surfaces:

- `hipfire tui` (or bare `hipfire`) opens the full ratatui application and its
  Chat tab.
- `hipfire chat [model]` is a lightweight line-oriented multi-turn client for
  terminals and scripts.

Both use the native OpenAI-compatible service documented in
[`SERVE.md`](SERVE.md).

## Full terminal UI

```bash
hipfire
# or
hipfire tui
```

Open the Chat tab, select a downloaded model in Models, or use `/model <tag>`.
The TUI supports streaming, multiline input (`Ctrl+O`), cancellation, saved
sessions, model/system/sampling commands, regeneration/edit/copy actions, and
live serve status. Run `/help` in Chat for the current command list.

The TUI asks the native CLI to start a detached service when the configured
endpoint is offline. Config, registry, model lifecycle, and HTTP calls all use
the shared Rust crates; there is no separate script runtime.

## Lightweight chat command

```bash
hipfire chat qwen3.6:35b-a3b-mq4r
```

```text
you> Explain wave32 WMMA briefly.
assistant> ...
you> /clear
you> /exit
```

Supported local commands are `/clear`, `/exit`, and `/quit`. Flags:

| Flag | Purpose |
|---|---|
| `-t, --temp <f>` | Explicit temperature for this session |
| `--top-p <f>` | Explicit nucleus probability |
| `-n, --max-tokens <n>` | Per-turn generation cap |
| `--system <text>` | Initial system message |
| `--no-color` | Compatibility flag; the line client emits no ANSI color |

If no healthy service exists, the command starts a tracked detached native
service on the configured host/port with lazy model loading. That service
remains available after the line client exits; use `hipfire stop` when you want
to release it.

## Resolution and request behavior

- Model names use the same registry/alias/path resolver as `run` and `serve`.
- The service resolves load-time KV/speculation/model overrides from
  `config.toml`, `models.toml`, registry recommendations, and compatible env.
- Explicit chat sampling flags are sent on every turn. Omitted sampling fields
  follow the service's per-model/registry/daemon fallback contract.
- The full `messages` array is sent each turn, so conversation context is
  preserved by request content rather than hidden client state.
- Streaming answer and reasoning deltas share the same Rust HTTP client used by
  the TUI.

## Thinking and model framing

Reasoning models may emit a thinking stream before the answer. The HTTP API
exposes it as `reasoning_content`; visible answer text remains `content`.

Mode, semantic effort, and hard cap are **independent** — see
[`CONFIG.md`](CONFIG.md) for keys/defaults and [`SERVE.md`](SERVE.md) for
per-request HTTP fields (`reasoning_effort`, `enable_thinking`,
`max_think_tokens`, legacy `thinking_budget`). Effort never maps to a token
budget. Unsupported or cross-family values are dropped with a warning (server
log + OpenAI `hipfire` response metadata), not silently reinterpreted.

Chat framing and stop behavior are model-specific. Use the exact registry tag
and do not assume Qwen `<think>` conventions apply to LFM, Gemma, Glimmer, or
DeepSeek. The LFM framing smoke route is `scripts/serve_harness.py` with the
exact `lfm2.5:*` tag; see [`VALIDATION.md`](VALIDATION.md).

## Troubleshooting

| Symptom | Action |
|---|---|
| Model not found | `hipfire list -r`, then `hipfire pull <tag>` |
| Service will not start | `hipfire diag`; inspect `~/.hipfire/serve.log` |
| Port conflict | `hipfire ps`; stop the owned service with `hipfire stop` |
| Truncated answer | Raise `--max-tokens`; on Qwen, if the think span was explicitly force-closed, raise `max_think_tokens` or leave it uncapped |
| Need richer session controls | Use `hipfire tui` and its Chat tab |

The service has no authentication or TLS. Local chat should use a loopback bind;
see [`SERVE.md`](SERVE.md) before exposing it to a network.
