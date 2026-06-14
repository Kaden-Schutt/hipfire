# Serve (OpenAI-compatible HTTP)

The daemon exposes an OpenAI-compatible HTTP API on `localhost:11435`
(configurable via `port` in [CONFIG.md](CONFIG.md)). `hipfire run` will
automatically route through this daemon when it's up, skipping the
2–5 s cold-start cost on every invocation.

## Start

```bash
hipfire serve            # foreground, ctrl-c to stop
hipfire serve -d         # background; pid in ~/.hipfire/serve.pid, log in ~/.hipfire/serve.log
```

`-d` pre-warms `default_model` (`hipfire config set default_model
qwen3.5:9b`) so the first request returns tokens immediately.

```bash
hipfire ps               # list running daemons + their pid/port/model
hipfire stop             # graceful shutdown
```

## Endpoints

`POST /v1/chat/completions`

```bash
curl -N http://localhost:11435/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3.5:9b",
      "messages": [{"role": "user", "content": "hi"}],
      "stream": true
    }'
```

Streams JSON-encoded SSE chunks until `[DONE]`. Set `"stream": false`
for a single response payload.

`POST /v1/completions` — non-chat raw completion. Same shape as OpenAI.

`GET /v1/models` — list of locally-available tags.

`GET /health` — returns 200 once the model is loaded; 503 during load.
First-load can take 30 s–2 min on a cold kernel cache.

## Auto-routing from `hipfire run`

```bash
hipfire serve -d                      # daemon up
hipfire run qwen3.5:9b "..."          # this hits HTTP, ~0 ms cold start
HIPFIRE_LOCAL=1 hipfire run qwen3.5:9b "..."  # forced one-shot, skips HTTP
```

Auto-routing detects the running daemon by reading
`~/.hipfire/serve.pid` and probing `/health`. If the daemon is loaded
on a different model than the request asks for, it'll evict and reload
— that takes the cold-start hit on the first request after the model
switch.

## Idle eviction

`idle_timeout` (default 300 s) frees VRAM after 5 minutes with no
requests. The next request reloads automatically. Set to 0 in
[CONFIG.md](CONFIG.md) to keep weights resident forever — useful when
you've got spare VRAM and want zero-latency for sporadic requests.

## Multi-process safety

Only one daemon at a time can hold port 11435. If you see "address in
use" errors:

```bash
pkill -9 daemon bun
rm ~/.hipfire/serve.pid
hipfire serve -d
```

## Logs

- `~/.hipfire/serve.log` — daemon stdout/stderr. Tail this during
  first-load to watch layer upload progress.
- `~/.hipfire/serve.pid` — current daemon pid (deleted on graceful
  shutdown; stale pid file is the most common reason `hipfire stop`
  reports "not running").
