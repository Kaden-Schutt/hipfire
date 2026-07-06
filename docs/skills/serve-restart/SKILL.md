---
name: serve-restart
description: Cleanly stop, free the port, and restart `hipfire serve`. Use when serve "Failed to start (port in use)", a stale daemon holds VRAM, an os-error-2/JSON-parse pre-warm crash left a zombie singleton, or you just want a guaranteed-fresh daemon. Kills both bun CLI serve and the spawned target/release/examples/daemon, reaps stale ~/.hipfire/daemon.pid + serve.pid, fuser-frees the port, then relaunches.
---

# Cleanly restart hipfire serve

Serve thrash burned hours on 2026-05-25: stale `daemon.pid` singletons
("FATAL already running"), zombie daemons holding 30 GB VRAM (next load
OOMs), and pid-by-name kills missing the actual port owner. This skill
does the whole cycle atomically.

## One-shot

```
scripts/serve-restart.sh [port] [-- <extra hipfire serve args>]
```

Default port 11435. Env honored: `HIPFIRE_MODELS_DIR`, `HIPFIRE_VERIFY_GRAPH`.

It: kills `cli/index.ts serve` + `examples/daemon`, `fuser -k <port>/tcp`,
removes `~/.hipfire/{daemon,serve}.pid`, waits the port free, relaunches
detached, tails to `warm-up complete`. (It no longer touches
`/tmp/hipfire-gpu.lock` — that is an flock'd file the kernel auto-releases
on holder death; deleting it would break GPU mutual exclusion.)

## Just kill, don't restart

```
scripts/serve-restart.sh --kill-only 11435
```

## Why pid-by-name isn't enough

`pkill -f examples/daemon` matched a bash wrapper while the real owner of
:11435 survived. Always also `fuser -k 11435/tcp`. A daemon can show 39 GB
RSS — kill it before the next load or you OOM with "0 MB free". Verify
VRAM frees: `rocm-smi --showmeminfo vram | grep Used` → ~10 MB idle.
