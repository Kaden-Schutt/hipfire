---
name: serve-restart
description: Cleanly stop, free the port, and restart hipfire serve. Use when serve fails with "port in use", a stale daemon holds VRAM, daemon.pid/serve.pid is stale after a crash, or you need a fresh daemon before a serve harness or multi-turn test. Script readiness is not guaranteed — always verify /health and logs before proceeding.
---

# serve-restart

Atomic cleanup + optional relaunch for the hipfire serve/daemon stack.
Prefer the scripted path over ad-hoc `pkill` so the port owner and pidfiles
are reaped together — but treat the script as a blunt shared-machine tool,
not an ownership-safe default.

## Ownership-safe default (tracked PID only)

The **only** ownership-safe default is plain tracked stop against the live
pidfile. It validates the tracked PID and does not orphan-reap by name or
`fuser` the port:

```bash
hipfire stop                 # tracked serve.pid only — ownership-safe default
hipfire ps                   # is anything tracked/running?
tail -f ~/.hipfire/serve.log
```

## Shared-machine tools (require approval + owner checks)

These paths force-reap exact-name orphan processes and/or free the port with
`fuser` **without** per-user ownership validation of those orphan/port owners.
On any machine that may host another user's serve, multi-GPU job, or long-
running bench: **get explicit shared-machine approval**, confirm process/port
ownership yourself, then proceed.

```bash
hipfire stop --force [port]  # + orphan daemon reap + fuser free port
hipfire stop --all [port]    # --force plus orphan quantize procs
hipfire restart -d           # stop --force semantics, then start again (default 0.0.0.0)
scripts/serve-restart.sh --kill-only [port]   # teardown only — then loopback serve
scripts/serve-restart.sh [port] [-- <extra hipfire serve args>]  # relaunch binds 0.0.0.0
```

Do **not** label `--force`, `--all`, `restart`, or the script as safer
PID-validated alternatives. Only plain `hipfire stop` is the ownership-safe
tracked-PID path.

## Script behavior (not a readiness guarantee)

```bash
# From repo root. Default port 11435.
# Prefer kill-only + loopback serve for local testing (see Security bind).
scripts/serve-restart.sh --kill-only [port]
hipfire serve 127.0.0.1:<port> -d   # or foreground without -d

# Full scripted relaunch binds 0.0.0.0 — requires explicit all-interface approval:
scripts/serve-restart.sh [port] [-- <extra hipfire serve args>]
```

What the script does, in order:

1. `kill -9` every match of `cli/index.ts serve`, `examples/daemon`, and `bun.*serve`
2. `fuser -k <port>/tcp` so the real TCP owner dies even if name-match missed it
3. Removes `~/.hipfire/daemon.pid` and `~/.hipfire/serve.pid`
4. Waits until `ss -ltn` no longer shows `:<port>` (fails if still busy)
5. Prints a VRAM Used line via `rocm-smi --showmeminfo vram` when available
6. Unless `--kill-only`: clears `~/.hipfire/serve.log`, launches
   `setsid bun cli/index.ts serve 0.0.0.0 <port> …` detached, and tails until
   `warm-up complete` / `port in use` / `JSON Parse` / `FATAL` (or 120s)

**Security bind (mandatory):** serve has **no authentication and no TLS**
([`docs/SERVE.md`](../../../docs/SERVE.md)). Anyone who can reach the bind
address can call every endpoint. The scripted relaunch **hardcodes
`0.0.0.0`** (all interfaces).

- **Local testing default:** `scripts/serve-restart.sh --kill-only [port]`,
  then `hipfire serve 127.0.0.1:<port> -d` (loopback only).
- **Do not** use the script’s non-`--kill-only` relaunch for routine local
  harness/bench work.
- All-interface exposure (`0.0.0.0`, or the script’s default relaunch) requires
  **explicit user approval** plus a trusted/firewalled network or an
  authenticated TLS reverse proxy you control. Do not publish the raw port.

**Fail-closed readiness (mandatory):** the script may still exit successfully
after seeing `port in use` / `JSON Parse` / `FATAL` in the tail, or after the
120s poll timeout without observing readiness. That is **not** a guaranteed-
fresh daemon. Before any harness or benchmark proceeds you **must** manually
verify all of the following; on any failure **block** (do not continue):

1. Log shows warm-up / ready, with **no** `port in use`, `JSON Parse`, or `FATAL`
2. `curl -sf http://127.0.0.1:<port>/health` succeeds
3. Timeout without readiness ⇒ treat as failed restart; re-run `--kill-only`,
   confirm port + VRAM, relaunch once on **loopback** (unless all-interface
   exposure was explicitly approved), re-check health

Env the relaunch inherits: full caller env, including `HIPFIRE_MODELS_DIR`,
`HIPFIRE_MODEL`, `HIPFIRE_VERIFY_GRAPH`, and other serve knobs.

Use `scripts/serve-restart.sh` when:

- plain `hipfire stop` is insufficient and shared-machine approval is in hand
- `hipfire stop --force` still leaves `:11435` busy or VRAM held
- the checkout under test is the repo `cli/index.ts`, not `~/.hipfire/bin`
- a crash left both bun serve and `examples/daemon` without a valid pidfile

Default local pattern after approval for destructive teardown:

```bash
scripts/serve-restart.sh 11435 --kill-only
hipfire serve 127.0.0.1:11435 -d
```

Use full scripted relaunch (binds `0.0.0.0`) **only** with explicit
all-interface approval (trusted firewall or authenticated reverse proxy).

## Guards (do not skip)

- **Do not** `rm /tmp/hipfire-gpu.lock` (or `$HIPFIRE_GPU_LOCKFILE`). It is an
  `flock`'d path; unlinking it breaks mutual exclusion because a new acquirer
  locks a fresh inode. The kernel drops the flock when the holder dies
  (`scripts/serve-restart.sh`, `scripts/gpu-lock.sh`).
- **Do not** kill by name alone. `pkill -f examples/daemon` can match a shell
  wrapper while the real TCP owner survives — always free the port with
  `fuser -k <port>/tcp` (the script does this).
- **Ask before** running `--force`, `--all`, `restart`, or the script on a
  machine that may host someone else's serve, multi-GPU job, or long-running
  bench. Those paths use `kill -9` / name orphan reap / `fuser` and are not
  selective across users.
- **Ask before** any all-interface bind. Scripted relaunch and
  `hipfire serve` / `hipfire serve -d` without an explicit host default to
  `0.0.0.0` with **no auth/TLS**. Local testing → `127.0.0.1` only.
- Prefer one-run env overrides while diagnosing (`HIPFIRE_MODEL=…`,
  `HIPFIRE_KV_MODE=q8`) over editing `~/.hipfire/config.json`.
- After kill-only, confirm idle VRAM before the next load:
  `rocm-smi --showmeminfo vram | grep Used` — expect near-idle MB, not tens of GB.

## Verify (required before harness/bench)

```bash
ss -ltn | grep 11435 || echo "port free"
curl -sf http://127.0.0.1:11435/health && echo OK
tail -20 ~/.hipfire/serve.log
# Block if health fails, log shows FATAL / port in use / JSON Parse, or
# restart timed out without readiness.
```

If restart fails with singleton / port errors, re-run `--kill-only`, confirm
VRAM and port, then relaunch once on **loopback**
(`hipfire serve 127.0.0.1:11435 -d`) unless all-interface exposure was
explicitly approved, and re-run the checks above. Deeper runtime failures →
`.agents/skills/hipfire-autoheal/` (after `.agents/skills/hipfire-diag/` if
bring-up is unclear).

## Related

- Script owner: `scripts/serve-restart.sh`
- GPU lock owner: `scripts/gpu-lock.sh` (`HIPFIRE_GPU_LOCKFILE`, default `/tmp/hipfire-gpu.lock`)
- Product lifecycle: `hipfire serve` / `stop` / `restart` / `ps` in `cli/index.ts`
- Runtime triage: `.agents/skills/hipfire-autoheal/`
