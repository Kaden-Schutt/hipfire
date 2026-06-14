# Serve: 27B hetero PFlash + AR, 92k context (2026-05-25)

Goal: `hipfire serve` qwen3.6:27b AWQ+GPTQ, PFlash prompt-compress, AR
decode (no DFlash), ctx ≥92k. Drafter on gfx1031, target on gfx906.

## Wired (3 commits)

- `prefill_drafter_device` CLI config (-1=share, >0=sibling) → daemon.
- Daemon: persistent `init_with_device` sibling drafter, compress runs
  there, decode on target, bind_thread, host-side kept-IDs handoff.
- `[pflash] LOADED / gen / BYPASS / LOAD FAILED` stderr logging.

## Validated (direct daemon, fits 92k)

target 27B@gfx906 + 0.8B drafter@gfx1031, q8 KV, 96k cap, compat=true:
- 12004 → 3684 tok (0.30, 1.6s), prefill 186 tok/s, AR 17.7 tok/s
- 40004 → 12100 tok (0.30, 14.8s) — no OOM at 96k. Fits 92k goal.
- AR coding correct (`fn helper_7` from 19k-tok needle, helper picked).

## Remaining serve-path bugs (issues #1,#5-9 in memory)

Per-model config drops on wire; stale models.json alias; CASK
auto-attach unbypassable; sidecar dead symlink → os error 2; serve
HTTP JSON-framing crash on ~40k prompts → stale daemon.pid singleton.
Direct daemon works; CLI serve needs config-propagation + pid + framing
fixes. AR fine. Compress proven, not yet through HTTP.
