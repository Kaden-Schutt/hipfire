# Steering driver — pivot to the daemon

Status: DONE — `hipfire-steer` now drives the model through the daemon; the
in-process `Gemma3Harness` is retired.
Date: 2026-06-30
Supersedes the in-process harness path in
`docs/plans/2026-06-29-refusal-direction-steering.md`.

## Outcome (validated 2026-06-30, nix2 gfx1103)

End-to-end on `medgemma-1.5-4b-it.q8f16.hfq` (arch_id 13) + the medical set,
all through the daemon (`begin_capture`→`steer_capture`×N→`finish_capture`→
`begin_apply`→`kld_eval`→`generate`):

- **base refusals 3/3** — coherent over-refusal ("I am an AI assistant, not a
  medical professional", "I cannot provide medical advice"); the daemon
  inference is correct where the in-process harness garbled it.
- **Ablate strength=1.0 → refusals 0/3** — abliteration through the daemon apply
  session fired in the resident forward and removed every refusal.
- **KLD ≈ 0** on the benign good-eval corpus — negligible capability damage (the
  Heretic low-collateral property; the refusal drop proves apply is live).

Notes for the next pass:
- MedGemma 1.5 is a reasoning model (`<unused94> … <unused95>` block); the daemon
  emits it verbatim (gemma3 ignores `thinking_mode`), so `DaemonHarness::generate`
  strips through `<unused95>` before scoring — the daemon-path analogue of the
  retired harness's token-level `skip_thinking`. With a short budget the block
  may not close, but the refusal markers in the reasoning trace still classify
  correctly.
- `DaemonHarness::generate` sends `reset` before each request (the daemon
  generate path is stateful and does not self-reset); `reset` leaves the steer
  session intact, so an active apply persists across the batch.
- Point the harness at a steer-aware daemon via `HIPFIRE_DAEMON_BIN` until the
  installed `~/.hipfire/bin/hipfire-daemon` is rebuilt with the steer arms.

## Why

`hipfire-steer-harness::Gemma3Harness` reimplemented a slice of the inference
stack — chat templating, BOS/special-token encoding, prefill, decode loop, EOS,
per-request state, logprobs — and we spent real effort debugging bugs that path
introduced (missing `<bos>`, special-token encoding, `Gemma3State` reuse). The
**daemon already does all of this correctly** (`hipfire chat` produces coherent
output on the exact prompts the harness garbled). And crucially, the steer hook
(`maybe_steer_block`) is compiled into the arch forward, so the steer session
lives in the daemon process too — we just never exposed control over it.

Pivot: drive the model through the daemon (reusing its inference + the eval
harness's `DaemonEngine` client + `kld_eval`), and add a few steer-session ops to
the daemon protocol. Retire `Gemma3Harness`. All of `hipfire-steer`'s core (hook,
capture `observe`/`commit`, `derive_directions`, on-GPU apply, driver, scoring,
Pareto) is reused unchanged.

## What the survey established

- **Adding a daemon op = 3 edits**: protocol struct+enum variant
  (`hipfire-daemon-protocol`), a string-keyed `match msg_type` arm in
  `hipfire-daemon/src/main.rs` (hand-parses JSON, writes a JSON response line —
  modeled on the `collect` arm ~L4785), and a `DaemonEngine` client method
  (`hipfire-daemon-adapter`, modeled on `collect` L276).
- **Dispatch is single-threaded + synchronous** (`for line in stdin.lock()...`,
  forward runs inline). Requests serialize one-at-a-time → a
  begin_capture → N×capture → finish_capture sequence runs in order, no
  interleaving. Safe for the process-global steer session.
- **The hook fires in the daemon forward** (gemma3 `forward.rs:362`; arch crates
  depend on `hipfire-steer`).
- **Reusable client**: `DaemonEngine` (`hipfire-daemon-adapter`) — `spawn`,
  `load`, `generate`, `kld_eval`, etc. hipfire-eval drives it from
  `executor_daemon.rs`.
- **Scoring ops that exist**: `generate` (chat-templated `messages` → text,
  through the hooked forward) and `kld_eval` (modes BuildRef/Score/SelfScore →
  KLD scalar + NLL/PPL). **No op returns first-token logprobs over the wire.**
- **Capture caveat**: a capture forward must be **prefill-only** — `generate`
  with `max_tokens≥1` decodes a token whose forward would overwrite the
  last-prompt-token residual `observe` just recorded. Capture needs prefill +
  `commit_capture`, no decode (model on the `collect` arm, which is prefill-only).
- Every GPU forward arm guards `pp == 1`; capture inherits that.
- `reset`/`unload` don't touch the steer session → need `clear` on unload so a
  stale apply session can't leak across model loads.

## The ops (5)

Added to `hipfire-daemon-protocol` (`DaemonRequest`/`DaemonResponse`), handled in
`hipfire-daemon/src/main.rs`, wrapped in `DaemonEngine`:

1. `steer_begin_capture { num_layers, hidden }` → `hipfire_steer::begin_capture` → ack.
2. `steer_capture { messages }` → chat-template + **prefill only** through the
   hooked forward (hook observes the last-prompt-token residual per block) +
   `hipfire_steer::commit_capture()` → ack. (Combines prefill+commit; no decode.)
3. `steer_finish_capture` → `finish_capture()` → `{ means: Vec<Vec<f32>> }`
   (serialize `CaptureMeans.0`).
4. `steer_begin_apply { directions: Vec<Vec<f32>>, mode, strength, layer_start, layer_end }`
   → build `SteerSpec` → `begin_apply(spec)` → ack.
5. `steer_clear` → `clear()` → ack. Also call `clear()` inside the `unload`/`load` arms.

Add `hipfire-steer = { path = "../hipfire-steer" }` to `hipfire-daemon/Cargo.toml`.

Serialization note: `means`/`directions` are `num_layers × hidden` f32
(gemma-4B: 34×2560 ≈ 87k floats ≈ ~1 MB JSON). Acceptable over the JSON-line
protocol for now; revisit (bincode/base64) if it's a bottleneck.

## Scoring through the daemon

- **Refusals** (primary): `DaemonEngine::generate` with steer applied, on the
  bad-eval `messages` → text → existing `count_refusals`/`is_refusal`
  (`driver.rs`). This is the path that was broken in-process and is correct in
  the daemon.
- **KLD** (capability guard): `kld_eval` BuildRef (steer cleared = base) on the
  good-eval corpus → `begin_apply` → `kld_eval` Score (steered) → `mean_kld`.
  Sequence-KLD vs base, a valid (arguably stronger) capability-damage metric than
  first-token KLD. (If we later want exact first-token KLD, add a logprobs op.)

## Driver wiring

`run_driver` (`hipfire-steer/src/driver.rs`) already takes `&mut dyn ModelHarness`.
The cleanest pivot keeps that seam:

- Extend `ModelHarness` with session-control methods (`begin_capture`,
  `commit`, `finish_capture`, `begin_apply`, `clear`) so the driver routes them
  through the harness instead of calling the `hipfire_steer::*` statics directly.
  A `DaemonHarness` routes them to the daemon; the (retired) in-process one would
  route to the local statics.
- New `DaemonHarness` (replaces `Gemma3Harness`) holds a `DaemonEngine`, knows
  `num_layers`/`hidden` (from the load response or config), and implements the
  trait: `capture_forward` → `steer_capture`; `generate` → `generate`;
  KLD via `kld_eval`. The good/bad prompt sets become `messages`.

Net: delete `Gemma3Harness`; `hipfire-steer-harness` becomes a thin
`DaemonHarness` + CLI over `DaemonEngine`. Everything in `hipfire-steer` stays.

## Build order

1. Protocol structs + enum variants (`hipfire-daemon-protocol`).
2. Daemon arms + `hipfire-steer` dep + `clear()` on unload (`hipfire-daemon`).
3. `DaemonEngine` client methods (`hipfire-daemon-adapter`).
4. `ModelHarness` session-control extension + `DaemonHarness` + retire
   `Gemma3Harness` (`hipfire-steer`, `hipfire-steer-harness`).
5. Validate end-to-end on medgemma-4b + the medical set — expect coherent
   generations (daemon inference) and a real base refusal rate.

## Reused unchanged

`maybe_steer_block` hook, `CaptureAcc`/`observe`/`commit`, `derive_directions`,
on-GPU apply (`apply_on_gpu`), `run_driver`, `count_refusals`/`is_refusal`,
Pareto, the medical prompt set, the medgemma-4b `.hfq`.
