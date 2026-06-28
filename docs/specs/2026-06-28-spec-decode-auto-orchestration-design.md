<!-- Copyright (c) 2026 Kaden Schutt -->
# Spec-decode auto-orchestration: unified `spec_decode`, per-model selection, DFlash→MTP long-ctx handoff

**Status:** Design APPROVED 2026-06-28 (brainstormed with user). Phase 1 ready to
build; depends on the config-driven MTP foundation (branch
`feat/mtp-config-default`).

## 1. Motivation

Spec-decode today is three separate knobs (`dflash_mode` / `mtp_mode` /
`ngram_mode`), each with its own config value + env override. A user must know
(a) which mechanism their model supports and (b) which flag turns it on. That is
the opposite of "enable spec decode."

Two measured facts drive the orchestration:
- **DFlash is short-ctx-good, long-ctx-bad.** Its drafter re-attends over the
  full target-derived context every spec step (O(ctx)/step), so τ falls with
  context and DFlash crosses **below AR around ~24k** (see
  `docs/specs/2026-06-26-dflash-triattn-drafter-eviction.md`).
- **MTP is ctx-independent.** Its head reads only the trunk's final hidden
  state, so per-step cost does not grow with context — lower short-ctx τ, but it
  never degrades.

So the best decode for a **dense** model is DFlash early, MTP late; for **MoE**
(A3B) there is no DFlash draft at all, so it's MTP-only. The goal is one knob
("spec decode on") that resolves the right mechanism per model, hands off
DFlash→MTP at long context, auto-fetches the needed head/draft, and is
reachable from both a TUI and CLI flags.

## 2. Unified `spec_decode` config

```
spec_decode: off | auto        (default: off = AR)
```
- `off` → AR (unchanged default).
- `auto` → resolved per model at load:
  - dense + DFlash draft + MTP head → DFlash until ctx≈threshold, **evict ring +
    hand off to MTP** beyond it.
  - MoE, or dense with only an MTP head → MTP.
  - dense with only a DFlash draft → DFlash (no long-ctx fallback).
  - neither available → AR.

The existing `dflash_mode` / `mtp_mode` / `ngram_mode` stay as **explicit
per-mechanism overrides** for power users; `spec_decode=auto` is the orchestrated
default path. Dense vs MoE is resolved by `arch_id` (qwen3.5/3.6 dense = 5; A3B
MoE = 6; etc.). Availability is a disk + registry lookup for the model's MTP head
(bundled sidecar) and DFlash draft (separate registry SKU).

## 3. Auto-download UX

When spec-decode (or `dflash_mode`/`mtp_mode`) is enabled — in the TUI **or** via
`config set` — a config-time check runs against the registry + disk:
- **on disk** → nothing.
- **not on disk, in registry** → notify once:
  *"drafter for [model] not found on disk — DFlash for [model] will be downloaded
  when this config is saved"* (and the MTP-head analog). The download fires on
  **save**, not on toggle.
- **not on disk, not in registry** → *"no drafter/head available for [model]"* —
  the config saves but stays AR until one exists.

No silent multi-GB pull, no surprise; it's a local registry+disk check so the
notification is instant.

## 4. Config surfaces — three, consistent (llama.cpp/llama-server parity)

The TUI is the friendly default, but **nothing is TUI-only**. Every spec knob is
reachable identically from:
1. **TUI** (`hipfire config` / `hipfire`) — already exposes `dflash_mode` /
   `mtp_mode` (`crates/hipfire-tui/src/hipfire/config.rs`).
2. **`hipfire config set <key> <value>`** — persisted, CLI.
3. **`hipfire serve|run --spec <off|auto|dflash|mtp>`** — per-invocation,
   command-flag style for people who live on the command line.

The unified `spec_decode`/`--spec auto` and the auto-download behavior must be
wired into all three identically.

## 5. Phasing

- **Phase 1 (shippable; builds on the config-driven MTP PR):** the unified
  `spec_decode` toggle + per-model selection (dense→DFlash, MoE→MTP) +
  auto-download-on-save + the three config surfaces. **No handoff yet** — just
  "pick the right mechanism per model, fetch it if missing."
- **Phase 2 (the perf feature):** the DFlash→MTP long-ctx **handoff**.

## 6. Phase 2 — the DFlash→MTP handoff

- **Trigger:** ctx crossover (default ~24k, tunable) where DFlash τ drops toward
  AR-parity. Adaptive-τ (switch when running τ < 1.1) is a later refinement.
- **Mechanism:** at the threshold, **evict the DFlash hidden ring** (reclaims the
  O(ctx) bloat VRAM) and route subsequent steps through the MTP head.
- **VRAM:** requires both the DFlash draft and the MTP head resident (the ring is
  freed at handoff, but the draft stays loaded for the next request's short-ctx
  phase). When both won't fit, **degrade gracefully to MTP-only** — never force
  an OOM for the handoff.
- **Validation:** long-ctx (≥24k/32k) dense decode tok/s + τ AND the three-tier
  DFlash coherence gate at the handoff boundary — the mechanism switch must not
  introduce an attractor.

## 7. Open questions / risks

- The mid-generation mechanism switch must preserve KV + recurrent state across
  the handoff (the #462 cross-request state-bleed class — here intra-request).
- Adaptive-τ needs running-τ tracking in the decode loop.
- Auto-downloading a multi-GB DFlash draft on save: the UX must surface the
  size/ETA so "save" isn't a silent long stall.

## References
- `docs/specs/2026-06-26-dflash-triattn-drafter-eviction.md` — DFlash long-ctx degradation + the O(ctx) re-attention.
- Config-driven MTP rework (`feat/mtp-config-default`, daemon.rs qwen35 MTP dispatch) — the Phase-1 foundation.
- `crates/hipfire-tui/src/hipfire/config.rs`, `knobs.rs` — the existing TUI config surface.
