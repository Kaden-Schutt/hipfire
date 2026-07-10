# autoresearch supervisor (`hipfire-ar`) — Design

**Date:** 2026-07-06  **Status:** Built (`autoresearch/ar/hipfire_ar.py`), spec for reference.

## Why
The autoresearch *logic* was already correct and proven — `driver_v3.sh` self-terminates on per-kernel
exhaustion (K consecutive deads), targets from the BOD census, and feeds a tried-lever digest to codex. What
was missing was a reliable **supervision/persistence/control layer**. Instead of using `driver_v3`, the campaign
ran `fire_moe.sh` + a flat hardcoded prompt wrapped in a `ka.sh` keep-alive — which hardcoded targets, re-tried
dead levers, ran away (a detached keep-alive with no leash), and scattered state in `/tmp`. `hipfire-ar` is the
missing layer: it wraps the proven pieces, is non-ephemeral, and is controllable by both Claude and codex with
**mechanical** runaway prevention.

## Principle
Reuse, don't rebuild. `hipfire-ar` orchestrates the existing `oracle_profile.sh` (census→BOD), `driver_v3.sh`
(self-terminating loop), the `check/update/gen_digest.py` exhaustion trio, and `rollover_v2.sh` (fold). It adds
only: a persistent store, a role-scoped CLI, and bounds enforcement. Python — the trio + census tooling are
already Python.

## Roles (single tool, capability-scoped)
- **operator / Claude:** `start · stop · status · why · bod · ingest`
- **agent / codex:** `why · status · bod · certify` — read the state + submit candidates, nothing else.

## Runaway prevention — defense in depth (the whole point)
1. `driver_v3` self-terminates on **global exhaustion** (every candidate kernel hit K deads). **No keep-alive.**
2. Every run carries a **codex-call budget + wall-TTL**; the supervisor auto-stops past either. Single-owner
   **lock per (box, card)** — one loop per card, nothing piles up.
3. **Every `certify` is bounds-checked in the tool** (not the prompt): a submission on an EXHAUSTED kernel, an
   OFF-TARGET (non-census) kernel, or past budget/TTL is **REFUSED (exit 3)**. Codex can be as eager as it
   likes; the tool is the leash. This replaces prompt-discipline ("never re-try a DEAD lever") — which codex
   sometimes ignored — with mechanical rejection.

## Store (non-ephemeral)
SQLite at `autoresearch/state/ar.db` (gitignored) + the git-committed `autoresearch/ledger/*.jsonl` for durable
history. `ingest` pulls the jsonl ledger + `bod_<arch>.json` snapshots into SQLite (idempotent). Tables:
`attempts` (arch,kernel,lever,verdict,delta,profile,ts), `bod` (arch,kernel,wall%,l2,mem_busy,occ,vgpr),
`runs` (id,arch,model,card,status,budget,calls,ttl,pid). Replaces the `/tmp` scatter (exhaustion.json, BOD,
loop_progress.log).

## Claude/codex-shaped surface (signal, not logs)
- `ar bod --arch X` — ranked candidate kernels with **coverage**: wall%, L2%/bound-class (DRAM-thrash⇒traffic
  lever, cache-resident⇒ALU/occ lever), tried/win counts, best-win%, and **OPEN vs EXHAUSTED**. One glance
  shows where lever room remains (e.g. `residual_*` 7.9% wall, 1 tried, OPEN).
- `ar why <kernel> --arch X` — every lever tried on that kernel + verdict + delta + why-dead. The tried/failed
  feedback surface codex pulls before authoring.
- `ar status` — running loops: calls/budget, ttl-left, pid.

## Model-agnostic
`arch/model/card/budget/ttl` are CLI params → env into `driver_v3` (which already reads `ARCH`/`BOD`/`MODEL`).
Nothing hardcodes a3b/mq4/gfx1201.

## Data flow
`census (oracle_profile) → bod_<arch>.json → ar ingest → SQLite`; then
`ar start → driver_v3 (target from bod → codex asks ar why → ar certify gates → ab_certify_v2p grades → ledger)
→ ar ingest → ar bod/why/status`. Fold/advance via `rollover_v2`, then re-census + re-ingest.

## Out of scope (kept as human gate)
Auto Stage-1→Stage-2 bridge (operator runs the brainstorm→queue). Branch-safety patch to `driver_v3`
(`checkout -B` → create-or-resume + `loop/cardN_recovered`) is a follow-up edit, tracked separately.

## Testing
No-GPU: `ingest` idempotency, `bod`/`why` against a fixture ledger, `certify` refusal on exhausted/off-target/
over-budget (exit 3). Live: ingest the real ledger (done — 490 attempts, gfx1201+gfx1151) and confirm `ar bod`
ranks + marks EXHAUSTED correctly (verified: gate_up EXHAUSTED, residual OPEN).
