---
name: project-specdecode-metrics-system
description: Unified spec-decode metrics system (SpecMetrics + drain_extra_metrics) landed on chaingun, P1-P6
metadata:
  node_type: memory
  type: project
  originSessionId: a5c90d05-ac89-4d1e-9531-d57a03dc27b9
---

Unified speculative-decode metrics system, DONE + pushed on chaingun (2026-07-05,
P1-P6). Replaces 3 divergent `done`-event schemas + process-global atomic
telemetry with one accumulator every strategy feeds, plus per-strategy
specialized blocks living in their own crates.

Core: `crates/hipfire-specdecode/src/metrics.rs` `SpecMetrics` — per-request
accumulator recording PRIMITIVES (`record_window(proposed, accepted, committed)`)
so core takes no dep on a strategy crate; `tau/accept_rate/mean_draft_len/
mean_committed/to_json`. Emitted once via `hipfire-serving-core/src/spec_metrics.rs`
`emit_spec_done(...)`. Eval parses the unified fields in
`hipfire-eval/src/performance.rs` (normalize + non-negative allowlist).

KEY ARCHITECTURAL FACT (non-obvious): only the DSpark/MTP path holds a real
`Speculator` trait object, so `Speculator::drain_extra_metrics()` works there
(DSpark confidence-truncation ext in `dspark_core.rs`). The **qwen35 DFlash** and
**deepseek4** daemon paths are FREE-FUNCTION calls (no Speculator) — their
specialized metrics can't drain via the trait. P6 solution: **thread-local
per-request accumulators** (daemon spec loop is synchronous, one request per
worker thread → reset-at-request-start + read/to_json at done). Migrated off
process-global `AtomicU64` statics: DDTree-meta (`specdecode/stats.rs`) and
qwen35 seed-oracle (`speculative.rs`), both drained into the DFlash done ext in
`generate.rs`. Retired the env-gated per-cycle eprintln (HIPFIRE_DFLASH_SEED_ORACLE
removed; env-docs auto-regen dropped it). deepseek4 `SpecStepResult` carries no
specialized metrics (n_accepted flows via record_window). Follow-up DONE
(`fc6eeffc1`): the two same-named public types are structurally different, so
deepseek4's was RENAMED to `deepseek4::spec_decode::SpecWindow` (lightweight MTP
window) rather than merged onto the core DeltaNet `SpecStepResult` — leaving one
`SpecStepResult` in the tree; daemon unaffected (reads fields, not the type name).

The system diagnosed the DSpark under-training: done ext showed
`mean_draft_len=1.0`, `dspark.mean_confidence=0.063 < conf_threshold 0.1` (the
confidence head truncates every block). See [[project-dspark-native-trainer]].

Validated: change is OBSERVE-ONLY (telemetry recording + drain, no decode logic)
→ greedy output byte-identical. `coherence-gate-dflash.sh` CONFIRMED on nix2 with
the 9B DFlash pair (qwen3.5-9b-mq4 + drafts/qwen3.5-9b-mq4.dflash.hfq): 0 hard
fails across dflash+ddtree prose/code. Re-run + CONFIRMED again on `8641e526c`
(after the deepseek4 SpecWindow rename): 4/4 cases 0 hard fails (dflash-prose
τ=8.65 clean, dflash-code + ddtree-b12-prose soft-warn only [unique_ratio 0.24 /
0.30 — inherent code/prose repetition, non-blocking], ddtree-b12-code clean).
Note the gate falls back 27B→9B and labels cases `27b-*` regardless of the pair
actually used. GOTCHA when launching it: don't double-background (`nohup … &`
inside a run_in_background Bash) — the outer shell exits immediately and the
"exit 0" notification fires before the gate finishes; run one level of
backgrounding and wait on the `no hard errors` footer (avoid a `pgrep -f
coherence-gate-dflash` waiter — it self-matches its own command line).
