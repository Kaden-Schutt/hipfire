# Canonical Modularization and Runtime Architecture Plan

## 0) Source of truth

This file is the canonical decision record for architecture and modularization execution.
It is intentionally compact and backed by evidence links into the historical archive in `docs-old`.

## 1) Scope and invariant

- Do not split the hot Qwen3.5 serving path until correctness and gate stability are maintained.
- Preserve behavior parity for speculative decode and DFlash-related flows during extraction.
- Run `./tests/coherence-gate-dflash.sh` after touching kernels, quant paths, dispatch, rotation, rmsnorm, or speculative decode.
- Run `./tests/no-gpu-ci.sh` before workflow-only handoff changes.

## 2) Accepted modularization target

The target crates for modular boundaries remain:
- `hipfire-daemon`
- `hipfire-model` (created; owns model-source contracts and artifact identity helpers)
- `hipfire-prompt` (created; owns prompt framing and Jinja rendering)
- `hipfire-state`
- `hipfire-generate`
- `hipfire-coherence` (created; owns detector policy and report row serialization helpers)
- `hipfire-rocm`
- `hipfire-evidence` (created; owns evidence provenance and hash helpers)

A `bun`-free control plane remains desirable but is deferred behind verified seam extraction.

Current prompt boundary status:
- `hipfire-prompt` owns `AssistantPrefix`, `Role`, `Message`, `ToolCall`, `ChatFrame`, and `JinjaChatFrame`.
- `hipfire-runtime::prompt_frame` remains a compatibility re-export and implements the prompt tokenizer trait for the runtime tokenizer.
- The Rust server forwards structured chat `messages` to the daemon and keeps `prompt` as the last-user-text compatibility fallback, avoiding nested ChatML.

Current model boundary status:
- `hipfire-model` owns `ModelSource`, `TensorInfo`, `QuantConfig`, model artifact format detection, role-sidecar filtering, display-name derivation, and quant preference ranking.
- `hipfire-runtime::model_source` remains a compatibility facade and still owns concrete HFQ/safetensors openers until those loaders move.

Current state boundary status:
- `hipfire-state` owns sequence-state handles, parsed handle contracts, page descriptors, worker memory/runtime view structs, generic reservation helpers, and JSON rendering for state descriptors.
- `hipfire-daemon` still owns loaded-model state maps, Qwen3.5 checkpoint attach/fork/release behavior, and backend-specific GPU state materialization.

Current scheduler boundary status:
- `hipfire-scheduler` owns Rust parity contracts for priority classes, prefill/decode scheduler policy, model-worker compatibility, request-session drafts, prefill queue selection, decode active-set batching, backpressure, opportunistic dispatch, and deadline aging.
- Bun scheduler code remains the live server control plane until Rust server paths consume the shared scheduler crate.

Current evidence boundary status:
- `hipfire-evidence` owns stable hash, model/tag hash, directory digest, file hash, and HFQ metadata extraction helpers.
- `hipfire-runtime::eval_harness` still owns eval execution and artifact writing, but now consumes the shared evidence provenance helpers.

Current coherence boundary status:
- `hipfire-coherence` owns detector profile selection, detector-bank construction, agentic prompt detection, and report row serialization.
- `hipfire-runtime::coherence_runtime` still owns daemon orchestration, prompt execution, token event capture, and artifact assembly.

## 3) Execution sequence

### Phase A — Stabilize and unify status
1. Lock behavior for existing runtime paths.
2. Confirm gate pass outcomes and no-regression smoke cases.
3. Keep non-functional cleanup out of the critical path.

### Phase B — Boundary extraction in slices
1. Introduce explicit typed interfaces around one subsystem at a time.
2. Build parity tests around each seam before and after migration.
3. Move ownership only when existing callers are reduced to contract calls.

### Phase C — Evidence + serving maturity
1. Move session + scheduler logic into stable service boundaries.
2. Keep runtime state ownership explicit and validated by existing smoke matrices.
3. Maintain queue/attach behavior as a strict compatibility requirement.

### Phase D — Rust CLI/server replacement
1. Port command/runtime entry points only where equivalent behavior is proven.
2. Keep command-level behavior parity and environment handling stable.
3. Remove legacy wrappers once replacement is at parity.

## 4) Open risks and how we close them

- Hidden coupling: runtime bootstrap order and session ownership lifecycles.
- Speculative/quality coupling: DFlash, MTP, and prompt attachment interactions.
- Documentation drift: operational workflows becoming stale during architecture churn.

To close: update this file whenever scope changes and keep canonical references up to date.

## 5) Evidence-first links

These are the archive files with current implementation-relevant decisions:

- `docs-old/plans/modular-runtime-architecture.md`
- `docs-old/plans/stabilize-before-extraction.md`
- `docs-old/plans/session-serving-feature-chart.md`
- `docs-old/plans/multi-model-session-state-serving.md`
- `docs-old/plans/priority-microbatching-scheduler.md`
- `docs-old/plans/rust-cli-server-port.md`
- `docs-old/plans/v1-architecture-roadmap.md`
