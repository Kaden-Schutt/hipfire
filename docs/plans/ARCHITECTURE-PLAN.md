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
- `hipfire-rocm` (created; owns ROCm backend evidence contracts)
- `hipfire-daemon-adapter` (created; owns daemon JSONL process-client adapter)
- `hipfire-daemon-protocol` (created; owns daemon JSONL request/response contracts)
- `hipfire-evidence` (created; owns evidence provenance and hash helpers)

A `bun`-free control plane remains desirable but is deferred behind verified seam extraction.

Current prompt boundary status:
- `hipfire-prompt` owns `AssistantPrefix`, `Role`, `Message`, `ToolCall`, `ChatFrame`, and `JinjaChatFrame`.
- `hipfire-runtime::prompt_frame` remains a compatibility re-export and implements the prompt tokenizer trait for the runtime tokenizer.
- The Rust server forwards structured chat `messages` to the daemon and keeps `prompt` as the last-user-text compatibility fallback, avoiding nested ChatML.
- The Rust server builds typed daemon load parameters from config and preserves explicit DFlash mode plus configured TriAttention sidecars through `hipfire-daemon-protocol`.

Current model boundary status:
- `hipfire-model` owns `ModelSource`, `TensorInfo`, `QuantConfig`, model artifact format detection, role-sidecar filtering, display-name derivation, and quant preference ranking.
- `hipfire-model` owns common model-load request/parameter contracts used by daemon protocol clients and future direct library adapters.
- `hipfire-runtime::model_source` remains a compatibility facade and still owns concrete HFQ/safetensors openers until those loaders move.

Current state boundary status:
- `hipfire-state` owns sequence-state handles, parsed handle contracts, page descriptors, worker memory/runtime view structs, generic reservation helpers, and JSON rendering for state descriptors.
- `hipfire-daemon` still owns loaded-model state maps, Qwen3.5 checkpoint attach/fork/release behavior, and backend-specific GPU state materialization.

Current scheduler boundary status:
- `hipfire-scheduler` owns Rust parity contracts for priority classes, prefill/decode scheduler policy, model-worker compatibility, request-session drafts, prefill queue selection, decode active-set batching, backpressure, opportunistic dispatch, and deadline aging.
- Bun scheduler code remains the live server control plane until Rust server paths consume the shared scheduler crate.

Current generate boundary status:
- `hipfire-generate` owns typed generation sampling policy, text generation request/event structs, generate-batch prefill/decode envelopes, batch/preflight JSON validation, prefix-hash compute/JSON helpers, Qwen3.5 prefill/decode backend plan and selector policy, and scheduler metadata helpers.
- `hipfire-daemon-protocol` re-exports the generate-owned text request and token/done/error event structs for daemon JSONL generate traffic, so the protocol crate no longer owns duplicate generate contracts.
- `hipfire-daemon` consumes the shared generate-batch prefill, prefix-hash preflight, decode envelope, prefix-hash helpers, validation, and Qwen3.5 backend-plan/selector contracts while preserving its existing execution paths.
- `hipfire-daemon` still owns model-specific execution, Qwen3.5 runtime orchestration, and JSONL adapter dispatch until later migration slices consume more shared logic.

Current CPU/backend boundary status:
- `hipfire-cpu` owns deterministic BF16 CPU oracle helpers, dense FFN/projection module contracts, backend selection evidence structs, and JSON rendering for module outputs.
- `hipfire-arch-qwen35::ffn_bf16` remains the compatibility facade for Qwen3.5 mode/env parsing and re-exports the shared CPU oracle contracts.

Current ROCm/backend boundary status:
- `hipfire-rocm` owns ROCm device identity, backend-path classification, dense FFN/projection module execution evidence, and JSON rendering for ROCm module outputs.
- Qwen3.5 dense FFN trace output records shared ROCm evidence for the existing `weight_gemv_swiglu_residual` path without moving HIP dispatch or kernel code.

Current daemon protocol boundary status:
- `hipfire-daemon-protocol` owns typed daemon JSONL request/response structs, including load/generate request envelopes and token/done/error responses.
- `hipfire-server::daemon::protocol` remains a compatibility re-export so server callers keep their current paths while the wire JSON stays unchanged.
- `hipfire-daemon` consumes the shared generate request contract opportunistically for common generate fields while preserving raw-JSON fallbacks for legacy/daemon-only fields.
- `hipfire-daemon` also consumes generate-owned batch prefill/preflight/decode contracts, validators, prefix-hash helpers, selector policy, and scheduler metadata so daemon-local duplicate batch structs, hash helpers, validation helpers, and Qwen3.5 batch policy helpers are retired.
- `hipfire-daemon` also consumes the shared load request contract for common load fields (`model`, `max_seq`, `physical_cap`, `dflash_mode`, `draft`, `kv_cache`, `cask_sidecar`) while preserving raw-JSON fallbacks for legacy/daemon-only load fields.
- `hipfire-daemon-protocol` re-exports the model-owned load request/params so protocol callers keep stable paths while load contract ownership moves into `hipfire-model`.

Current daemon adapter boundary status:
- `hipfire-daemon-adapter` owns the async stdio JSONL process client, daemon binary discovery, load/ping/unload/generate response loops, and stale-response filtering.
- `hipfire-server::daemon::engine` remains a compatibility re-export for server and CLI callers while the adapter becomes reusable outside the HTTP server crate.

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
