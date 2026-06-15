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
- `hipfire-config` (created; owns shared CLI/server config and local path helpers)
- `hipfire-eval` (created; owns the eval runner binary adapter and harness implementation)
- `hipfire-hash` (created; owns stable file/byte hash primitives shared by model and evidence contracts)
- `hipfire-model` (created; owns model-source contracts and artifact identity helpers)
- `hipfire-prompt` (created; owns prompt framing and Jinja rendering)
- `hipfire-state`
- `hipfire-generate`
- `hipfire-coherence` (created; owns detector policy and report row serialization helpers)
- `hipfire-rocm` (created; owns ROCm backend evidence contracts)
- `hipfire-npu` (created; owns NPU opt-in artifact/admission contracts)
- `hipfire-daemon-adapter` (created; owns daemon JSONL process-client adapter)
- `hipfire-daemon-protocol` (created; owns daemon JSONL request/response contracts)
- `hipfire-evidence` (created; owns evidence provenance, host-profile contracts/policy, and evidence hash policy)

A `bun`-free control plane remains desirable but is deferred behind verified seam extraction.

Current prompt boundary status:
- `hipfire-prompt` owns `AssistantPrefix`, `Role`, `Message`, `ToolCall`, `ChatFrame`, and `JinjaChatFrame`.
- `hipfire-prompt` owns OpenAI chat-role/content conversion helpers used by the Rust server to build structured daemon messages and last-user prompt fallbacks.
- `hipfire-prompt` owns assistant-turn prompt-history fingerprinting and canonical tool-call argument JSON used by daemon multi-turn prefix-cache identity.
- `hipfire-prompt` owns assistant-prefix wire-label parsing for `plain`, `open_think`, and `closed_think` request paths.
- `hipfire-prompt` owns load-time chat-template source resolution for env-file, per-model override file, and embedded model-template precedence.
- `hipfire-prompt` owns the pure prompt-normalization transform for line endings, NBSP, trailing line whitespace, and cold newline runs; runtime keeps only the env/config opt-in wrapper.
- `hipfire-runtime` implements the prompt tokenizer trait for its tokenizer while prompt frame types are imported directly from `hipfire-prompt`.
- `hipfire-daemon` now consumes `hipfire-prompt` directly for daemon prompt framing, prompt normalization, assistant-prefix labels, tool-call fingerprints, and structured message history instead of routing those types through the runtime compatibility facade.
- The Rust server forwards structured chat `messages` to the daemon and keeps `prompt` as the last-user-text compatibility fallback, avoiding nested ChatML.
- The Rust server builds typed daemon load parameters from config and preserves explicit DFlash mode plus configured TriAttention sidecars through `hipfire-daemon-protocol`.

Current model boundary status:
- `hipfire-model` owns `ModelSource`, `TensorInfo`, `QuantConfig`, model artifact format detection, role-sidecar filtering, display-name derivation, filesystem-safe artifact stem derivation, and quant preference ranking.
- `hipfire-model` owns `ModelWorkerKey` identity, feature-flag normalization, worker-key id construction, worker-key compatibility comparison, model-worker runtime id construction, daemon `worker_id` / `worker_key_id` alias parsing, and generate worker/model identity requirement policy used by scheduler/state/generate/control-plane adapters.
- `hipfire-model` owns model architecture id constants and family classification helpers used by generate/backend-selection and daemon Qwen3.5 runtime guard contracts.
- `hipfire-model` owns tokenizer signature/fingerprint policy, HFQ tokenizer metadata payload selection (`tokenizer` vs `gguf_meta`), optional safetensors `tokenizer.json` sidecar read policy, and HFQ chat-template extraction used by runtime tokenizer/prompt compatibility checks; runtime still owns tokenizer parsing and encode/decode.
- `hipfire-model` owns parameterized local model discovery helpers for direct paths, model-directory lookup, aliases, fuzzy scans, quant preference ranking, sidecar exclusion, and adjacent DFlash draft sidecar discovery; server, CLI, and eval adapters provide environment-specific paths.
- `hipfire-model` owns the concrete GGUF artifact parser plus model-source opening policy for HFQ files and safetensors directories while runtime supplies the remaining concrete HFQ/safetensors loader constructors.
- `hipfire-model` owns common model-load request/parameter and loaded-response contracts used by daemon protocol clients and future direct library adapters.
- `hipfire-model` owns common model-load parameter construction from config values, including `auto` mode elision, explicit `dflash_mode=off`, and empty sidecar filtering, now consumed by both Rust server and CLI run load paths.
- `hipfire-model` owns eval model-manifest row construction for local-file/tag identity, file/tag hashes, HFQ metadata hashes, architecture IDs, embedded quantization hashes, and model-specific hash/HFQ metadata compatibility helpers consumed by evidence/runtime paths. It consumes `hipfire-hash` directly instead of re-exporting generic hash helpers.
- `hipfire-model` owns OpenAI-compatible `/v1/models` list response rendering consumed by the Rust server.
- Runtime and arch callers now consume `hipfire-model` model-source contracts directly; `hipfire-runtime::gguf` remains a compatibility re-export and `hipfire-runtime::model_source` remains only the concrete HFQ/safetensors opener adapter until those loaders move.

Current state boundary status:
- `hipfire-state` owns sequence-state handles, Qwen3.5 session/checkpoint handle policy, parsed handle contracts, prefix-hash data shapes, reserve/describe/release/release-sessions/unload-worker/fork/checkpoint request metadata, checkpoint source-residency, prefix-hash, and logical-position validation policy, page descriptors, generate-session state-kind validation, required-kind, set-compatibility, prefix-hash canonicalization, and Qwen3.5 compatible-kind label policy, model artifact memory accounting, worker memory/runtime view structs that carry model-owned worker ids, arena backend support policy over model-owned architecture classification, generic reservation helpers, reservation sizing/projection and memory-pressure policy, allocator/spill policy vocabulary, LRU checkpoint eviction candidate ordering, and JSON rendering for state descriptors, reserve-session success/rejection responses, `describe_state_done` responses, release-state responses, release-sessions responses, unload-worker responses, worker allocator policy status, and runtime-workers health summaries.
- `hipfire-daemon` consumes the shared checkpoint request metadata while still owning loaded-model state maps, Qwen3.5 checkpoint attach/fork/release behavior, and backend-specific GPU state materialization.

Current scheduler boundary status:
- `hipfire-scheduler` owns Rust parity contracts for priority classes, prefill/decode scheduler policy, model-worker compatibility, request-session drafts, prefill queue selection, decode active-set batching, backpressure, opportunistic dispatch, deadline aging, and server health JSON contracts for scheduler-derived prefill/decode/state-cache metadata while consuming state-owned state-kind compatibility policy.
- `hipfire-scheduler` consumes model-owned worker-key identity helpers directly from `hipfire-model` without re-exporting them through scheduler.
- Rust server `/health` consumes scheduler-owned health JSON builders while live Rust request queuing remains daemon-serial.
- Bun scheduler code remains the live batching control plane until Rust server request paths consume the shared scheduler queue/session contracts.

Current generate boundary status:
- `hipfire-generate` owns typed generation sampling policy, text/VL generation request structs, generation event structs, generate-batch prefill/decode envelopes, semantic boundary checkpoint and prefill checkpoint hook contracts, prepared prefill/result and fused dense batch contract types, decode step result contracts, batch/preflight JSON validation, `prefix_hash_preflight_done` response rendering, Qwen3.5 `generate_batch_prefill_session_done`/`generate_batch_prefill_done` response rendering, Qwen3.5 `generate_batch_decode_step_done` response rendering, Qwen3.5 prefix-hash compute/JSON helpers over the state-owned hash shape without re-exporting that state contract, Qwen3.5 prefill/decode backend plan and selector policy, fused prefill preflight helpers, prefill scratch batch sizing policy, and scheduler metadata helpers that render compatible state kinds from `hipfire-state`.
- `hipfire-generate` owns OpenAI chat-message to structured text-generate request construction, including the last-user `prompt` compatibility fallback and full `messages` forwarding.
- `hipfire-generate` owns OpenAI chat completion and streaming chunk JSON response rendering consumed by the Rust server.
- `hipfire-generate` owns prompt-only structured text-generate request construction, generation sampling default/override merging, and worker/tools/system request decoration consumed by Rust server and CLI daemon adapters.
- `hipfire-cli run` consumes `hipfire-generate` structured text-generate request construction directly instead of pre-rendering ChatML locally.
- `hipfire-generate` consumes model-owned architecture classification for Qwen3.5 dense/MoE batch backend decisions and decode scheduler fallback metadata instead of owning local arch-id checks.
- `hipfire-daemon-protocol` uses generate-owned text request and token/done/error event structs for daemon JSONL generate traffic without re-exporting them as compatibility aliases.
- `hipfire-daemon` consumes the shared text/VL generation request contracts, generate-batch prefill, prefix-hash preflight, decode envelope/result, semantic boundary checkpoint, prefill checkpoint hook, prepared prefill/result, and fused dense batch contracts, state-owned prefix-hash/checkpoint metadata, prefix-hash helpers, validation and prefill/preflight/decode-step response rendering, Qwen3.5 backend-plan/selector contracts, fused prefill preflight helpers, and prefill scratch sizing policy while preserving its existing execution paths.
- `hipfire-daemon` still owns model-specific execution, Qwen3.5 runtime orchestration, and JSONL adapter dispatch until later migration slices consume more shared logic.

Current CPU/backend boundary status:
- `hipfire-cpu` owns deterministic BF16 CPU oracle helpers, dense FFN/projection module contracts, generic `ModuleInvocation`/`ModuleOutput` vocabulary, the shared `BackendSelection` contract, backend selection evidence structs, and JSON rendering for module outputs.
- `hipfire-arch-qwen35::ffn_bf16` owns Qwen3.5 mode/env parsing and exposes shared CPU/NPU oracle and artifact contracts only within the arch crate instead of re-exporting backend helper APIs publicly.

Current ROCm/backend boundary status:
- `hipfire-rocm` owns ROCm device identity, backend-path classification, dense FFN/projection module execution evidence, and JSON rendering for ROCm module outputs.
- Qwen3.5 dense FFN trace output records shared ROCm evidence for the existing `weight_gemv_swiglu_residual` path without moving HIP dispatch or kernel code.

Current NPU/backend boundary status:
- `hipfire-npu` owns XDNA1 SwiGLU module artifact and admission contracts for explicit NPU opt-in paths.
- `hipfire-arch-qwen35::ffn_bf16` consumes the shared XDNA1 artifact and admission contracts for env-derived `HIPFIRE_QWEN35_XDNA1_XCLBIN` / `HIPFIRE_QWEN35_XDNA1_INSTR` paths while Qwen3.5 still owns XRT handle creation, FFI dispatch, and GPU fallback execution.

Current daemon protocol boundary status:
- `hipfire-daemon-protocol` owns typed daemon JSONL request/response envelopes while model load request/loaded-response and generate token/done/error payloads live in shared boundary crates.
- `hipfire-server` consumes `hipfire-daemon-protocol` directly for daemon JSONL request/response contracts; the old server-local protocol re-export has been retired.
- `hipfire-daemon` consumes the shared generate request contract opportunistically for common generate fields while preserving raw-JSON fallbacks for legacy/daemon-only fields.
- `hipfire-daemon` also consumes generate-owned batch prefill/preflight/decode contracts, validators, prefix-hash helpers, selector policy, scratch sizing policy, scheduler metadata, and state-owned prefix-hash/checkpoint metadata so daemon-local duplicate batch structs, hash helpers, validation helpers, checkpoint request structs, and Qwen3.5 batch policy helpers are retired.
- `hipfire-daemon` also consumes the shared load request contract for common load fields (`model`, `max_seq`, `physical_cap`, `dflash_mode`, `draft`, `kv_cache`, `cask_sidecar`) while preserving raw-JSON fallbacks for legacy/daemon-only load fields.
- `hipfire-daemon-protocol` uses model-owned load request/params and loaded response structs for daemon JSONL traffic without re-exporting them as compatibility aliases.

Current daemon adapter boundary status:
- `hipfire-daemon-adapter` owns the async stdio JSONL process client, daemon binary discovery and missing-binary diagnostics, load/ping/unload/generate response loops, stale-response filtering, and daemon startup resource lease policy/helpers.
- `hipfire-cli run` consumes `hipfire-daemon-adapter` and `hipfire-daemon-protocol` directly for daemon-backed execution instead of reaching those contracts through the server crate.
- `hipfire-server` consumes `hipfire-daemon-adapter` directly for daemon-backed execution; the old server-local engine re-export has been retired.
- `hipfire-coherence` consumes `hipfire-daemon-adapter` for daemon binary discovery while retaining its coherence-specific raw JSONL runner and detector orchestration.

Current config boundary status:
- `hipfire-config` owns `HipfireConfig`, config-file loading, per-model override merging, and `~/.hipfire` / model-directory path helpers.
- `hipfire-cli` and `hipfire-server` consume `hipfire-config` directly for config and local model discovery paths; the old `hipfire-server::config` compatibility facade has been retired.
- `hipfire-cli` and `hipfire-server` consume `hipfire-model` display/list helpers directly instead of re-exporting them through local model modules.

Current evidence boundary status:
- `hipfire-hash` owns stable byte hash, file hash, fallback file hash, and deterministic score primitives used by model identity and evidence records.
- `hipfire-evidence` owns stable evidence hash policy, directory digest, eval status, host-profile and sourced-field contracts, host-profile hardware-kind/bucket/bandwidth/hash policy, model/tag hash and HFQ metadata compatibility wrappers that delegate to `hipfire-model`, eval reference/slice/llama integrity verifiers, the standard evidence artifact catalog, catalog-based evidence artifact directory discovery, evidence artifact collection status policy, standard evidence artifact contract/JSON rendering, comparison/admission artifact contract rendering, admission required/observed evidence catalogs, the generic evidence artifact record contract/JSON renderer, external evidence record selection/annotation, quality/performance/phase-timing/memory/launch-count/MoE-router/profiling/module-evidence/DFlash-trace/Path-C-trace metric detection/projection policies, comparison metric-direction policy, admission quality/review policy, admission verdict policy, run-provenance contract/JSON rendering, run-metadata artifact contract/JSON rendering, and artifact-index entry contract/JSON rendering.
- `hipfire-evidence` owns comparison, admission, prompt-ledger, and host-profile artifact-index entry variants so eval no longer mutates those index JSON shapes locally.
- `hipfire-eval` owns the `hipfire-eval` binary adapter, eval execution, artifact writing, and harness implementation. It consumes shared model, evidence, and coherence crates directly for model identity, artifact/provenance/status contracts, host-profile reporting, sourced-field values, and daemon-backed coherence rows.
- `hipfire-eval` exposes an explicit `--executor daemon` mode for protocol-backed eval rows. The daemon executor uses `hipfire-daemon-adapter`, `hipfire-model`, and `hipfire-generate` to run smoke model-load, finite greedy decode, and repeated greedy reset/recall through one resident daemon model load; speed anchors still need daemon-backed implementations.
- Runtime examples now consume `hipfire-evidence` eval integrity verifiers directly, and the old `hipfire-runtime::eval_common` compatibility facade has been retired.
- Runtime host-profile code now consumes `hipfire-eval` directly, and the old `hipfire-runtime::eval_harness` compatibility facade has been retired.

Current coherence boundary status:
- `hipfire-coherence` owns detector profile selection, detector-bank construction, agentic prompt detection, coherence run input/output contracts, report row serialization, coherence artifact serialization, daemon-backed coherence execution, prompt execution, token event capture, and detector report assembly.
- Runtime `coherence_probe` and eval harness paths now consume `hipfire-coherence` directly; the old `hipfire-runtime::coherence_runtime` compatibility facade has been retired.

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
