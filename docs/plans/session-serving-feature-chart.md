# Canonical Pointer

This index points to the current canonical source in the archive.

- [docs-old/plans/session-serving-feature-chart.md](../docs-old/plans/session-serving-feature-chart.md)

## Why this file exists

The historical implementation record is preserved in `docs-old`; this page is kept as an explicit stable pointer for workflow tooling and review references.

## Current Overlay

- Rust server chat requests now forward structured daemon `messages` while keeping `prompt` as the last-user-text compatibility fallback.
- Rust server OpenAI chat-message conversion now reuses `hipfire-prompt` helpers instead of route-local prompt-boundary mapping.
- Rust server model loads now build typed daemon `LoadParams` from config, preserving explicit DFlash mode and configured TriAttention sidecars over the shared JSONL contract.
- The daemon now opportunistically consumes shared typed generate and load request contracts while keeping raw JSON fallbacks for legacy and daemon-only fields.
- Daemon generate request/token/done/error protocol types now reuse `hipfire-generate` contracts instead of duplicating generate structs in `hipfire-daemon-protocol`.
- Daemon text/VL generation request contracts, generate-batch prefill/preflight/decode contracts, semantic boundary checkpoints, prefill checkpoint hooks, prepared prefill/result, decode result, and fused dense batch contracts, validators, prefix-hash helpers, Qwen3.5 batch backend selectors, fused prefill preflight helpers, scratch sizing policy, and decode scheduler metadata now reuse `hipfire-generate` ownership; prefix-hash data shapes, checkpoint request metadata, and model artifact memory accounting now reuse `hipfire-state` ownership instead of daemon-local duplicates.
- Eval evidence artifact kinds and expected-metric catalogs now reuse `hipfire-evidence` ownership instead of runtime-local lists.
- Eval runtime-evidence directory ingestion now uses `hipfire-evidence` catalog-based artifact path discovery instead of runtime-local filename lists.
- Coherence run contracts, artifact serialization, daemon-backed execution, prompt execution, token capture, and detector report assembly now reuse `hipfire-coherence` ownership while `hipfire-runtime::coherence_runtime` keeps the existing import path.
- Daemon load request/parameter and loaded-response protocol types now reuse `hipfire-model` contracts instead of duplicating common model-load structs in `hipfire-daemon-protocol`.
- Rust server and CLI model discovery now reuse `hipfire-model` local artifact discovery helpers instead of server-local scanning.
- Eval DFlash draft auto-discovery now reuses `hipfire-model` sidecar discovery instead of eval-local candidate parsing.
- Rust scheduler worker-key identity and compatibility helpers now reuse `hipfire-model` ownership instead of scheduler-local model identity code.
- Generate Qwen3.5 dense/MoE batch backend selection now reuses `hipfire-model` architecture classification instead of local numeric arch checks.
- Runtime tokenizer compatibility signatures now reuse `hipfire-model` fingerprint policy while tokenizer parsing and encode/decode stay in `hipfire-runtime`.
- Eval output and runtime-evidence model stems now reuse `hipfire-model` artifact identity helpers instead of eval-local stem sanitization.
- Eval model manifests now reuse `hipfire-model` row construction for file/tag identity, HFQ metadata hashes, architecture IDs, and embedded quantization hashes.
- Runtime model-source opening now reuses `hipfire-model` HFQ/safetensors path policy while keeping concrete loader constructors in `hipfire-runtime`.
- Evidence model/tag hash and HFQ metadata compatibility helpers now delegate to `hipfire-model` instead of carrying duplicate model-specific parsing.
- Eval reference/slice/llama integrity verifiers now reuse `hipfire-evidence` ownership while `hipfire-runtime::eval_common` keeps the existing import path.
- Eval and host-profile reporting now reuse `hipfire-evidence` eval-status, host-profile, and sourced-field contracts instead of eval-harness-local JSON shapes.
- Eval host-profile hardware-kind, bucket, bandwidth, and hash policy now reuse `hipfire-evidence` ownership instead of eval-harness-local helpers.
- `hipfire-eval` now owns the `hipfire-eval` binary adapter and eval harness implementation while `hipfire-runtime::eval_harness` keeps the existing import path.
- Daemon model-worker id construction and sequence-state arena support policy now reuse `hipfire-state` ownership instead of daemon-local policy helpers.
- Daemon Qwen3.5 sequence-state session/checkpoint handle construction now reuses `hipfire-state` ownership instead of daemon-local policy helpers.
- Daemon `describe_state` response rendering now reuses `hipfire-state` JSON helpers instead of daemon-local JSON construction.
- Daemon release-state response rendering now reuses `hipfire-state` JSON helpers instead of daemon-local JSON construction.
- Daemon startup resource lease policy and lock helpers now reuse `hipfire-daemon-adapter` ownership instead of daemon-local helpers.
- Eval artifact row records now reuse `hipfire-evidence` record contracts instead of eval-local JSON construction.
- Eval comparison, admission, and evidence artifacts now reuse `hipfire-evidence` run-provenance contracts instead of eval-local provenance structs.
- Eval artifact index entries now reuse `hipfire-evidence` index rendering instead of eval-local JSON construction.
- Eval external evidence ingestion now reuses `hipfire-evidence` record selection and annotation helpers instead of eval-local JSON mapping.
- Eval run metadata artifacts now reuse `hipfire-evidence` run metadata contracts and JSON rendering instead of eval-local schema construction.
- Eval comparison artifacts now reuse `hipfire-evidence` metric-direction policy instead of eval-local metric classification.
- Eval admission findings now reuse `hipfire-evidence` quality/review policy instead of eval-local rejection classification.
- Eval admission required/observed evidence catalogs now reuse `hipfire-evidence` ownership instead of eval-local hard-coded lists.
- Eval admission verdicts now reuse `hipfire-evidence` verdict policy instead of eval-local promote/review/reject mapping.
- Eval evidence artifacts now reuse `hipfire-evidence` collection status policy instead of eval-local collected/requested/disabled/not-collected mapping.
- Eval standard evidence artifacts now reuse `hipfire-evidence` JSON rendering instead of eval-local schema construction.
- Eval comparison/admission artifacts now reuse `hipfire-evidence` JSON rendering instead of direct eval-local struct serialization.
