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
- Coherence run contracts and artifact serialization now reuse `hipfire-coherence` ownership while runtime keeps daemon execution.
- Daemon load request/parameter and loaded-response protocol types now reuse `hipfire-model` contracts instead of duplicating common model-load structs in `hipfire-daemon-protocol`.
- Rust server and CLI model discovery now reuse `hipfire-model` local artifact discovery helpers instead of server-local scanning.
- Eval DFlash draft auto-discovery now reuses `hipfire-model` sidecar discovery instead of eval-local candidate parsing.
- Rust scheduler worker-key identity and compatibility helpers now reuse `hipfire-model` ownership instead of scheduler-local model identity code.
- Generate Qwen3.5 dense/MoE batch backend selection now reuses `hipfire-model` architecture classification instead of local numeric arch checks.
- Eval output and runtime-evidence model stems now reuse `hipfire-model` artifact identity helpers instead of eval-local stem sanitization.
- Daemon model-worker id construction and sequence-state arena support policy now reuse `hipfire-state` ownership instead of daemon-local policy helpers.
- Daemon startup resource lease policy and lock helpers now reuse `hipfire-daemon-adapter` ownership instead of daemon-local helpers.
- Eval artifact row records now reuse `hipfire-evidence` record contracts instead of eval-local JSON construction.
- Eval comparison, admission, and evidence artifacts now reuse `hipfire-evidence` run-provenance contracts instead of eval-local provenance structs.
