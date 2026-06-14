# Canonical Pointer

This index points to the current canonical source in the archive.

- [docs-old/plans/session-serving-feature-chart.md](../docs-old/plans/session-serving-feature-chart.md)

## Why this file exists

The historical implementation record is preserved in `docs-old`; this page is kept as an explicit stable pointer for workflow tooling and review references.

## Current Overlay

- Rust server chat requests now forward structured daemon `messages` while keeping `prompt` as the last-user-text compatibility fallback.
- Rust server model loads now build typed daemon `LoadParams` from config, preserving explicit DFlash mode and configured TriAttention sidecars over the shared JSONL contract.
- The daemon now opportunistically consumes shared typed generate and load request contracts while keeping raw JSON fallbacks for legacy and daemon-only fields.
- Daemon generate request/token/done/error protocol types now reuse `hipfire-generate` contracts instead of duplicating generate structs in `hipfire-daemon-protocol`.
- Daemon generate-batch prefill/preflight/decode contracts, semantic boundary checkpoints, prefill checkpoint hooks, validators, prefix-hash helpers, Qwen3.5 batch backend selectors, scratch sizing policy, and decode scheduler metadata now reuse `hipfire-generate` ownership instead of daemon-local duplicates.
- Daemon load request/parameter protocol types now reuse `hipfire-model` contracts instead of duplicating common model-load structs in `hipfire-daemon-protocol`.
