# Canonical Pointer

This index points to the current canonical source in the archive.

- [docs-old/plans/session-serving-feature-chart.md](../docs-old/plans/session-serving-feature-chart.md)

## Why this file exists

The historical implementation record is preserved in `docs-old`; this page is kept as an explicit stable pointer for workflow tooling and review references.

## Current Overlay

- Rust server chat requests now forward structured daemon `messages` while keeping `prompt` as the last-user-text compatibility fallback.
- Rust server model loads now build typed daemon `LoadParams` from config, preserving explicit DFlash mode and configured TriAttention sidecars over the shared JSONL contract.
- The daemon now opportunistically consumes shared typed generate and load request contracts while keeping raw JSON fallbacks for legacy and daemon-only fields.
