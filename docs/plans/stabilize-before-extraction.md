# Stabilize Before Extraction

This plan has been merged into
[`modular-runtime-architecture.md`](modular-runtime-architecture.md).

The short version remains unchanged: stabilize correctness gates and typed
boundaries before splitting large hot-path files such as `qwen35.rs`,
`dispatch.rs`, or daemon serving state.

