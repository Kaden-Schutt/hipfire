# Bugs To Investigate

This is a lightweight reminder list. Add a short description, or record
revision + file + line number with a one-line explanation. Do not turn entries
into full investigations here.

- Dense native multi-row decode does not match `serial_reference` when
  `HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW=1`; see
  `crates/hipfire-runtime/examples/daemon.rs` fused dense decode path.
- Explicit fused dense decode parity is sensitive to active-vs-resident Qwen35
  session lifecycle ordering.
- 128 fully admitted Qwen35 requests can OOM with FP32 KV/DeltaNet session
  state; current stress gate only passes with deterministic backpressure.
- Semantic-boundary checkpoint cloning can OOM under high fan-in; stress disables
  it with `HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS=0`.
- Server reset/eviction/backpressure cleanup still needs focused tests for
  pending prefill/decode promises and HTTP 503 behavior.
