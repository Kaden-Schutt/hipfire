# Bugs To Investigate

This is a lightweight reminder list. Add a short description, or record
revision + file + line number with a one-line explanation. Do not turn entries
into full investigations here.

- Qwen3 no-output-gate FullAttention faults in fused Q/K/V MQ4 projection;
  split projection should be used until the fused kernel is shape-audited.

## [High] cli/index.ts is a monolithic file
- Category: Maintainability
- Location: cli/index.ts
- Summary: The file is ~490KB, indicating poor module boundaries and mixed responsibilities.
- Suggested fix: Break down into smaller, focused modules (e.g., config, daemon management, CLI commands).
- Scope: Architectural
- Confidence: High

## [High] crates/rdna-compute/src/dispatch.rs is excessively large
- Category: Maintainability
- Location: crates/rdna-compute/src/dispatch.rs
- Summary: The file is ~1.67MB, acting as a massive god-file for kernel dispatching.
- Suggested fix: Split dispatch logic by architecture or kernel family into smaller files.
- Scope: Architectural
- Confidence: High

## [High] crates/hipfire-runtime/examples/daemon.rs is a massive monolith
- Category: Maintainability
- Location: crates/hipfire-runtime/examples/daemon.rs
- Summary: The file is ~16.5K lines, indicating poor module boundaries for the HTTP server and orchestration layer.
- Suggested fix: Extract routing, state management, and request lifecycle logic into separate modules under `src/`.
- Scope: Architectural
- Confidence: High

## [High] Excessive use of .unwrap() leading to potential panics
- Category: Reliability / Maintainability
- Location: Project-wide (e.g., crates/hipfire-quantize/src/main.rs, crates/hipfire-arch-deepseek4/src/forward.rs)
- Summary: The codebase heavily relies on `.unwrap()` on Results and Options, which can cause the daemon or CLI to crash abruptly on unexpected inputs.
- Suggested fix: Replace `.unwrap()` with proper error handling using `Result` and `?`, or provide descriptive `expect()` messages.
- Scope: Cross-cutting
- Confidence: High

## [Medium] Excessive global state via OnceLock and thread_local!
- Category: Architecture / Maintainability
- Location: Project-wide (e.g., crates/hipfire-arch-qwen35/src/qwen35.rs, crates/rdna-compute/src/dispatch.rs)
- Summary: Global variables and thread-locals are used extensively for caching and environment configuration, making testing difficult and hiding dependencies.
- Suggested fix: Inject configuration and state through structs/context objects instead of relying on global statics.
- Scope: Architectural
- Confidence: High

## [High] Missing unit tests for critical path logic in dispatch.rs
- Category: Testing
- Location: crates/rdna-compute/src/dispatch.rs
- Summary: A 46,000-line file that manages critical GPU dispatch logic contains only a single test (`mq_signs_128_deterministic`).
- Suggested fix: Add unit tests for routing logic, fallback choices, and error handling.
- Scope: Local (but high impact)
- Confidence: High

## [High] Unsafe block memory mapping and unchecked aliasing in llama.rs
- Category: Reliability / Security
- Location: crates/hipfire-runtime/src/llama.rs
- Summary: Usage of `unsafe` with `gpu.mq_x_rot.as_ref().unwrap().buf.alias()` combines panics and unsafe pointer aliasing.
- Suggested fix: Validate buffer initialization before attempting unsafe aliasing and provide safe abstractions for GPU memory management.
- Scope: Architectural
- Confidence: High
