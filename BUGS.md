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
