# Test Entry Points

Put pass/fail checks in this directory. A file belongs here when its primary
contract is "return 0 when the behavior is acceptable, return non-zero when it
is not."

Examples:

- CI-safe checks such as `no-gpu-ci.sh`
- hardware gates such as `coherence-gate-dflash.sh`, `pp-gate.sh`, and
  `speed-gate.sh`
- server or daemon smoke tests that validate an end-to-end behavior
- pytest files and small shell regression tests

Keep reusable helpers, diagnostics, data conversion, profiling, corpus tooling,
installation helpers, and benchmark runners in `scripts/` or `benchmarks/`.
Rust GPU/kernel harnesses can remain as Cargo examples under `crates/*/examples`
when they are built and invoked through Cargo, but any repo-level wrapper that
decides pass/fail should live here.

Do not add compatibility symlinks or wrappers under `scripts/` for new tests.
Update callers and documentation to use `./tests/<name>` directly.
