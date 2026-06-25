# AGENTS.md - tests and gates

This subtree owns enforcement wrappers, smoke tests, and CI gates. Keep shell
gates focused on orchestration and evidence collection; put reusable runtime
admission logic in `hipfire-eval` first.

## Gate Policy

- `./tests/coherence-gate-dflash.sh` is the canonical correctness gate for
  kernels, quant formats, dispatch, fusion, rotation, rmsnorm, and spec-decode
  changes.
- `./tests/no-gpu-ci.sh` is the default no-GPU handoff check for workflow-only
  changes.
- GPU gates must acquire `hipfire gpu-lock` unless they exclusively drive a
  daemon path that acquires resource leases itself.
- Preserve byte-identical prompts by reading committed files from
  `benchmarks/prompts/`; do not inline new benchmark prompts in shell heredocs
  when the exact text matters.
- When adding or repairing model/runtime admission coverage, update
  `crates/hipfire-eval/` first and keep shell scripts as wrappers where useful.
