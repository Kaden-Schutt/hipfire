# AGENTS.md - scripts

This subtree owns reusable shell workflows, sweeps, profiling helpers, and
benchmarks that are not formal test gates.

## Script Rules

- GPU scripts must acquire `hipfire gpu-lock` unless they exclusively drive a
  daemon path that acquires resource leases itself.
- Scripts that compare model behavior or speed must read committed prompt files
  from `benchmarks/prompts/` when exact prompt shape matters.
- Keep artifact names aligned with the root canonical naming convention. Update
  old spellings found in scripts as part of the fix.
- Prefer small, composable shell wrappers around existing Rust binaries. Do not
  put Python in runtime or inference execution paths.
