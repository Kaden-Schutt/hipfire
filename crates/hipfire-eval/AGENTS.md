# AGENTS.md - hipfire-eval

`hipfire-eval` is the first home for model/runtime evidence. Add or repair eval
batteries and suites here before adding new shell-only admission checks.

## Evidence Policy

- Add or update `hipfire-eval` batteries/suites for speed, coherence,
  DFlash/DDTree/Path C, PFlash, agentic/tool-call, long-context, quality, and
  server-runtime admission tests.
- Keep shell gates as wrappers when they provide hook integration, baseline
  comparison, or environment setup that belongs outside the Rust harness.
- Use committed prompt files from `benchmarks/prompts/` for benchmark-sensitive
  runs, and preserve prompt md5 evidence when recording results.
- Keep no-GPU paths working. `./tests/no-gpu-ci.sh` checks `hipfire-eval`, so
  avoid GPU-only assumptions in shared executor code.
