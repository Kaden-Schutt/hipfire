# AGENTS.md - benchmarks

Benchmark work is forensic: keep prompts, inputs, and result context stable
enough that another run can explain a regression.

## Prompt Discipline

- Prompt structure dictates tau. Use byte-identical prompts from
  `benchmarks/prompts/*.txt` for cross-session, cross-agent, or cross-commit
  comparisons.
- Record the prompt path and md5 alongside benchmark results.
- Do not store canonical benchmark prompts under `/tmp`; use
  `benchmarks/prompts/`, `~/.hipfire/datasets/`, or a committed script.
- Whitespace changes in benchmark prompts or prompt-building scripts can change
  acceptance and tok/s. Treat them as behavior changes.

## GPU Runs

- Benchmark scripts that use GPU examples or direct binaries must acquire the
  native mutex with `hipfire gpu-lock` unless they delegate to a gate that
  already does.
- Tight standard deviation on speculative-decode benches is suspicious. Eyeball
  decoded output when tau or tok/s looks too good.
- Store durable outputs under `benchmarks/results/` or the documented baseline
  directories, not ad hoc temp paths.
