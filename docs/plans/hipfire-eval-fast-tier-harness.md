# Rust `hipfire eval` Fast-Tier Harness

## Summary

Add `hipfire eval` as a Bun CLI command that delegates to a new Rust
`hipfire-eval` runner in `hipfire-runtime`. The first implementation ships
the shared harness framework plus the `fast` tier, with `targeted` and
`extensive` accepted as CLI concepts but only running batteries that are
implemented.

The v1 harness reuses existing Hipfire fixtures and quality infrastructure,
avoids vendoring third-party benchmark code, preserves prompt/model/binary
metadata discipline, and keeps profiling disabled by default.

## Key Changes

- Add a Rust binary at `crates/hipfire-runtime/src/bin/hipfire_eval.rs`, backed
  by shared modules under `crates/hipfire-runtime/src/eval_harness/`.
- Add `hipfire eval ...` in `cli/index.ts`.
- Resolve the eval binary in this order:
  - `HIPFIRE_EVAL_BIN`
  - repo dev build `target/release/hipfire-eval`
  - installed `~/.hipfire/bin/hipfire-eval`
- Pass remaining CLI args through unchanged.
- Update install scripts to build and copy `hipfire-eval` alongside daemon and
  helper binaries.

Public run surface:

```bash
hipfire eval --model <model> --tier fast
hipfire eval --model <model> --tier fast --out <dir>
hipfire eval --model <model> --tier fast --draft <draft> --dflash auto
hipfire eval --model <model> --battery smoke,quality,speed
hipfire eval --model <model> --tier fast --profile passive
```

Core Rust interfaces:

- `EvalConfig`: model path/name, optional draft, tier, selected batteries,
  output dir, KV mode, max tokens, profiling mode.
- `EvalTier`: `fast`, `targeted`, `extensive`.
- `BatteryId`: `smoke`, `quality`, `retrieval`, `longctx`, `speed`,
  `dflash`, `prompt_shape`, `structured`, `vision`, `cask`, `profile`.
- `EvalResult`: one JSONL row per subcase with pass/fail/skip, metrics,
  prompt hash, model/draft identifiers, commit SHA, binary hash, arch, ROCm,
  KV mode, and timing fields.

## Implementation Details

Fast tier batteries:

- `smoke`: one boot session covering load metadata, tokenizer canaries, finite
  greedy decode, and multi-turn reset/recall.
- `quality`: one canary pass using the existing KLD/PPL/reference-slice logic
  from the quality-baseline harness. MQ4 is treated as the control row when
  multiple variants are provided; MQ3/MQ6 compare against the same pinned
  fixture.
- `retrieval`: one Hipfire-native synthetic retrieval canary generated from
  committed/seeded fixtures, not third-party code.
- `speed`: one loaded model session measuring pp32, pp128, TTFT, and short
  decode on hash-pinned prompts.
- `dflash`: AR runs first, then DFlash only when `--dflash auto|on` and a draft
  is provided or discoverable. Missing/disabled draft records a skipped row,
  not a failed eval.
- `prompt_shape` and `structured`: reuse the boot session with committed
  whitespace/template and tool-call/JSON-ish canary prompts.

Optional batteries:

- `vision` and `cask` are defined in the CLI/result schema but disabled by
  default in v1.
- They return explicit `skipped: not implemented in fast v1` unless selected
  after their implementation lands.

Profiling:

- Default is `--profile off`.
- `--profile passive` runs only representative anchors after scored metrics
  are collected and writes artifacts under the output directory.
- Profiling data is evidence only; it does not affect pass/fail scoring.

Result output:

- Default output directory when run from the repo:
  `benchmarks/results/eval/<utc>-<model-stem>-<tier>/`.
- Default output directory outside the repo:
  `~/.hipfire/eval/results/<utc>-<model-stem>-<tier>/`.
- Write `manifest.json`, `results.jsonl`, `summary.md`, and optional
  `artifacts/`.
- Every prompt used by scored batteries must have a recorded hash; canonical
  prompts stay out of `/tmp`.

## Test Plan

No-GPU checks:

- Cargo check the new `hipfire-eval` binary with the same runtime features used
  by existing eval examples.
- Unit-test CLI parsing, tier expansion, battery selection, skip reasons,
  fixture hashing, output path generation, and JSONL/manifest serialization.
- Add Bun CLI tests for `hipfire eval --help`, binary resolution, and
  pass-through args.

Quality regression checks:

- Run the quality canary against an existing pinned reference slice with
  `--quality-max-chunks 1`.
- Confirm repeated same-arch runs produce stable KLD/PPL/argmax rows within
  expected noise.

GPU smoke:

- Run `hipfire eval --tier fast` on a small local HFQ model with short token
  limits.
- Confirm one-load grouping for smoke/prompt/structured and one-load grouping
  for speed/DFlash.
- Confirm DFlash records AR first, then DFlash, and skips cleanly when no draft
  is enabled.

Profiling validation:

- Run one `--profile passive` pass and compare scored anchor metrics against an
  unprofiled run.
- Reject profiling hooks if they move medians outside normal within-session
  variance.

## Assumptions

- v1 targets Hipfire HFQ models first; GGUF/llama comparison remains available
  through existing quality examples until promoted into the unified harness.
- Third-party benchmark repos in `~/.hipfire/third_party` are concept
  references only and are not vendored.
- Dense Qwen 3.5 variants do not need separate performance runs when the
  matching Qwen 3.6 dense model is available; MoE variants remain separate
  because router hit rates can change performance.
- DFlash remains opt-in and is never claimed as a scored result unless AR
  coherence is clean for the same prompt set.
- `targeted` and `extensive` are accepted CLI tiers in v1, but unimplemented
  batteries must report explicit skipped rows rather than silently disappearing.

## Prior Partial Implementation Notes

Before this plan was parked, an initial scaffold had been prototyped with:

- `crates/hipfire-runtime/src/eval_harness/mod.rs`
- `crates/hipfire-runtime/src/bin/hipfire_eval.rs`
- `cli/eval.ts`
- `cli/eval.test.ts`
- Cargo/install/CLI wiring for `hipfire-eval`

That scaffold verified the reporting contract and CLI delegation, but daemon-
backed quality, speed, DFlash, and smoke execution were intentionally explicit
skip rows rather than scored model results.
