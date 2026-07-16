# COR-003 fresh-session handover: native MTP cancellation

> **Historical handover — superseded (2026-07-16).** This document records
> the remaining gaps as observed on 2026-07-15. The native Qwen35 and DeepSeek
> in-flight cancellation work, including production-owned lifecycle tests, is
> now implemented as part of COR-003. Fresh CPU, DFlash coherence, and
> multi-turn serving evidence passed on 2026-07-16: workspace tests, DFlash
> report `/tmp/coherence-dflash-20260716-110721.md` with no hard or soft
> warnings, and serving report `/tmp/serve-multiturn-20260716-110919.md`.
> `git diff --check` also passed. Transactional Qwen speculative-target
> loading remains deferred to `SPEC-003`; preserve this handover as forensic
> evidence, but use [the canonical device-mesh tracker](../../.agent-progress/device-mesh-refactor-tracker.md)
> for current status.

## Objective

Finish COR-003 without creating a separate tracker task. Native Qwen35 and
DeepSeek MTP prefill must honor cancellation *during* their long token/chunk
loops, restore draft/target guards correctly, and publish no first token after
cancellation.

## Current state

The sealed-turn, PP, DeepSeek discard, and DSpark cancellation work is in the
working tree. Targeted tests pass, but review found two remaining gaps:

1. Native Qwen35 and DeepSeek drafters only check abort before/after blocking
   `mtp_prefill`; their internal loops ignore cancellation.
2. Existing reset-order tests use helpers, not real
   `generate_qwen35`/`generate_deepseek4_spec` control paths.

## Required design

- Extend the existing optional `MtpDrafter` prefill cancellation seam; do not
  create a second trait or a new tracker task.
- Native MTP implementations must check the callback at bounded chunk/token
  boundaries and return cancellation promptly.
- `generate_spec` rechecks cancellation before first-token admission/output.
- On cancellation/error after state exists: release borrows, restore target and
  PBS/guard state, run canonical reset including DeepSeek decode-cache zeroing,
  then emit abort/error envelopes. No parser/emitter finalization or cache
  insertion is allowed.
- Preserve the sealed-boundary policy: no approximate rollback or token pop.

## Likely files

- `crates/hipfire-runtime/src/spec.rs`
- `crates/hipfire-arch-qwen35/src/mtp_spec.rs`
- `crates/hipfire-arch-qwen35/src/forward.rs`
- `crates/hipfire-arch-deepseek4/src/spec_impl.rs`
- `crates/hipfire-runtime/src/dspark_core.rs`
- `crates/hipfire-runtime/examples/daemon.rs`

## TDD acceptance cases

1. Cancel between native MTP chunks: no first token/wire event; target/PBS
   guards restored.
2. Cancel during final-head work: same result.
3. Qwen35 and DeepSeek production caller paths each show reset before terminal
   envelope and no cache/turn contamination on a following request.
4. DSpark cancellation behavior remains passing.
5. Existing non-cancelled MTP behavior remains unchanged.

## Verification

```bash
nix develop --command cargo test -p hipfire-runtime --example daemon --locked
nix develop --command cargo test -p hipfire-runtime --lib --locked
nix develop --command cargo test -p hipfire-arch-qwen35 --lib --locked
nix develop --command cargo test -p hipfire-arch-deepseek4 --lib --locked
nix develop --command cargo check -p hipfire-runtime --example daemon
git diff --check
```

Do not commit. Preserve pre-existing unrelated modifications and repository-wide
format drift; format only touched files.
