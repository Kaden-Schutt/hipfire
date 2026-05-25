# MQ-Lloyd batched prefill — follow-up checklist

**Scope:** future work to add batched-prefill support for `DType::MQ3G256Lloyd`
and `DType::MQ2G256Lloyd` (and MoE-LA/FA support for plain `DType::MQ3G256`,
upstream issue #179) to `crates/hipfire-arch-qwen35/src/qwen35.rs`.

**Date created:** 2026-05-06.
**Created by:** Claude (Opus 4.7) at end of the Lloyd-MQ3 perf-tuning session
on `lloyd-max-mq3-spike` (commits `4ba2f24` … `50b384e`). Companion to
the session devlog at `benchmarks/results/devlog_20260506_lloyd_mq4_extension.md`.

## Current state (safe but latent)

The Lloyd-MQ3 / Lloyd-MQ2 work landed on this branch is **functionally
correct** because of a multi-gate safety net:

1. **`is_batchable_la` (qwen35.rs:3660)** — explicit allowlist of dtypes
   the batched prefill path supports. Currently:
   `MQ4G256 | HFQ4G256 | MQ6G256 | HFQ6G256` always; `MQ3G256` only on
   gfx11/12 archs with WMMA. **`MQ3G256Lloyd` and `MQ2G256Lloyd` are
   NOT in this allowlist.**
2. **`forward_prefill_batch` (qwen35.rs:3517)** — checks
   `is_batchable_la(weight.gpu_dtype, arch)` for every weight in every
   layer. If any returns false, `eligible = false` →
   per-token `forward_scratch` fallback (lines 3550-3582).
3. **`moe_ffn_all_mq4` (qwen35.rs:3707)** — for MoE layers, requires
   every routed/shared FFN weight to be MQ4. Otherwise drops the
   layer's MoE eligibility, also forcing per-token fallback.

Net effect today: any model containing Lloyd-MQ3 / Lloyd-MQ2 weights, OR
plain MQ3 + MoE, takes the per-token `forward_scratch` path during
prefill. This is correct (forward_scratch is the same B=1 path used
for decode), just slower than batched prefill would be.

## The latent landmine

The `is_mq` / `is_6bit` / `is_mq3` matchers downstream (lines 4063,
4235, 4264, 4311, 4360, 4625, 4655, 4693, 4768, 4919) do NOT include
Lloyd dtypes. They also do not include plain `MQ3G256` in the MoE-LA
(4768) and MoE-FA (4919) variants — issue #179 from upstream.

Today these matcher gaps are dead code (the gates above prevent any
Lloyd / MoE-MQ3 weight from reaching them). But if a future PR enables
batched prefill for Lloyd by adding to `is_batchable_la` WITHOUT also
updating these matchers AND adding a Lloyd-specific GEMM dispatch arm,
the matcher's `else` branch would silently fall through to
`gemm_qkvza_hfq4g256` reading Lloyd weights at HFQ4 stride
(112 vs 136 byte mismatch) — corrupted prefill, fluent-looking but
wrong tokens. The bug class has hit this codebase before; the original
warning was raised by a parallel agent investigation on gfx906 MQ2/MQ3.

## Required checklist before declaring batched-prefill MQ-Lloyd done

Verbatim from the originating warning:

> 1. Grep the arch crate to confirm both new dtypes appear in every
>    relevant matcher:
>    `grep -nE 'is_mq = matches!|qkv_is_mq = matches!|wo_is_mq = matches!|ffn_is_mq = matches!|w_down_is_mq = matches!|fa_.*_is_mq = matches!' crates/hipfire-arch-qwen35/src/qwen35.rs`
>    There should be 28 hits in qwen35.rs, each one listing every
>    loader-producible rotated quant (MQ4G256 + MQ6G256 + MQ3G256 + the
>    new MQ3-Lloyd + MQ2-Lloyd, where the latter two replace or sit
>    alongside the existing entries depending on whether they're rotated
>    like MQ3/MQ2). *(Note: at the time this doc was written the grep
>    returned 10 hits, not 28 — most of the additional matchers from the
>    pre-modular-split era have been refactored away. The principle still
>    holds: enumerate every site that gates rotated-quant dispatch.)*
>
> 2. Pay specific attention to the MoE-LA matcher at qwen35.rs:4768
>    and the MoE-FA matcher at qwen35.rs:4919. They currently drop
>    plain MQ3G256 (upstream issue #179, opened 2026-05-06). If #179's
>    fix hasn't merged before this rebase, the MQ3 entries on those two
>    lines need to be added as part of this PR rather than landing as a
>    separate one. The MoE-batched bodies were duplicated inline from
>    the dense LA/FA bodies and the dropped MQ3 was a copy-paste error;
>    the new Lloyd dtypes will inherit the same gap unless explicitly
>    added.
>
> 3. After matcher updates, validate end-to-end on a real model:
>    produce a small `qwen3.5-{0.8b,4b}.mq3-lloyd` via
>    `hipfire-quantize --format mq3-lloyd`, run
>    `./scripts/coherence-gate.sh` on a config that hits the
>    prefill-batched path (any prompt of more than ~16 tokens), and
>    read the report. If first-128-token unique-token-ratio is
>    suspiciously high or coherence is visibly broken while the test
>    passes hard-fail thresholds, that's the symptom of stride-
>    mismatched prefill — open the matchers and re-grep.
>
> 4. Add `qwen3.5-9b.mq3-lloyd` (or whatever your perplexity-validated
>    reference is) to the coherence-gate matrix at
>    `scripts/coherence-gate.sh:84-103`. Without this, future PRs that
>    touch MoE or per-layer dispatch can regress MQ-Lloyd silently.
>    *(2026-05-06 update: 4B + 9B Lloyd-MQ3 rows have been added to the
>    short battery in commit 4ba2f24. Future MoE / batched-prefill work
>    should keep them green and add longer-prompt rows that hit the
>    batched path once it's enabled.)*
>
> 5. The reference document
>    `docs/perf-checkpoints/2026-05-06-quant-dispatch-matrix.md`
>    (commit `cf00664` on the `feat/gfx906-hfq6-hfq8-analysis` branch
>    on Kevin's fork) is a per-quant × workload × arch matrix that
>    should be updated to include the two new Lloyd dtypes once they
>    land.
>
> The audit method that produced this checklist is in
> `docs/plans/gfx906-mq6-mq8-port.md §5.4 part 2` (also Kevin's fork).
> Treat it as the dispatch-wiring sibling of `coherence-gate.sh` —
> build alone is insufficient; runtime dispatch must also be confirmed
> for every new dtype.

## What NOT to do (the trap)

**Do not add `MQ3G256Lloyd` / `MQ2G256Lloyd` to the `is_mq*` matchers
without also (a) adding them to `is_batchable_la`, (b) adding a
Lloyd-specific GEMM dispatch arm, and (c) writing or routing to a
batched Lloyd-prefill kernel.** Doing only the matcher update is dead
code today (gated out upstream); doing it together with `is_batchable_la`
without the dispatch arm is the silent-corruption bug. Land all four
together or none.

The same constraint applies to plain MQ3 + MoE: don't relax
`moe_ffn_all_mq4` without simultaneously fixing the MoE-LA / MoE-FA
matchers AND adding the `gemm_qkvza_hfq3g256_wmma` arms in those
branches.

## Pointers

- `crates/hipfire-arch-qwen35/src/qwen35.rs:3517` — `forward_prefill_batch` eligibility
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3660` — `is_batchable_la` allowlist
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3707` — `moe_ffn_all_mq4` MoE gate
- `crates/hipfire-arch-qwen35/src/qwen35.rs:4063+` — dense LA matchers
- `crates/hipfire-arch-qwen35/src/qwen35.rs:4360+` — dense FA matchers
- `crates/hipfire-arch-qwen35/src/qwen35.rs:4768`  — MoE-LA matcher (#179)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:4919`  — MoE-FA matcher (#179)
- `kernels/src/gemv_mq3g256_lloyd.{,gfx1100.}hip` — current Lloyd-MQ3 GEMV (B=1 only)
- `kernels/src/fused_gate_up_mq3g256_lloyd.{,gfx1100.}hip` — fused gate+up (B=1 only)
- Plan: `docs/plans/PR-115-lload-max-codebooks-mq3.md`
- Devlog: `benchmarks/results/devlog_20260506_lloyd_mq4_extension.md`
