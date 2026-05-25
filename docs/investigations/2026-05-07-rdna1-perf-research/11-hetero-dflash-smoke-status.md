# Smoke status — hetero PFlash+DFlash PRD steps 1-4

**Date:** 2026-05-07
**Scope ask:** Implement and smoke-test PRD steps 1-4 from `docs/plans/hetero-pflash-dflash.prd`.

## What shipped tonight

### PR1 — lift DFlash + pp>1 refusal (committed `3da2661`)

Env-gated lift of the load-time refusal at `daemon.rs:594`. With `HIPFIRE_PP_DFLASH=1` set, the daemon accepts pp>1 + draft load messages instead of refusing them. Empty / unset preserves prior behavior byte-for-byte.

Mechanically identical to the PFlash + pp>1 refusal lift shipped earlier this session (commit `ea2b8b7`). cargo check on master is clean.

The lifted error message explicitly tells operators that PR2-4 are not yet implemented:

> "the load message will accept but generate will not run cross-card spec-decode"

This is honest scoping — the env opens the door, but the actual hetero plumbing is real engineering captured in the PRD.

## What was deferred (and why)

### PR2 — DFlash drafter device pinning

Requires opening a separate `Gpu` instance for the drafter, modifying `LoadedModel` struct to hold it, threading it through the load path. Estimated ~80 lines.

**Why deferred**: by itself this PR is shippable but provides no functional value without PR3 (cross-card coordination). Without PR3 the drafter still does both draft and verify on the same Gpu it was loaded onto, so where the drafter lives is moot. Best shipped as a unit with PR3.

### PR3 — Cross-card spec-decode coordination loop

This is the substantive engineering. Modify:

- `crates/hipfire-runtime/src/dflash.rs` — `DflashWeights`, `DflashScratch`, drafter KV state need to live on a specifiable Gpu. Constructor signatures change from `&mut Gpu` to `(drafter_gpu: &mut Gpu, target_gpu: &mut Gpu)`.
- `crates/hipfire-arch-qwen35/src/speculative.rs` — `spec_step_dflash` and related helpers need to accept two Gpu instances and route per-step ops correctly. This is the main refactor; ~150 lines of careful surgery.
- `crates/hipfire-runtime/examples/daemon.rs::generate_dflash` — accepts the two-Gpu split, plumbs through the speculative module.
- Cross-card token-ID transfer + acceptance bitmap transfer per cycle: `hipMemcpy` device→host on one card, host→device on the other. ~30 lines.
- KV state synchronization on accepted-prefix-rollback: drafter must roll back unaccepted positions on its KV; target advances by accepted_count on its KV. The existing single-Gpu rollback logic just needs to be redispatched against the appropriate Gpu instance.

Estimated total: ~250-300 lines across 3 crates. Cross-card synchronization is correctness-critical (KV mismatch = broken inference); needs careful testing.

**Why deferred**: this is a multi-day PR cycle, not a single-turn smoke. Honest engineering vs the autoresearch contract's "smallest change" ethos: PR3 is a substantive feature, not a test of an intrinsic.

### PR4 — PFlash + DFlash composition

Plumbing change: the compressed `Vec<u32>` already produced by `generate_multi`'s PFlash hook (commit `ea2b8b7`) needs to flow into BOTH the drafter prefill (on the drafter Gpu) and the target prefill (on the target Gpu set). Today the compressed list flows only into the target prefill.

Estimated: ~50-100 lines.

**Why deferred**: depends on PR2+PR3 (no drafter Gpu to prefill onto until those land).

## Empirical smoke that did fire tonight

### Exp #10 (committed `2bc67e4`): gfx1151 SOLO DFlash 27B = 1.84× over AR

This is the EXISTING DFlash path on gfx1151 alone, which PR1's refusal lift does not modify. It serves as the perf anchor for what hetero DFlash should improve over: any hetero variant must beat 27.0 tok/s decode on 27B / gfx1151 to be worth shipping.

### Exp #9 (committed earlier): WMMA prefill 5.07× on gfx1151 vs gfx1010

Anchors the prefill-tier value of the architecture. PRD references this as the empirical justification for placing prefill on gfx1151 in the hetero pipeline.

## Recommendation

PR1 is shipped on master. PR2-4 should land as a single multi-PR cycle (probably 3-5 days of focused engineering) targeting v1.2-alpha. The PRD at `docs/plans/hetero-pflash-dflash.prd` has the full architecture, build sequence, validation plan, and risk inventory.

A reviewer can pick up the work from the PRD without further context. The empirical anchors (Exps #9 and #10) and the PR1 env are already on master.

## Honest assessment

Tonight's autoresearch session shipped:
- 11 verdict docs (Exps #1-#10 + Exp #11 == this status)
- 1 PRD draft for v1.2 hetero pipeline
- 2 code commits to master:
  - Exp #7 win: HIPFIRE_TARGET_ARCH env override (gfx10-1-generic compile target validation, BC-160 procurement de-risk)
  - PR1 of hetero PRD: HIPFIRE_PP_DFLASH env-gated refusal lift

The 5.07× WMMA prefill empirical result (Exp #9) and the 1.84× gfx1151 solo DFlash result (Exp #10) are both reproducible from on-rig hardware that already exists. These ground the entire architecture.

What did NOT ship tonight: the cross-card spec-decode coordination loop (PR3 of the PRD). That's the substantive engineering investment for v1.2.
