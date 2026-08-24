# Amendment — explicit `sched_group_barrier` partially decouples VGPR from clause depth, but does not close the MQ4-GL gap

**Date:** 2026-08-16
**Lifecycle:** `historical`. Amendment record. Prior records **unchanged**:

- [`2026-08-16-gfx1201-codebook-lookup-lowering-defect.md`](2026-08-16-gfx1201-codebook-lookup-lowering-defect.md)
- [`2026-08-16-amendment-mq4gl-vgpr-clause-tension.md`](2026-08-16-amendment-mq4gl-vgpr-clause-tension.md)
- [`2026-08-16-amendment-host-sync-tax-invalidates-wall-clock.md`](2026-08-16-amendment-host-sync-tax-invalidates-wall-clock.md)

---

## What this refines

The VGPR/clause amendment concluded the register-pressure vs clause-depth
tradeoff was **structural**. That conclusion was reached while leaving the
instruction interleave entirely to the machine scheduler. AMD's Composable
Kernel solves the analogous problem on CDNA with explicit
`__builtin_amdgcn_sched_group_barrier` directives, so the tradeoff was retested
with the interleave *requested* rather than hoped for.

**Result: the tension is a scheduler artifact for moderate register reductions
and genuinely structural for large ones.** The prior record's blanket
"structural" framing is too strong; its practical conclusion survives.

## Context worth recording separately

`__builtin_amdgcn_sched_group_barrier` **compiles clean for gfx1201** (verified:
`hipcc --genco --offload-arch=gfx1201 -O3`, exit 0). It is fully available on
RDNA4. Before this work, `grep -rlE "sched_group_barrier|sched_barrier|iglp_opt"
kernels/src/` matched **0 of 880** kernel files — the technique was entirely
unused in this codebase.

RDNA mask note: `v_wmma_*` is VALU-class, so the RDNA analogue of CK's `MFMA`
mask (8) is `VALU` (2). Relevant masks here are `VMEM_READ` = 32, `DS_READ` =
256, `VALU` = 2.

## Measured — static counters (compile-time facts)

`m=4096 k=5120`, gfx1201, r4 entry point, zero spills throughout.

| variant | VGPR | clauses | maxVMEM | static |
|---|---|---|---|---|
| baseline, `default` profile | 119 | 14 | 11 | 1802 |
| baseline, `memory-clause` profile | 109 | 12 | 16 | 1787 |
| **V1b** `(32,4)(256,32)(2,32)` before X loads | **94** | **13** | **18** | 1863 |
| V1a `(32,8)(256,32)(2,32)` | 94 | 13 | 12 | 1859 |
| V2 interleaved `4x{(32,2)(256,8)(2,8)}` | 94 | 14 | 12 | 1857 |
| V3 grouped inside per-row, before `pk` | 153 | 12 | **4** | 1827 |
| low-VGPR scalar accumulators, no directives | **62** | **8** | 2 | 972 |

**The decoupling is real at moderate scale.** V1b holds 13 clauses and *raises*
max-consecutive-VMEM from 11 to 18 while cutting 25 registers — better than the
`memory-clause` profile on both VGPR and VMEM depth. Directives can therefore
buy register headroom without surrendering clause depth, which the prior record
asserted was impossible.

**It fails at extreme scale.** Explicit scalar accumulators reach 62 VGPR but
collapse to 8 clauses / maxVMEM 2, and adding directives did not recover them.
Past roughly a 25-register reduction the tradeoff reasserts itself.

V3 is recorded as a loser: placing the VMEM group inside the per-row body before
the packed-index load *raised* VGPR to 153 and collapsed maxVMEM to 4. Directive
placement matters more than ratio.

## Wall clock — NOT resolved by this record

The variants were timed host-side (`Instant` around launch +
`device_synchronize`), which
[`2026-08-16-amendment-host-sync-tax-invalidates-wall-clock.md`](2026-08-16-amendment-host-sync-tax-invalidates-wall-clock.md)
establishes carries a ~22.1 µs constant term on this stack. Reported medians were
47.8–48.6 µs across V1b and V2 with no separation between them. **Those numbers
cannot support a wall-clock verdict** and none is claimed here. The task was
dispatched before the instrument problem was understood.

What can be said: no variant produced a *large* wall-clock improvement, and the
static-counter gains did not visibly translate. Whether a ~25-VGPR occupancy gain
is worth anything device-side is **unmeasured**.

## Disposition

- **Not adopted.** No variant is in the tree; the kernel remains at HEAD.
- The reason is scale, not principle: MQ4-GL's device-side deficit against
  `gemv_hfq4g256` is **27%** (412.3 vs 568.0 GB/s). A 25-register occupancy
  improvement is not plausibly worth 27%, so further scalar-path tuning is not
  the lever. That judgement follows from the device-side numbers, not from these
  host-timed ones.
- The technique is **not discredited** and should not be treated as such. It
  demonstrably decouples the two quantities that the prior record called
  inseparable. It was declined here because the target is too far away, and it
  remains available for any kernel where the gap is small enough for occupancy
  to matter.

## Open

- **Device-side timing of V1b.** The one measurement that would settle whether
  25 VGPRs at equal clause depth is worth anything. The pattern is recorded
  verbatim above and is cheap to reconstruct.
- Whether the twelve Lloyd/GL kernels that now share the branchless-LDS-codebook
  shape (commits `e7b0fa56c`, `08aa566f4`) would benefit. They mix
  `global_load`, `ds_load` and FMA in an inner loop, which is the shape these
  directives address. Untested.
- All figures are single-shape, single-arch, batch-1, synthetic-weight
  microbenchmarks. No end-to-end decode measurement exists.
