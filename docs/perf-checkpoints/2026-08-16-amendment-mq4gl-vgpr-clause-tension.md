# Amendment — MQ4-GL's byte advantage does not convert to wall-clock: VGPR and clause depth are in direct tension

**Date:** 2026-08-16
**Lifecycle:** `historical`. Amendment record. The original is **unchanged**:

- [`2026-08-16-gfx1201-codebook-lookup-lowering-defect.md`](2026-08-16-gfx1201-codebook-lookup-lowering-defect.md)

**What this closes:** the original's § "Open, not concluded" left this open —
*"Whether MQ4-GL's 4.0625 bpw is a net win is undetermined by this record. At
HFQ4G256's 307 GB/s, GL's 10.65 MB would take 34.7 µs — 4.4% faster than
uniform, matching its byte advantage exactly. Whether that is reachable is
open."*

It is **not reachable on this kernel structure**, and the reason is structural
rather than a coding deficiency.

**What this does NOT change:** the original's central finding — the
divergent-branch codebook lowering defect and its ~2.7-2.8× fix — stands
unmodified and is unaffected by anything here.

---

## Measured

Same fixture as the original: gfx1201 / Navi 48, hipcc 7.14.60850, `m=4096
k=5120`, batch-1 GEMV, synthetic FWHT-rotated weights, warm median n=30 after 5
warmup. Timings below are back-to-back in one thermal state except where noted.

### Wall clock

| source | scheduler profile | R2 | R4 | R8 | uniform HFQ4G256 |
|---|---|---|---|---|---|
| HEAD (`814d8fd95`) | default | 38.16 / 40.03 µs | **37.21 / 39.16 µs** | 39.66 / 42.96 µs | 34.32 / 36.14 µs |
| optimization variants | default | 43.79 µs | 42.23 µs | 43.85 µs | 36.16 µs |
| optimization variants | memory-clause | — | best 36.57, median 38-41 µs | — | 34-36 µs |

MQ4-GL moves **10.65 MB**; uniform HFQ4G256 moves **11.14 MB** (4.6% more).
Uniform is nonetheless faster in every configuration measured. The best GL
result anywhere (36.57 µs) sits against a uniform arm measuring 34.3-36.3 µs in
the same runs — at or inside run-to-run noise, never a reliable win.

### Register counts, radiowave CLI, HEAD source

| profile | r2 | r4 | r8 | r4 clauses | r4 maxVMEM |
|---|---|---|---|---|---|
| default | 94 | 120 | 214 | 15 | 10 |
| **memory-clause** | **93** | **109** | 203 | 12 | **16** |
| max-ilp | 116 | 196 | 240 | 13 | 8 |
| iterative-ilp | 108 | 180 | 240 (**spill 138**) | 12 | 16 |
| pipeline-ilp | 116 | 196 | 240 (spill 2) | 13 | 8 |

Reference: `gemv_hfq4g256_multirow` is **93** VGPR at r4 with 16 clauses and
maxVMEM 16.

---

## The mechanism — why the gap does not close

Every source variant that pushed VGPR toward HFQ4G256's 93 **lost wall clock,
because clause count fell with it**:

| variant | r4 VGPR | r4 clauses | r4 median |
|---|---|---|---|
| baseline | 121 | 15 | ~37.0 µs |
| `sc_ptrs` (scale pointer array) | **88** | **9** | 40.4 µs |
| explicit scalar accumulators | **82** | **10** | 40.4 µs |
| `float4` X hoist | 121 | collapsed | 41.0 µs |

Reaching the target register class costs more in clause depth than it recovers
in occupancy. That is not a tuning plateau; it is a property of the format's
layout.

**MQ4-GL keeps two memory streams** — a 128 B index region and a separate fp16
scale region at `M*gpr*128` — so it needs an extra base pointer and an extra
index computation live per row. With R rows in flight, that address state is
what competes for registers. HFQ4G256 carries `scale`/`min` **inline in the
group**, so one stream and one pointer suffice.

This is the separated scale region's cost surfacing. The kernel-design analysis
argued the split was actively beneficial — 128 B group stride is cleanly
aligned where uniform's 136 B stride rotates, and per-row-contiguous scales
hoist in one coalesced load. Both of those are true and were confirmed. They are
simply outweighed at multirow occupancy by the register cost of keeping two
streams clauseable simultaneously.

`memory-clause` is a genuine free win on top of any source (r4 120 -> 109,
maxVMEM 10 -> 16, zero spills) and it narrows but does not close the gap.

---

## Methodological finding — manifest `scheduler_profile` is not evidence

`radiowave::Compiler::certify_existing` (`crates/radiowave/src/lib.rs:606`)
hardcodes `scheduler_profile: SchedulerProfile::Default` at **:641**. Because
`rdna-compute` compiles with hipcc directly and then *certifies* the object
rather than compiling through `radiowave::Compiler::compile` (lib.rs:537, called
by nothing outside radiowave), **every runtime manifest reports `"default"`
regardless of what produced the object.**

During this work a `memory-clause` object was loaded while its own manifest
described it as `default`, and that field was read as evidence twice before the
hardcode was found. Additionally `cache_hash` covers source + arch + flags +
toolchain + ABI but **not** the profile, so `pair_valid` accepts an object
compiled under a different profile as valid for the current one.

Consequence for anyone reading these records: **a `scheduler_profile` value in a
`.radiowave.json` produced by the runtime cache carries no information.** Only
manifests emitted by `radiowave compile` directly are trustworthy on that field.
Both issues are being plumbed; until then, treat runtime-manifest profile fields
as unset rather than as `default`.

---

## Disposition

- MQ4-GL (qt=40) remains a **size and codec-MSE win with neutral-to-negative
  wall clock** on this kernel structure: 130 vs 136 B/group (−4.41%), −20.9%
  codec MSE, and no reliable decode speed advantage.
- No admission decision is implied. Nothing here authorizes or blocks shipping
  qt=40; it is still not dispatch-resolvable.
- Source variants pursuing register reduction were **not landed** — they measure
  42.23 µs at r4 against HEAD's 37.21/39.16 in the same thermal state, i.e. a
  regression, and their apparent advantage existed only under a profile the
  committed source could not select.

## Open

- **HEAD source × `memory-clause` is unmeasured for wall clock.** The
  memory-clause timings above were taken on the regressed variant source. That
  cell is the honest best case and requires the profile plumbing to exist first.
  Its register counts (r4 = 109) are already the best measured.
- Whether a **single-region layout** for qt=40 — folding the fp16 scale back
  into the group at 130 B — would relieve the register pressure while preserving
  4.0625 bpw. This was declined during kernel design on alignment grounds; the
  VGPR/clause tension is new evidence that was not available then.
- R=3 and R=6 were probed (r3 = 97 VGPR / 12 clauses / maxVMEM 17; r6 = 90 VGPR
  / 14 clauses / maxVMEM 8) but not measured cleanly on HEAD source. The
  occupancy cliff between r4 (109-120) and r8 (203-214) suggests an optimum
  between them.
- All figures remain single-shape, single-arch, batch-1, synthetic-weight
  microbenchmarks. No end-to-end model decode measurement exists for any arm.
