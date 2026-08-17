# Amendment — every wall-clock figure in this campaign carried a ~22 µs host sync tax; device-side timing widens the MQ4-GL deficit from 7% to 27%

**Date:** 2026-08-16
**Lifecycle:** `historical`. Amendment record. Both originals are **unchanged**:

- [`2026-08-16-gfx1201-codebook-lookup-lowering-defect.md`](2026-08-16-gfx1201-codebook-lookup-lowering-defect.md)
- [`2026-08-16-amendment-mq4gl-vgpr-clause-tension.md`](2026-08-16-amendment-mq4gl-vgpr-clause-tension.md)

**Commits:** `58a1f1dce` (device-side instrumentation + scheduler-profile plumbing)

---

## What this corrects

Every microsecond figure in both originals was measured as host `Instant`
elapsed around `kernel launch + hipDeviceSynchronize`. **That is not kernel
time.** On gfx1201 it carries a launch-invariant term of **~22.1 µs**.

Measured directly (`crates/hipfire-runtime/examples/dispatch_floor_probe.rs`),
host minus device time per shape, same kernel, ascending bytes:

| m | bytes | host µs | device µs | **delta** |
|---|---|---|---|---|
| 512 | 1.33 MB | 40.02 | 17.64 | **22.38** |
| 1024 | 2.66 MB | 40.78 | 18.68 | **22.10** |
| 2048 | 5.32 MB | 47.98 | 25.88 | **22.10** |
| 3072 | 7.99 MB | 50.58 | 28.40 | **22.18** |
| 4096 | 10.65 MB | 61.59 | 39.44 | **22.15** |
| 6144 | 15.97 MB | 73.32 | 50.68 | **22.64** |
| 8192 | 21.30 MB | 82.31 | 60.16 | **22.15** |
| 12288 | 31.95 MB | 93.99 | 71.00 | **22.99** |

Dead flat across a 24× byte range. It is the synchronization round-trip, **not
dispatch**: HIP's dispatch floor is ~2.13 µs
([`redline-pub/docs/DISPATCH-FLOOR.md`](../../../redline-pub/docs/DISPATCH-FLOOR.md)),
roughly 10× smaller. Retained PM4 in that same reference reaches
0.176–0.211 µs/dispatch.

The arm-to-arm differences both originals reasoned about were **2–3 µs**, i.e.
roughly 13% of an uncontrolled 22 µs term.

## The corrected measurement

Device-side, via the in-tree HIP-event profiler (`profile::start`/`stop`), which
the GEMV wrappers already fed with byte counts through `profile::begin_timer`.
`m=4096 k=5120`, median over 35 launches, bytes as the wrapper accounts them
(weights + x + y):

| kernel | B/group | bytes | device µs | GB/s | % of ~640 peak |
|---|---|---|---|---|---|
| `gemv_hfq4g256` (uniform) | 136 | 11.18 MB | **19.680** | **568.0** | **89%** |
| `gemv_mq4g256gl_multirow` | 130 | 10.69 MB | 25.920 | 412.3 | 64% |
| `gemv_mq4g256_lloyd` | 160 | 13.14 MB | 31.120 | 422.4 | 66% |
| `gemv_mq4g256gl` single-row | 130 | 10.69 MB | 30.840 | 346.5 | 54% |

The multirow row is a median across the r2/r4/r8 entry points, which share one
kernel name in the profiler.

## What changes, and by how much

**The sync tax was compressing the comparison, not inflating it.** A constant
added to both sides of a ratio pulls it toward 1.

| | host-timed (as published) | device-timed |
|---|---|---|
| MQ4-GL vs uniform | 8.4% slower | **31.7% slower** |
| bandwidth-efficiency deficit | 7.2% | **27%** |

So [`2026-08-16-amendment-mq4gl-vgpr-clause-tension.md`](2026-08-16-amendment-mq4gl-vgpr-clause-tension.md)
reached the right verdict — MQ4-GL's 4.41% byte advantage does not convert to
wall clock — but **understated the deficit by roughly 4×**. Its numeric table
should not be cited; this record's should.

`gemv_hfq4g256` reaching **568 GB/s, 89% of peak**, is the substantive finding
behind that verdict. It is not a kernel with headroom left to concede; a
competing format needs a genuinely better decode path, not incremental tuning.

## What is unaffected

- **The lowering defect and its fix.** The original checkpoint's central result —
  a `?:` codebook chain lowering to divergent branches, 115 `s_cbranch_execz`
  reduced to 0, ~2.7–2.8× — rests on **static instruction counts from the
  emitted code object**, which are compile-time facts no timer can distort. The
  speedup ratio was host-timed, so its *magnitude* inherits this correction, but
  its existence and mechanism do not.
- **The VGPR/clause tradeoff mechanism.** `sc_ptrs` at 88 VGPR / 9 clauses and
  explicit scalar accumulators at 82 VGPR / 10 clauses versus a 121 VGPR / 15
  clause baseline are all manifest counters, not timings. The tradeoff is real.
  Which side of it wins was decided on host timings and is now open again.
- **All correctness evidence.** Parity, the codebook-order probe, edge cases and
  tail shapes are value comparisons, independent of timing.

## Methodological rules this establishes

1. **A host-timed single-kernel A/B on this stack cannot resolve differences
   below roughly 20 µs.** Use `profile::start`/`stop`; the wrappers already
   carry byte counts, so GB/s comes out without re-deriving traffic by hand.
2. **Never fit a floor from a sweep of small shapes.** Two earlier versions of
   the probe were wrong in ways worth recording: sweeping `m=1..128` left DPM at
   low clocks and fitted a 31.9 µs intercept which, subtracted, implied
   3.6 TB/s on a ~640 GB/s part; and an *ascending* sweep lets clocks ramp
   during the fit, biasing the intercept upward. Measure host-minus-device per
   shape instead of inferring an intercept.
3. **Report the instrument with the number.** Both originals present host
   medians as kernel time without qualification.

## Open

- **Per-R device timing.** r2/r4/r8 share a profiler kernel name, so the 412 GB/s
  figure is a blend. Which R wins device-side is unmeasured.
- **Whether the 27% deficit is addressable at all.** At 568 GB/s the uniform arm
  is near saturation; closing 27% means MQ4-GL must reach ~525 GB/s, well above
  anything measured for it.
- **The `memory-clause` profile's device-side effect.** It cuts r4 VGPR 120 → 109
  and lifts max-VMEM 10 → 16, and is now genuinely selectable (`58a1f1dce`), but
  has only ever been evaluated on host timings.
- All figures remain single-shape, single-arch, batch-1, synthetic-weight
  microbenchmarks. **No end-to-end model decode measurement exists for any arm.**
