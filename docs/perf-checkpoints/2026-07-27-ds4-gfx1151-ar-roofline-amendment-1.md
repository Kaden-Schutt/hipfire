# Amendment 1 to `2026-07-27-ds4-gfx1151-ar-roofline`

**Amends (does not modify):**
[`2026-07-27-ds4-gfx1151-ar-roofline.md`](./2026-07-27-ds4-gfx1151-ar-roofline.md)

**Reason:** the original was written without reading the active campaign ledger
at `.codeinsight+research/ds4-gfx1151-campaign/ledger.jsonl`, which already
contained a same-day result falsifying its central recommendation.

**Date:** 2026-07-27

---

## Correction 1 — the fusion recommendation is REJECTED, not open

The original checkpoint concluded that the small-kernel bucket's 3.02 ms of
dispatch cost (1704 launches × 1.77 µs) was "the only path past 191 GB/s" and
that halving launch count was worth ~1.5 ms.

Ledger entry `2026-07-27-gate-e-swa-compressor-product-rejection` had already
tested exactly this:

| | control | bundle |
| --- | ---: | ---: |
| launches/step | 2320 | 2089 (**−231**) |
| full-prefix micro | — | **+1.477%** (494.98 GPU µs saved) |
| settled product (`auto`) | 28.0679 tok/s | **28.0795 tok/s (+0.041%)** |

The micro magnitude *agrees with this checkpoint's own floor measurement* —
231 dispatches × 1.77 µs = 409 µs against 495 µs measured — so the floor number
survives. **What does not survive is the inference that removing dispatches
raises shipping throughput.** Ledger diagnosis:

> launch-count reduction is not the shipping bottleneck for these copy kernels
> once the retained one-IB path is resident; future cooperative-tape candidates
> must remove retained-specific waits/acquires or command execution cost
> without equivalently accelerating HipGraph.

**Disposition: the fusion lever moves from "top kernel-side lever" to
Rejected/null** for launch-count-reduction framing. A fusion that also removes
retained-path waits/acquires is a different candidate and remains open.

## Correction 2 — GPU-µs is not throughput; transfer ratio ~3%

The original reasoned in GPU-µs throughout ("max kernel-side recovery 6.41 ms →
34.9 tok/s"). The Gate E pair shows 495 µs of *measured, bit-exact* GPU saving
producing 0.041% of product throughput. Any ms-to-tok/s arithmetic in the
original — including the 191 GB/s ceiling projection — is **Exploratory**, not
Measured, and should not be cited as a throughput forecast.

## Correction 3 — product baseline is 28.079 tok/s, not 27.70

Settled screening instrument, `auto` (retained PM4) arm: median **28.079 tok/s**,
spread 0.149%, `measurement_valid: true`, 10 runs/arm after 10 warmups. The
`hip` drift-control arm is 25.238. The 27.70 figure cited throughout the
original session is superseded. **Historical.**

The original's own in-process figures (28.31–28.50 via `--ar-ref`) sit near the
`auto` product number, which is reassuring for the instrument but does not make
them product claims.

## Correction 4 — `HIPFIRE_HC_CTRL_T1024` is unvalidated at product level

The original recorded +0.65% (35.31 → 35.08 ms) from interleaved `--ar-ref`
pairs, with rocprof confirming 0.942 → 0.764 ms/step on the kernel.

`--ar-ref` is an in-process reference and is structurally a **micro**, not the
settled screening product instrument. Given Gate E's 1.477% micro → 0.041%
product, a 0.23 ms micro gain cannot be asserted as throughput.

**Disposition: Measured (micro only). Not a promotion candidate.** It needs a
settled screening product run before any claim or default flip — in addition to
the `serve_harness.py` numerics sign-off already noted, since the kernel is not
bit-exact.

One reason it *might* transfer where Gate E did not: this is a kernel-duration
reduction (real command execution cost), not a launch-count reduction, and the
ledger's own implication names "command execution cost" as a category that
should still convert. Untested either way.

---

## What in the original still stands

- **Byte accounting, 5.48 GB/token.** Transport-independent; derived from the
  HFQ tensor table, closes to 0.09% of file size. Confirmed by the top-k 6→4
  measurement (predicted +11.2%, measured +12%).
- **Per-family GB/s.** Kernel durations are durations regardless of transport;
  both big GEMV families at 94–104% of the 206–210 GB/s ceiling still means they
  are finished.
- **The 1.77 µs dispatch floor itself** — Gate E's micro corroborates it.
- **All seven Rejected/null levers** (clocks, wave64, rocBLAS/hipBLAS, rocWMMA,
  rocPRIM/hipCUB, `mq_rotate_x` re-grid, `copyBuffer` d2d). Independent of this
  correction.
- **The `scoregrid` launch defect** (grid sized by `head_dim`, indexed as head).
- **The bytes conclusion**, which this amendment *strengthens*: if GPU-µs
  savings transfer at ~3%, then reducing bytes — which changes the work itself
  rather than its scheduling — is by elimination the remaining AR lever. The
  MFP3-dense projection stays Exploratory and unmeasured.

## Process note

The campaign ledger is the authoritative record of what has been tried on this
route. It was not consulted before the original checkpoint was written, which
cost a falsified recommendation. **Read
`.codeinsight+research/ds4-gfx1151-campaign/ledger.jsonl` before proposing any
ds4 gfx1151 lever**, and prefer its settled screening product instrument over
any in-process reference for throughput claims.
