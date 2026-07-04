# R1 — sustained W4A8: is the L3→L1 feed the wall?

R0 established the compute ceiling (dense int8 ~56 TOPS, **W4A8 ~112 TOPS**, int4
weights = the 2× lever, II=1 confirmed on hardware). R1 asks the go/no-go: can the
weight **feed** keep that fed?

## The arithmetic

At ~112 TOPS = 56e12 int8×int4 MACs/s, each int4 weight (0.5 B) is reused across M
prefill rows, so the weight feed needed is `56e12 / (2·M) B/s`:

| M (prefill batch / reuse) | weight feed needed |
|---|---|
| 256 | ~109 GB/s |
| 1024 | ~27 GB/s |
| 4096 | ~6.8 GB/s |

So W4A8 goes compute-bound only if the feed reaches tens of GB/s at large M.

## R1a — naive single-column feed (`r1a_feed.cc` + `r1a_run.py`)

> **Superseded by R1b (below).** The ~0.9 GB/s here is a **fixed per-call overhead
> artifact**, not the feed: the byte totals were too small to clear a ~16 ms
> device-load/dispatch floor. The true byte-proportional rate is **~12 GB/s**.
> Kept for the record and because the *core-vs-DMA* invariances still hold.

A single worker streams int8 tiles from an L3 (host) tensor via one objectFIFO and
touches every byte; differential host timing over total bytes gives the feed rate.

Measured (min of fresh-process runs, NPU `performance`):
- **~0.9 GB/s**, and it is **DMA-bound not core-bound**: a minimal-touch kernel
  (reads 1 vector; DMA still moves the whole tile) gives the *same* 0.88 vs 0.93
  GB/s, and the rate is proportional to bytes and independent of tile size
  (4 KB vs 16 KB both ~0.94 GB/s → not per-transfer latency).
- **Corroborated** by the earlier two-pass dequant: 262144 int4 = 128 KB in
  129.8 µs = 0.99 GB/s.

## Caveat: this is a FLOOR, not the feed ceiling

0.9 GB/s is ~2% of one shim-DMA channel's theoretical (~40 GB/s at 32 B/cyc ×
1.267 GHz npuclk). This config is a single column, single objectFIFO/DMA channel,
default `rt.fill` from one host BO — the worst case. **Do not conclude W4A8 is
feed-dead from this number.** The real feed ceiling needs:
- **8 columns in parallel** (8 shim DMA channels) — the aggregate is the number
  that matters vs the table above.
- larger/burst descriptors and possibly weights staged in a faster region.

## R1b — the 0.9 GB/s was a fixed-overhead artifact; the feed byte-rate is ~12 GB/s

R1a timed the whole `@jit` call as a **single shot**. On the current toolchain
(measured 2026-07-05, halo aie2p, mlir-aie `+886d932`) that call carries **~16 ms
of FIXED per-call overhead** — device load + BO alloc + dispatch — independent of
bytes. At R1a's small totals the feed is far below 16 ms, so the single-shot rate
*is* the overhead, not the feed. R1a even says it used "differential host timing";
the ~0.9 GB/s is what you get when the fixed cost still dominates the byte term.

**Fix: fit `call_ms = fixed_ms + slope·bytes` across an N_TILES sweep** (the
differential slope cancels the fixed cost), with totals large enough (tens of MB)
that the feed clears the 16 ms floor. Driver: `sweep_r1b.py` (via `run_r1b.sh`),
one fresh process per point (pyxrt segfaults on repeat under py3.14), min-of-N.

Measured, single column, `feed_sum` touch, `TILE_N=4096`:

| bytes | 32 MB | 64 MB | 128 MB | fit |
|---|---|---|---|---|
| call_ms (depth 4) | 18.75 | 21.05 | 26.56 | **slope 12.8 GB/s**, fixed 16.0 ms, R²=0.998 |

So the byte-proportional feed cost is **~12 GB/s, not 0.9** — ~14× higher. Against
the W4A8 table above a **single** column already clears M=4096 (6.8 GB/s) with
margin; 8 columns clear M=1024 (27). **The feed is not the wall for prefill.**

### But ~12 GB/s is a lower bound, and it's mostly host BO sync, not the feed

The slope is nearly **DEPTH-INSENSITIVE** (11.2 / 12.0 / 12.8 GB/s at FIFO depth
1 / 8 / 4 — within noise, non-monotonic). FIFO depth is an on-NPU DMA knob; its
irrelevance means the byte cost lives mostly in the **host→device BO sync** (which
precedes the kernel and depth can't touch), not the on-NPU feed. The `@jit` run
does sync-then-feed in series, so the measured slope satisfies
`1/12.8 = 1/sync + 1/feed` ⇒ **the true on-NPU feed is ≥ 12 GB/s** and hidden
above the sync. Both readings point the same way: feed is fine for prefill.

### Three-way status (what worked, what the toolchain blocked)

- **M1 host single-shot** (r1b_run.py): dominated by the 16 ms fixed cost —
  reproduces R1a's mistake. Kept as the baseline that exposes the overhead.
- **M2 on-device core timer**: NOT viable here — `aie::tile::current().cycles()`
  fails to link (undefined `::get_cycles()`); aie2p kernels timestamp via
  `event0/event1` + the trace unit. Host-side 194 fencing is also unreachable:
  IRON's concrete `run()` bundles BO sync + execute. **So the differential slope
  (sweep_r1b.py) is the validated host-side stand-in for M2.**
- **M3 shim-DMA `PORT_RUNNING` trace** (r1b_trace_run.py, UNSEALED scaffold): the
  only way to isolate the feed's own busy-cycle bandwidth from the host sync.
  Elevated from "refinement" to the **required decisive step** by the
  depth-insensitive result. Needs an on-hw iteration loop to seal.

### Next

1. Seal M3 (shim MM2S `PORT_RUNNING`) to get the feed's true busy-cycle rate,
   separated from host BO sync.
2. Multi-column aggregate slope (the NoC/mem-controller **saturation knee**, per
   docs/192-193 — aggregate ≠ 8× linear), then add the W4A8 `mac_4x16_16x16`
   compute and sweep M for the feed/compute crossover — the go/no-go.

### Toolchain notes (current, drifted from R1a's pin)

- Active venv is mlir-aie `2026-05 +886d932`, which **removed `aie.iron.placers`**;
  `Program.resolve_program()` now takes no placer (auto `--aie-place-tiles` pass).
  R1a/R0b's committed `SequentialPlacer()` import no longer imports here — R1b
  drops it. (Older pinned-March env with placers not present on the box now.)
- Env bring-up: `aie` package is under `<venv>/.../mlir_aie/python` (namespace
  pkg — `__file__` is None, resolve via `__path__`); pyxrt ships with XRT under
  `/opt/xilinx/xrt/python`. Both go on `PYTHONPATH` (see `run_r1b.sh`).
