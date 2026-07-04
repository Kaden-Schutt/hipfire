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

### Depth-insensitive because bandwidth-bound (not because of host sync)

The slope is nearly **DEPTH-INSENSITIVE** (11.2 / 12.0 / 12.8 GB/s at FIFO depth
1 / 8 / 4 — within noise, non-monotonic). This was first read as "byte cost is
mostly host BO sync." **M3 (trace) below disproves that**: the depth-insensitivity
is because the receive DMA is already ~91% busy — **bandwidth-bound, so more
buffering can't help** — not because the feed is hidden under sync.

### M3 — SEALED: on-NPU feed is ~13 GB/s, bandwidth-bound (trace unit)

The differential slope still can't split feed from host BO sync (both scale with
bytes). The trace unit can: it timestamps on-NPU events, and host→device sync
precedes kernel start so it is **not in the trace window**. `r1b_trace_run.py`
traces the compute tile's S2MM ch0 (the feed-receive port) with
`PORT_RUNNING/STALLED/IDLE` and reports span (feed duration) + busy fraction.

Measured (single column, `TILE_N=4096`, no trace-buffer overflow), stable across
128 / 256 / 512-tile feeds:

| metric | value | meaning |
|---|---|---|
| **FEED_GBS (active cycles)** | **14.4 GB/s** | exactly 512 cyc/tile = 8 B/cyc @ 1.8 GHz — dead stable |
| FEED_GBS (span/wall) | ~13 GB/s | includes ~9% inter-tile idle |
| BUSY_FRAC | 0.89–0.92 | PORT_RUNNING / span |
| STALL | ~0.2% | negligible → not core-consume-limited |

So the ~12 GB/s host slope **was the real feed** (host BO sync is overlapped /
negligible in the byte term), and the single-column feed is a genuine on-NPU
**~13–14 GB/s, ~91% busy = bandwidth-bound**. That is why FIFO depth did nothing.
Against the W4A8 table: one column clears M=4096 (6.8 GB/s) with 2× margin; the
open question is only how far 8 columns aggregate before the NoC/mem-controller
knee.

### Three-way status (what worked, what the toolchain blocked)

- **M1 host single-shot** (r1b_run.py): dominated by the 16 ms fixed cost —
  reproduces R1a's mistake. Kept as the baseline that exposes the overhead.
- **M2 on-device core timer**: NOT viable here — `aie::tile::current().cycles()`
  fails to link (undefined `::get_cycles()`); `event0/event1` markers also did not
  surface as INSTR events in the trace. Host-side 194 fencing is unreachable too:
  IRON's concrete `run()` bundles BO sync + execute. **The differential slope
  (sweep_r1b.py) is the validated host-side stand-in.**
- **M3 core-DMA `PORT_RUNNING` trace** (r1b_trace_run.py): **SEALED** — 14.4 GB/s
  active, busy 0.91, host-sync-free. This is the decisive on-NPU number.

### Next

1. Multi-column aggregate: replicate the feed across 8 columns and trace the knee
   (NoC/mem-controller, per docs/192-193 — aggregate ≠ 8× linear). ~14 GB/s ×
   columns is the ceiling to chase against the W4A8 M-table.
2. Add the W4A8 `mac_4x16_16x16` compute and sweep M for the feed/compute
   crossover — the go/no-go.
3. (Optional) trace the shim MM2S directly (`shimtile_events`) for the DDR-read
   view; the core-receive seal already bounds the end-to-end feed at 14.4 GB/s.

### Toolchain notes (current, drifted from R1a's pin)

- Active venv is mlir-aie `2026-05 +886d932`, which **removed `aie.iron.placers`**;
  `Program.resolve_program()` now takes no placer (auto `--aie-place-tiles` pass).
  R1a/R0b's committed `SequentialPlacer()` import no longer imports here — R1b
  drops it. (Older pinned-March env with placers not present on the box now.)
- Env bring-up: `aie` package is under `<venv>/.../mlir_aie/python` (namespace
  pkg — `__file__` is None, resolve via `__path__`); pyxrt ships with XRT under
  `/opt/xilinx/xrt/python`. Both go on `PYTHONPATH` (see `run_r1b.sh`).
