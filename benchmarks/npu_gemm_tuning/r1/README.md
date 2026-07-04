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

### Aggregate — MEASURED: 8-column feed saturates at ~55 GB/s (the NoC knee)

`r1b_cols_run.py` / `r1b_cols_trace_run.py` run COLS single-column feeds
concurrently, each pinned to its own column (`Tile(col=i, row=2)` — auto-placement
otherwise stacks them on column 0 sharing one shim) and traced per-column.
Aggregate = total bytes / global concurrent span (distinct per-column regions):

| COLS | AGG GB/s | per-col | MEAN_BUSY | vs 1-col linear |
|---|---|---|---|---|
| 1 | 13.4 | 13.4 | 0.93 | — |
| 2 | 25.8 | 12.9 | 0.90 | 1.9× |
| 4 | 44–45 | 11.0–11.3 | 0.77–0.80 | ~3.3× |
| 8 | 54–56 | 7.0 | 0.47–0.49 | ~4.1× |

The aggregate **saturates at ~55 GB/s**: 1→2 is near-linear, then the per-column
rate falls (13.4→7.0) and the receive DMA busy fraction collapses (0.93→0.49) —
the shims spend half their time starved. That is the shared LPDDR5X/NoC/mem-
controller knee predicted by docs/192-193 (aggregate ≠ COLS×).

**Go/no-go for W4A8 prefill** (feed needed = `56e12 / (2·M)` B/s, per the table
up top, vs the ~56 GB/s ceiling):

- **M ≥ ~512 → compute-bound** (the good case): M=1024 needs 27 GB/s (met by ~3
  columns), M=4096 needs 6.8 (one column). W4A8 prefill runs at the compute
  ceiling here.
- **M ≲ 500 → feed-bound**: M=256 needs 109 GB/s, above the 56 ceiling.

So the crossover is **M ≈ 500**. For realistic prefill batch sizes (M ≥ 512) the
feed is not the limiter — the earlier "is the feed the wall?" question resolves
**no** for prefill. Only small-batch/decode-shaped work stays feed-bound.

**Distinct-region firm-up (caveat resolved).** The table above shares one input
BO (all columns read the same DDR region), which could bias the ceiling via
locality. Re-run with `DISTINCT=1` — one big BO of `COLS×PER`, each column reading
its own offset slice via a `simple_tiler` tap (a flat `[PER],[1]` tap illegally
lowers to a per-element `repeat_count`; the tiler gives the linear `[1,1,1,PER]`
BD) — lands at **54–56 GB/s at 8 columns** (busy 0.47–0.49), within run-to-run
jitter (~4%) of the shared number. So locality gave no material bias: the ~55 GB/s
ceiling is real for distinct per-column regions, i.e. real weight feed.

Notes: XRT's ~5 inout-buffer limit forces the single-BO approach either way
(2×COLS separate BOs segfaults at COLS≥3). N_TILES ≤ 255 with `DISTINCT` (the tile
loop is a DMA `repeat_count`, capped at 255). COLS=8 + 8 trace flows overruns the
router, so trace ≤4 columns (`TRACE_COLS`) while all 8 feed — traced columns feel
the same contention.

### What the ~55 GB/s ceiling actually is: the NPU fabric link (not the DRAM)

~55 GB/s is suspiciously low for a ~120-TOPS (W4A8) engine on a 256 GB/s LPDDR5X
system, so we chased the mechanism. It is **the NPU's link into the SoC data
fabric**, upstream of the memory controllers — not a per-controller/channel limit
and not our dataflow. Evidence, all pointing the same way:

- **Address-stride invariant** (`STRIDE` in `r1b_cols_trace_run.py`): spreading the
  8 columns' regions 256 MB apart across a 2 GB range — forcing distinct memory
  controllers for any interleave — gives 55.3 / 55.2 / 55.1 GB/s vs. adjacent.
  Flat. A one-controller bottleneck would scale when spread; it doesn't.
- **Stream-density invariant** (`r1b_streams_run.py`, ROWS = streams/shim): one
  interface tile drives its full 128-bit NoC input (1 col × 2 streams = 28.8 GB/s
  = 128 b × 1.8 GHz), but the *aggregate* is hit by ~2 tiles (2 col × 2 rows =
  53.6) and adding tiles/streams doesn't exceed ~55 (8×2 = 46.6, worse).
- **Depth/burst invariant**: FIFO depth 4→32 and tile 4 K→16 K both flat at ~56.
- **The memory system is not the limit**: a CPU memcpy spreads ~26 M read-beats
  evenly across all 12 `amd_df/..._read_data_beats_dram_N` channels — the full
  bandwidth is there; the NPU just has a narrow on-ramp.

So ~55 GB/s is a hard NPU-side ceiling, placement- and parallelism-invariant. It's
a deliberate design point: XDNA2 is a *reuse* machine (weights/activations live in
the 8×512 KB memtiles), so its DRAM link is provisioned for CNN / LLM-prefill
(compute-bound, on-chip-fed), not for decode (M=1, pure weight streaming). Caveat:
this box's DF PMU does not cleanly attribute XDNA2 DRAM traffic (`upstream_io` /
`cfi` counters read ~0 during the feed), so the conclusion rests on the behavioral
invariances above, not a direct counter or datasheet figure.

### Next

1. Add the W4A8 `mac_4x16_16x16` compute at M ≥ 512 and confirm sustained TOPS
   sits at the compute ceiling (feed proven sufficient there).
2. (Optional) trace the shim MM2S directly (`shimtile_events`) for the DDR-read
   view; the core-receive seal already bounds the end-to-end feed.

### Three-way status (what worked, what the toolchain blocked)

- **M1 host single-shot** (r1b_run.py): dominated by the 16 ms fixed cost —
  reproduces R1a's mistake. Kept as the baseline that exposes the overhead.
- **M2 on-device core timer**: NOT viable here — `aie::tile::current().cycles()`
  fails to link (undefined `::get_cycles()`); `event0/event1` markers also did not
  surface as INSTR events in the trace. Host-side 194 fencing is unreachable too:
  IRON's concrete `run()` bundles BO sync + execute. **The differential slope
  (sweep_r1b.py) is the validated host-side stand-in.**
- **M3 core-DMA `PORT_RUNNING` trace** (r1b_trace_run.py + r1b_cols_trace_run.py):
  **SEALED** — 14.4 GB/s single-column active (busy 0.91), ~56 GB/s 8-column
  aggregate, host-sync-free. The decisive on-NPU numbers.

### Toolchain notes (current, drifted from R1a's pin)

- Active venv is mlir-aie `2026-05 +886d932`, which **removed `aie.iron.placers`**;
  `Program.resolve_program()` now takes no placer (auto `--aie-place-tiles` pass).
  R1a/R0b's committed `SequentialPlacer()` import no longer imports here — R1b
  drops it. (Older pinned-March env with placers not present on the box now.)
- Env bring-up: `aie` package is under `<venv>/.../mlir_aie/python` (namespace
  pkg — `__file__` is None, resolve via `__path__`); pyxrt ships with XRT under
  `/opt/xilinx/xrt/python`. Both go on `PYTHONPATH` (see `run_r1b.sh`).
