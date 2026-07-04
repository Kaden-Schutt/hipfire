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

## Next (R1b)

Measure the *aggregate* multi-column feed ceiling (8 workers, 8 fifos), then add
the W4A8 `mac_4x16_16x16` compute and sweep M to find where sustained TOPS
crosses from feed-bound to compute-bound. That crossover (and whether the
aggregate feed clears ~7–27 GB/s at usable M) is the actual go/no-go.

Harness note: `@jit` + IRON `Tensor`s (from R0b) works; repeated runs in one
process segfault pyxrt (py3.14) so each measurement uses a fresh process.
