# Redline context-bucketed retained-tape close-out

**Date:** 2026-07-11

**Host:** `hiptrx`, Radeon AI PRO R9700 (`gfx1201`)

**Model:** `qwen3.6-35b-a3b.mq4r`, FWHT3 KV, ordinary AR

**Clock policy:** automatic; no clock or performance-level override

## Question

The retained AQL/PM4 recording uses the physical KV capacity for every flash-
attention tile grid so one immutable tape remains valid as context grows. At a
2K live context with a 32K cache, each of the ten FullAttention layers launches
256 tile rows although only 16 do work; the rest immediately return.

Would retaining a context-bounded grid recover measurable decode throughput?

## Experiment

The experimental implementation kept the recorded max-capacity tape and all
kernargs, partial-buffer strides, and reduction limits unchanged. Between
completed replays it patched only each FA tile dispatch's grid:

```text
live_tiles = ceil(seq_len / 128)
grid.y     = live_tiles
```

Both transports were implemented: AQL kernel-dispatch packet geometry and the
`DISPATCH_DIRECT` words in the existing PM4 indirect buffer. The PM4 queue,
AQL packet, IB address, command count, kernels, and kernargs remained stable.
`HIPFIRE_REPLAY_CONTEXT_BUCKETS=0` supplied the full-grid control.

The daemon log confirmed transitions from `1/256` at sequence length 50 through
successive 128-token boundaries. The sampled control and candidate produced
identical assistant bytes, token counts, think/answer split, finish reasons,
and recall results, with zero attractors or empty turns.

## Result

### Fixed-context retained PM4

Each row used 100 decode positions, three warmups, five measured repetitions,
automatic clocks, and the HipGraph arm reported in-frame. The 8K comparison was
bookended with two independent processes per arm.

| Context | Full-grid control | Live-grid candidate | Change |
| ---: | ---: | ---: | ---: |
| 2,048 | 185.161 tok/s | 183.825 tok/s | **−0.72%** |
| 8,192 | 176.116 tok/s | 176.108 tok/s | **−0.005%** |

At 2K the candidate removed 240 of 256 tile rows per FullAttention layer and
still regressed. Its in-frame HipGraph anchor was faster than the control
(160.783 versus 157.727 tok/s), so under-clocking does not explain that result.

At 8K the two control medians were 175.917 and 176.316 tok/s; the two candidate
medians were 176.414 and 175.803 tok/s. Their aggregate is exactly neutral and
the apparent single-process positive result was arm drift.

### Sampled eight-turn serve harness

The candidate averaged 168.2 tok/s versus 167.9 for the disabled-bucket
control (+0.18%), but the fixed-context bookend above shows that difference is
within process variance. All eight outputs were byte-identical and both arms
had zero attractors, zero empty responses, one shared length-cap turn, and the
same 2/3 recall on turns 7 and 8.

## Verdict

Empty flash-tile rows are not the long-context bottleneck: their early-return
cost is effectively hidden beside the ten real FA layers and the much hotter
MoE body. Removing up to 93.75% of the launched rows did not improve retained
replay.

The experimental mutable-geometry implementation was therefore reverted. The
product keeps the simpler immutable max-capacity tape, and ordinary-AR work
moves to suballocation-aware dependency boundaries. This result also means a
future 256-token FA tile must earn its win from the useful tile body and memory
traffic, not merely from halving empty grid rows.
