---
title: Context-bucketed retained FA grids are neutral or regressive on gfx1201
date: 2026-07-11
tags: [redline,pm4,aql,flash-attention,context,gfx1201,negative-result]
---

On hiptrx R9700 at automatic clocks, a literal retained-grid experiment patched
FWHT3 flash-attention dispatch geometry from the 32K physical capacity to the
exact live `ceil(seq_len/128)` tile count between completed replays. Kernargs,
partial strides, reduction limits, queue, and PM4 IB shape remained fixed.

The path was bit-identical in the sampled eight-turn serve harness, but the
bookended 8K product result was 176.116 full-grid versus 176.108 tok/s live-grid
(−0.005%). At 2K, removing 240/256 tile rows per FA layer measured 185.161
versus 183.825 tok/s (−0.72%) despite a faster candidate HipGraph anchor. Empty
tile early exits are effectively free on this workload. The mutable AQL/PM4
geometry code was reverted; do not reopen without a different mechanism.
