---
title: gfx1201 indexed MoE gate-up buffer RT clears spills with scalar bases and staged lifetimes
date: 2026-07-11
tags: [gfx1201,moe,gate-up,buffer-load,redline,performance]
---

The active A3B decode kernel is
`gemv_hfq4g256_moe_gate_up_indexed.hip`, not the similarly named inactive
`_k8_indexed.hip` R40/LDS source. The active control is 96 VGPR / 20 SGPR /
zero scratch.

Two row-local SRDs spilled 12 B/lane and the first expert-wide SRD spilled
16 B/lane. Descriptor count was therefore not the root cause. Giving every
load a distinct scalar `soffset` was worse (28 B). The winning lowering uses
one expert-wide SRD, two shared scalar row/group bases, lane displacement in
`voffset`, and consumes all four gate headers before loading the up headers.
It compiles at 95 VGPR / 16 SGPR / zero scratch. Default/global gfx1201 and
gfx1100 controls remain byte-identical; `HIPFIRE_GFX12_WEIGHT_LOAD_POLICY=global`
is the rollback.

On hiptrx R9700 at automatic clocks, the corrected real-grid benchmark
(fixed to allocate `K_TOP * MI` outputs) produced identical global/buffer FNV64
`80003edb5bbe030b`. Its cold 256-expert arm improved from 18.8 to 18.3 us
(about 2.7%); the hot eight-expert arm was neutral. Retained-PM4 FWHT3 8K
process medians averaged 175.755 global versus 176.234 buffer (+0.272%); Q8
tg128 was neutral (+0.045%). The sampled eight-turn AR session was behaviorally
identical with zero attractors and improved average decode from 166.95 to
168.18 tok/s (+0.734%), including 153.9 to 155.0 at 17.6K context. MTP/spec
decode was disabled and untouched throughout.
