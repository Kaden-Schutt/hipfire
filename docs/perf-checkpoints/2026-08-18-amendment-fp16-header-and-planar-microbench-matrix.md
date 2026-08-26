# Amendment: fp16 headers beat f32 on BOTH ends; planar wins all three axes

- **Date:** 2026-08-18
- **Lifecycle:** `historical`
- **Amends / extends:**
  [`2026-08-18-amendment-fp16-header-wins-132b-packing-loses.md`](2026-08-18-amendment-fp16-header-wins-132b-packing-loses.md),
  which isolated the prefill variable but left planar unmeasured and decode
  untested.
- **Host:** k9lin, gfx1201, HIP 7.14
- **Harnesses:** `bench_mq4c_slab_alignment` (prefill WMMA GEMM),
  `bench_vgpr_cap_sweep` (decode GEMV)

## 1 · The matrix

Ratios vs the v1 control; **below 1.000 is faster than v1**. Prefill from the
residual `_bt` GEMM at M=K=5120, batches 8/16/32, min of 5 interleaved samples.
Decode from the residual GEMV at m=k=5120, min per run, two clean runs shown
(a third was discarded — cold/contended clocks put every arm ~30% high).

|layout|B/grp|decode r2|decode r3|prefill bt8|prefill bt12|
|---|---|---|---|---|---|
|v1 (f32 hdr, 136 B)|136|1.000|1.000|1.000|1.000|
|mq4c as shipped (fp16 hdr, 132 B interleaved)|132|**0.921**|**0.947**|1.073 .. 1.119|1.030 .. 1.058|
|fp16 hdr at v1 geometry (136 B, payload +8)|136|0.944|0.962|**0.953 .. 0.970**|0.987 .. 0.999|
|**planar (128 B payload plane + 4 B header plane)**|**132**|0.948|0.965|0.966 .. 0.977|0.972 .. 0.998|

All arms pass their correctness gate. Every fp16-header GEMM arm returns
rel-l2 **2.63e-4** against an f64 host reference vs v1's **9.61e-4** — the fp16
header is also ~3.7x more accurate, because v1's f32 scale is converted to
`_Float16` in-kernel anyway and rounds twice.

## 2 · Why the two ends disagree

- **Decode GEMV is bandwidth-bound**, running at **866-944 GB/s**. Bytes
  dominate, so the smallest container wins: interleaved 132 B mq4c is fastest.
- **Prefill WMMA GEMM is latency-bound**, running at **110-130 GB/s**, far
  under peak. Bytes are nearly free; *alignment* dominates, so the 132 B
  interleaved layout loses badly (payload at `132g+4` is 4-byte aligned half
  the time while both kernels still issue 96 `global_load_b128`).

This is the whole explanation for the original puzzle — why qt=45 "streams
2.94% fewer bytes yet prefills slower," and why the 6.6-8.1% GEMV microbench
win never reached model scale.

## 3 · Two shippable conclusions

**(a) An in-place fp16-header upgrade to hfq4g256 is a free ~5% on both ends.**
The `fp16 hdr at v1 geometry` arm keeps v1's exact 136 B stride and payload
offset (+8) and rewrites only the header: two f32 -> one packed fp16 pair,
with 4 B left as padding. That means:
- **no size change** (still 136 B/group, same file length)
- **no requantization** — the scale/zero values are unchanged, only re-encoded
- **no nibble touch** — a full-file pass over the mq4c repack already proved 0
  nibble flips across 95,119,360 groups
- measured **-3.8 .. -5.5% decode** and **-3.0 .. -4.7% prefill (bt8)**

The known caveat carries over: some groups store a scale below the fp16 normal
floor (min observed 1.370e-06 vs floor 6.104e-05) and land in subnormals, where
max drift was 0.261 steps against a 0.5 flip boundary — a 1.9x margin, not the
45x a sampled probe suggested. Fine at 4 bits; re-verify before reusing at
sub-4-bit or finer grouping.

**(b) Planar is the only layout that beats v1 on all three axes.**
Payload plane at a 128 B stride (perfectly 16 B aligned, better than v1's 136)
followed by a contiguous 4 B/group header plane. Still 132 B/group.
- prefill **0.966 .. 0.977** (faster than v1, and 11-14 points better than
  interleaved mq4c)
- decode **0.948 .. 0.965** (faster than v1; ~2-3 points behind interleaved
  mq4c, the two-stream cost, as predicted)
- size **97.06%** of v1 — the full 2.43% win preserved

Planar costs interleaved-mq4c about 2 points of decode and buys back 11-14
points of prefill. That is the trade to take.

Adopting planar is free right now: qt=45 is unreleased, and the repack is a
pure header rewrite that can emit either layout. KLD is provably unchanged —
identical values, relocated — so the measured 0.043776 / 0.587566 carry over
without re-scoring.

## 4 · Artifacts

Diagnostic kernels (bench-only; not wired into dispatch):
- `kernels/src/gemm_mq4cpad_residual_wmma_gfx12_bt.hip`
- `kernels/src/gemm_mq4cplanar_residual_wmma_gfx12_bt.hip`
- `kernels/src/gemv_mq4cpad_residual.hip`
- `kernels/src/gemv_mq4cplanar_residual.hip`

Each is byte-identical to its mq4c parent apart from stride, payload offset,
and symbol names (verified by round-tripping the substitutions back to source).

Reproduce:
```
cargo run --release -p rdna-compute --example bench_mq4c_slab_alignment   # prefill
cargo run --release -p rdna-compute --example bench_vgpr_cap_sweep        # decode
```

## 5 · Method notes

- Decode ratios move 2-3 points between runs from thermal drift; always take
  minimums and discard a run where the v1 control itself is far off its
  own best (one run put every arm ~30% high).
- The planar GEMV derives the header address from the payload address
  (`hplane + ((gp - A) >> 5)`) instead of threading a second pointer through
  eight call sites, since `(row*gpr + g)*4` is exactly `(gp - A)/32`.
