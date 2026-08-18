# Amendment: the fp16 header at model scale — prefill win survives, decode win inverts

- **Date:** 2026-08-18
- **Lifecycle:** `historical`
- **Amends:**
  [`2026-08-18-amendment-fp16-header-and-planar-microbench-matrix.md`](2026-08-18-amendment-fp16-header-and-planar-microbench-matrix.md)
  § 3, which reported the fp16 header as "a free ~5% on both ends" from
  isolated benches. Prefill holds. **Decode does not — it reverses sign.**
- **Host:** hiptrx, gfx1201, HIP 7.14
- **Commit:** `b815dd97a`

## 1 · The measurement

qt=45 was rebuilt in the "pad" layout — v1's exact geometry with the header
re-encoded: `[0..4)` fp16 header, `[4..8)` zero padding, `[8..136)` 128 B
nibbles. Same file size as qt=13. `hipfire bench --runs 5 --json`, dense
Qwen3.8-27B `ctl`, serial, with the v1 control run **before and after** in one
session to bracket thermal drift.

|run|decode tok/s|prefill tok/s|ttft ms|
|---|---|---|---|
|v1 before|34.70|404.70|59.30|
|**qt=45 pad**|**33.90**|**415.40**|**57.80**|
|v1 after|34.60|401.30|59.80|

Versus the v1 mean (34.65 / 403.0): **prefill +3.1%, decode −2.2%, ttft −2.9%**.

KLD unchanged from every other qt=45 layout — WT2 0.043423, v6sel 0.587705 —
confirming again that these layout moves are pure relocations.

## 2 · What the microbenches got right and wrong

|axis|isolated bench|model scale|verdict|
|---|---|---|---|
|prefill|−3.0% (faster than v1)|**+3.1% faster**|direction AND magnitude survived|
|decode|−5.0% (faster than v1)|**−2.2% slower**|**sign inverted**|

This is the sharpest result of the campaign on methodology. A residual-GEMM
microbench predicted model prefill well. A residual-GEMV microbench predicted
model decode *backwards*. Decode at model scale runs many more distinct kernels
than the residual GEMV, and it is bandwidth-bound near peak, so a single
kernel's schedule improvement does not compose.

**Prefill microbenches are usable as directional evidence. Decode microbenches
are not.** Any decode claim needs `hipfire bench` with the control bracketed.

## 3 · The fp16 header has a consistent signature: buy prefill, sell decode

Three formats now measured at model scale, all 136 B/group:

|format|decode|prefill|KLD (WT2)|
|---|---|---|---|
|qt=13 v1 (two f32 headers)|34.65|403.0|0.043776|
|qt=44 v2 (two fp16 pairs, per-128)|33.40 (−3.6%)|**420.4 (+4.3%)**|**0.039033 (−10.8%)**|
|qt=45 pad (one fp16 pair, per-256)|33.90 (−2.2%)|415.4 (+3.1%)|0.043423 (−0.8%)|

Both fp16-header formats trade decode for prefill, in the same direction and
roughly proportional magnitude. That is the header's real signature — not the
"free win on both ends" the isolated benches suggested.

## 4 · Consequence: qt=45 has no defensible niche

**qt=44 dominates qt=45 pad**: better KLD by 10 points and better prefill,
giving up 1.5% decode. Both are the same size. qt=45 pad is qt=44 with the
quality win deleted — the difference is that qt=44 spends its 8 header bytes on
per-128 granularity and earns −10.8% KLD, while qt=45 spends 4 and pads the
rest.

The full qt=45 design space, measured:

|layout|size|decode|prefill|niche|
|---|---|---|---|---|
|interleaved 132|97.06%|34.50|388.8|none — worst prefill|
|planar 132|97.06%|34.20|394.0|2.43% size for 1-2% throughput|
|pad 136|100%|33.90|415.4|dominated by qt=44|

The one thing qt=45 pad still does that qt=44 cannot: it is reachable from an
**already-distributed `.mq4` by a pure header rewrite**, with no parent
checkpoint and no requantization, since the nibbles are untouched and the size
is identical. Whether +3.1% prefill for −2.2% decode justifies shipping a
format for that path is a product call, not a measurement one.

## 5 · Two defects worth recording

**The load path hardcoded the stride.** `qwen35.rs` computed
`expected = m * gpr * 132` from a literal, so the first pad artifact failed to
load with `blob length mismatch: expected 27033600, got 27852800` — the same
204800 groups at two strides. The comment above it had explicitly argued for
hardcoding the number rather than sharing the constant, which is exactly what
broke. Now derived from `rdna_compute::MQ4C_GROUP_BYTES`.

**A conversion agent broke the oracle's control arm.** It replaced the shared
`gfx12_weight_cache_policy.inc` include in `mq4c_parity` with two hand-rolled
defines so its own arm would compile standalone. That compiled out v1's
`weight_rsrc` declaration while leaving its uses, so the **control** stopped
building. An oracle whose control cannot build measures nothing. Test
infrastructure shared between an arm and its control must stay shared.

Both were caught by a loud failure rather than a silent one, which is the only
reason each cost one cycle instead of a debugging session.
