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

## 6 · CORRECTION (same day): the occupancy explanation is falsified

An earlier draft of this file attributed the decode penalty to an occupancy
cliff — the qt=45 decode GEMV at 97 VGPR / 12 waves against v1's 93 / 16, with
decode being bandwidth-bound so a 4-of-16 wave loss should cost real
memory-level parallelism. That mechanism was tested and is **wrong**.

Unpacking the header through a native `ext_vector_type(2)` of `_Float16`
instead of mask/shift removes the integer temp and brings the kernel to
**94 VGPR / 16 waves, zero spills** — v1's occupancy exactly. Model scale, v1
bracketed:

|build|VGPR/waves|isolated bench|model decode|model prefill|
|---|---|---|---|---|
|v1|93 / 16|15.0-15.6 us|34.65|403-407|
|qt=45 mask/shift|97 / 12|**14.5 us**|33.90|415.4|
|qt=45 packed-half|94 / 16|16.0 us|**34.00**|415.9|

Restoring occupancy moved model decode **33.90 -> 34.00**: one 0.1 tok/s
reporting quantum against a 0.75 tok/s deficit. The cliff explains essentially
none of the gap.

Two things follow.

**The real cause is inherent unpack ALU.** v1 loads two dwords that ARE the
scale and zero; qt=45 loads one dword and must take it apart. Prefill is
latency-bound at ~1/5 of peak bandwidth and profits from the cheaper load;
decode is bandwidth-bound near peak and simply pays the extra ALU. No register
trick recovers it, because nothing about the register allocation is the problem.

**Third consecutive inversion from the isolated decode microbench, and the
sharpest.** It called the packed-half change a 10% REGRESSION (14.5 -> 16.0 us,
slower than v1) while model scale measured +0.3%. It also reproduces this
campaign's earlier "the occupancy-12 build is the fastest" result exactly — a
finding that is now shown to be an artifact of measuring one kernel in
isolation. Two separate conclusions in this repo rested on that microbench and
both were wrong at model scale.

The packed-half unpack is KEPT: at model scale it is neutral-to-marginally
better, it is correct (oracle 6.966e-8), spill-free, and 16 waves is the more
robust operating point under concurrency. It is simply not a fix for the decode
deficit, and the deficit appears not to be fixable.

## 7 · Three mechanisms proposed for the decode deficit, three falsified

The ~2% model-scale decode gap between qt=45 and v1 was chased through three
independent hypotheses. Every one was implemented, measured at model scale with
the v1 control bracketed, and refuted.

|build|VGPR/waves|FMA form|isolated decode|**model decode**|
|---|---|---|---|---|
|v1|93 / 16|98 `v_fma_f32`|15.4 us|**34.65**|
|qt=45 mask/shift|97 / 12|112 `v_fma_mix_f32`|14.5 us|33.90|
|qt=45 packed-half|94 / **16**|112 mix|16.0 us|**34.00**|
|qt=45 + FMA barrier|102 / 12|**98 plain**, dual 68->70|14.9 us|33.90|

**(a) Occupancy cliff.** qt=45 sat at 97 VGPR / 12 waves against v1's 93 / 16.
Unpacking through a native `ext_vector_type(2)` of `_Float16` reaches 94 / 16 —
v1's occupancy exactly, spill-free. Model decode moved 33.90 -> 34.00, one 0.1
tok/s quantum on a 0.75 tok/s gap.

**(b) FMA form / dual-issue.** The decode histogram showed the backend folding
the f16->f32 widening into the dequant FMA as `v_fma_mix_f32` (VOP3, not
VOPD-eligible): 98 packable `v_fma_f32` became 112 that issue one at a time,
with 4 fewer dual-issue slots. Forcing `sc`/`zp` to materialise as opaque f32
(empty asm, `"+v"` constraint, zero instructions) restored v1's form exactly —
98 `v_fma_f32`, zero mix, dual-issue 68 -> 70. Model decode: **33.90.** No
change at all.

**(c) Earlier, in § 5 and prior files:** 16-byte load misalignment, a redundant
f16->f32->f16 round trip, and lost `dwordx2` merging. All three refuted.

**What this actually establishes.** Perturbing the residual GEMV in three
independent directions — register pressure, unpack form, FMA form — moved model
decode by at most 0.1 tok/s. That bounds this kernel's contribution to the gap
at essentially zero. The deficit is **not in the kernel we spent three
experiments on**; it is distributed across the remaining decode kernels, which
are all still on the mask/shift unpack: `gemv_mq4cg256`, `gemv_mq4cg256_multirow`,
`fused_qkv_mq4cg256`, `fused_qkvza_mq4cg256`, `fused_gate_up_mq4cg256`.

**Fourth consecutive decode-microbench inversion.** The isolated bench ranked
these builds 14.5 / 16.0 / 14.9 us; model scale ranked them 33.90 / 34.00 /
33.90. It got the ordering wrong every time, including preferring the build that
model scale liked least. The isolated decode bench in this repo has now
mispredicted four out of four times and should be treated as a mechanism probe
only — never as a verdict.

**Kept:** the packed-half unpack (94 VGPR / 16 waves), because it measured
marginally best at model scale, has the lowest register pressure, and 16 waves
is the more robust operating point under concurrency. The FMA barrier is
reverted: it cost 8 VGPRs and 4 waves for zero model-scale gain.

**Recommendation: stop here.** The whole remaining deficit is 2% of decode,
three experiments have each bought under 0.3%, and the format that matters
(qt=44) carries the same signature with a -10.8% KLD payoff that dwarfs it.
