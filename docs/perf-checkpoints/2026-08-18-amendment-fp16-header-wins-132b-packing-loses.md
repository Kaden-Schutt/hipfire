# Amendment: the fp16 header is a throughput WIN; the 132 B packing is the loss

- **Date:** 2026-08-18
- **Lifecycle:** `historical`
- **Amends:** [`2026-08-18-amendment-mq4c-prefill-gap-is-tuning-debt.md`](2026-08-18-amendment-mq4c-prefill-gap-is-tuning-debt.md)
  § 4, which concluded the residual difference was "a memory-timing artifact of
  the row stride" and rated a planar split as "may not help either." That was
  under-specified; this file isolates the variable and gives it a number.
- **Host:** k9lin, gfx1201, HIP 7.14
- **Harness:** `crates/rdna-compute/examples/bench_mq4c_slab_alignment.rs`

## 1 · The question

qt=44 kept v1's 136 B stride, swapped f32 headers for fp16, and **gained**
prefill (+4.5%). qt=45 used an even cheaper header (4 B vs 8 B) at a 132 B
stride and **lost** on both ends. Two variables moved at once — header
semantics and geometry — so neither earlier measurement could attribute the
delta.

## 2 · The isolation

Arm C: mq4c's 4 B per-256 fp16 header laid out at **exactly v1's geometry** —
header at +0, 4 B pad at +4, 128 B payload at **+8**, 136 B stride. Identical
header values and identical nibbles to arm B; only the byte placement differs,
so arm B's host reference validates arm C unchanged.

**A first attempt at this arm was wrong and is worth recording.** It changed
only the stride (132 -> 136) and left the payload at +4. That is not an
isolation: `136g + 4` preserves mq4c's 4-mod-8 payload alignment, so it varied
geometry only partially and produced a null result (+6.4..9.0% at bt8, *worse*
at bt12) that nearly buried the finding.

Payload address sequences, which is the whole story:

|container|payload address|mod 16|
|---|---|---|
|v1|`136g + 8`|{8, 0} — always >= 8B aligned|
|mq4c|`132g + 4`|{4, 8, 12, 0} — 4B aligned half the time|
|**arm C**|`136g + 8`|{8, 0} — identical to v1|

Both kernels emit **96 `global_load_b128`**. Same count — but mq4c's wide
loads are genuinely unaligned half the time.

## 3 · Result

M = K = 5120, seeded weights encoded once into every container, device events,
64 warmups, 200 launches/sample, 5 interleaved samples, minimum as lead
statistic. Ratio vs v1; below 1.0 is faster than v1.

|batch|tile|mq4c (132 B)|**arm C (fp16 hdr, v1 geometry)**|
|---|---|---|---|
|8|bt8|1.1127|**0.9688**|
|12|bt8|1.0901|**0.9551**|
|16|bt8|1.0924|**0.9526**|
|32|bt8|1.0749|**0.9657**|
|8|bt12|1.0590|**0.9889**|
|12|bt12|1.0615|1.0018|
|16|bt12|1.0429|1.0015|
|32|bt12|1.0240|**0.9884**|

Correctness: arm C rel-l2 **2.634e-4**, bit-identical to arm B and 3.7x better
than v1's 9.613e-4. Not a fast-but-wrong decode.

## 4 · What this means

- **The fp16 header is strictly better than the f32 header on throughput too,
  not just on bytes.** At matched geometry it beats v1 by 3.1-4.7% at bt8 and
  ties-to-beats at bt12. This is the same direction qt=44 showed with its
  +4.5% prefill, now isolated from every other variable.
- **The 132 B packing costs 11-14 points at bt8** (+11.27% -> -3.12% is a
  14.4-point swing), which swamps the header gain and is the entire reason
  qt=45 measured slower. It is a geometry cost, not a container-semantics cost.
- The earlier "latency-bound, so fewer bytes buys nothing" observation stands
  and now has a mechanism: in a latency-bound kernel an unaligned wide load
  costs more than the 2.43% of traffic the smaller group saves.

## 5 · The design this implies

Arm C is **not** shippable — the 4 B pad returns the entire 2.43% size win. But
it points directly at a layout that keeps both:

**Planar split.** Store headers and payloads in separate planes rather than
interleaved:
- payload plane: 128 B/group stride — **perfectly 16-byte aligned always**,
  strictly better than v1's 136 B, and a power of two
- header plane: 4 B/group, contiguous, 1/33 of total traffic
- total: still 132 B/group, so the 2.43% size win is fully preserved

This should beat arm C, since 128 B is better aligned than 136 B. It requires
a container change, which is free right now: qt=45 is unreleased, and the
repack is a pure header rewrite that can emit either layout. KLD is provably
unchanged — the same values, relocated.

Before building it, note the cost: two streams instead of one. Decode GEMV
reads each row's weights exactly once with no reuse, so the header plane adds
~2 extra cache lines per row there. Prefill, which is where the 11-14 points
live, should not care. Bench decode and prefill separately.

## 6 · Reproduce

```
cargo run --release -p rdna-compute --example bench_mq4c_slab_alignment
```
Arms: v1 / mq4c / mq4cpad, bt8 and bt12, batches 8/12/16/32. Correctness gate
runs before timing; arm C is validated against arm B's host reference.
Diagnostic kernel: `kernels/src/gemm_mq4cpad_residual_wmma_gfx12_bt.hip`
(byte-identical to the mq4c `_bt` kernel apart from stride 136 and payload +8).
