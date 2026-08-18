# Amendment: the mq4c prefill gap is tuning debt, not a container cost

- **Date:** 2026-08-18
- **Lifecycle:** `historical`
- **Amends:** [`2026-08-18-qwen38-27b-mq4c-qt45-kld-and-throughput.md`](2026-08-18-qwen38-27b-mq4c-qt45-kld-and-throughput.md)
  (unchanged; this file corrects its framing of § 3)
- **Host:** k9lin, gfx1201, HIP 7.14
- **Harness:** `crates/rdna-compute/examples/bench_mq4c_slab_alignment.rs` (new)

## 1 · What the original checkpoint got wrong

§ 3 reported "prefill is genuinely ~2.9% slower than v1" as though it were a
property of the 132 B container. That framing is wrong, and the comparison it
implies is not like-for-like.

|family|kernel files|residual-GEMM variants|
|---|---|---|
|`hfq4g256` (qt=13)|169|**54**|
|`mq4g256v2` (qt=44)|13|3|
|`mq4cg256` (qt=45)|13|3|

v1's residual GEMM has been ground through 54 variants: `_k2`, `_k4`,
`_ksplit`, `_ksplit_det`, `_kt`, `_mw`, `_wmma2`, seven `muse_*` gfx1100
specialisations, four gfx942 MFMA revisions, and a nine-point gfx906 tile
sweep (`x8` … `x64`). qt=44 and qt=45 each have exactly one WMMA kernel, one
`_bt`, and one GEMV. The measurement was never "v1 container vs mq4c
container" — it was a mature autotuned family against a first-pass port.

## 2 · The gap reproduces at kernel level, and is larger than 2.9%

Isolated residual `_bt` GEMM, M = K = 5120, seeded logical weights encoded
ONCE into both containers so the arms differ only by container. Device events,
64 warmups, 200 launches/sample, 5 samples, arms interleaved, minimum is the
lead statistic.

|batch|bt8 mq4c vs v1|bt12 mq4c vs v1|
|---|---|---|
|8|+11.97%|+5.75%|
|12|+9.30%|+6.12%|
|16|+11.18%|+5.08%|
|32|+7.80%|+4.21%|

Correctness gate passed first: rel-l2 **2.6e-4 (mq4c)** vs **9.6e-4 (v1)**
against each container's own f64 host dequant. mq4c is ~3.7x MORE accurate,
matching the earlier qt=44 GEMM parity result. This is not a fast-but-wrong
decode.

## 3 · Three hypotheses, all falsified — do not re-run these

**(a) 16-byte load misalignment.** 132 % 16 == 4, so a 2-group slab (264 B)
starts 8-byte aligned on odd slabs while v1's 272 B slab is always 16-byte
aligned; the LDS staging loop issues `uint4v_t` (dwordx4) copies. Arithmetic
checked out: 5 of 10 slabs misaligned at K=5120 vs 0 of 10 for v1.

REFUTED because those kernels are not on the dispatch path. The compiled
module list from the scoring run contains only
`gemm_mq4cg256_residual_wmma_gfx12_bt{8,12}.hsaco` and the qkv/qkvza/gate_up
`_bt` siblings. The LDS-staged `[16][264]` variants never compiled. The `_bt`
kernels that DO run use 4-byte scalar loads (`gp+4+(kt+n)*8+kg*4`), which are
4-byte aligned in both containers. A planned 528 B-slab fix was cancelled
before it was written.

**(b) Redundant fp16 round trip.** The header decode reads
`(_Float16)__half2float(__ushort_as_half(u16))` — f16 -> f32 -> f16 — at 40
sites across 10 mq4c/v2 kernels, against v1's single `v_cvt_f16_f32`.

REFUTED. Replacing all 40 with `__builtin_bit_cast(_Float16, u16)` changed the
benchmark by nothing (bt8 +11.99% after vs +11.97% before; correctness digits
bit-identical). LLVM already folds the round trip — f16->f32 is lossless and
f32->f16 of an exactly-representable value is exact. The edit was reverted
rather than landed as a phantom fix.

**(c) Lost dwordx2 merging.** 132 % 8 == 4 makes the mq4c payload base
alternate 8-byte / 4-byte aligned, which should block the compiler from
merging adjacent 4-byte loads.

REFUTED at ISA level. gfx1201 device asm emits **96 `global_load_b128` in
BOTH** kernels; the only delta is v1's 3 `global_load_b64` where mq4c has 3
extra `global_load_b32`. Three instructions cannot produce 5-12%.

## 4 · What the ISA actually says

|metric|v1 `_bt8`|mq4c `_bt8`|
|---|---|---|
|total VALU|2467|**2461**|
|`v_cvt_f16_f32`|102|**96**|
|`global_load_b128`|96|96|
|dequant macro|identical|identical|

mq4c issues fewer instructions and fewer converts, loads equally wide, and
streams 2.94% fewer bytes — and is still slower. Achieved bandwidth is
110-130 GB/s, well under this part's peak, so both kernels are **latency-bound,
not bandwidth-bound**. That is also why the smaller footprint buys nothing
here, and why the 6.6-8.1% residual-GEMV microbench win did not survive to
model scale.

The remaining structural difference is the row stride itself: 20 x 132 =
**2640 B** vs 20 x 136 = **2720 B**. Sixteen lanes read sixteen consecutive
rows simultaneously (`sr = rs + ml`), so the lane-to-address stride IS the row
stride, and those two values distribute across memory channels differently.
That is a memory-timing artifact, not something visible in an instruction
count.

## 5 · Disposition

The prefill delta is **tuning debt on a first-pass kernel**, not a cost of the
132 B container. The honest statement of § 3 is: at equal tuning effort the
comparison is unknown; at current tuning effort mq4c gives identical quality
and decode parity for 2.43% fewer bytes and ~2.9% less prefill throughput.

Next lever, in order of cost:
1. `crates/radiowave` already sweeps `scheduler_profile` (Default, MaxIlp,
   IterativeIlp, MemoryClause, PipelineIlp) and `unroll` per translation unit,
   with append-only JSONL round accounting and per-arch `RecipeEvidence`. It
   has no recipes for the 13 mq4c or 13 v2 TUs. That is the cheapest search.
2. Port the variant axes that actually paid for v1 — `_k2`/`_k4` K-unroll and
   `_ksplit` — to the mq4c `_bt` residual, and bench with the harness below.
3. Only then consider a container change (a planar header/payload split gives
   a 128 B payload stride at identical total size), and only with the
   latency-bound finding above in mind: it may not help either.

Reproduce with:
`cargo run --release -p rdna-compute --example bench_mq4c_slab_alignment`
It gates correctness before timing and reports min/median/max + GB/s + ratio
vs v1 for both bt tiles across batches 8/12/16/32.
