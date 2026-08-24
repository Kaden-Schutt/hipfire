# Amendment: qt=45 planar does NOT beat v1 at model scale — it is a size lever

- **Date:** 2026-08-18
- **Lifecycle:** `historical`
- **Amends:** [`2026-08-18-qt45-planar-beats-v1-on-all-three-axes.md`](2026-08-18-qt45-planar-beats-v1-on-all-three-axes.md)
  — that file's title and § 3 headline are **WRONG at model scale** and are
  corrected here. Its KLD results, oracle results, and microbench numbers stand.
- **Host:** hiptrx, gfx1201, HIP 7.14
- **Commit:** `bd49c1b48`

## 1 · The correction

The amended file concluded planar "beats v1 on prefill, decode AND size
simultaneously." That was measured on the **isolated residual GEMM and residual
GEMV**. It does not hold for the full model.

`hipfire bench --runs 5 --json`, dense Qwen3.8-27B `ctl`, serial (the daemon
holds a single-instance lock), v1 run BEFORE and AFTER planar in one session to
bracket thermal drift:

|run|decode tok/s|prefill tok/s|ttft ms|
|---|---|---|---|
|v1 ctl (qt=13)|34.60|402.10|59.70|
|**qt=45 planar**|**34.20**|**394.00**|**60.90**|
|v1 ctl (repeat)|34.60|400.50|59.90|

v1 reproduces exactly — 34.60 both times, prefill spread 0.4% — so this is a
real difference, not drift. At model scale planar is **-1.2% decode** and
**-1.8% prefill** against v1.

For completeness, all three qt=45/qt=13 variants through the same harness:

|arm|decode|prefill|size GB|
|---|---|---|---|
|v1 (qt=13)|34.60|401.3 (mean)|15.663|
|qt=45 interleaved|34.50|388.80|15.282|
|qt=45 planar|34.20|394.00|15.282|

Planar's contribution is real but narrower than claimed: it recovers **+1.3%
prefill** over interleaved (388.8 -> 394.0) at a cost of **-0.9% decode**
(34.50 -> 34.20). Neither qt=45 layout beats v1 on throughput at model scale.

## 2 · Corrected disposition

qt=45 planar is a **size lever**, not a speed lever:

- quality: identical or slightly better than v1 (WT2 0.043423 vs 0.043776;
  v6sel 0.587705 vs 0.587566) — unchanged from the amended file, and confirmed
  to six decimals across the layout change
- size: **97.06%** of v1, -380 MB
- throughput: **~1-2% behind v1** at model scale

That is still worth shipping for a size-constrained deployment. It is not the
free win the amended file described.

## 3 · Why the microbench did not translate, again

This is the third instance in one session:

1. the mq4c residual GEMV measured 6.6-8.1% faster than v1 and produced no
   model-level decode change,
2. removing 40 redundant fp16 round-trips changed a benchmark by nothing
   because LLVM already folded them,
3. and now the residual GEMM measured 2-3% faster than v1 while full prefill
   came in 1.8% slower.

The mechanism is the same each time: the residual GEMM is one kernel among
many. Full prefill also runs qkv, qkvza, gate_up, attention, and the norms, and
the alignment argument that made the residual kernel faster does not
automatically apply to all of them. A single-kernel A/B bounds what a change
*can* do; it does not measure what it *does*.

**Rule for this project going forward: no format-level throughput claim without
a model-scale `hipfire bench` with the control bracketed in the same session.**
The isolated bench is for mechanism, not for the headline.

## 4 · Open follow-up

The residual GEMM got faster while total prefill got slower, so at least one
other converted mq4c kernel is now slower than its v1 twin. Candidates, in
order of prefill weight: `gemm_qkvza_mq4cg256_wmma_gfx12_bt`,
`gemm_gate_up_mq4cg256_wmma_gfx12_bt`, `gemm_qkv_mq4cg256_wmma.gfx12`, and the
two LDS-staging variants whose slab changed 264 -> 256 B. Extending
`bench_mq4c_slab_alignment` to cover the qkv/qkvza/gate_up shapes would locate
it; that harness currently benches only the residual shape.

## 5 · What still stands from the amended file

- KLD: 0.043423 / 0.587705, matched to six decimals across the relocation.
- Oracle: 6.966e-8 on gfx1201, with a fixture that asserts its own
  discriminating power first.
- Zero register spills in all 13 kernels; LDS 12416 -> 12288 B on the two
  staging GEMMs.
- Decode correctness verified by real autoregressive generation on the planar
  artifact (coherent Python, 32.4 tok/s warm), which is the only thing that
  exercises `gemv_mq4cg256` and `gemv_mq4cg256_multirow` — prefill scoring and
  the residual microbench both miss them.
