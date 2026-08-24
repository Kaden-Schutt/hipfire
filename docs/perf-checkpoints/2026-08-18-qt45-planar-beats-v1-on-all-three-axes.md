# qt=45 planar: faster than v1 on prefill AND decode, at 2.43% fewer bytes

- **Date:** 2026-08-18
- **Lifecycle:** `historical` — evidence under the exact fixture and method
  below. Not a current default, not an automatic baseline, not an admission
  decision.
- **Commit:** `bd49c1b48`
- **Hosts:** k9lin (microbench) and hiptrx (KLD), both gfx1201, HIP 7.14
- **Model:** dense Qwen3.8-27B, `ctl` Q8 rung (lm_head + embed protected)
- **Disposition:** planar is the shipping layout for qt=45. The interleaved
  form measured in
  [`2026-08-18-qwen38-27b-mq4c-qt45-kld-and-throughput.md`](2026-08-18-qwen38-27b-mq4c-qt45-kld-and-throughput.md)
  never shipped and should not be resurrected.

## 1 · The change

Per weight tensor with `m` rows and `gpr = K/256`:

    [ payload plane : m*gpr*128 ][ header plane : m*gpr*4 ]

Total is still `m*gpr*132`, so `MQ4C_GROUP_BYTES`, every on-disk length check,
and the load path are unchanged. The header is the same packed dword (low 16
fp16 scale, high 16 fp16 zero, governing all 256 weights), just relocated.

Interleaved put the payload at `132g+4` — 4-byte aligned for half the groups —
while every one of these kernels issues 96 `global_load_b128`. Planar puts it
at `(row*gpr+g)*128`, 16-byte aligned for every group.

## 2 · Quality: provably unchanged, and confirmed

Because only byte placement moved, the values are bit-identical and KLD must
carry over. It does, to six decimals on both references:

|arm|WT2 KLD|v6sel KLD|bytes|
|---|---|---|---|
|qt=13 v1 `ctl`|0.043776|0.587566|15,662,615,552|
|qt=45 interleaved (never shipped)|0.043423|0.587705|15,282,138,112|
|**qt=45 planar**|**0.043423**|**0.587705**|**15,282,138,112**|

Reference `8a21364051d844b97c122e2c895f56d8`, `n_ctx=2048`, `top_k=256`,
`n_chunk=24`, 24,552 tokens, prefill scoring, kv q8/q8. Recipe:
`HIPFIRE_Q8_CLASSES="" --format mq4c --q8-router --imatrix
Qwen3.8-27B-imatrix.gguf --awq-alpha 0.55`.

## 3 · Throughput: the point of the exercise

Residual GEMM at M=K=5120, min of 5 interleaved samples, device events, 64
warmups, 200 launches/sample. Decode GEMV at m=k=5120, min of 3 runs. Ratio vs
the qt=13 control; below 1.000 is faster than v1.

|axis|interleaved|**planar**|
|---|---|---|
|prefill bt8|1.073 .. 1.119|**0.967 .. 0.978**|
|prefill bt12|1.024 .. 1.062|**0.979 .. 0.989**|
|decode|0.921 .. 0.947|0.950 .. 0.957|
|size|97.06%|97.06%|

Planar gives back ~2 points of decode versus interleaved — the predicted cost
of reading two streams instead of one — and gains 11-14 points of prefill.
**It is the only layout measured that beats v1 on prefill, decode, and size
simultaneously.** End-to-end prefill scoring on the real 27B rose to 168 tok/s
from 164.

## 4 · Why the two axes wanted different things

Decode GEMV runs at 866-944 GB/s, near this part's peak: **bandwidth-bound**,
so the smallest container wins and interleaved-132 was fastest. Prefill WMMA
GEMM runs at 110-130 GB/s, far under peak: **latency-bound**, so bytes are
nearly free and alignment dominates. Planar is the layout that satisfies both,
because it shrinks the container AND aligns the payload.

## 5 · Implementation notes worth keeping

- The two LDS-staging GEMMs drop from a 264 B/row slab to 256 B, which deletes
  their ragged 8-byte tail copy (256 is a whole number of 16-byte vectors).
  LDS per workgroup 12416 -> 12288 B.
- The fused kernels concatenate several independent tensors by row range, so
  each needs its OWN header-plane base from its OWN row count and a LOCAL row
  index (`lrow = sr - qkv_m`, ...). One shared base decodes the wrong headers
  for every tensor but the first.
- The multirow r8 GEMV spilled 14 SGPRs when each header was addressed inline
  as `hplane + ((gp - A) >> 5)`; precomputing `row_hptrs[R]` restores zero
  spills. Every kernel is spill-free.
- Elsewhere that shift trick is worth keeping: `(row*gpr + g)*4` is exactly
  `(gp - A)/32`, so the header address needs no second pointer threaded
  through call sites.

## 6 · The oracle earned its keep — and its old fixture could not have

`crates/rdna-compute/examples/mq4c_parity.rs` previously built its fixture from
a Gaussian with sigma=0.011. That gives every group nearly the same scale
(~0.0044) and zero (~0), so **an off-by-one plane read would have passed
silently**. A fixture that cannot distinguish neighbouring headers cannot
detect a header read from the wrong place.

It now builds a deliberately discriminating fixture (scale range
1.0e-3 .. 2.4e-2, zero range -2.0 .. 2.0), **asserts the fixture is
discriminating before asserting the result**, and additionally checks that
payload bytes reinterpreted as a header decode far away from any true header.
A wrong-plane emulation produces rel-L2 4.1e3 and a NaN scale. On gfx1201 the
real run reports **6.966e-8** against its own exact dequant.

The bench correctness gate also behaved correctly: the moment the kernels went
planar while the harness still packed interleaved, the mq4c arm returned
NaN/inf and FAILED. A layout change that does not announce itself that way did
not actually happen.

## 7 · Fixtures

- artifact: `/home/kaden/qcal/q38.planar.mq4c`, 15,282,138,112 B
- refs: `qwen3.8-27b.ref_wt2.bin`, `qwen3.8-27b.ref_v6sel-814d8fd.bin`
- oracle: `cargo run --release -p rdna-compute --example mq4c_parity`
- prefill: `cargo run --release -p rdna-compute --example bench_mq4c_slab_alignment`
- decode: `cargo run --release -p rdna-compute --example bench_vgpr_cap_sweep`
- warm the kernel cache serially before any concurrent scoring; two
  `eval_hipfire` processes racing to publish the same module produce
  `post-publish pair invalid` and kill an arm.
