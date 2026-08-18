# MQ4C / v1.5 (qt=45): same KLD as qt=13 at 2.43% fewer bytes

- **Date:** 2026-08-18
- **Lifecycle:** `historical` — evidence under the exact fixture and method
  below. Not a current default, not an automatic baseline, not an admission
  decision. Newest file != current baseline.
- **Commit:** `07ac623f2` (`fix(qwen35): route qt=45 through the arch dispatch`)
- **Host:** hiptrx, gfx1201 (Radeon AI PRO R9700, 34.2 GB), HIP 7.14
- **Model:** dense Qwen3.8-27B, `ctl` Q8 rung (lm_head + embed protected)
- **Disposition:** MQ4C ships as a size lever, not a speed lever. It buys
  380 MB at identical quality and decode parity, and costs ~2.9% prefill.

## 1 · What qt=45 is

132 B/group: `[0..2)` fp16 scale, `[2..4)` fp16 zero (shared by all 256
weights), `[4..132)` 128 B of nibbles. Against qt=13's 136 B (f32 scale +
f32 zero) that is 4.125 bpw vs 4.25 bpw. The nibble payload is byte-identical
to qt=13; only the header is re-encoded. Unlike qt=44 there is no half-select
and no `cndmask` — the header is per-256, not per-128.

## 2 · Quality — the headline

Reference `8a21364051d844b97c122e2c895f56d8`, `n_ctx=2048`, `top_k=256`,
`n_chunk=24`, 24,552 tokens, prefill scoring, kv q8/q8.

|arm|size B|WT2 KLD|v6sel KLD|
|---|---|---|---|
|qt=13 v1 `ctl`|15,662,615,552|0.043776|0.587566|
|**qt=45 repack**|**15,282,138,112**|**0.043776**|**0.587566**|
|qt=45 direct|15,282,138,112|0.043423|0.587705|
|qt=44 v2 `ctl`|15,662,615,552|0.039033|0.544517|

**The repack reproduces qt=13 to six decimals on BOTH references.** That is
the expected consequence of the repack being a pure header rewrite: an
earlier full-file pass measured 0 nibble flips across all 95,119,360 groups,
so the only delta is f32 -> fp16 header rounding, and that rounding is below
KLD resolution at this fixture.

The direct path (`--format mq4c` from the parent) is a wash against the
repack: -0.81% on WT2, +0.02% on v6sel. Both are fine; neither is a quality
argument for one over the other.

## 3 · Throughput — parity on decode, a real prefill cost

`hipfire bench --runs 5 --json`, serial (the daemon holds a single-instance
lock), daemon rebuilt at the tested commit, same thermal state, HIP_VISIBLE_DEVICES=0.

|arm|decode tok/s|prefill tok/s|ttft ms|size GB|
|---|---|---|---|---|
|qt=13 v1 `ctl`|34.60|400.60|59.90|15.663|
|qt=44 v2 `ctl`|33.30|417.20|57.50|15.663|
|qt=45 direct|34.50|388.80|61.70|15.282|
|qt=45 repack|34.50|389.80|61.60|15.282|

**Read the stdev correctly.** Every arm reports `stdev = 0.0` across all five
runs. That is the bench's 0.1 tok/s reporting granularity, not evidence of
stability. The 34.5-vs-34.60 decode gap is exactly one quantum wide and
should be read as parity, not as a 0.3% loss.

**The microbench win did not translate.** `bench_vgpr_cap_sweep` measured the
uncapped v1.5 residual GEMV at 14.54 us vs v1's 15.82 us — 6.6-8.1% faster,
past the ~2.9% floor implied by header traffic alone. At full-model scale
that becomes 34.5 vs 34.6: nothing. A single-kernel microbench win is not a
model-level win, and this is a clean example of the gap.

Prefill is genuinely ~2.9% slower than v1 (389 vs 401), with ttft tracking it
(+3%). qt=44 remains the prefill winner at +4.1% over v1.

## 4 · The defect this checkpoint had to clear first

qt=45 was fully implemented below the arch layer — 13 kernels, all 16
launchers, KernelKey variants, family dispatch, quantizer, passing decode
oracle — and `crates/hipfire-arch-qwen35/src/qwen35.rs` still carried **79
`DType::MQ4G256V2` dispatch sites and zero for `MQ4CG256`**. Its only mq4c
mentions were load-path.

Symptom: **8-12 tok/s against ~160 for the equivalent qt=44 arm**, with
*correct output*. Every prefill admission gate refused the batched path and
dispatch degraded to the one reachable mq4c kernel, per-row `gemv_mq4cg256`.
Only three mq4c modules ever compiled, not one of them a GEMM.

No guard could fire: the fallback kernel was the *correct* mq4c kernel, just
the slow one. The qt=45 cross-route guards watch for qt=45 bytes reaching a
`*Hfq4G256*`/`*Mq4G256V2*` key, which never happened.

The direct diagnostic is the kernel cache, and it is now part of the score
script: after a serial warm run, `ls .hipfire_kernels/gfx1201 | grep -c
'^gemm_.*mq4cg256'` must be 7. It was 0.

## 5 · Method notes worth keeping

- **Warm the kernel cache serially before concurrent scoring.** Two
  `eval_hipfire` processes racing to publish the same module produced
  `post-publish pair invalid for fused_silu_mul_mq_rotate_a...` and killed an
  arm. `eval_hipfire` takes no global lock, so concurrency is otherwise safe;
  the cache is the shared resource.
- **`cargo build -p X --example Y` does not build X's binaries.** `--example`
  restricts target selection, so `hipfire-quantize` silently stayed a day
  stale and failed the qt=0 fall-through guard with `use_mq4c` absent from
  its own diagnostic. Third time this class has bitten this campaign.
- Interactive `examples/run` charges JIT compilation to the first generation:
  16.1 tok/s cold vs 32.4 warm on the identical prompt. Never quote a cold
  number.

## 6 · Where the three formats actually sit

They are three distinct points, not a ranking:

- **qt=13 v1** — 4.25 bpw baseline.
- **qt=44 v2** — same bytes, **-10.8% KLD**, +4.1% prefill, -3.8% decode.
  The quality lever.
- **qt=45 mq4c** — **-2.43% bytes at identical quality**, decode parity,
  -2.9% prefill. The size lever.

qt=45 does not compete with qt=44 and does not supersede qt=13 on quality —
it moves the same quality into fewer bytes. For an already-distributed `.mq4`
the repack reaches that point with no parent checkpoint and no requantization.

## 7 · Fixtures

- target (repack): `/home/kaden/qcal/q38.ctl.mq4c`, 15,282,138,112 B
- target (direct): `/home/kaden/qcal/q38.mq4c.mq4c`, 15,282,138,112 B
- v1 twin: `/home/kaden/qcal/q38.ctl.mq4`, 15,662,615,552 B
- v2 twin: `/home/kaden/qcal/q38.mq4v2.mq4`, 15,662,615,552 B
- direct recipe: `HIPFIRE_Q8_CLASSES="" hipfire-quantize --input <parent>
  --output <out> --format mq4c --q8-router --imatrix Qwen3.8-27B-imatrix.gguf
  --awq-alpha 0.55`
- refs: `qwen3.8-27b.ref_wt2.bin`, `qwen3.8-27b.ref_v6sel-814d8fd.bin`
- oracle: `cargo run --release -p rdna-compute --example mq4c_parity` —
  8.751e-8 vs its own exact dequant; PASS on both k9lin and hiptrx.
- decoded output eyeballed on the direct artifact (standing rule: numbers
  never prove coherence) — coherent Python with correct type hints and
  docstring structure, no attractor or degeneracy.
