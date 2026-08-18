# MQ4 v2.0 (qt=44) — 10.8% better KLD than qt=13 at byte-identical size

**Date:** 2026-08-18
**Lifecycle:** `historical` — evidence under the exact fixture and method below.
Not a current default, not an automatic baseline, not an admission decision.
Newest file != current baseline. See
[`README.md`](README.md) in this directory before citing.

## Result

Dense Qwen3.8-27B, gfx1201 (hiptrx), prefill scoring, 24,552 tokens, 24 chunks.

| reference | qt=13 (MQ4G256) | qt=44 (MQ4G256V2) | delta |
|---|---|---|---|
| WT2 prose tripwire | 0.043776 | **0.039033** | **−10.83%** |
| v6 conversation selector | 0.587566 | **0.544517** | **−7.33%** |

Same artifact size to the byte: **15,662,615,552** for both. qt=44's payload
nibbles are byte-identical to qt=13; only the 8 header bytes change meaning.
Data-free — no calibration, no imatrix requirement beyond what qt=13 already
used, no search.

WT2 mean NLL 1.847483, PPL 6.3438. Decode 151–162 tok/s, i.e. production speed,
not a research path.

## Fixture

- artifact: `/home/kaden/qcal/q38.mq4v2.mq4`, md5 prefix `dba291397f01`,
  15,662,615,552 B, census qt44 496 / Q8F16 50 / F16 801 (305 + 496 AWQ sidecars)
- recipe, byte-identical to the qt=13 baseline apart from `--format`:
  `--format mq4v2 --q8-router --imatrix Qwen3.8-27B-imatrix.gguf --awq-alpha 0.55`
  with `HIPFIRE_Q8_CLASSES=""`
- references: `qwen3.8-27b.ref_wt2.bin`, `qwen3.8-27b.ref_v6sel-814d8fd.bin`
- scoring: `eval_hipfire --max-chunks 24 --kv-mode q8 --kv-v q8 --scoring-mode prefill`,
  `HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_LLOYD_GFX12=1`
- commit `a6ccc922e`

## Format

Group = 256 weights, stride 136 B, 8-byte aligned, `K % 256 == 0`. Identical to
qt=13 except the header:

| offset | qt=13 | qt=44 |
|---|---|---|
| `[0..4)` | f32 scale, all 256 | fp16 scale h0 + fp16 zero h0 (weights 0–127) |
| `[4..8)` | f32 zero, all 256 | fp16 scale h1 + fp16 zero h1 (weights 128–255) |
| `[8..136)` | 128 B nibbles | byte-identical |

The container is HFQ4-v2; qt=44 is that container plus the same offline FWHT-256
(seeds 42/1042) that makes qt=13 out of HFQ4. An unrotated sibling is therefore
free for the qt=6 line, untested here.

## Three independent defects, all silent

qt=44 scored WT2 12.137559 (PPL ~1e6) on its first two attempts and 16.705139 /
17.104609 on the next two. Every one was a wiring defect, none was the format,
and not one produced an error until a guard was added:

1. **Kernel routing.** `DType::MQ4G256V2` was added to existing v1 match arms —
   16 sites (13 in `qwen35.rs`, 3 in the dispatch families) of the shape
   `DType::MQ4G256 | DType::MQ4G256V2 =>`. That compiles, loads, and decodes v2
   bytes with v1 kernels.
2. **FWHT never applied.** `llama.rs` gates rotation on dtype and qt=44 was in
   none of the lists, so it fell to the arm doing rmsnorm without the rotate.
   `is_batchable_la` additionally refused it from the batched WMMA prefill path.
3. **AWQ sidecars silently dropped.** `DType::supports_awq_sidecar()` omitted
   qt=44, so all 496 sidecars were ignored and the engine computed `(W·s)·x`
   instead of `(W·s)·(x/s) = W·x` — a per-channel scale error on every
   projection. The doc block above that predicate describes the identical May
   2026 regression, which cost ~5 hours and produced "fluent-but-nonsensical
   token soup"; the predicate had been centralised specifically so that adding a
   dtype is a one-line edit. It was still missed, because the audit that found
   (1) and (2) looked at kernel routing, not capability gates.

Additionally, `forward_slots.rs` — the multi-slot batched prefill path that
`--scoring-mode prefill` drives — hardcoded seven `*Hfq4G256` kernel keys and
never consulted the weight dtype. Its own header comment documented why that was
correct: *"MQ4G256 is byte-identical to HFQ4G256, only the input activations are
pre-rotated."* True for qt=13; qt=44 voids it. `qwen35.rs` hardcoded twelve more.
All nineteen now select through container-aware helpers.

## Why none of it was visible

A v1 kernel fed qt=44 bytes `bit_cast`s an fp16 (scale, zero) pair to f32 and
gets ~1e-14, so every weight collapses to numerically zero. It cannot fail: every
bit pattern is a valid finite f32, the nibbles are read correctly, and stride,
alignment, group count and `K % 256` are all identical. It runs at **full speed**
and returns noise. Measured NLL 13.847 against `ln(248320) = 12.422` — slightly
worse than uniform, exactly what zeroed projections give.

Byte count, dtype census, tensor count, tok/s, a clean build, and
`cargo test --no-run` are all blind to this. **Fast and wrong is the signature of
a correct encoder feeding the wrong decoder.**

## What made it findable

Two oracles, both committed:

- `crates/hipfire-runtime/examples/mq4v2_parity.rs` — host-vs-GPU decode oracle
  for the GEMV. Builds groups whose halves occupy **deliberately disjoint** ranges
  ([-1,1] vs [96,160]) and asserts the fixture is discriminating *before*
  asserting the result, so a kernel reading only half 0's header fails by
  construction. gfx1201: worst relative error **2.426e-7**, fixture separation
  1.009e0.
- `crates/hipfire-runtime/examples/mq4v2_gemm_parity.rs` — cross-check for the
  WMMA GEMMs. Encodes one weight set into **both** containers and runs each
  through its own kernel against its own exact dequant, avoiding a host model of
  fp16 conversion, tiling and accumulation order that would be likelier wrong
  than the kernel. Sweeps batch size to cover the `bt8`/`bt12` bodies scoring
  actually compiles.

The GEMM oracle also confirms the format's premise directly: every live v2 GEMM
is **~3.7× more accurate than v1** at equal bytes.

| kernel | v1 rel-rms | v2 rel-rms |
|---|---|---|
| residual, batch 1 / 8 / 12 / 16 / 32 | 7.9e-4 – 9.6e-4 | **2.65e-4 – 2.80e-4** |
| gate_up (2 outputs) | 9.839e-4 | **2.618e-4** |
| qkvza (4 outputs) | 9.839e-4 | **2.618e-4** |

A third guard now hard-errors when a qt=44 weight reaches a `*Hfq4G256*` kernel
key, naming the dtype and key. It fired on the first run after the prefill fix
and located the twelve remaining `qwen35.rs` sites immediately, instead of costing
another 26-minute measurement.

**Caveat on the GEMM oracle:** `build_weights` is deterministic and unseeded, so
the multi-output tests run identical weights in every slot. That validates header
decode but would not catch cross-weight contamination (e.g. a kernel reading
`a_z`'s header for `a_beta`). Worth seeding per-slot before relying on it for a
different bug class.

## Not claimed here

- No throughput comparison. Decode ran 151–162 tok/s, which is production speed,
  but the v1/v1.5/v2 throughput sweep in
  [`2026-08-17-mq4-v2-affine-2x128-fp16-header.md`](2026-08-17-mq4-v2-affine-2x128-fp16-header.md)
  was measured on `gemv_hfq4g256_multirow`, which is **not** on the dense MQ4
  projection path. A shipped-path throughput number still needs measuring on the
  `fused_*` kernels and `gemv_hfq4g256_residual`.
- gfx1030 regresses 14–18% on the v2 header (Infinity Cache makes that kernel
  VALU-bound rather than bandwidth-bound). Arch-gating is undecided.
- MoE is unported: one MoE site
  (`gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched`) has no v2 launcher, and
  the MoE kernel family was explicitly out of scope. Dense-only result.
- The unrotated HFQ4-v2 sibling (qt=6 line) is implied by the container split but
  untested.
- N=1 on each reference. No repeat runs.
