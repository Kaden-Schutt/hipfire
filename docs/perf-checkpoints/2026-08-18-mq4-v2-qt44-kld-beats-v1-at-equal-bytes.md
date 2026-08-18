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

---

## Amendment — v2 against the Q8-protection ladder; the floor drops 16.5%

Same fixture, same host, same references, same day. Scoring parallelised across
hiptrx's four gfx1201 GPUs (both references concurrently per arm).

| arm | Q8 scope | size | qt=13 KLD | **qt=44 KLD** | Δ |
|---|---|---|---|---|---|
| `ctl` | head+embed | 15.663 GB | 0.043776 | **0.039033** | −10.83% |
| `ctl2` | +`ssm_out` | 16.464 GB | 0.036746 | **0.032495** | −11.57% |
| `attn` | +`ssm_out`+`attn` | 19.504 GB | — | **0.025437** | new arm |

**Prior best measured KLD on this model was 0.030479** (`ssm_in`, 18.614 GB, qt=13).
The `attn` arm reaches **0.025437 — 16.5% below that floor.**

v6 conversation selector, which the original ladder never scored for any rung
above `ctl` (its open gap 3):

| arm | v6sel KLD |
|---|---|
| `ctl` | 0.544517 |
| `ctl2` | 0.484145 |
| `attn` | **0.368886** |

### Composition is multiplicative to within a percent

Predicted from `ctl`'s ratio `r = 0.039033 / 0.043776 = 0.891656` applied to the
qt=13 ladder:

| arm | predicted | measured | error |
|---|---|---|---|
| `ctl2` | 0.032765 | 0.032495 | **0.8%** |
| `attn` | 0.024603 | 0.025437 | 3.4% |

So the header change and Q8 class protection are effectively independent levers —
unsurprising, since they act on disjoint tensor sets, but now measured rather than
assumed. A qt=44 arm's KLD can be estimated from its qt=13 twin before building it.

### A strict Pareto win

**qt=44 `ctl2` (0.032495 @ 16.464 GB) beats qt=13 `attn_full` (0.033862 @ 17.355
GB)** — better quality, 0.891 GB smaller, and one Q8 class cheaper. On the qt=13
ladder `attn_full` cost 31.70 tok/s against `ctl2`'s 33.20, so the qt=44 arm should
also be faster; unmeasured pending the GEMV port below.

### The ladder's top two rungs are NOT reproducible as documented

`HIPFIRE_Q8_CLASSES` accepts exactly `{lm_head, embed, router, ssm_out, attn}`
(`q8_class_of`, main.rs:7000). **`attn_full` and `ssm_in` are not valid values** —
they were labels in the earlier record, produced by quantize binary
`c644fc7f9272`. Passing either is silently ignored.

Proof: arms requested as `…,ssm_out,attn_full` and `…,ssm_out,ssm_in` both emitted
**md5 `a30ac345a3b0`, byte-identical to each other** at 16,464,182,272 B — i.e. the
plain `ctl2` scope, twice. A size assertion against the qt=13 ladder caught both and
refused to score them, so no false comparison entered this record. Without that
assertion the campaign would have logged two "new" arms that were the same artifact.

Note also that current `attn` spans full attention **and** `linear_attn`
(main.rs:7036), so it is a superset of the old `attn_full` and `ssm_in` rungs
combined — which is why the `attn` arm here is 19.504 GB, larger than either.

### Census of the new arm

```
q38.v2-attn.mq4   19,504,421,888 B   md5 2b0b4f2a4d3f
  attn(full)      {F16: 32,  Q8F16: 64}     full attention promoted
  linear_attn(dn) {F16: 144, Q8F16: 288}    all DeltaNet projections promoted
  mlp             {qt44: 192, F16: 192}     only MLP remains 4-bit
  lm_head/embed   {Q8F16: 1 each}
```

### Open: qt=44 cannot generate yet, only score

`hipfire bench` fails on every qt=44 arm at decode:

```
forward_scratch decode: qt=44 gemv_mq4g256v2: no GEMV_MQ4G256V2 plain source exists
```

`eval_hipfire --scoring-mode prefill` never touches the plain/multirow decode GEMV,
so the KLD numbers above are unaffected and valid. But autoregressive decode does,
and `gemv_hfq4g256.hip` (18 header sites) and `gemv_hfq4g256_multirow.hip` are not
among the 11 ported kernels. Until they are, **qt=44 is score-only and not
shippable**, and no shipped-path decode throughput number exists for it. The
refusal is a deliberate stub — it errors rather than decoding v2 bytes with a v1
kernel, which is the failure this whole record is about.

An earlier claim in this campaign that the plain/multirow GEMVs were "not on the
dense path" was drawn from a **cumulative** kernel cache whose entries predated the
run by two days. It is correct for prefill scoring and wrong for decode. Kernel-cache
evidence is only admissible after clearing the cache.

---

## Amendment 2 — shipped-path throughput measured; qt=44 generates

`hipfire bench --runs 5`, `max_tokens` 128, batch 1, hiptrx gfx1201, daemon
rebuilt at this commit. qt=13 rows from this record's ladder amendment 2.

| arm | size GB | KLD | decode tok/s | prefill tok/s | ttft ms |
|---|---|---|---|---|---|
| qt=13 `ctl` | 15.663 | 0.043776 | 34.60 | 401.3 | 59.8 |
| **qt=44 `ctl`** | 15.663 | **0.039033** | **33.40** | **420.40** | **57.10** |
| qt=13 `ctl2` | 16.464 | 0.036746 | 33.20 | 396.3 | 60.6 |
| **qt=44 `ctl2`** | 16.464 | **0.032495** | **32.00** | **414.20** | **57.90** |
| qt=13 `attn_full` | 17.355 | 0.033862 | 31.70 | 383.0 | 62.7 |
| qt=13 `ssm_in` | 18.614 | 0.030479 | 29.70 | 355.7 | 67.5 |
| **qt=44 `attn`** | 19.504 | **0.025437** | **28.20** | **357.70** | **67.10** |

### v2 costs decode and buys prefill

At byte-identical size, measured twice independently:

| | decode | prefill | ttft |
|---|---|---|---|
| `ctl` (15.663 GB) | **−3.47%** | **+4.76%** | **−4.52%** |
| `ctl2` (16.464 GB) | **−3.61%** | **+4.52%** | **−4.46%** |

The agreement across two sizes makes this a real effect, not noise. Decode at
batch 1 is bandwidth-bound but VALU-exposed enough that v2's extra `v_cndmask` +
two `v_cvt_f32_f16` per group per row cost ~3.5%. Prefill goes the other way:
the WMMA bodies load BOTH half headers unconditionally into `sc0/zp0/sc1/zp1`
and select per K-tile, which schedules better than v1's two `bit_cast`s.

**This corrects § 6 of [`docs/quant-formats/mq4-v2.md`](../quant-formats/mq4-v2.md).**
That section reported v2 as throughput-neutral on gfx1201 (ratio 0.9645 at R=2),
measured on `gemv_hfq4g256_multirow` — which the live-kernel trace established is
not on the dense projection path. The shipping path costs 3.5% decode. The spec
already flagged its own numbers as a proxy needing re-measurement; this is that
measurement.

### The Pareto win, now with measured decode

**qt=44 `ctl2` dominates qt=13 `attn_full` on all three axes:**

| | KLD | decode | size |
|---|---|---|---|
| qt=13 `attn_full` | 0.033862 | 31.70 | 17.355 GB |
| **qt=44 `ctl2`** | **0.032495** | **32.00** | **16.464 GB** |

Better quality, faster, smaller, and one Q8 protection class cheaper. Not a
tradeoff — strict dominance.

Against the old floor: qt=44 `attn` gives **−16.5% KLD for −5.1% decode and +4.8%
size** versus qt=13 `ssm_in`. Its 28.20 tok/s also lands within 0.5% of the 28.34
predicted by byte-scaling from `ssm_in`, so the bandwidth-bound model from the
original ladder still holds across the container change.

### qt=44 generates — eyeballed, not inferred

Decode required two more kernels (`gemv_mq4g256v2.hip`, 18 header sites, and
`gemv_mq4g256v2_multirow.hip`), since the daemon's autoregressive path uses the
plain and multirow GEMVs that prefill scoring never touches. With those ported,
`hipfire run` on the qt=44 `ctl` arm produced a correct vertical-scan
`longest_common_prefix`, accurate O(S) complexity analysis, and a valid
sort-based alternative — coherent prose and correct code, not token soup. Per this
project's standing rule, that eyeball is the correctness evidence for the decode
half-select; a wrong predicate silently applies the wrong scale to half of every
tensor and no throughput number would reveal it.

### Operational note — `hipfire bench` cannot be parallelised

KLD scoring parallelises across hiptrx's four GPUs (`eval_hipfire` takes no global
lock), and doing so cut the ladder sweep from ~26 min to ~6. `hipfire bench` does
NOT: the daemon enforces a single-instance guard and the second invocation dies
with `FATAL: hipfire daemon already running (PID …)` regardless of
`HIP_VISIBLE_DEVICES`. Bench arms must be run serially, killing the daemon between
them — and by PID or `/proc/*/comm`, never `pkill -f`, which self-matches the ssh
command and drops the connection.
