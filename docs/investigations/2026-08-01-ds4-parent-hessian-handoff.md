# DeepSeek V4 Flash 0731 parent-Hessian handoff

Date: 2026-08-01 (America/Phoenix)

Branch: `ds4-cdna-test-fail`

Pre-checkpoint HEAD: `b15edf38d35843b7a9d31bb609214d6abb172d4b`

Host: `mi300x` (`gfx942`, ROCm `/opt/rocm/core-7.14`)

## Executive state

The Hipfire-native activation dumper and rocBLAS Hessian builder work, but the
only complete 554-tensor capture was driven by the quantized DeepSeek V4 Flash
0731 MQ2R P3 artifact. It was **not** driven by the original parent checkpoint.

The generated activations and Hessians are therefore rejected as input to the
parent-derived GPTQ procedure. Preserve them as quant-self-calibration and
collector-validation evidence, but do not promote, rename, or consume them as
parent Hessians.

No pre-quant KLD/PPL baseline was recorded. No GPTQ bake has consumed these
Hessians, and no GPTQ/Hessian/quantization process was active when this handoff
was written.

The next gate is not GPTQ. It is a correct, fail-closed Hipfire forward for the
original mixed-precision parent checkpoint, followed by saved parent logits and
measured MQ2L/MQ2R KLD against those logits.

## What this checkpoint contains

### Activation producer

`crates/hipfire-arch-deepseek4/src/forward.rs` adds an environment-gated P3
activation recorder at the actual DeepSeek forward projection boundaries.

- Environment: `HIPFIRE_DS4_DENSE_ACT_DIR`
- File contract: `[u32 rows][u32 K][rows * K * f32]`
- Logical tensor names match the 554-tensor P3 map.
- Shared inputs are downloaded once and fanned out to all consuming tensor
  names.
- Batched prefill records all active rows, including the eight grouped rows per
  token consumed by `wo_a`.
- Finalization patches row counts only after a successful run.

`crates/hipfire-arch-deepseek4/examples/deepseek4_prefill_bench.rs` exposes the
recorder as `--dump-dense-acts DIR` and refuses ambiguous benchmark settings:
one repetition, no warmup, one variant/batch/E8 arm, no prefix/AR reference,
and a positive token count.

### Hessian consumer

`crates/hip-bridge/examples/collect_e8_hessian_rocblas.rs` reads one or more
activation files, computes each 256-channel `X^T X` block with rocBLAS FP32
GEMM on gfx942, canonicalizes the independent rocBLAS triangles to exact
symmetry, validates finite entries and nonnegative diagonals, and writes the
`E8H1` `.hblk` contract consumed by `hipfire-quantize --hessian-dir`.

This utility is model-agnostic with respect to activation provenance. The
producer determines whether a resulting Hessian is a parent Hessian, a
quant-self Hessian, or invalid. The utility does not make that claim itself.

## Preserved rejected capture

Root:

`/mnt/scratch/quantization/deepseek-v4-flash-0731-native-hessian`

The directory name predates the provenance correction and is misleading.
"Native" here only meant that Hipfire produced F32 activation buffers and
rocBLAS produced the Gram matrices. It did not mean that the original parent
weights produced those activations.

| Item | Value |
|---|---:|
| Corpus tokens | 1,024 WikiText tokens |
| Corpus MD5 | `83b0205a304bf4e52172ecdb05f2e895` |
| Capture time | 22.1196 s under instrumentation |
| Source artifact | `deepseek-v4-flash-0731.mq2r` |
| Source SHA-256 | `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce` |
| Activation files | 554 |
| Activation rows | 875,520 |
| Activation bytes | 13,899,927,888 |
| Raw Hessian files | 554 |
| Raw Hessian bytes | 2,212,502,008 |
| Symmetric Hessian files | 554 |
| Symmetric Hessian bytes | 2,212,502,008 |
| Representative symmetric hash | `head.weight.hblk` = `abc5f736949b27528356d3cbfc6abe5ecca85ad49f380d16a46229f1dad4d53d` |

Subdirectories:

- `p3-wikitext-1024-acts`
- `p3-wikitext-1024-hblk`
- `p3-wikitext-1024-hblk-symmetric`
- four smaller `smoke-*` directories

The loader discovered the sibling MTP artifact during the capture load, but
the benchmark executed ordinary target-only batched prefill. No MTP or
speculative forward generated these activations.

The capture directory contains no provenance manifest, parent-logit baseline,
KLD result, or exact saved invocation. That absence is itself a provenance
failure and must not be repaired retrospectively by inference.

## Root cause of the rejected provenance

The work conflated two different meanings of "native":

1. Hipfire generated and stored activation buffers as F32.
2. The original parent checkpoint generated the activation distribution.

Only the first statement was true. The benchmark CLI accepted an HFQ/MQ2R
model path and loaded the quantized P3 artifact directly. Converting its
intermediate values to F32 does not turn it into the parent model.

The original checkpoint is present and fits on the MI300X, but the current
DeepSeek safetensors loader cannot execute its formats correctly. It indexes
the tensors, then uploads unrecognized FP8/I8 payloads as raw bytes without
pairing their `.scale` tensors or selecting matching kernels.

## Original parent checkpoint inventory

Path:

`/mnt/scratch/models/DeepSeek-V4-Flash-0731`

The checkpoint occupies approximately 156 GiB across 48 safetensors shards
and contains 72,317 tensors.

| Safetensors dtype | Tensors | Payload GiB | Meaning |
|---|---:|---:|---|
| `I8` | 35,328 | 138.000 | Two packed E2M1 FP4 values per byte for routed experts |
| `F8_E8M0` | 35,718 | 8.625 | UE8M0 block scales, primarily expert per-32 scales |
| `F8_E4M3` | 390 | 5.871 | Dense FP8 weights with 128 by 128 scaling |
| `BF16` | 445 | 2.763 | Embeddings, norms, head, and other parent tensors |
| `F32` | 433 | 0.141 | Sinks, biases, and other full-precision tensors |
| `I64` | 3 | 0.017 | Hash-routing tables |

Relevant configuration:

- `quant_method = fp8`
- `fmt = e4m3`
- `scale_fmt = ue8m0`
- dense `weight_block_size = [128, 128]`
- `expert_dtype = fp4`
- expert FP4 scale group = 32 along K
- `num_experts_per_tok = 6`
- 256 routed experts
- 43 target layers plus one MTP layer

## Proposed parent-calibration backend

Implement a DS4-owned `Ds4ParentBackend`; do not extend the generic MQ2R byte
heuristics.

Admission must require all of:

- `model_type = deepseek_v4`
- `quant_method = fp8`
- `expert_dtype = fp4`
- exact weight/scale dtype and shape contracts
- `gfx942`

Any missing scale, unexpected dtype/shape, or unsupported device fails the
load. There is no fallback to `Raw`, MQ2R, Qwen, or a generic gfx11/gfx12
path.

### Dense weights

Decode `E4M3 * UE8M0` dense weights to resident BF16 on the GPU. The stored
values are exactly representable in BF16 because BF16 has a wider exponent and
mantissa than E4M3 and the UE8M0 multiplier is a power of two. After the decode
oracle passes, release the original dense FP8 code/scale buffers.

The 5.871 GiB dense tier expands to approximately 11.742 GiB.

### Routed experts

Keep all expert E2M1 codes and UE8M0 scales compressed in HBM. Do not expand
all 256 experts. Route tokens first, decode only each selected expert matrix
into a reusable BF16 scratch allocation, execute it through the gfx942 BF16
MFMA path, and reuse the scratch for the next matrix.

Expected resident model footprint is approximately 162--166 GiB, leaving about
25 GiB for state, KV, activation, and decode scratch on a 192 GB MI300X. Do not
load MTP for parent KLD/Hessian collection.

### Parent activation semantics

Reproduce the bundled parent implementation's arithmetic rather than silently
running a higher-precision reinterpretation:

- Before every FP8/FP4 linear, apply dynamic E4M3 activation quantization with
  a per-128 UE8M0 power-of-two scale, then dequantize for the BF16 MFMA
  correctness path.
- Mirror the explicit FP4 simulation points in the indexer.
- Mirror the explicit FP8 simulation of non-RoPE KV dimensions.
- Preserve top-k 6 routing and all 256 parent experts.
- Keep DSpark and MTP disabled.

Reuse the existing DS4 attention, compressor, routing, Hyper-Connections, and
state control flow. Branch on a DS4 model-owned backend, not on a process-wide
architecture flag. No Qwen-owned body changes.

## Required gate order

1. **Inventory gate** — **PASSED 2026-08-02.** All 72,317 source tensors
   accounted for; every native weight has exactly one valid scale companion;
   MTP is explicitly excluded. See "Gate status" below.
2. **Codec gate** — **PASSED 2026-08-02.** GPU E4M3/UE8M0 and E2M1/UE8M0
   decode matches an independent CPU oracle on fixed edge cases and sampled
   checkpoint values, bit for bit. See "Gate status" below.
3. **Linear gate** — **PASSED 2026-08-02.** Dense and expert matmul outputs
   match the checkpoint's bundled operator semantics on fixed inputs and on
   real checkpoint tensors. See "Gate 3 evidence" below.
4. **One-layer gate** — **PASSED 2026-08-02.** 16-token layer canary, finite
   state, 14/14 sub-block checks against an f64 CPU oracle. See "Gate 4
   evidence" below.
5. **Parent-forward gate** — **PASSED 2026-08-02.** Full 43-layer forward at
   32 and 256 tokens, finite logits, deterministic in-process hash, coherent
   next-token. See "Gate 5 evidence" below.
6. **Pre-GPTQ quality gate**: save parent reference logits, then measure the
   existing MQ2L and MQ2R artifacts on the exact same token IDs, positions,
   tokenizer, RoPE convention, and engine fingerprint. Record KLD/PPL before
   any GPTQ mutation.
7. **Hessian canary**: capture 1,024 parent tokens and verify the 554-tensor
   map, row counts, finite/nonnegative Hessians, exact symmetry, and consumer
   compatibility.
8. **Calibration expansion**: accumulate diverse fixed 1K shards to 8K, 16K,
   and 32K tokens; stop when quant decisions and quality stabilize.
9. **GPTQ**: only after gates 1--8, apply `gptq.rs` to original parent weights
   and compare RTN versus GPTQ against the saved parent logits.

## Gate status (updated 2026-08-02)

Gates 1-5 are closed. The original parent checkpoint runs end to end on the
MI300X and produces coherent output. Gate 6 (parent logit baseline + MQ2L/MQ2R
KLD) is the next work.

### What landed

Commit `f8b98f0a2` (branch `ds4-cdna-test-fail`) adds
`crates/hipfire-arch-deepseek4/src/parent/`:

| module | role |
|---|---|
| `mod.rs` | `Ds4ParentBackend` admission: `model_type=deepseek_v4`, `quant_method=fp8`, `fmt=e4m3`, `scale_fmt=ue8m0`, `weight_block_size=[128,128]`, `expert_dtype=fp4`, exact gfx942. No env override, no portable fallback. |
| `inventory.rs` | Gate 1. Tensor accounting, scale pairing, dtype/shape contract, MTP exclusion. |
| `codec.rs` | Gate 2 CPU oracle. E4M3/UE8M0/E2M1 codecs, dense 128x128 and expert per-32 dequant, bit-exact `fast_log2_ceil`/`fast_pow2`/`fast_round_scale` activation-quant reference. |
| `manifest.rs` | The mandatory evidence manifest, with `validate()`. |
| `plog.rs` | Gate 6's parent-logit container and KLD/PPL comparator. |

Four new gfx942 kernels: `dequant_fp8_e4m3_ue8m0_blk128_to_bf16`,
`dequant_fp4_e2m1_ue8m0_g32_to_bf16`, `act_quant_fp8_ue8m0_inplace`
(block 128 at linears, 64 at the KV simulation sites), and
`act_quant_fp4_ue8m0_g32_inplace`.

Two executable gates:
`examples/ds4_parent_inventory_gate.rs`, `examples/ds4_parent_codec_gate.rs`.

### Gate 1 evidence

Run on `mi300x` (gfx942) against `/mnt/scratch/models/DeepSeek-V4-Flash-0731`:

- 72,317 tensors seen, `assert_complete(72317)` PASS, walk time 0.082 s.
- 35,718 scale pairings verified; **zero** orphan scales, zero unquantized
  tensors carrying a scale, zero non-expert `I8`, zero unknown dtypes.
- Main tower 67,612 tensors / 145.301 GiB; 4,705 MTP tensors excluded.
- Index SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`;
  config SHA-256 `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`;
  all 48 shard SHA-256s recorded in the emitted manifest.

**VRAM residency projection (main tower, weights only):**

| tier | treatment | GiB |
|---|---|---:|
| dense `F8_E4M3` | decoded to resident BF16 (2x stored) | 10.910 |
| routed experts | `I8` + `F8_E8M0` left compressed | 137.062 |
| `BF16` | as stored | 2.634 |
| `F32` | as stored | 0.132 |
| `I64` | as stored | 0.017 |
| **total** | | **150.756** |

Against a 192 GiB card that is **41.244 GiB of headroom**, so the parent
forward fits with MTP excluded. This is weights only — KV, activations, and
expert decode scratch come out of the headroom.

### Gate 2 evidence

`ds4_parent_codec_gate` on `mi300x` (gfx942): **13/13 PASS, exit 0.** Every
comparison is **bit-exact** against the CPU oracle, not tolerance-based.

- Dense FP8: exhaustive 256x256 (65,536 elements), ragged 260x300 (catches
  `floor` where `ceil` is required, in both dimensions), NaN propagation for
  scale byte `0xFF` and E4M3 `0x7F`/`0xFF`.
- Expert FP4: exhaustive 64x512 (32,768 elements), explicit nibble-order
  assertion.
- Activation quant: FP8 at block 128 and 64, FP4 at group 32, including
  power-of-two amax, just-above-power-of-two amax, values under the `1e-4`
  and `6*2^-126` floors, all-zero groups, single outliers, and exact RNE
  midpoints.
- Real checkpoint samples: `layers.3.attn.wq_a.weight` (`F8_E4M3 [1024,4096]`)
  decoded to min -0.117188 / max 0.117188 / mean -7e-6 / std 0.023066 /
  0.001 % exact zeros; `layers.3.ffn.experts.0.w1.weight`
  (`I8 [2048,2048]` logical `[2048,4096]`) to min -0.125 / max 0.125 /
  mean 2.4e-5 / std 0.025293 / 12.77 % exact zeros. Both trained-looking; the
  expert's zero fraction is expected given E2M1's zero codes.

### Findings worth carrying forward

1. **`__builtin_amdgcn_cvt_pk_fp8_f32` on gfx942 is FNUZ, not OCP.** Its max
   finite magnitude is 240 and its NaN encoding is `0x80`; the parent
   checkpoint uses OCP `float8_e4m3fn` with max 448. Using the hardware
   builtin would have silently saturated every activation above 240 — a
   quality bug no coherence check would catch. `act_quant_fp8_ue8m0_inplace`
   therefore implements OCP RNE in software, cross-checked against
   `__hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3)` over 101 vectors
   with zero mismatches.
2. **E2M1 nibble order is low-nibble-first**, confirmed decisively by the
   checkpoint's own packer at `inference/convert.py:30-33`
   (`stack([low, high], dim=-1).flatten`), and again on real bytes in Gate 2.
   Distributional evidence alone was *not* decisive here, because adjacent
   logical positions share a 32-wide scale group, so swapping nibbles never
   crosses a scale boundary.
3. **`inference/convert.py::cast_e2m1fn_to_e4m3fn` is not the decode path.**
   It is an opt-in FP4→FP8 re-packing utility selected by `main`'s
   `expert_dtype` argument. This checkpoint declares `expert_dtype = fp4`, and
   `model.py::linear()` consumes the FP4 weights directly through `fp4_gemm`
   with their per-32 E8M0 scales. Do not let the `MAX_OFFSET_BITS = 6`
   arithmetic in that function leak into the decoder.
4. **The bundled reference cannot be executed.** `mi300x` has no torch, numpy,
   safetensors, or tilelang. `parent::codec` is consequently the *only*
   numerical cross-check that exists, which is why it is tested exhaustively
   over all 256 E4M3 codes, all 256 UE8M0 bytes, and all 16 E2M1 codes rather
   than spot-checked.
5. **The parent checkpoint's tensor names already match hipfire's DS4 loader**
   (`layers.{l}.attn.wq_a.weight`, `embed.weight`, ...). That is not a
   coincidence: the on-disk checkpoint is post-`convert.py`, and `convert.py`'s
   rename table produces exactly those names. No name mapping layer is needed
   for Gate 3.
6. `engine.rocm_path` in the emitted manifest reads `/opt/rocm-7.0.2`, not
   `/opt/rocm/core-7.14`, because it reports what `hipfire_config::rocm::root()`
   resolves. The kernels were compiled with `/opt/rocm/core-7.14/bin/hipcc`.
   Both installs are present on the host; if the discrepancy matters for a
   published result, pin `HIPFIRE_ROCM_PATH` before the producing run.

### Gate 3 evidence

`ds4_parent_linear_gate` on `mi300x` (gfx942), seed `0xD54CA7E32026`:
**11 PASS, 1 INCONCLUSIVE, 0 FAIL**, exit 0.

The acceptance criterion is relative, not an invented tolerance. The CPU
oracle (`parent::gemm_ref`) runs in two modes: `Exact` (f64 ground truth) and
`ReferenceOrder` (f32, reproducing the tilelang block structure exactly). Then
`err_ref = ||ReferenceOrder - Exact|| / ||Exact||` is the bundled kernel's own
rounding, and the GPU must not be materially worse.

| signal | result | reading |
|---|---|---|
| bias: mean signed err / stddev | <= 0.08 all cases; **0.005** on both real tensors | no misplaced scale |
| `err_gpu / err_ref` | 0.64 - 2.56x (bar: 4x) | same summation-order class |
| `err_gpu / err_seq_f32` | 0.54 - 0.83x | MFMA is *tighter* than naive sequential f32 |

Bias is the defect detector and magnitude is not: a misplaced scale shows up
as bias at any magnitude, while a different-but-valid summation tree shows up
as unbiased noise. Both signals are clean, on synthetic cases with deliberately
wide UE8M0 exponent spread and on `layers.3.attn.wq_a.weight` and
`layers.3.ffn.experts.0.w1.weight` from the real checkpoint.

The one INCONCLUSIVE case is a tiny expert shape where `err_ref == 0` — every
term happened to be exactly representable, so no rounding occurred anywhere
and the comparison measures nothing. It is reported as INCONCLUSIVE rather
than PASS on purpose. **This is a real trap for later gates:** narrow scale
ranges make GPU and oracle agree bit-for-bit and produce a pass that is
evidence of nothing. Any future numeric gate must assert `err_ref > 0` before
counting a case.

### Full-checkpoint residency (`ds4_parent_residency_gate`, mi300x)

All 43 layers with routed experts, MTP excluded:

| tier | GiB | bytes |
|---|---:|---:|
| dense, decoded to resident BF16 | 10.910 | 11,714,691,072 |
| routed experts, compressed | 137.062 | 147,169,738,752 |
| BF16 | 2.634 | 2,828,377,344 |
| F32 | 0.132 | 141,262,684 |
| I64 | 0.017 | 18,616,320 |
| **total** | **150.756** | **161,872,686,172** |

Byte-identical to the Gate 1 projection, which was derived independently from
the safetensors index — two separate code paths agreeing exactly. 41.244 GiB
headroom on the 192 GiB card. Full load 40.7 s at 3.79 GiB/s; a
`ParentLoadPlan { layers: 0..1 }` loads in 3.4 s, which is what Gate 4 should
iterate against.

### Additional findings

7. **`gemm_bf16_mfma.gfx942.hip` had never been executed.** It was compile-
   and disassembly-validated on 2026-05-19 but its runtime validation was
   still marked PENDING and it had no Rust wrapper. It now has
   `Gpu::gemm_bf16_mfma_gfx942` and is bit-exact against both an F32 reference
   and rocBLAS on every shape tested. Throughput is 13.8 TFLOP/s vs rocBLAS
   40.9 at `n=32768, k=1024, m=32` — adequate for calibration (a 1K-token
   forward is order 13 TFLOP total), so correctness and capture-hook
   friendliness win over raw speed here.
8. **BF16 decode is exact, not an approximation.** Every UE8M0 scale is a
   power of two, and an E4M3 code (3-bit mantissa) or E2M1 code (1-bit
   mantissa) times a power of two is exactly representable in BF16 (7-bit
   mantissa, wide exponent). So each product term is the identical real number
   in both the reference's FP8xFP8 formulation and ours, and exact in FP32.
   Only summation order differs. This is why Gate 3 can demand near-parity
   rather than a loose tolerance.
9. **`parent_linear_dense` destroys `x_bf16`**, matching the reference's
   `inplace=True` activation quantization at the linear boundary. A caller
   that reuses the buffer for a second projection double-quantizes silently.
   The forward must fill a per-linear activation scratch from the residual;
   never hand it a buffer that has to survive.

### Gate 4 evidence

`ds4_parent_layer_gate` on `mi300x` (gfx942), layer 0, 16 rows seeded from
real `embed.weight` token rows: **PASS, 14/14 oracle checks**, exit 0.
Output finite (0 NaN, 0 Inf, 0 exactly-zero elements), L2 506.71,
110.9 ms, 27.5 MiB scratch, layer load 0.70 s.

Every stage with an f64 reference in `parent::layer_ref` was downloaded and
compared. Agreement is 3.7e-9 to 9.5e-7 max-abs across all HC and norm
stages — f32 round-off, nothing more:

| stage | max abs | mean rel |
|---|---:|---:|
| `hc_pre_attn.y` / `.post` / `.comb` | 3.7e-9 / 1.5e-8 / 8.9e-8 | ~1e-8 - 3e-7 |
| `attn_norm` | 1.5e-8 | 2.8e-8 |
| `hc_pre_ffn.y` / `.post` / `.comb` | 2.4e-7 / 1.2e-7 / 4.2e-7 | ~1e-7 - 3e-7 |
| `ffn_norm` | 2.4e-7 | 3.0e-8 |
| `hc_post_ffn` | 9.5e-7 | 3.7e-8 |
| routing (hash, layer 0) | 0 index mismatches, 86 distinct experts | weight sum err 2.4e-7 |
| expert SwiGLU (shared + routed 254) | scale err 6.7e-9 | — |

**Closed-form norm check.** RMSNorm forces per-row RMS to 1, so post-norm L2
is `sqrt(rows*dim) * mean_abs(weight)`. Using the checkpoint's own BF16 norm
weights (`attn_norm` mean_abs 0.029486, `ffn_norm` 0.225120):

| norm | predicted | measured | rel err |
|---|---:|---:|---:|
| `attn_norm` | 7.548 | 7.539 | 0.12% |
| `ffn_norm` | 57.63 | 58.31 | 1.2% |

This is the check that catches a `+1` offset convention, a wrong eps, or a
transposed norm weight. The reference RMSNorm (`model.py:197-202`) is a
direct multiply with **no offset**, and the real weights are tightly clustered
and strictly positive, consistent with that. Note `attn_norm`'s mean weight is
an order of magnitude below `ffn_norm`'s, so the L2 drop from 237 to 7.5
across that stage is correct, not a collapse.

### The constant-leak finding

The single most valuable result of this gate is what the *standalone* forward
avoided. Reusing the MQ2R production path as a reference would have silently
inherited at least three serving-tuned constants that contradict the
checkpoint:

1. **`route_scale`** — production reads `HIPFIRE_DEEPSEEK4_ROUTE_SCALE` with
   `unwrap_or(2.2)` and never reads the config's `routed_scaling_factor`
   (**1.5**).
2. **`norm_topk_prob`** — present in config, hardwired on in production
   forward.
3. **HC `mhc_pre`** — production uses F16 weights, a hardcoded eps, and an
   env-driven `post_scale`; the reference is `2*sigmoid(...)`.

Each would have produced finite, plausible, wrong parent logits. Every
constant in `parent/*` now traces to `config.json` or a checkpoint tensor, and
no parent path reads an environment variable to choose a number. **Treat that
as a standing rule for the remaining gates.**

Separately, this is worth a maintainer's judgement on its own terms: if
`route_scale = 2.2` in the MQ2R serving path is a deliberate, measured tuning,
it should be recorded as such; if it is drift from `routed_scaling_factor`,
it may be depressing quantized-artifact quality independently of this work.

### Parent semantics already reproduced (from `inference/model.py`)

- `Expert.forward` (592-611): `up = clamp(up, -10, 10)` but
  `gate = clamp(gate, max=10)` — **`gate` has no lower clamp**. The routing
  weight multiplies `silu(gate)*up` **before** `w2`, not the expert's output.
- `MoE.forward` (614-649): `y` accumulates in **f32**, routed experts first,
  then `y += shared_experts(x)`, then cast back.
- `Gate.forward` (551-590): `linear(x.float(), weight.float())`, so no
  activation quantization. `sqrt(softplus(x))` scoring. Bias shifts scores for
  **selection only**; the returned weight is the *uncorrected* score, then
  normalized, then scaled by 1.5.
- Attention (442-549): FP8 in-place simulation at block **64** on the
  non-RoPE KV dims. RoPE is **interleaved** —
  `view_as_complex(unflatten(-1, (-1, 2)))` pairs dims `(2i, 2i+1)`. Layers
  with `compress_ratio == 0` disable YaRN (484-485).

### Gate 5 evidence

`ds4_parent_forward_gate` on `mi300x` (gfx942), all 43 layers with routed
experts, MTP excluded. **PASS at both 32 and 256 tokens.**

| | 32 tok | 256 tok |
|---|---:|---:|
| load | 18.48 s | 19.06 s |
| forward | 4.478 s | 24.81 s |
| residency | 161,872,686,172 B | same, delta 0 vs Gate 1 |
| logits | L2 8453.5, mean -1.164, std 3.990 | L2 2.300e4 |
| NaN / Inf | 0 / 0 | 0 / 0 |
| determinism (in-process) | bit-identical | bit-identical |
| `.plog` | 16,547,864 B | 132,382,744 B |

**Coherence.** `"The capital of France is"` gives top-1 `" Paris"` at logit
30.19, clear of `"Paris"` (25.52) and `"巴黎"` (25.16). This is the first
coherent output from the parent checkpoint under hipfire.

Note the gate also prints an argmax decode of the *fixed pseudo-random* token
sequence, and that output is gibberish. It is correctly labelled
diagnostic-only: the input is PRNG-generated token ids, not text, so its
continuation is meaningless. The real prompt is the coherence signal. Do not
read the fixed-sequence decode as a failure — and do not read a *plausible*
fixed-sequence decode as a success either.

**Stack stability.** The HC residual norm grows from ~400 to ~631,000 over 43
layers, geometric mean 1.1917 per layer, with no monotonic trend (the last
three ratios are 1.0523, 0.9892, 0.9412) and no per-layer ratio outside
[0.25, 4]. This is not obviously wrong: `hc_post` applies a learned
`post = 2*sigmoid(...)` factor in (0, 2), and both `hc_pre` and the final
RMSNorm are scale-invariant, so absolute stream magnitude never reaches the
logits. `hc_post` is oracle-verified to 9.5e-7. Recorded as an observation to
re-check against the reference if it ever becomes runnable.

### The act-quant BF16-domain finding

While building the compressor, the host FP8 act-quant oracle was found to
disagree with the GPU kernel by max_abs 0.25 / mean_rel 2.2e-3 — **contradicting
Gate 2**, which had certified that pair bit-exact.

Root cause was an implicit input-domain contract, not a kernel bug.
`kernel.py:41-102` declares `in_dtype = BF16` and the GPU kernel reads a BF16
buffer, but the host oracle accepted f32 and computed `amax` at full precision.
When an f32 amax sits just above a power-of-two boundary of `amax/448` and BF16
rounds it back onto the boundary, `fast_log2_ceil` differs by one and the scale
differs by exactly 2x:

| | amax | amax/448 | `fast_log2_ceil` | scale |
|---|---:|---:|---:|---:|
| f32 | 224.4 | 0.500892 | 0 | 1.0 |
| BF16 | 224.0 | 0.5 exactly | -1 | 0.5 |

Sparse groups, large absolute error — exactly the observed signature. The host
oracles now round to BF16 internally (idempotent), so the misuse is impossible
rather than merely documented. The GPU kernel was correct and is unchanged.
Gate 2 re-run: **14/14**, including a new case driven by full-f32 near-boundary
values that fails without the fix. Gate 3 and Gate 4 conclusions hold — both
already staged BF16-rounded activations.

Lesson for the remaining gates, alongside the `err_ref > 0` rule from Gate 3:
**an oracle's input domain is part of its contract.** A CPU reference that
silently accepts a wider type than the kernel it checks will agree on
synthetic data and diverge on real data.

### Sequence-length coverage — read before trusting a gate run

`compress_ratios` gives 2 layers at ratio 0, 21 at ratio 4, and 20 at ratio 128.
The number of compress events is `floor(rows / ratio)`, so **short runs do not
exercise the compressed path**:

| tokens | ratio-4 windows | ratio-128 windows |
|---:|---:|---:|
| 32 | 8 | **0** |
| 256 | 64 | 2 |
| 1024 | 256 | 8 |

Gate 5's first run at 32 tokens therefore produced zero compress events on all
20 ratio-128 layers — 47% of the stack ran an SWA-only fallback while the gate
reported PASS, finite and coherent. The 256-token run engages them. This is
pinned by `parent::compressor::tests::compress_events_require_enough_rows` so
the requirement is explicit rather than inferable from a token count.

Residual gap: the *integrated* ratio-128 attention path has been exercised at
256 tokens as part of the full forward, but not with per-layer instrumentation
confirming the compressed positions were consumed. The compressor component
itself was verified standalone on real layer 3 at 128 rows. Gate 6's 1024-token
calibration run should carry that instrumentation.

### Which quantized artifacts Gate 6 may compare against

There are **two generations** of DS4 quants on `mi300x` and they are easy to
confuse: same recipes, near-identical file sizes, different base checkpoints.
A KLD against the wrong generation measures the difference between two
*models*, not between a model and its quantization, and is meaningless.

**Eligible for Gate 6 — derived from the 0731 parent:**

| path | bytes | sha256 |
|---|---:|---|
| `quantization/deepseek-v4-flash-0731-mq2r-p3/artifacts/deepseek-v4-flash-0731.mq2r` | 82,191,359,851 | `cbf2bbcf…9318cce` |
| `quantization/deepseek-v4-flash-0731-mq2lloyd/artifacts/deepseek-v4-flash-0731.mq2lloyd` | 86,184,307,563 | see `identity/artifacts.sha256` |

The MQ2R sha256 matches the pin recorded above for the rejected Hessian
capture, which confirms it is the same P3 artifact. Both directories carry an
`identity/` tree (`artifacts.sha256`, `engine-fingerprint.json`,
`trunk-census.txt`, `qt35-actual.tsv`) — use it, do not re-derive provenance.

**NOT eligible — quants of the pre-0731 base:**

| path | bytes | date |
|---|---:|---|
| `models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r` | 82,191,362,222 | 2026-07-24 |
| `models/existing-deepseek-v4-flash/deepseek-v4-flash.mq2lloyd` | 86,184,307,283 | 2026-05-27 |

Note the MQ2R pair differs by only **2,371 bytes** and the MQ2-Lloyd pair by
**280**. Size is not a discriminator here; only the hash and the path are.

> The `route_scale` perplexity numbers recorded earlier in this document
> (1.5 → 7.0131, 2.2 → 6.0804 at ctx256) were measured on the **Jul 24
> pre-0731** MQ2R. They are a valid statement about that artifact and about
> the routed-vs-shared gain in general, but they are **not** a statement about
> the 0731 artifact and must not be cited as one. This was caught by asking
> the obvious provenance question about a file that had simply been assumed
> current — the same failure mode, one level up, that this whole document
> exists to prevent. Provenance discipline applies to what you compare
> against, not only to the thing under test.

### Quant-tier structure (identical across both generations)

`dump_hfq_dtypes` over all four artifacts:

| tier | MQ2-Lloyd build | MQ2R build |
|---|---|---|
| routed experts `ffn.experts.N.w{1,2,3}` | qt=19 MQ2-Lloyd, 33,024 tensors, 277,025,390,592 elems | **byte-identical count** |
| shared expert `ffn.shared_experts.w{1,2,3}` | **qt=3 Q8_0** (8-bit) | **qt=35 MFP4G32E8SOA** (4-bit) |
| router `ffn.gate.weight` | qt=3 Q8_0 | qt=35 FP4-E8 |
| dense tier totals | 389 × qt=3, 807 × qt=1 | 554 × qt=35, 641 × qt=1 |

This is the mechanism behind the `route_scale` puzzle. `route_scale` is
exactly the routed:shared gain
(`y = s·⟨E_i⟩_w + E_shared(x)`, weights normalized to sum `s`). Between the
two builds the **routed tier is unchanged and the shared tier was halved in
precision**, so the optimal value is necessarily build-specific — a constant
defaulted in the Q8-shared era cannot also be right for a 4-bit shared expert.
That a *routing* constant is sensitive to which tier the *shared expert* lives
in is strong evidence for quantization compensation rather than for a routing
bug in hipfire, since a genuine routing bug would not care.

Independently: the router `gate.weight` itself went Q8 → FP4, so a 4-bit
router perturbs *which* experts are selected, not merely how much they
contribute. Two builds are not comparable at a fixed `route_scale` for that
reason alone.

### route_scale sweep on the 0731 MQ2R (standalone PPL)

`deepseek4_perplexity`, ctx 256 / warmup 8 / offset 0, wikitext2 slice md5
`83b0205a304bf4e52172ecdb05f2e895`, fresh process per point, target verified
by sha256 against `cbf2bbcf…9318cce` before the sweep ran:

| route_scale | 1.2 | 1.5 | 1.8 | **2.0** | 2.2 | 2.4 | 2.6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPL | 15.59 | 9.13 | 6.63 | **6.03** | 6.84 | 7.30 | 7.90 |

A sharp, unambiguous minimum at **2.0**, monotone on both sides. Three things
follow.

First, **the optimum is build-specific.** The pre-0731 MQ2R measured earlier
gave 7.01 at 1.5 and 6.08 at 2.2; the 0731 build gives 9.13 at 1.5 and 6.84 at
2.2. Same recipe, different base weights, materially different curve. Neither
the hardcoded 2.2 nor the checkpoint's 1.5 is the optimum for this artifact.

Second, **the compensation is large.** At the checkpoint's declared 1.5 the
0731 MQ2R scores 9.13 against 6.03 at its optimum — a 51% PPL penalty. A
routed:shared gain that has to move this far to recover quality is strong
evidence that MQ2 expert quantization is losing substantial routed-expert
magnitude, i.e. the constant is compensating for a quantization defect rather
than expressing a model property.

Third, and this is the actionable one: **the real fix is at quantization
time.** `route_scale` is a single scalar applied uniformly to the routed
branch. If the underlying problem is per-expert magnitude loss, recovering it
in the quantizer would let 1.5 be both correct and better, and would remove a
per-artifact tuning knob from the serving path entirely.

> This is standalone PPL and is a **bracket only**. The Gate 6 objective is
> minimum **KLD against the parent**, a different criterion: PPL rewards a
> quantized model for being confidently right on the corpus, whereas KLD
> rewards it for matching the teacher's distribution including where the
> teacher is uncertain. Expect the KLD optimum near but not necessarily at
> 2.0, and report both.

### Gate 6 in progress — the parent forward is wrong at long sequence

Gate 6's first real measurement did exactly what a gate is for: it falsified
something Gate 5 had passed.

**Infrastructure landed.** `ds4_tokenize_corpus` pins a real corpus to a
byte-identical token-id file that every run consumes, so no producer
re-tokenizes. `ds4_quant_plog` captures an HFQ model's logits into the same
`HFPLOG01` container as the parent and **requires** `--expect-sha256`, refusing
to load an artifact whose hash does not match. All three 1024-token captures
live under
`/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/`:

| file | bytes | note |
|---|---:|---|
| `tokens.bin` | 4,096 | 1024 ids, sha256 `48b0f834…5bb86dd`, from the wikitext2 slice md5 `83b0205a…` |
| `parent_1024.plog` | 529,530,904 | batched prefill |
| `mq2r_1024.plog` | 529,530,904 | sha256-verified `cbf2bbcf…9318cce` |
| `mq2lloyd_1024.plog` | 529,530,904 | sha256-verified `a6195336…` |

**A comparator bug found first.** PPL came back at 5.7e6 (parent) and 1.8e6
(MQ2R) — worse than the uniform 129,280 — while both models emitted coherent
text. `.plog` row `t` predicts token `t+1`, but `compare()` scored row `t`
against `token_ids[t]`. A shift scan settled it empirically:
`argmax == token_ids[t]` in **0/48** sampled rows for both files, against
`argmax == token_ids[t+1]` in 13/48 (parent) and 26/48 (MQ2R). The shift now
happens inside `compare()`, same remedy as the BF16 `amax` fix — an
interface's input convention is part of its contract. KLD never used targets
and was unaffected.

**The real finding.** With PPL corrected:

| model | PPL | KLD vs parent | top-1 vs parent |
|---|---:|---:|---:|
| parent | **163.89** | — | — |
| MQ2R | 14.70 | 7.676 | 0.3008 |
| MQ2-Lloyd | 14.56 | 7.659 | 0.3008 |

**The parent is 11x worse than its own 2-bit quantization.** That is not
possible for a correct teacher.

Triangulation makes it conclusive. Comparing the two quantized artifacts
**against each other** — different recipes, one with a Q8 shared tier and one
with FP4-E8 — gives KLD **0.102**, top-1 **0.873**, p95 0.373. Two independent
quantizations converge; the parent diverges from both by the same amount, to
six digits of identical top-1 (308/1024). The parent is the outlier, which
also means the production DS4 path is fine and the defect is in the parent
code written this session.

Accuracy by position, identical tokens for all three:

| bucket | parent | mq2r | lloyd |
|---|---:|---:|---:|
| [1,32) | 0.292 | 0.292 | 0.292 |
| [32,64) | 0.542 | 0.625 | 0.708 |
| [64,127) | 0.458 | 0.708 | 0.625 |
| [128,256) | 0.458 | 0.583 | 0.583 |
| [256,512) | 0.333 | 0.458 | 0.375 |
| [512,1022) | **0.208** | 0.500 | 0.542 |

The parent decays progressively while both quants stay flat, and all three
agree exactly at [1,32) — a clean control. There is **no break at 128**, so
this is not the ratio-128 compressor switching on; it is cumulative in
context length. The parent runs batched prefill while the quantized path runs
sequential decode, so the batched SWA/compressor bookkeeping is the leading
suspect.

**Why Gate 5 missed it.** Gate 5 checked finiteness, in-process determinism,
stage-norm sanity, and a **5-token** prompt. None of those exercise long
context. Its 256-token run *did* emit gibberish on the fixed sequence, and
that was explained away as "pseudo-random input, so a meaningless
continuation" — a reasonable-sounding story that turned out to be covering a
real defect. **A gate that cannot fail teaches nothing:** next-token accuracy
against a real corpus, compared to a known-good reference, is the check that
would have caught this on day one, and it costs one perplexity number.

Diagnostic scripts are on the box at `/root/plog_shift_scan.py` (alignment)
and `/root/plog_pos_scan.py` (accuracy by position); both sample rows rather
than streaming 529 MB and run in seconds.

> Until this is fixed, `parent_1024.plog` is **not** a baseline and no KLD
> number derived from it means anything. The route_scale KLD optimization,
> Gate 7 Hessians, and Gate 9 GPTQ all depend on a correct teacher and are
> blocked behind it.

#### Elimination ledger — what the defect is NOT

Each closed with evidence at a realistic operating point, not by inspection.
Recorded because the search space is large and re-treading it is the main way
this stalls.

| hypothesis | verdict | evidence |
|---|---|---|
| `.plog` row misalignment | closed | shift 0 optimal; `argmax==tok[t]` 0/48 for both files |
| PPL scoring off-by-one | **was real, fixed** | row `t` scored against `token_ids[t]`; shift moved inside `compare()` |
| batch size / prefill width | closed | a 128-token run **bit-matches** 1024 on every shared bucket |
| compressed-index `offset=0` | closed | sentinel proves idx 5 → COMP 2005, never SWA 1005 |
| wave64 `__shfl_down` width | closed | fixed defensively; recaptured plog **byte-identical** |
| sequential decode vs batched prefill | closed | sequential slightly *worse* (0.484 vs 0.516) |
| Hyper-Connections gain | closed | f64 oracle ~1e-7 at layers 0/5/20/40 on grown residuals; `post` mean 0.37, not saturating |
| MoE component paths | closed | f64 oracle ~1e-7 at L5 row 0: routing exact, weight sum exactly 1.5, expert-35 5.7e-6, shared 2.9e-6 |
| RoPE frequency tables | closed | bit-identical to f64 transcription, both policies; constants from `config.json`, not `ModelArgs` defaults |
| RoPE table *selection* on ratio>0 | closed | GPU q vs correct YaRN oracle 1.4e-6, vs wrong plain table 21.3 |
| ratio-0 attention | closed | f64 oracle **flat** across position (row 0 1.19e-6, row 100 1.43e-6) |
| ratio>0 main attention | closed | same 1e-6 floor and flatness on layers 2 and 3 |
| row 0's 6e5 residual | red herring | row 0's input direction aligns gate/up (cos 0.498 vs −0.009 isotropic); all three models agree exactly in [1,32); massive first-token activations are documented and functional |

**Two methodology notes worth more than any single elimination.**

*Know the oracle's arithmetic domain.* A layer bisect reported the MoE block
diverging from an f64 reference by 6.4e-3 at layer 0 — apparently four orders
above the 1e-6 floor every other block shows. But BF16 carries 8 significant
bits, so its unit roundoff is `2^-8 = 3.9e-3`, and `parent_linear_expert` runs
BF16 MFMA while HC and the norms run f32/host-f64 paths. An f64 oracle *should*
disagree with a bf16 GEMM at ~4e-3, so the apparent signal is the same order as
the expected floor. This is the third instance this session of the same class
of mistake (BF16-vs-f32 `amax`, the plog target shift). **A comparison is only
meaningful once its floor is established on a case known to be correct, or the
oracle is domain-matched by rounding operands to bf16.**

*Beware relative error on pathological magnitudes.* The same bisect reports FFN
divergence growing to 4e0 deep in the stack — but the residual there is ~6e5,
and a relative metric against a huge, possibly degenerate input misleads in
either direction.

bf16 also cannot explain the gap on its own: both quantized models run at bf16
or worse and score 14.6 against the parent's 163.89.

### Root cause found — `hc_post` contracted the wrong `comb` axis (2026-08-02)

`Block.hc_post` contracted the wrong axis of the sinkhorn `comb` matrix.
`model.py:692` is
`torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`, which
broadcasts to `comb[A,B] * residual[A,d]` and sums over `A`, the **first** hc
axis:

```text
y[B,d] = sum_A comb[A][B] * residual[A,d]
```

`hc_mix_4stream_batched` contracts the **second** (kernel line 46,
`A[stream_out * HC_MULT + s_in]`), so it requires `comb^T`. The parent passed
`comb` untransposed. Production reaches the same kernel with its comb already
in the kernel's orientation, which is why production was never affected.

The axis is load-bearing, not a naming convention: `hc_split_sinkhorn`
(`kernel.py:401-423`) ends its loop on `comb / comb.sum(-2)`, so the **columns**
sum to 1. Contracting `A` is norm-preserving; contracting the other axis
weights the residual by row sums, which are not 1, and amplifies it every
layer.

**Why it survived thirteen-plus eliminated hypotheses.**
`layer_ref.rs::hc_post_ref`, the host oracle, contracted the same wrong axis.
Every HC comparison therefore agreed to ~1e-7 while the model was badly wrong,
and the layer-0 bisect put the median row at the floor (p50 3.576e-7). A shared
misreading between `parent/forward.rs` and `parent/*_ref` is invisible by
construction — the documented cost of the standalone-parent decision, now paid.
No test caught it either; the tests encode the same convention.

Fixed in `dc4a6cd8f`. `comb` now has one meaning across the parent (reference
orientation), converted at the single kernel boundary.

### Post-fix state — parent wins at 256, two defects remain

| tokens | parent pre → post | mq2r | lloyd |
|---|---|---|---|
| 128 | 23.638 → 11.538 | 9.297 | 8.734 |
| 256 | 29.644 → **8.619** | 11.080 | 11.289 |
| 512 | 63.498 → 17.162 | 11.693 | 12.167 |
| 1024 | 163.892 → 59.507 | 14.703 | 14.564 |

Two independent defects remain, cleanly separated by the data:

1. **L37 → L38 step of 8.2x** (14221.72 → 116669.77), present at *all four*
   lengths including 128 where `index_topk=512` selection is a complete no-op.
   Independent of any long-context path. `LoaderOracle` verified the loader
   bit-exactly but sampled only layers 4 and 26 — layers 36-39 were never
   checked, and one malformed tensor produces exactly this signature.
2. **Accuracy step at position 512.** Buckets are position-anchored and
   length-invariant. The parent beats mq2r in four of six buckets and holds
   through `[256,512)` at 0.583, then drops to 0.333 in `[512,1022)` where mq2r
   manages 0.500. `index_topk = 512`. Prior indexer work eliminated *causality*,
   not *selection correctness*, and ran at 128 tokens where top-k selects
   everything and the selection logic is never exercised.

Reading KLD here: it rose at 1024 (7.676 → 8.793) while PPL fell 2.75x and
top-1 rose 0.3008 → 0.3408, and fell at 128 (5.297 → 4.768).
`KLD(P_parent || Q_quant)` is weighted by the parent's own distribution, so a
sharper-but-still-wrong parent scores higher. Judge by PPL, top-1 and buckets.

Only compare PPL and geo-mean growth **within a length** — both are
length-dependent (geo mean 1.1684 at 128, 1.1498 at 512, 1.1410 at 1024).
Position-bucket accuracy is the one length-invariant metric.

### Gates 7-9 are conditional on a value test (decided 2026-08-02)

Gates 7-9 are **no longer a given**. The independent GPTQ cross-reference
(`crates/hipfire-quantize/reference_gptq/`) found the solver math already
correct — exact agreement with the paper reference in f64 — so there is no
broken GPTQ to fix. The entire program therefore rests on an unproven premise:
that parent-derived Hessians and parent-KLD calibration produce measurably
better children than the pipeline already ships.

**The gate:** once the parent is correct, re-quantize *one* MQ2R build with
parent-derived Hessians and a parent-KLD-swept `route_scale`, and measure
against the shipped **14.703 at 1024 tokens**. Proceed to the full program only
if it wins. If it does not, stop — that is a cheap answer, not a wasted one.

Cheap wins to keep regardless of the outcome:

- `route_scale` at quantization time: MQ2R standalone PPL goes 6.63 → **6.03** →
  6.84 across 1.8/2.0/2.2.
- The `gptq.rs` absolute-vs-fractional damp footgun: it takes `initial_damp` as
  an absolute value while `e8_gptq` uses fractional `LAMBDA*mean(diag)`. DS4
  rides the safe path; other callers must pre-multiply by `mean(diag)`.
- E8H1 discards cross-block Hessian mass (Frobenius ≈1.25e3 on an N=24, K=512
  draw). By design, but it is an accuracy ceiling no calibration data can lift.

### Not yet done

Gates 6-9. Specifically:

- **No pinned parent logit baseline yet.** Both Gate 5 runs used
  `--skip-shard-hashes`, so their manifests carry placeholders rather than a
  pin, and the token sequences were pseudo-random rather than a real corpus.
  Those `.plog` files are smoke artifacts and **must not** be promoted to the
  Gate 6 baseline.
- Gate 6 needs: a real 1024-token corpus with a recorded hash, a full-shard-hash
  manifest, cross-process determinism confirmed (only in-process is proven so
  far), and then MQ2L/MQ2R logits captured on byte-identical token ids for the
  KLD comparison. `plog::compare` is written and unit-tested but has never
  consumed a real parent logit file. **Use only the 0731-derived artifacts
  listed above, verified by sha256 before the run, not by filename.**
- Forward cost is 24.8 s per 256 tokens, so a 1024-token capture is order
  100 s per model plus load. Tractable; measure before scaling to the 8K/16K/32K
  expansion in gate 8.


## Producer boundary

For GPTQ, accumulate the operand actually consumed by the parent weight
matmul: the post-dynamic-activation-quantization/dequantization matrix. Record
the pre-quant matrix only as optional diagnostic evidence. The boundary must
be named in the output manifest; "F32 activations" is not sufficient
provenance.

Accumulate 256-channel `X^T X` blocks online with rocBLAS and write the
Hessians directly. The intermediate `.acts` format remains useful for codec
and collector debugging, but the production parent run should avoid another
13 GiB activation dump.

## Mandatory evidence manifest

Every parent-logit and Hessian bundle must include:

- source shard/index hashes
- engine commit and dirty diff hash
- producer binary hash
- ROCm path/version and GPU architecture
- tokenizer hash
- exact token IDs or corpus hash
- model configuration and RoPE convention
- activation capture boundary (`pre_quant` or `post_dynamic_fp8`)
- per-tensor row counts and shapes
- logits/Hessian hashes
- KLD/PPL command and result artifacts

No artifact without this manifest is eligible for GPTQ or a quality claim.

## Do not do

- Do not use the preserved 554 MQ2R-driven Hessians for parent GPTQ.
- Do not delete or overwrite the preserved capture.
- Do not call an activation parent-derived merely because its buffer dtype is
  F32.
- Do not let unrecognized `F8_E4M3`, `F8_E8M0`, or packed expert `I8` fall
  through to `DType::Raw`.
- Do not dequantize all 256 experts simultaneously.
- Do not begin GPTQ before the parent logit baseline and existing-quant KLD
  are durable.
- Do not treat coherent text alone as numerical validation of the parent
  forward.
