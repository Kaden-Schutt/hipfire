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

1. **Inventory gate**: all 72,317 source tensors accounted for; every native
   weight has exactly one valid scale companion; MTP is explicitly excluded.
2. **Codec gate**: GPU E4M3/UE8M0 and E2M1/UE8M0 decode matches an independent
   CPU oracle on fixed edge cases and sampled checkpoint values.
3. **Linear gate**: dense and expert matmul outputs match the checkpoint's
   bundled operator semantics on fixed inputs.
4. **One-layer gate**: 16-token layer canary with finite state and logits.
5. **Parent-forward gate**: full 43-layer 32-token coherent output; finite
   logits and deterministic fixed-input hashes.
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
