# DS4 heterogeneous G5: dense attention branch overlap

Date: 2026-08-06  
Branch: `ds4-beta-staging`  
Implementation commit: `2d82b2516`  
Status: accepted G5 increment; G5 remains open against the 70 tok/s target

## Lever

The gfx1100-owned attention block formerly serialized four independent branches
after the shared RMSNorm/FWHT preparation:

- Q-LoRA projection,
- joint KV projection,
- the main compressor,
- the indexer compressor.

The heterogeneous-only path now splits Q-LoRA into a shared preparation and a
projection phase. It records a same-device fork event, runs KV plus both
compressors on a persistent secondary gfx1100 stream while Q-LoRA projects on
the primary stream, then joins before RoPE, indexer, attention, and HC mix.
The ordinary single-device, EP, prefill, and Qwen routes are unchanged. No
arithmetic or reduction order changed.

Relevant source:

- `crates/hipfire-arch-deepseek4/src/forward.rs`: `q_lora_prepare`,
  `q_lora_project`, and `ds4_attn_block_heterogeneous`.
- `crates/hipfire-arch-deepseek4/src/heterogeneous.rs`: persistent secondary
  stream plus fork/join event ownership and teardown.

## Canonical fixture

- Model: DeepSeek V4 Flash 0731 MQ2R P3.
- Model SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`.
- Prompt: `benchmarks/prompts/ds4_heterogeneous_code_2048.txt`.
- Prompt MD5: `593234a767e71b97a3a4dad6431b47ce`.
- 2,048 prompt tokens, 512 generated tokens, batch 1, greedy, top-k 6,
  Q8 KV configuration, speculation off.
- Direct heterogeneous HIP: gfx1100 dense/non-routed owner plus gfx1151 routed
  expert owner.
- Binary SHA-256:
  `2e8b9f9e5ac23c234d88b857b0a80261999b6f9113972ddc36b9bb468404c43e`.

## Product result

The prior canonical screen at `bdafb6915` measured 29.454558677 tok/s.
The accepted candidate produced three fresh-process samples:

| Sample | Prefill tok/s | Decode tok/s | Decode seconds |
|---|---:|---:|---:|
| 1 | 30.664962912 | 30.043910485 | 17.041722989 |
| 2 | 30.583196961 | 30.080781893 | 17.020834160 |
| 3 | 30.111782142 | 29.954655432 | 17.092501737 |

Decode median is **30.043910485 tok/s**, a **+2.000885%** gain over the
matched pre-lever screen. The full min-to-max spread is 0.126126461 tok/s, or
0.419807% of the median. Prefill median is 30.583196961 tok/s.

Every sample produced 2,491 identical bytes:

- MD5: `ee05ab4f07393fb7d624d966a7dde4af`.
- SHA-256:
  `3611840208334c77b3cfcf85984786920deabd550ba83311645f413d3ba6608b`.

## Mechanism proof

A 32-token selected-decode `rocprofv3` diagnostic trace recorded separate
gfx1100 compute queues. The primary queue executed 837.019135 ms of kernels;
the new side queue executed 93.137921 ms. Their measured concurrent interval
was 50.947987 ms, or 54.701658% of side-queue work and 1.592125 ms per
generated token. The profiler-perturbed 25.485 tok/s diagnostic number is
excluded from product performance evidence.

Raw trace SHA-256:
`88e32c9248e0b270164f852e432f2f99f3b317a520bb11467e1a0dec232232b3`.

## Evidence

Durable evidence on hipx:

`/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g5/`

- `dense-attn-overlap-product-screen/`
- `dense-attn-overlap-promotion/sample-2/`
- `dense-attn-overlap-promotion/sample-3/`
- `dense-attn-overlap-mechanism/`

## Validation and exclusions

- `scripts/fmt-changed.sh`: pass.
- `git diff --check`: pass.
- `cargo check -p hipfire-arch-deepseek4 -p rdna-compute`: pass.
- `cargo test -p hipfire-arch-deepseek4 --lib`: 251 passed, 1 ignored.
- No PM4/Redline work was started before the G5 target.
- No speculative decode, RCCL retry, generic gfx11 product image, weight,
  quant, top-k, sampling, or KV-policy change was made.

Next: retain this accepted increment and structurally rewrite/screen the
gfx1100 dense MFP4E8 SoA GEMV path; do not retry the already-rejected U4/U8
unroll variants.
