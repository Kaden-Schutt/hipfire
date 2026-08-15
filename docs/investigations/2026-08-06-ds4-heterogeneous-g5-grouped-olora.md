# DS4 heterogeneous G5: grouped gfx1100 O-LoRA

Date: 2026-08-06  
Branch: `ds4-beta-staging`  
Implementation commit: `1f4f4c558`  
Status: accepted G5 increment; G5 remains open against the 70 tok/s target

## Lever

The heterogeneous attention path formerly issued eight independent
`MFP4G32E8SOA` O-LoRA GEMVs per attention layer on the gfx1100 dense owner.
There are 43 attention layers, so this cost 344 small launches per generated
token. The gfx1151 and gfx942 routes already had grouped 2-D O-LoRA kernels.

The heterogeneous-only route now invokes one grouped 2-D kernel per attention
layer. It reuses the arithmetic-identical width-32 kernel source and compiles an
exact `gfx1100` code object with a distinct symbol. The ordinary single-device,
EP, MTP, prefill and Qwen paths retain their prior dispatch. No arithmetic,
weights, quantization or reduction order changed.

Relevant source:

- `crates/hipfire-arch-deepseek4/src/forward.rs`: `OloraSchedule` and the
  heterogeneous-only grouped dispatch in `attn_stub`.
- `crates/rdna-compute/src/gemv.rs`:
  `gemv_mfp4g32_e8_soa_grouped_gfx1100`.
- `crates/rdna-compute/src/kernels.rs`: exact-gfx1100 grouped source identity.

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
  `ee8ba3b612995c60454bed81590bba7631b0de80738eb95f96b50fafdcdfcfd6`.

## Product result

The accepted pre-lever median at `2d82b2516` was 30.043910485 tok/s. The
candidate produced three fresh-process samples:

| Sample | Prefill tok/s | Decode tok/s | Decode seconds |
|---|---:|---:|---:|
| 1 | 32.634410057 | 32.062364971 | 15.968878168 |
| 2 | 32.760970125 | 32.002912953 | 15.998543656 |
| 3 | 32.747268256 | 31.980247999 | 16.009882100 |

Decode median is **32.002912953 tok/s**, a **+6.520464%** gain over the
accepted pre-lever median. The full min-to-max spread is 0.082116972 tok/s, or
0.256592% of the median. Prefill median is 32.747268256 tok/s.

Every sample produced 2,491 identical bytes:

- MD5: `ee05ab4f07393fb7d624d966a7dde4af`.
- SHA-256:
  `3611840208334c77b3cfcf85984786920deabd550ba83311645f413d3ba6608b`.

## Mechanism proof

The exact-gfx1100 micro used the in-model O-LoRA shape, `G=8`, `M=1024`,
`K=4096`, and nine weight replicas for a 161,611,776-byte working set. Across
seven alternating trials:

- sequential eight-launch median: 0.122209 ms;
- grouped one-launch median: 0.055395 ms;
- isolated speedup: 2.2061x;
- 8,192 outputs: raw-bit exact;
- occurrence-weighted projection at 43 layers: 2.872964 ms/token saved.

The product result saves 2.037460 ms/token relative to the accepted pre-lever
median. This is smaller than the isolated projection, as expected, but clears
the product admission threshold without relying on the micro for sizing.

Micro log SHA-256:
`0a19f5b328fc50fc87b508cdabb67cba59b78a8ebe62f13f80e1cd1df73bcc47`.

## Evidence

Durable evidence on hipx:

`/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g5/`

- `gfx1100-e8-grouped-micro/`
- `gfx1100-e8-grouped-product-screen/`
- `gfx1100-e8-grouped-promotion/sample-2/`
- `gfx1100-e8-grouped-promotion/sample-3/`

Product run-log SHA-256 values, in sample order:

- `e5c6cba874bd7c360d38d1b1085f96dc9ab0e3f4fabd51639e89204ccaa31bbc`
- `2a3852c31b1e95ff9764e0f1a6969d04d8365da8320016029a0021c30bf7abc8`
- `f315e835078882d836ac3fbf3d2a330793de2a7a571ff063fac633a3cb11b492`

## Validation and exclusions

- `scripts/fmt-changed.sh`: pass.
- `git diff --check`: pass.
- `cargo check -p hipfire-arch-deepseek4 -p rdna-compute`: pass.
- `cargo test -p hipfire-arch-deepseek4 --lib`: 251 passed, 1 ignored.
- No generic `gfx11` code object was placed on the product hot path; the common
  source is compiled for exact gfx1100.
- No PM4/Redline work was started before the G5 target.
- No speculative decode, RCCL retry, weight, quant, top-k, sampling or
  KV-policy change was made.

Next: retain this accepted increment and profile the new 32.00 tok/s route.
Continue G5 with only occurrence-weighted candidates projecting at least 2%.
