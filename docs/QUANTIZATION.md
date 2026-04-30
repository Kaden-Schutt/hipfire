# Quantization

How hipfire stores weights and KV cache. This is the design / math
side. For the user-facing "how do I quantize my model" page, see
[QUANTIZE.md](QUANTIZE.md).

## Weight formats

All weight formats group elements into 256-wide blocks (G256). Each
block has independent scale + zero-point metadata. The bitwidth and
whether a Walsh-Hadamard rotation runs before quantization defines the
four production formats.

| Format | Bits | Rotation | Bytes / 256 elements | Use case |
|---|---|---|---|---|
| HFQ4-G256 | 4 | none | 136 (8 hdr + 128 data) | Llama / Qwen3 / dense |
| HFQ6-G256 | 6 | none | 200 | Dense, higher quality |
| MQ4-G256 | 4 | FWHT | 136 | Qwen 3.5+ hybrid |
| MQ6-G256 | 6 | FWHT | 200 | Qwen 3.5+ higher quality |
| MQ3-G256 | 3 | FWHT | 104 (8 hdr + 96 data) | Sub-4-bit bandwidth play (≥9B models only) |
| MQ2-G256 | 2 | FWHT | 72 (8 hdr + 64 data) | Reserved — uniform-grid collapses; pending Lloyd-Max codebook |

**Sub-4-bit caveat**: MQ3 and MQ2 reuse the production HFQ3/HFQ2
decode kernels with a pre-rotated `x` (no separate kernel). Local
validation pass on Qwen 3.5 / 3.6 (gfx1100, master `c448d5e`):

- **MQ3 quality threshold ≈ 9B.** 27B + 9B both produce fluent on-topic
  output across the 4-prompt coherence battery; 4B partially collapses
  (recognises intent but loops in `<think>` / mixes languages); 0.8B
  produces gibberish. Don't recommend MQ3 below 9B.
- **MQ2 with the current uniform 4-level codebook collapses at every
  size tested** (0.8B / 4B / 9B → multilingual mojibake / symbol soup
  on all 4 prompts). Path D Lloyd-Max non-uniform codebooks (per-block
  squared-error-minimising, see PRD §5.2) are the planned remediation;
  until then `--format mq2` is reserved and gated behind an explicit
  opt-in flag.

There is no WMMA prefill path for MQ3 or MQ2 yet, so prefill falls
back to per-row GEMV until the kernel lands in a follow-up PR. The
eligibility check in `qwen35::forward_prefill_batch` (`is_batchable_la`)
correctly excludes MQ3/MQ2 from the batched fast path; per-token
`forward_scratch` handles them via `weight_gemv`'s MQ3/MQ2 dispatch
arms. Engine wiring is correct — the quality verdict is purely about
the format's expressiveness on each model size, not a runtime bug.

Header layout (8 bytes): 4 bytes scale (f32-bitcast-from-f16) + 4 bytes
zero point. Data: bitwidth × 256 / 8 bytes = 128 (4-bit) or 192 (6-bit)
of packed nibbles / sextets.

For embedding tables: always Q8F16 (32 elements per block, 1-byte
scale + 32 int8 codes — total 33 bytes). Q4-grade is too lossy for
large-dim embeddings; the per-token absolute lookup error compounds
through the rest of the network.

For 1D norms / scale vectors: F16 unmodified. They're tiny (one float
per dim) and precision-sensitive.

## FWHT rotation (the M in MQ)

Each 256-wide group is multiplied element-wise by a `±1` sign vector,
then transformed by a fast Walsh-Hadamard transform, then divided by
sqrt(256). The same operation runs on the input vector at inference
time (kernels apply the inverse), so the GEMV math is unchanged.

What this buys: outliers in the original weight distribution get
*spread* across the group, making the post-rotation distribution more
uniform. Quantization to 4 bits is then less destructive — the per-
group dynamic range is narrower.

The sign vectors are deterministic (PRNG seeds 42 / 1042) and shared
between the quantizer and the engine; no per-model calibration.

```
quant time:    w' = WHT(signs ⊙ w) / 16
                     → quantize w' to 4 bits per block
inference:     y = w'·x  =  WHT(signs ⊙ w)·x / 16
                          =  signs ⊙ w · WHT(x) / 16    (WHT self-adjoint)
                            └────────┘    └──────┘
                            stored MQ4    rotate_x_mq
```

So the engine does `rotate_x_mq` on the input vector once per layer
and the quantized GEMV consumes the pre-rotated input. The cancellation
is exact in fp32 arithmetic; in mixed precision there's a small
numerical drift but well below the per-block quantization error.

## When MQ vs HFQ matters

The FWHT rotation was calibrated against the Qwen 3.5 / 3.6 weight
distributions (DeltaNet hybrid models). On those, MQ4 hits Q8-level
quality at Q4 bandwidth — the project's central claim.

On Llama-style dense models that weren't trained against this weight
space, the rotation still works mathematically (the cancellation is
exact) but provides no quality lift over plain HFQ4. It only adds the
runtime `rotate_x_mq` kernel-launch cost.

**Practical rule**: pick MQ for Qwen 3.5+, HF for everything else.
The CLI defaults reflect this — `hipfire quantize <safetensors-dir>`
defaults to `--format mq4`, `hipfire quantize <file.gguf>` defaults to
`--format hf4`.

## KV cache (asym format)

The KV cache stores per-token K and V for every prefix position.
Memory grows linearly with seq_len, so quantizing it has out-sized
impact on long-context inference.

```
mode      K layout                              V layout
─────────────────────────────────────────────────────────
q8        Q8_0 (32-element blocks)              Q8_0
asym4     Lloyd-Max rotated 4-bit               Q8_0
asym3     Lloyd-Max rotated 3-bit  (default)    Q8_0
asym2     rotated 2-bit                         Q8_0
```

K and V are quantized differently because they have different
sensitivities. K participates in the softmax — small numerical errors
get exponentiated and shift attention mass between tokens, especially
on multi-turn recall (`"Kaden"` becomes `"Kendall"` if you go too
aggressive). V is the value bank that the attention probabilities
already weight; modest noise smears across all positions and matters
less.

So K gets the rotation + Lloyd-Max scalar quantization at low
bitwidth, V stays Q8.

Lloyd-Max here means: per-block, find the K-bit codebook that
minimizes squared error against the rotated K vector for that head,
not a uniform scale + zero. The codebook is two floats (min / max)
plus 2^K codes implied uniformly between them — same storage cost as
asymmetric uniform, slightly better fit on the actual K distribution.

## Asym KV per arch defaults

Set in `cli/index.ts::archDefaults`. Override with `hipfire config set
kv_cache <mode>` or `HIPFIRE_KV_MODE=<mode>` env.

| Arch | Default | Reason |
|---|---|---|
| gfx1100 (7900 XTX) | asym3 | 24 GB VRAM affords it; quality matches Q8 |
| gfx1101 (7900 XT) | asym3 | Same |
| gfx1102 (7800 XT) | asym3 | Same |
| gfx1030 (V620 / 6800 XT) | asym3 | 32 GB on V620 — plenty of headroom |
| gfx1031 (6700 XT) | asym3 | 12 GB |
| gfx1032 (6600 XT) | asym2 | 8 GB — tighter quant for headroom |
| gfx1010 (5700 XT) | asym2 | 8 GB |
| gfx1013 (BC-250 APU) | asym2 | 14 GB shared, prioritize ctx length |
| gfx1151 (Strix Halo APU) | asym2 | 16 GB shared |
| gfx1200 (9070 XT) | asym3 | 16 GB |
| (default fall-through) | asym3 | Includes gfx94x / MI300X — override to `q8` if you have spare HBM |

Override with `hipfire config set kv_cache <mode>` or
`HIPFIRE_KV_MODE=<mode>` env.

## Why a custom format at all

llama.cpp's GGUF Q4_K_M has nearly the same on-disk size and in-place
GEMV cost. The win comes from two places:

1. **Fused projections**. hipfire's GEMV+GEMM kernels for HFQ4 / MQ4
   fuse the 3-way QKV (or 6-way QKVZA in DeltaNet) into one kernel
   launch. This is where the 1.7–2.1× decode lead over llama.cpp
   comes from in the bench table.
2. **WMMA prefill**. Batched MQ4 GEMM uses RDNA3's WMMA intrinsic
   directly — it's smaller-batch-friendly than the cuBLAS-style
   replacement llama.cpp's ROCm path uses.

Neither is exotic — they're both engineering, not algorithmic
breakthroughs. But they add up.

## Format file format (.hfq / .mq / .hf)

```
0x00  "HFQM"        (magic, 4 bytes)
0x04  version       (u32 LE = 1)
0x08  arch_id       (u32 LE — 0=llama, 1=qwen3, 5=qwen3_5, 6=qwen3_5_moe)
0x0C  n_tensors     (u32 LE)
0x10  metadata_offset  (u64 LE)
0x18  data_offset      (u64 LE — 4096-aligned)

[metadata_offset .. data_offset]:
  metadata_json  (UTF-8 JSON: config, tokenizer, gguf_meta, source)
  tensor_index:
    n_tensors: u32 LE
    for each tensor:
      name_len: u16 LE
      name: utf-8
      quant_type: u8     (0=Q4F16G64, 1=F16, ..., 6=HFQ4G256, 13=MQ4G256, ...)
      n_dims: u8
      shape: u32 LE × n_dims
      group_size: u32 LE
      data_size: u64 LE

[data_offset .. EOF]:
  tensor data, sequential, in tensor_index order
```

`HfqFile::open` mmaps the file; `tensor_data(name)` returns a slice
into the mapping. The daemon never copies tensor bytes — they go
straight from mmap to a HIP `hipMalloc` + `hipMemcpy` upload.
