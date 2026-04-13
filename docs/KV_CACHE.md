# KV Cache — asymmetric rotated K + Q8 V (asym3 family)

hipfire stores the K/V cache asymmetrically: **K is rotated-quantized at
2/3/4-bit with Lloyd-Max centroids, V is stored as Q8_0 in normal space**.
Only K needs the rotation machinery because only K suffers precision damage
from aggressive quantization — V reads go through an existing Q8_0 flash
reduce path unchanged.

This is what the `kv_cache` config key selects (`q8`, `asym4`, `asym3`, `asym2`).
`asym3` is the default on every RDNA3/RDNA4 card.

## The problem asym solves

Standard Q8_0 KV cache is 544 bytes/head/position on Qwen 3.5 (head_dim=256).
A full 128k context on 9B (4 KV heads × 8 full-attention layers) = 4.6 GB just
for KV, before factoring weights + activations. Untenable on 12 GB cards.

The obvious answer is "compress K more aggressively." But the obvious bad answer
is "just drop bits from K." 4-bit per-channel K collapses outlier components
that carry rare-token identity. Multi-turn recall fails — Qwen 3.5 9B with
naive 4-bit K will answer "What is my name?" with "Kendall" instead of the
actual "Kaden" from two turns earlier.

asym3 fixes this in two pieces:

1. **Rotate K before quantizing**, so outlier energy spreads across channels
   instead of concentrating on a few high-variance dimensions.
2. **Keep V at Q8**, because V errors propagate directly into the attention
   output and multi-turn recall is especially sensitive.

## Byte layout

All asym modes store K as:

```
[4 B scale][head_dim/2 packed nibbles]    (asym4 — 4 bits per element)
[4 B scale][head_dim packed ~3 bits]       (asym3 — 3 bits per element, packed in 8-thread blocks)
[4 B scale][head_dim/4 nibbles]            (asym2 — 2 bits per element)
```

V is Q8_0 as before: `[head_dim bytes][2 B scale]`.

Compression per head per position on Qwen 3.5 (head_dim=256):

| Mode | K bytes | V bytes | Total | vs fp32 | vs Q8 baseline |
|---|---:|---:|---:|:---:|:---:|
| fp32 (theoretical) | 1024 | 1024 | 2048 | 1.0× | 0.27× |
| q8 | 272 | 272 | 544 | 3.76× | 1.0× |
| asym4 | 132 | 272 | 404 | 5.1× | 1.35× |
| **asym3** | **100** | **272** | **372** | **5.5×** | **1.46×** |
| asym2 | 68 | 272 | 340 | 6.0× | 1.60× |

## Rotation — block-diagonal 2×2 Givens, not full WHT

The first shipped attempt (`givens4`) used pure random-angle 2×2 Givens
rotations in-place before quantization. This passed single-turn tests, and
shipped in v0.1.4. It *failed* multi-turn rare-token recall.

**Root cause** (fixed in v0.1.5): the K kernel had a silent half-coverage
bug on `head_dim=256` models. The old layout was `tid*4 × 32 threads = 128`
dims — the second half of each 256-dim head was never written. Qwen 3.5 uses
head_dim=256 exclusively. Invisible to md5 comparisons, perf benchmarks, and
single-turn tests (because the lower 128 dims alone are enough for common
tokens). Only multi-turn rare-token recall broke, and only on 9B where outlier
components concentrate in the uncovered upper 128 dims.

Fix: explicit 2-pass loop (`half=0,1`) in every K kernel. Lands alongside the
asym family in v0.1.5.

## Lloyd-Max centroids — why they matter

K distribution post-RoPE is approximately N(0, 1/head_dim). A uniform grid of
quantization values (like naive 2/3/4-bit) wastes most codepoints on the
low-probability tails.

Lloyd-Max solves for the centroid positions that minimize mean-squared error
on a fixed distribution. For N(0, 1/256):

- `TURBO_C4` — 16 values optimized for 4-bit
- `TURBO_C3_256` — 8 values for 3-bit (head_dim=256 variant)
- `TURBO_C2` — 4 values for 2-bit

Tables are pre-computed and compiled in as constant memory. The quantize
kernel picks the closest centroid index per element; the dequantize side
looks it up.

## Flash-only constraint

All three asym modes are **flash-only**. There is no non-flash read kernel for
rotated-K cache. The `flash_mode` config key (auto/always/never) applies only
when `kv_cache=q8` — for asym modes the flash path is hard-wired.

`run_fa_layer_body` (the non-flash fallback) runs only for:
- Qwen 3 (non-DeltaNet architectures)
- Weights that don't pass `is_batchable_la` (non-MQ/HFQ quantization)

The TUI surfaces this: when `kv_cache=asym*` is selected, the `flash_mode` row
shows `(ignored — asym is flash-only)`.

## Batched prefill + flash reduce

On prefill, hipfire batches tokens through a fused FA path. Every asym mode
takes `attention_flash_asym{2,3,4}_tile_batched` for the per-head score
computation, then a shared `attention_flash_asym_reduce_batched` for the
softmax + V accumulation. Value-side is Q8_0 for all three, so the reduce
kernel is mode-agnostic.

The whitelist `fa_batched_ok` in `qwen35.rs` enables batched prefill when:
1. `kv_cache ∈ {q8, asym4, asym3, asym2}`, AND
2. Every FA layer's weights pass `is_batchable_la` (MQ4/MQ6/HFQ4/HFQ6)

Everything else falls back to per-token gather/scatter, which is correct but slow.

## When to pick each mode

Default is `asym3` on every RDNA3+ card, `asym2` on tight 8 GB cards. Override
via `hipfire config set kv_cache <mode>` or the TUI.

| Mode | Best for |
|---|---|
| `q8` | Numerical-parity debugging; short-context workloads where asym3's extra compute isn't worth it |
| `asym4` | "Just a bit more headroom than q8" — rarely the right pick |
| `asym3` | **Default**. Best compression/quality/speed tradeoff, recall-safe |
| `asym2` | 8 GB cards (gfx1010, gfx1032). Slightly more quality loss; still passes common multi-turn tests |

## Measured decode tok/s (9B Qwen 3.5, gfx1030, ctx=4096)

| KV mode | Decode | Compression |
|---|---:|:---:|
| q8 | baseline | 3.76× |
| asym4 | 116 | 5.1× |
| **asym3** | **120** | **5.5×** |
| asym2 | 116 | 6.0× |

asym3 is slightly *faster* than q8 because the rotated-K read path is smaller
(3 bits/element vs 8 bits) and the compute overhead of centroid lookup is
absorbed into the already-launch-bound dispatch.

## Multi-turn recall verification

The release gate for any KV change is the "Kaden" test:

```
user:  "My name is Kaden."
asst:  "Hello Kaden!"
user:  "What is my name? One word."
```

Expected: `Kaden`. Observed on Qwen 3.5 9B:

| KV mode | Answer |
|---|---|
| q8 | Kaden ✓ |
| givens4 (v0.1.4, retired) | **Kendall** ✗ — K head_dim=256 coverage bug |
| asym4 | Kaden ✓ |
| asym3 | Kaden ✓ |
| asym2 | Kaden ✓ (sometimes "The name is Kaden.") |

The test is part of the quality gate in v0.1.5+.

## Pointers to implementation

- Rust — `crates/engine/src/llama.rs` (KvCache constructors)
- Rust — `crates/engine/src/qwen35.rs` (`fa_batched_ok`, flash dispatch)
- Dispatch — `crates/rdna-compute/src/dispatch.rs` (`attention_flash_asym{2,3,4}_batched`, `kv_cache_write_asym{4,3,2}_{fused,batched}`)
- Kernels — `kernels/src/attention_flash_asym{2,3,4}_tile{,_batched}.hip` + `attention_flash_asym_reduce_batched.hip`
- Kernels — `kernels/src/kv_cache_write_asym_k_givens{2,3,4}{,_batched}.hip` (K write kernels — historical naming preserved from the rotation code)
- Centroid tables — `kernels/src/turbo_common.h` (`TURBO_C2`, `TURBO_C3_256`, `TURBO_C4`)

## Related

- [QUANTIZATION.md](QUANTIZATION.md) — MQ4/MQ6 weight format (the other side of the compression story)
- [LEGACY_HFQ.md](LEGACY_HFQ.md) — retired HF4/HF6 format (still loads but not produced)
