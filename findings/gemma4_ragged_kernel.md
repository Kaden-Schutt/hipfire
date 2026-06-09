# HFQ4G128 Ragged-Kernel Bug — Analysis & Options

**Date:** 2026-06-08  
**Status:** Open  
**Severity:** Silent correctness (data corruption) — 9.1% of expert down_proj weight data dropped  
**Affected models:** Gemma 4 26B-A4B (any quant format using HFQ4G128 for down_proj)

## Root Cause

`gemv_hfq4g128.hip` line 18:

```c
const int groups_per_row = K / 128;  // truncating division
```

When `K % 128 ≠ 0`, the quantizer writes `ceil(K/128)` groups (it uses
ceiling division) but the kernel reads only `floor(K/128)` groups. The
last partial group — containing real weight data — is silently skipped.

The same pattern appears in all 4 G128 kernel files:

| File | Line |
|------|------|
| `gemv_hfq4g128.hip` | 18 |
| `gemv_paro_q4g128_moe_down_k8_indexed_batched.hip` | 42 |
| `gemv_paro_q4g128_moe_gate_up_indexed.hip` | 45 |
| `gemv_paro_q4g128_moe_gate_up_k8_indexed_batched.hip` | 53 |

## Affected Dimensions (Gemma 4 26B-A4B)

| Weight tensor | K | `K/128` groups | `ceil(K/128)` groups | Dropped elements | % lost |
|---|---|---|---|---|---|
| expert `down_proj` | 704 | 5 | 6 | 64 | **9.1%** |
| dense FFN `down_proj` | 2112 | 16 | 17 | 64 | **3.0%** |
| expert `gate_up_proj` | 2816 | 22 | 22 | 0 | — |
| dense `gate_up_proj` | 2816 | 22 | 22 | 0 | — |
| attention (all) | 2816 | 22 | 22 | 0 | — |

Root cause is Gemma 4's `moe_intermediate=704` and
`dense_intermediate=2112` — neither is a multiple of 128.

## Not Affected

- **Q8_0** kernels: 32-byte block size. 704/32=22, 2112/32=66 — exact.
- **MQ4G256** kernels: 256-byte groups. 2816/256=11 — exact. But
  704%256=192, 2112%256=64 — would be ragged if MQ4G256 were used for
  down_proj (same class of bug, different group size).
- **Qwen 3.5-A3B**: all intermediate sizes (1408, 2048) are multiples of
  128. Not affected.
- **Gemma 4 12B dense**: intermediate=9216, hidden=2304 — both multiples
  of 128. Not affected.

## Quantizer Behavior (Correct)

The quantizer in `quantize_hfq4g128()` does ceiling division and pads
the last partial group with `min_val` (the zero-point):

```rust
let n_blocks = (n + group_size - 1) / group_size;  // ceiling
// Padded elements → min_val → nibble 0 → dequant to (scale*0 + min_val) = min_val
```

So the weight file contains all data. The kernel just doesn't read it.

## Options

### Option A — Fix the kernel (recommended)

Change `K / 128` → `(K + 127) / 128` and add a bounds guard on `x[]`
reads for the last partial group.

For the last group, indices `g*128 + tid*4 + {0,1,2,3}` may exceed K.
Guard each x load: `x[min(idx, K-1)]` is wrong (duplicates the last
element). Instead, zero out-of-bounds loads:

```c
float load_x(int idx, int K, const float* x) {
    return (idx < K) ? x[idx] : 0.0f;
}
```

Padded nibbles dequant to `min_val`. Guarded x loads return `0.0` for
OOB indices. Contribution of padded elements: `min_val × 0.0 = 0.0`. ✓
Real elements: correctly dequantized × real x. ✓

**Pros:**
- Fixes all G128 kernels for all future models
- Minimal performance impact (one branch per element in the last group only)
- 4 kernel files to patch, ~5 lines each
- Weight format unchanged — no re-quantization needed

**Cons:**
- None meaningful

**Estimated change:** ~20 lines across 4 kernel files + recompile.

### Option B — Pad weights at quantize time + pass padded K

Pad f32 weight rows to `next_multiple_of_128(K)` with zeros before
quantizing. Store `padded_K` as the kernel's K parameter so `K/128`
(truncating) still computes the correct group count.

**Pros:**
- No kernel changes
- Mathematically clean: padded weights are zero, so padded × zero-padded-x = 0

**Cons:**
- Weight files slightly larger (up to 127 extra f32 → 127 extra nibbles per row)
- Must track `padded_K` separately from real K in the model metadata
- All callers (load path, GEMV dispatch) need to pass `padded_K`
- More invasive than Option A across the stack

### Option C — Refuse at load time

Add a validation gate: when loading a model, check if any HFQ4G128
tensor has `K % 128 ≠ 0`. Refuse to load with a clear error message
suggesting re-quantization with `--expert-q8` or using MQ4G256 instead.

**Pros:**
- Minimal code (~10 lines in the load path)
- Prevents silent corruption for future models
- Good defense-in-depth regardless of which other option is chosen

**Cons:**
- Doesn't fix the problem — just makes it loud
- Forces larger weight files (Q8 is ~33 bpw vs HFQ4G128's ~4.5 bpw)
- 26B model balloons from ~16 GB to 27.5 GB (current all-Q8 workaround)

### Option D — Q8 fallback for ragged-K weights (current workaround)

Already implemented via `--expert-q8` quantizer flag. Forces Q8_0 for
all expert weights. Q8's 32-byte block size divides all Gemma 4
dimensions exactly.

**Pros:**
- Already working — 26B-A4B all-Q8 model produces coherent output
- Q8 has higher per-element fidelity than HFQ4G128 (8-bit vs 4-bit)
- No kernel changes needed

**Cons:**
- 27.5 GB model vs ~16 GB with a fixed HFQ4G128
- For K=704 expert down_proj: Q8 uses `704 × 34/32 = 748 B/row` vs
  HFQ4G128's `704 × 72/128 = 396 B/row` — 1.9× larger
- Doesn't fix the kernel for future models with different architectures
- On 24 GB GPUs (RX 7900 XTX), 27.5 GB doesn't fit; ~16 GB would

## Recommendation

**Option A** — fix the kernels. It's the smallest change, fixes the root
cause permanently for all models, has zero measurable perf impact, and
doesn't require re-quantization or weight format changes.

**Additionally:** add Option C as a defense-in-depth validation gate at
load time, independent of the kernel fix. This catches any future format
that might have similar alignment requirements.

## Historical Context

- Bug discovered Session 24 (2026-06-07) via per-layer HF oracle comparison
- Expert gate_up showed 20–35% error (MQ4G256 quality on small weights)
- Expert down_proj showed 3.7× error (HFQ4G128 dropping 9.1% of data)
- Dense FFN down_proj also affected (same K=2112%128=64)
- Workaround: all-Q8 quantization (`--expert-q8` flag, commit `f9f0c574`)
- All-Q8 model verified coherent via oracle argmax=818 match
