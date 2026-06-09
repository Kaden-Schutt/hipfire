# Adversarial review: HFQ4G128 ragged-kernel fix options

**Review of:** `findings/gemma4_ragged_kernel.md` (2026-06-08)  
**Reviewer:** DS4 agent, adversarial pass  
**Date:** 2026-06-08

## Summary

The original document correctly identifies the root cause (`K/128`
truncating division in `gemv_hfq4g128.hip` line 18) and the affected
dimensions. However, it understates the kernel-change complexity, omits
a quantize-time defense option, and understates the scope of the bug
(which affects **~75 group-quantized kernel files**, not just 4).

The recommended approach is still **fix the kernels** (ceiling division
+ x-padding in callers), but the mechanics differ from what the document
describes.

---

## Flaw 1 — Option A understates kernel-change complexity

The document says "~5 lines each" and proposes an OOB guard on x[]
reads:

```c
float load_x(int idx, int K, const float* x) {
    return (idx < K) ? x[idx] : 0.0f;
}
```

This doesn't address the **quad-unrolled loop structure** in the 3
ParoQuant indexed variants (`gemv_paro_q4g128_moe_gate_up_indexed.hip`,
etc.). Those kernels unroll groups into quads-of-4 with tail handling:

```c
const int quads = groups_per_row >> 2;   // groups / 4
const int tail = groups_per_row & 3;     // groups % 4
```

For K=704, `ceil(704/128)=6`, `quads=1`, `tail=2`. Groups 0-3 go
through the quad loop (all full — no OOB issue). Groups 4-5 are handled
by the `if (tail >= 1)` / `if (tail >= 2)` tail blocks. Group 5 is the
partial group (real elements 640-703, padded 704-767). Threads 16-31
(tid×4 = 64-124) in that group have all-4-elements past-end.

The guarding must go into the `TAIL_DOG` macro or each tail block.
That's not "~5 lines" — it requires either:

**(a)** A guarded variant of the macro (doubles the macro surface):  
```c
#define TAIL_DOG_GUARDED(b0, b1, sc, zp, base_idx, a, K) \
    (a) += (idx< K ? (sc*(float)((b0)&0xFu)+zp)*x[(base_idx)] : 0.0f) + ...
```
plus conditionals to select the right macro per tail block.

**(b)** A `bool partial = ((K % 128) != 0)` and branching inside each
tail block to pick guarded vs. unguarded loads.

Either way, **15-25 lines per variant, not 5**. The simple
`gemv_hfq4g128.hip` (plain for-loop) is the easy case.

### Simpler alternative not considered: x-padding in caller

A cleaner approach is:

1. **Kernel:** change `K/128` → `(K+127)/128`. **No loop body changes.**
   No guards, no macro variants, no tail modifications.

2. **Rust caller:** zero-pad the x vector to `ceil(K/128)*128` before
   dispatch. For single-token decode, this is ≤508 bytes (127 floats) —
   negligible. For batched variants, each row of the activation batch
   gets padded.

3. **Why this works:** padded nibbles dequantize to `zero = min_val`
   (not 0.0f — see Flaw 2). Padded x positions are exactly `0.0f`.
   Contribution: `min_val * 0.0f = 0.0f`. ✓

The kernel diff reduces to literally one line per file: `s/K \/ 128/(K + 127) \/ 128/`.
No loop nesting changes, no macro variants, no branch divergence concerns.

The Rust-side change is 2-3 lines per call site: compute `padded_k =
(k + 127) / 128 * 128`, zero-pad x.

**Recommendation:** use x-padding, not OOB guards. It's mechanically
simpler across all kernel variants and has zero GPU-side branch cost.

---

## Flaw 2 — Option A misstates padded nibble dequantization

The document says:

> Padded nibbles are `0` → dequant to `zero` (= `min_val`). Guarded x
> reads return `0.0` for OOB indices. Contribution of padded elements:
> `min_val × 0.0 = 0.0`.

The first sentence is correct — but dequant of nibble 0 is `zero`, and
`zero` is `min_val`, **not 0.0**. The document then uses `0.0` for x
(guarded) which makes `min_val × 0.0 = 0.0`. This is arithmetically
correct but conceptually conflates two different zeros:

| Source | Value | Reason |
|--------|-------|--------|
| Padded weight nibble → dequant | `min_val` | Nibble 0 → scale×0 + zero = zero = min_val |
| Guarded/OOB x read | `0.0f` | Branch returns 0.0 for idx ≥ K |
| Product | `0.0f` | Anything × 0.0 = 0.0 |

This matters because if someone were to implement protection via a
saturating clamp (`idx = min(idx, K-1)`) instead of zeroing, they'd get
`min_val × x[K-1]` — a systematic positive bias that passes most tests
but silently corrupts that column. The document should explicitly say:
**zero the contribution, do not clamp the index**.

---

## Flaw 3 — Option B silently degrades model quality

> Pad f32 weight rows to next_multiple_of_128(K) with zeros before quantizing.

Zero-padding weights with 0.0f changes the group min/max computation for
the last group. If real weights are e.g. [0.3, 0.8], padding with 0.0
extends the range from [0.3, 0.8] to [0.0, 0.8], consuming an extra
(0.3 / 15.0) of the 4-bit quantization range on dead values. Every
*real* weight in that group loses ~0.02 effective bits of precision.

For the Gemma4 expert down_proj (K=704, group=128), the 6th group has 64
real weights. If their range is e.g. [0.02, 0.35] and zero-pad extends
min to 0.0, the resolution loss is `(0.02 / (0.35-0.0)) / (0.02 /
(0.35-0.02))` ≈ 6% worse quantization error for 64 weights — on top of
the 4-bit quantization error already present. This is a per-group
quality regression, not a "mathematically clean" solution as the
document claims.

**Verdict:** avoid Option B. Quantizing zero-padded rows introduces a
quality regression with no compensating benefit over Option A.

---

## Flaw 4 — Scope is ~75 kernel files, not 4

The document lists only the 4 G128 ParoQuant variant files. But the grep
in the codebase reveals **every group-quantized kernel** — HFQ4G256,
MQ4G256, MQ3G256, HFQ3G256, HFQ2G128, HFQ6G256, etc. — uses the same
`K / group_size` truncating pattern. That's ~75 HIP files.

| Family | Group size | File count (approx) | Affected by K%group_size≠0? |
|--------|-----------|---------------------|---------------------------|
| HFQ4G128 | 128 | 4 | ✅ (Gemma4 down_proj) |
| ParoQ4G128 | 128 | 3 | ✅ (same shapes) |
| HFQ4G256 | 256 | ~40 | ✅ (Gemma4 down_proj 704%256=192, 2112%256=64) |
| MQ4G256 | 256 | ~8 | ✅ (ditto) |
| MQ3G256 | 256 | ~4 | ✅ |
| HFQ3G256 | 256 | ~4 | ✅ |
| MQ2G256 | 256 | ~6 | ✅ |
| HFQ6G256 | 256 | ~6 | ✅ |
| HFQ2G128 | 128 | 1 | ✅ |
| HFQ4G1024 | 1024 | 1 | ✅ (K%1024 check) |
| HFQ4G512 | 512 | 1 | ✅ |

**Verdict:** the bug is pervasive and systematic. The fix (whether
ceiling-division or metadata or refusal) should cover ALL group-quantized
kernels, not just G128. The document should be retitled to reflect this
scope.

The good news: for currently-shipping models (Qwen3.5/3.6, LLaMA arch),
all intermediate/hidden dimensions happen to be multiples of common
group sizes, so the bug is latent. It only manifests with Gemma4's
unusual `moe_intermediate=704`. But it **will** manifest again for any
future architecture with non-aligned dimensions.

---

## Flaw 5 — Option C is a valid defense-in-depth but too narrow

> Add a validation gate: when loading a model, check if any HFQ4G128
> tensor has K % 128 ≠ 0.

This check should cover **all group-quantized formats**, not just
HFQ4G128. The gate should be keyed off the format's group size, which is
already known at load time:

```rust
fn validate_group_alignment(dtype: DType, k: usize) -> Result<()> {
    let g = dtype.group_size();  // 128 for HFQ4G128, 256 for MQ4G256, etc.
    if g > 0 && k % g != 0 {
        return Err(format!("K={k} not multiple of group_size={g} for {dtype:?}"));
    }
    Ok(())
}
```

Additionally: this check should happen at **quantize time**, not just
load time. The quantizer should refuse to produce group-quantized
weights for K%group_size≠0 and fall back to Q8 (which has no group
constraint). This is better UX than a load-time refusal — the user gets
a working model with Q8 for the non-aligned tensors.

The `--expert-q8` quantizer flag (commit `f9f0c574`) partially
implements this for expert weights, but doesn't cover dense FFN
down_proj (K=2112, 2112%128=64, 2112%256=64) or arbitrary future shapes.

---

## Flaw 6 — Missing option: store groups_per_row in model metadata

The quantizer already knows the exact group count (`n_blocks`). Storing
it per-tensor in the model metadata would eliminate the computation
error entirely:

```rust
struct TensorMeta {
    dtype: DType,
    shape: [usize; 2],  // [M, K]
    groups_per_row: u32,  // = ceil(K / group_size), stored at quantize time
    ...
}
```

The kernel receives `groups_per_row` as a parameter instead of deriving
it from `K / group_size`. This:
- Eliminates the bug at the root (no computation, no error)
- Works for any group size, any K
- No ceiling division, no x-padding, no OOB guards
- Backward-compatible: add field to metadata, kernels that ignore it use
  the old K/group_size formula (still buggy, but a per-kernel rollout)

**Verdict:** this is the cleanest long-term fix and should be adopted as
the canonical approach. Existing kernels can be mechanically patched
(ceiling division) as the fast path.

---

## Updated Recommendation

**Phase 1 (immediate, fix the bug):** Ceiling-division + x-padding.

- Change `K / group_size` → `(K + group_size - 1) / group_size` in all
  group-quantized GEMV kernels (~75 files, mechanical sed).
- For each Rust call site, zero-pad x to `ceil(K/group_size) * group_size`.
  This is 2-3 lines per GPU method, concentrated in the `gemv.rs` dispatch.
- This fixes correctness for all current and future models with zero
  kernel loop-body changes.

**Phase 2 (load-time defense):** Group-alignment validation gate.

- At model load, for every group-quantized tensor, assert
  `K % dtype.group_size() == 0`. Refuse to load with a clear error if
  the kernel fix hasn't been deployed to all kernels yet. Once Phase 1
  covers all kernels, this gate becomes a defense-in-depth check.

**Phase 3 (quantize-time defense):** Auto-fallback in quantizer.

- In the quantizer, when a weight tensor has `K % group_size ≠ 0`,
  auto-fallback to Q8 for that tensor instead of silently producing
  a ragged group. This is better UX than load-time refusal.

**Phase 4 (long-term):** Store `groups_per_row` in model metadata.

- Add `groups_per_row: u32` to the per-tensor metadata. Kernels receive
  it as a parameter, eliminating the `K / group_size` computation
  entirely. Roll out per-kernel-family. Backward-compatible (old kernels
  ignore the metadata field; old models without the field use ceiling
  division).

---

## Items the original document got right

- Root cause identification: `K / 128` truncating division, quantizer
  writes `ceil(K/128)` groups. ✓
- Affected dimensions correctly identified (704, 2112). ✓
- Recognition that Q8_0 (32-byte blocks) is immune. ✓
- Recommendation of Option A (kernel fix) + Option C (validation gate)
  as defense-in-depth. ✓ (though the mechanics need refinement)
- Option D accurately characterized as a workaround, not a fix. ✓

---

## Items the original document missed

1. The quad-unrolled loop structure in ParoQuant kernels makes OOB
   guarding more invasive than stated.
2. Padded nibble dequant is `min_val`, not `0.0f` — the distinction
   matters for implementation correctness.
3. Option B (zero-pad weights) introduces a per-group quality regression
   by altering the last group's min/max range.
4. ~75 kernel files have the same pattern, not just 4.
5. Validation gate should cover all group-quantized formats, not just
   HFQ4G128.
6. Quantize-time auto-fallback to Q8 is better UX than load-time refusal.
7. Storing `groups_per_row` in model metadata is the cleanest long-term
   fix and eliminates the class of bug entirely.
8. x-padding in callers is mechanically simpler than OOB guards and
   should be the recommended implementation of Option A.
