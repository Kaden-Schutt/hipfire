# Review of updated ragged-kernel fix plan (rev 2026-06-09)

**Review of:** `findings/gemma4_ragged_kernel.md` (rev 2026-06-09)  
**Reviewer:** DS4 agent (session 24b)  
**Date:** 2026-06-09

## Verdict

The corrected root-cause model is **right** and materially changes the
fix approach. My earlier analysis (ceiling-division + x-padding, no
re-quant) was wrong because it assumed per-row `ceil`-packing. The
quantizer packs the whole `[M*K]` tensor as a flat group stream; the
kernel strides per-row with `floor(K/g)*block`. When `K%g≠0` these
disagree and the stride error accumulates across rows → whole-row
corruption for rows > 0. This matches the 3.7× oracle error.

I agree with the expanded scope, the priority ordering (A→B→C→D), and
the rejection of no-re-quant approaches. Below are the items I think
need correction or strengthening.

---

## ✅ Things the updated plan gets right

1. **Root cause is whole-tensor vs per-row mismatch.** Verified:
   `quantize_hfq4g128` operates on `f32_data.len()` = M\*K flat slice.
   `dequant_hfq4` in `dots_ocr.rs:670` walks `n_groups =
   n_elements.div_ceil(group_size)` continuously. The kernel strides
   with `floor(K/128)*72`. These disagree for K%128≠0. ✓

2. **Re-quant is mandatory.** You cannot fix this with kernel-only
   changes because the on-disk bytes are whole-tensor packed. No amount
   of ceiling division, x-padding, or metadata helps when the group
   boundaries don't align with row boundaries. ✓

3. **3.7× error is consistent with whole-row corruption, not 9% loss.**
   If only the last group were dropped per row, the error would be
   bounded by the last group's contribution (~9%). The observed 3.7×
   error across ALL expert down_proj output is only possible if later
   rows are reading misaligned scale/zero/nibbles. ✓

4. **MQ\* rotated format caveat.** FWHT rotation operates on
   fixed-size segments; a ragged tail changes the rotation basis.
   Keeping the fallback to Q8 for rotated formats on non-aligned K is
   the right call. ✓

5. **Option C should be both quantize-time and load-time.**
   Quantize-time auto-fallback to Q8 is strictly better UX than
   load-time refusal. ✓

6. **Rejection of whole-tensor-aware kernel.** Trying to make the
   kernel understand the flat layout would require group-boundary
   cross-referencing per row — impractical for a GEMV inner loop. ✓

---

## ⚠️ Items that need correction

### 1. `profile.rs` does NOT under-allocate GPU buffers

The document states:

> The runtime byte allocator truncates too — `profile.rs:157` /
> `profile.rs:188` use `k / 128` — so it under-sizes the weight buffer
> relative to the (larger) continuous on-disk blob.

This is **incorrect.** The `profile.rs` functions
(`hfq4g128_weight_bytes`, `gemv_hfq4g128_bytes`, etc.) are used **only
for performance counter byte-counting** (`begin_timer` in `gemv.rs`).
They estimate the bytes transferred so the profiler can report
bandwidth. They are never used for GPU allocation.

The actual allocation path is:

```rust
// crates/hipfire-runtime/src/hfq.rs:633 (and gemma4.rs:553)
let buf = gpu.upload_raw(data, &[data.len()])?;
```

`upload_raw` (`dispatch.rs:1176`) does `self.hip.malloc(data.len())`
— the full on-disk blob size. No truncation. The GPU buffer contains
the entire whole-tensor-packed weight data.

**Impact on the plan:** Option A step 3 says "fix `profile.rs:157`
and `:188` (`k / group`) to `ceil`, or they under-size the
re-quantized weight → OOB device read." After re-quantization to
per-row packing, the kernel's per-row stride will be correct, and the
GPU buffer will still be fully uploaded. The `profile.rs` fix is a
nice-to-have for accurate bandwidth reporting but is NOT a correctness
requirement. It should be reclassified as a cleanup, not a ship-blocker.

### 2. CPU dequant path must also be updated

The plan's Option A focuses on the quantizer and GPU kernels but
doesn't mention the CPU dequantization paths. After per-row
re-quantization, the on-disk layout changes from whole-tensor-flat to
per-row-grouped. Any code that reads the weight data using the old
flat-layout assumption will break.

Known CPU consumers:

- `dequant_hfq4()` in `crates/hipfire-arch-dots-ocr/src/dots_ocr.rs:670`
  — walks `n_groups = n_elements.div_ceil(group_size)` continuously.
  **Will produce wrong results** after per-row re-quant unless updated
  to stride per-row.

- `repack_awq_to_hfq4g128()` in `crates/hipfire-runtime/src/hfq.rs:930`
  — this one IS already per-row (`groups_per_row = in_dim / group_size`,
  `bytes_per_row = groups_per_row * 72`). **Already correct** after
  re-quant (assuming truncating division is also fixed to ceil for
  ragged K). But note: this function uses truncating division for
  `groups_per_row` — same class of bug if K%128≠0 and the input is AWQ.
  Not urgent (AWQ weights tend to have aligned K) but worth noting.

- The `load_gemma4_weight` / `load_weight_tensor` functions — these
  upload raw bytes without interpreting the layout, so they're layout-
  agnostic. No change needed.

**Recommendation:** Add a step to Option A: audit and update all CPU
dequant paths to use per-row stride = `ceil(K/g) * block_bytes`.

### 3. Format version bump or layout flag needed

Per-row re-quantization changes the on-disk binary layout. Old models
(whole-tensor-packed) will be misinterpreted by new code (per-row-
stride), and vice versa. The plan doesn't address backward
compatibility.

Options:

- **Bump `HFQ_VERSION`** from 1 to 2. Old code refuses v2 models with
  "unsupported format version." New code reads v2 with per-row stride.
  Old v1 models are still loadable by new code (detect version, use
  flat stride). This is the safest approach.

- **Add a per-tensor layout flag** (whole-tensor vs per-row) in the
  metadata. More granular but more complex.

**Recommendation:** bump `HFQ_VERSION` to 2 and handle v1/v2 in the
load path. Old models (v1) continue to work (using the old stride,
which is correct for their layout since all v1 dims were aligned). New
models with ragged K get v2 layout and per-row stride.

### 4. Weight file size change is understated

The plan says "weight files grow by ≤ g-1 extra nibbles per row." The
actual growth is per-row tail groups: for K=704 with g=128, each row
gains 1 full group (72 bytes) — not just a few nibbles.

For the 26B-A4B expert down_proj per layer:
- 128 experts × M=2816 rows × ceil(704/128)=6 groups × 72 B = 155.6 MB
- Old: 128 × 2816 × ceil(2816×704/128)×72 / 2816 ≈ 128 × 1115136 = 142.7 MB
- Delta: ~13 MB per layer's expert down_proj

Over 30 layers: ~390 MB total growth across all expert down_proj
tensors. For dense down_proj: ~3.4 MB per layer, ~102 MB over 30
layers. Total model growth: ~500 MB (from ~15.6 GB to ~16.1 GB). Small
but worth stating precisely.

---

## ⚠️ Items that need strengthening

### 5. The quantizer change should be a separate function, not in-place

The plan says to change the quantizer to pack per-row. Rather than
modifying the existing `quantize_hfq4g128` in-place (which would break
the CPU dequant for dots_ocr and any other consumer), introduce a new
function like `quantize_hfq4g128_per_row(f32_data: &[f32], m: usize,
k: usize) -> Vec<u8>` that takes shape information and groups within
rows. Keep the old function for backward compat (or gate on version).

### 6. The "dense FFN down_proj runs for every token" observation deserves more emphasis

The plan correctly notes:

> Dense FFN down_proj (K=2112) runs for every token, not just routed
> experts — likely the higher-impact corruption, despite the MoE framing.

This is an important insight. The dense FFN is on ALL 30 layers and
processes EVERY token. If it's corrupted, the entire model output is
garbage — not just the MoE branch. The expert down_proj only activates
for routed tokens, so its corruption is partially masked by the dense
FFN path. The dense FFN down_proj corruption (K=2112, 2112%128=64) is
the PRIMARY source of model-level garbage, not the expert weights.

This should be reflected in the priority: if we fix only the quantizer
and re-quantize only dense FFN down_proj to Q8, we might already get a
coherent model even with broken expert HFQ4G128. (The all-Q8 model's
coherence confirms this — Q8 dense FFN + Q8 experts works.)

### 7. Quad-unrolled kernel tail handling complexity is real but manageable

The plan acknowledges "~15–25 lines per variant, not '~5 lines'" for
the ParoQuant indexed kernels. This is accurate. However, it should
also note that after per-row re-quantization, the kernel's ceiling
division + OOB guard is still needed because the per-row tail group
exists (it's just properly padded now). The tail group has padded
nibbles (→ min_val) and the kernel must zero the contribution of those
elements. The x-padding approach (zero-pad activation to
`ceil(K/g)*g`) is still the cleanest kernel-side implementation — one
line per kernel, no tail block modifications.

---

## Items I now retract from my earlier adversarial review

- **"Scope is ~75 kernel files"** — I stand by the number but the
  updated plan correctly distinguishes "raw file count" from "genuinely
  affected reachable kernels." The actual affected set depends on which
  formats reach which weights with ragged K. Not all 75 files will fire
  for any given model.

- **"x-padding in caller is simplest"** — still true mechanically
  (one-line kernel change), but now that re-quant is mandatory anyway,
  the x-padding benefit is smaller relative to the total change. The
  OOB guard approach is equally valid in the context of a full re-quant.

---

## Revised priority list

1. **Quantizer:** new `quantize_hfq4g128_per_row` function. Gate on
   `HFQ_VERSION=2`. Handle the per-row tail group correctly (stats over
   real elements only, pad nibbles to zero-point).

2. **HFQ_VERSION bump:** v2 = per-row layout. v1 loader unchanged.
   New code handles both.

3. **Re-quantize 26B-A4B** with the new per-row quantizer for
   down_proj tensors (dense K=2112, expert K=704).

4. **Kernel ceiling division + OOB handling** in the GEMV/GEMM kernels
   that will touch the re-quantized weights. For Gemma4 26B:
   `gemv_hfq4g128.hip` and the 3 ParoQuant indexed variants.

5. **CPU dequant audit:** update `dequant_hfq4` and any other flat-
   layout consumer to use per-row stride.

6. **profile.rs cleanup:** fix bandwidth estimation (truncating → ceil).
   Not correctness-critical.

7. **Load-time validation gate** (Option C): refuse ragged-K group-
   quant tensors that haven't been re-quantized. Defense-in-depth.

8. **Quantize-time auto-fallback:** when K%g≠0, auto-select Q8 for that
   tensor instead of producing a potentially-broken HFQ4.
