# Ragged-K Group-Quant Kernel Bug — Analysis & Fix Plan

**Date:** 2026-06-08 (rev 2026-06-09: corrected root-cause model)
**Status:** Open
**Severity:** Silent correctness — near-total corruption of any group-quantized
weight whose `K % group_size ≠ 0` (NOT a bounded "% dropped"; see below)
**Affected models:** Gemma 4 26B-A4B today; latent in every model with a
non-group-aligned `K` on a per-row-stride GEMV/GEMM kernel

> This file supersedes the original analysis and the `*_plan_rev_ds4.md`
> adversarial review. Both earlier documents shared the same incorrect
> root-cause model (per-row `ceil`-packing). The corrected model is below.

## Corrected Root Cause

Two layers disagree about how grouped weights are laid out in memory.

**The quantizer packs the WHOLE tensor as one continuous group stream — it
does NOT pad per row.** `quantize_hfq4g128()` (`main.rs:2983-2984`):

```rust
let n = f32_data.len();                 // entire [M,K] tensor (or whole expert slice)
let n_blocks = (n + group_size - 1) / group_size;   // ceil over M*K, not per-row
```

The canonical CPU dequant agrees — `dequant_hfq4()` (`dots_ocr.rs:670-693`)
walks `n_groups = n_elements.div_ceil(group_size)` continuously with
`base = g * group_size`, no per-row boundaries. Padding (tail nibbles →
zero-point) happens **once, at the very end of the tensor**.

**The kernel assumes per-row groups with a per-row stride** computed by
truncating division (`gemv_hfq4g128.hip:18-20`):

```c
const int groups_per_row = K / 128;            // floor
const int row_bytes = groups_per_row * 72;
const char* row_ptr = A + (long long)row * row_bytes;
```

These two layouts coincide **only when `K % group_size == 0`**. When `K` is
ragged:

- Groups straddle output rows. For K=704, group 5 holds flat elements
  640–767 — row 0's tail (640–703) **and** row 1's head (704–767) — under a
  single `(scale, zero)` pair.
- The kernel's `row_ptr = A + row * floor(K/g) * 72` is too short by
  `(ceil−floor)*72` bytes, so the stride error **accumulates every row**.
  Row 0 reads real data; every subsequent row reads progressively
  misaligned bytes.

So this is **not "9.1 % of data dropped."** It is whole-row corruption for
all rows > 0. The Session-24 oracle measured **3.7× error** on expert
down_proj — consistent with stride misalignment, and flatly inconsistent
with a 9 % data-loss theory. The earlier "% lost" table was an artifact of
the wrong (per-row `ceil`-packing) model and has been removed.

The runtime byte allocator truncates too — `profile.rs:157` /
`profile.rs:188` use `k / 128` — so it under-sizes the weight buffer
relative to the (larger) continuous on-disk blob. The kernel reads a prefix
of that larger blob, which is why there is no OOB crash, just silent
garbage.

## Scope — systematic across all group formats

The whole-tensor packing is shared by every group quantizer
(`quantize_hfq4g256` `main.rs:960-961`, `quantize_mq4g256` `main.rs:822-823`,
etc.), and the per-row-stride assumption is shared by their GEMV/GEMM
kernels. So the bug class is **systematic**, not specific to the 4 G128
files. It is *latent* only because shipping models (Qwen3.5/3.6, LLaMA-arch)
have group-aligned intermediate/hidden dims; Gemma 4's
`moe_intermediate=704` / `dense_intermediate=2112` are the first to trip it.

A loose grep (`/128`,`/256` …) matches ~300 `.hip` files, but most are
tile/seq/head-dim chunking. The genuinely affected set is the *format
GEMV/GEMM kernels reachable by a ragged-K weight* — dozens, arch-gated, not
"~75". Don't trust a raw file count; trust "does the quantizer pack
whole-tensor AND does the kernel stride per-row" (both true → affected).

### Affected Gemma 4 26B-A4B dimensions

| Weight tensor | K | K % 128 | K % 256 |
|---|---|---|---|
| expert `down_proj` | 704 | 64 | 192 |
| dense FFN `down_proj` | 2112 | 64 | 64 |
| expert/dense `gate_up_proj`, attention | 2816 | 0 | 0 |

Root cause: 704 and 2112 are not multiples of 128 or 256. **Dense FFN
down_proj (K=2112) runs for every token, not just routed experts** — likely
the higher-impact corruption, despite the MoE framing.

### Not affected

- **Q8_0**: 32-byte blocks. 704/32 and 2112/32 are exact → immune. (This is
  why the `--expert-q8` workaround is coherent.)
- **Qwen 3.5-A3B** (1408, 2048) and **Gemma 4 12B dense** (9216, 2304): all
  group-aligned.

## Fix Options

> Hard constraint discovered above: **the current on-disk bytes are
> whole-tensor packed.** Any per-row-stride kernel needs row boundaries to
> align to group boundaries. Therefore **every "no re-quant" proposal is
> invalid** — including the original "just change `K/128→ceil` + guard x",
> the x-padding variant, and the groups_per_row-metadata variant. All of
> them read bytes that don't exist as per-row groups.

### Option A — Per-row re-quantization + kernel `ceil` (recommended core)

1. **Quantizer → per-row grouping.** Group within each row, never across
   rows. For the tail group, compute `(scale, min)` over **real elements
   only** and pad the trailing nibbles to the zero-point. This is exactly
   the technique the existing partial-group path already uses
   (`main.rs:2989-2992, 3007-3016`) — applied per row instead of once at
   tensor end. **This changes the on-disk layout → re-quantization is
   mandatory.** No range expansion, no quality loss (see "Rejected: naive
   zero-pad" below).
2. **Kernels:** `groups_per_row = (K + g - 1) / g`; row stride
   `groups_per_row * block`. Tail group: **zero the contribution for
   out-of-range lanes — do NOT clamp the index.** Padded nibble dequants to
   `min_val` (= the stored zero-point, `scale*0 + zero`), so the correct
   neutralizer is multiplying by a zeroed activation, not `min_val·x[K-1]`
   (a clamp would inject systematic bias that passes smoke tests).
   - Plain loop (`gemv_hfq4g128.hip`): trivial.
   - Quad-unrolled ParoQuant variants
     (`...down_k8_indexed_batched.hip:47-134` etc.): the guard goes in the
     `TAIL_DOG`/tail blocks, ~15–25 lines per variant — **not** "~5 lines."
3. **Buffer sizing:** fix `profile.rs:157` and `:188` (`k / group`) to
   `ceil`, or they under-size the re-quantized weight → OOB device read.

**Rotated formats (MQ\*) caveat:** FWHT rotation operates on fixed
group-size segments (`signs1`/`signs2`). A ragged tail changes the rotation
basis, so per-row padding must extend to the rotation-segment boundary and
re-apply the rotation on the padded segment — a stride change alone is
insufficient. The quantizer already refuses rotated formats on non-aligned K
and falls back (`main.rs:6646-6649`); keep that fallback until a rotated
per-row path is implemented.

**Pros:** correct for all current/future models; permanent root fix.
**Cons:** requires re-quantizing affected artifacts; weight files grow by
the per-row tail padding (≤ `g-1` extra nibbles per row).

### Option B — Robustness belt: store `groups_per_row` in metadata

Add `groups_per_row: u32` (= per-row `ceil(K/g)`) to per-tensor metadata and
pass it to kernels instead of recomputing. **This is NOT a standalone fix.**
It is only well-defined once Option A's per-row padding makes rows align to
group boundaries; on whole-tensor-packed bytes there is no integer
`groups_per_row` and it reads garbage exactly like the broken path. Adopt it
*with* Option A as a readability/robustness guard against future
recomputation drift — never as a substitute for re-quant.

### Option C — Validation gate (all formats, quantize-time + load-time)

Generalize beyond HFQ4G128: for any group-quantized tensor, check
`K % group_size == 0`.

- **Quantize time (preferred UX):** when a weight is ragged, auto-fall back
  to Q8 for that tensor and emit a notice — the user gets a working model.
  `--expert-q8` (`f9f0c574`) does this for experts but not dense down_proj
  or arbitrary future shapes.
- **Load time:** refuse with a clear error if a ragged group-quant tensor is
  encountered, so corruption can never be silent again.

Requires a `DType::group_size()` accessor (does not exist yet — must be
added). Good defense-in-depth regardless of A.

### Option D — Q8 fallback (current shipped workaround)

`--expert-q8` forces Q8_0 for expert weights; Q8's 32-byte block divides all
Gemma 4 dims. **Verified coherent (oracle argmax match), already shipped.**
Interim only: the all-Q8 26B model is ~27.5 GB and **does not fit the 24 GB
gfx1100 (k9lin) — the primary deploy target.** Viable solely on the ≥96 GB
boxes (hipx/hiptrx). A fixed HFQ4 path (~16 GB) is what unblocks the primary
target, so D is not a substitute for A.

### Rejected approaches

- **"No re-quant" kernel-only fixes** (original Option A, x-padding,
  metadata-only): invalid — on-disk bytes are whole-tensor packed.
- **Naive Option B "pad f32 rows with 0.0 then quantize"**: expands the tail
  group's min/max to include 0.0, wasting quantization levels on dead values
  (a real per-group precision regression). Avoided entirely by Option A's
  per-row "stats over real elements, pad nibbles to zero-point."
- **Whole-tensor-aware kernel** (consume the existing continuous layout):
  rejected — row-straddling groups can't map to a workgroup-per-row GEMV
  cleanly; per-row padding is the pragmatic path.

## Recommendation

Ship **A (per-row re-quant + kernel `ceil` + `profile.rs` sizing)** as the
permanent fix, **B** folded in as a metadata robustness guard, and **C** as a
quantize-time + load-time gate so this class of bug can never be silent
again. Keep **D** as the interim path for ≥96 GB boxes until A lands.

## Historical Context

- Bug discovered Session 24 (2026-06-07) via per-layer HF oracle comparison.
- Expert down_proj: 3.7× error (HFQ4G128). Expert gate_up: 20–35 % (MQ4G256
  on small weights). Dense FFN down_proj (K=2112) also affected.
- Workaround: all-Q8 quantization (`--expert-q8`, `f9f0c574`); verified
  coherent via oracle argmax=818 match.
- 2026-06-09: root-cause model corrected from "per-row `ceil`-packing,
  last group dropped" to "whole-tensor continuous packing vs. per-row stride"
  after verifying the quantizer, CPU dequant, and `profile.rs` against the
  measured 3.7× error.
