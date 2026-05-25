# `gemm_hfq6g256` — non-residual batched MQ6 / HFQ6 GEMM

**Branch:** `feat/mq6-gemm-nonresidual`.
**Targets:** every arch that already ships an HFQ6 residual variant — gfx906, gfx1010–1012, gfx1030–1036, gfx1100–1102, gfx1150–1152. gfx1200/1201 explicitly **out of scope** for this plan (see §gfx12).
**Motivation:** kill the memset+residual workaround in the mixed-format batched prefill fallback (issue #249) and remove the active-stream ordering hazard introduced by it.

*This plan folds three adversarial reviews:* Claude self-review, GLM5 (`mq6_gemm_plan_rev_glm5.md`), Gemini (`mq6_gemm_plan_rev_gemini.md`). Original plan claimed "Phase 1 is on par with memset+residual perf"; reviewers found that wrong (Phase 1 was a perf regression on gfx11 as drafted), the KLD tolerance contradicted the byte-exact criterion, the LOC estimate was ~2× off, and three call sites + a CPU-reference test infra were missed. Plan rewritten below; rev files deleted after fold-in.

## Why (with the do-nothing baseline)

After issue #249's fix, the mixed-format fallback in
`crates/hipfire-arch-qwen35/src/qwen35.rs::batched_gemm_single_weight`
dispatches the MQ6/HFQ6 arm as:

```rust
gpu.hip.memset_async(&y.buf, 0, bytes, stream)?;     // launch 1
gpu.gemm_hfq6g256_residual(&w.buf, x, y, m, k, n)?;  // launch 2, Y += A·X
```

Two costs vs an ideal `Y = A·X` kernel:

1. **2× Y bandwidth on the wv projection** — memset writes zeros, residual GEMM reads them back, then writes the result.
2. **+1 launch** of the memset itself.

The third cost is **correctness fragility**: the memset must be ordered on `active_stream` (commit `dad04563` learned this the hard way — the original sync `hipMemset` on the null stream raced against active-stream kernels and silently produced NaN logits). Any future caller of `batched_gemm_single_weight` who forgets the stream rule re-introduces the bug.

**The same `memset_async + gemm_hfq6g256_residual` pattern** lives at `dispatch.rs:10240` in **`gemm_hfq6g256_batched_lmhead`** (the DFlash draft-side lmhead). That call site retains the active-stream hazard until non-residual ships (GLM5 §2.1).

### Profile-first gate (Phase 0, hard go/no-go)

Per GLM5 §4.1: the 1-2% prefill estimate in the original draft is analytical, not measured. The memset may pipeline with the prior kernel's tail, and launch overhead may be absorbed by the next kernel's dispatch latency. **Before writing kernels, measure.**

Phase 0 deliverable: a microbench under `crates/hipfire-runtime/examples/bench_mq6_residual_overhead.rs` that times two paths on the kmd2 wv shape (`m=256, k=4096`) and the w_down shape (`m=4096, k=12288`) at `batch_size ∈ {64, 256, 1024}` on gfx1151:

- **Path A (current):** `memset_async + gemm_hfq6g256_residual` — measured per call, averaged over 1000 iterations after 100 warmup.
- **Path B (emulated non-residual):** the same kernel called against a pre-zeroed Y allocated once at startup — isolates the memset from the GEMM's measured cost.

**Go/no-go**: implementation proceeds only if Path A measures **≥ 1.0% slower** than Path B on at least one shape at `batch_size ≥ 256` on a fresh process (per `docs/methodology/perf-benchmarking.md`). If the gap is <1.0%, the bandwidth win isn't there; the correctness benefit (eliminating the stream hazard) alone may not be worth 4 new kernels — fall back to §"Alternative if Phase 0 says no" below.

## Strategy: template the existing kernel, don't fork it (Gemini §1)

The original plan called for 4 new `.hip` files derived by copy-paste. Gemini correctly flagged this as maintenance debt: 6-bit dequant is subtle (200-byte groups, 4-per-3-bytes packing), and any bug fix would have to be propagated to 2 copies per arch variant. The recipe is:

1. **Extract the K-loop + dequant + accumulation into a `__device__`-templated helper** in each existing residual `.hip` file. The helper produces `val` (the dot-product output). The current kernel function becomes a thin wrapper that does `Y[...] += val`.
2. **Add a sibling `extern "C" __global__` entry point** in the same file that wraps the same helper with `Y[...] = val`.
3. **Both entry points compile into the same `.hsaco` blob.** The dispatch layer picks by kernel name (`"gemm_hfq6g256"` vs `"gemm_hfq6g256_residual"`).

Concrete sketch for `kernels/src/gemm_hfq6g256_residual.hip`:

```cpp
template<bool RESIDUAL>
__device__ __forceinline__ void
gemm_hfq6g256_impl(const char* A, const float* X, float* Y, int M, int K, int batch_size) {
    /* current 200-line body, ending with: */
    if (tid == 0) {
        const long long off = (long long)(batch_start + b) * M + row;
        if constexpr (RESIDUAL) Y[off] += val;
        else                     Y[off]  = val;
    }
}

extern "C" __launch_bounds__(32, 8) __global__ void
gemm_hfq6g256_residual(const char* A, const float* X, float* Y, int M, int K, int N)
{ gemm_hfq6g256_impl<true>(A, X, Y, M, K, N); }

extern "C" __launch_bounds__(32, 8) __global__ void
gemm_hfq6g256(const char* A, const float* X, float* Y, int M, int K, int N)
{ gemm_hfq6g256_impl<false>(A, X, Y, M, K, N); }
```

Same pattern for the 3 other variants:
- `gemm_hfq6g256_residual_wmma_k2.hip` → exports both `_residual_wmma_k2` and `_wmma_k2`.
- `gemm_hfq6g256_residual_wave64_dp4a.hip` → exports both names.
- `gemm_hfq6g256_residual_fp16.hip` → exports both names.

Build implications: `kernels.rs` declares one `_SRC` constant per file (unchanged) and the kernel-name strings used by `ensure_kernel` get the second name added. **No new files**, no doubled binary size of the hot kernel body (the compiler emits two thin entry-point preambles around shared device code).

If template-based extraction proves obstructive at HIP build time (e.g. AMD's clang has an issue with `if constexpr` in a `__device__` context — unlikely on ROCm 7.x but verify), fall back to a preprocessor `#define RESIDUAL` macro controlling the store — same effect, uglier source.

**Risk this introduces:** modifying the existing residual kernels carries non-zero regression risk for the production residual paths. Mitigation: byte-exact byte-stable harness against pre-modification kernel binary (capture `.hsaco` md5 before/after; the residual path must produce identical output). If the templating refactor changes residual codegen at all, back out the refactor and write new files (the original plan's approach).

## Dispatch — unified entry point (Gemini §2)

Instead of duplicating the residual dispatch tree, add a private impl with a `residual: bool` flag and two thin entries:

```rust
fn gemm_hfq6g256_impl(
    &mut self,
    a_raw: &GpuTensor, x: &GpuTensor, y: &GpuTensor,
    m: usize, k: usize, batch_size: usize,
    residual: bool,
) -> HipResult<()> {
    self.bind_thread()?;
    // Existing dispatch tree (line 10406-10422), with kernel names switched
    // by the residual flag:
    //   - wmma: "gemm_hfq6g256_residual_wmma_k2" vs "gemm_hfq6g256_wmma_k2"
    //   - wave64_dp4a: "gemm_hfq6g256_residual_wave64_dp4a" vs "gemm_hfq6g256_wave64_dp4a"
    //   - fp16: "gemm_hfq6g256_residual_fp16" vs "gemm_hfq6g256_fp16"
    //   - scalar wave32 fallback: "gemm_hfq6g256_residual" vs "gemm_hfq6g256"
    // ...
}

pub fn gemm_hfq6g256_residual(&mut self, ...) -> HipResult<()> {
    self.gemm_hfq6g256_impl(a_raw, x, y, m, k, batch_size, /*residual=*/true)
}

pub fn gemm_hfq6g256(&mut self, ...) -> HipResult<()> {
    self.gemm_hfq6g256_impl(a_raw, x, y, m, k, batch_size, /*residual=*/false)
}
```

This means an arch arm added to one variant is automatically picked up by the other. Future-drift risk that Gemini §2 flagged collapses.

### `ensure_fp16_x` cache invalidation (GLM5 §3.2)

The WMMA wrapper calls `self.ensure_fp16_x(x, batch_size * k)` and caches the f16 conversion at `self.fp16_x_source_ptr`. If non-residual + residual calls share a forward pass on the SAME X (e.g. lmhead non-residual then a subsequent residual projection on the same hidden state — possible in DFlash), the cached f16 buffer is correctly reused. If they're called on DIFFERENT X but the source pointer collides, the cache returns stale bytes.

The existing `gemm_f16_batched_lmhead` at `dispatch.rs:10266` handles this by setting `self.fp16_x_source_ptr = std::ptr::null_mut()` to force re-conversion. The non-residual wrapper does NOT need this (the X pointer is the source-of-truth cache key; if it's the same pointer with the same `n*k`, the bytes are the same; if not, the cache misses correctly). But a comment in the impl pointing to this invariant prevents future-me from "optimizing" it broken.

## Scope (call sites converted, real and acknowledged)

| Call site | Action |
|---|---|
| `qwen35.rs::batched_gemm_single_weight` MQ6 arm | Replace `memset_async + gemm_hfq6g256_residual` with `gemm_hfq6g256`. Drops 2 LOC. |
| `dispatch.rs::gemm_hfq6g256_batched_lmhead` (line 10240) | Same conversion. Same active-stream hazard removed. (GLM5 §2.1) |
| `llama.rs:1719` (`gemm_hfq6g256_residual` on `wo`) | **No change** — this IS a residual add (`x_batch += wo · x`). Audited; documented. (GLM5 §2.4) |
| `llama.rs:1786` (`gemm_hfq6g256_residual` on `w_down`) | **No change** — same as above. Audited. (GLM5 §2.4) |

## gfx12 — explicit out-of-scope decision (my §7, GLM5 §1.2)

HFQ6 has no `_wmma_gfx12` variant on the residual side (only HFQ4 does, at `dispatch.rs:7831`). The original plan's target list said "gfx1200/1201" but the dispatch tree had no gfx12 arm. Reviewers flagged this honestly.

**Decision:** drop gfx1200/1201 from this plan's target list. On gfx12, both residual and non-residual MQ6 take the FP16 fallback today. Adding `gemm_hfq6g256_residual_wmma_gfx12` (and its non-residual sibling, automatic under the templatize approach) is a separate task that an owner should pick up alongside the kmd2 / mixed-format productionization story. Open a follow-up issue and link.

## Acceptance criteria (rewritten to remove the self-contradiction GLM5 §3.1 caught)

1. `cargo check -p rdna-compute -p hipfire-arch-qwen35 -p hipfire-runtime` clean.
2. **Production residual kernels' `.hsaco` md5 unchanged** post-templatize. The refactor must not perturb existing codegen (mitigates the regression risk of touching shipped kernels). Capture md5 of `.hipfire_kernels/gfx1151/gemm_hfq6g256_residual*.hsaco` before refactor, compare after. If any blob differs, the refactor failed — investigate or revert to "new file" approach.
3. **Pick ONE comparison rule and stick with it:**
   - The non-residual kernel's output `Y_new[i, j]` MUST be **bit-identical** to `(memset(Y_old, 0) ; residual(Y_old))[i, j]` for every `(i, j)`. Floating-point: `val + 0.0 == val` for finite val; we're comparing bits AFTER WMMA accumulate-and-store, so the only edge cases are signed-zero outputs (`-0.0 + 0.0 = +0.0`) and NaN payloads. Both are tolerable in this context (logits don't end at `-0.0`, and a NaN output is already a fatal-class bug regardless of payload). Comparison: `f32_to_bits(y_new[i]) == f32_to_bits(y_old[i])` mod the signed-zero exemption.
   - **ulp tolerance is NOT used.** If bit-identicality fails, the templatize refactor changed codegen; back out per criterion 2.
4. **Parity vs CPU f32 reference** (Gemini §3, GLM5 §2.2). Extend the existing `crates/hipfire-runtime/examples/test_hfq6_gemm.rs` to add a `--non-residual` mode. The existing CPU reference is the ground truth; both residual-pre-zeroed and the new non-residual must match it to the existing test's tolerance (single ulp under reduction order). This catches dequant bugs in the shared template that the criterion-3 self-parity wouldn't.
5. **Poison test** (Gemini §5). In the same test binary, initialize Y with `f32::NAN` before calling the non-residual kernel. Every output element must be overwritten (no NaN survives). Catches tiling / `local_bs` guard mistakes that would silently leak uninitialized memory after the memset goes away.
6. **KLD on `qwen3.5-9b.mq4-kmd2-q8conv1d` byte-identical** to the post-#249 baseline of `0.155438` at q8 KV, n=20 chunks. Same code path, same FP rounding order, must yield same bits. If it differs, the residual-side refactor leaked something. *(Removed the original plan's ±0.001 tolerance — was inconsistent with the byte-exact claim per GLM5 §3.1 and my §2.)*
7. **Perf gate**: ≥ 0.5% prefill speedup on the kmd2 model at n=1024 across 3 fresh-process runs (per `docs/methodology/perf-benchmarking.md`), uncontended GPU. Below 0.5% means the bandwidth optimization didn't carry — but **don't auto-revert** (GLM5 §4.2): the correctness benefit (no active-stream hazard) and the lmhead-path fix are real even at 0% perf. Threshold is for "did we measure what Phase 0 predicted?"; decision to keep/revert is a judgement call.

Coherence-gate: **not in the acceptance set** — the gate's models are uniform-format and don't exercise `gemm_hfq6g256` (non-residual) at all (my §5, GLM5 §5.6). Adding a kmd2 row to the gate matrix is its own task and not blocking on this PR.

## Test plan

1. **Phase 0 microbench** (go/no-go gate) — `bench_mq6_residual_overhead.rs`. See §"Profile-first gate" above.
2. **Extended `test_hfq6_gemm.rs`** (criterion 3 + 4 + 5 combined):
   - Add `--mode {residual, non-residual}` flag (default: both).
   - Shapes: the **real model shapes** flagged in my §4 — `(m=256, k=4096)` for wv on 9B, `(m=4096, k=12288)` for w_down on 9B, plus the existing power-of-two matrix. *(Not random, not the original plan's `m ∈ {1024,2048,4096}`.)*
   - Batch sizes: `{1, 16, 64, 256, 1024}`. Include `1` so single-token decode is covered.
   - For each shape × batch_size × mode: random-init weight + x, compute CPU reference, run kernel, compare bits to CPU ref (criterion 4) and to memset+residual (criterion 3) and confirm no NaN-poison survives (criterion 5).
3. **End-to-end eval**: rebuild `eval_hipfire`, re-run kmd2 q8 n=20. KLD must be byte-identical to baseline `0.155438` (criterion 6). PPL must be byte-identical to `9.2070`.
4. **Uniform-MQ6 regression smoke**: run `eval_hipfire` on `qwen3.5-9b.mq6` before/after. Must be byte-identical (uniform path is untouched; this is a defense against accidental codegen drift in the residual sibling). Tied to criterion 2.

## Sequencing (issue #1 from my review fixed)

The original plan's sequencing had Phase 1 wire `qwen35.rs` to a scalar-only kernel, causing a perf regression on gfx11 (which falls out of WMMA into scalar). Rewritten:

- **Phase 0**: microbench. Output is a single tok/s delta number per shape. Implementer reviews → go or kill. ≤ 1 day.
- **Phase 1**: templatize the **4 existing residual kernels** in place, exposing the second `extern "C"` entry point in each. NO dispatch wrapper yet, NO qwen35.rs change. Validate criterion 2 (`.hsaco` md5 unchanged for the residual entry points). ≤ 1 day if templatize works; ≤ 2 days if fallback to preprocessor macros.
- **Phase 2**: add the `gemm_hfq6g256_impl` private + the two thin pub entry points in `dispatch.rs`. Run criteria 3, 4, 5 (the test_hfq6_gemm.rs extension). ≤ 1 day.
- **Phase 3**: wire `batched_gemm_single_weight` MQ6 arm AND `gemm_hfq6g256_batched_lmhead` to the new entry point. Run criteria 6, 7 (KLD parity, perf gate). ≤ half a day.

Total: **3-4 working days end-to-end**, dominated by Phase 1 (the templatize correctness audit). The original plan's "half a day per arch variant in Phase 2" was wrong (my review §6, GLM5 §4.3) — the work is the validation pipeline, not the per-variant kernel edit.

**Crucially**: by landing Phases 1+2 before Phase 3, the production residual path stays on its existing WMMA fast path the entire time. The qwen35.rs call site keeps using `memset_async + gemm_hfq6g256_residual` until Phase 3 swaps it. **No perf regression possible mid-stack.**

## Profiling-bytes placeholder (GLM5 §2.3)

The residual wrapper uses `gemv_hfq4g256_bytes(m, k)` with a "placeholder until hfq6 profiling added" comment (`dispatch.rs:10446`). Phase 2 inherits this. Out of scope for this plan to fix — open a follow-up issue noting the placeholder is now load-bearing for two wrappers and add an `hfq6_bytes` helper that accounts for the 200 B/group stride.

## Alternative if Phase 0 says no

If the microbench finds <1% overhead, the kernel proliferation isn't justified by perf. Two lower-cost alternatives:

(a) **Inline the memset+residual into a dispatch.rs helper** named something like `gemm_hfq6g256_into_zero`. Same kernel, same launch count, but the helper encapsulates the active-stream rule so future callers can't get it wrong. ~30 LOC. Closes the correctness hazard without touching kernels.

(b) **Do nothing.** The perf cost is in noise floor, and the active-stream rule is documented in CLAUDE.md and at the call site. The cost of leaving it is one comment paragraph in `batched_gemm_single_weight`.

Default if Phase 0 says no: (a). It pays for itself the first time another non-uniform format ships.

## Risks (updated from all three reviews)

- **WMMA accumulator init semantics**: the residual WMMA kernel may initialize its WMMA accumulator by *loading* `Y_prev` rather than starting from a register-zero. Under templatize, `if constexpr (RESIDUAL)` needs to gate both the *initial load* and the *final store* of the accumulator. If both paths exist, fine; if the accumulator init is implicit (e.g. WMMA always starts from zero in registers regardless of Y), the residual path is currently loading Y for no reason and the non-residual is a no-op simplification. Verify by reading `gemm_hfq6g256_residual_wmma_k2.hip` BEFORE writing the templatize patch. (Mine + Gemini §3 partial overlap.)
- **`.hsaco` codegen drift from templatize**: covered by criterion 2 — if any blob differs, abort the templatize approach and fall back to new files. The original-plan "new file" path remains the conservative escape hatch.
- **Tiling artifacts in non-residual** (Gemini §5): if the kernel has an off-by-one in `local_bs` guards, an output row may not get written. With memset+residual, that row reads as zero (still wrong, but bounded). With non-residual, it reads as undefined GPU memory garbage. Criterion 5 (poison test) is the catch.
- **Phase 0 measurement noise**: ±10-15% within-session drift is documented on gfx1100 (CLAUDE.md). Phase 0 microbench must run in fresh processes per `docs/methodology/perf-benchmarking.md`, otherwise the go/no-go decision is unreliable.
- **dp4a `ensure_q8_1_mmq_x` scratch and `capture_mode` guard**: mirror the residual wrapper exactly, don't simplify. Same scratch, same `!self.capture_mode` predicate.
- **`ensure_fp16_x` cache invalidation**: covered above. Comment in the impl prevents future regression.
- **gfx10 archs**: no dedicated fast path; uses FP16 fallback or scalar wave32. Per the residual side's status. Documented in §Targets, no action needed.

## Out of scope (consolidated)

- gfx1200/1201 WMMA path for both residual and non-residual HFQ6 (open follow-up).
- Fused mixed-format kernel (`gemm_qkv_mixed_hfq4_hfq4_hfq6` etc.) — Gemini §recommendations elevated this; I'm explicitly **not** elevating it. The fused-kernel path is a far larger task (~500-1000 LOC per shape) and shouldn't ride on this PR. If kmd2 ever becomes a default and Phase 7 perf shows launch-overhead-bound, plan it separately.
- Profiling-bytes accurate `hfq6_bytes` helper (open follow-up).
- Coherence-gate kmd2 row (separate task).
- Non-residual variants for MQ3 / MQ2 / MFP4 (same pattern, no current consumer).

## File layout summary

| Path | Action | LOC delta |
|---|---|---:|
| `kernels/src/gemm_hfq6g256_residual.hip` | templatize, expose `gemm_hfq6g256` entry | +~25 |
| `kernels/src/gemm_hfq6g256_residual_wmma_k2.hip` | same | +~25 |
| `kernels/src/gemm_hfq6g256_residual_wave64_dp4a.hip` | same | +~25 |
| `kernels/src/gemm_hfq6g256_residual_fp16.hip` | same | +~25 |
| `crates/rdna-compute/src/kernels.rs` | add kernel-name strings for the new entry points | +~8 |
| `crates/rdna-compute/src/dispatch.rs` | extract impl + thin pub entry point | +~30 |
| `crates/hipfire-arch-qwen35/src/qwen35.rs` | rewrite MQ6 arm of `batched_gemm_single_weight` | -5/+2 |
| `crates/rdna-compute/src/dispatch.rs::gemm_hfq6g256_batched_lmhead` | switch to non-residual | -5/+2 |
| `crates/hipfire-runtime/examples/test_hfq6_gemm.rs` | extend with `--non-residual`, real-model shapes, poison test | +~80 |
| `crates/hipfire-runtime/examples/bench_mq6_residual_overhead.rs` | new microbench (Phase 0) | +~100 |

**Total: ~325 LOC**, mostly templatize boilerplate. (Original plan's "~150 LOC" was off by 2× per GLM5 §1.3 — corrected here.)
