# RoughQuant — Phase 3 scope: real packed format (graded channel protection)

Scope for the SHIPPABLE form of the verified sim result (foldable channel
protection halves mq4 KLD). The bf16 sim can't settle absolute coherence/perf;
this is the design + plan to do so on the real GEMV. Incorporates the review
finding that **weight importance is a spectrum (multiple levels), not just
activation energy** — so the format is GRADED, not binary protect/bulk.

## Key design realization: no rotation, no fold

The dead PCA path needed a foldable rotation. **Channel protection does not** —
keeping specific channels precise is just storing them at higher precision. The
clean, low-risk realization:

```
W stored as: mq4(W)                          [unchanged MQ4G256, all channels]
           + sparse correction sidecar:      [for the protected channel set S]
               residual R[:,c] = (W - dequant(mq4(W)))[:,c]  for c in S,  bf16/Q8
y = mq4_gemv(W_mq4, x)  +  sparse_gemv(R_S, x_S)
```

- No rotation, no fold, no residual-stream surgery.
- Reuses the existing mq4 GEMV verbatim; adds a small **sparse correction GEMV**
  over |S| channels (readers: |S| input cols; writers: |S| output rows).
- The sidecar stores the *residual* (exact − mq4), so a protected channel becomes
  effectively bf16/Q8 with no double-quant.

## Optional perf optimization: #5 permutation → gather-free correction

The sparse correction needs a runtime **gather** of `x[S]` (readers) and a
**scatter** into `y[S]` (writers) — implemented as `rq_gather_f32` /
`rq_scatter_add_f32`. The *data* there is trivial (~|S| floats), but as standalone
kernels they add launch + alloc overhead on the decode hot path (cf. the measured
`add_inplace_f32` ~4.4% of decode time on gfx1100/gfx1151; less relevant on a
bandwidth-bound 780M). The **#5 residual-stream permutation eliminates them**:

- Permute the residual channel order (OFFLINE, baked into weights) so the protected
  set S is a **contiguous block** (e.g. channels `0..|S|`). Then the reader gather
  becomes a pointer-offset **slice** `x_normed[0..|S|]` and the writer scatter
  becomes a contiguous **slice-add** `y[0..|S|] += c`. No gather/scatter kernel, no
  index buffer.
- #5 is **runtime-free** (verified bijective in `permutation-bijectivity.md`): the
  stream is simply *stored* in the permuted order with the permutation propagated
  across embed rows + every reader input-col + every writer output-row + RMSNorm γ
  + lm_head + tied embed. No per-token transform (unlike the dead PCA rotation).
- It reorders the hidden/residual axis only, **upstream of the q/k/v head split**,
  so there is **no RoPE interaction** (the RoPE constraint is on #4 per-head-dims).
- For the **current binary format (residual-only protection) this removes the
  gather/scatter entirely** — one global #5 makes the single shared set contiguous
  everywhere. It only becomes partial ("most") if per-weight NON-residual protection
  is later added (each such projection's salient input cols would need its own free
  #1 hidden-dim permute; one global perm can't cluster all sets at once).
- It does **not** save the correction GEMV MACs (~|S|/d_in ≈ 5% on protected
  projections) — only the gather/scatter overhead, which is exactly the part that
  wasn't already free.

⇒ This is the real consumer for the (verified but unbuilt) **production #5 applier**:
not "folding a rotation" (channel protection needs no fold) but making the
correction gather-free. The producer would emit S as a contiguous range + bake the
#5 permutation; the lowered-path correction then reads/writes dense slices.

## Graded importance (the review insight)

Not binary. Tiers by importance LEVEL:

| tier | precision | ~size | who |
|---|---|---|---|
| T0 super-critical | bf16 (maybe fp32 scalars) | ~0.1–1% | super-weight channels |
| T1 important | Q8 | ~2–8% | the shared outlier set |
| T2 bulk | mq4 (FWHT 4-bit) | ~90%+ | everything else |
| (T3 void) | — | 0% | NONE — void was catastrophic; no dead tail |

**Importance = multi-signal, not diag(H) alone.** Candidates to combine/compare:
- activation energy `E[x²]` = diag(H) (current);
- output-error contribution `‖W[:,c]‖²·E[x²]` (product);
- OBS/GPTQ sensitivity `(ΔW)²/[H⁻¹]_cc` (accounts for compensation by other
  channels — the OBS saliency, more principled than raw magnitude/energy);
- (gold standard, expensive) empirical per-channel KLD ablation.

## De-risk experiments (sim, before kernels — cheap)

1. **Graded vs binary (iso-bits):** does {T0 bf16 + T1 Q8 + T2 mq4} beat {binary
   bf16-protect + mq4} at equal avg-bits on KLD? Tests the multi-level thesis.
   (Note: Q8 *weight* protection broke generation in one test but that was the
   Q8-DeltaNet-state confound; re-test Q8 weight tiers with FP32_STATE + the real
   format.)
2. **Importance metric bake-off:** for a fixed bit budget, which selector
   (energy / product / OBS) picks the channels that minimize KLD? Add
   `HIPFIRE_RQ4_SALIENCY=obs` alongside diag/wnorm/product.
3. **Sidecar size sweep:** KLD + tok/s vs |S| and per-tier precision, to find the
   knee on the real format.

## Build status (2026-06-17)

**Step 1 (producer) DONE + verified — commit `fb3d403b`.** `--format roughquant`
(`rq`) emits real MQ4G256 bulk + a bf16 correction sidecar over the diag(H)-selected
shared residual set (reader cols + writer rows). Key numerics de-risk: the feared
"sim ≠ real mq4" gap was only the *sim* storing its recon as lossy bf16 — NOT a
dequant mismatch. `dequant_mq4g256` (added to the quantize crate) is bit-identical
to the GEMV kernel (which rotates x, so its effective W = inverse-FWHT(stored) =
dequant), so `R = W − dequant_mq4g256(packed)` makes protected channels exact on the
REAL kernel. In-place self-check on 0.8B @ 5%: protected-channel recon
max-err = **1.19e-4** (= bf16 rounding of R only). Indices in
`metadata["roughquant_sidecar"]`; values in `<name>.rqcorr` bf16 tensors; absent
sidecar = plain mq4 (backward-compatible).

Producer simplification vs the graded table above: v1 is BINARY (bf16-protect +
mq4-bulk), the validated mechanism. Graded T0/T1 tiers and embed-Q8 (for size
fairness vs mq4, which Q8s the tied embed — the current rq file leaves it bf16,
hence 838 MB) are follow-ups once the binary verdict lands.

## Remaining implementation plan

1. ~~**Producer**~~ — DONE (`fb3d403b`), see above.
2. **Loader/format**: read `metadata["roughquant_sidecar"]`, load each `.rqcorr`
   bf16 tensor + a u32 channel-index buffer to GPU. Mirror the AWQ-sidecar loader
   precedent (`load_awq_scale_for`, qwen35.rs:2729).
3. **Correction GEMV (compose existing primitives, NO new kernel) — PROVEN on real
   GPU, `rq_real_gemv_check` example, commit pending**: reader = gather `x[S]`→`xs`
   then dense `gemv_f32` of `corr[m×|S|]` added into `y`; writer = `gemv_f32` of
   `corr[|S|×k]·x` then scatter-add into `y[S]`. Kernel-level proof result: with
   `x` nonzero only on `S`, `mq4_kernel(x) + gemv_f32(corr,x_S)` reconstructs the
   protected channels to **bf16 precision** (corrected err 3.5e-3–1.3e-2 vs
   uncorrected mq4 1.1–5.0, i.e. 100–400× reduction) on the real
   `gemv_mq4g256_with_rotate` kernel. Two findings:
   - `dequant_mq4g256` is **bit-identical** to the kernel's effective weights
     (|kernel − cpu(recon·x)| ≈ 1e-6), so the producer's residual is exact — no
     producer/kernel dequant mismatch.
   - **`gemv_f32`'s tree reduction requires a power-of-2 block** (`block=256.min(k)`),
     so for the small correction width `|S|` (<256, not power-of-2) the gather/corr
     MUST be **padded to the next power of two** (zeros don't change the dot). The
     forward wiring + a fused gfx1100/gfx1201 kernel are the only remaining work.
4. **Forward wiring (the big, hot-path, coherence-gated step) — LARGER THAN
   EXPECTED, see finding**: the KLD path uses only `forward_scratch`, which
   delegates to `forward_scratch_layers` (≈4,200 lines, qwen35.rs:21830). Critical
   finding: **projections are FUSED, not separate GEMVs** — `fused_qkvza_*`
   (q+k+v+z+α+β in one kernel), `fused_gate_up_*` (gate+up in one), plus many
   arch-specific branches (dp4a / paro / prerotated / Lloyd). So the bf16
   correction cannot drop in "after each projection GEMV"; it needs a correction
   pass after each fused block, slicing the fused output to the right sub-range
   (wqkv→q/k/v/z/a/b; gate_up→gate/up), replicated across every branch, plus the
   writer corrections (wo, w_down). The side-map (per-layer correction bundle on
   the 4 layer structs — chosen 2026-06-17) avoids the WeightTensor 120-site churn,
   but the *apply* sites are numerous and branch-dense. This is the dominant cost
   and is what the product decision below should weigh.
5. **Validation gates (the actual verdict):**
   - KLD on the **real GEMV** (not sim) vs bf16 — must keep the ~half-mq4 win.
   - **Coherence battery** (`scripts/roughquant_coherence_battery.sh`, canonical
     prompts, FP32 state) on the real format — must not regress vs mq4.
   - **Fresh-probe tok/s** (`scripts/probe_commits.sh`): the sparse correction
     adds ~|S|/d_in extra GEMV FLOPs (~5–10% at |S|≈75, d≈1024); must quantify the
     decode tok/s cost and confirm the quality/bit/speed trade beats just shipping
     mq6 where VRAM allows.

## Perf budget (rough)

Sparse correction over |S|≈75 of 1024-dim residual readers ≈ 7% extra MACs on
those weights; writers similar. Net decode-time overhead likely a few %. If that
holds and KLD stays halved with coherence intact, it's a real intermediate
operating point between mq4 (4.25b) and mq6 (6.25b) that mq5 doesn't fill.

## Forward-wiring attempt (2026-06-17) — hand path DONE, lowered path is the blocker

Implemented the full correction stack and wired it into the **hand** forward path
(`forward_scratch_layers`):
- **Kernels** (`kernels/src/rq_correction.hip`): `rq_gather_f32` (xs = x[S], padded
  to pow2) + `rq_scatter_add_f32` (y[S] += c) + dispatch wrappers.
- **Loader** (`load_rq_corrections`): reads `metadata["roughquant_sidecar"]` +
  `.rqcorr` bf16 tensors into a `(layer_idx, RqProj) -> RqCorr` side-map on
  `Qwen35Weights` (pads reader corr to pow2). Verified: 138 reader + 48 writer
  corrections loaded on 0.8B @ 5%.
- **Apply** (`rq_apply_readers` / `rq_apply_writer`): reader = explicit
  original-frame `rmsnorm(x)` → gather S → `gemv_f32(corr)` → `add_inplace`;
  writer = `gemv_f32(corr)` → `rq_scatter_add`. Wired at all 12 DeltaNet +
  FullAttn projection sites. Gated on side-map presence ⇒ **byte-identical for
  non-roughquant models** (coherence-safe).

**Two blocking findings (why the clean verdict isn't measured yet):**
1. **The default decode path is the LOWERED super-op executor, NOT the hand
   path.** `forward_scratch_layers` returns `forward_scratch_layers_lowered` at
   entry unless `HIPFIRE_FORWARD_LOWERED=0`. The lowered path runs each layer as a
   `LayerProgram` through `hipfire_dispatch::pipeline::superop::run_layer_program`
   — a generic cross-crate super-op executor with NO per-projection hook. Wiring
   the correction there (correction super-ops in the IR, or a post-projection bind
   hook) is a separate, larger, cross-crate task than the hand path.
2. **The hand path is NOT a valid baseline — it diverges from lowered.** On plain
   mq4 (no sidecar, FP32 DeltaNet state, ctx 1024): hand path KLD = **0.598** vs
   lowered = **0.158** for the SAME weights. The "byte-identical" claim in the
   lowered-path docstring does not hold under this config (likely an FP32-state or
   staleness interaction — a pre-existing issue, not roughquant). So the hand path
   can't produce the clean ~0.084 verdict number.

**Update (deferred 2026-06-17):** further isolation showed the hand path is not
merely divergent but **catastrophically broken** — plain **bf16** through the hand
path gives self-KLD **13.89** vs lowered **0.000** (the hand path produces
near-random output, independent of roughquant). So "fix the hand path" = resurrect
dead code, and the only working forward is the lowered super-op executor. Per the
product owner, #10c is **DEFERRED** at proven-mechanism. The rq→hand routing is now
gated behind opt-in `HIPFIRE_RQ_HAND=1`; by default rq models use the working
(uncorrected) lowered path and stay coherent, with the correction stack dormant.
The verdict requires wiring the correction into the lowered super-op executor
(`run_layer_program`) — the scoped follow-up, to be gated by the product call (vs
mq6).

**Mechanism IS proven**: in the hand path, corrections reduce KLD 0.598 → 0.571
(real reduction; the math + kernels + loader are correct, consistent with the
standalone `rq_real_gemv_check` proof). The remaining work for the shippable
verdict is wiring the correction into the lowered super-op executor and measuring
KLD there (lowered mq4 0.158 → target ~0.084), then coherence + tok/s.

## Open / decision (updated 2026-06-17 — DECISION POINT)

Everything cheap and decisive is now DONE: importance science (diag ρ=0.90),
producer (verified), and the **kernel-level proof** that the correction
reconstructs protected channels to bf16 precision on the real GEMV. What remains
is ONLY the expensive part: wiring the correction into the fused, arch-branched
decode path (`forward_scratch_layers`) + coherence + tok/s — a multi-hour,
gate-heavy hot-path surgery across many fused-kernel branches (see step 4).

The product question is now sharp and should be answered BEFORE paying that cost:
**is the proven ~half-mq4-KLD win (4.25b→~4.8b effective) worth the fused-path
forward surgery + a per-fused-block correction GEMV, vs simply shipping mq6
(6.25b) where VRAM allows?** The intermediate operating point only matters where
mq6 doesn't fit but mq4 does and the quality gap bites. Inputs to the call:
- sim KLD halving is real and coherence-positive (established);
- real-GEMV protected-channel exactness is proven (kernel proof);
- the missing number is decode tok/s overhead of the correction on the real fused
  path — which requires doing the wiring to measure.

Recommendation: gate the forward-wiring spend on a product decision (or do a
single-projection spike to estimate tok/s overhead before committing to all
branches). Cross-model (7B/9B) confirmatory is independent and safe (the role
classifier + d_model-from-roles fix), and validates generality regardless.
