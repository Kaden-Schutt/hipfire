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
3. **Correction GEMV (compose existing primitives, likely no new kernel)**: reader
   = gather `x[S]`→`xs` then dense `gemv_f32`/residual of `corr[m×|S|]` added into
   `y`; writer = `gemv_f32` of `corr[|S|×k]·x` then scatter-add into `y[S]`. Reuse
   `dispatch.rs::gemv_f32` + a small gather (or the `*_residual` y+=A·x variants).
   gfx1100 + gfx1201 (RDNA4 mandatory) per cross-arch rule.
4. **Forward wiring (the big, hot-path, coherence-gated step)**: qwen35 has ~10
   forward variants (`forward_from_x_gpu`, `forward_scratch`, `forward_prefill_*`)
   with linears dispatched inline via `dispatch_ref()`→`GemvFamily` — there is NO
   single `linear()` helper, so the correction must be applied at each residual
   reader/writer projection. Two options: (a) add `rq_corr: Option<RqCorrection>`
   to `WeightTensor` (clean, but ~120 `rq_corr: None` constructor sites across
   crates — no `Default`), or (b) a side `HashMap` in the qwen35 model keyed by
   projection, applied within qwen35.rs only (lower cross-crate churn). Decide
   before starting; this is the multi-hour, gate-heavy part.
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

## Open / decision

- Cross-model (7B/9B) confirmatory (the role classifier + d_model-from-roles fix
  now make this safe).
- Is the ~half-KLD / few-% win worth a new format + sparse-GEMV kernel vs just
  using mq6 where VRAM allows? Product call once real-GEMV coherence + tok/s land.
