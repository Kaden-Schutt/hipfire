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

## Implementation plan (only after the sim de-risks confirm graded wins)

1. **Producer** (`hipfire-quantize`): emit MQ4G256 bulk + a `roughquant-sidecar`
   record = {channel ids (shared residual set, role-based per the audit fix),
   per-tier precision, residual values}. Reuse the role classifier + multi-signal
   selector.
2. **Loader/format**: extend the HFQ tensor schema with the sidecar; keep
   backward-compat (absent sidecar = plain mq4).
3. **GEMV**: `mq4_gemv` + a sparse correction kernel (small dense GEMV over |S|
   channels, bf16/Q8). gfx1100 + gfx1201 (RDNA4 mandatory) per repo cross-arch rule.
4. **Validation gates (the actual verdict):**
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
