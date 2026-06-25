# Scope: norm-only OQ+ recovery for real qwen3.5 (block-local distillation)

Date: 2026-06-23. Branch: chaingun. Follows the Supra-50M recovery probes
(commits 33b42b09 / ef7ebdc2 / 56c5ed51) and their hardened held-out result:
**OQ+ W4A8 norm-only recovery generalizes to unseen text (~51% of the quant-loss,
Path-A exportable, zero runtime cost); KVarN+CASK merge recovery does NOT
generalize (~22%, dead lever).** This scopes taking the *OQ+ norm-recovery* lever
to the real qwen3.5 model.

## The core constraint

To get a gradient on a RMSNorm weight you must backprop through everything
downstream of that norm. End-to-end that needs the full differentiable qwen3.5
forward — and hipfire-train has **no differentiable gated-DeltaNet** (its
`ssm_block` uses a simplified `gated_scan`, not qwen3.5's conv1d+silu+split +
A_log/dt_bias + delta-rule scan). Porting that forward+backward (gradchecked) is
the large multi-session cost the teacher/student-split memory warned about.

**Block-local distillation removes that need for most norms.** Instead of one
end-to-end loss, tune each norm to make its *own block's output* match the
teacher's captured output. Each norm then needs backward through ONE block, and —
critically — if we capture the teacher's intermediate residuals, the
non-differentiable mixer can be bypassed entirely.

## Norm inventory (qwen3.5-0.8b, 24 layers = 18 linear_attn + 6 full_attn)

Per layer: `input_layernorm` (→ token mixer) + `post_attention_layernorm` (→ MLP),
plus one final norm. ~49 norm tensors total.

| Norm | mixer downstream | differentiable in hipfire-train today? |
|------|------------------|----------------------------------------|
| 24× MLP norm (post_attn → gate/up/down + swiglu) | SwiGLU MLP | **YES** (block.rs) |
| 6× attn-input norm (→ q/k/v, GQA) | GQA attention | **YES** (ops::attention) |
| 1× final norm (→ logits) | matmul | **YES** |
| 18× DeltaNet-input norm (→ DeltaNet in-projs) | gated delta rule | **NO** (needs port) |

So **31 of 49 norm tensors are recoverable with EXISTING differentiable ops**;
only the 18 DeltaNet-input norms need the DeltaNet backward.

## Phase A — the cheap, shippable port (NO DeltaNet math)

Recover the 31 "easy" norms via block-local distillation against daemon-captured
teacher activations. The trick that kills the DeltaNet dependency: capture the
teacher's **pre-MLP residual** `x_mid_b` (post-mixer, pre-`post_attention_layernorm`)
in addition to block input/output. Then:

- **MLP-norm recovery (all 24 layers, incl. the 18 DeltaNet layers):** feed
  captured `x_mid_b` → trainable `post_attention_layernorm` → gate/up (OQ+
  sim-quant) → swiglu → down → residual; MSE vs captured block output `y_b`.
  Backprop only through the MLP. **No mixer forward at all** — the DeltaNet is
  skipped because its output is the captured `x_mid_b` input. Pure existing ops.
- **Attn-input-norm recovery (6 full_attn layers):** captured block input `x_b`
  → trainable `input_layernorm` → q/k/v (OQ+) → GQA → o → residual; MSE vs the
  captured post-attn residual `x_mid_b`. Existing GQA forward+backward.

### Phase A work items
1. **Teacher capture (daemon, forward-only):** extend the existing hidden-state
   dump (`HIPFIRE_DUMP_HIDDEN_ALL` all-rows mode, per-layer; used by the
   kv-compression work) to also emit, per block per chunk: residual-in `x_b`,
   pre-MLP residual `x_mid_b`, residual-out `y_b`. One bf16 forward over a calib
   slice (`benchmarks/calib/calib-1m.txt`). Land via the v2 label cache
   (`hipfire-train checkpoint.rs::save_labels`) per the teacher/student split.
2. **Student blocks in hipfire-train:** a qwen3.5 MLP block (reuse block.rs
   swiglu+linears) and a full-attn block (reuse GQA), both loading OQ+-quantized
   projections from the `.hfq` (DeltaNetLayerWeights / attn weights;
   `oqplus_simquant` already exists for the sim, real loader path is the
   qwen35 qt=33 arm). Trainable param = the one norm scale; everything else frozen.
3. **Block-local distill loop:** AdamW on the norm scale, MSE(student_block_out,
   captured y_b/x_mid_b). Reuse the per-iter free + NaN-guard hygiene from
   `recovery_generalization_supra50m`.
4. **Export + close the loop on the REAL model:** write recovered norms into the
   real `.hfq` via Path-A (`hfq_patch.rs` overwrites BF16 norm bytes in place — no
   kernel/format change), then measure KLD-vs-bf16 + coherence gate on the actual
   daemon (`hipfire-eval`). This is the only step that proves block-local MSE
   recovery translates to end-to-end quality on qwen3.5.

### Phase A risk / open questions
- Block-local MSE is a proxy for end-to-end KL (standard layer-wise distillation,
  AdaRound-style). Step 4 is the real validator; if block-local under-delivers,
  fall back to a short end-to-end norm FT over the differentiable subset (still no
  DeltaNet backward if we keep DeltaNet mixers frozen via captured `x_mid`).
- Capture storage: per-block × 3 residuals × calib tokens × bf16. Bounded; stream
  to disk per chunk (the dump infra already streams).
- Unknown until measured: what fraction of the ~51% Supra-50M norm-recovery the
  31-norm subset delivers on qwen3.5. Hypothesis: most of it — MLP norms gate the
  largest matrices (gate/up/down) where W4 error concentrates.

## Phase B — DeltaNet-input norms (the real port, gated on Phase A payoff)

Only worth it if Phase A's measured recovery leaves meaningful headroom on the 18
DeltaNet-input norms. Needs a differentiable gated-DeltaNet block in hipfire-train:
conv1d+silu+split, A_log/dt_bias gating, the delta-rule recurrence — forward AND
gradchecked backward (mirrors the existing `gradcheck_gated_scan`/`ssm_block`
pattern but for the real delta rule). Large; defer.

**STE shortcut to consider first:** treat the DeltaNet mixer as frozen forward-only
(run it in the daemon, capture its output) and pass the input-norm gradient through
the recurrence as identity (STE) — we validated STE works for the KV path this
session. Approximate, but may recover most of the DeltaNet-input-norm headroom
without the full backward. Cheap to try before committing to Phase B.

## Bottom line

The qwen3.5 norm-recovery port is **not** one big DeltaNet-forward port. Phase A
ships a real, exportable OQ+ quality recovery (31/49 norms, incl. every MLP norm)
using only differentiable ops hipfire-train already has + an incremental daemon
capture hook. Phase B (DeltaNet norms) is gated on Phase A measuring leftover
headroom, and even then an STE shortcut may avoid the full backward.
