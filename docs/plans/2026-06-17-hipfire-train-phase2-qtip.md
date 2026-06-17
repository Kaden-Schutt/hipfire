# hipfire-train Phase 2: QTIP-style tuning → coherent quantized Supra-50M

Status: IN PROGRESS (Q0 started 2026-06-17)
Builds on: Phase 0 (docs/plans/2026-06-17-hipfire-train-phase0.md) — verified
fp32 forward+backward+AdamW+LoRA, and the full Supra-50M training loop.

## Goal

Get QTIP-style fine-tuning running and use it to take a QTIP-quantized Supra-50M
back toward coherence. QTIP "fine-tuning" (per the vendored QTIP source, see the
Phase 0 conversation) is **quantization-error recovery via distillation, codes
frozen** — NOT QLoRA task FT. Trainable = the surviving fp params (RMSNorms,
optionally LoRA adapters as extra recovery capacity); the trellis codes never
move.

## Architecture decision: decode QTIP → fp32 once, frozen base

The codes never change during recovery FT, so we decode each quantized linear to
fp32 **once** at load and keep it as the frozen base. The training forward stays
the verified fp32 path (no GPU trellis-decode kernel needed). The fp32
*original* model is the teacher; the QTIP-dequant model is the student.

`cpu_fwht_256` is orthogonal ((1/16)²·H² = I, signs involutive) ⇒ the inverse
rotation is the same routine with sign vectors swapped. Verified.

## Milestones

- **Q0 — QTIP quantize→dequant + damage.**
  - ✅ `qtip_quant.rs`: vendored encoder (1MAD codebook, beam Viterbi, decode,
    scales) + FWHT; `qtip_quantize_dequant(W, bits, beam)` → fp32 hatW.
    `examples/qtip_roundtrip.rs` PASS (CPU): Gaussian recon MSE/var 0.082 @2-bit,
    0.022 @3-bit, 3-bit beats 2-bit.
  - TODO Q0b: quantize Supra-50M's linears, build the student model, measure the
    teacher↔student gap (CE loss increase + per-token KL) — the gap recovery FT
    must close. (Beam encode is CPU-heavy at model scale; tune beam / parallelize
    with rayon, or quantize a subset first.)
- **Q1 — distillation loss op.** ✅ DONE: `kernels/src/distill_train.hip`,
  `ops/distill.rs`, `examples/gradcheck_distill.rs` PASS (1.4e-4). KL(teacher‖
  softmax(student)); d_logits = softmax(student) − teacher_p.
- **Q2 — recovery FT loop.** ✅ DONE: norm grads threaded out of block/model
  backward (`BlockLoraGrad` gains dnorm1/dnorm2; `model_distill_backward` +
  `recovery_params`/`flatten_recovery_grads`). `examples/recovery_ft_supra50m.rs`:
  QTIP-3 student distilled to fp32 teacher, training LoRA + layernorms (codes
  frozen). Mean KL **0.915 → 0.0296 nats/token in 120 steps** on gfx1103.
  QTIP-style tuning works end to end.
  - Caveat: distills on a FIXED small set (no tokenizer yet) — demonstrates the
    mechanism; broad coherence needs real calibration text (Q3).
- **Q3 — coherence.** IN PROGRESS.
  - ✅ step 1: tokenizer (`hipfire_model::tokenizer::Tokenizer`, no new dep) +
    greedy generation. `examples/generate_supra50m.rs`: the fp32 teacher
    generates COHERENT English ("…a fascinating and complex topic. In the late
    Middle Ages, the Roman Empire was a powerful … force that had a profound
    impact on the world"). This validates the ENTIRE hipfire-train fp32 forward
    against real output — strongest end-to-end check yet, beyond gradchecks.
  - TODO step 2: distill the QTIP student on real text (encode a corpus with the
    tokenizer), then generate from the student BEFORE vs AFTER recovery FT and
    compare to the teacher — the actual coherence-recovery demonstration.

## Simplifications vs full QTIP (documented, revisit later)

- **No Hessian-aware LDLQ** yet: groups are beam-encoded independently (no
  cross-group error feedback). Quality is below full QTIP; recovery FT is meant
  to claw it back. LDLQ integration is a later increment.
- **Vendored encoder**, not shared with `hipfire-quantize` (bin-only crate). The
  decoded values are faithful QTIP but not byte-compatible with the inference
  Qtip3 artifact format — fine here (we run our own fp32 forward); revisit if the
  tuned model must load in the inference engine (would need a hipfire-quantize
  lib extraction + format match).
- **Sequential interleaved quantize-and-FT** (real QTIP quantizes one linear at a
  time, fine-tuning the still-fp ones between) is collapsed to quantize-all-then-
  recover. Simpler; can add interleaving if recovery is insufficient.
