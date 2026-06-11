# Gemma4 Softcap-Aware LM-Head Calibration -- Design Doc

**Status:** Research-grade feasibility + skeleton only. Full implementation deferred.
**Env gate:** `HIPFIRE_GEMMA4_SOFTCAP_CALIB=1` (no-op skeleton added to quantizer).

---

## Objective

Standard AWQ calibration for the lm_head weight W minimizes

    || Wq * x  -  W * x ||

where Wq is the quantized weight. For Gemma4, the actual deployed operation is

    softcap( Wq * x )  =  tanh((Wq*x) / 30) * 30

The tanh compresses large logits toward +/-30. An outlier logit at raw z=60 is mapped to
softcap(60)=29.5. Quantization error dz at that position contributes
dz * tanh'(z/30)/30 to the final logit, where tanh'(u)=1-tanh^2(u) -> 0 as |u| -> inf.
The softcap attenuates large-magnitude quant errors before sampling, unlike a plain
lm_head where the full dz reaches the sampler.

The softcap-aware objective exploits this: instead of minimizing ||(Wq-W)*x||, minimize

    || tanh(Wq*x / cap)*cap  -  tanh(W*x / cap)*cap ||

This reweights the residual loss by the softcap sensitivity at each logit position.
Rows already pushed to saturation accept more quant error; near-zero logit rows (rare
tokens, delimiter tokens) get tighter error budget.

### Practical KLD impact estimate

lm_head is 248,320 x 2,304 (gemma4 12B). At Q8 (default) reconstruction error is tiny
(||dz|| ~0.01 RMS per row). Net KLD improvement over standard Q8: **~0-2%** on wt2.
For MQ4 lm_head (--lm-head-format mq4) where errors are ~8x larger: **~10-25%**,
making this calibration most relevant in that regime.

---

## Code Locations

### Softcap constant

`crates/hipfire-arch-gemma4/src/config.rs:75`

```rust
pub final_logit_softcapping: f32, // 30.0 -- tanh(x/30)*30
```

Read from `config.json["final_logit_softcapping"]`, defaults to 0.0 if absent.

### Forward path (where softcap fires)

`crates/hipfire-arch-gemma4/src/forward.rs:425-437`:

```rust
// 5) LM head -> logits (tied embed bytes via lm_head.buf alias).
weight_gemv(gpu, &weights.lm_head, &state.tmp, &state.logits)?;

// 6) Final logit softcap: logits = tanh(logits / cap) * cap.
if cfg.final_logit_softcapping > 0.0 {
    gpu.logit_softcap_f32(&state.logits, cfg.vocab_size, cfg.final_logit_softcapping)?;
}
```

`state.tmp` is the post-final-RMSNorm hidden state [dim] (dim=2304 for 12B). This is
the activation x that calibration would capture.

### Embed / lm_head quantization in the quantizer

`crates/hipfire-quantize/src/main.rs` -- lm_head is the tied embedding; both enter the
Q8 arm at line ~6646-6649:

```rust
} else if kmap_level == QuantLevel::Q8 {
    // K-map says Q8 (embed, lm_head, router)
    let q = quantize_q8f16(&f32_data);
    (q, QuantType::Q8F16, 32u32, "Q8_F16")
```

The kmap resolves embed_tokens.weight and lm_head.weight to QuantLevel::Q8 via
`is_q8_tensor` (line ~3293-3297). For --lm-head-format override see line ~6709
(QuantLevel::Override), where MQ4/MQ6 lm_head experiments route.

### Activation threading (NOT YET WIRED)

The activation x (= state.tmp at forward time) is NOT threaded through the quantizer.
The quantizer runs offline on weight bytes only.

The existing `ActivationCapture` trait (`crates/rdna-compute/src/dispatch.rs:302`) is a
scaffold -- the Gpu field and weight_gemv hook are TODO stubs (see
`crates/hipfire-runtime/src/bin/collect_imatrix.rs:170-175`). WeightTensor has no `name`
field. Activations currently reach the quantizer only via the offline GGUF imatrix file
(`--imatrix <path>`, loaded by `load_imatrix` at line ~3853).

---

## Implementation Approach

### Step 1 -- Collect lm_head activations at calibration time

Add a named hook in `forward.rs` before the `weight_gemv(lm_head)` call, downloading
`state.tmp` [dim] per token into a corpus matrix X of shape [n_tokens, dim]. This is
cheaper than wiring the full ActivationCapture chain -- a lm_head-specific tap suffices.

Memory: dim=2304 * n_tokens * 2 bytes FP16 = ~450 MB per 100k tokens. Fine for 1-5k
token calibration corpora.

### Step 2 -- Softcap-aware scale computation (offline)

Given W [vocab x dim] and corpus activations X [n x dim]:

1. Compute Z_ref = W * X^T, apply softcap -> S_ref.
2. For candidate quantized Wq: compute Zq = Wq * X^T, apply softcap -> Sq.
3. Minimize KL(softmax(S_ref) || softmax(Sq)) over quantization parameters.

For AWQ-style channel rescaling with scale vector s [dim]:

    argmin_s || tanh( (W*s * (x/s)) / cap ) * cap
              - tanh( (Wq(s) * (x/s)) / cap ) * cap ||_F

Unlike standard AWQ (closed-form s), this has no closed form -- needs a ~20-point alpha
grid search (same as standard AWQ), with softcap applied to each candidate's output.
Extra cost: O(vocab * n_tokens * 20) tanhf() calls, ~1B ops at 5k-token corpus,
sub-second on CPU.

### Step 3 -- Runtime (tied-embed constraint)

The AWQ sidecar mechanism already exists (WeightTensor.awq_scale, loaded from
`<name>.awq_scale` sidecar). The scale applies ONLY to the lm_head dispatch (after final
norm) -- NOT to the embedding lookup (raw token index, no dot product with x). This is
a structural constraint: the sidecar must be lm_head-specific, not embed_tokens-generic.

Implementation: emit sidecar under `model.language_model.lm_head.awq_scale` (distinct
from `embed_tokens`). The loader already ties lm_head bytes to embed_tokens
(GpuTensor alias at `gemma4.rs:261`), so the awq_scale tensor can be loaded into the
WeightTensor.awq_scale field of lm_head without touching embed_tokens.

---

## Effort Estimate

| Sub-task | Effort |
|---|---|
| Hook state.tmp dump in forward.rs (lm_head site only) | 1-2h |
| Corpus dump binary (collect_softcap_calib) | 2-4h |
| Offline scale optimizer (Rust or Python script) | 4-8h |
| AWQ sidecar emit for lm_head in quantizer | 2-4h |
| Runtime: split AWQ application (lm_head only, not embed lookup) | 1-2h |
| Eval: KLD delta vs standard Q8 / MQ4 lm_head | 1-2h (needs GPU) |

**Total: ~1.5-2 dev days.**

---

## Feasibility Verdict

**FEASIBLE at modest effort (~1.5-2 dev days). Low priority for Q8 lm_head (~1-2% KLD
win). Meaningful for MQ4 lm_head (~10-25% KLD win). Recommended gate: measure MQ4
lm_head KLD vs embed+down_F16 baseline (0.071) before investing.**

The tied-embedding constraint is the only structural wrinkle -- solved by name-keying the
sidecar to the lm_head dispatch site, not the shared bytes.
