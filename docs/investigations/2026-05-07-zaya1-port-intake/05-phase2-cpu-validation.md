# 05 - Phase 2 CPU validation results

**Date:** 2026-05-07
**Branch:** `feat/zaya1-port-intake`
**Predicate:** 04-phase1-results.md (reference dump landed; Phase 2
unblocked via the CPU-Rust workaround that skips HFQ).

## Summary

Two of the six "free components" on the contract list NRMSE-validated
against the PyTorch reference dump at sub-bf16-ULP precision:

```
=== ZayaRMSNorm @ layer 0 input_norm ===
  PASS NRMSE = 1.659e-3  layer_00 input_norm
=== ZayaRMSNorm @ layer 1 input_norm ===
  PASS NRMSE = 1.664e-3  layer_01 input_norm
=== ResidualScaling @ layer 0 (hidden_states only) ===
  PASS NRMSE = 2.412e-3  layer_00 res_scale.hidden_states
=== ResidualScaling @ layer 1 (residual path) ===
  PASS NRMSE = 0.000e0  layer_01 res_scale.residual
=== ResidualScaling @ layer 1 (hidden_states path) ===
  PASS NRMSE = 2.353e-3  layer_01 res_scale.hidden_states
=== ZayaRMSNorm @ final_norm ===
  PASS NRMSE = 1.652e-3  final_norm

=== ALL PASS (bf16-ULP threshold = 5e-3) ===
```

Threshold = 5e-3, derived from bf16's ~7-mantissa-bit precision.
Observed values: 0 to 2.4e-3, well clear.

## Methodology

The CPU-Rust workaround (per Phase 1 results doc) consumes:

1. The model's bf16 safetensors directly (38 KB subset extracted on
   hiptrx via `scripts/arch-intake/extract_phase2_subset.py`, scp'd
   back). HFQ writer for ZAYA1 is NOT required for component-level
   validation; only weight bytes are.
2. The PyTorch reference activation dump from
   `scripts/arch-intake/dump_zaya_reference.py` (v3 augmented to
   capture all positional inputs + multi-tensor outputs; 1279 tensors
   captured; `/tmp/zaya-port/refs/refs-canonical-v3/`).

`crates/hipfire-arch-zaya/examples/cpu_validate_phase2.rs` runs each
component on its real layer-N input (loaded from the ref dump),
computes the NRMSE against the layer-N output (also from the ref
dump), and asserts NRMSE < 5e-3.

This validates that the COMPONENT MATH is correct end-to-end against
PyTorch. The CPU impl is what hipfire's RDNA kernel must match; if
this passes, the kernel's job is reduced to "produce these same f32
values" which is a much cleaner contract.

## What's covered

| Component | Status | NRMSE | Notes |
|---|---|---|---|
| ZayaRMSNorm | PASS | 1.65-1.66e-3 (3 distinct sites) | Including final_norm |
| ResidualScaling (layer 0 hidden_states) | PASS | 2.41e-3 | First-layer special case (no residual path) |
| ResidualScaling (layer 1 residual) | PASS | 0e0 | Bit-exact (math-only mul/add) |
| ResidualScaling (layer 1 hidden_states) | PASS | 2.35e-3 | Standard not-first-layer path |

The 0.000e0 on the residual path is a bit-exact match; both sides
compute `(x + bias) * scale` with bf16 inputs and the rounding
sequence happens to land identically. The hidden_states path's ~2.4e-3
NRMSE is the expected bf16 round-trip cost (we cast to bf16 in the
RMSNorm output BEFORE multiplying by weight, per PyTorch's order of
ops; that bf16 round-trip is the sole error source here).

## What's NOT covered (yet)

Four free components from the contract list still need validation:

1. **SwiGLU** - need finer hooks inside `MLP.forward` to capture
   pre-activation and post-activation; bf16 weights for `linear_fc1`
   per expert, plus the activation function evaluation. ~1 hour of
   harness extension.
2. **GQA path** (post-CCA standard attention) - need self_attn
   internal hooks (post-RoPE Q, post-RoPE K, post-attention before
   o_proj). ~2 hours of harness extension.
3. **partial_rotary_factor=0.5** - ZAYA1 rotates first 64 of 128 head
   dims. Validation needs Q before-rotary and after-rotary intermediate
   hooks. ~1 hour.
4. **MLP-based MoE router + top-1 routing** - have router input/output
   captured; need to run my CPU router_mlp + softmax + top-k=1 against
   it. Need router weight tensors (router_mlp.0/1/2 + balancing biases)
   extracted. ~2 hours.

These are landable as follow-up commits once the relevant weights are
extracted to a larger subset file (the current subset is 38 KB, only
covers RMSNorm + res_scale).

## Implementation notes

CPU impls live in `examples/cpu_validate_phase2.rs`:

- `zaya_rmsnorm_cpu` follows `ZayaRMSNorm.forward` (modeling_zaya.py:167)
  exactly, including the f32-variance / bf16-roundtrip / cast-then-multiply
  ordering. Doing the cast on the row before applying the weight is
  what gives the matching NRMSE; doing it after produces ~5e-3.
- `residual_scaling_cpu` is a straight elementwise `(x + bias) * scale`
  per `ResidualScaling.forward` (modeling_zaya.py:907).
- `nrmse` uses f64 accumulators internally to avoid measurement-side
  precision artifacts.

The bf16 round-trip in RMSNorm matters for matching; without it (i.e.
keeping the variance-scaled rows in f32 and multiplying by f32 weight
directly) the NRMSE jumps to ~5e-3, right at the threshold. With the
PyTorch-faithful order, we land at 1.65e-3 with comfortable headroom.

## Hardware

CPU only. The validator runs on the dev box (no GPU needed). All
activations and weights are CPU-resident; the validation took
~150 ms total for all 6 sites.

## RDNA target consideration

The CPU NRMSE numbers set the bar for the eventual RDNA kernel. When
these components land as HIP kernels for gfx1201:

- RMSNorm on gfx1201: wave32 reduction across hidden_size=2048 with
  the f32 variance computation in registers, then packed-fp16
  multiply by weight. Existing rmsnorm.hip (per qwen35) is the
  template; the only ZAYA1 bit is the weight tensor name and the
  f32 cast ordering. Should land at the same 1.65e-3 NRMSE.
- ResidualScaling: pure elementwise add+mul+add+mul; either fused
  with the residual-add pre-norm path or a small standalone kernel.
  The 0e0 NRMSE on the residual path suggests it can be bit-exact in
  the kernel too, given the same f32 accumulation order.

## Next steps

1. Extract a larger weight subset on hiptrx (router weights + a couple
   of expert FCs + RoPE inv_freq) and scp back.
2. Add SwiGLU + partial-RoPE + MLP-router validators to
   `cpu_validate_phase2.rs`.
3. Re-dump with finer hooks inside MLP and self_attn.

These do not gate any other phase. The Phase 6.A decision (per
MANUAL_REVIEW.md) is the actual scoping bottleneck for end-to-end
ZAYA1.
