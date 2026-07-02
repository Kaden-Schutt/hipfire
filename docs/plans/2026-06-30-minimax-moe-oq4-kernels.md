# Indexed-MoE OQ4 kernels — MiniMax expert OQ4++ (Stage 2)

Status: design + study DONE 2026-06-30. Implementation not started.
Prereq: Stage 1 (dense OQ4++/router OQ8++) committed `bb7e2583d`.

## Why

`--format oq4++` on MiniMax (arch 10) currently quantizes dense attention →
OQ4++ and the router → OQ8++ (commit bb7e2583d), but **routed experts stay
MQ4** because there are NO indexed-MoE OQ GEMV kernels. Only the
`gemv_hfq4g256_moe_*` / `hfq6` / `paro` MQ-family indexed kernels exist in
`kernels/src/`. LFM2 and Qwen hit the same wall ("until the indexed MoE OQ
kernels exist"). Experts are the bulk of the MoE, so a full OQ4++ MoE needs
these kernels. This benefits lfm2/qwen/minimax alike.

## Format delta (the whole trick)

HFQ4 (mq4) indexed expert block = **136 B/group**:
`[f32 scale | f32 zero-point | 128 unsigned nibbles]`, dequant
`w = sc*nib + zp` (affine/asymmetric).

OQ4G256 = **132 B/group** `[f32 scale | 128 SIGNED nibbles]`, dequant
`w = sext4(nib) * sc` (symmetric, NO zero-point). This is exactly the
`INTERLEAVED_BLOCK` (132) section that `pack_oq4_arch_combined`
(crates/hipfire-arch-minimax/src/minimax.rs) already produces. At decode the
dense OQ4 path (`gemv_oq4_grouped.hip`) runs W4A16 (unpack int4 × f32 act) —
same as mq4 — so the MoE OQ4 kernels are W4A16 too (no act int8-quant needed).

So each OQ4 MoE kernel = the corresponding HFQ4 kernel with:
- block stride 136 → **132**
- scale at `gp+0` (same), nibbles at `gp+4` (was `gp+8`), drop the `zp` read
- dequant `(sc*nib + zp)` → `sc * sext4(nib)` where
  `sext4(n) = ((int)n << 28) >> 28` per nibble.

## Kernels to write (kernels/src/)

Mirror the HFQ4 entry points minimax/forward.rs dispatches:
1. `gemv_oq4g256_moe_gate_up_k8_indexed`            (decode; from gemv_hfq4g256_moe_gate_up_indexed.hip)
2. `gemv_oq4g256_moe_gate_up_k8_indexed_batched`    (prefill; from _gate_up_indexed_batched.hip)
3. `gemv_oq4g256_moe_down_k8_indexed_batched_expanded` (from _down_k8_indexed_batched_expanded.hip)
4. (reuse) `moe_down_combine_k8_batched` — operates on f32 expert outputs, dtype-agnostic.

Optional wave64 variants later (perf only; correctness path is wave32).

## Integration

- **dispatch (rdna-compute)**: add `gpu.gemv_oq4g256_moe_*` methods + KernelKey
  entries; add kernel-source names to the hash/registry list (see
  gen_kernel_hashes). Mirror the hfq4 method signatures exactly.
- **quantizer (main.rs)**: in the is_minimax expert branch, when an OQ format is
  requested, emit each expert w1/w2/w3 as OQ4G256 (imatrix-AWQ via
  `imatrix_weights_for`, NO per-expert LDLQ — no per-expert Hessian fits).
- **loader (minimax.rs)**: fuse per-expert OQ4 w1‖w3 into the gate_up blob in the
  132B interleaved layout and w2 into the down blob; build the expert ptr tables
  (already done for MQ4 — add the OQ4 dtype branch). Repack on-disk OQ4G256
  (130B [f16 scale|128 nib]) → 132B [f32 scale|128 nib] per group, per expert.
- **forward (forward.rs)**: dispatch the oq4 kernels when expert dtype is Oq4G256
  (add arms next to the existing hfq4/hfq6/mq2l/mq3l arms at L264/336/557/617/1093).
- **calib (calibration.rs + forward.rs)**: per-expert imatrix capture via taps
  around the indexed-expert GEMV (lfm2 `decode_step_capture` pattern); names
  `model.layers.{l}.block_sparse_moe.experts.{e}.w{1,2,3}`, imatrix-only.

## Validation

Tiny fixture: emit-fixture minimax → quantize oq4++ (now experts OQ4 too) →
tiny_quant_probe ar-hash loads+decodes; compare logits vs the MQ4-expert build
for sanity. Add a kernel-level CPU-reference check if practical. Coherence gate
after, since this touches kernels/quant/dispatch.
```
```
