# hipfire vs hipEngine ParoQuant comparison — what to port for Qwen3.6 / A3B

Scope: identify the concrete forward-path and loader divergences between
hipfire's `feat/paroquant-native` (last `3abbfa4c`, Qwen3-0.6B and Qwen3.5-0.8B
dense working, Qwen3.6 attractor) and shisa-ai's hipEngine reference
(`/tmp/hipengine-survey`, target `Qwen3.6-35B-A3B-PARO-full4096-e5-packed`).

Read-only investigation. All citations are `file:line`.

---

## Top-5 concrete diffs (ranked by suspected impact)

### 1. GemmaRMSNorm `(1 + weight)` offset is dropped on the ParoQuant load path

hipfire's ParoQuant loader bypasses the `+= 1.0` bake that the HFQ loader
performs:

- `crates/hipfire-arch-qwen35/src/qwen35.rs:778-792` — `load_norm_weight` (HFQ
  path) explicitly does `for v in &mut f32_data { *v += 1.0; }` with the
  comment `// Qwen3.5 RMSNorm: output = x * rsqrt(var+eps) * (1 + weight)`.
- `crates/hipfire-arch-qwen35/src/qwen35.rs:1794-1804` — `paro_load_norm`
  (does `+= 1.0`) is **defined but never called**.
- `crates/hipfire-arch-qwen35/src/qwen35.rs:1806-1815` — `paro_load_norm_raw`
  (does NOT add 1.0) is the only function used at qwen35.rs:1843, 1845,
  1869, 1879, 1889, 1894, 1895, 1896.

hipEngine's rmsnorm kernels add the `1.0f` **inside the kernel**:

- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/norm/rmsnorm.hip:80`
  `const float scale = 1.0f + bf16_bits_to_float(weight[idx]);` (repeated at
  lines 116, 164-171, 199, 231).
- `/tmp/hipengine-survey/hipengine/loading/qwen35_paro.py:1496-1506` documents
  the convention: "Parent `load_paro_rmsnorm_module` first turns checkpoint
  offsets into direct scales in FP16 via `(weight + 1)`."

hipfire's `gpu.rmsnorm_f32` expects pre-baked `(1 + weight)`. So under
ParoQuant load, every `attn_norm`, `ffn_norm`, `q_norm`, `k_norm`, and the
final `output_norm` runs with weights shifted by ~−1.0. For Qwen3.5/Qwen3.6
GemmaRMSNorm this corrupts every layer — LESSONS-LEARNED.md:585-590 calls out
exactly this: "Materializing effective normal RMSNorm weights recovered the
same slice to KLD 0.1916 and made layer 0 match HF at FP16-level error."

Qwen3-0.6B and Qwen3.5-0.8B may still emit coherent tokens because the
quantizer for the small models possibly already pre-baked the `+1` into the
saved weights, or because the dense small models tolerate the offset error;
the 27B-A3B model is qualitatively more sensitive. (Cannot fully verify
without inspecting the actual .safetensors; high-confidence suspect.)

### 2. MoE FFN loader is unimplemented — panics on Qwen3.6/A3B

- `crates/hipfire-arch-qwen35/src/qwen35.rs:1902` — `_ => panic!("ParoQuant
  MoE loading not yet implemented (layer {i})")`.

hipfire's HFQ MoE loader (qwen35.rs:2143-2210, `load_moe_ffn`) expects HFQ
quantizer output naming, which is **incompatible** with the shisa-ai PARO
checkpoint:

| Concept             | hipfire HFQ loader (qwen35.rs:2155-2179) | hipEngine PARO checkpoint (qwen35_paro.py:373-432) |
|---------------------|------------------------------------------|----------------------------------------------------|
| Router              | `mlp.gate.weight` (size `n_exp`)         | source `mlp.gate.weight` + `mlp.shared_expert_gate.weight` are concatenated row-wise into `mlp.router_shared_gate.weight` size `n_exp+1` |
| Shared-expert gate  | `mlp.shared_expert_gate.weight` (1×dim)  | fused into router (last row of `router_shared_gate.weight`) |
| Routed experts      | per-expert `experts.{X}.gate_up_proj.weight` (fused gate∥up) + `experts.{X}.down_proj.weight`, separate quant per expert | per-expert `experts.{X}.{gate_proj,up_proj,down_proj}.{qweight,qzeros,scales}` **stacked across experts** at load into `mlp.experts.stacked_{gate,up,down}_qweight_pack8_decode` + `_qzeros` + `_scales` |
| Rotation metadata   | per-expert (one `ParoRotation` per WeightTensor) | **shared across experts**: `mlp.experts.{gate_up,down}_weight_{pairs,theta,channel_scales}` (one rotation for all expert gates+ups, one for all expert downs) — `qwen35_paro.py:5590-5604, 5644-5662` |
| Shared expert       | three FP16 dense tensors                 | three packed PARO W4 sidecars `shared_expert.{gate_proj,up_proj,down_proj}.{qweight,qzeros,scales,pairs,theta,channel_scales}` (qwen35_paro.py:5777-5853) |

This means hipfire cannot just unpack the panic — even reading the same
safetensors keys, the file does NOT contain hipfire's expected `gate_up_proj`
fused tensor. The repack-AWQ→HFQ4G128 helper at
`crates/hipfire-arch-qwen35/src/qwen35.rs:1096` repacks one matrix at a time
and would need a new code path for "stacked across experts" expert weights.

### 3. LinearAttention `in_proj_a/b` are dense `.weight`, not ParoQuant — already partially handled but lossy

- `crates/hipfire-arch-qwen35/src/qwen35.rs:1872-1873` — calls `paro_load_wt`
  on `linear_attn.in_proj_a` and `.in_proj_b`. The fallback at
  qwen35.rs:1785-1791 detects "no `.qweight` present" and falls back to
  `load_fp16_weight_from_source`, which converts to **F32** and stores as
  `DType::F32` (qwen35.rs:1145).

hipEngine treats these as dense BF16 and calls a dual GEMV directly on the
**un-rotated** hidden state:

- `/tmp/hipengine-survey/hipengine/runtime/qwen35_paro.py:1110-1123` —
  `dense_dual_gemv_out_bf16(hidden.ptr, a_weight.ptr, b_weight.ptr, ...)`. No
  rotation, no AWQ scale.
- `/tmp/hipengine-survey/hipengine/loading/qwen35_paro.py:342-343, 1996-1997`
  declares the shapes `(linear_num_value_heads, hidden_size)` with raw
  `.weight` semantics.

hipfire's `forward_from_x_gpu` (qwen35.rs:2763-2765) feeds these through
`weight_gemv` on the ALREADY-rmsnormed `tmp` — same as hipEngine reads the
hidden state. Likely correct in principle once the norm in #1 is fixed; the
F32 vs FP16 storage costs ~2× memory but is numerically fine.

### 4. Alpha/beta gate transform is computed in a separate pre-kernel; GDN expects post-transformed inputs

- `crates/hipfire-arch-qwen35/src/qwen35.rs:2766-2768` — hipfire calls
  `fused_sigmoid_alpha_gate_f32(&beta_out, &alpha_out, &dt_bias, &a_log, n_v_heads)`
  **before** conv1d. After this, `alpha_out` holds `exp(-exp(a_log) *
  softplus(alpha + dt_bias))` and `beta_out` holds `sigmoid(beta)`.

hipEngine performs the **identical** transform but **fused inside the GDN
recurrence kernel**, reading raw `a/b/dt_bias/a_log`:

- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip:148-149`
  `const float beta = sigmoid_f32(...b[v_head]); const float decay = expf(-expf(a_log[v_head]) * softplus_f32(...a[v_head] + dt_bias[v_head]));`

This is **semantically equivalent** assuming the same scalar transform; not
an immediate suspect but worth a numerical-equivalence sanity check, because
hipfire's `fused_sigmoid_alpha_gate_f32` (rdna-compute/src/dispatch.rs:22219)
writes back into the same buffers, and the call site at qwen35.rs:2766
passes them by `&` — if any buffer is reused before the kernel completes
async, you have a race. (Lower likelihood; verify by reading the kernel
body.)

### 5. RoPE: partial-interleaved layout assumption

hipfire's full-attention forward (qwen35.rs:2908-2911) applies
`rope_partial_interleaved_f32` with `n_rot = head_dim * partial_rotary_factor`
expected to be 64 for Qwen3.5/3.6 (`head_dim=256`,
`partial_rotary_factor=0.25`). LESSONS-LEARNED.md:591-597 confirms this is
correct for Qwen3.5; hipEngine carries `rotary_dim` from `ModelSpec`
(loading/qwen35_paro.py:130 `rotary_dim = int(head_dim * partial_rotary_factor)`).

The config-parse path in hipfire (qwen35.rs:179-181) reads
`partial_rotary_factor` from `config.text_config.partial_rotary_factor` OR
nested `rope_parameters.partial_rotary_factor`. Shisa-ai's Qwen3.6-A3B may
ship the factor under a different key (HF Qwen3.6 ships it directly under
`text_config.partial_rotary_factor`). Worth double-checking against the
actual JSON. Low-likelihood culprit (would also break Qwen3.5-0.8B).

---

## MoE loader sketch — Rust pseudocode for qwen35.rs:75 panic arms

hipfire MUST pick a stacking strategy. Two viable approaches:

**Option A (lower-effort):** Keep hipfire's per-expert layout, but read
shisa-ai's per-expert keys directly (`experts.{X}.gate_proj.qweight` etc.)
and locally repack into hipfire's fused `gate_up` HFQ4G128 form. Reuses
existing `moe_ffn_decode` dispatch (qwen35.rs:2292+) unchanged.

**Option B (matches hipEngine):** Use the stacked layout + indexed dispatch.
Lower memory fragmentation; aligns with hipEngine's grouped/c1 kernels.
Requires new kernels and new `WeightTensor` arity (one rotation shared
across experts) — bigger lift.

Sketch for Option A (Qwen3.6 dense-MoE A3B):

```rust
// File: crates/hipfire-arch-qwen35/src/qwen35.rs
// Replaces the panic at qwen35.rs:1902 inside `load_weights_paroquant`.

(LayerType::FullAttention, true) | (LayerType::LinearAttention, true) => {
    // Build the same DeltaNetMoeLayerWeights / FullAttnMoeLayerWeights as the
    // HFQ paths (qwen35.rs:1727-1764), but populate `ffn` via a new
    // `paro_load_moe_ffn` that walks PARO sidecar keys instead of HFQ ones.

    let attn_block = /* same as the !is_moe arms, with paro_load_wt/raw_norm */;

    let ffn = paro_load_moe_ffn(source, gpu, &p, config, i as u16)?;
    // The non-FFN attn block matches qwen35.rs:1864-1900 (DeltaNet) or
    // a new ParoQuant FullAttention arm modeled on qwen35.rs:1885-1900.

    layers.push(LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
        attn_norm: paro_load_norm(...) // NOTE: use _norm (with +1), NOT _raw
        // ... rest as in the dense arm ...
        ffn,
    }));
}

fn paro_load_moe_ffn(
    source: &dyn ModelSource, gpu: &mut Gpu, p: &str,
    config: &Qwen35Config, layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let dim = config.dim;
    let qc = source.quant_config().unwrap();
    let (gs, kr) = (qc.group_size, qc.krot);

    // Router. shisa-ai ships `mlp.router_shared_gate.weight` size [n_exp+1, dim]
    // (rows 0..n_exp are router, row n_exp is the shared_expert_gate scalar).
    // If only the legacy split keys are present, fall back to concatenating
    // them at load (matches qwen35_paro.py:1052-1056).
    let router = paro_load_combined_router(source, gpu, p, n_exp, dim)?;
    let shared_expert_gate = /* slice last row of router OR load standalone */;

    // Shared expert: PARO W4 packed sidecars (paro_load_wt fallback handles
    // both `.qweight` present and dense `.weight` fallback).
    let shared_expert = SharedExpertWeights {
        gate: paro_load_wt(source, gpu, &format!("{p}.mlp.shared_expert.gate_proj"), smi, dim, gs, kr)?,
        up:   paro_load_wt(source, gpu, &format!("{p}.mlp.shared_expert.up_proj"),   smi, dim, gs, kr)?,
        down: paro_load_wt(source, gpu, &format!("{p}.mlp.shared_expert.down_proj"), dim, smi, gs, kr)?,
    };

    // Routed experts. shisa-ai's checkpoint stores them per-expert as
    // `experts.{X}.gate_proj/up_proj/down_proj.{qweight,qzeros,scales,
    // pairs,theta,channel_scales}` — NOT fused into gate_up_proj. We fuse
    // gate||up at load to match hipfire's existing `gate_up` WeightTensor
    // shape, then build the device-side pointer table.
    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let g = paro_load_wt(source, gpu, &format!("{p}.mlp.experts.{x}.gate_proj"), mi, dim, gs, kr)?;
        let u = paro_load_wt(source, gpu, &format!("{p}.mlp.experts.{x}.up_proj"),   mi, dim, gs, kr)?;
        let gate_up = fuse_gate_up(gpu, g, u, mi, dim)?; // concat rows on GPU
        let down = paro_load_wt(source, gpu, &format!("{p}.mlp.experts.{x}.down_proj"), dim, mi, gs, kr)?;
        experts.push(ExpertWeights { gate_up, down });
    }

    // Pointer tables — identical to qwen35.rs:2188-2199.
    /* gu_ptrs / dn_ptrs / expert_gate_up_ptrs / expert_down_ptrs ... */

    Ok(MoeFfnWeights {
        router, experts, shared_expert, shared_expert_gate,
        expert_gate_up_ptrs, expert_down_ptrs, layer_idx, expert_shape: None,
    })
}
```

Caveat: hipEngine's PARO checkpoint ships ONE rotation tuple per stacked
projection (`mlp.experts.gate_up_weight_pairs/theta/channel_scales`), shared
across all experts. If a shisa-ai checkpoint is in stacked form, the
`paro_load_wt` per-expert calls above will not find per-expert `pairs/theta/
channel_scales` and will need either (a) a stacked-expert kernel or (b) a
broadcast helper that attaches the shared rotation to each expert
WeightTensor's `paro` field. Option (b) is the smallest delta.

For the **shared_expert_gate** scalar gate: shisa-ai fuses it into
`router_shared_gate.weight` as row `n_exp`. The forward path
(qwen35.rs:2498-2499) currently does two separate weight_gemv calls
(router and shared_gate). Either:
- Load row n_exp into a separate `WeightTensor` of shape `(1, dim)` to keep
  existing dispatch, OR
- Add a fused router+gate weight_gemv that emits `[n_exp+1]` and slice.
Option (a) is the smallest delta to ship.

---

## Safetensors key map (shisa-ai PARO, A3B)

From `/tmp/hipengine-survey/hipengine/loading/qwen35_paro.py:279-486`:

```
# Top-level
model.language_model.embed_tokens.weight        # FP16, tied to lm_head
model.language_model.norm.weight                # FP16, needs +1 bake

# Per layer (linear_attention)
layers.{L}.input_layernorm.weight                                  # FP16 +1
layers.{L}.post_attention_layernorm.weight                          # FP16 +1
layers.{L}.linear_attn.in_proj_{qkv,z,out_proj}.{qweight,qzeros,scales,theta,pairs,channel_scales}
layers.{L}.linear_attn.in_proj_{a,b}.weight                         # dense FP16
layers.{L}.linear_attn.conv1d.weight                                # FP32, shape [qkv_width, 1, kernel_dim]
layers.{L}.linear_attn.A_log                                        # FP32 [n_v_heads]
layers.{L}.linear_attn.dt_bias                                      # FP32 [n_v_heads]
layers.{L}.linear_attn.norm.weight                                  # FP32 [linear_value_head_dim] +1

# Per layer (full_attention)
layers.{L}.self_attn.{q,k,v,o}_proj.{qweight,qzeros,scales,theta,pairs,channel_scales}
layers.{L}.self_attn.q_norm.weight                                  # FP16 +1 (head_dim)
layers.{L}.self_attn.k_norm.weight                                  # FP16 +1 (head_dim)

# MoE FFN (per layer, when num_experts > 0)
#   RAW upstream shisa-ai keys (what `from_pretrained` writes):
layers.{L}.mlp.gate.weight                                          # FP16 [n_exp, dim]   — router
layers.{L}.mlp.shared_expert_gate.weight                            # FP16 [1, dim]
layers.{L}.mlp.experts.{X}.gate_proj.{qweight,qzeros,scales,theta,pairs,channel_scales}
layers.{L}.mlp.experts.{X}.up_proj.{qweight,qzeros,scales,theta,pairs,channel_scales}
layers.{L}.mlp.experts.{X}.down_proj.{qweight,qzeros,scales,theta,pairs,channel_scales}
layers.{L}.mlp.shared_expert.{gate_proj,up_proj,down_proj}.{qweight,qzeros,scales,theta,pairs,channel_scales}

#   hipEngine-derived stacked keys (built at LOAD time from RAW; see
#   qwen35_paro.py:1022-1107 prepare_qwen35_paro_moe_c1_host_tensors):
layers.{L}.mlp.router_shared_gate.weight                            # FP16 [n_exp+1, dim] = concat(gate, shared_expert_gate)
layers.{L}.mlp.experts.stacked_{gate,up,down}_qweight               # stacked over expert dim
layers.{L}.mlp.experts.stacked_{gate,up,down}_qweight_pack8_decode  # transpose-packed for decode kernels
layers.{L}.mlp.experts.stacked_{gate,up,down}_qzeros
layers.{L}.mlp.experts.stacked_{gate,up,down}_scales
layers.{L}.mlp.experts.{gate_up,down}_weight_{pairs,theta,channel_scales}   # shared rotation across all experts
```

Hipfire SHOULD prefer reading the raw upstream keys
(`mlp.gate.weight`, per-expert `experts.{X}.{...}.qweight`) directly and skip
hipEngine's load-time stacking. That avoids the stacked kernel surface and
matches hipfire's per-expert dispatch.

The `ROOT_PREFIXES` normalize (qwen35_paro.py:104-108) strips
`model.language_model.` from all names — hipfire already prepends it (e.g.
qwen35.rs:1786 `format!("model.language_model.{prefix}")`). Match.

---

## Risk list — what could still go wrong after porting these diffs

1. **Quantizer-side bug already in the checkpoint.** Task notes that z-lab's
   `Qwen3.6-27B-PARO` drops FP16 fallback weights for MoE — shisa-ai's
   `Qwen3.6-35B-A3B-PARO-full4096-e5-packed` is the fix. If we accidentally
   download z-lab's checkpoint instead of shisa-ai's, the attractor will
   reappear regardless of code-side fixes.
2. **`(1 + weight)` semantics for `linear_attn.norm.weight`.** The Gemma
   convention applies to the full-dim norms; whether it applies to the
   per-head `linear_attn.norm.weight` (loaded via `paro_load_f32` at
   qwen35.rs:1877) is unverified. hipEngine's GDN kernel at gdn.hip:243 does
   `* norm_weight[idx]` directly without `+1.0f`, suggesting NO offset on
   this one. Treat as no-offset; do not bake +1 here.
3. **`partial_rotary_factor` config-key drift.** Qwen3.6 may surface it
   under a different JSON key than Qwen3.5; if hipfire reads `0.0` it does
   full-head RoPE and corrupts everything (LESSONS-LEARNED.md:591-597
   matches this failure mode exactly — KL ~12.7 collapse).
4. **Tied embedding vs separate lm_head.** hipfire's ParoQuant loader
   (qwen35.rs:1849-1855) assumes tied embeddings — uses
   `embed_tokens.weight` for `lm_head`. If Qwen3.6-A3B ships an explicit
   `lm_head.weight` (as hipEngine's `model.language_model.lm_head` or
   `lm_head` per `_runner.py:510`), hipfire will silently use the wrong
   tensor.
5. **Per-expert vs stacked rotation broadcast.** If the shisa-ai checkpoint
   stores ONLY the stacked `experts.gate_up_weight_{pairs,theta,
   channel_scales}` and not per-expert pairs, the `paro_load_wt` per-expert
   call will panic on `pairs not found`. Need a stacked-mode helper.
6. **Expert weight ordering within stacked tensors.** hipEngine's
   `_stack_expert_refs` (loading/qwen35_paro.py:1058+) presumably stacks in
   ascending expert index; if hipfire reads them per-index it implicitly
   matches, but if hipfire ever loads the `stacked_*` form, the in-stack
   ordering convention must be preserved.
7. **FP16 vs F32 storage for `in_proj_a/b`.** `load_fp16_weight_from_source`
   stores as F32 (qwen35.rs:1145). Decode `weight_gemv` on `DType::F32`
   takes the dense-FP32 path with no rotation/AWQ scale — correct for these,
   but uses 2× memory. Cosmetic; not a correctness risk.
8. **Conv1d weight shape.** hipEngine validates
   `(qkv_width, 1, kernel_dim)` (loading/qwen35_paro.py:1998); hipfire
   reads a flat `qkv_dim * conv_kernel_dim` buffer (qwen35.rs:1876).
   Compatible because the data is contiguous, but if shisa-ai's safetensors
   ships `(1, qkv_width, kernel_dim)` (some HF dumps do this), hipfire will
   read transposed garbage. Verify the on-disk order.
9. **Fused-pre-conv `sigmoid_alpha_gate` race.** hipfire's pattern
   (compute alpha/beta in a separate kernel, then read in GDN) requires a
   stream barrier between them. If the two kernels happen to be on
   different streams (or capture into the same hipGraph node), reads in
   GDN may see stale values. Check `gpu.fused_sigmoid_alpha_gate_f32` and
   `gpu.gated_delta_net_q8` for stream coherency.
10. **No `prompt_normalize` reproducibility.** Per CLAUDE.md "Prompt-structure
    τ sensitivity," any A/B comparison against hipEngine must use
    byte-identical prompts. The 4-gram attractor itself may be marginally
    sensitive to prompt whitespace; do not compare hipfire vs hipEngine
    outputs on different bytes.
