// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 in-memory weight layout: per-layer weight structs, the MoE FFN
//! weight bundle, the `Qwen35Weights` model container, and GPU-storage /
//! free helpers.

use super::*;

/// Weights for a DeltaNet (linear attention) layer.
pub struct DeltaNetLayerWeights {
    pub attn_norm: GpuTensor,   // input_layernorm [dim]
    pub wqkv: WeightTensor,     // in_proj_qkv [6144, dim] → Q+K+V concat
    pub wz: WeightTensor,       // in_proj_z [2048, dim] → gate Z
    pub w_alpha: WeightTensor,  // in_proj_a [n_heads, dim] → decay
    pub w_beta: WeightTensor,   // in_proj_b [n_heads, dim] → update
    pub a_log: GpuTensor,       // A_log [n_heads] — learnable log-decay
    pub dt_bias: GpuTensor,     // dt_bias [n_heads]
    pub conv_weight: GpuTensor, // conv1d.weight [conv_channels, 1, 4] → F32
    pub norm_weight: GpuTensor, // norm.weight [head_dim] — gated output norm
    pub wo: WeightTensor,       // out_proj [dim, d_inner]
    pub ffn_norm: GpuTensor,    // post_attention_layernorm [dim]
    pub w_gate: WeightTensor,   // mlp.gate_proj
    pub w_up: WeightTensor,     // mlp.up_proj
    pub w_down: WeightTensor,   // mlp.down_proj
    pub bf16_down_shadow: Option<Bf16DownShadow>,
}

/// Weights for a full attention (gated) layer — similar to Qwen3 but with q+gate split.
pub struct FullAttnLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,  // q_proj [4096, dim] — 2x wide (query + gate)
    pub wk: WeightTensor,  // k_proj
    pub wv: WeightTensor,  // v_proj
    pub wo: WeightTensor,  // o_proj
    pub q_norm: GpuTensor, // q_norm [head_dim]
    pub k_norm: GpuTensor, // k_norm [head_dim]
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
    pub bf16_down_shadow: Option<Bf16DownShadow>,
}

// ─── MoE FFN weights (Qwen3.5-MoE / A3B) ────────────────────────────────
//
// Replaces the dense (w_gate, w_up, w_down) triple with N+1 expert FFNs
// gated by a router, plus a shared always-on expert.
//
// A3B specifics:
//   num_experts = 256, top_k = 8, moe_intermediate = 512, hidden = 2048
//   shared_expert_intermediate = 512 (same as routed)
//
// Per-layer storage:
//   router:               [num_experts, hidden]  MQ4G256 / Q8
//   shared_expert_gate:   [1, hidden]            MQ4G256 / Q8 — projects to scalar
//   experts[X].gate_up:   [2*moe_intermediate, hidden]  MQ4G256
//   experts[X].down:      [hidden, moe_intermediate]    MQ4G256
//   shared_expert.gate:   [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.up:     [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.down:   [hidden, shared_expert_intermediate]   MQ4G256
//
// The quantizer (hipfire-quantize) splits the safetensors 3D
// `mlp.experts.gate_up_proj` / `down_proj` tensors per-expert into
// `mlp.experts.{X}.gate_up_proj.weight` / `down_proj.weight` so the loader
// can fish them out by index. The shared expert is stored with separate
// gate_proj + up_proj + down_proj (it is not fused in safetensors either).

pub struct ExpertWeights {
    pub gate_up: WeightTensor, // [2 * moe_intermediate, hidden] — fused (gate || up)
    pub down: WeightTensor,    // [hidden, moe_intermediate]
}

/// Shared expert storage — unlike routed experts, gate_proj and up_proj are
/// NOT fused in the safetensors, so we keep them separate here too. The
/// forward path does two GEMVs + silu_mul + down GEMV.
pub struct SharedExpertWeights {
    pub gate: WeightTensor, // [shared_expert_intermediate, hidden]
    pub up: WeightTensor,   // [shared_expert_intermediate, hidden]
    pub down: WeightTensor, // [hidden, shared_expert_intermediate]
}

pub struct MoeFfnWeights {
    pub router: WeightTensor, // [num_experts, hidden]
    /// Routed expert weights. Populated when this layer is fully resident
    /// (`paged_experts == false`); **empty `Vec`** when `paged_experts == true`
    /// (the [`hipfire_runtime::weight_pager::WeightPager`] owns the buffers, and the
    /// indexed kernels read pointers from `expert_*_ptrs` which the pager
    /// patches per-token via `patch_expert_ptr_table`).
    pub experts: Vec<ExpertWeights>, // num_experts (= 256 for A3B); empty in paged mode
    pub shared_expert: SharedExpertWeights,
    pub shared_expert_gate: WeightTensor, // [1, hidden] — row-vector projecting to scalar
    /// Device-side array of `unsigned long long` pointers, one per
    /// expert's `gate_up.buf`. Indexed at runtime by the GPU top-K
    /// kernel's output so the indexed MoE GEMV can stay capture-safe.
    pub expert_gate_up_ptrs: GpuTensor, // [num_experts * 2] f32 slots = num_experts × u64
    pub expert_down_ptrs: GpuTensor,      // [num_experts * 2] f32 slots = num_experts × u64

    /// Layer index. Stable identity used to key
    /// [`hipfire_runtime::weight_pager::WeightId::Expert`] entries.
    pub layer_idx: u16,

    /// Per-expert tensor shapes. `None` in non-paged mode (shapes are read
    /// from `experts[i].gate_up.{m, k}` etc.); `Some` in paged mode where
    /// `experts` is empty but kernels still need m/k for kernel-arg setup.
    /// Qwen3.5-MoE-A3B has uniform per-expert shape so one descriptor per
    /// layer suffices for v0.1.
    pub expert_shape: Option<hipfire_runtime::weight_pager::ExpertShape>,
    pub expert_gate_up_dtype: Option<DType>,
    pub expert_down_dtype: Option<DType>,

    /// ParoQuant only: shared per-layer rotation sidecars for the routed
    /// experts. shisa-ai's PARO checkpoint quantizes all 256 experts with
    /// one rotation tuple per projection-group (gate||up vs down), so we
    /// upload the sidecars ONCE per layer and broadcast a non-owning
    /// `ParoRotation` (built via `DeviceBuffer::from_raw`) into every
    /// `ExpertWeights.gate_up.paro` / `ExpertWeights.down.paro`. The
    /// owning storage lives here so the aliases stay valid for the
    /// lifetime of the layer. `None` for HFQ MoE (per-tensor PARO sidecars
    /// or no PARO at all).
    pub paro_shared: Option<MoeParoSidecars>,
}

/// Owning storage for the per-layer shared ParoQuant rotation sidecars.
/// One tuple per projection-group:
///   - `gate_up_*`: applied to the post-RMSNorm hidden activation (K = hidden_dim).
///     Shared by all 256 experts' gate AND up projections, and by the fused
///     gate_up `WeightTensor`'s `paro` alias.
///   - `down_*`: applied to the post-SiLU intermediate activation (K = mi).
///     Shared by all 256 experts' down projection.
pub struct MoeParoSidecars {
    pub gate_up_pairs: GpuTensor,
    pub gate_up_theta: GpuTensor,
    pub gate_up_channel_scales: GpuTensor,
    pub down_pairs: GpuTensor,
    pub down_theta: GpuTensor,
    pub down_channel_scales: GpuTensor,
    pub krot: u32,
    pub group_size: u32,
}

pub struct DeltaNetMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wqkv: WeightTensor,
    pub wz: WeightTensor,
    pub w_alpha: WeightTensor,
    pub w_beta: WeightTensor,
    pub a_log: GpuTensor,
    pub dt_bias: GpuTensor,
    pub conv_weight: GpuTensor,
    pub norm_weight: GpuTensor,
    pub wo: WeightTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub struct FullAttnMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub enum LayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
    // A3B / qwen3_5_moe: same attention as above, MoE FFN instead of dense.
    // Loader + forward path TODO — adding the variants now so the enum is
    // forward-compatible and downstream code that pattern-matches gets a
    // compile-time hint to handle the new case.
    DeltaNetMoe(DeltaNetMoeLayerWeights),
    FullAttnMoe(FullAttnMoeLayerWeights),
}

/// RoughQuant real-format protected-channel correction (one per quantized
/// residual projection that carries a `.rqcorr` sidecar). Applied at GEMV time as
/// `y += R_S · x_S`, restoring the protected channels to bf16 precision over the
/// mq4 bulk. See `docs/roughquant/phase3-real-format-scope.md`.
pub enum RqCorr {
    /// Residual reader (protected input COLUMNS): gather `xs = x_normed[S]` (padded
    /// to power-of-2 `np`), `out += corr[m×np] · xs`.
    Reader {
        corr: GpuTensor, // [m × np] f32
        idx: GpuTensor,  // [n_idx] i32 — residual channel ids (gather source)
        m: usize,
        n_idx: usize,
        np: usize,
    },
    /// Residual writer (protected output ROWS): `c = corr[|S|×k] · input`, then
    /// scatter-add `out[S[j]] += c[j]`.
    Writer {
        corr: GpuTensor, // [n_s × k] f32
        idx: GpuTensor,  // [n_s] i32 — residual channel ids (scatter target)
        n_s: usize,
        k: usize,
    },
}

/// Which projection within a layer a correction targets (keys the side-map).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RqProj {
    Wqkv,
    Wz,
    Walpha,
    Wbeta,
    Wq,
    Wk,
    Wv,
    Wgate,
    Wup,
    Wo,
    Wdown,
}

pub struct Qwen35Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
    pub slab_storage: Option<ModelGpuStorage>,
    /// RoughQuant real-format corrections, keyed by `(layer_idx, projection)`.
    /// Empty for all non-roughquant models (backward-compatible: the forward
    /// applies nothing). Populated by `load_rq_corrections` from the
    /// `metadata["roughquant_sidecar"]` index + `<name>.rqcorr` tensors.
    pub rq_corrections: std::collections::HashMap<(u32, RqProj), RqCorr>,

    /// Weight pager (MAD-93 v0.1). `Some` only when the model was loaded
    /// with `Qwen35Config::paged_experts == true`. The forward path uses
    /// interior mutability (`borrow_mut`) at the MoE dispatch site to call
    /// `ensure_resident` / `patch_expert_ptr_table`. `None` means the model
    /// is fully resident — no behavior change vs main.
    pub pager: Option<std::cell::RefCell<hipfire_runtime::weight_pager::WeightPager>>,
}

impl Qwen35Weights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Qwen35Weights {
            token_embd,
            output_norm,
            output,
            layers,
            pager,
            slab_storage,
            ..
        } = self;
        let slabs = slab_storage.as_ref();
        free_tensor_maybe_slab(gpu, slabs, token_embd);
        free_tensor_maybe_slab(gpu, slabs, output_norm);
        free_weight_tensor_maybe_slab(gpu, slabs, output);
        for layer in layers {
            match layer {
                LayerWeights::DeltaNet(l) => {
                    free_tensor_maybe_slab(gpu, slabs, l.attn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wqkv);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wz);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_alpha);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_beta);
                    free_tensor_maybe_slab(gpu, slabs, l.a_log);
                    free_tensor_maybe_slab(gpu, slabs, l.dt_bias);
                    free_tensor_maybe_slab(gpu, slabs, l.conv_weight);
                    free_tensor_maybe_slab(gpu, slabs, l.norm_weight);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wo);
                    free_tensor_maybe_slab(gpu, slabs, l.ffn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_gate);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_up);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_down);
                }
                LayerWeights::FullAttn(l) => {
                    free_tensor_maybe_slab(gpu, slabs, l.attn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wq);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wk);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wv);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wo);
                    free_tensor_maybe_slab(gpu, slabs, l.q_norm);
                    free_tensor_maybe_slab(gpu, slabs, l.k_norm);
                    free_tensor_maybe_slab(gpu, slabs, l.ffn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_gate);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_up);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_down);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    free_tensor_maybe_slab(gpu, slabs, l.attn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wqkv);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wz);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_alpha);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_beta);
                    free_tensor_maybe_slab(gpu, slabs, l.a_log);
                    free_tensor_maybe_slab(gpu, slabs, l.dt_bias);
                    free_tensor_maybe_slab(gpu, slabs, l.conv_weight);
                    free_tensor_maybe_slab(gpu, slabs, l.norm_weight);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wo);
                    free_tensor_maybe_slab(gpu, slabs, l.ffn_norm);
                    free_moe_ffn_maybe_slab(gpu, slabs, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    free_tensor_maybe_slab(gpu, slabs, l.attn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wq);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wk);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wv);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wo);
                    free_tensor_maybe_slab(gpu, slabs, l.q_norm);
                    free_tensor_maybe_slab(gpu, slabs, l.k_norm);
                    free_tensor_maybe_slab(gpu, slabs, l.ffn_norm);
                    free_moe_ffn_maybe_slab(gpu, slabs, l.ffn);
                }
            }
        }
        // MAD-93 v0.1: in paged mode, the pager owns expert weight allocations
        // (the per-layer `free_moe_ffn` loops ran no-ops since `ffn.experts`
        // was empty). Drain the pager's resident set back to the GPU pool here.
        if let Some(pager_cell) = pager {
            pager_cell.into_inner().free_all(gpu);
        }
        if let Some(storage) = slab_storage {
            storage.free_gpu(gpu);
        }
    }

    /// Multi-GPU companion to `free_gpu`. Each layer freed on its
    /// band-owning device per `gpus.device_for_layer(i)`; `token_embd`
    /// freed on dev 0; `output_norm + output` on `gpus.output_device`.
    /// Mirror of `load_weights_multi` placement. The `pager` field is
    /// always `None` on the multi path (paged-experts is not wired into
    /// pp>1 yet); a non-None pager would need its own per-band drain
    /// strategy and is rejected at load.
    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        debug_assert!(
            self.pager.is_none(),
            "free_gpu_multi: pager must be None on pp>1 path"
        );
        let _ = gpus.devices[0].free_tensor(self.token_embd);
        let out_dev = gpus.output_device;
        let _ = gpus.devices[out_dev].free_tensor(self.output_norm);
        let _ = gpus.devices[out_dev].free_tensor(self.output.buf);
        for (i, layer) in self.layers.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let gpu = &mut gpus.devices[dev_idx];
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wqkv.buf);
                    let _ = gpu.free_tensor(l.wz.buf);
                    let _ = gpu.free_tensor(l.w_alpha.buf);
                    let _ = gpu.free_tensor(l.w_beta.buf);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    let _ = gpu.free_tensor(l.w_gate.buf);
                    let _ = gpu.free_tensor(l.w_up.buf);
                    let _ = gpu.free_tensor(l.w_down.buf);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wq.buf);
                    let _ = gpu.free_tensor(l.wk.buf);
                    let _ = gpu.free_tensor(l.wv.buf);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    let _ = gpu.free_tensor(l.w_gate.buf);
                    let _ = gpu.free_tensor(l.w_up.buf);
                    let _ = gpu.free_tensor(l.w_down.buf);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wqkv.buf);
                    let _ = gpu.free_tensor(l.wz.buf);
                    let _ = gpu.free_tensor(l.w_alpha.buf);
                    let _ = gpu.free_tensor(l.w_beta.buf);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wq.buf);
                    let _ = gpu.free_tensor(l.wk.buf);
                    let _ = gpu.free_tensor(l.wv.buf);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
            }
        }
    }
}

pub fn validate_paged_moe_decode_expert_cache(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
) -> Result<(), String> {
    if !config.paged_experts || config.num_experts == 0 {
        return Ok(());
    }
    let pager = weights
        .pager
        .as_ref()
        .ok_or_else(|| "paged Qwen35-MoE decode requires a weight pager".to_string())?;
    let layer = weights
        .layers
        .iter()
        .find_map(|layer| match layer {
            LayerWeights::DeltaNetMoe(layer) => Some(layer.ffn.layer_idx),
            LayerWeights::FullAttnMoe(layer) => Some(layer.ffn.layer_idx),
            _ => None,
        })
        .ok_or_else(|| "paged Qwen35-MoE decode requires a MoE layer".to_string())?;
    pager
        .borrow()
        .would_fit_largest_expert_module_set(layer, config.num_experts_per_tok)
        .map_err(|e| format!("paged Qwen35-MoE decode expert cache too small: {e}"))
}

fn free_moe_ffn(gpu: &mut Gpu, ffn: MoeFfnWeights) {
    let _ = gpu.free_tensor(ffn.router.buf);
    let _ = gpu.free_tensor(ffn.shared_expert_gate.buf);
    let _ = gpu.free_tensor(ffn.shared_expert.gate.buf);
    let _ = gpu.free_tensor(ffn.shared_expert.up.buf);
    let _ = gpu.free_tensor(ffn.shared_expert.down.buf);
    let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
    let _ = gpu.free_tensor(ffn.expert_down_ptrs);
    for e in ffn.experts {
        let _ = gpu.free_tensor(e.gate_up.buf);
        let _ = gpu.free_tensor(e.down.buf);
    }
    // ParoQuant MoE: free the owning shared sidecars (per-expert `paro` fields
    // alias these and must NOT be freed separately — they're non-owning views).
    if let Some(s) = ffn.paro_shared {
        let _ = gpu.free_tensor(s.gate_up_pairs);
        let _ = gpu.free_tensor(s.gate_up_theta);
        let _ = gpu.free_tensor(s.gate_up_channel_scales);
        let _ = gpu.free_tensor(s.down_pairs);
        let _ = gpu.free_tensor(s.down_theta);
        let _ = gpu.free_tensor(s.down_channel_scales);
    }
}

pub struct ModelGpuStorage {
    pub(crate) slabs: Vec<GpuTensor>,
    pub(crate) bytes: usize,
}

impl ModelGpuStorage {
    pub(crate) fn new(slabs: Vec<GpuTensor>, bytes: usize) -> Self {
        Self { slabs, bytes }
    }

    fn contains_tensor(&self, tensor: &GpuTensor) -> bool {
        let ptr = tensor.buf.as_ptr() as usize;
        self.slabs.iter().any(|slab| {
            let start = slab.buf.as_ptr() as usize;
            let end = start.saturating_add(slab.buf.size());
            ptr >= start && ptr < end
        })
    }

    fn free_gpu(self, gpu: &mut Gpu) {
        for slab in self.slabs {
            let _ = gpu.free_tensor(slab);
        }
    }
}

fn free_tensor_maybe_slab(gpu: &mut Gpu, slabs: Option<&ModelGpuStorage>, tensor: GpuTensor) {
    if slabs.is_some_and(|s| s.contains_tensor(&tensor)) {
        std::mem::forget(tensor);
    } else {
        let _ = gpu.free_tensor(tensor);
    }
}

fn free_weight_tensor_maybe_slab(gpu: &mut Gpu, slabs: Option<&ModelGpuStorage>, wt: WeightTensor) {
    if let Some(scale) = wt.awq_scale {
        let _ = gpu.free_tensor(scale);
    }
    free_tensor_maybe_slab(gpu, slabs, wt.buf);
}

fn free_moe_ffn_maybe_slab(gpu: &mut Gpu, slabs: Option<&ModelGpuStorage>, ffn: MoeFfnWeights) {
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.router);
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.shared_expert_gate);
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.shared_expert.gate);
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.shared_expert.up);
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.shared_expert.down);
    let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
    let _ = gpu.free_tensor(ffn.expert_down_ptrs);
    for e in ffn.experts {
        free_weight_tensor_maybe_slab(gpu, slabs, e.gate_up);
        free_weight_tensor_maybe_slab(gpu, slabs, e.down);
    }
}
