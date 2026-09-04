// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 weight structs (dense / MoE layers), EP shard provenance and seals,
//! `Qwen35Weights`, and the persistent DeltaNet state (`DeltaNetState`).

use super::config::LayerType;
use super::config::Qwen35Config;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::screen_weight_tensor;
use hipfire_runtime::MmqScreenable;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

// ─── Weight structs ─────────────────────────────────────────────────────

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
    /// Additive output biases, present only on escha dense exports
    /// (Qwen3.8-27B). Base Qwen3.8-27B has `attention_bias: false` and no MLP
    /// bias — these are Escha's end-to-end output correction, and they cannot
    /// be folded into a weight because they are additive. `None` for every
    /// other model, which is why they are Option rather than a zero vector:
    /// a zero add per projection per layer is real work for no effect.
    pub biases: Option<DeltaNetBiases>,
    /// Escha trellis metadata, one per coded projection. `Some` only when the
    /// weights are `Escha2T16`/`Escha3T16`, which is the signal this layer
    /// must bypass the fused MQ paths entirely — each projection needs its
    /// own rin-rotated activation, so FusedQkvza/gate_up have nothing to
    /// share.
    pub escha: Option<DeltaNetEscha>,
}

/// See `DeltaNetLayerWeights::escha`.
pub struct DeltaNetEscha {
    pub qkv: crate::qwen35::escha::EschaProj,
    pub z: crate::qwen35::escha::EschaProj,
    pub o: crate::qwen35::escha::EschaProj,
    pub gate: crate::qwen35::escha::EschaProj,
    pub up: crate::qwen35::escha::EschaProj,
    pub down: crate::qwen35::escha::EschaProj,
    /// `slots`-long run of zeros for the indexed GEMV; sized for the largest
    /// batch the model was built for so decode (slots=1) reads a prefix.
    pub ids: GpuTensor,
    /// `0..MAX` — the grouped GEMM's slot permutation, which for a dense
    /// linear is the identity.
    pub iota: GpuTensor,
}

/// See `DeltaNetLayerWeights::biases`. Each is `[oc]` f32, applied to the
/// projection's output. `in_proj_a`/`in_proj_b` have none — they are on
/// escha's `ignore` list and ship as plain weights.
pub struct DeltaNetBiases {
    pub qkv: GpuTensor,
    pub z: GpuTensor,
    pub o: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub down: GpuTensor,
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
    /// See `DeltaNetLayerWeights::biases`.
    pub biases: Option<FullAttnBiases>,
    /// See `DeltaNetLayerWeights::escha`.
    pub escha: Option<FullAttnEscha>,
}

/// See `DeltaNetLayerWeights::escha`.
pub struct FullAttnEscha {
    pub q: crate::qwen35::escha::EschaProj,
    pub k: crate::qwen35::escha::EschaProj,
    pub v: crate::qwen35::escha::EschaProj,
    pub o: crate::qwen35::escha::EschaProj,
    pub gate: crate::qwen35::escha::EschaProj,
    pub up: crate::qwen35::escha::EschaProj,
    pub down: crate::qwen35::escha::EschaProj,
    pub ids: GpuTensor,
    /// See `DeltaNetEscha::iota`.
    pub iota: GpuTensor,
}

/// See `DeltaNetLayerWeights::biases`.
pub struct FullAttnBiases {
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub o: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub down: GpuTensor,
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

/// Owning storage for a layer's packed routed experts — one device buffer per
/// (layer, projection) covering ALL experts.
///
/// `experts` still carries one [`WeightTensor`] view per routed expert so the
/// CPU fallback and every existing indexed dispatch keep their exact metadata
/// and pointer-table ABI. Those views are non-owning subranges of these two
/// buffers; only this owner pair may be returned to the GPU pool.
///
/// Two producers build this: `try_load_packed_mq4_experts` (uniform MQ4) and
/// `escha::load_escha_moe_experts` (Escha-W2). It is `pub` because the latter
/// hands the owners back across the crate boundary to
/// `examples/escha_moe_block_gate`, which loads a layer's experts directly and
/// must free them exactly once.
///
/// ## Why this is not merely tidier
///
/// The HIP allocator rounds every allocation up to a 2 MiB granule. At A3B
/// shapes a Q8_0 gate_up is 2.125 MiB and a Q8_0 down is 1.0625 MiB, so 20,480
/// independent per-expert buffers (40 layers x 256 experts x 2 projections)
/// occupy 4 MiB and 2 MiB each — 64.4 GB of granules for 34.2 GB of weights.
/// Packing each (layer, projection) into ONE buffer pays the rounding once per
/// buffer instead of once per expert and recovers ~30 GB. Measured: 67.9 GB ->
/// 37.6 GB of GTT for the whole escha-35b model on gfx1151 (37 587 996 672 B
/// delta, `scripts/escha-gtt-probe.sh`).
pub struct PackedExpertOwners {
    pub gate_up: GpuTensor,
    pub down: GpuTensor,
}

/// SP2: build the per-expert (gate_up, down) quant-tier tables that
/// [`hipfire_dispatch::families::moe::MoeDtypes`] uses to detect an
/// intra-layer mixed-tier layer.
///
/// A table is `Some(vec)` only when the layer genuinely spans >1 distinct
/// tier; a uniform layer — or paged mode where `experts` is empty — yields
/// `None`, which `MoeResolution::resolve` collapses to the unchanged uniform
/// fast path. We pre-filter to `None` for the uniform/empty cases here so the
/// common path allocates nothing and is byte-identical to before SP2.
pub(crate) fn per_expert_tier_tables(
    ffn: &MoeFfnWeights,
) -> (Option<Vec<DType>>, Option<Vec<DType>>) {
    if let Some(global) = ffn.global_expert_dtypes.as_ref() {
        let gu: Vec<DType> = global.iter().map(|(g, _)| *g).collect();
        let dn: Vec<DType> = global.iter().map(|(_, d)| *d).collect();
        (mixed_tier_table(gu), mixed_tier_table(dn))
    } else {
        let gu: Vec<DType> = ffn.experts.iter().map(|e| e.gate_up.gpu_dtype).collect();
        let dn: Vec<DType> = ffn.experts.iter().map(|e| e.down.gpu_dtype).collect();
        (mixed_tier_table(gu), mixed_tier_table(dn))
    }
}

/// Collapse a per-expert dtype column to `None` when it is empty or uniform,
/// `Some` only when it spans >1 distinct tier. Pure (no GPU weights) so it is
/// unit-testable in isolation; `per_expert_tier_tables` is the GPU-weight
/// adapter over it.
fn mixed_tier_table(tiers: Vec<DType>) -> Option<Vec<DType>> {
    match tiers.first() {
        // Empty (paged mode) or uniform → uniform fast path.
        None => None,
        Some(&first) if tiers.iter().all(|&d| d == first) => None,
        Some(_) => Some(tiers),
    }
}
/// Fallible per-expert tag mapping for the pinned graded MQ4R family.
///
/// Tags 0..6 retain exact V1 pair meaning. Tags 7..18 are the frozen MQV2
/// mixed-layout identities from the kernel contract — V1/V2 never collapse.
/// Every other ordered pair, every GL dtype in either position, and every
/// unknown dtype is `Err`. Single source of truth consumed by both
/// projections via one stored tag.
pub fn mixed_expert_tag(gate_dtype: DType, down_dtype: DType) -> HipResult<u8> {
    // GL in either position is always rejected – the tag-branched decoder has
    // no GL branch and would silently mis-decode as MQ4.
    if matches!(gate_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
        || matches!(down_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
    {
        return Err(HipError::new(
            0,
            &format!("graded EP: GL dtype not supported (gate={gate_dtype:?} down={down_dtype:?})"),
        ));
    }
    match (gate_dtype, down_dtype) {
        // Tags 0..6 — V1 pair identities (unchanged).
        (DType::MQ4G256, DType::MQ6G256) => Ok(0),
        (DType::MQ4G256, DType::MQ2G256Lloyd) => Ok(1),
        (DType::MQ4G256, DType::MQ4G256) => Ok(2),
        (DType::MQ4G256, DType::MQ3G256Lloyd) => Ok(3),
        (DType::MQ4G256, DType::MFP4G32E8) => Ok(4),
        (DType::MQ4G256, DType::MFP3G32E8) => Ok(5),
        (DType::MQ4G256, DType::MFP2G32E8) => Ok(6),
        // Matching non-MQ4 V1 pairs reuse the same tag numbers.
        (DType::MQ6G256, DType::MQ6G256) => Ok(0),
        (DType::MQ2G256Lloyd, DType::MQ2G256Lloyd) => Ok(1),
        (DType::MQ3G256Lloyd, DType::MQ3G256Lloyd) => Ok(3),
        (DType::MFP4G32E8, DType::MFP4G32E8) => Ok(4),
        (DType::MFP3G32E8, DType::MFP3G32E8) => Ok(5),
        (DType::MFP2G32E8, DType::MFP2G32E8) => Ok(6),
        // Tags 7..18 — frozen MQV2 mixed identities (never collapse to 0..6).
        (DType::MQ4G256V2, DType::MQ4G256V2) => Ok(7),
        (DType::MQ6G256V2, DType::MQ6G256V2) => Ok(8),
        (DType::MQ4G256V2, DType::MQ6G256) => Ok(9),
        (DType::MQ4G256V2, DType::MQ2G256Lloyd) => Ok(10),
        (DType::MQ4G256V2, DType::MQ4G256) => Ok(11),
        (DType::MQ4G256, DType::MQ4G256V2) => Ok(12),
        (DType::MQ4G256V2, DType::MQ3G256Lloyd) => Ok(13),
        (DType::MQ4G256V2, DType::MFP4G32E8) => Ok(14),
        (DType::MQ4G256V2, DType::MFP3G32E8) => Ok(15),
        (DType::MQ4G256V2, DType::MFP2G32E8) => Ok(16),
        (DType::MQ4G256V2, DType::MQ6G256V2) => Ok(17),
        (DType::MQ4G256, DType::MQ6G256V2) => Ok(18),
        _ => Err(HipError::new(
            0,
            &format!("graded EP: unsupported dtype pair gate={gate_dtype:?} down={down_dtype:?}"),
        )),
    }
}

pub(crate) fn dtype_from_quant_type(qt: u8) -> HipResult<DType> {
    match qt {
        13 => Ok(DType::MQ4G256),
        15 => Ok(DType::MQ6G256),
        19 => Ok(DType::MQ2G256Lloyd),
        20 => Ok(DType::MQ3G256Lloyd),
        30 => Ok(DType::MQ4G256Lloyd),
        34 => Ok(DType::MFP4G32E8),
        36 => Ok(DType::MFP3G32E8),
        38 => Ok(DType::MQ2G256GL),
        39 => Ok(DType::MQ3G256GL),
        40 => Ok(DType::TQ2G128),
        41 => Ok(DType::BQ1G128),
        44 => Ok(DType::MQ4G256V2),
        45 => Ok(DType::MQ4CG256),
        // Neutral-size Magnum V2 family (qt47-50): preserve qtype distinction
        // through WeightTensor/GpuTensor; do not map to legacy MQ2/3/5/6.
        47 => Ok(DType::MQ6G256V2),
        48 => Ok(DType::MQ5G256V2),
        49 => Ok(DType::MQ3G256V2),
        50 => Ok(DType::MQ2G256V2),
        // qt=6 (HFQ4G256) and qt=37 (MFP2G32E8) are shipped formats and MUST stay
        // mapped here. Dropping an arm from this match is not a compile error — it
        // degrades to "graded EP: unsupported quant_type", so the loss stays
        // invisible until a model of that format fails to load.
        6 => Ok(DType::HFQ4G256),
        37 => Ok(DType::MFP2G32E8),
        3 => Ok(DType::Q8_0),
        1 => Ok(DType::F16),
        2 => Ok(DType::F32),
        other => Err(HipError::new(
            0,
            &format!("graded EP: unsupported quant_type {other}"),
        )),
    }
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
    /// Two allocation owners for the uniform MQ4 packed path. `None` preserves
    /// the literal per-expert ownership used by mixed quant, Paro, paged, and
    /// EP-streaming routes.
    pub(crate) packed_expert_owners: Option<PackedExpertOwners>,
    pub shared_expert: SharedExpertWeights,
    pub shared_expert_gate: WeightTensor, // [1, hidden] — row-vector projecting to scalar
    /// Device-side array of `unsigned long long` pointers, one per
    /// expert's `gate_up.buf`. Indexed at runtime by the GPU top-K
    /// kernel's output so the indexed MoE GEMV can stay capture-safe.
    pub expert_gate_up_ptrs: GpuTensor, // [num_experts * 2] f32 slots = num_experts × u64
    pub expert_down_ptrs: GpuTensor,      // [num_experts * 2] f32 slots = num_experts × u64

    /// Route A MoE-AWQ: per-expert down `awq_scale` pointer table
    /// (`[num_experts * 2]` f32 = num_experts × u64). `Some` only when the
    /// `.hfq` carries per-expert `down_proj.awq_scale` sidecars (all-or-none).
    /// Holds *non-owning* device pointers into each `experts[i].down.awq_scale`
    /// — freed as a buffer only; the scales are freed via
    /// `ExpertWeights::down.free_all`.
    pub expert_down_awq_ptrs: Option<GpuTensor>,

    /// Per-expert mixed-precision decode: `[num_experts]` u8 (DType::Raw,
    /// 1 B/expert) dtype-tag table. `Some` only when the layer's routed
    /// experts carry MIXED down dtypes (graded MQ6 hot / MQ2-Lloyd cold);
    /// the merged dtype-tag-branched down kernel reads `tags[expert_id]`
    /// per block (0=MQ6, 1=MQ2-Lloyd). `None` ⇒ uniform path, byte-identical.
    /// Owned device buffer (no aliasing) — freed as a buffer in free_moe_ffn.
    pub expert_dtype_tags: Option<GpuTensor>,

    /// Layer index. Stable identity used to key
    /// [`hipfire_runtime::weight_pager::WeightId::Expert`] entries.
    pub layer_idx: u16,

    /// Per-expert tensor shapes. `None` in non-paged mode (shapes are read
    /// from `experts[i].gate_up.{m, k}` etc.); `Some` in paged mode where
    /// `experts` is empty but kernels still need m/k for kernel-arg setup.
    /// Qwen3.5-MoE-A3B has uniform per-expert shape so one descriptor per
    /// layer suffices for v0.1.
    pub expert_shape: Option<hipfire_runtime::weight_pager::ExpertShape>,

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

    /// EP global (gate_up_dtype, down_dtype) table — CPU-side immutable
    /// snapshot of the *full-model* expert dtypes (`len == num_experts`).
    /// `Some` only on the EP `load_weights_ep_rank` path; `None` preserves
    /// byte-identical single-GPU behavior. When present, every graded-mix
    /// decision (uniform/mixed flags, representative dtypes, tier tables,
    /// dummy layout sizes, device tag upload) is derived from this global
    /// table, never from the compact local `experts` slice.
    pub(crate) global_expert_dtypes: Option<Box<[(DType, DType)]>>,

    /// EP streaming dummies: one owned zero buffer per distinct
    /// non-owned storage layout. Non-owned global slots alias into the
    /// matching entry. Owned so `free_moe_ffn` can reclaim them.
    pub(crate) ep_dummy_buffers: Vec<GpuTensor>,

    /// Escha-W2 (Task 10): per-layer H128 transform tables + decode scratch.
    /// `Some` only for layers loaded from an Escha-W2 checkpoint.
    ///
    /// This is also the layer's escha MARKER. The loader decodes the trellis
    /// and stores the experts as `Q8_0`, so `experts[i].gate_up.gpu_dtype` no
    /// longer says "escha" by the time dispatch resolves the layer; only this
    /// field does. `moe_ffn_decode_impl` threads it into
    /// `MoeParams::escha`, whose `has_escha()` drives both the f16
    /// router-logit round-trip and the H128-wrapped routed executor.
    pub escha: Option<super::escha::EschaMoeTables>,
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
/// Immutable source identity captured before any EP GPU allocation.
/// Exact equality over canonical path, platform file identity (dev, ino),
/// length, mtime, arch_id, exact metadata_json, ordered tensor manifest
/// (name, quant_type, shape, group_size, data_offset, data_size) with
/// absolute offsets (base offset included), and overlay status.
/// Not a hash – any reordering or header difference is inequality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35HfqSourceIdentity {
    pub canonical_path: std::path::PathBuf,
    pub dev: u64,
    pub ino: u64,
    pub file_len: u64,
    pub mtime_secs: i64,
    pub mtime_nanos: u32,
    pub arch_id: u32,
    pub metadata_json: String,
    pub tensor_manifest: Vec<(String, u8, Vec<u32>, u32, usize, usize)>,
    pub has_overlay: bool,
}

impl Qwen35HfqSourceIdentity {
    pub fn capture(hfq: &HfqFile) -> Self {
        let path = hfq.path().to_path_buf();
        let canonical = std::fs::canonicalize(&path).unwrap_or(path.clone());
        let (dev, ino, file_len, mtime_secs, mtime_nanos) = {
            match std::fs::metadata(&path) {
                Ok(md) => {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        let mtime = md
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
                        (
                            md.dev(),
                            md.ino(),
                            md.len(),
                            mtime.map(|d| d.as_secs() as i64).unwrap_or(0),
                            mtime.map(|d| d.subsec_nanos()).unwrap_or(0),
                        )
                    }
                    #[cfg(not(unix))]
                    {
                        let mtime = md
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
                        (
                            0u64,
                            0u64,
                            md.len(),
                            mtime.map(|d| d.as_secs() as i64).unwrap_or(0),
                            mtime.map(|d| d.subsec_nanos()).unwrap_or(0),
                        )
                    }
                }
                Err(_) => (0, 0, 0, 0, 0),
            }
        };
        let tensors = hfq.tensors();
        let manifest = tensors
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    t.quant_type,
                    t.shape.clone(),
                    t.group_size,
                    t.data_offset,
                    t.data_size,
                )
            })
            .collect();
        Self {
            canonical_path: canonical,
            dev,
            ino,
            file_len,
            mtime_secs,
            mtime_nanos,
            arch_id: hfq.arch_id,
            metadata_json: hfq.metadata_json.clone(),
            tensor_manifest: manifest,
            has_overlay: hfq.has_overlay(),
        }
    }
}

/// Frozen config fingerprint for EP seal. Contains every Qwen35Config primitive.
/// Equality is exact (f32 via to_bits). EP admission still rejects paged/REAP but
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35EpConfigFingerprint {
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub norm_eps_bits: u32,
    pub eos_token: u32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta_bits: u32,
    pub partial_rotary_factor_bits: u32,
    pub is_vl_text: bool,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub conv_kernel_dim: usize,
    pub hidden_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub has_shared_expert: bool,
    pub norm_topk_prob: bool,
    pub layer_types: Vec<LayerType>,
    pub paged_experts: bool,
    pub vram_budget_bytes: u64,
    pub has_reap_keep: bool,
}

impl Qwen35EpConfigFingerprint {
    pub fn capture(config: &Qwen35Config) -> Self {
        Self {
            dim: config.dim,
            n_layers: config.n_layers,
            vocab_size: config.vocab_size,
            norm_eps_bits: config.norm_eps.to_bits(),
            eos_token: config.eos_token,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            rope_theta_bits: config.rope_theta.to_bits(),
            partial_rotary_factor_bits: config.partial_rotary_factor.to_bits(),
            is_vl_text: config.is_vl_text,
            mrope_interleaved: config.mrope_interleaved,
            mrope_section: config.mrope_section,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            conv_kernel_dim: config.conv_kernel_dim,
            hidden_dim: config.hidden_dim,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            moe_intermediate_size: config.moe_intermediate_size,
            shared_expert_intermediate_size: config.shared_expert_intermediate_size,
            has_shared_expert: config.has_shared_expert,
            norm_topk_prob: config.norm_topk_prob,
            layer_types: config.layer_types.clone(),
            paged_experts: config.paged_experts,
            vram_budget_bytes: config.vram_budget_bytes,
            has_reap_keep: config.reap_keep.is_some(),
        }
    }
}

/// Device-pointer-free descriptor for a GpuTensor. Excludes DeviceBuffer pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTensorDescriptor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub byte_len: usize,
}

impl GpuTensorDescriptor {
    pub fn from_tensor(t: &GpuTensor) -> Self {
        Self {
            shape: t.shape.clone(),
            dtype: t.dtype,
            byte_len: t.buf.size(),
        }
    }
}

/// Paro sidecar descriptor excluding pointers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParoDescriptor {
    pub krot: u32,
    pub group_size: u32,
    pub is_alias: bool,
    pub pairs: GpuTensorDescriptor,
    pub theta: GpuTensorDescriptor,
    pub channel_scales: GpuTensorDescriptor,
}

/// Weight tensor descriptor excluding device pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightTensorDescriptor {
    pub gpu_dtype: DType,
    pub m: usize,
    pub k: usize,
    pub row_stride: usize,
    pub buf: GpuTensorDescriptor,
    pub awq_scale: Option<GpuTensorDescriptor>,
    pub paro: Option<ParoDescriptor>,
}

impl WeightTensorDescriptor {
    pub fn from_weight(w: &WeightTensor) -> Self {
        Self {
            gpu_dtype: w.gpu_dtype,
            m: w.m,
            k: w.k,
            row_stride: w.row_stride,
            buf: GpuTensorDescriptor::from_tensor(&w.buf),
            awq_scale: w.awq_scale.as_ref().map(GpuTensorDescriptor::from_tensor),
            paro: w.paro.as_ref().map(|p| ParoDescriptor {
                krot: p.krot,
                group_size: p.group_size,
                is_alias: p.is_alias,
                pairs: GpuTensorDescriptor::from_tensor(&p.pairs),
                theta: GpuTensorDescriptor::from_tensor(&p.theta),
                channel_scales: GpuTensorDescriptor::from_tensor(&p.channel_scales),
            }),
        }
    }
}

/// Per-expert local descriptor: global id + gate_up/down descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35LocalExpertDescriptor {
    pub global_expert_id: usize,
    pub gate_up: WeightTensorDescriptor,
    pub down: WeightTensorDescriptor,
}
/// Complete rank weight/layout seal. Immutable, allocation-free comparison via `matches_*`.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35RankSeal {
    pub token_embd: GpuTensorDescriptor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensorDescriptor,
    pub output: WeightTensorDescriptor,
    pub moe_has_mq6: bool,
    pub has_pager: bool,
    pub lm_head_aliases_embd: bool,
    pub layer_seals: Vec<Qwen35LayerSeal>,
    pub global_expert_dtypes: Vec<Vec<(DType, DType)>>,
    pub local_expert_descriptors: Vec<Vec<Qwen35LocalExpertDescriptor>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Qwen35LayerSeal {
    DeltaNet {
        attn_norm: GpuTensorDescriptor,
        wqkv: WeightTensorDescriptor,
        wz: WeightTensorDescriptor,
        w_alpha: WeightTensorDescriptor,
        w_beta: WeightTensorDescriptor,
        a_log: GpuTensorDescriptor,
        dt_bias: GpuTensorDescriptor,
        conv_weight: GpuTensorDescriptor,
        norm_weight: GpuTensorDescriptor,
        wo: WeightTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        w_gate: WeightTensorDescriptor,
        w_up: WeightTensorDescriptor,
        w_down: WeightTensorDescriptor,
    },
    FullAttn {
        attn_norm: GpuTensorDescriptor,
        wq: WeightTensorDescriptor,
        wk: WeightTensorDescriptor,
        wv: WeightTensorDescriptor,
        wo: WeightTensorDescriptor,
        q_norm: GpuTensorDescriptor,
        k_norm: GpuTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        w_gate: WeightTensorDescriptor,
        w_up: WeightTensorDescriptor,
        w_down: WeightTensorDescriptor,
    },
    DeltaNetMoe {
        attn_norm: GpuTensorDescriptor,
        wqkv: WeightTensorDescriptor,
        wz: WeightTensorDescriptor,
        w_alpha: WeightTensorDescriptor,
        w_beta: WeightTensorDescriptor,
        a_log: GpuTensorDescriptor,
        dt_bias: GpuTensorDescriptor,
        conv_weight: GpuTensorDescriptor,
        norm_weight: GpuTensorDescriptor,
        wo: WeightTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        moe: Qwen35MoeFfnSeal,
    },
    FullAttnMoe {
        attn_norm: GpuTensorDescriptor,
        wq: WeightTensorDescriptor,
        wk: WeightTensorDescriptor,
        wv: WeightTensorDescriptor,
        wo: WeightTensorDescriptor,
        q_norm: GpuTensorDescriptor,
        k_norm: GpuTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        moe: Qwen35MoeFfnSeal,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeFfnSeal {
    pub router: WeightTensorDescriptor,
    pub shared_gate: WeightTensorDescriptor,
    pub shared_up: WeightTensorDescriptor,
    pub shared_down: WeightTensorDescriptor,
    pub shared_expert_gate: WeightTensorDescriptor,
    pub expert_gate_up_ptrs: GpuTensorDescriptor,
    pub expert_down_ptrs: GpuTensorDescriptor,
    pub expert_down_awq_ptrs: Option<GpuTensorDescriptor>,
    pub expert_dtype_tags: Option<GpuTensorDescriptor>,
    pub layer_idx: u16,
    pub has_packed_owners: bool,
    pub global_expert_dtypes: Option<Vec<(DType, DType)>>,
    pub num_local_experts: usize,
}

impl Qwen35RankSeal {
    pub fn capture(weights: &Qwen35Weights, expert_to_rank: Option<&[u8]>, rank: usize) -> Self {
        let mut global_expert_dtypes = Vec::with_capacity(weights.layers.len());
        let mut local_expert_descriptors = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let ffn = match layer {
                LayerWeights::DeltaNetMoe(weights) => Some(&weights.ffn),
                LayerWeights::FullAttnMoe(weights) => Some(&weights.ffn),
                _ => None,
            };
            let Some(ffn) = ffn else {
                global_expert_dtypes.push(Vec::new());
                local_expert_descriptors.push(Vec::new());
                continue;
            };
            global_expert_dtypes.push(
                ffn.global_expert_dtypes
                    .as_ref()
                    .map(|dtypes| dtypes.to_vec())
                    .unwrap_or_default(),
            );
            let owned: Vec<usize> = match expert_to_rank {
                Some(map) => map
                    .iter()
                    .enumerate()
                    .filter_map(|(global_id, &owner)| (owner as usize == rank).then_some(global_id))
                    .collect(),
                None => (0..ffn.experts.len()).collect(),
            };
            let mut locals = Vec::with_capacity(ffn.experts.len());
            for (local_pos, expert) in ffn.experts.iter().enumerate() {
                locals.push(Qwen35LocalExpertDescriptor {
                    global_expert_id: owned.get(local_pos).copied().unwrap_or(local_pos),
                    gate_up: WeightTensorDescriptor::from_weight(&expert.gate_up),
                    down: WeightTensorDescriptor::from_weight(&expert.down),
                });
            }
            locals.sort_by_key(|descriptor| descriptor.global_expert_id);
            local_expert_descriptors.push(locals);
        }
        let layer_seals = weights
            .layers
            .iter()
            .map(|l| match l {
                LayerWeights::DeltaNet(w) => Qwen35LayerSeal::DeltaNet {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wqkv: WeightTensorDescriptor::from_weight(&w.wqkv),
                    wz: WeightTensorDescriptor::from_weight(&w.wz),
                    w_alpha: WeightTensorDescriptor::from_weight(&w.w_alpha),
                    w_beta: WeightTensorDescriptor::from_weight(&w.w_beta),
                    a_log: GpuTensorDescriptor::from_tensor(&w.a_log),
                    dt_bias: GpuTensorDescriptor::from_tensor(&w.dt_bias),
                    conv_weight: GpuTensorDescriptor::from_tensor(&w.conv_weight),
                    norm_weight: GpuTensorDescriptor::from_tensor(&w.norm_weight),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    w_gate: WeightTensorDescriptor::from_weight(&w.w_gate),
                    w_up: WeightTensorDescriptor::from_weight(&w.w_up),
                    w_down: WeightTensorDescriptor::from_weight(&w.w_down),
                },
                LayerWeights::FullAttn(w) => Qwen35LayerSeal::FullAttn {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wq: WeightTensorDescriptor::from_weight(&w.wq),
                    wk: WeightTensorDescriptor::from_weight(&w.wk),
                    wv: WeightTensorDescriptor::from_weight(&w.wv),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    q_norm: GpuTensorDescriptor::from_tensor(&w.q_norm),
                    k_norm: GpuTensorDescriptor::from_tensor(&w.k_norm),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    w_gate: WeightTensorDescriptor::from_weight(&w.w_gate),
                    w_up: WeightTensorDescriptor::from_weight(&w.w_up),
                    w_down: WeightTensorDescriptor::from_weight(&w.w_down),
                },
                LayerWeights::DeltaNetMoe(w) => Qwen35LayerSeal::DeltaNetMoe {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wqkv: WeightTensorDescriptor::from_weight(&w.wqkv),
                    wz: WeightTensorDescriptor::from_weight(&w.wz),
                    w_alpha: WeightTensorDescriptor::from_weight(&w.w_alpha),
                    w_beta: WeightTensorDescriptor::from_weight(&w.w_beta),
                    a_log: GpuTensorDescriptor::from_tensor(&w.a_log),
                    dt_bias: GpuTensorDescriptor::from_tensor(&w.dt_bias),
                    conv_weight: GpuTensorDescriptor::from_tensor(&w.conv_weight),
                    norm_weight: GpuTensorDescriptor::from_tensor(&w.norm_weight),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    moe: Qwen35MoeFfnSeal {
                        router: WeightTensorDescriptor::from_weight(&w.ffn.router),
                        shared_gate: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.gate),
                        shared_up: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.up),
                        shared_down: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.down),
                        shared_expert_gate: WeightTensorDescriptor::from_weight(
                            &w.ffn.shared_expert_gate,
                        ),
                        expert_gate_up_ptrs: GpuTensorDescriptor::from_tensor(
                            &w.ffn.expert_gate_up_ptrs,
                        ),
                        expert_down_ptrs: GpuTensorDescriptor::from_tensor(&w.ffn.expert_down_ptrs),
                        expert_down_awq_ptrs: w
                            .ffn
                            .expert_down_awq_ptrs
                            .as_ref()
                            .map(GpuTensorDescriptor::from_tensor),
                        expert_dtype_tags: w
                            .ffn
                            .expert_dtype_tags
                            .as_ref()
                            .map(GpuTensorDescriptor::from_tensor),
                        layer_idx: w.ffn.layer_idx,
                        has_packed_owners: w.ffn.packed_expert_owners.is_some(),
                        global_expert_dtypes: w
                            .ffn
                            .global_expert_dtypes
                            .as_ref()
                            .map(|b| b.to_vec()),
                        num_local_experts: w.ffn.experts.len(),
                    },
                },
                LayerWeights::FullAttnMoe(w) => Qwen35LayerSeal::FullAttnMoe {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wq: WeightTensorDescriptor::from_weight(&w.wq),
                    wk: WeightTensorDescriptor::from_weight(&w.wk),
                    wv: WeightTensorDescriptor::from_weight(&w.wv),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    q_norm: GpuTensorDescriptor::from_tensor(&w.q_norm),
                    k_norm: GpuTensorDescriptor::from_tensor(&w.k_norm),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    moe: Qwen35MoeFfnSeal {
                        router: WeightTensorDescriptor::from_weight(&w.ffn.router),
                        shared_gate: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.gate),
                        shared_up: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.up),
                        shared_down: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.down),
                        shared_expert_gate: WeightTensorDescriptor::from_weight(
                            &w.ffn.shared_expert_gate,
                        ),
                        expert_gate_up_ptrs: GpuTensorDescriptor::from_tensor(
                            &w.ffn.expert_gate_up_ptrs,
                        ),
                        expert_down_ptrs: GpuTensorDescriptor::from_tensor(&w.ffn.expert_down_ptrs),
                        expert_down_awq_ptrs: w
                            .ffn
                            .expert_down_awq_ptrs
                            .as_ref()
                            .map(GpuTensorDescriptor::from_tensor),
                        expert_dtype_tags: w
                            .ffn
                            .expert_dtype_tags
                            .as_ref()
                            .map(GpuTensorDescriptor::from_tensor),
                        layer_idx: w.ffn.layer_idx,
                        has_packed_owners: w.ffn.packed_expert_owners.is_some(),
                        global_expert_dtypes: w
                            .ffn
                            .global_expert_dtypes
                            .as_ref()
                            .map(|b| b.to_vec()),
                        num_local_experts: w.ffn.experts.len(),
                    },
                },
            })
            .collect();
        Self {
            token_embd: GpuTensorDescriptor::from_tensor(&weights.token_embd),
            embd_format: weights.embd_format,
            output_norm: GpuTensorDescriptor::from_tensor(&weights.output_norm),
            output: WeightTensorDescriptor::from_weight(&weights.output),
            moe_has_mq6: weights.moe_has_mq6,
            has_pager: weights.pager.is_some(),
            lm_head_aliases_embd: weights.lm_head_aliases_embd,
            layer_seals,
            global_expert_dtypes,
            local_expert_descriptors,
        }
    }
    pub fn matches_config(&self, other: &Self) -> bool {
        self == other
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35EpShardInfo {
    pub(crate) rank: u8,
    pub(crate) rank_count: u8,
    pub(crate) expert_to_rank: Box<[u8]>,
    pub device_id: i32,
    pub source_identity: std::sync::Arc<Qwen35HfqSourceIdentity>,
    pub config_fingerprint: Qwen35EpConfigFingerprint,
    pub rank_seal: Qwen35RankSeal,
}

impl Qwen35EpShardInfo {
    /// Owning rank for this shard (0 <= rank < rank_count).
    pub fn rank(&self) -> u8 {
        self.rank
    }
    /// Total number of ranks in the EP group (exactly 4 for the MQ4R route).
    pub fn rank_count(&self) -> u8 {
        self.rank_count
    }
    /// Global expert → owning rank map (`len == config.num_experts`, each entry < rank_count).
    pub fn expert_to_rank(&self) -> &[u8] {
        &self.expert_to_rank
    }
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    pub fn source_identity(&self) -> &Qwen35HfqSourceIdentity {
        &self.source_identity
    }
    pub fn config_fingerprint(&self) -> &Qwen35EpConfigFingerprint {
        &self.config_fingerprint
    }
    pub fn rank_seal(&self) -> &Qwen35RankSeal {
        &self.rank_seal
    }
}
/// Transactional pending owner for EP `load_moe_ffn`. Allocations publish only on commit;
/// any failure rolls back every populated field on the owner device with sync/first-error preservation.
pub(crate) struct PendingEpMoeFfn {
    pub(crate) router: Option<WeightTensor>,
    pub(crate) shared_gate: Option<WeightTensor>,
    pub(crate) shared_up: Option<WeightTensor>,
    pub(crate) shared_down: Option<WeightTensor>,
    pub(crate) shared_gate_scalar: Option<WeightTensor>,
    pub(crate) experts: Vec<ExpertWeights>,
    pub(crate) packed_owners: Option<PackedExpertOwners>,
    pub(crate) dummy_buffers: Vec<GpuTensor>,
    pub(crate) gate_up_ptrs: Option<GpuTensor>,
    pub(crate) down_ptrs: Option<GpuTensor>,
    pub(crate) awq_ptrs: Option<GpuTensor>,
    pub(crate) dtype_tags: Option<GpuTensor>,
    pub(crate) global_dtypes: Option<Box<[(DType, DType)]>>,
    pub(crate) layer_idx: u16,
}

impl PendingEpMoeFfn {
    pub(crate) fn new(layer_idx: u16) -> Self {
        Self {
            router: None,
            shared_gate: None,
            shared_up: None,
            shared_down: None,
            shared_gate_scalar: None,
            experts: Vec::new(),
            packed_owners: None,
            dummy_buffers: Vec::new(),
            gate_up_ptrs: None,
            down_ptrs: None,
            awq_ptrs: None,
            dtype_tags: None,
            global_dtypes: None,
            layer_idx,
        }
    }
    /// Roll back every populated field on the owner device. Binds + synchronizes the
    /// owner, attempts every free, preserves the initiating error as primary and attaches
    /// the first cleanup failure as context. Returns the enriched error.
    pub(crate) fn rollback(mut self, gpu: &mut Gpu, err: HipError) -> HipError {
        let mut first_cleanup: Option<HipError> = None;
        let mut record_cleanup = |e: HipError| {
            if first_cleanup.is_none() {
                first_cleanup = Some(e);
            }
        };
        let _ = gpu.bind_thread();
        let _ = gpu.hip.device_synchronize();
        if let Some(t) = self.dtype_tags.take() {
            if let Err(e) = gpu.free_tensor(t) {
                record_cleanup(e);
            }
        }
        if let Some(t) = self.awq_ptrs.take() {
            if let Err(e) = gpu.free_tensor(t) {
                record_cleanup(e);
            }
        }
        if let Some(t) = self.down_ptrs.take() {
            if let Err(e) = gpu.free_tensor(t) {
                record_cleanup(e);
            }
        }
        if let Some(t) = self.gate_up_ptrs.take() {
            if let Err(e) = gpu.free_tensor(t) {
                record_cleanup(e);
            }
        }
        for d in self.dummy_buffers.drain(..) {
            if let Err(e) = gpu.free_tensor(d) {
                record_cleanup(e);
            }
        }
        if let Some(owners) = self.packed_owners.take() {
            for e in self.experts.drain(..) {
                free_weight_metadata_only(gpu, e.gate_up);
                free_weight_metadata_only(gpu, e.down);
            }
            if let Err(e) = gpu.free_tensor(owners.gate_up) {
                record_cleanup(e);
            }
            if let Err(e) = gpu.free_tensor(owners.down) {
                record_cleanup(e);
            }
        } else {
            for e in self.experts.drain(..) {
                if let Some(e1) = free_weight_checked(gpu, e.gate_up) {
                    record_cleanup(e1);
                }
                if let Some(e2) = free_weight_checked(gpu, e.down) {
                    record_cleanup(e2);
                }
            }
        }
        if let Some(w) = self.shared_gate_scalar.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(w) = self.shared_down.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(w) = self.shared_up.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(w) = self.shared_gate.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(w) = self.router.take() {
            if let Some(e) = free_weight_checked(gpu, w) {
                record_cleanup(e);
            }
        }
        if let Some(cleanup) = first_cleanup {
            HipError::new(
                0,
                &format!("{} (cleanup: {})", err.message, cleanup.message),
            )
        } else {
            err
        }
    }
    pub(crate) fn commit(
        self,
        shared_expert: SharedExpertWeights,
        gate_up_ptrs: GpuTensor,
        down_ptrs: GpuTensor,
        awq_ptrs: Option<GpuTensor>,
        dtype_tags: Option<GpuTensor>,
    ) -> MoeFfnWeights {
        let router = self.router.expect("pending commit: router missing");
        let shared_gate_scalar = self
            .shared_gate_scalar
            .expect("pending commit: shared_gate_scalar missing");
        MoeFfnWeights {
            router,
            experts: self.experts,
            packed_expert_owners: self.packed_owners,
            shared_expert,
            shared_expert_gate: shared_gate_scalar,
            expert_gate_up_ptrs: gate_up_ptrs,
            expert_down_ptrs: down_ptrs,
            expert_down_awq_ptrs: awq_ptrs,
            expert_dtype_tags: dtype_tags,
            layer_idx: self.layer_idx,
            expert_shape: None,
            paro_shared: None,
            global_expert_dtypes: self.global_dtypes,
            ep_dummy_buffers: self.dummy_buffers,
            // The EP / pending-commit path does not carry escha layers (an
            // Escha-W2 checkpoint has one code tensor per layer, not the
            // per-expert tensors EP sharding streams).
            escha: None,
        }
    }
}

/// Internal checked free for a WeightTensor: attempts every sidecar and buffer free,
/// returns the first HipError if any, otherwise None. Non-public; used only by Pending rollback.
fn free_weight_checked(gpu: &mut Gpu, w: WeightTensor) -> Option<HipError> {
    let mut first: Option<HipError> = None;
    let mut record = |e: HipError| {
        if first.is_none() {
            first = Some(e);
        }
    };
    if let Some(paro) = w.paro {
        if !paro.is_alias {
            if let Err(e) = gpu.free_tensor(paro.pairs) {
                record(e);
            }
            if let Err(e) = gpu.free_tensor(paro.theta) {
                record(e);
            }
            if let Err(e) = gpu.free_tensor(paro.channel_scales) {
                record(e);
            }
        }
    }
    if let Some(awq) = w.awq_scale {
        if let Err(e) = gpu.free_tensor(awq) {
            record(e);
        }
    }
    if let Err(e) = gpu.free_tensor(w.buf) {
        record(e);
    }
    first
}

pub struct Qwen35Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
    /// True when any MoE FFN projection in the loaded model is MQ6. gfx1151's
    /// grouped-i8 MQ4 shortcut is model-level unsafe for these promoted A3B
    /// checkpoints, even in layers whose local routed experts remain MQ4.
    pub moe_has_mq6: bool,

    /// Weight pager (MAD-93 v0.1). `Some` only when the model was loaded
    /// with `Qwen35Config::paged_experts == true`. The forward path uses
    /// interior mutability (`borrow_mut`) at the MoE dispatch site to call
    /// `ensure_resident` / `patch_expert_ptr_table`. `None` means the model
    /// is fully resident — no behavior change vs main.
    pub pager: Option<std::cell::RefCell<hipfire_runtime::weight_pager::WeightPager>>,

    /// True when the tied lm_head aliases the embedding table buffer
    /// (single-GPU path). When true, `output.buf` is a non-owning view of
    /// `token_embd.buf` and must NOT be freed in `free_gpu`.
    pub lm_head_aliases_embd: bool,

    /// Immutable EP shard provenance. `Some` only when loaded via
    /// `load_weights_ep_rank` on the exact 4×gfx1201 MQ4R route; all
    /// ordinary (single-GPU, TP, paged) loads leave `None`.
    pub(crate) ep_shard: Option<Qwen35EpShardInfo>,
}

impl Qwen35Weights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        if !self.lm_head_aliases_embd {
            self.output.free_all(gpu);
        }
        for layer in self.layers {
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
            }
        }
        // MAD-93 v0.1: in paged mode, the pager owns expert weight allocations
        // (the per-layer `free_moe_ffn` loops ran no-ops since `ffn.experts`
        // was empty). Drain the pager's resident set back to the GPU pool here.
        if let Some(pager_cell) = self.pager {
            pager_cell.into_inner().free_all(gpu);
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
        self.output.free_all(&mut gpus.devices[out_dev]);
        for (i, layer) in self.layers.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let gpu = &mut gpus.devices[dev_idx];
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
            }
        }
    }
}
impl Qwen35Weights {
    /// Immutable EP shard provenance, if loaded via `load_weights_ep_rank`.
    /// `None` on every ordinary single-GPU/TP/paged load.
    pub fn ep_shard(&self) -> Option<&Qwen35EpShardInfo> {
        self.ep_shard.as_ref()
    }
}

impl MmqScreenable for Qwen35Weights {
    fn screen_mmq_weights(&self, gpu: &mut Gpu) -> (usize, usize) {
        let (mut safe, mut unsafe_count) = (0usize, 0usize);
        screen_weight_tensor(&self.output, gpu, &mut safe, &mut unsafe_count);
        for layer in &self.layers {
            match layer {
                LayerWeights::DeltaNet(weights) => {
                    for weight in [
                        &weights.wqkv,
                        &weights.wz,
                        &weights.w_alpha,
                        &weights.w_beta,
                        &weights.wo,
                        &weights.w_gate,
                        &weights.w_up,
                        &weights.w_down,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                LayerWeights::FullAttn(weights) => {
                    for weight in [
                        &weights.wq,
                        &weights.wk,
                        &weights.wv,
                        &weights.wo,
                        &weights.w_gate,
                        &weights.w_up,
                        &weights.w_down,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                // Routed and shared experts live outside ordinary WeightTensor
                // storage in paged/EP modes. Screen the resident attention and
                // dense router weights here; expert screening is separate work.
                LayerWeights::DeltaNetMoe(weights) => {
                    for weight in [
                        &weights.wqkv,
                        &weights.wz,
                        &weights.w_alpha,
                        &weights.w_beta,
                        &weights.wo,
                        &weights.ffn.router,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                LayerWeights::FullAttnMoe(weights) => {
                    for weight in [
                        &weights.wq,
                        &weights.wk,
                        &weights.wv,
                        &weights.wo,
                        &weights.ffn.router,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
            }
        }
        (safe, unsafe_count)
    }
}

fn free_moe_ffn(gpu: &mut Gpu, ffn: MoeFfnWeights) {
    // Escha-W2 transform tables + decode scratch. Owned outright (nothing
    // aliases them), so free before the experts they describe.
    if let Some(e) = ffn.escha {
        e.free_gpu(gpu);
    }
    ffn.router.free_all(gpu);
    ffn.shared_expert_gate.free_all(gpu);
    ffn.shared_expert.gate.free_all(gpu);
    ffn.shared_expert.up.free_all(gpu);
    ffn.shared_expert.down.free_all(gpu);
    let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
    let _ = gpu.free_tensor(ffn.expert_down_ptrs);
    // Non-owning pointer table — free the buffer only; the per-expert scales it
    // points into are owned by `experts[i].down.awq_scale` and freed below via
    // `e.down.free_all`.
    if let Some(t) = ffn.expert_down_awq_ptrs {
        let _ = gpu.free_tensor(t);
    }
    // Owned device buffer (built from per-expert gpu_dtype). Free it.
    if let Some(t) = ffn.expert_dtype_tags {
        let _ = gpu.free_tensor(t);
    }
    if let Some(owners) = ffn.packed_expert_owners {
        // Packed expert WeightTensors are non-owning views. Free only metadata
        // that remains individually owned, then return each layer blob once.
        for e in ffn.experts {
            free_weight_metadata_only(gpu, e.gate_up);
            free_weight_metadata_only(gpu, e.down);
        }
        let _ = gpu.free_tensor(owners.gate_up);
        let _ = gpu.free_tensor(owners.down);
    } else {
        for e in ffn.experts {
            e.gate_up.free_all(gpu);
            e.down.free_all(gpu);
        }
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
    for d in ffn.ep_dummy_buffers {
        let _ = gpu.free_tensor(d);
    }
}

/// Free a [`WeightTensor`]'s owning sidecars without freeing its weight buffer.
/// Used only for non-owning views into [`PackedExpertOwners`].
fn free_weight_metadata_only(gpu: &mut Gpu, weight: WeightTensor) {
    if let Some(paro) = weight.paro {
        if !paro.is_alias {
            let _ = gpu.free_tensor(paro.pairs);
            let _ = gpu.free_tensor(paro.theta);
            let _ = gpu.free_tensor(paro.channel_scales);
        }
    }
    if let Some(awq) = weight.awq_scale {
        let _ = gpu.free_tensor(awq);
    }
}

// ─── State ──────────────────────────────────────────────────────────────

/// Persistent state for DeltaNet layers across tokens.
/// State quantization mode for DeltaNet S matrix.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StateQuant {
    FP32,
    Q8,
    Q4,
}

pub struct DeltaNetState {
    /// S matrix storage — FP32 or Q8 depending on quant mode
    pub s_matrices: Vec<GpuTensor>,
    /// Per-head scale factors (only used for Q8 mode)
    pub s_scales: Vec<GpuTensor>,
    /// Conv ring buffer: [n_deltanet_layers × conv_channels × (kernel_size-1)] FP32
    pub conv_states: Vec<GpuTensor>,
    /// Per-element f16 error-feedback residual for Q8 state requant (sigma-delta
    /// noise-shaping). Empty unless Q8 + `HIPFIRE_DN_STATE_EF`. Same element count
    /// as `s_matrices`; carries the previous step's quant error so the next
    /// requant cancels it — DeltaNet's contractive decay damps the shaped noise,
    /// yielding ~FP32-grade state at Q8's byte container.
    pub s_ef_residual: Vec<GpuTensor>,
    /// Current quantization mode
    pub quant: StateQuant,
}

impl DeltaNetState {
    /// EF residual for a delta-layer, if error-feedback is active (Q8 + flag).
    /// `None` ⇒ callers pass null ⇒ kernel uses the legacy stochastic-rounding requant.
    #[inline]
    pub fn ef_residual(&self, idx: usize) -> Option<&GpuTensor> {
        self.s_ef_residual.get(idx)
    }

    /// Non-owning single-lane view into state allocated by
    /// [`Self::new_batched_with_quant`]. Used only to seed prompts through the
    /// existing sequential prefill path. The returned view must not be freed.
    pub(crate) fn q8_lane_view(
        &self,
        config: &Qwen35Config,
        lane: usize,
        batch: usize,
    ) -> HipResult<Self> {
        if self.quant != StateQuant::Q8 || lane >= batch {
            return Err(HipError::new(
                0,
                "DeltaNet q8_lane_view requires Q8 state and a valid lane",
            ));
        }
        let n_heads = config.linear_num_value_heads;
        let hd = config.linear_value_head_dim;
        let s_elems = n_heads * hd * hd;
        let scale_elems = n_heads * hd;
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_elems = conv_channels * (config.conv_kernel_dim - 1);

        let byte_view = |t: &GpuTensor, off: usize, bytes: usize, dtype: DType| {
            let ptr = unsafe { (t.buf.as_ptr() as *mut u8).add(off) as *mut std::ffi::c_void };
            GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) },
                shape: vec![bytes / dtype.size()],
                dtype,
            }
        };
        Ok(Self {
            s_matrices: self
                .s_matrices
                .iter()
                .map(|t| byte_view(t, lane * s_elems, s_elems, DType::Raw))
                .collect(),
            s_scales: self
                .s_scales
                .iter()
                .map(|t| byte_view(t, lane * scale_elems * 4, scale_elems * 4, DType::F32))
                .collect(),
            conv_states: self
                .conv_states
                .iter()
                .map(|t| byte_view(t, lane * conv_elems * 4, conv_elems * 4, DType::F32))
                .collect(),
            s_ef_residual: self
                .s_ef_residual
                .iter()
                .map(|t| byte_view(t, lane * s_elems * 2, s_elems * 2, DType::F16))
                .collect(),
            quant: StateQuant::Q8,
        })
    }

    pub fn new(gpu: &mut Gpu, config: &Qwen35Config) -> HipResult<Self> {
        Self::new_with_quant(gpu, config, StateQuant::Q8)
    }

    pub fn new_with_quant(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<Self> {
        Self::new_batched_with_quant(gpu, config, quant, 1)
    }

    /// Allocate lane-major recurrent state for independent-sequence decode.
    ///
    /// The ordinary state has an implicit batch of one.  This variant keeps
    /// the same per-layer vectors, but every tensor is laid out as
    /// `[batch, ...single-lane shape...]`.  It is intentionally consumed only
    /// by [`Qwen35DecodeBatchState`]: passing it to sequential prefill would
    /// advance lane 0 and leave the other lanes stale.
    pub fn new_batched_with_quant(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        quant: StateQuant,
        batch: usize,
    ) -> HipResult<Self> {
        assert!(batch > 0, "DeltaNetState batch must be non-zero");
        let n_delta_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        let s_dim = config.linear_key_head_dim; // 128
        let n_heads = config.linear_num_value_heads; // 16
        let s_size_per_lane = n_heads * s_dim * s_dim; // 16 * 128 * 128 = 262144
        let s_size = batch * s_size_per_lane;

        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = batch * conv_channels * (config.conv_kernel_dim - 1);

        // Error-feedback (sigma-delta) requant for Q8 state — DEFAULT ON as of
        // 2026-06-08. q8_ef ≈ FP32 coherence at −0.7% decode vs FP32's −4.5% (best
        // spec-decode τ too), and far better than stochastic Q8 — DFlash 27b-prose
        // unique_ratio 0.625 vs 0.555, max_freq 0.055 vs 0.078. Also makes the DN
        // state DETERMINISTIC (no stochastic dither). Opt OUT with
        // HIPFIRE_DN_STATE_EF=0. Q8-only (FP32 has no requant; Q4 EF is future
        // work; the multi-GPU band split is still stochastic — new_with_quant_multi
        // leaves s_ef_residual empty). Residual is f16 per-element.
        let ef_enabled = quant == StateQuant::Q8
            && hipfire_config::developer_var("HIPFIRE_DN_STATE_EF")
                .map(|v| v != "0")
                .unwrap_or(true);

        // GpuTensor has no freeing Drop (free needs &mut Gpu). Mirror
        // alloc_k_v_vmm_filtered: on any mid-loop failure free every tensor
        // already pushed before propagating.
        let mut s_matrices = Vec::with_capacity(n_delta_layers);
        let mut s_scales = Vec::with_capacity(n_delta_layers);
        let mut conv_states = Vec::with_capacity(n_delta_layers);
        let mut s_ef_residual = Vec::with_capacity(if ef_enabled { n_delta_layers } else { 0 });
        let result = (|| -> HipResult<()> {
            for _ in 0..n_delta_layers {
                match quant {
                    StateQuant::FP32 => {
                        s_matrices.push(gpu.zeros(&[s_size], DType::F32)?);
                        s_scales.push(gpu.zeros(&[batch * n_heads], DType::F32)?);
                    }
                    StateQuant::Q8 => {
                        // int8 state: s_size bytes (1 byte each), per-row scales
                        let buf = gpu.hip.malloc(s_size)?;
                        if let Err(e) = gpu.hip.memset(&buf, 0, s_size) {
                            let _ = gpu.hip.free(buf);
                            return Err(e);
                        }
                        s_matrices.push(GpuTensor {
                            buf,
                            shape: vec![s_size],
                            dtype: DType::F32,
                        });
                        s_scales.push(gpu.zeros(&[batch * n_heads * s_dim], DType::F32)?);
                    }
                    StateQuant::Q4 => {
                        // 4-bit nibble-packed: s_size/2 bytes, per-row scales
                        let buf = gpu.hip.malloc(s_size / 2)?;
                        if let Err(e) = gpu.hip.memset(&buf, 0, s_size / 2) {
                            let _ = gpu.hip.free(buf);
                            return Err(e);
                        }
                        s_matrices.push(GpuTensor {
                            buf,
                            shape: vec![s_size / 2],
                            dtype: DType::F32,
                        });
                        s_scales.push(gpu.zeros(&[batch * n_heads * s_dim], DType::F32)?);
                    }
                }
                if ef_enabled {
                    s_ef_residual.push(gpu.zeros(&[s_size], DType::F16)?);
                }
                conv_states.push(gpu.zeros(&[conv_state_size], DType::F32)?);
            }
            Ok(())
        })();
        if let Err(err) = result {
            for tensor in s_matrices
                .drain(..)
                .chain(s_scales.drain(..))
                .chain(conv_states.drain(..))
                .chain(s_ef_residual.drain(..))
            {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(err);
        }
        Ok(Self {
            s_matrices,
            s_scales,
            conv_states,
            s_ef_residual,
            quant,
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.s_matrices {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_scales {
            let _ = gpu.free_tensor(t);
        }
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_ef_residual {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Reset all DeltaNet recurrent buffers to zero in place. Lets callers
    /// reuse a single `DeltaNetState` across independent chunks/sequences
    /// without allocating per chunk (which leaks since DeltaNetState has no
    /// Drop). Mirrors `ModelSlot::reset_state` in speculative.rs.
    ///
    /// Returns `Err` on the first HIP memset/memset_async failure so production
    /// rollback can attest `rolled_back:false`.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                for s in &self.s_matrices {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
                for s in &self.s_scales {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
                for s in &self.conv_states {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
                for s in &self.s_ef_residual {
                    gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                }
            }
            None => {
                for s in &self.s_matrices {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
                for s in &self.s_scales {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
                for s in &self.conv_states {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
                for s in &self.s_ef_residual {
                    gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                }
            }
        }
        Ok(())
    }

    /// Multi-GPU companion to `new_with_quant`. Each LA-layer's state is
    /// allocated on the device that owns the layer in the multi-GPU band
    /// split: `gpus.devices[gpus.device_for_layer(orig_layer_idx)]` for the
    /// `orig_layer_idx` of the LA-layer. Returns the state alongside the
    /// `la_to_device` mapping the daemon needs to route reset memsets to
    /// the correct device.
    pub fn new_with_quant_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<(Self, Vec<u8>)> {
        let s_dim = config.linear_key_head_dim;
        let n_heads = config.linear_num_value_heads;
        let s_size = n_heads * s_dim * s_dim;
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        let mut s_matrices = Vec::new();
        let mut s_scales = Vec::new();
        let mut conv_states = Vec::new();
        let mut la_to_device: Vec<u8> = Vec::new();

        for (orig_layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != LayerType::LinearAttention {
                continue;
            }
            let dev_idx = gpus.device_for_layer(orig_layer_idx);
            la_to_device.push(dev_idx as u8);
            let g = &mut gpus.devices[dev_idx];
            // g.hip.malloc/memset bypass the Stage 2 bind_thread audit
            // (HipRuntime methods don't carry a device id). Bind explicitly
            // before any raw HIP ops so allocations land on the right device.
            g.bind_thread()?;
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(g.zeros(&[s_size], DType::F32)?);
                    s_scales.push(g.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::Q8 => {
                    let buf = g.hip.malloc(s_size)?;
                    g.hip.memset(&buf, 0, s_size)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
                StateQuant::Q4 => {
                    let buf = g.hip.malloc(s_size / 2)?;
                    g.hip.memset(&buf, 0, s_size / 2)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size / 2],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
            }
            conv_states.push(g.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok((
            Self {
                s_matrices,
                s_scales,
                conv_states,
                // EF residual not wired for the multi-GPU band split (would need
                // per-device residual alloc routed by device_for_layer); empty ⇒
                // ef_residual() returns None ⇒ kernel uses the stochastic path.
                s_ef_residual: Vec::new(),
                quant,
            },
            la_to_device,
        ))
    }

    /// Free per-LA-layer tensors on the devices listed in `la_to_device`
    /// (the second tuple element returned by `new_with_quant_multi`).
    pub fn free_gpu_multi(self, gpus: &mut Gpus, la_to_device: &[u8]) {
        for (i, t) in self.s_matrices.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.s_scales.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.conv_states.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        // Empty today (multi-GPU EF not wired); free if/when residuals land.
        for (i, t) in self.s_ef_residual.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::DType;

    // ── SP2 — per-expert mixed-tier table builder (CPU-pure) ──────────────
    // `mixed_tier_table` is the testable core of `per_expert_tier_tables`:
    // empty/uniform columns collapse to None (uniform fast path), only a
    // genuinely multi-tier column yields Some(table).
    #[test]
    fn mixed_tier_table_empty_is_none() {
        // Paged mode: no resident experts → uniform fast path.
        assert_eq!(mixed_tier_table(Vec::new()), None);
    }

    #[test]
    fn mixed_tier_table_uniform_is_none() {
        // The common case: every expert one tier → None → byte-identical
        // uniform path, no allocation surfaced to MoeDtypes.
        let tiers = vec![DType::MQ4G256; 4];
        assert_eq!(mixed_tier_table(tiers), None);
        // Single-expert uniform column is also None.
        assert_eq!(mixed_tier_table(vec![DType::MQ6G256]), None);
    }

    #[test]
    fn mixed_tier_table_mixed_is_some_preserving_order() {
        // A re-quant overlay bumped experts 1 and 3 to MQ6 → Some, and the
        // table preserves per-expert order/dtype so dispatch buckets correctly.
        let tiers = vec![
            DType::MQ4G256,
            DType::MQ6G256,
            DType::MQ4G256,
            DType::MQ6G256,
        ];
        assert_eq!(mixed_tier_table(tiers.clone()), Some(tiers));
    }

    #[test]
    fn mixed_tier_table_mixed_first_differs() {
        // Guard against an off-by-one where only expert[0] is compared:
        // here every later expert differs from expert[0].
        let tiers = vec![DType::MQ4G256, DType::MQ6G256, DType::MQ6G256];
        assert_eq!(mixed_tier_table(tiers.clone()), Some(tiers));
    }

    #[test]
    fn dtype_from_quant_type_neutral_v2_one_to_one() {
        // Each qt maps one-to-one to its V2 DType and exact block bytes;
        // V2 DTypes are distinct from legacy MQ2/3/5/6.
        assert_eq!(dtype_from_quant_type(47).unwrap(), DType::MQ6G256V2);
        assert_eq!(dtype_from_quant_type(48).unwrap(), DType::MQ5G256V2);
        assert_eq!(dtype_from_quant_type(49).unwrap(), DType::MQ3G256V2);
        assert_eq!(dtype_from_quant_type(50).unwrap(), DType::MQ2G256V2);
        // Legacy unchanged.
        assert_eq!(dtype_from_quant_type(15).unwrap(), DType::MQ6G256);
        assert_ne!(dtype_from_quant_type(47).unwrap(), DType::MQ6G256);
        assert_ne!(dtype_from_quant_type(49).unwrap(), DType::MQ3G256);
        // Bad qt fails closed.
        assert!(dtype_from_quant_type(99).is_err());
        // qt44/45 still map to their V2/C DTypes.
        assert_eq!(dtype_from_quant_type(44).unwrap(), DType::MQ4G256V2);
        assert_eq!(dtype_from_quant_type(45).unwrap(), DType::MQ4CG256);
    }

    // ── mixed_expert_tag — frozen V1 0..6 + MQV2 7..18 ────────────────────
    // Exact ordered-pair → tag map; V1/V2 never collapse; unknown pairs Err.

    #[test]
    fn mixed_expert_tag_v1_pairs_retain_tags_0_through_6() {
        use DType::*;
        let accepted: &[(DType, DType, u8)] = &[
            (MQ4G256, MQ6G256, 0),
            (MQ4G256, MQ2G256Lloyd, 1),
            (MQ4G256, MQ4G256, 2),
            (MQ4G256, MQ3G256Lloyd, 3),
            (MQ4G256, MFP4G32E8, 4),
            (MQ4G256, MFP3G32E8, 5),
            (MQ4G256, MFP2G32E8, 6),
            (MQ6G256, MQ6G256, 0),
            (MQ2G256Lloyd, MQ2G256Lloyd, 1),
            (MQ3G256Lloyd, MQ3G256Lloyd, 3),
            (MFP4G32E8, MFP4G32E8, 4),
            (MFP3G32E8, MFP3G32E8, 5),
            (MFP2G32E8, MFP2G32E8, 6),
        ];
        for &(g, d, tag) in accepted {
            assert_eq!(
                mixed_expert_tag(g, d).expect("V1 accepted pair"),
                tag,
                "V1 gate={g:?} down={d:?}"
            );
        }
    }

    #[test]
    fn mixed_expert_tag_mqv2_pairs_emit_exact_tags_7_through_18() {
        use DType::*;
        // Frozen contract: exact tags 7..18; no V2→V1 collapse.
        let accepted: &[(DType, DType, u8)] = &[
            (MQ4G256V2, MQ4G256V2, 7),
            (MQ6G256V2, MQ6G256V2, 8),
            (MQ4G256V2, MQ6G256, 9),
            (MQ4G256V2, MQ2G256Lloyd, 10),
            (MQ4G256V2, MQ4G256, 11),
            (MQ4G256, MQ4G256V2, 12),
            (MQ4G256V2, MQ3G256Lloyd, 13),
            (MQ4G256V2, MFP4G32E8, 14),
            (MQ4G256V2, MFP3G32E8, 15),
            (MQ4G256V2, MFP2G32E8, 16),
            (MQ4G256V2, MQ6G256V2, 17),
            (MQ4G256, MQ6G256V2, 18),
        ];
        for &(g, d, tag) in accepted {
            assert_eq!(
                mixed_expert_tag(g, d).expect("MQV2 accepted pair"),
                tag,
                "MQV2 gate={g:?} down={d:?}"
            );
        }
    }

    #[test]
    fn mixed_expert_tag_v2_never_collapses_to_v1_tags() {
        use DType::*;
        // Former collapses that incorrectly reused tags 0..6 must now be distinct.
        assert_ne!(
            mixed_expert_tag(MQ4G256V2, MQ6G256).unwrap(),
            mixed_expert_tag(MQ4G256, MQ6G256).unwrap()
        );
        assert_ne!(
            mixed_expert_tag(MQ4G256V2, MQ4G256V2).unwrap(),
            mixed_expert_tag(MQ4G256, MQ4G256).unwrap()
        );
        assert_ne!(
            mixed_expert_tag(MQ4G256V2, MQ4G256).unwrap(),
            mixed_expert_tag(MQ4G256, MQ4G256).unwrap()
        );
        assert_ne!(
            mixed_expert_tag(MQ4G256, MQ4G256V2).unwrap(),
            mixed_expert_tag(MQ4G256, MQ4G256).unwrap()
        );
        assert_eq!(mixed_expert_tag(MQ4G256V2, MQ6G256).unwrap(), 9);
        assert_eq!(mixed_expert_tag(MQ4G256V2, MQ4G256V2).unwrap(), 7);
        assert_eq!(mixed_expert_tag(MQ4G256V2, MQ4G256).unwrap(), 11);
        assert_eq!(mixed_expert_tag(MQ4G256, MQ4G256V2).unwrap(), 12);
    }

    #[test]
    fn mixed_expert_tag_unknown_pairs_err() {
        use DType::*;
        // Cross-direction and unsupported V2 combos stay Err.
        let rejected: &[(DType, DType)] = &[
            (MQ6G256, MQ4G256),
            (MQ6G256V2, MQ4G256V2),
            (MQ6G256V2, MQ4G256),
            (MQ6G256, MQ6G256V2),
            (MQ6G256V2, MQ6G256),
            (MQ4G256V2, MQ5G256V2),
            (MQ2G256V2, MQ2G256V2),
            (MQ3G256V2, MQ3G256V2),
            (MQ5G256V2, MQ5G256V2),
            (MQ4G256Lloyd, MQ4G256Lloyd),
            (Q8_0, Q8_0),
            (F16, F16),
            (ParoQ4G128, ParoQ4G128),
            (MQ2G256Lloyd, MQ4G256),
            (MQ3G256Lloyd, MQ4G256V2),
            (MFP4G32E8, MQ4G256),
        ];
        for &(g, d) in rejected {
            let err = mixed_expert_tag(g, d).expect_err(&format!("gate={g:?} down={d:?}"));
            assert!(
                err.message.contains("unsupported dtype pair"),
                "gate={g:?} down={d:?}: {}",
                err.message
            );
        }
    }

    #[test]
    fn mixed_expert_tag_gl_either_side_err() {
        use DType::*;
        for gate in [MQ2G256GL, MQ3G256GL, MQ4G256, MQ4G256V2] {
            for down in [MQ2G256GL, MQ3G256GL, MQ4G256, MQ4G256V2] {
                let gl =
                    matches!(gate, MQ2G256GL | MQ3G256GL) || matches!(down, MQ2G256GL | MQ3G256GL);
                if !gl {
                    continue;
                }
                let err = mixed_expert_tag(gate, down).expect_err("GL must reject");
                assert!(
                    err.message.contains("GL dtype not supported"),
                    "gate={gate:?} down={down:?}: {}",
                    err.message
                );
            }
        }
    }
}
