//! dots.ocr model types: `DotsOcrConfig`, `DotsOcrWeights`, and the
//! free-function entry points for the vision-tower forward pass.
//!
//! The **text side** is wholly delegated to [`hipfire_arch_qwen2`]:
//! [`DotsOcrConfig::text`] is a `Qwen2Config`, [`DotsOcrWeights::text`]
//! is a `Qwen2Weights`, and the per-decode `State` (carried by the
//! `Architecture` trait) is `Qwen2State`. The text forward path is
//! `hipfire_arch_qwen2::qwen2::forward_step{,_greedy}` invoked directly
//! by the daemon (no wrapper), which keeps the hot-path static-dispatch
//! invariant from the trait module's design.
//!
//! The **vision side** lives entirely in this module. It owns a
//! 42-block `DotsVisionTransformer` (RMSNorm + SwiGLU + 2-D RoPE +
//! non-causal attention, per §2.2 of the plan) plus a LayerNorm-based
//! `PatchMerger` (per §2.4). The encoder is one-shot: it takes a
//! preprocessed patch tensor (produced by [`crate::image`]) and emits
//! merged visual tokens that the daemon splices into the prompt at
//! `<|imgpad|>` positions during prefill.
//!
//! # Bring-up status (rev 0)
//!
//! - [`DotsOcrConfig::from_hfq`] — full parser landing soon. Stub
//!   currently returns the dots.ocr-shipped defaults (the model only
//!   has one published checkpoint, so the dependency on metadata is
//!   small for bring-up).
//! - [`DotsOcrWeights::load`] — text-side delegated to
//!   `Qwen2Weights::load`; vision-side currently a stub that returns
//!   an empty struct. Vision weight load lands together with
//!   `vision_forward` in phase 2c.
//! - [`vision_forward`] — stub returning an error. Real implementation
//!   in phase 2c.
//!
//! # TODO(transformer-extraction)
//!
//! The cross-arch dequant + GPU-upload helpers (e.g. `load_f16_gpu`,
//! `load_f32_gpu`, HFQ4-dequant) are duplicated from
//! `hipfire-arch-qwen35-vl::qwen35_vl`. They land here in phase 2c
//! with matching markers on both sides for the eventual consolidation
//! PR (`hipfire_runtime::transformer::vision_*`).

use hip_bridge::HipResult;
use hipfire_arch_qwen2::qwen2::{Qwen2Config, Qwen2Weights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ─────────────────────────────────────────────────────────────

/// dots.ocr vision-tower model-shape constants. Parsed from
/// `hfq.metadata_json[config][vision_config]`.
///
/// Field notes:
/// - `embed_dim`: 1536. Matches the text decoder's `hidden_size`.
/// - `num_hidden_layers`: 42 (3.5× the text decoder's 28).
/// - `num_attention_heads`: 12, `head_dim = embed_dim / num_attention_heads = 128`.
/// - `intermediate_size`: 4224 (smaller than text FFN at 8960).
/// - `patch_size`: 14, `spatial_merge_size`: 2 → effective patch
///   stride 28 after merger.
/// - `temporal_patch_size`: 1. dots.ocr does not model time.
/// - `num_channels`: 3 (RGB).
/// - `use_bias`: false for every linear inside the block (attn QKV,
///   attn proj, FFN fc1/fc2/fc3). Only `patch_embed.proj` and the
///   merger MLP have bias on disk.
/// - `post_norm`: true. After the 42-block stack, apply RMSNorm before
///   the merger (`vision_tower.post_trunk_norm.weight`).
/// - `rms_norm_eps`: 1e-5 (note: 100× larger than text's 1e-6 — keep
///   them separate, do not unify).
#[derive(Debug, Clone)]
pub struct DotsVisionConfig {
    pub embed_dim: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub num_channels: usize,
    pub use_bias: bool,
    pub post_norm: bool,
    pub rms_norm_eps: f32,
    /// Post-merger output dim. Must equal the text decoder's
    /// `hidden_size` so the merged visual tokens can be spliced in as
    /// drop-in `embed_dim` vectors during prefill.
    pub out_hidden_size: usize,
}

impl DotsVisionConfig {
    /// Defaults matching the published dots.ocr checkpoint. Used as
    /// fallbacks when individual `vision_config.*` keys are missing.
    pub fn dots_ocr_defaults() -> Self {
        Self {
            embed_dim: 1536,
            num_hidden_layers: 42,
            num_attention_heads: 12,
            head_dim: 128,
            intermediate_size: 4224,
            patch_size: 14,
            spatial_merge_size: 2,
            temporal_patch_size: 1,
            num_channels: 3,
            use_bias: false,
            post_norm: true,
            rms_norm_eps: 1e-5,
            out_hidden_size: 1536,
        }
    }
}

/// Outer dots.ocr config: text + vision side-by-side.
///
/// The text side is a full `Qwen2Config` (28-layer dense decoder with
/// `tie_word_embeddings=false` — note divergence from Qwen2-1.5B-Instruct,
/// which has `tie=true`). The vision side carries the
/// `DotsVisionTransformer` constants.
#[derive(Debug, Clone)]
pub struct DotsOcrConfig {
    pub text: Qwen2Config,
    pub vision: DotsVisionConfig,
}

impl DotsOcrConfig {
    /// Parse a `DotsOcrConfig` out of an HFQ file's metadata.
    ///
    /// Text side delegates to `Qwen2Config::from_hfq` (which already
    /// handles the `text_config` nesting that dots.ocr uses). Vision
    /// side reads `config.vision_config.*` with `dots_ocr_defaults()`
    /// fallbacks for missing keys.
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let text = Qwen2Config::from_hfq(hfq)?;
        let vision = parse_vision_config(&hfq.metadata_json)
            .unwrap_or_else(DotsVisionConfig::dots_ocr_defaults);
        Ok(Self { text, vision })
    }
}

fn parse_vision_config(metadata_json: &str) -> Option<DotsVisionConfig> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    let vc = meta.get("config")?.get("vision_config")?;
    let defaults = DotsVisionConfig::dots_ocr_defaults();

    let embed_dim = vc.get("embed_dim").and_then(|v| v.as_u64())
        .or_else(|| vc.get("hidden_size").and_then(|v| v.as_u64()))
        .map(|v| v as usize)
        .unwrap_or(defaults.embed_dim);
    let num_attention_heads = vc.get("num_attention_heads").and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(defaults.num_attention_heads);
    let num_hidden_layers = vc.get("num_hidden_layers").and_then(|v| v.as_u64())
        .or_else(|| vc.get("depth").and_then(|v| v.as_u64()))
        .map(|v| v as usize)
        .unwrap_or(defaults.num_hidden_layers);
    let head_dim = vc.get("head_dim").and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(embed_dim / num_attention_heads);
    let intermediate_size = vc.get("intermediate_size").and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(defaults.intermediate_size);
    let patch_size = vc.get("patch_size").and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(defaults.patch_size);
    let spatial_merge_size = vc.get("spatial_merge_size").and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(defaults.spatial_merge_size);
    let temporal_patch_size = vc.get("temporal_patch_size").and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(defaults.temporal_patch_size);
    let num_channels = vc.get("num_channels").and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(defaults.num_channels);
    let use_bias = vc.get("use_bias").and_then(|v| v.as_bool())
        .unwrap_or(defaults.use_bias);
    let post_norm = vc.get("post_norm").and_then(|v| v.as_bool())
        .unwrap_or(defaults.post_norm);
    let rms_norm_eps = vc.get("rms_norm_eps").and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(defaults.rms_norm_eps);
    // Text-hidden-size for out_hidden_size fallback (matches the
    // post-merger output dim — must match text decoder).
    let out_hidden_size = vc.get("out_hidden_size").and_then(|v| v.as_u64())
        .or_else(|| {
            meta.get("config")?
                .get("text_config")
                .and_then(|tc| tc.get("hidden_size"))
                .and_then(|v| v.as_u64())
        })
        .map(|v| v as usize)
        .unwrap_or(defaults.out_hidden_size);

    Some(DotsVisionConfig {
        embed_dim,
        num_hidden_layers,
        num_attention_heads,
        head_dim,
        intermediate_size,
        patch_size,
        spatial_merge_size,
        temporal_patch_size,
        num_channels,
        use_bias,
        post_norm,
        rms_norm_eps,
        out_hidden_size,
    })
}

// ─── Vision weights ─────────────────────────────────────────────────────

/// Per-block weights for one `DotsVisionBlock`. All linears are
/// bias-free (`use_bias=false`); only the norm scales are present.
///
/// Layout on disk:
/// ```text
/// vision_tower.blocks.{i}.norm1.weight        [embed_dim]
/// vision_tower.blocks.{i}.attn.qkv.weight     [3*embed_dim, embed_dim]
/// vision_tower.blocks.{i}.attn.proj.weight    [embed_dim, embed_dim]
/// vision_tower.blocks.{i}.norm2.weight        [embed_dim]
/// vision_tower.blocks.{i}.mlp.fc1.weight      [intermediate_size, embed_dim]
/// vision_tower.blocks.{i}.mlp.fc3.weight      [intermediate_size, embed_dim]
/// vision_tower.blocks.{i}.mlp.fc2.weight      [embed_dim, intermediate_size]
/// ```
///
/// `fc13_proj` is the load-time concat of `fc1` and `fc3` along the
/// output axis (`[2 * intermediate_size, embed_dim]`) so the SwiGLU
/// MLP runs as one GEMM instead of two — the (a) option from §5
/// phase 2 of the plan. Lands together with `vision_forward` in
/// phase 2c.
pub struct DotsVisionBlockWeights {
    pub norm1_w: GpuTensor,
    /// `[3 * embed_dim, embed_dim]` F16 fused Q/K/V projection.
    /// `use_bias=false`. Stored on GPU in F16 for `gemm_f16`; HFQ4 /
    /// Q8 source quant types are dequantized at load time per the
    /// qwen35-vl pattern (vision tower is one-shot per image so f16
    /// dequant is acceptable; batched HFQ4 GEMM is a future perf pass).
    pub qkv_w: GpuTensor,
    /// `[embed_dim, embed_dim]` F16 attention output projection.
    /// `use_bias=false`.
    pub proj_w: GpuTensor,
    pub norm2_w: GpuTensor,
    /// `[2 * intermediate_size, embed_dim]` F16 — load-time concat of
    /// `fc1` and `fc3` along the M axis. `silu(y[:H]) * y[H:]` after
    /// this GEMM. `use_bias=false`.
    pub fc13_proj: GpuTensor,
    /// `[embed_dim, intermediate_size]` F16. `use_bias=false`.
    pub fc2: GpuTensor,
}

/// Full dots.ocr vision tower weights. Owned by [`DotsOcrWeights`].
///
/// Layout on disk (top-level, vision-tower-relative):
/// ```text
/// vision_tower.patch_embed.patchifier.proj.weight    [embed_dim, 3, 14, 14]
/// vision_tower.patch_embed.patchifier.proj.bias      [embed_dim]
/// vision_tower.patch_embed.patchifier.norm.weight    [embed_dim]   (RMSNorm)
/// vision_tower.blocks.{0..42}                         (see DotsVisionBlockWeights)
/// vision_tower.post_trunk_norm.weight                 [embed_dim]   (post-stack RMSNorm)
/// vision_tower.merger.ln_q.weight                     [embed_dim]   (LayerNorm)
/// vision_tower.merger.ln_q.bias                       [embed_dim]
/// vision_tower.merger.mlp.0.weight                    [6144, 6144]
/// vision_tower.merger.mlp.0.bias                      [6144]
/// vision_tower.merger.mlp.2.weight                    [out_hidden_size, 6144]
/// vision_tower.merger.mlp.2.bias                      [out_hidden_size]
/// ```
pub struct DotsVisionWeights {
    /// Conv2d-style patch projection, F16. Reshape on load from
    /// `[embed_dim, 3, 14, 14]` to `[embed_dim, 588]` (= 3 * 14 * 14)
    /// for the GEMM. Has bias.
    pub patch_embed_w: GpuTensor,
    pub patch_embed_b: GpuTensor,
    /// RMSNorm applied right after patch_embed projection.
    pub patch_embed_norm: GpuTensor,
    pub blocks: Vec<DotsVisionBlockWeights>,
    /// Post-trunk RMSNorm, applied to the encoder output before the
    /// merger (because `post_norm=true`).
    pub post_trunk_norm: GpuTensor,
    /// PatchMerger pre-norm: LayerNorm (not RMSNorm — note divergence)
    /// with bias.
    pub merger_ln_w: GpuTensor,
    pub merger_ln_b: GpuTensor,
    /// `mlp.0`: linear(merge_dim → merge_dim), F16. Bias on disk.
    pub merger_fc1_w: GpuTensor,
    pub merger_fc1_b: GpuTensor,
    /// `mlp.2`: linear(merge_dim → out_hidden_size), F16. Bias on disk.
    /// `mlp.1` is GELU (no params); slot 2 carries the second linear.
    pub merger_fc2_w: GpuTensor,
    pub merger_fc2_b: GpuTensor,
}

impl DotsVisionWeights {
    /// Return all GPU buffers to the pool. Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.patch_embed_w);
        let _ = gpu.free_tensor(self.patch_embed_b);
        let _ = gpu.free_tensor(self.patch_embed_norm);
        for b in self.blocks {
            let _ = gpu.free_tensor(b.norm1_w);
            let _ = gpu.free_tensor(b.qkv_w);
            let _ = gpu.free_tensor(b.proj_w);
            let _ = gpu.free_tensor(b.norm2_w);
            let _ = gpu.free_tensor(b.fc13_proj);
            let _ = gpu.free_tensor(b.fc2);
        }
        let _ = gpu.free_tensor(self.post_trunk_norm);
        let _ = gpu.free_tensor(self.merger_ln_w);
        let _ = gpu.free_tensor(self.merger_ln_b);
        let _ = gpu.free_tensor(self.merger_fc1_w);
        let _ = gpu.free_tensor(self.merger_fc1_b);
        let _ = gpu.free_tensor(self.merger_fc2_w);
        let _ = gpu.free_tensor(self.merger_fc2_b);
    }
}

// ─── Outer weights wrapper ──────────────────────────────────────────────

/// dots.ocr weights: text decoder + vision tower side-by-side.
///
/// Text-side load delegates to `Qwen2Weights::load` unchanged
/// (dots.ocr stores text weights as `model.*`, identical to plain
/// Qwen2). Vision-side load happens after.
pub struct DotsOcrWeights {
    pub text: Qwen2Weights,
    pub vision: DotsVisionWeights,
}

impl DotsOcrWeights {
    /// Load both text and vision weights from a dots.ocr HFQ file.
    pub fn load(
        hfq: &mut HfqFile,
        cfg: &DotsOcrConfig,
        gpu: &mut Gpu,
    ) -> Result<Self, String> {
        let text = Qwen2Weights::load(hfq, &cfg.text, gpu)?;
        let vision = load_vision_weights(hfq, &cfg.vision, gpu)
            .map_err(|e| format!("dots-ocr: load_vision_weights failed: {e:?}"))?;
        Ok(Self { text, vision })
    }

    /// Free both halves' GPU buffers.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.text.free_gpu(gpu);
        self.vision.free_gpu(gpu);
    }
}

/// Load all dots.ocr vision-tower weights from an HFQ file.
///
/// Tensor name layout (verified against the safetensors manifest at
/// `docs/plans/qwen_2.0_vlm_plus_dots_ocr.dots_ocr_manifest.txt`):
///
/// - `vision_tower.patch_embed.patchifier.proj.{weight,bias}` — Conv2d
///   weight is 4-D `[embed_dim, 3, 14, 14]` on disk; we reshape to a
///   2-D linear `[embed_dim, 3*14*14 = 588]` (free, contiguous memory).
/// - `vision_tower.patch_embed.patchifier.norm.weight` — RMSNorm scale.
/// - For each of `num_hidden_layers` blocks:
///   `vision_tower.blocks.{i}.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,
///   mlp.fc2,mlp.fc3}.weight` — all linears `use_bias=false` per §2.2.
///   `fc1` and `fc3` are concatenated along the output (M) axis at
///   load time into `fc13_proj` so the SwiGLU MLP runs as one GEMM
///   instead of two (option (a) of plan §5 phase 2).
/// - `vision_tower.post_trunk_norm.weight` — RMSNorm scale.
/// - `vision_tower.merger.ln_q.{weight,bias}` — LayerNorm (NOT
///   RMSNorm; note divergence from vision blocks).
/// - `vision_tower.merger.mlp.{0,2}.{weight,bias}` — both linears
///   carry bias; `mlp.1` is GELU (no params).
pub fn load_vision_weights(
    hfq: &HfqFile,
    cfg: &DotsVisionConfig,
    gpu: &mut Gpu,
) -> HipResult<DotsVisionWeights> {
    let h = cfg.embed_dim;
    let intermediate = cfg.intermediate_size;
    let patch_dim = cfg.num_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let merge_dim = h * cfg.spatial_merge_size * cfg.spatial_merge_size;
    eprintln!(
        "  loading dots-ocr vision tower: embed_dim={h} layers={} intermediate={intermediate} \
         patch_dim={patch_dim} merge_dim={merge_dim}",
        cfg.num_hidden_layers,
    );

    // ── patch_embed ───────────────────────────────────────────────
    //
    // Conv2d weight on disk is [embed_dim, 3, 14, 14] = [embed_dim,
    // 588] elements when flattened C-major. `load_f16_or_dequant` sees
    // only the byte stream; the 4-D shape is metadata.
    //
    // n_elements = h * patch_dim for the GEMM shape `[h, patch_dim]`.
    let patch_embed_w = load_f16_or_dequant(
        hfq, gpu, "vision_tower.patch_embed.patchifier.proj.weight", h * patch_dim,
    )?;
    let patch_embed_b = load_bias_f32(
        hfq, gpu, "vision_tower.patch_embed.patchifier.proj.bias", h,
    )?;
    let patch_embed_norm = load_norm_weight_raw(
        hfq, gpu, "vision_tower.patch_embed.patchifier.norm.weight", h,
    )?;

    // ── blocks ────────────────────────────────────────────────────
    let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        if i % 7 == 0 {
            eprintln!("  loading vision block {i}/{}", cfg.num_hidden_layers);
        }
        let p = format!("vision_tower.blocks.{i}");
        let norm1_w = load_norm_weight_raw(hfq, gpu, &format!("{p}.norm1.weight"), h)?;
        let qkv_w = load_f16_or_dequant(hfq, gpu, &format!("{p}.attn.qkv.weight"), 3 * h * h)?;
        let proj_w = load_f16_or_dequant(hfq, gpu, &format!("{p}.attn.proj.weight"), h * h)?;
        let norm2_w = load_norm_weight_raw(hfq, gpu, &format!("{p}.norm2.weight"), h)?;
        // Load-time concat: fc13_proj = [fc1; fc3] along the output
        // (M) axis. Both tensors get dequantized to F16, then we
        // concatenate the F16 bytes (= row-wise concat in matrix-
        // shape since rows are independent in row-major storage).
        let fc13_proj = load_f16_or_dequant_concat_rows(
            hfq, gpu,
            &format!("{p}.mlp.fc1.weight"),
            &format!("{p}.mlp.fc3.weight"),
            intermediate * h, intermediate * h,
        )?;
        let fc2 = load_f16_or_dequant(hfq, gpu, &format!("{p}.mlp.fc2.weight"), h * intermediate)?;
        blocks.push(DotsVisionBlockWeights { norm1_w, qkv_w, proj_w, norm2_w, fc13_proj, fc2 });
    }

    // ── post-trunk norm + merger ──────────────────────────────────
    let post_trunk_norm = load_norm_weight_raw(
        hfq, gpu, "vision_tower.post_trunk_norm.weight", h,
    )?;
    eprintln!("  loading vision merger");
    // ln_q is a LayerNorm, NOT an RMSNorm — but `load_norm_weight_raw`
    // is just "F16/F32 bytes → f32 upload, no offset". Same shape, fine
    // to reuse. The LayerNorm-ness is in how the FORWARD kernel uses
    // it (gpu.layernorm_batched, which also takes a bias).
    let merger_ln_w = load_norm_weight_raw(hfq, gpu, "vision_tower.merger.ln_q.weight", h)?;
    let merger_ln_b = load_bias_f32(hfq, gpu, "vision_tower.merger.ln_q.bias", h)?;
    let merger_fc1_w = load_f16_or_dequant(
        hfq, gpu, "vision_tower.merger.mlp.0.weight", merge_dim * merge_dim,
    )?;
    let merger_fc1_b = load_bias_f32(hfq, gpu, "vision_tower.merger.mlp.0.bias", merge_dim)?;
    let merger_fc2_w = load_f16_or_dequant(
        hfq, gpu, "vision_tower.merger.mlp.2.weight", cfg.out_hidden_size * merge_dim,
    )?;
    let merger_fc2_b = load_bias_f32(
        hfq, gpu, "vision_tower.merger.mlp.2.bias", cfg.out_hidden_size,
    )?;

    Ok(DotsVisionWeights {
        patch_embed_w,
        patch_embed_b,
        patch_embed_norm,
        blocks,
        post_trunk_norm,
        merger_ln_w,
        merger_ln_b,
        merger_fc1_w,
        merger_fc1_b,
        merger_fc2_w,
        merger_fc2_b,
    })
}

// ─── Loader helpers (TODO(transformer-extraction): cross-arch dupes) ────

/// Load an F32 norm scale (no `+= 1.0` offset). Mirrors
/// `hipfire-arch-qwen2::qwen2::load_norm_weight_raw` —
/// both are the same shape (RMSNorm w/o +1 bake). Dots.ocr also uses
/// this for the merger's LayerNorm weight (the bias is loaded
/// separately via `load_bias_f32`).
///
/// TODO(transformer-extraction): pull this + the qwen2 + qwen35
/// variants into `hipfire_runtime::transformer::norm` during the
/// consolidation PR.
fn load_norm_weight_raw(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data_vec(name)
        .unwrap_or_else(|| panic!("dots-ocr: tensor not found: {name}"));
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        qt => panic!("dots-ocr: expected F16/F32 for norm {name}, got qt={qt}"),
    };
    assert_eq!(
        f32_data.len(), n,
        "dots-ocr: norm {name} has {} elements, expected {n}", f32_data.len(),
    );
    gpu.upload_f32(&f32_data, &[n])
}

/// Load a bias tensor as F32 on GPU. Mirrors
/// `hipfire-arch-qwen2::qwen2::load_bias_f32`. Same accepted
/// quant_types (F16 / F32 only — biases are tiny, never worth
/// quantising).
///
/// TODO(transformer-extraction): see norm helper.
fn load_bias_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data_vec(name)
        .unwrap_or_else(|| panic!("dots-ocr: tensor not found: {name}"));
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        qt => panic!("dots-ocr: expected F16/F32 for bias {name}, got qt={qt}"),
    };
    assert_eq!(
        f32_data.len(), n,
        "dots-ocr: bias {name} has {} elements, expected {n}", f32_data.len(),
    );
    gpu.upload_f32(&f32_data, &[n])
}

/// Load a linear weight and ensure it ends up as F16 on GPU,
/// dequantising HFQ4/Q8 → F16 at load time if needed.
///
/// Mirrors `hipfire-arch-qwen35-vl::qwen35_vl::load_f16_gpu` — the
/// vision tower is one-shot per image (not in the per-decode-step hot
/// path) so dequantising on load and using `gemm_f16` for the batched
/// projection is cheaper than wiring batched HFQ4 GEMM kernels for
/// every per-block linear. Promotion to native quantised batched GEMM
/// is a deferred perf pass under the Δ ≥ 5 % rule.
///
/// `n_elements` is the logical element count of the resulting `[M, K]`
/// matrix (= M * K). Used for shape verification on the GPU upload.
///
/// TODO(transformer-extraction): pull this + qwen35-vl's load_f16_gpu
/// into `hipfire_runtime::transformer::vision_weights` during the
/// consolidation PR.
fn load_f16_or_dequant(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    n_elements: usize,
) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data_vec(name)
        .unwrap_or_else(|| panic!("dots-ocr: tensor not found: {name}"));
    match info.quant_type {
        1 => {
            // F16 — upload directly.
            assert_eq!(
                data.len(), 2 * n_elements,
                "dots-ocr: {name} F16 has {} bytes, expected 2 * {n_elements} = {}",
                data.len(), 2 * n_elements,
            );
            gpu.upload_raw(&data, &[n_elements])
        }
        2 => {
            // F32 → cast to F16 then upload.
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            gpu.upload_raw(&f16_bytes, &[n_elements])
        }
        6 | 7 => {
            // HFQ4G256 / HFQ4G128 → dequant to F32, cast to F16, upload.
            let group_size = info.group_size as usize;
            let f32_data = dequant_hfq4(&data, n_elements, group_size);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            gpu.upload_raw(&f16_bytes, &[n_elements])
        }
        qt => panic!(
            "dots-ocr: unsupported weight quant_type {qt} for {name}. \
             load_f16_or_dequant handles qt ∈ {{1 (F16), 2 (F32), 6 (HFQ4G256), 7 (HFQ4G128)}}.",
        ),
    }
}

/// Load TWO linear weights and concatenate them along the output (M)
/// axis at load time into a single F16 GPU tensor. Used for the
/// SwiGLU fc1+fc3 fusion: both have shape `[intermediate, embed_dim]`,
/// concatenated they form `fc13_proj` of `[2*intermediate, embed_dim]`
/// so the SwiGLU MLP runs as one batched GEMM instead of two — option
/// (a) of plan §5 phase 2.
///
/// Concatenation happens AFTER dequantisation, so source quant_types
/// don't need to match (though in practice they will for a single
/// HFQ file). Output is always F16 on GPU.
fn load_f16_or_dequant_concat_rows(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name_a: &str,
    name_b: &str,
    n_elements_a: usize,
    n_elements_b: usize,
) -> HipResult<GpuTensor> {
    // Dequantise + cast to F16 bytes for each side, then concatenate.
    let bytes_a = dequant_to_f16_bytes(hfq, name_a, n_elements_a);
    let bytes_b = dequant_to_f16_bytes(hfq, name_b, n_elements_b);
    let mut combined = Vec::with_capacity(bytes_a.len() + bytes_b.len());
    combined.extend_from_slice(&bytes_a);
    combined.extend_from_slice(&bytes_b);
    gpu.upload_raw(&combined, &[n_elements_a + n_elements_b])
}

/// Shared helper: dequant any supported source quant_type to F16 byte
/// stream (little-endian per-element). Returns the f16 buffer ready
/// for `gpu.upload_raw`.
fn dequant_to_f16_bytes(hfq: &HfqFile, name: &str, n_elements: usize) -> Vec<u8> {
    let (info, data) = hfq.tensor_data_vec(name)
        .unwrap_or_else(|| panic!("dots-ocr: tensor not found: {name}"));
    match info.quant_type {
        1 => {
            // F16 — already F16, just hand back the bytes.
            assert_eq!(
                data.len(), 2 * n_elements,
                "dots-ocr: {name} F16 has {} bytes, expected {}", data.len(), 2 * n_elements,
            );
            data.to_vec()
        }
        2 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            f32_data.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes()).collect()
        }
        6 | 7 => {
            let group_size = info.group_size as usize;
            let f32_data = dequant_hfq4(&data, n_elements, group_size);
            f32_data.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes()).collect()
        }
        qt => panic!("dots-ocr: dequant_to_f16_bytes does not support quant_type {qt} for {name}"),
    }
}

/// Dequantise HFQ4G256 / HFQ4G128 to F32.
///
/// Block layout: `[scale: f32, zero: f32, group_size/2 bytes of
/// nibbles]`. Each nibble decodes to `scale * nibble + zero`. The
/// trailing group may have fewer than `group_size` elements (the
/// blob still allocates the full group_size in bytes, we truncate
/// `out` to `n_elements`).
///
/// Mirrors `hipfire-arch-qwen35-vl::qwen35_vl::dequant_hfq4`.
///
/// TODO(transformer-extraction): same destination as the helpers
/// above.
fn dequant_hfq4(data: &[u8], n_elements: usize, group_size: usize) -> Vec<f32> {
    let nibble_bytes = group_size / 2;
    let block_size = 8 + nibble_bytes; // 4-byte scale + 4-byte zero + nibbles
    let mut out = Vec::with_capacity(n_elements);
    let n_groups = n_elements.div_ceil(group_size);
    for g in 0..n_groups {
        let off = g * block_size;
        if off + 8 > data.len() { break; }
        let scale = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let zero = f32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        let nibbles = &data[off + 8..(off + block_size).min(data.len())];
        let base = g * group_size;
        for i in 0..group_size.min(n_elements.saturating_sub(base)) {
            let byte_idx = i / 2;
            if byte_idx >= nibbles.len() { break; }
            let nibble = if i % 2 == 0 {
                nibbles[byte_idx] & 0xF
            } else {
                nibbles[byte_idx] >> 4
            };
            out.push(scale * nibble as f32 + zero);
        }
    }
    out.truncate(n_elements);
    out
}

// ─── Vision-forward primitives ─────────────────────────────────────────

/// `linear_f16(W [out, in], X [n, in], bias [out]) -> Y [n, out]`.
///
/// Mirrors `hipfire-arch-qwen35-vl::qwen35_vl::linear_f16`. `gpu.gemm_f16`
/// produces `Y_t [out, n]`; we transpose to row-major `Y [n, out]` then
/// apply per-output-channel bias. Caller owns the returned tensor.
///
/// Used by the merger MLP and patch_embed (linears that have bias).
/// For the use_bias=false vision-block linears, see
/// [`linear_f16_no_bias`].
///
/// TODO(transformer-extraction): qwen35-vl has the same helper; pull
/// both into `hipfire_runtime::transformer::vision_linear` during the
/// consolidation PR.
#[allow(dead_code)]
pub(crate) fn linear_f16(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
    gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    gpu.transpose_f32(&yt, &y, out_dim, n)?;
    gpu.free_tensor(yt)?;
    gpu.bias_add_f32(&y, bias, n, out_dim)?;
    Ok(y)
}

/// `linear_f16(W [out, in], X [n, in]) -> Y [n, out]` — bias-free.
///
/// Identical to [`linear_f16`] minus the trailing `bias_add_f32`. Used
/// by the 42 `DotsVisionBlock` projections (qkv, proj, fc13, fc2) —
/// all of which have `use_bias=false` per §2.2 of the plan.
///
/// Saves one kernel launch per linear vs. calling `linear_f16` with
/// a zero-filled bias buffer.
#[allow(dead_code)]
pub(crate) fn linear_f16_no_bias(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
    gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    gpu.transpose_f32(&yt, &y, out_dim, n)?;
    gpu.free_tensor(yt)?;
    Ok(y)
}

// ─── Forward pass (phase 2c stub) ───────────────────────────────────────

/// Encode preprocessed patches through the 42-block vision tower and
/// the PatchMerger, returning post-merger visual embeddings ready for
/// `<|imgpad|>` substitution in the text prompt.
///
/// # Inputs
///
/// - `gpu`: compute context.
/// - `weights`: vision tower weights (`DotsVisionWeights`).
/// - `cfg`: vision config (`DotsVisionConfig`).
/// - `patches`: pre-extracted patch tensor in HF
///   `Qwen2VLImageProcessor` order — i.e. AFTER the
///   `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)` of §2.7. Shape is
///   `[grid_t * grid_h * grid_w, channels * temporal_patch_size *
///   patch_size * patch_size]`. For dots.ocr's
///   `temporal_patch_size=1`, this is `[N_patches, 3 * 14 * 14 = 588]`.
/// - `grid_h`, `grid_w`: post-patch grid dims (image_grid_thw without
///   the t-axis since temporal_patch_size=1).
///
/// # Output
///
/// `Vec<f32>` of shape `[N_patches / (spatial_merge_size^2),
/// out_hidden_size]` — one merged visual token per 2×2 spatial block.
///
/// # Phase 2c
///
/// Returns a "not yet implemented" error today. Real implementation
/// lands in phase 2c with:
/// 1. patch_embed GEMM + bias + RMSNorm
/// 2. 2-D RoPE prep (hpos/wpos reshape-permute-flatten)
/// 3. 42 blocks: RMSNorm → QKV → 2-D RoPE → vit_attention_f32
///    (non-causal) → o_proj → residual → RMSNorm → SwiGLU
///    (silu(fc13_y[:H]) * fc13_y[H:] → fc2) → residual
/// 4. post_trunk_norm (RMSNorm)
/// 5. merger: view(-1, 6144) → LayerNorm+bias → linear → GELU →
///    linear
///
/// # Gotchas (from Phase 0 item 2 — §2.9 of the plan)
///
/// - Attention scale is plain `1.0 / (head_dim as f32).sqrt()` —
///   no qk-norm, no learned scale, no `* -0.5` factor.
/// - For batch_size > 1, `image_grid_thw` builds a SINGLE flattened
///   sequence; cu_seqlens is image-major (cumsum of per-image
///   `t * h * w`) and must be `i32` for FA correctness.
/// - HF casts vision activations to bf16 at forward entry (line
///   493-494 in modeling_dots_vision.py). We compute in f32 and
///   cast the final merged tokens to f16/bf16 to match the text
///   decoder's embedding dtype before splicing.
/// - The merger output is already `out_hidden_size = text_hidden_size`.
///   NO additional projection layer between vision tower and text
///   embedding space — vision tokens substitute directly into the
///   `<|imgpad|>` positions via the daemon's `masked_scatter`-style
///   prefill loop.
pub fn vision_forward(
    _gpu: &mut Gpu,
    _weights: &DotsVisionWeights,
    _cfg: &DotsVisionConfig,
    _patches: &[f32],
    _grid_h: usize,
    _grid_w: usize,
) -> Result<Vec<f32>, String> {
    Err(
        "dots-ocr: vision_forward is a phase 2c stub. Implementation \
         pending — see docs/plans/qwen_2.0_vlm_plus_dots_ocr.md phase 2c."
            .to_string(),
    )
}

// ─── Token-id constants ─────────────────────────────────────────────────

/// `<|imgpad|>` — image-pad slot id, used by the daemon's prefill loop
/// to mark positions where merged visual tokens splice in.
pub const IMGPAD_ID: u32 = 151665;
/// `<|img|>` — image-start framing token.
pub const IMG_START_ID: u32 = 151666;
/// `<|endofimg|>` — image-end framing token.
pub const IMG_END_ID: u32 = 151667;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_defaults_match_plan() {
        let d = DotsVisionConfig::dots_ocr_defaults();
        assert_eq!(d.embed_dim, 1536);
        assert_eq!(d.num_hidden_layers, 42);
        assert_eq!(d.num_attention_heads, 12);
        assert_eq!(d.head_dim, 128);
        assert_eq!(d.intermediate_size, 4224);
        assert_eq!(d.patch_size, 14);
        assert_eq!(d.spatial_merge_size, 2);
        assert_eq!(d.temporal_patch_size, 1);
        assert!(!d.use_bias);
        assert!(d.post_norm);
        assert_eq!(d.rms_norm_eps, 1e-5);
        assert_eq!(d.out_hidden_size, 1536);
    }

    #[test]
    fn parse_vision_config_picks_up_overrides() {
        // Minimal fake metadata with a couple of vision_config overrides
        // — verifies the parser actually walks the JSON instead of
        // always returning defaults.
        let json = r#"{
          "config": {
            "vision_config": {
              "embed_dim": 2048,
              "num_hidden_layers": 24,
              "num_attention_heads": 16,
              "intermediate_size": 8192
            },
            "text_config": { "hidden_size": 2048 }
          }
        }"#;
        let cfg = parse_vision_config(json).unwrap();
        assert_eq!(cfg.embed_dim, 2048);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.head_dim, 2048 / 16);
        assert_eq!(cfg.intermediate_size, 8192);
        assert_eq!(cfg.out_hidden_size, 2048);
        // Untouched fields fall back to defaults.
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.spatial_merge_size, 2);
    }

    #[test]
    fn token_id_constants_match_plan() {
        // Sanity-checking the three image-framing token ids against
        // the values recorded in §2.5 of the bring-up plan. Mismatch
        // here means either the constants drifted or
        // tokenizer_config.json on disk no longer matches the plan.
        assert_eq!(IMGPAD_ID, 151665);
        assert_eq!(IMG_START_ID, 151666);
        assert_eq!(IMG_END_ID, 151667);
    }
}
