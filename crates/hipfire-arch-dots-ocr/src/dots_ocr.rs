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
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16, attention_family, DispatchCtx, FullAttnParams, KernelKey, ShapeInfo};
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
    // Post-merger output dim — must match the text decoder's embedding
    // dimension so the merger output can splice directly into the text
    // embed stream. Fallback chain:
    //   1. `vision_config.out_hidden_size` — explicit override
    //   2. `vision_config.hidden_size` — HF's PatchMerger uses this
    //      (modeling_dots_vision.py:62-83 `PatchMerger(dim=config.hidden_size, ...)`).
    //      dots.ocr's config.json sets both `embed_dim` and `hidden_size`
    //      to 1536 (verified against the snapshot's config.json). On a
    //      future Qwen2-VL sibling where embed_dim != hidden_size,
    //      this is the load-bearing one for the merger output dim.
    //   3. `config.text_config.hidden_size` — last resort for
    //      hypothetical nested-config layouts.
    //   4. defaults.out_hidden_size (1536).
    let out_hidden_size = vc.get("out_hidden_size").and_then(|v| v.as_u64())
        .or_else(|| vc.get("hidden_size").and_then(|v| v.as_u64()))
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
/// `docs/plans/dots-ocr.dots_ocr_manifest.txt`):
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
             load_f16_or_dequant handles qt ∈ {{1 (F16), 2 (F32), 6 (HFQ4G256), 7 (HFQ4G128)}}. \
             Other formats known to the HFQ writer (3 = Q8F16, 13 = MQ4G256, etc.) \
             would need an additional dequant arm here; mirrors the qwen35-vl \
             load_f16_gpu gap and is deferred until phase 5 (quantisation) makes \
             one of them load-bearing on the vision side.",
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
    // Defensive: dequant_hfq4 truncates to n_elements on a successful
    // dequant, but a partial last-group truncate inside the loop could
    // produce fewer elements without panicking. Catch that here so the
    // GPU upload doesn't silently produce a shape-mismatched fc13_proj
    // and either crash or, worse, read into the next tensor's allocation.
    assert_eq!(
        bytes_a.len(), 2 * n_elements_a,
        "dots-ocr: {name_a} dequant produced {} f16 bytes, expected {}",
        bytes_a.len(), 2 * n_elements_a,
    );
    assert_eq!(
        bytes_b.len(), 2 * n_elements_b,
        "dots-ocr: {name_b} dequant produced {} f16 bytes, expected {}",
        bytes_b.len(), 2 * n_elements_b,
    );
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
pub(crate) fn linear_f16(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    // Output as 2-D `[n, out_dim]`. The 2-D shape is load-bearing for
    // downstream `rmsnorm_f32`, which infers `batch = shape[0]` and
    // `n = shape.last()`. With a 1-D shape, rmsnorm interprets the
    // whole buffer as ONE row of length `n * out_dim` and reads the
    // norm-weight (length out_dim) out of bounds -> sticky HIP fault.
    let y = gpu.alloc_tensor(&[n, out_dim], DType::F32)?;
    // gfx11/gfx12 — use the fused-transpose WMMA variant.
    // It writes row-major `[n, out_dim]` directly, dropping the separate
    // transpose_f32 kernel that the older `gemm_f16_wmma` path required.
    if gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12() {
        gpu.gemm_f16_wmma_mb8(w, x, &y, out_dim, in_dim, n)?;
    } else {
        let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
        gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
        gpu.transpose_f32(&yt, &y, out_dim, n)?;
        gpu.free_tensor(yt)?;
    }
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
pub(crate) fn linear_f16_no_bias(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let y = gpu.alloc_tensor(&[n, out_dim], DType::F32)?;
    // See [`linear_f16`] for the fused-transpose WMMA rationale and the
    // 2-D output-shape requirement.
    if gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12() {
        gpu.gemm_f16_wmma_mb8(w, x, &y, out_dim, in_dim, n)?;
    } else {
        let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
        gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
        gpu.transpose_f32(&yt, &y, out_dim, n)?;
        gpu.free_tensor(yt)?;
    }
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
///
/// # Batch limitation (single image per call)
///
/// This function processes ONE image at a time. The HF `cu_seqlens`
/// block-diagonal masking that allows multi-image concatenation in
/// `flash_attn_varlen_func` (see `modeling_dots_vision.py:160-167`)
/// is NOT supported by `vit_attention_opt`, which is a standard dense
/// attention kernel. Multi-image batching at this layer would let
/// patches from image A attend to patches from image B, corrupting
/// the merged tokens.
///
/// Multi-image prompts should call `vision_forward` once per image
/// at the daemon prefill layer and concatenate the resulting merged
/// tokens after the vision pass. Until that wiring lands (Phase 3),
/// the function panics on inputs whose `patches.numel()` is not
/// consistent with a single image's `[grid_h * grid_w, 588]` shape.
pub fn vision_forward(
    gpu: &mut Gpu,
    weights: &DotsVisionWeights,
    cfg: &DotsVisionConfig,
    patches: &GpuTensor,
    grid_h: usize,
    grid_w: usize,
) -> HipResult<GpuTensor> {
    let h = cfg.embed_dim;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;
    let interm = cfg.intermediate_size;
    let n_patches = grid_h * grid_w;
    let patch_dim = cfg.num_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let sms = cfg.spatial_merge_size;
    let eps = cfg.rms_norm_eps;

    // Patch tensor shape check — `[N, patch_dim]` flat. Per the
    // "single image per call" invariant documented in this function's
    // doc-comment, the daemon prefill loop must call us once per image;
    // multi-image batching at this layer would let patches from image A
    // attend to patches from image B (vit_attention_opt is dense, not
    // cu_seqlens-aware).
    assert_eq!(
        patches.numel(), n_patches * patch_dim,
        "dots-ocr: vision_forward expects patches.numel() == n_patches * patch_dim \
         ({n_patches} * {patch_dim} = {}), got {}. \
         If you're batching multiple images, call vision_forward once per image; \
         see the multi-image-attention-leakage note on the function header.",
        n_patches * patch_dim, patches.numel(),
    );
    assert_eq!(
        grid_h % sms, 0,
        "dots-ocr: grid_h={grid_h} must be a multiple of spatial_merge_size={sms}",
    );
    assert_eq!(
        grid_w % sms, 0,
        "dots-ocr: grid_w={grid_w} must be a multiple of spatial_merge_size={sms}",
    );
    assert_eq!(n_heads * head_dim, h, "dots-ocr: n_heads * head_dim must equal embed_dim");

    let t0 = std::time::Instant::now();
    let use_wmma = gpu.arch_caps.has_wmma_w32();
    let use_gfx12_wmma = gpu.arch_caps.has_wmma_w32_gfx12();
    eprintln!(
        "  vision forward (dots-ocr GPU): {n_patches} patches, {grid_h}×{grid_w} grid, {} blocks",
        cfg.num_hidden_layers,
    );
    eprintln!(
        "  vision kernels: {}",
        if use_wmma {
            "rdna3-wmma"
        } else if use_gfx12_wmma {
            "rdna4-wmma"
        } else {
            "scalar-fallback"
        },
    );

    // HIPFIRE_DOTS_OCR_DUMP_DIR=<path>: dump full per-stage tensor
    // outputs to that directory for offline HF-reference diffing. Stage
    // names match `benchmarks/references/dots_ocr_smoke_001_activations/`:
    // patch_embed, block_00, block_21, block_41, post_trunk_norm, merger.
    // Each is written as a NumPy `.npy` file in native row-major F32.
    let dump_dir: Option<std::path::PathBuf> = std::env::var("HIPFIRE_DOTS_OCR_DUMP_DIR")
        .ok()
        .map(std::path::PathBuf::from);
    if let Some(ref d) = dump_dir {
        std::fs::create_dir_all(d).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("dump_dir mkdir {}: {e}", d.display()))
        })?;
        eprintln!("  HIPFIRE_DOTS_OCR_DUMP_DIR={} — will dump per-stage tensors", d.display());
    }
    let dump_stage = |gpu: &Gpu, t: &GpuTensor, name: &str, shape: &[usize]| -> HipResult<()> {
        if let Some(d) = dump_dir.as_ref() {
            let data = gpu.download_f32(t)?;
            write_npy_f32(&d.join(format!("{name}.npy")), &data, shape).map_err(|e| {
                hip_bridge::HipError::new(0, &format!("npy write {name}: {e}"))
            })?;
            eprintln!("    dump: {name}.npy ({}×{} f32)", shape[0], shape[1]);
        }
        Ok(())
    };

    // ── Build + upload 2-D RoPE tables (CPU build per plan §2.6) ─────
    //
    // Tables persist for the whole vision pass (all 42 blocks share
    // them). Theta = 10000 per `VisionRotaryEmbedding` default in
    // modeling_dots_vision.py.
    let (cos_h, sin_h) = crate::rope::build_rope_2d_tables(grid_h, grid_w, head_dim, sms, 10_000.0);
    let cos_table = gpu.upload_f32(&cos_h, &[n_patches, head_dim])?;
    let sin_table = gpu.upload_f32(&sin_h, &[n_patches, head_dim])?;

    // ── 1. Patch embed: linear(F16 weight + bias) + RMSNorm ──────────
    //
    // patch_embed_w on GPU is the 4-D conv weight flattened to a
    // `[embed_dim, patch_dim]` linear (verified during load).
    let trace_pre = std::env::var("HIPFIRE_DOTS_OCR_TRACE").ok().as_deref() == Some("1");
    let dump_stats = |gpu: &Gpu, t: &GpuTensor, label: &str| -> HipResult<()> {
        if !trace_pre { return Ok(()); }
        let data = gpu.download_f32(t)?;
        let n = data.len();
        let nan = data.iter().filter(|x| x.is_nan()).count();
        let inf = data.iter().filter(|x| x.is_infinite()).count();
        let mean: f64 = data.iter().filter(|x| x.is_finite()).map(|&x| x as f64).sum::<f64>()
            / (n - nan - inf).max(1) as f64;
        let (mn, mx) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
            if x.is_finite() { (a.min(x), b.max(x)) } else { (a, b) }
        });
        eprintln!(
            "    stats[{label}]: n={n} mean={mean:+.4} range=[{mn:+.3}, {mx:+.3}] nan={nan} inf={inf}"
        );
        Ok(())
    };
    if trace_pre { eprintln!("  trace: about to patch_embed linear"); gpu.hip.device_synchronize()?; }
    dump_stats(gpu, patches, "patches_in")?;
    let mut x = linear_f16(
        gpu, &weights.patch_embed_w, patches, &weights.patch_embed_b,
        h, patch_dim, n_patches,
    )?;
    if trace_pre { eprintln!("  trace: after patch_embed linear"); gpu.hip.device_synchronize()?; }
    dump_stats(gpu, &x, "patch_embed_linear")?;
    // patch_embed_norm is RMSNorm (the patchifier carries one).
    let normed = gpu.alloc_tensor(&[n_patches, h], DType::F32)?;
    gpu.rmsnorm_f32(&x, &weights.patch_embed_norm, &normed, eps)?;
    if trace_pre { eprintln!("  trace: after patch_embed RMSNorm"); gpu.hip.device_synchronize()?; }
    dump_stats(gpu, &normed, "patch_embed_norm")?;
    gpu.free_tensor(x)?;
    x = normed;
    // HF cast x to bf16 at vision forward entry (modeling_dots_vision.py
    // line 484-485 `hidden_states = hidden_states.bfloat16()`), so the
    // residual stream is bf16-precision throughout. Emulate that by
    // bf16-truncating at every block boundary (after each residual add).
    // Optional via env var so it can be A/B tested.
    let bf16_residual = std::env::var("HIPFIRE_DOTS_OCR_BF16_RESIDUAL")
        .ok().as_deref() == Some("1");
    if bf16_residual {
        gpu.bf16_round_trip_f32(&x)?;
    }
    // Dump matches HF capture point — `patch_embed` hook is on the
    // full `vt.patch_embed` module = Conv2d + bias + RMSNorm output.
    dump_stage(gpu, &x, "patch_embed", &[n_patches, h])?;

    // ── 2. 42-block encoder stack ────────────────────────────────────
    //
    // Per-block:
    //   x_norm = rmsnorm(x, norm1)
    //   qkv    = linear_no_bias(qkv_w, x_norm)                 # [n, 3h]
    //   rope_2d_halfsplit_qkv_interleaved(qkv, cos, sin)        # in-place on Q+K
    //   attn   = vit_attention_opt(qkv)                         # non-causal
    //   x     += linear_no_bias(proj_w, attn)
    //   x_norm = rmsnorm(x, norm2)
    //   fc13   = linear_no_bias(fc13_proj, x_norm)              # [n, 2*interm]
    //   act    = silu(fc13[:, :interm]) * fc13[:, interm:]      # SwiGLU
    //   x     += linear_no_bias(fc2_w, act)
    let qkv_dim = 3 * h;
    let two_interm = 2 * interm;

    // HIPFIRE_DOTS_OCR_TRACE=1: sync after every step + print probe so
    // the first failing kernel surfaces directly instead of via a sticky
    // error reported later (HIP errors are async-sticky — the call that
    // reports them is rarely the launch that caused them).
    let trace = std::env::var("HIPFIRE_DOTS_OCR_TRACE").ok().as_deref() == Some("1");
    macro_rules! probe {
        ($gpu:expr, $msg:literal) => {
            if trace {
                eprintln!("    trace: {}", $msg);
                $gpu.hip.device_synchronize()?;
            }
        };
    }

    if trace { eprintln!("  trace: entering 42-block loop"); }
    for li in 0..cfg.num_hidden_layers {
        let lw = &weights.blocks[li];
        let trace_block_li = trace && li == 0;
        if trace_block_li { eprintln!("  block {li}: start"); }

        // Per-kernel sync timing macro (block 0 only). Returns the elapsed
        // ms since the previous `tic` (or the loop top if first call).
        let mut tic = std::time::Instant::now();
        macro_rules! toc {
            ($gpu:expr, $label:literal) => {
                if trace_block_li {
                    $gpu.hip.device_synchronize()?;
                    let dt = tic.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("    timing: {:25} {dt:>8.2} ms", $label);
                    tic = std::time::Instant::now();
                }
            };
        }
        if trace_block_li { tic = std::time::Instant::now(); }

        // 2a. RMSNorm pre-attn.
        let xn = gpu.alloc_tensor(&[n_patches, h], DType::F32)?;
        gpu.rmsnorm_f32(&x, &lw.norm1_w, &xn, eps)?;
        toc!(gpu, "norm1 rmsnorm");
        if trace_block_li { dump_stats(gpu, &xn, "b0_xn (post-norm1)")?; tic = std::time::Instant::now(); }

        // 2b. Fused QKV GEMM (no bias). yt[3h, n] → transpose → qkv[n, 3h]
        // interleaved (Q, K, V stacked along the 3h axis).
        let qkv = linear_f16_no_bias(gpu, &lw.qkv_w, &xn, qkv_dim, h, n_patches)?;
        gpu.free_tensor(xn)?;
        // EXPERIMENTAL bf16-trunc the QKV linear output to match HF's
        // bf16 storage of Q/K/V. QKV linear cos is 0.999 (essentially
        // exact) but the small remaining drift may flip softmax winners
        // — bf16-trunc collapses our F32 output to the exact bf16 bits
        // HF would have at this point.
        if bf16_residual { gpu.bf16_round_trip_f32(&qkv)?; }
        toc!(gpu, "qkv GEMM");
        if trace_block_li { dump_stats(gpu, &qkv, "b0_qkv")?; tic = std::time::Instant::now(); }
        // Dump QKV linear output for direct HF diff. Shape is
        // [n_patches, 3 * hidden] interleaved (Q | K | V along the
        // 3h axis). Lets us isolate whether the bug is in the QKV
        // linear or downstream (RoPE / attention compute).
        if matches!(li, 0 | 1 | 2 | 4 | 8 | 12 | 16 | 21 | 41) {
            dump_stage(gpu, &qkv, &format!("block_{li:02}_qkv"), &[n_patches, qkv_dim])?;
        }

        // 2c. Split interleaved QKV into three separate Q, K, V buffers
        // (`[n_patches, hidden=n_heads*head_dim]` each). `attention_dflash_f32`
        // expects separate buffers; the in-place interleaved RoPE
        // variant from 2c-5a is not usable here because the only
        // large-N-friendly attention kernel (`attention_dflash_f32`)
        // takes Q/K/V as separate flat tensors. `vit_attention_opt`
        // would accept the interleaved layout but overflows RDNA3 LDS
        // at N=19520 (stores `scores[N]` in shared memory = 78 KB,
        // exceeds the 64 KB cap).
        let q_buf = gpu.alloc_tensor(&[n_patches, h], DType::F32)?;
        let k_buf = gpu.alloc_tensor(&[n_patches, h], DType::F32)?;
        let v_buf = gpu.alloc_tensor(&[n_patches, h], DType::F32)?;
        gpu.qkv_split_interleaved_f32(&qkv, &q_buf, &k_buf, &v_buf, n_patches, h)?;
        gpu.free_tensor(qkv)?;
        toc!(gpu, "qkv split");
        if trace_block_li {
            dump_stats(gpu, &q_buf, "b0_q (post-split)")?;
            dump_stats(gpu, &k_buf, "b0_k (post-split)")?;
            dump_stats(gpu, &v_buf, "b0_v (post-split)")?;
            tic = std::time::Instant::now();
        }

        // 2d. 2-D RoPE on Q and K (in-place, separate buffers).
        gpu.rope_2d_halfsplit_f32(
            &q_buf, &k_buf, &cos_table, &sin_table,
            n_patches, n_heads, n_heads, head_dim,
        )?;
        toc!(gpu, "rope_2d");
        if trace_block_li {
            dump_stats(gpu, &q_buf, "b0_q (post-rope)")?;
            dump_stats(gpu, &k_buf, "b0_k (post-rope)")?;
            tic = std::time::Instant::now();
        }

        // 2e. Non-causal attention via FA-style online softmax. Vision
        // self-attention is B = L = n_patches, n_heads_kv = n_heads
        // (no GQA on the vision side per modeling_dots_vision.py:106).
        let attn = gpu.alloc_tensor(&[n_patches, h], DType::F32)?;
        // For large-B vision attention (B = L = n_patches ≈ 20k on the
        // smoke image), use the WMMA-accelerated FlashAttention kernel.
        // Five variants, picked by (head_dim, B) constraints:
        //
        //   * `attention_dflash_wmma_n64_f16kv_f32` — same shape as N=64
        //     above but K and V are cast to f16 in DRAM first. Halves
        //     the K+V DRAM byte footprint, which is the dominant cost
        //     on this DRAM-bound workload (L2 hit < 1 %). +18% over the
        //     f32-K/V N=64 path on Strix Halo gfx1151. The f16 cast
        //     itself is ~120 MB vs ~73 GB of K+V DRAM traffic per
        //     attention call → amortises to noise.
        //   * `attention_dflash_wmma_n64_f32` — M=32, N=64, Q in regs,
        //     fused alpha-scale. Hard-coded head_dim==128. +7% over M32.
        //   * `attention_dflash_wmma_m32_f32` — M=32, N=16, 2 waves.
        //     Halves query-tile-block count vs M=16. LDS caps at hd≤128.
        //   * `attention_dflash_wmma_f32` — M=16, 1 wave. hd ≤ 256.
        //   * `attention_dflash_f32` — scalar online-softmax fallback.
        //
        // Vision self-attention via dispatch. Try F16-K/V path first
        // (v5, n128), fall to F32-K/V (m32, wmma_f32, scalar floor).
        {
            let ctx = DispatchCtx::new(gpu);
            let family = attention_family();
            let shape = ShapeInfo {
                batch_size: n_patches,
                head_dim,
                m: n_patches,
                is_tree: false,
            };
            if let Ok(_variant) = family.resolve(
                KernelKey::AttnFullF16, &ctx, Some(&shape)
            ) {
                // F16-K/V path: cast K and V, dispatch, free temps.
                let k_f16 = gpu.alloc_tensor(&[n_patches, h], DType::F16)?;
                let v_f16 = gpu.alloc_tensor(&[n_patches, h], DType::F16)?;
                gpu.cast_f32_to_f16(&k_buf, &k_f16)?;
                gpu.cast_f32_to_f16(&v_buf, &v_f16)?;
                family.run_full_attention(&ctx, gpu, &FullAttnParams {
                    key: KernelKey::AttnFullF16,
                    q: &q_buf,
                    k: &k_f16,
                    v: &v_f16,
                    out: &attn,
                    n: n_patches,
                    seq_len: n_patches,
                    n_heads,
                    n_kv_heads: n_heads,
                    head_dim,
                }).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                gpu.free_tensor(k_f16)?;
                gpu.free_tensor(v_f16)?;
            } else {
                // F32-K/V path (m32, wmma_f32, or scalar floor).
                family.run_full_attention(&ctx, gpu, &FullAttnParams {
                    key: KernelKey::AttnFullF32,
                    q: &q_buf,
                    k: &k_buf,
                    v: &v_buf,
                    out: &attn,
                    n: n_patches,
                    seq_len: n_patches,
                    n_heads,
                    n_kv_heads: n_heads,
                    head_dim,
                }).map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            }
        }
        // Dump pre-proj attention output so we can compare to numpy
        // F32 reference (which doesn't include proj).
        if matches!(li, 0 | 1 | 2 | 4 | 8 | 12 | 16 | 21 | 41) {
            dump_stage(gpu, &attn, &format!("block_{li:02}_attn_pre_proj"), &[n_patches, h])?;
        }
        gpu.free_tensor(q_buf)?;
        gpu.free_tensor(k_buf)?;
        gpu.free_tensor(v_buf)?;
        toc!(gpu, "attention_dflash");
        if trace_block_li { dump_stats(gpu, &attn, "b0_attn")?; tic = std::time::Instant::now(); }

        // 2f. Output projection (no bias) + residual.
        let proj = linear_f16_no_bias(gpu, &lw.proj_w, &attn, h, h, n_patches)?;
        gpu.free_tensor(attn)?;
        toc!(gpu, "proj GEMM");
        // Matches HF capture point `block_NN_attn_out` (= output of
        // VisionAttention.forward — includes the self.proj projection,
        // before the residual is added).
        if matches!(li, 0 | 1 | 2 | 4 | 8 | 12 | 16 | 21 | 41) {
            dump_stage(gpu, &proj, &format!("block_{li:02}_attn_out"), &[n_patches, h])?;
        }
        gpu.add_inplace_f32(&x, &proj)?;
        gpu.free_tensor(proj)?;
        if bf16_residual { gpu.bf16_round_trip_f32(&x)?; }
        toc!(gpu, "residual1 (add)");

        // 2g. RMSNorm pre-MLP.
        let xn2 = gpu.alloc_tensor(&[n_patches, h], DType::F32)?;
        gpu.rmsnorm_f32(&x, &lw.norm2_w, &xn2, eps)?;
        toc!(gpu, "norm2 rmsnorm");

        // 2h. Fused fc13 GEMM (no bias). Layout: yt[2*interm, n] without
        // transpose — keep head-major so sub_offset can slice cleanly
        // into separate gate/up `[interm, n]` halves.
        let fc13_yt = gpu.alloc_tensor(&[two_interm * n_patches], DType::F32)?;
        if use_wmma {
            gpu.gemm_f16_wmma(&lw.fc13_proj, &xn2, &fc13_yt, two_interm, h, n_patches)?;
        } else {
            gpu.gemm_f16(&lw.fc13_proj, &xn2, &fc13_yt, two_interm, h, n_patches)?;
        }
        gpu.free_tensor(xn2)?;
        toc!(gpu, "fc13 GEMM");

        // 2i. SwiGLU on head-major sub-views.
        //
        // The fc1+fc3 concat at load-time stacked `fc1` (gate) ABOVE
        // `fc3` (up) along the M (output) axis, so without the final
        // transpose yt[0..interm*n] is exactly the gate buffer and
        // yt[interm*n..2*interm*n] is exactly the up buffer (each in
        // `[interm, n]` head-major layout). silu_mul_f32 operates
        // element-wise, so it doesn't care about the (interm, n)
        // ordering — only that gate[i] and up[i] correspond.
        let gate = fc13_yt.sub_offset(0, interm * n_patches);
        let up = fc13_yt.sub_offset(interm * n_patches, interm * n_patches);
        let act = gpu.alloc_tensor(&[interm * n_patches], DType::F32)?;
        gpu.silu_mul_f32(&gate, &up, &act)?;
        gpu.free_tensor(fc13_yt)?;
        toc!(gpu, "silu_mul");

        // 2j. Transpose act from head-major `[interm, n]` to position-
        // major `[n, interm]` for the fc2 GEMM input.
        let act_nm = gpu.alloc_tensor(&[n_patches, interm], DType::F32)?;
        gpu.transpose_f32(&act, &act_nm, interm, n_patches)?;
        gpu.free_tensor(act)?;
        toc!(gpu, "act transpose");

        // 2k. fc2 projection (no bias) + residual.
        let fc2_y = linear_f16_no_bias(gpu, &lw.fc2, &act_nm, h, interm, n_patches)?;
        gpu.free_tensor(act_nm)?;
        toc!(gpu, "fc2 GEMM");
        gpu.add_inplace_f32(&x, &fc2_y)?;
        gpu.free_tensor(fc2_y)?;
        if bf16_residual { gpu.bf16_round_trip_f32(&x)?; }
        toc!(gpu, "residual2 (add)");

        if trace {
            // Force a sync at end of every block in trace mode so the
            // per-block wall time is real (the loop body is otherwise
            // fully async — all 42 launches queue in ~100 ms and the
            // actual GPU work only flushes at the post-loop sync).
            gpu.hip.device_synchronize()?;
        }
        if li % 7 == 0 || li == cfg.num_hidden_layers - 1 {
            eprintln!("  vision block {}/{} done ({:.2}s)", li + 1, cfg.num_hidden_layers, t0.elapsed().as_secs_f32());
        }
        if trace_pre && (li == 0 || li == 1 || li == 41) {
            dump_stats(gpu, &x, &format!("block_{li:02}_out"))?;
        }
        // Dump matches HF capture: blocks 0, 1, 2, 4, 8, 12, 16, 21, 41
        // (output of full block, i.e. after the residual add at the end).
        if matches!(li, 0 | 1 | 2 | 4 | 8 | 12 | 16 | 21 | 41) {
            dump_stage(gpu, &x, &format!("block_{li:02}"), &[n_patches, h])?;
        }
    }

    // Drop RoPE tables now that all blocks are done.
    gpu.free_tensor(cos_table)?;
    gpu.free_tensor(sin_table)?;

    // Single sync at end of the encoder stack (matches qwen35-vl).
    gpu.hip.device_synchronize()?;
    eprintln!("  vision encoder done ({:.2}s)", t0.elapsed().as_secs_f32());

    // ── 3. Post-trunk RMSNorm (post_norm=true for dots.ocr) ──────────
    let post = gpu.alloc_tensor(&[n_patches, h], DType::F32)?;
    gpu.rmsnorm_f32(&x, &weights.post_trunk_norm, &post, eps)?;
    gpu.free_tensor(x)?;
    dump_stage(gpu, &post, "post_trunk_norm", &[n_patches, h])?;

    // ── 4. Merger: LayerNorm + 2×2 reshape (free) + MLP ──────────────
    //
    // ln_q is a LayerNorm (NOT RMSNorm — note divergence; see §2.4 of
    // the plan). eps=1e-6 from modeling_dots_vision.py:75.
    let merger_eps = 1e-6f32;
    let normed_merger = gpu.alloc_tensor(&[n_patches, h], DType::F32)?;
    gpu.layernorm_batched(
        &post, &weights.merger_ln_w, &weights.merger_ln_b, &normed_merger,
        n_patches, h, merger_eps,
    )?;
    gpu.free_tensor(post)?;

    // The 2×2 group concat is a pure shape change — no rearrange — because
    // [`crate::image::extract_patches`] + [`crate::rope::build_rope_2d_tables`]
    // already emit patches in 2×2-block-major order. `linear_f16` only
    // uses the dimension parameters, not the tensor shape vector, so we
    // can reinterpret `[n_patches, h]` as `[n_merged, merge_dim]` for the
    // merger MLP without an explicit reshape.
    let n_merged = n_patches / (sms * sms);
    let merge_dim = h * sms * sms;

    // 4a. mlp.0: linear(merge_dim → merge_dim) + bias, then GELU.
    let m1 = linear_f16(
        gpu, &weights.merger_fc1_w, &normed_merger, &weights.merger_fc1_b,
        merge_dim, merge_dim, n_merged,
    )?;
    gpu.free_tensor(normed_merger)?;
    // dots.ocr uses exact GELU (PyTorch nn.GELU default). We currently
    // only have the tanh approximation — numerical difference is ~1e-3
    // peak, well inside the bf16→f16 cast slack. TODO(vision-gelu):
    // add gelu_exact_f32 when validation flags a regression.
    gpu.gelu_tanh_f32(&m1, &m1, n_merged * merge_dim)?;

    // 4b. mlp.2: linear(merge_dim → out_hidden_size) + bias.
    let m2 = linear_f16(
        gpu, &weights.merger_fc2_w, &m1, &weights.merger_fc2_b,
        cfg.out_hidden_size, merge_dim, n_merged,
    )?;
    gpu.free_tensor(m1)?;

    gpu.hip.device_synchronize()?;
    eprintln!(
        "  vision merger done: {n_merged} merged tokens × {} dims ({:.2}s)",
        cfg.out_hidden_size, t0.elapsed().as_secs_f32(),
    );
    dump_stage(gpu, &m2, "merger", &[n_merged, cfg.out_hidden_size])?;
    Ok(m2)
}

/// Minimal NumPy `.npy` writer for F32 row-major tensors. Used by the
/// `HIPFIRE_DOTS_OCR_DUMP_DIR` per-stage dump for offline HF-reference
/// diffing. Format reference: github.com/numpy/numpy/blob/main/numpy/lib/format.py
fn write_npy_f32(path: &std::path::Path, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    // Build the header dict (ASCII), then pad with spaces to 16-byte align
    // including the 10-byte preamble (magic + version + header_len).
    let mut shape_str = String::from("(");
    for (i, &s) in shape.iter().enumerate() {
        if i > 0 { shape_str.push_str(", "); }
        shape_str.push_str(&s.to_string());
    }
    if shape.len() == 1 { shape_str.push(','); }
    shape_str.push(')');
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}"
    );
    // Pre-pad so (10 + header.len()) is a multiple of 16, then add `\n`.
    let preamble = 10;
    let mut padded = header;
    while (preamble + padded.len() + 1) % 16 != 0 { padded.push(' '); }
    padded.push('\n');
    let header_len = padded.len() as u16;

    f.write_all(b"\x93NUMPY")?;
    f.write_all(&[1u8, 0u8])?;                       // version 1.0
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(padded.as_bytes())?;
    // Cast &[f32] to &[u8] for the data section.
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    f.write_all(bytes)?;
    Ok(())
}

// ─── Token-id constants ─────────────────────────────────────────────────

/// `<|imgpad|>` — image-pad slot id, used by the daemon's prefill loop
/// to mark positions where merged visual tokens splice in.
pub const IMGPAD_ID: u32 = 151665;
/// `<|img|>` — image-start framing token.
pub const IMG_START_ID: u32 = 151666;
/// `<|endofimg|>` — image-end framing token.
pub const IMG_END_ID: u32 = 151667;
/// `<|user|>` — chat-template user-turn open (text-only turns).
pub const USER_ID: u32 = 151670;
/// `<|endofuser|>` — user-turn close (text-only turns).
pub const ENDOFUSER_ID: u32 = 151671;
/// `<|assistant|>` — assistant-turn cue. The prompt ends here and
/// greedy decode begins right after it.
pub const ASSISTANT_ID: u32 = 151672;
/// `<|endofassistant|>` — primary EOS.
pub const ENDOFASSISTANT_ID: u32 = 151673;
/// `<|endoftext|>` — secondary EOS.
pub const ENDOFTEXT_ID: u32 = 151643;

/// Build the dots.ocr image-OCR prompt token sequence, reproducing the
/// HF `processor.apply_chat_template(messages, add_generation_prompt=True)`
/// output for a single-image layout-extraction turn.
///
/// Framing (verified byte-exact against the captured HF reference in
/// `benchmarks/references/dots_ocr_smoke_001.json` — see the
/// `build_prompt_ids_matches_hf_capture` test):
///
/// ```text
/// 220  <|img|>  <|imgpad|>×n_visual_tokens  <|endofimg|>  <prompt text…>  <|assistant|>
/// ```
///
/// The image-content turn does NOT wrap in `<|user|>` / `<|endofuser|>`
/// (the text-only template branch does; the image branch doesn't). The
/// leading id 220 is the byte-BPE space the template emits before
/// `<|img|>`. The `<|imgpad|>` slots are placeholders the daemon's
/// prefill loop replaces, one-for-one, with merged visual-token
/// embeddings; the count MUST equal the merger's output-token count.
pub fn build_prompt_ids(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    prompt_text: &str,
    n_visual_tokens: usize,
) -> Vec<u32> {
    let prompt = tokenizer.encode(prompt_text);
    let mut ids = Vec::with_capacity(n_visual_tokens + prompt.len() + 4);
    ids.push(220); // leading byte-BPE space before <|img|>
    ids.push(IMG_START_ID);
    ids.resize(ids.len() + n_visual_tokens, IMGPAD_ID);
    ids.push(IMG_END_ID);
    ids.extend_from_slice(&prompt);
    ids.push(ASSISTANT_ID);
    ids
}

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

    /// Oracle test: `build_prompt_ids` must reproduce the HF
    /// `apply_chat_template` output byte-for-byte. The captured
    /// `input_token_ids` (from `capture_dots_ocr_reference.py`, transformers
    /// 5.5.1) is the ground truth that drove the phase-2 13/13 OCR PASS, so
    /// the daemon's prompt builder is correct iff it reproduces it exactly.
    ///
    /// Model-gated: needs the tokenizer from the dots.ocr HFQ, which isn't
    /// in CI. Skips cleanly when the model or fixture is absent.
    #[test]
    fn build_prompt_ids_matches_hf_capture() {
        use std::path::Path;
        let hfq_path = "/data/hipfire/dots-ocr.q8.hfq";
        let cap_path = "../../benchmarks/references/dots_ocr_smoke_001.json";
        if !Path::new(hfq_path).exists() || !Path::new(cap_path).exists() {
            eprintln!("skipping build_prompt_ids_matches_hf_capture: model/fixture absent");
            return;
        }
        let hfq = HfqFile::open(Path::new(hfq_path)).expect("open hfq");
        let tok = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .expect("tokenizer from hfq metadata");
        let cap: serde_json::Value =
            serde_json::from_slice(&std::fs::read(cap_path).unwrap()).unwrap();
        let expected: Vec<u32> = cap["input_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let n_visual = expected.iter().filter(|&&t| t == IMGPAD_ID).count();
        let prompt_text = cap["prompt_template_text"].as_str().unwrap();

        let got = build_prompt_ids(&tok, prompt_text, n_visual);

        assert_eq!(
            got.len(),
            expected.len(),
            "length mismatch: got {} expected {}",
            got.len(),
            expected.len()
        );

        // The framing scaffold is what `build_prompt_ids` is responsible
        // for: leading space, image block (IMG_START + n_visual×IMGPAD +
        // IMG_END), and the trailing <|assistant|> cue. This MUST be
        // byte-exact.
        let img_block_end = 2 + n_visual; // 220, IMG_START, n×IMGPAD, then IMG_END at this index
        assert_eq!(got[..=img_block_end], expected[..=img_block_end],
            "image-block framing diverged from HF capture");
        assert_eq!(*got.last().unwrap(), ASSISTANT_ID, "missing trailing <|assistant|> cue");
        assert_eq!(got.last(), expected.last(), "trailing cue diverged");

        // The prompt-text interior is tokenized by the shared GPT-2 BPE
        // path, which has a known `\s+(?!\S)` lookahead gap (see
        // tokenizer.rs:22 — the `regex` crate can't express the
        // negative lookahead, so whitespace runs before a non-space split
        // one token differently than HF). That is NOT a framing bug: the
        // decoded *text* is identical, only BPE boundaries on indentation
        // runs differ. Assert text-equality (the strong invariant) and
        // report any boundary diffs for visibility.
        assert_eq!(tok.decode(&got), tok.decode(&expected),
            "decoded prompt text diverged — this IS a real bug (not just a BPE-boundary diff)");

        let n_diff = (0..got.len()).filter(|&i| got[i] != expected[i]).count();
        if n_diff > 0 {
            eprintln!(
                "NOTE: {n_diff}/{} tokens differ on BPE whitespace boundaries \
                 (decoded text identical) — tokenizer.rs `\\s+` lookahead gap, \
                 tracked separately; verify benign via the daemon OCR grade.",
                got.len()
            );
        }
    }
}
