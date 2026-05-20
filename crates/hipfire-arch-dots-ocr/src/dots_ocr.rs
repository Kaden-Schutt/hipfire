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
use rdna_compute::{Gpu, GpuTensor};

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
    pub qkv_w: GpuTensor,
    pub proj_w: GpuTensor,
    pub norm2_w: GpuTensor,
    /// `[2 * intermediate_size, embed_dim]` — load-time concat of
    /// `fc1` and `fc3`. `silu(y[:H]) * y[H:]` after this GEMM.
    pub fc13_proj: GpuTensor,
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
    /// Conv2d-style patch projection. Reshape on load from
    /// `[embed_dim, 3, 14, 14]` to `[embed_dim, 588]` for the GEMM.
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
    /// `mlp.0`: linear(merge_dim → merge_dim). Bias on disk.
    pub merger_fc1_w: GpuTensor,
    pub merger_fc1_b: GpuTensor,
    /// `mlp.2`: linear(merge_dim → out_hidden_size). Bias on disk.
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
            for t in [b.norm1_w, b.qkv_w, b.proj_w, b.norm2_w, b.fc13_proj, b.fc2] {
                let _ = gpu.free_tensor(t);
            }
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

/// Vision weight loader — phase 2c stub. Returns a "not yet
/// implemented" error so attempts to load a dots.ocr HFQ surface a
/// clear message at bring-up rather than silently emitting garbage.
///
/// Real loader will dequant HFQ4G256 / read F16 / read F32 per the
/// qwen35-vl pattern, reshape the 4-D `patch_embed.proj.weight` from
/// `[embed_dim, 3, 14, 14]` to `[embed_dim, 588]`, and concat fc1+fc3
/// at load time into `fc13_proj`.
pub fn load_vision_weights(
    _hfq: &HfqFile,
    _cfg: &DotsVisionConfig,
    _gpu: &mut Gpu,
) -> HipResult<DotsVisionWeights> {
    panic!(
        "dots-ocr: vision weight loader is a phase 2c stub. \
         Text-side load is wired; vision tower lands in the next commit. \
         Track in docs/plans/qwen_2.0_vlm_plus_dots_ocr.md phase 2c."
    );
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
