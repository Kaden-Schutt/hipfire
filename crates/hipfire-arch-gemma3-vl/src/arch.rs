// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3-VL ServingBackend (image-token splice). See LICENSE / NOTICE.

//! `Gemma3VlBackend` — the multimodal (`arch_id = 13`) serving backend.
//!
//! Unlike the dense-AR text families (qwen2, gemma3 text) that delegate wholesale
//! to [`run_simple_ar`], the vision path has a **bespoke prefill**: the SigLIP
//! tower encodes the image, the projector maps it to 256 text-hidden rows, and
//! those rows are spliced into the prompt at the `<image_soft_token>` placeholder
//! positions via `forward_step_with_embed`. So this backend overrides
//! [`ServingBackend::serve`] for the splice, then hands off to the **shared**
//! [`decode_loop`] for the identical streaming/stop greedy decode the text path
//! uses. Its [`SimpleAr`] impl covers the text-only fallback (no image bytes) and
//! the per-step decode the loop drives.

use hipfire_arch_gemma3::{forward_step, forward_step_with_embed, Gemma3Config, Gemma3State};
use hipfire_runtime::arch::{
    decode_loop, ArchCaps, GenerateCtx, ServeOutcome, ServingBackend, SimpleAr,
};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};

use crate::config::Gemma3VlConfig;
use crate::loader::Gemma3VlWeights;
use crate::{preprocess_image_bytes, project, vision_forward};

/// Owns the typed configs + multimodal weights + decode state behind the
/// object-safe serving surface. Built by the daemon from [`crate::load_vl`].
pub struct Gemma3VlBackend {
    pub text_cfg: Gemma3Config,
    pub vl_cfg: Gemma3VlConfig,
    pub weights: Gemma3VlWeights,
    pub state: Gemma3State,
}

impl Gemma3VlBackend {
    pub fn new(
        text_cfg: Gemma3Config,
        vl_cfg: Gemma3VlConfig,
        weights: Gemma3VlWeights,
        state: Gemma3State,
    ) -> Self {
        Self {
            text_cfg,
            vl_cfg,
            weights,
            state,
        }
    }

    /// Encode one image's bytes through SigLIP + projector into `mm_tokens × th`
    /// text-hidden rows on the host (row-major, ready to splice at the image
    /// placeholders). GPU scratch is freed before returning.
    ///
    /// `pub` so a caller (the daemon's vision-embedding cache) can encode a
    /// single frame on a cache miss and splice the cached rows back via
    /// [`Gemma3VlBackend::serve_with_embeds`].
    pub fn encode_image(&self, gpu: &mut Gpu, bytes: &[u8]) -> Result<Vec<f32>, String> {
        let patches = preprocess_image_bytes(bytes, &self.vl_cfg.vision)?;
        let vis = vision_forward(gpu, &self.weights.vision, &self.vl_cfg.vision, &patches)
            .map_err(|e| format!("gemma3-vl: vision_forward: {e:?}"))?;
        let img_embeds_gpu = project(gpu, &self.weights.projector, &self.vl_cfg, &vis)
            .map_err(|e| format!("gemma3-vl: project: {e:?}"))?;
        gpu.free_tensor(vis)
            .map_err(|e| format!("gemma3-vl: free vis: {e:?}"))?;
        let img_embeds = gpu
            .download_f32(&img_embeds_gpu)
            .map_err(|e| format!("gemma3-vl: download img_embeds: {e:?}"))?;
        gpu.free_tensor(img_embeds_gpu)
            .map_err(|e| format!("gemma3-vl: free img_embeds: {e:?}"))?;
        Ok(img_embeds)
    }

    /// Splice pre-encoded image rows into the prompt and decode. The second half
    /// of [`ServingBackend::serve`], split out so a caller can supply `img_embeds`
    /// from a cache (skipping SigLIP+projector on a hit) instead of re-encoding.
    ///
    /// `img_embeds` is `n_images * mm_tokens_per_image * text_hidden` row-major
    /// f32, in image order — exactly what `encode_image` concatenated per image
    /// produces. Prefill feeds text tokens via `forward_step` and each image
    /// placeholder the next projected row via `forward_step_with_embed` (no embed
    /// scaling — projector output is already in the decoder's input space), then
    /// the shared [`decode_loop`] streams the output.
    pub fn serve_with_embeds(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
        img_embeds: &[f32],
        n_images: usize,
    ) -> Result<ServeOutcome, String> {
        let eos = tok
            .special_token_id("<end_of_turn>")
            .unwrap_or(self.text_cfg.eos_token_id);
        let ids = tok.encode(ctx.prompt);
        if ids.is_empty() {
            return Err("gemma3-vl serve: empty prompt after tokenize".to_string());
        }
        let th = self.vl_cfg.text_hidden_size;
        let expected = n_images * self.vl_cfg.mm_tokens_per_image * th;
        if img_embeds.len() != expected {
            return Err(format!(
                "gemma3-vl serve_with_embeds: img_embeds len {} != n_images {} * mm {} * th {} = {}",
                img_embeds.len(),
                n_images,
                self.vl_cfg.mm_tokens_per_image,
                th,
                expected,
            ));
        }
        let stream = splice_image_tokens(&self.vl_cfg, &ids, n_images);

        let mut img_row = 0usize;
        for &id in &stream {
            if id == self.vl_cfg.image_token_index {
                let row = &img_embeds[img_row * th..(img_row + 1) * th];
                forward_step_with_embed(
                    gpu,
                    &self.weights.text,
                    &self.text_cfg,
                    &mut self.state,
                    row,
                )
                .map_err(|e| format!("gemma3-vl prefill embed splice: {e:?}"))?;
                img_row += 1;
            } else {
                forward_step(gpu, &self.weights.text, &self.text_cfg, &mut self.state, id)
                    .map_err(|e| format!("gemma3-vl prefill forward_step: {e:?}"))?;
            }
        }

        decode_loop(gpu, self, tok, eos, ctx, stream.len(), stream.len())
    }
}

impl SimpleAr for Gemma3VlBackend {
    /// Text-only prefill (no image): one token at a time, like the gemma3 text
    /// path. The image splice lives in [`ServingBackend::serve`], not here.
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        for &t in tokens {
            forward_step(gpu, &self.weights.text, &self.text_cfg, &mut self.state, t)
                .map_err(|e| format!("gemma3-vl prefill forward_step: {e:?}"))?;
        }
        Ok(())
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        debug_assert_eq!(
            pos, self.state.next_pos,
            "gemma3-vl decode pos {pos} drifted from internal next_pos {}",
            self.state.next_pos
        );
        let _ = pos;
        forward_step(
            gpu,
            &self.weights.text,
            &self.text_cfg,
            &mut self.state,
            token,
        )
        .map_err(|e| format!("gemma3-vl decode forward_step: {e:?}"))
    }

    fn logits(&self) -> &GpuTensor {
        &self.state.logits
    }

    fn vocab_size(&self) -> usize {
        self.text_cfg.vocab_size
    }
}

/// Expand each begin-of-image (`boi`) placeholder in a tokenized prompt into the
/// full image block `[boi, image_soft_token × mm, eoi]` — mirroring the HF
/// Gemma3 processor, which replaces each `<start_of_image>` with the soft tokens
/// + `<end_of_image>`. Multi-image: every `boi` in the prompt is expanded, so a
/// prompt with `n_images` markers yields `n_images` blocks (`n_images × mm`
/// placeholders) consuming the projected rows in order.
///
/// If the framed prompt carries fewer markers than `n_images` (e.g. a caller
/// that didn't template the markers), the missing blocks are prepended after any
/// leading `<bos>`, so an image request is never silently dropped.
fn splice_image_tokens(vl: &Gemma3VlConfig, ids: &[u32], n_images: usize) -> Vec<u32> {
    let mm = vl.mm_tokens_per_image;
    let push_block = |out: &mut Vec<u32>| {
        out.push(vl.boi_token_index);
        out.extend(std::iter::repeat(vl.image_token_index).take(mm));
        out.push(vl.eoi_token_index);
    };

    let mut out = Vec::with_capacity(ids.len() + n_images * (mm + 2));
    let mut expanded = 0usize;
    for &id in ids {
        if id == vl.boi_token_index {
            push_block(&mut out);
            expanded += 1;
        } else {
            out.push(id);
        }
    }

    // Fewer markers than images: prepend the deficit after a leading <bos>=0.
    if expanded < n_images {
        let insert_at = usize::from(ids.first() == Some(&0));
        let mut blocks = Vec::with_capacity((n_images - expanded) * (mm + 2));
        for _ in 0..(n_images - expanded) {
            push_block(&mut blocks);
        }
        out.splice(insert_at..insert_at, blocks);
    }
    out
}

impl ServingBackend for Gemma3VlBackend {
    fn arch_id(&self) -> u32 {
        13
    }

    fn caps(&self) -> ArchCaps {
        ArchCaps {
            vision: true,
            ..ArchCaps::default()
        }
    }

    fn eos_token(&self) -> u32 {
        self.text_cfg.eos_token_id
    }

    /// Multimodal serve: encode each image (if any) through SigLIP + projector,
    /// splice the projected rows into the prompt at the image placeholders during
    /// prefill, then run the shared [`decode_loop`]. With no images this is the
    /// plain gemma3 text path (tokenize → prefill → decode_loop).
    ///
    /// Multi-image: `ctx.images` carries one entry per image (a video's frames
    /// arrive as a stack of images). Each is encoded to `mm` rows; the rows are
    /// concatenated in image order and consumed left-to-right at the prompt's
    /// `image_soft_token` placeholders.
    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        let eos = tok
            .special_token_id("<end_of_turn>")
            .unwrap_or(self.text_cfg.eos_token_id);
        let ids = tok.encode(ctx.prompt);
        if ids.is_empty() {
            return Err("gemma3-vl serve: empty prompt after tokenize".to_string());
        }

        // Text-only request: no splice, plain dense-AR prefill + shared loop.
        if ctx.images.is_empty() {
            self.prefill(gpu, &ids)?;
            return decode_loop(gpu, self, tok, eos, ctx, ids.len(), ids.len());
        }

        // Vision: each image → SigLIP → projector → mm text-hidden rows (host),
        // concatenated in image order, then spliced + decoded.
        let th = self.vl_cfg.text_hidden_size;
        let n_images = ctx.images.len();
        let mut img_embeds: Vec<f32> =
            Vec::with_capacity(n_images * self.vl_cfg.mm_tokens_per_image * th);
        for bytes in ctx.images {
            img_embeds.extend(self.encode_image(gpu, bytes)?);
        }
        self.serve_with_embeds(gpu, tok, ctx, &img_embeds, n_images)
    }

    fn reset_session(&mut self, _gpu: &mut Gpu, _session_id: &str) -> Result<(), String> {
        self.state.reset();
        Ok(())
    }

    fn unload(self: Box<Self>, gpu: &mut Gpu) {
        let b = *self;
        b.weights.free_gpu(gpu);
        b.state.free_gpu(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SigLipConfig;

    fn backend_cfg() -> Gemma3VlConfig {
        Gemma3VlConfig {
            vision: SigLipConfig {
                hidden_size: 1152,
                num_hidden_layers: 27,
                num_attention_heads: 16,
                intermediate_size: 4304,
                patch_size: 14,
                image_size: 896,
                num_channels: 3,
                layer_norm_eps: 1e-6,
            },
            mm_tokens_per_image: 4, // tiny for the test
            image_token_index: 262144,
            boi_token_index: 255999,
            eoi_token_index: 256000,
            text_hidden_size: 2560,
            gemma_norm_offset: 1.0,
        }
    }

    #[test]
    fn splice_expands_marker_in_place() {
        let vl = backend_cfg();
        // user … <boi> … prompt  →  user … [boi, img×4, eoi] … prompt
        let ids = vec![10, 11, vl.boi_token_index, 12, 13];
        let out = splice_image_tokens(&vl, &ids, 1);
        assert_eq!(
            out,
            vec![10, 11, 255999, 262144, 262144, 262144, 262144, 256000, 12, 13]
        );
        // 4 image placeholders present for the prefill embed-splice.
        assert_eq!(out.iter().filter(|&&t| t == 262144).count(), 4);
    }

    #[test]
    fn splice_prepends_after_bos_when_no_marker() {
        let vl = backend_cfg();
        let ids = vec![0u32, 10, 11]; // leading <bos>=0
        let out = splice_image_tokens(&vl, &ids, 1);
        assert_eq!(
            out,
            vec![0, 255999, 262144, 262144, 262144, 262144, 256000, 10, 11]
        );
    }

    #[test]
    fn splice_expands_each_marker_for_multi_image() {
        let vl = backend_cfg();
        // Two <boi> markers → two image blocks, 8 placeholders total.
        let ids = vec![0u32, vl.boi_token_index, vl.boi_token_index, 12];
        let out = splice_image_tokens(&vl, &ids, 2);
        assert_eq!(out.iter().filter(|&&t| t == 262144).count(), 8);
        assert_eq!(out.iter().filter(|&&t| t == vl.boi_token_index).count(), 2);
        assert_eq!(out.iter().filter(|&&t| t == vl.eoi_token_index).count(), 2);
    }

    #[test]
    fn splice_prepends_deficit_blocks_when_markers_missing() {
        let vl = backend_cfg();
        // 3 images requested, prompt has no markers → 3 blocks prepended after <bos>.
        let ids = vec![0u32, 10, 11];
        let out = splice_image_tokens(&vl, &ids, 3);
        assert_eq!(out.iter().filter(|&&t| t == vl.boi_token_index).count(), 3);
        assert_eq!(out.iter().filter(|&&t| t == 262144).count(), 12);
        assert_eq!(out[0], 0, "bos stays first");
        assert_eq!(out[1], vl.boi_token_index, "blocks after bos");
    }
}
