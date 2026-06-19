// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 multimodal projector. See LICENSE / NOTICE.

//! `multi_modal_projector`: maps the SigLIP `[num_patches, vision_hidden]`
//! features to `[mm_tokens_per_image, text_hidden]` image embeddings that splice
//! into the gemma3 text stream.
//!
//! Gemma3 pipeline: reshape the patch grid `64×64`, **avg-pool 4×4** → `16×16`
//! (256 tokens), **`mm_soft_emb_norm`** (gemma RMSNorm, `(1+w)`-baked at ingest),
//! then **`mm_input_projection_weight`** (a `[vision_hidden, text_hidden]`
//! parameter applied as `normed @ W`, no bias). The avg-pool is done host-side
//! for bring-up (18 MB download, trivial); norm + projection run on GPU.

use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::Gemma3VlConfig;

/// GPU-resident projector weights.
pub struct ProjectorWeights {
    /// `mm_soft_emb_norm.weight` `[vision_hidden]` — gemma RMSNorm ((1+w)-baked).
    pub soft_emb_norm: GpuTensor,
    /// `mm_input_projection_weight` transposed to `[text_hidden, vision_hidden]`
    /// so `linear` (X @ Wᵀ) computes the model's `normed @ W`.
    pub input_projection_t: GpuTensor,
}

impl ProjectorWeights {
    pub fn load(hfq: &HfqFile, cfg: &Gemma3VlConfig, gpu: &mut Gpu) -> HipResult<Self> {
        let vh = cfg.vision.hidden_size;
        let th = cfg.text_hidden_size;
        let soft_emb_norm = upload(
            hfq,
            gpu,
            "multi_modal_projector.mm_soft_emb_norm.weight",
            vh,
        )?;
        // Stored [vision_hidden, text_hidden]; transpose to [text_hidden, vision_hidden].
        let w = read_f32(hfq, "multi_modal_projector.mm_input_projection_weight");
        assert_eq!(
            w.len(),
            vh * th,
            "gemma3-vl: mm_input_projection size mismatch"
        );
        let mut wt = vec![0.0f32; th * vh];
        for i in 0..vh {
            for j in 0..th {
                wt[j * vh + i] = w[i * th + j];
            }
        }
        let input_projection_t = gpu.upload_f32(&wt, &[th * vh])?;
        Ok(Self {
            soft_emb_norm,
            input_projection_t,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.soft_emb_norm);
        let _ = gpu.free_tensor(self.input_projection_t);
    }
}

fn read_f32(hfq: &HfqFile, name: &str) -> Vec<f32> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("gemma3-vl: projector tensor not found: {name}"));
    match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        qt => panic!("gemma3-vl: expected F16/F32/BF16 for {name}, got qt={qt}"),
    }
}

fn upload(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let v = read_f32(hfq, name);
    assert_eq!(
        v.len(),
        n,
        "gemma3-vl: {name} has {} elems, expected {n}",
        v.len()
    );
    gpu.upload_f32(&v, &[n])
}

/// Avg-pool the `[grid·grid, vision_hidden]` patch features down to
/// `[pool·pool, vision_hidden]` (Gemma3's `AvgPool2d(k=s=pool_factor)` over the
/// `grid×grid` spatial layout), host-side. `feats` is row-major
/// `[grid², vision_hidden]` (patch index = row·grid + col).
fn avg_pool(feats: &[f32], grid: usize, pool: usize, vh: usize) -> Vec<f32> {
    let factor = grid / pool;
    let inv = 1.0 / (factor * factor) as f32;
    let mut out = vec![0.0f32; pool * pool * vh];
    for pr in 0..pool {
        for pc in 0..pool {
            let o = (pr * pool + pc) * vh;
            for dr in 0..factor {
                for dc in 0..factor {
                    let gr = pr * factor + dr;
                    let gc = pc * factor + dc;
                    let s = (gr * grid + gc) * vh;
                    for d in 0..vh {
                        out[o + d] += feats[s + d];
                    }
                }
            }
            for d in 0..vh {
                out[o + d] *= inv;
            }
        }
    }
    out
}

/// Project the SigLIP encoder output to `mm_tokens_per_image` image embeddings.
/// `vision_out` is the GPU `[grid², vision_hidden]` from `vision_forward`;
/// returns a GPU `[mm_tokens_per_image, text_hidden]` tensor (caller frees).
pub fn project(
    gpu: &mut Gpu,
    proj: &ProjectorWeights,
    cfg: &Gemma3VlConfig,
    vision_out: &GpuTensor,
) -> HipResult<GpuTensor> {
    let vh = cfg.vision.hidden_size;
    let th = cfg.text_hidden_size;
    let grid = cfg.vision.grid_side();
    let pool = cfg.pool_side();
    let n_tok = cfg.mm_tokens_per_image;

    // Avg-pool host-side (download → pool → upload).
    let feats = gpu.download_f32(vision_out)?;
    debug_assert_eq!(feats.len(), grid * grid * vh);
    let pooled = avg_pool(&feats, grid, pool, vh);
    let pooled_gpu = gpu.upload_f32(&pooled, &[n_tok * vh])?;

    // mm_soft_emb_norm (RMSNorm over vision_hidden, per token).
    let normed = gpu.alloc_tensor(&[n_tok * vh], DType::F32)?;
    gpu.rmsnorm_batched(&pooled_gpu, &proj.soft_emb_norm, &normed, n_tok, vh, 1e-6)?;
    gpu.free_tensor(pooled_gpu)?;

    // Projection: [n_tok, vh] @ Wᵀ[th, vh]ᵀ = [n_tok, th]. (gemm → transpose.)
    let yt = gpu.alloc_tensor(&[th * n_tok], DType::F32)?;
    gpu.gemm_f32_batched(&proj.input_projection_t, &normed, &yt, th, vh, n_tok)?;
    gpu.free_tensor(normed)?;
    let out = gpu.alloc_tensor(&[n_tok * th], DType::F32)?;
    gpu.transpose_f32(&yt, &out, th, n_tok)?;
    gpu.free_tensor(yt)?;
    Ok(out)
}
