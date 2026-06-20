// SPDX-License-Identifier: Apache-2.0
// hipfire — SigLIP vision encoder forward. See LICENSE / NOTICE.

//! `vision_forward`: SigLIP ViT over a `[num_patches, 3·patch²]` patch tensor →
//! `[num_patches, hidden]` features. Mirrors `hipfire-arch-qwen35-vl`'s
//! `vision_forward` minus the 2D-RoPE and spatial merger: SigLIP uses a learned
//! position embedding (a plain add) and bidirectional attention. Reuses
//! `gemm_f32_batched`/`transpose_f32`/`bias_add_f32` (linear), `layernorm_batched`
//! (LayerNorm+bias), `vit_attention_opt` (bidirectional), `gelu_tanh_f32`, and
//! `add_inplace_f32` (residual). F32 throughout (vision weights ship F32).
//!
//! Output `[num_patches=4096, hidden=1152]` feeds the multimodal projector
//! (avg-pool → `mm_soft_emb_norm` → `mm_input_projection`), the next phase.

use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::SigLipConfig;
use crate::vision::SigLipWeights;

/// Batched linear `Y[n, out] = X[n, in] · W[out, in]ᵀ + bias`, F32 — the SigLIP
/// analogue of qwen35-vl's `linear_f16` (gemm → transpose → bias-add).
///
/// NOTE: kept on `gemm_f32_batched`+`transpose_f32`. A swap to
/// `gemm_f32_register_tiled` (which would also drop the transpose) corrupted the
/// vision embeddings — the model then reported "cannot process images" — so the
/// register-tiled path is layout-incompatible for these shapes and was reverted.
/// The attention kernel, not this GEMM, was the encode bottleneck.
fn linear_f32(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    // Y_t[out, n] = W[out, in] @ X[n, in]ᵀ
    let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
    gpu.gemm_f32_batched(w, x, &yt, out_dim, in_dim, n)?;
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    gpu.transpose_f32(&yt, &y, out_dim, n)?;
    gpu.free_tensor(yt)?;
    gpu.bias_add_f32(&y, bias, n, out_dim)?;
    Ok(y)
}

/// Run the SigLIP encoder. `patches` is row-major `[num_patches, 3·patch²]`
/// (im2col of the 896×896 image at 14×14 stride-14, channel-major within each
/// patch — matching the flattened Conv2d weight layout). Returns the GPU tensor
/// `[num_patches, hidden]` of post-`post_layernorm` features (caller frees).
pub fn vision_forward(
    gpu: &mut Gpu,
    weights: &SigLipWeights,
    cfg: &SigLipConfig,
    patches: &[f32],
) -> HipResult<GpuTensor> {
    let h = cfg.hidden_size;
    let n = cfg.num_patches();
    let inter = cfg.intermediate_size;
    let num_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    let patch_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size;
    let eps = cfg.layer_norm_eps;

    assert_eq!(
        patches.len(),
        n * patch_dim,
        "gemma3-vl: vision_forward expects {n}×{patch_dim} patch values, got {}",
        patches.len()
    );

    // Patch embedding: linear(patch_embed_w [h, patch_dim]) + bias → [n, h].
    let x_patches = gpu.upload_f32(patches, &[n * patch_dim])?;
    let x = linear_f32(
        gpu,
        &weights.patch_embed_w,
        &x_patches,
        &weights.patch_embed_b,
        h,
        patch_dim,
        n,
    )?;
    gpu.free_tensor(x_patches)?;
    // + learned position embedding (fixed grid, direct add — no interpolation).
    gpu.add_inplace_f32(&x, &weights.pos_embed)?;

    for lw in &weights.layers {
        // ── self-attention block (LN1 → attn → residual) ──
        let tmp = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.layernorm_batched(&x, &lw.ln1_w, &lw.ln1_b, &tmp, n, h, eps)?;
        let qkv = linear_f32(gpu, &lw.qkv_w, &tmp, &lw.qkv_b, 3 * h, h, n)?;
        gpu.free_tensor(tmp)?;
        let attn_out = gpu.alloc_tensor(&[n * h], DType::F32)?;
        // Bidirectional ViT attention (no causal mask), scale 1/√head_dim.
        // vit_attention_opt: tiled K/V via shared memory, 4 queries/block —
        // ~3× the naive vit_attention_f32 (the encode hot spot: ≈110s/image of
        // the naive path on gfx1151), same math + signature.
        gpu.vit_attention_opt(&qkv, &attn_out, n, h, num_heads, head_dim)?;
        gpu.free_tensor(qkv)?;
        let proj = linear_f32(gpu, &lw.out_w, &attn_out, &lw.out_b, h, h, n)?;
        gpu.free_tensor(attn_out)?;
        gpu.add_inplace_f32(&x, &proj)?;
        gpu.free_tensor(proj)?;

        // ── MLP block (LN2 → fc1 → gelu-tanh → fc2 → residual) ──
        let tmp2 = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.layernorm_batched(&x, &lw.ln2_w, &lw.ln2_b, &tmp2, n, h, eps)?;
        let fc1 = linear_f32(gpu, &lw.fc1_w, &tmp2, &lw.fc1_b, inter, h, n)?;
        gpu.free_tensor(tmp2)?;
        gpu.gelu_tanh_f32(&fc1, &fc1, n * inter)?;
        let fc2 = linear_f32(gpu, &lw.fc2_w, &fc1, &lw.fc2_b, h, inter, n)?;
        gpu.free_tensor(fc1)?;
        gpu.add_inplace_f32(&x, &fc2)?;
        gpu.free_tensor(fc2)?;
    }

    // Final post_layernorm → [n, h].
    let out = gpu.alloc_tensor(&[n * h], DType::F32)?;
    gpu.layernorm_batched(&x, &weights.post_ln_w, &weights.post_ln_b, &out, n, h, eps)?;
    gpu.free_tensor(x)?;
    Ok(out)
}
