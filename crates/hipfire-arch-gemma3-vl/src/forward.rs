// SPDX-License-Identifier: Apache-2.0
// hipfire — SigLIP vision encoder forward. See LICENSE / NOTICE.

//! `vision_forward`: SigLIP ViT over a `[num_patches, 3·patch²]` patch tensor →
//! `[num_patches, hidden]` features. Mirrors `hipfire-arch-qwen35-vl`'s
//! `vision_forward` minus the 2D-RoPE and spatial merger: SigLIP uses a learned
//! position embedding (a plain add) and bidirectional attention.
//!
//! Mixed precision (encode is bandwidth-bound on unified-memory gfx1151): the
//! per-layer linears run `gemm_bf16_x_bf16_wmma` (bf16 weights, f32 accumulation
//! in the matrix cores) and attention runs the bf16 `flash_attn_bf16` (online
//! softmax, no causal mask); `layernorm_batched`, `gelu_tanh_f32`, `bias_add_f32`,
//! and `add_inplace_f32` stay F32 (negligible cost). The patch-embed linear stays
//! F32 (its `k = 3·patch² = 588` is not a multiple of 16, so no WMMA).
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

/// BF16-weight linear `Y[n, out] = X[n, in] · W[out, in]ᵀ + bias`.
///
/// `gemm_bf16_x_bf16_wmma`: bf16 weight `[out, in]`, f32 activation staged to
/// bf16 once internally, **f32 accumulation in the matrix cores**, f32 output
/// already `[n, out]` (no transpose). On unified-memory gfx1151 this halves
/// weight bandwidth — the dominant cost — and uses the WMMA units f32 GEMM
/// can't. 108 of these run per image, so it's the tower's hot loop.
fn linear_bf16(
    gpu: &mut Gpu,
    w_bf16: &GpuTensor,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    gpu.gemm_bf16_x_bf16_wmma(w_bf16, x, &y, out_dim, in_dim, n)?;
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

    // Optional per-category timing (HIPFIRE_VISION_PROFILE=1): device-sync around
    // each op group and accumulate. acc = [gemm, attn, norm, elem]. The syncs
    // serialize the pipeline, so totals are upper bounds — use for *relative*
    // attribution, not absolute speed.
    let profile = std::env::var("HIPFIRE_VISION_PROFILE").is_ok();
    let mut acc = [0f64; 4];
    macro_rules! timed {
        ($i:expr, $e:expr) => {{
            if profile {
                gpu.hip.device_synchronize()?;
            }
            let __t = std::time::Instant::now();
            let __r = $e?;
            if profile {
                gpu.hip.device_synchronize()?;
                acc[$i] += __t.elapsed().as_secs_f64();
            }
            __r
        }};
    }

    // Patch embedding: linear(patch_embed_w [h, patch_dim]) + bias → [n, h].
    let x_patches = gpu.upload_f32(patches, &[n * patch_dim])?;
    let x = timed!(
        0,
        linear_f32(
            gpu,
            &weights.patch_embed_w,
            &x_patches,
            &weights.patch_embed_b,
            h,
            patch_dim,
            n,
        )
    );
    gpu.free_tensor(x_patches)?;
    // + learned position embedding (fixed grid, direct add — no interpolation).
    timed!(3, gpu.add_inplace_f32(&x, &weights.pos_embed));

    for lw in &weights.layers {
        // ── self-attention block (LN1 → attn → residual) ──
        let tmp = gpu.alloc_tensor(&[n * h], DType::F32)?;
        timed!(
            2,
            gpu.layernorm_batched(&x, &lw.ln1_w, &lw.ln1_b, &tmp, n, h, eps)
        );
        let qkv = timed!(0, linear_bf16(gpu, &lw.qkv_w, &tmp, &lw.qkv_b, 3 * h, h, n));
        gpu.free_tensor(tmp)?;
        // Cast fused qkv → bf16, then bidirectional bf16 flash attention (online
        // softmax, no causal mask). The flash kernel reads the fused qkv's q/k/v
        // offsets directly, so no split is needed.
        let qkv_bf16 = gpu.alloc_tensor(&[n * 3 * h], DType::BF16)?;
        timed!(1, gpu.cast_f32_to_bf16(&qkv, &qkv_bf16));
        gpu.free_tensor(qkv)?;
        let attn_out = gpu.alloc_tensor(&[n * h], DType::F32)?;
        timed!(
            1,
            gpu.flash_attn_bf16(&qkv_bf16, &attn_out, n, h, num_heads, head_dim)
        );
        gpu.free_tensor(qkv_bf16)?;
        let proj = timed!(
            0,
            linear_bf16(gpu, &lw.out_w, &attn_out, &lw.out_b, h, h, n)
        );
        gpu.free_tensor(attn_out)?;
        timed!(3, gpu.add_inplace_f32(&x, &proj));
        gpu.free_tensor(proj)?;

        // ── MLP block (LN2 → fc1 → gelu-tanh → fc2 → residual) ──
        let tmp2 = gpu.alloc_tensor(&[n * h], DType::F32)?;
        timed!(
            2,
            gpu.layernorm_batched(&x, &lw.ln2_w, &lw.ln2_b, &tmp2, n, h, eps)
        );
        let fc1 = timed!(
            0,
            linear_bf16(gpu, &lw.fc1_w, &tmp2, &lw.fc1_b, inter, h, n)
        );
        gpu.free_tensor(tmp2)?;
        timed!(3, gpu.gelu_tanh_f32(&fc1, &fc1, n * inter));
        let fc2 = timed!(0, linear_bf16(gpu, &lw.fc2_w, &fc1, &lw.fc2_b, h, inter, n));
        gpu.free_tensor(fc1)?;
        timed!(3, gpu.add_inplace_f32(&x, &fc2));
        gpu.free_tensor(fc2)?;
    }

    if profile {
        eprintln!(
            "[vision-profile] gemm={:.2}s attn={:.2}s norm={:.2}s elem={:.2}s (sum={:.2}s)",
            acc[0],
            acc[1],
            acc[2],
            acc[3],
            acc.iter().sum::<f64>()
        );
    }

    // Final post_layernorm → [n, h].
    let out = gpu.alloc_tensor(&[n * h], DType::F32)?;
    gpu.layernorm_batched(&x, &weights.post_ln_w, &weights.post_ln_b, &out, n, h, eps)?;
    gpu.free_tensor(x)?;
    Ok(out)
}
