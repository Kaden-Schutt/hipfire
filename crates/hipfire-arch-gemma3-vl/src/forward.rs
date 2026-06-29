// SPDX-License-Identifier: Apache-2.0
// hipfire — SigLIP vision encoder forward. See LICENSE / NOTICE.

//! `vision_forward`: SigLIP ViT over a `[num_patches, 3·patch²]` patch tensor →
//! `[num_patches, hidden]` features. Mirrors `hipfire-arch-qwen35-vl`'s
//! `vision_forward` minus the 2D-RoPE and spatial merger: SigLIP uses a learned
//! position embedding (a plain add) and bidirectional attention.
//!
//! Mixed precision (encode is bandwidth-bound on unified-memory gfx1151): the
//! per-layer linears run `gemm_bf16_x_bf16_wmma` (bf16 weights, f32 accumulation
//! in the matrix cores); attention runs the f16-KV matrix-core flash
//! (`attention_dflash_wmma_m64_n128_f16kv_v3_f32`, head_dim padded 72→128) on
//! WMMA archs, falling back to the generic bf16 `flash_attn_bf16` otherwise.
//! `layernorm_batched`, `gelu_tanh_f32`, `bias_add_f32`, and `add_inplace_f32`
//! stay F32 (negligible cost). The patch-embed linear stays F32 (its
//! `k = 3·patch² = 588` is not a multiple of 16, so no WMMA).
//!
//! Output `[num_patches=4096, hidden=1152]` feeds the multimodal projector
//! (avg-pool → `mm_soft_emb_norm` → `mm_input_projection`), the next phase.
//!
//! ## Scratch lifetime (RAII via `OwnedTensor`)
//!
//! Every per-call scratch tensor is an [`OwnedTensor`]: dropping it (on success
//! OR on any `?`-propagated error) enqueues its pooled buffer into the `Gpu`'s
//! deferred-free mailbox. There are therefore NO manual `free_tensor` calls and
//! NO error-path bookkeeping: plain `?` is leak-free. `gpu.reclaim_pending()` at
//! the bottom of EACH layer-loop iteration returns that layer's scratch to the
//! pool for reuse in the next layer (peak VRAM stays flat across all layers); a
//! final `reclaim_pending()` returns the residual stream + patch-embed scratch.
//! The only non-owned allocation is `out` — the result escapes to the caller, so
//! it is a plain `GpuTensor` that never enters the mailbox.

use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor, OwnedTensor};

use crate::config::SigLipConfig;
use crate::vision::SigLipWeights;

/// Debug: when `HIPFIRE_VISION_DUMP=<dir>` is set, write a vision-tower stage
/// to `<dir>/<name>.bin` (raw f32 LE) + `<dir>/<name>.json` ({"shape":[...]}) so
/// `benchmarks/vision/diff_dumps.py` can bisect against an HF reference. No-op
/// when unset. Errors are swallowed (diagnostic-only).
pub(crate) fn maybe_dump_stage(gpu: &mut Gpu, t: &GpuTensor, name: &str, shape: &[usize]) {
    let dir = match std::env::var("HIPFIRE_VISION_DUMP") {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(data) = gpu.download_f32(t) {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/{name}.bin"), bytes);
        let _ = std::fs::write(
            format!("{dir}/{name}.json"),
            format!("{{\"shape\":{shape:?}}}"),
        );
    }
}

/// Batched linear `Y[n, out] = X[n, in] · W[out, in]ᵀ + bias`, F32.
///
/// `gemm_f32_batched` already writes its result as `[n, out]` (one warp per
/// (out, token) cell, `Y[n*out_dim + m]`), which is exactly the layout we want —
/// so NO transpose is applied. (A prior version transposed the result, which
/// scrambled every patch embedding: the patch-embed conv came out as the
/// transpose of the correct values, breaking vision grounding entirely. Bisected
/// against the HF SigLIP reference — see benchmarks/vision/diff_dumps.py.)
///
/// Returns an [`OwnedTensor`]: the caller owns the result and it frees itself on
/// drop. `&y` deref-coerces to `&GpuTensor` at the kernel call sites.
fn linear_f32(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<OwnedTensor> {
    let y = gpu.alloc_owned(&[n * out_dim], DType::F32)?;
    gpu.gemm_f32_batched(w, x, &y, out_dim, in_dim, n)?;
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
///
/// Returns an [`OwnedTensor`] (frees itself on drop). See `linear_f32`.
fn linear_bf16(
    gpu: &mut Gpu,
    w_bf16: &GpuTensor,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<OwnedTensor> {
    let y = gpu.alloc_owned(&[n * out_dim], DType::F32)?;
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
    let x_patches = gpu.upload_owned_f32(patches, &[n * patch_dim])?;
    maybe_dump_stage(gpu, &x_patches, "patches_raw", &[n, patch_dim]);
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
    drop(x_patches); // scratch consumed — enqueue for the boundary reclaim.

    // + learned position embedding (fixed grid, direct add — no interpolation).
    timed!(3, gpu.add_inplace_f32(&x, &weights.pos_embed));
    // HF's `embeddings` hook captures patch+position embedding together; match it.
    maybe_dump_stage(gpu, &x, "patch_embed", &[n, h]);

    for (li, lw) in weights.layers.iter().enumerate() {
        // ── self-attention block (LN1 → attn → residual) ──
        // Per-iteration scratch is `OwnedTensor`: it drops at the end of this
        // loop body (or instantly on a `?` error), enqueueing for reclaim. No
        // explicit frees, and any early `?` is leak-free.
        let tmp = gpu.alloc_owned(&[n * h], DType::F32)?;
        timed!(
            2,
            gpu.layernorm_batched(&x, &lw.ln1_w, &lw.ln1_b, &tmp, n, h, eps)
        );
        let qkv = timed!(0, linear_bf16(gpu, &lw.qkv_w, &tmp, &lw.qkv_b, 3 * h, h, n));
        drop(tmp);
        // Bidirectional attention (no causal mask). Fast path on WMMA archs
        // (RDNA3/3.5/4): the f16-KV matrix-core flash, which needs head_dim=128 —
        // split the fused qkv into padded q(f32)/k(f16)/v(f16), run, then unpad
        // (~22× the generic kernel on this shape; zero-padded dims contribute
        // nothing to QKᵀ/PV). Fallback on non-WMMA archs: the generic bf16 flash
        // over the fused qkv directly.
        let attn_out = gpu.alloc_owned(&[n * h], DType::F32)?;
        // WMMA f16-KV flash by default (fast path). After the q-prescale fix
        // (correct 1/sqrt(head_dim) scale despite the kernel's fixed 1/sqrt(hdp))
        // this path is numerically equivalent to the generic bf16 flash —
        // verified byte-identical decode output across both. HIPFIRE_GEMMA3_
        // VISION_NOWMMA=1 forces the generic path for A/B diagnostics.
        let use_wmma = gpu.arch_caps.has_wmma_w32()
            && std::env::var("HIPFIRE_GEMMA3_VISION_NOWMMA")
                .ok()
                .as_deref()
                != Some("1");
        if use_wmma {
            let hdp = 128usize;
            let q_pad = gpu.alloc_owned(&[n * num_heads * hdp], DType::F32)?;
            let k_pad = gpu.alloc_owned(&[n * num_heads * hdp], DType::F16)?;
            let v_pad = gpu.alloc_owned(&[n * num_heads * hdp], DType::F16)?;
            // The WMMA flash below bakes a fixed 1/sqrt(hdp) softmax scale, but
            // SigLIP's real head_dim (e.g. 72) != hdp (128). Pre-scale Q by
            // sqrt(hdp/head_dim) so the effective scale is the correct
            // 1/sqrt(head_dim); without this the attention is over-smoothed
            // (~0.75x scores) and vision features lose spatial discrimination.
            let q_scale = (hdp as f32 / head_dim as f32).sqrt();
            timed!(
                1,
                gpu.attn_split_pad_f16kv(
                    &qkv, &q_pad, &k_pad, &v_pad, n, h, num_heads, head_dim, hdp, q_scale,
                )
            );
            drop(qkv);
            let attn_pad = gpu.alloc_owned(&[n * num_heads * hdp], DType::F32)?;
            timed!(
                1,
                gpu.attention_dflash_wmma_m64_n128_f16kv_v3_f32(
                    &q_pad, &k_pad, &v_pad, &attn_pad, n, n, num_heads, num_heads, hdp,
                )
            );
            drop(q_pad);
            drop(k_pad);
            drop(v_pad);
            timed!(
                1,
                gpu.attn_unpad(&attn_pad, &attn_out, n, num_heads, head_dim, hdp)
            );
        } else {
            let qkv_bf16 = gpu.alloc_owned(&[n * 3 * h], DType::BF16)?;
            timed!(1, gpu.cast_f32_to_bf16(&qkv, &qkv_bf16));
            drop(qkv);
            timed!(
                1,
                gpu.flash_attn_bf16(&qkv_bf16, &attn_out, n, h, num_heads, head_dim)
            );
        }
        let proj = timed!(
            0,
            linear_bf16(gpu, &lw.out_w, &attn_out, &lw.out_b, h, h, n)
        );
        drop(attn_out);
        timed!(3, gpu.add_inplace_f32(&x, &proj));
        drop(proj);

        // ── MLP block (LN2 → fc1 → gelu-tanh → fc2 → residual) ──
        let tmp2 = gpu.alloc_owned(&[n * h], DType::F32)?;
        timed!(
            2,
            gpu.layernorm_batched(&x, &lw.ln2_w, &lw.ln2_b, &tmp2, n, h, eps)
        );
        let fc1 = timed!(
            0,
            linear_bf16(gpu, &lw.fc1_w, &tmp2, &lw.fc1_b, inter, h, n)
        );
        drop(tmp2);
        timed!(3, gpu.gelu_tanh_f32(&fc1, &fc1, n * inter));
        let fc2 = timed!(0, linear_bf16(gpu, &lw.fc2_w, &fc1, &lw.fc2_b, h, inter, n));
        drop(fc1);
        timed!(3, gpu.add_inplace_f32(&x, &fc2));
        drop(fc2);
        maybe_dump_stage(gpu, &x, &format!("block_{li:02}"), &[n, h]);
        // Per-iteration reclaim: this layer's scratch (tmp/qkv/q_pad/…/fc2) goes
        // back to the pool so the next layer reuses it — peak VRAM stays flat
        // across the encoder's layers instead of growing with depth. No-op while
        // a graph is live (self-gated); the residual stream `x` is still alive,
        // so it is never reclaimed here.
        gpu.reclaim_pending();
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

    // Final post_layernorm → [n, h]. `out` is a PLAIN `GpuTensor` (not owned):
    // it escapes to the caller, who frees it, so it must never enter the mailbox.
    let out = gpu.alloc_tensor(&[n * h], DType::F32)?;
    gpu.layernorm_batched(&x, &weights.post_ln_w, &weights.post_ln_b, &out, n, h, eps)?;
    drop(x); // residual stream scratch — enqueue before the reclaim below.
    maybe_dump_stage(gpu, &out, "pre_merger", &[n, h]);
    // Boundary reclaim: return the residual stream `x` (and anything still queued)
    // to the pool. No-op under capture.
    gpu.reclaim_pending();
    Ok(out)
}
