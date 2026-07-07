// SPDX-License-Identifier: Apache-2.0
// hipfire — SigLIP vision tower weights + loader. See LICENSE / NOTICE.

//! GPU-resident SigLIP weights (`vision_tower.vision_model.*`) and the loader.
//!
//! SigLIP is a standard ViT: Conv2d patch embed (k=s=patch, loaded as a linear
//! `[hidden, 3·patch²]`), a learned position-embedding table, N encoder layers
//! (LayerNorm+bias → bidirectional self-attn → residual; LayerNorm+bias →
//! gelu-tanh MLP → residual), and a final `post_layernorm`. Matrix weights
//! (qkv/out/fc1/fc2) load as **BF16** for the `gemm_bf16_x_bf16_wmma` matrix-core
//! linears; biases, LayerNorms, and the position embedding stay F32. q/k/v are
//! **fused at load** into one `[3·hidden, hidden]` weight matching the fused-qkv
//! layout `flash_attn_bf16` consumes.
//!
//! The forward lives in `forward.rs` (next phase); this file is weights only.

use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::quant::{dequant_oq8g256, dequant_q8f16, f16_to_f32};

use crate::config::SigLipConfig;

/// Per-layer SigLIP encoder weights. LayerNorms carry bias (`beta`); attention
/// is square (`hidden→hidden` for q/k/v/out). q/k/v are fused into `qkv_*`.
pub struct SigLipLayerWeights {
    pub ln1_w: GpuTensor,
    pub ln1_b: GpuTensor,
    pub qkv_w: GpuTensor, // fused [3·hidden, hidden]
    pub qkv_b: GpuTensor, // fused [3·hidden]
    pub out_w: GpuTensor, // [hidden, hidden]
    pub out_b: GpuTensor,
    pub ln2_w: GpuTensor,
    pub ln2_b: GpuTensor,
    pub fc1_w: GpuTensor, // [intermediate, hidden]
    pub fc1_b: GpuTensor,
    pub fc2_w: GpuTensor, // [hidden, intermediate]
    pub fc2_b: GpuTensor,
}

/// GPU-resident SigLIP vision tower.
pub struct SigLipWeights {
    /// Conv2d patch embedding as a linear weight `[hidden, 3·patch²]` + bias.
    pub patch_embed_w: GpuTensor,
    pub patch_embed_b: GpuTensor,
    /// Learned position embedding `[num_patches, hidden]` (fixed grid — no
    /// interpolation needed since image_size/patch_size is fixed).
    pub pos_embed: GpuTensor,
    pub layers: Vec<SigLipLayerWeights>,
    pub post_ln_w: GpuTensor,
    pub post_ln_b: GpuTensor,
}

impl SigLipWeights {
    pub fn load(hfq: &mut HfqFile, cfg: &SigLipConfig, gpu: &mut Gpu) -> Result<Self, String> {
        load_vision_weights(hfq, cfg, gpu)
            .map_err(|e| format!("gemma3-vl: vision load failed: {e:?}"))
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.patch_embed_w,
            self.patch_embed_b,
            self.pos_embed,
            self.post_ln_w,
            self.post_ln_b,
        ] {
            let _ = gpu.free_tensor(t);
        }
        for l in self.layers {
            for t in [
                l.ln1_w, l.ln1_b, l.qkv_w, l.qkv_b, l.out_w, l.out_b, l.ln2_w, l.ln2_b, l.fc1_w,
                l.fc1_b, l.fc2_w, l.fc2_b,
            ] {
                let _ = gpu.free_tensor(t);
            }
        }
    }
}

// ─── loader ─────────────────────────────────────────────────────────────────

/// Read a tensor to a host F32 vector. Handles float sources (F16/F32/BF16) and
/// dequantizes a quantized vision tower back to f32 (`hfq --vision-quant`, e.g.
/// `.vloq8+`: Q8F16 qt=3, Oq8G256 qt=35).
fn read_f32(hfq: &HfqFile, name: &str) -> Vec<f32> {
    // tensor_data_vec (pread) not tensor_data (mmap): on UMA APUs the loader
    // drops the mmap, so the mmap accessor returns None.
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("gemma3-vl: vision tensor not found: {name}"));
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
        // Quantized vision tower (`hfq --vision-quant`, e.g. `.vloq8+`): Q8F16
        // (qt=3, group 32) matrix weights + position embedding, Oq8G256 (qt=35,
        // group 256) for the wider projections. Dequant to f32 here so the rest
        // of the tower stays float (matrix weights are re-narrowed to bf16 by
        // `load_gpu_bf16`); `n` = element count from the tensor's shape.
        3 => dequant_q8f16(&data, info.shape.iter().product::<u32>() as usize),
        35 => dequant_oq8g256(&data, info.shape.iter().product::<u32>() as usize),
        qt => panic!("gemma3-vl: expected F16/F32/BF16/Q8F16/Oq8G256 for {name}, got qt={qt}"),
    }
}

fn load_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let v = read_f32(hfq, name);
    assert_eq!(
        v.len(),
        n,
        "gemma3-vl: tensor {name} has {} elems, expected {n}",
        v.len()
    );
    gpu.upload_f32(&v, &[n])
}

/// f32 → bf16 bits (round-to-nearest-even, top 16 bits of the f32).
fn f32_to_bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    let lsb = (u >> 16) & 1;
    ((u + 0x7fff + lsb) >> 16) as u16
}

/// Upload an f32 host slice as a `DType::BF16` device tensor (for the bf16 WMMA
/// linears — the matrix-core path halves weight bandwidth, the UMA bottleneck).
fn upload_bf16(gpu: &Gpu, vals: &[f32]) -> HipResult<GpuTensor> {
    let bytes: Vec<u8> = vals
        .iter()
        .flat_map(|&x| f32_to_bf16_bits(x).to_le_bytes())
        .collect();
    let mut t = gpu.upload_raw(&bytes, &[vals.len()])?;
    t.dtype = DType::BF16;
    Ok(t)
}

/// Like [`load_gpu`] but stores the weight as BF16 (matrix weights only — biases
/// and norms stay F32).
fn load_gpu_bf16(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let v = read_f32(hfq, name);
    assert_eq!(
        v.len(),
        n,
        "gemma3-vl: tensor {name} has {} elems, expected {n}",
        v.len()
    );
    upload_bf16(gpu, &v)
}

fn load_vision_weights(
    hfq: &mut HfqFile,
    cfg: &SigLipConfig,
    gpu: &mut Gpu,
) -> HipResult<SigLipWeights> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let patch_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size; // 3*14*14 = 588
    let n_patches = cfg.num_patches();
    let vp = "vision_tower.vision_model";

    // Conv2d patch_embedding.weight is [hidden, 3, patch, patch]; flatten to the
    // linear `[hidden, 3·patch²]`. (Row-major flatten matches im2col patch order.)
    let patch_embed_w = load_gpu(
        hfq,
        gpu,
        &format!("{vp}.embeddings.patch_embedding.weight"),
        h * patch_dim,
    )?;
    let patch_embed_b = load_gpu(
        hfq,
        gpu,
        &format!("{vp}.embeddings.patch_embedding.bias"),
        h,
    )?;
    let pos_embed = load_gpu(
        hfq,
        gpu,
        &format!("{vp}.embeddings.position_embedding.weight"),
        n_patches * h,
    )?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("{vp}.encoder.layers.{i}");
        // Fuse q/k/v (each [h,h]) into one [3h,h] weight + [3h] bias, in q,k,v
        // order — the layout vit_attention_f32 expects from a fused qkv.
        let mut qkv_w = read_f32(hfq, &format!("{lp}.self_attn.q_proj.weight"));
        qkv_w.extend(read_f32(hfq, &format!("{lp}.self_attn.k_proj.weight")));
        qkv_w.extend(read_f32(hfq, &format!("{lp}.self_attn.v_proj.weight")));
        assert_eq!(
            qkv_w.len(),
            3 * h * h,
            "gemma3-vl: fused qkv_w size mismatch at layer {i}"
        );
        let mut qkv_b = read_f32(hfq, &format!("{lp}.self_attn.q_proj.bias"));
        qkv_b.extend(read_f32(hfq, &format!("{lp}.self_attn.k_proj.bias")));
        qkv_b.extend(read_f32(hfq, &format!("{lp}.self_attn.v_proj.bias")));

        // Matrix weights → BF16 (consumed by gemm_bf16_x_bf16_wmma); biases and
        // LayerNorm params stay F32.
        layers.push(SigLipLayerWeights {
            ln1_w: load_gpu(hfq, gpu, &format!("{lp}.layer_norm1.weight"), h)?,
            ln1_b: load_gpu(hfq, gpu, &format!("{lp}.layer_norm1.bias"), h)?,
            qkv_w: upload_bf16(gpu, &qkv_w)?,
            qkv_b: gpu.upload_f32(&qkv_b, &[3 * h])?,
            out_w: load_gpu_bf16(hfq, gpu, &format!("{lp}.self_attn.out_proj.weight"), h * h)?,
            out_b: load_gpu(hfq, gpu, &format!("{lp}.self_attn.out_proj.bias"), h)?,
            ln2_w: load_gpu(hfq, gpu, &format!("{lp}.layer_norm2.weight"), h)?,
            ln2_b: load_gpu(hfq, gpu, &format!("{lp}.layer_norm2.bias"), h)?,
            fc1_w: load_gpu_bf16(hfq, gpu, &format!("{lp}.mlp.fc1.weight"), inter * h)?,
            fc1_b: load_gpu(hfq, gpu, &format!("{lp}.mlp.fc1.bias"), inter)?,
            fc2_w: load_gpu_bf16(hfq, gpu, &format!("{lp}.mlp.fc2.weight"), h * inter)?,
            fc2_b: load_gpu(hfq, gpu, &format!("{lp}.mlp.fc2.bias"), h)?,
        });
    }

    let post_ln_w = load_gpu(hfq, gpu, &format!("{vp}.post_layernorm.weight"), h)?;
    let post_ln_b = load_gpu(hfq, gpu, &format!("{vp}.post_layernorm.bias"), h)?;

    Ok(SigLipWeights {
        patch_embed_w,
        patch_embed_b,
        pos_embed,
        layers,
        post_ln_w,
        post_ln_b,
    })
}
