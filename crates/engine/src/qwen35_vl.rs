//! Qwen3.5-VL vision encoder: SigLIP-2 ViT + spatial merger.
//! GPU path: gemm_f16 (9 VGPRs), layernorm (13), gelu (8), vit_attention, transpose.

use crate::hfq::HfqFile;
use crate::llama::{f16_to_f32, f32_to_f16};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub mlp_dim: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub spatial_merge_size: usize,
    pub norm_eps: f32,
}

pub fn vision_config_from_hfq(hfq: &HfqFile) -> Option<VisionConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let vc = config.get("vision_config")?;

    let hidden_size = vc.get("hidden_size")?.as_u64()? as usize;
    let num_heads = vc.get("num_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let num_layers = vc.get("depth").and_then(|v| v.as_u64()).unwrap_or(27) as usize;
    let mlp_dim = vc.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(4304) as usize;
    let patch_size = vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let temporal_patch_size = vc.get("temporal_patch_size").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    let out_hidden_size = vc.get("out_hidden_size").and_then(|v| v.as_u64())
        .or_else(|| config.get("text_config").and_then(|tc| tc.get("hidden_size")).and_then(|v| v.as_u64()))
        .unwrap_or(4096) as usize;
    let spatial_merge_size = vc.get("spatial_merge_size").and_then(|v| v.as_u64()).unwrap_or(2) as usize;

    Some(VisionConfig {
        hidden_size, num_heads, head_dim: hidden_size / num_heads,
        num_layers, mlp_dim, patch_size, temporal_patch_size,
        out_hidden_size, spatial_merge_size, norm_eps: 1e-6,
    })
}

// ─── GPU-side weights ────────────────────────────────────────────────────────

pub struct VisionLayerWeights {
    pub norm1_w: GpuTensor, pub norm1_b: GpuTensor,
    pub qkv_w: GpuTensor, pub qkv_b: GpuTensor,
    pub proj_w: GpuTensor, pub proj_b: GpuTensor,
    pub norm2_w: GpuTensor, pub norm2_b: GpuTensor,
    pub fc1_w: GpuTensor, pub fc1_b: GpuTensor,
    pub fc2_w: GpuTensor, pub fc2_b: GpuTensor,
}

pub struct VisionWeights {
    pub patch_embed_w: GpuTensor, pub patch_embed_b: GpuTensor,
    pub pos_embed: GpuTensor,
    pub layers: Vec<VisionLayerWeights>,
    pub merger_norm_w: GpuTensor, pub merger_norm_b: GpuTensor,
    pub merger_fc1_w: GpuTensor, pub merger_fc1_b: GpuTensor,
    pub merger_fc2_w: GpuTensor, pub merger_fc2_b: GpuTensor,
}

impl VisionWeights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.patch_embed_w);
        let _ = gpu.free_tensor(self.patch_embed_b);
        let _ = gpu.free_tensor(self.pos_embed);
        for l in self.layers {
            for t in [l.norm1_w, l.norm1_b, l.qkv_w, l.qkv_b, l.proj_w, l.proj_b,
                      l.norm2_w, l.norm2_b, l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b] {
                let _ = gpu.free_tensor(t);
            }
        }
        let _ = gpu.free_tensor(self.merger_norm_w);
        let _ = gpu.free_tensor(self.merger_norm_b);
        let _ = gpu.free_tensor(self.merger_fc1_w);
        let _ = gpu.free_tensor(self.merger_fc1_b);
        let _ = gpu.free_tensor(self.merger_fc2_w);
        let _ = gpu.free_tensor(self.merger_fc2_b);
    }
}

// ─── Weight loading ──────────────────────────────────────────────────────────

fn load_f32_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data(name)
        .unwrap_or_else(|| panic!("vision tensor not found: {name}"));
    let vals: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        6 | 7 => dequant_hfq4(data, n, info.group_size as usize),
        _ => panic!("expected F16/F32/HFQ4 for {name}, got qt={}", info.quant_type),
    };
    gpu.upload_f32(&vals[..n], &[n])
}

fn load_f16_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data(name)
        .unwrap_or_else(|| panic!("vision tensor not found: {name}"));
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    match info.quant_type {
        1 => {
            // F16 — upload directly. Shape records element count, not byte count.
            gpu.upload_raw(data, &[n])
        }
        6 | 7 => {
            // HFQ4 — dequantize to F32, then convert to F16 for gemm_f16.
            // Shape records element count, not byte count.
            let f32_data = dequant_hfq4(data, n, info.group_size as usize);
            let f16_bytes: Vec<u8> = f32_data.iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            gpu.upload_raw(&f16_bytes, &[n])
        }
        other => panic!("{name}: unsupported vision quant_type={other} (expected F16=1, HFQ4=6/7)"),
    }
}

/// Dequantize HFQ4 blocks to F32, using actual group_size (128 or 256).
/// G256 block: [scale:f32, zero:f32, 128 bytes nibbles] = 136 bytes per 256 values.
/// G128 block: [scale:f32, zero:f32, 64 bytes nibbles] = 72 bytes per 128 values.
fn dequant_hfq4(data: &[u8], n: usize, group_size: usize) -> Vec<f32> {
    let nibble_bytes = group_size / 2;
    let block_size = 8 + nibble_bytes; // 4+4 scale/zero + nibbles
    let mut out = Vec::with_capacity(n);
    let n_groups = n.div_ceil(group_size);
    for g in 0..n_groups {
        let off = g * block_size;
        if off + 8 > data.len() { break; }
        let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        let zero = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
        let nibbles = &data[off+8..(off+block_size).min(data.len())];
        let base = g * group_size;
        for i in 0..group_size.min(n.saturating_sub(base)) {
            let byte_idx = i / 2;
            if byte_idx >= nibbles.len() { break; }
            let nibble = if i % 2 == 0 { nibbles[byte_idx] & 0xF } else { nibbles[byte_idx] >> 4 };
            out.push(scale * nibble as f32 + zero);
        }
    }
    out.truncate(n);
    out
}

pub fn load_vision_weights(hfq: &HfqFile, config: &VisionConfig, gpu: &mut Gpu) -> HipResult<VisionWeights> {
    let h = config.hidden_size;
    // Detect vision weight format (F16 direct vs HFQ4 auto-dequant) and log once.
    // HFQ4 vision weights (qt=6 G256, qt=7 G128) are dequantized to F16 at load
    // time for the gemm_f16 path — there is no GPU HFQ4 kernel for vision yet.
    // See CHANGELOG.md "v0.1.7-alpha.4 / Vision" for details.
    if let Some((info, _)) = hfq.tensor_data("model.visual.patch_embed.proj.weight") {
        let fmt = match info.quant_type {
            1 => "F16 (direct)",
            6 => "HFQ4-G256 (dequanting to F16 on load)",
            7 => "HFQ4-G128 (dequanting to F16 on load)",
            other => &format!("qt={other}"),
        };
        eprintln!("  vision weight format: {fmt}");
    }
    eprintln!("  loading vision weights (GPU)...");
    let patch_embed_w = load_f16_gpu(hfq, gpu, "model.visual.patch_embed.proj.weight")?;
    let patch_embed_b = load_f32_gpu(hfq, gpu, "model.visual.patch_embed.proj.bias", h)?;
    let pos_embed = load_f32_gpu(hfq, gpu, "model.visual.pos_embed.weight", 2304 * h)?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        if i % 9 == 0 { eprintln!("  loading vision block {i}/{}...", config.num_layers); }
        let p = format!("model.visual.blocks.{i}");
        layers.push(VisionLayerWeights {
            norm1_w: load_f32_gpu(hfq, gpu, &format!("{p}.norm1.weight"), h)?,
            norm1_b: load_f32_gpu(hfq, gpu, &format!("{p}.norm1.bias"), h)?,
            qkv_w: load_f16_gpu(hfq, gpu, &format!("{p}.attn.qkv.weight"))?,
            qkv_b: load_f32_gpu(hfq, gpu, &format!("{p}.attn.qkv.bias"), 3 * h)?,
            proj_w: load_f16_gpu(hfq, gpu, &format!("{p}.attn.proj.weight"))?,
            proj_b: load_f32_gpu(hfq, gpu, &format!("{p}.attn.proj.bias"), h)?,
            norm2_w: load_f32_gpu(hfq, gpu, &format!("{p}.norm2.weight"), h)?,
            norm2_b: load_f32_gpu(hfq, gpu, &format!("{p}.norm2.bias"), h)?,
            fc1_w: load_f16_gpu(hfq, gpu, &format!("{p}.mlp.linear_fc1.weight"))?,
            fc1_b: load_f32_gpu(hfq, gpu, &format!("{p}.mlp.linear_fc1.bias"), config.mlp_dim)?,
            fc2_w: load_f16_gpu(hfq, gpu, &format!("{p}.mlp.linear_fc2.weight"))?,
            fc2_b: load_f32_gpu(hfq, gpu, &format!("{p}.mlp.linear_fc2.bias"), h)?,
        });
    }

    let merge_dim = h * config.spatial_merge_size * config.spatial_merge_size;
    eprintln!("  loading vision merger...");
    Ok(VisionWeights {
        patch_embed_w, patch_embed_b, pos_embed, layers,
        merger_norm_w: load_f32_gpu(hfq, gpu, "model.visual.merger.norm.weight", h)?,
        merger_norm_b: load_f32_gpu(hfq, gpu, "model.visual.merger.norm.bias", h)?,
        merger_fc1_w: load_f16_gpu(hfq, gpu, "model.visual.merger.linear_fc1.weight")?,
        merger_fc1_b: load_f32_gpu(hfq, gpu, "model.visual.merger.linear_fc1.bias", merge_dim)?,
        merger_fc2_w: load_f16_gpu(hfq, gpu, "model.visual.merger.linear_fc2.weight")?,
        merger_fc2_b: load_f32_gpu(hfq, gpu, "model.visual.merger.linear_fc2.bias", config.out_hidden_size)?,
    })
}

// ─── GPU vision forward (no CPU roundtrips for compute) ──────────────────────

/// gemm_f16 produces Y[M,N]. We need [N,M]. This helper does GEMM + transpose + bias.
fn linear_f16(
    gpu: &mut Gpu, w: &GpuTensor, x: &GpuTensor, bias: &GpuTensor,
    out_dim: usize, in_dim: usize, n: usize,
) -> HipResult<GpuTensor> {
    // GEMM: Y_t[out_dim, n] = W[out_dim, in_dim] @ X[n, in_dim]^T
    let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
    gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
    // Transpose: Y[n, out_dim]
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    gpu.transpose_f32(&yt, &y, out_dim, n)?;
    gpu.free_tensor(yt)?;
    // Bias
    gpu.bias_add_f32(&y, bias, n, out_dim)?;
    Ok(y)
}

pub fn vision_forward(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    config: &VisionConfig,
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> HipResult<Vec<f32>> {
    let h = config.hidden_size;
    let n = grid_h * grid_w;
    let patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
    let t0 = std::time::Instant::now();

    eprintln!("  vision forward (GPU): {} patches, {}x{} grid", n, grid_h, grid_w);

    // Upload patches [n, patch_dim]
    let x_patches = gpu.upload_f32(patches, &[n * patch_dim])?;

    // Patch embedding: linear_f16 → [n, h]
    let mut x = linear_f16(gpu, &weights.patch_embed_w, &x_patches, &weights.patch_embed_b, h, patch_dim, n)?;
    gpu.free_tensor(x_patches)?;

    // Add position embeddings (first n*h elements of pos_embed)
    gpu.add_inplace_f32(&x, &weights.pos_embed)?;

    // Scratch buffers reused across layers
    let qkv_dim = 3 * h;

    for li in 0..config.num_layers {
        let lw = &weights.layers[li];

        // LayerNorm1 → tmp
        let tmp = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.layernorm_batched(&x, &lw.norm1_w, &lw.norm1_b, &tmp, n, h, config.norm_eps)?;

        // QKV projection → [n, 3h]
        let qkv = linear_f16(gpu, &lw.qkv_w, &tmp, &lw.qkv_b, qkv_dim, h, n)?;
        gpu.free_tensor(tmp)?;

        // Self-attention on GPU: qkv[n, 3h] → attn_out[n, h]
        let attn_out = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.vit_attention_f32(&qkv, &attn_out, n, h, config.num_heads, config.head_dim)?;
        gpu.free_tensor(qkv)?;

        // Output projection → [n, h]
        let proj = linear_f16(gpu, &lw.proj_w, &attn_out, &lw.proj_b, h, h, n)?;
        gpu.free_tensor(attn_out)?;

        // Residual: x += proj
        gpu.add_inplace_f32(&x, &proj)?;
        gpu.free_tensor(proj)?;

        // LayerNorm2 → tmp
        let tmp2 = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.layernorm_batched(&x, &lw.norm2_w, &lw.norm2_b, &tmp2, n, h, config.norm_eps)?;

        // MLP: fc1 → GELU → fc2
        let fc1 = linear_f16(gpu, &lw.fc1_w, &tmp2, &lw.fc1_b, config.mlp_dim, h, n)?;
        gpu.free_tensor(tmp2)?;
        gpu.gelu_tanh_f32(&fc1, &fc1, n * config.mlp_dim)?;

        let fc2 = linear_f16(gpu, &lw.fc2_w, &fc1, &lw.fc2_b, h, config.mlp_dim, n)?;
        gpu.free_tensor(fc1)?;

        // Residual: x += fc2
        gpu.add_inplace_f32(&x, &fc2)?;
        gpu.free_tensor(fc2)?;
    }

    // Single sync at end of all layers (avoids per-layer sync overhead)
    gpu.hip.device_synchronize()?;
    eprintln!("  vision forward complete ({:.2}s)", t0.elapsed().as_secs_f32());

    // Spatial merge: [n, h] → [n_merged, merge_dim] (CPU rearrange, small data)
    let sms = config.spatial_merge_size;
    let merged_h = grid_h / sms;
    let merged_w = grid_w / sms;
    let n_merged = merged_h * merged_w;
    let merge_dim = h * sms * sms;

    // LayerNorm all patches
    let normed = gpu.alloc_tensor(&[n * h], DType::F32)?;
    gpu.layernorm_batched(&x, &weights.merger_norm_w, &weights.merger_norm_b, &normed, n, h, config.norm_eps)?;
    gpu.free_tensor(x)?;

    // Download for 2x2 rearrange (only ~3.6MB, one-time cost)
    let normed_data = gpu.download_f32(&normed)?;
    gpu.free_tensor(normed)?;

    let mut merged = vec![0.0f32; n_merged * merge_dim];
    for my in 0..merged_h {
        for mx in 0..merged_w {
            let out_idx = my * merged_w + mx;
            for dy in 0..sms {
                for dx in 0..sms {
                    let src = (my * sms + dy) * grid_w + (mx * sms + dx);
                    let sub = dy * sms + dx;
                    merged[out_idx * merge_dim + sub * h..out_idx * merge_dim + sub * h + h]
                        .copy_from_slice(&normed_data[src * h..src * h + h]);
                }
            }
        }
    }

    // Merger MLP on GPU
    let merged_gpu = gpu.upload_f32(&merged, &[n_merged * merge_dim])?;
    let m1 = linear_f16(gpu, &weights.merger_fc1_w, &merged_gpu, &weights.merger_fc1_b, merge_dim, merge_dim, n_merged)?;
    gpu.free_tensor(merged_gpu)?;
    gpu.gelu_tanh_f32(&m1, &m1, n_merged * merge_dim)?;

    let m2 = linear_f16(gpu, &weights.merger_fc2_w, &m1, &weights.merger_fc2_b, config.out_hidden_size, merge_dim, n_merged)?;
    gpu.free_tensor(m1)?;

    let result = gpu.download_f32(&m2)?;
    gpu.free_tensor(m2)?;

    eprintln!("  vision done: {} tokens × {} dims ({:.2}s)",
        n_merged, config.out_hidden_size, t0.elapsed().as_secs_f32());
    Ok(result)
}
