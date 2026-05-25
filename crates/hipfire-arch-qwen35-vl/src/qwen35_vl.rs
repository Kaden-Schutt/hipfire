// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5-VL vision encoder: SigLIP-2 ViT + spatial merger.
//! GPU path: gemm_f16 (9 VGPRs), layernorm (13), gelu (8), vit_attention, transpose.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
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
    /// Number of entries in the learned `pos_embed` table. The table is laid
    /// out as a square grid of side `sqrt(num_position_embeddings)` and
    /// bilinearly interpolated to each image's `(grid_h, grid_w)` at forward
    /// time — see `fast_pos_embed_interpolate`.
    pub num_position_embeddings: usize,
    /// Base θ for the 2D vision rotary frequencies (HF default 10000.0).
    /// Read from `vision_config.rope_theta` if present, else defaulted —
    /// future variants that override `Qwen3_5VisionRotaryEmbedding(theta=…)`
    /// will pick up the correct value automatically.
    pub rope_theta: f32,
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
    let num_position_embeddings = vc.get("num_position_embeddings")
        .and_then(|v| v.as_u64()).unwrap_or(2304) as usize;
    let rope_theta = vc.get("rope_theta")
        .and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;

    Some(VisionConfig {
        hidden_size, num_heads, head_dim: hidden_size / num_heads,
        num_layers, mlp_dim, patch_size, temporal_patch_size,
        out_hidden_size, spatial_merge_size, num_position_embeddings,
        rope_theta, norm_eps: 1e-6,
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
    /// Learned position-embedding table `(num_position_embeddings, hidden)`,
    /// resident on CPU because every image bilinearly interpolates a different
    /// `(grid_h, grid_w)` slice out of it (`fast_pos_embed_interpolate`). Cost
    /// is small enough (~10 MB at 2304×1152 F32) that keeping it on host is
    /// strictly simpler than re-uploading shards per image.
    pub pos_embed: Vec<f32>,
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

fn load_f32_cpu(hfq: &HfqFile, name: &str, n: usize) -> Vec<f32> {
    let (info, data) = hfq.tensor_data(name)
        .unwrap_or_else(|| panic!("vision tensor not found: {name}"));
    let mut vals: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        6 | 7 => dequant_hfq4(data, n, info.group_size as usize),
        _ => panic!("expected F16/F32/HFQ4 for {name}, got qt={}", info.quant_type),
    };
    vals.truncate(n);
    vals
}

fn load_f32_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let vals = load_f32_cpu(hfq, name, n);
    gpu.upload_f32(&vals, &[n])
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

    // Arch advisory. The vision tower kernels (gemm_f16, layernorm_batched,
    // vit_attention_f32, apply_rope_2d_vision_f32, gelu_tanh_f32, etc.) are
    // arch-neutral HIP and should COMPILE on any ROCm-supported GPU, but the
    // 12/12 quality bench against llama.cpp was only run on gfx1100 (RX 7900 XT).
    // Other archs are unvalidated — we don't refuse, but we warn so users can
    // distinguish "tested" from "should work in theory."
    // Tracked validation: gfx1100, gfx1101, gfx1102 (all RDNA3 wave32).
    match gpu.arch.as_str() {
        "gfx1100" | "gfx1101" | "gfx1102" => {}
        other => {
            eprintln!("  ⚠ vision tower not yet validated on {other}; \
                       results may differ from gfx1100 reference. See \
                       benchmarks/vision/comparison-2026-05-23.md for the \
                       gfx1100 baseline.");
        }
    }

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
    let pos_embed = load_f32_cpu(hfq, "model.visual.pos_embed.weight", config.num_position_embeddings * h);

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

// ─── CPU-side per-image precomputes (pos_embed interp + 2D rotary) ───────────

/// Bilinearly interpolate the learned `(K×K, hidden)` position-embedding table
/// down to the actual `(grid_h, grid_w)`, then reorder into the 2×2 spatial-
/// merge-grouped layout that `extract_patches` emits. Output: `[n*hidden]`.
///
/// HF reference: `Qwen3_5VisionModel.fast_pos_embed_interpolate`. For each
/// output position `(py, px)` the table is sampled at the four corners
/// `(floor(py'), floor(px'))`, `(floor(py'), ceil(px'))`, `(ceil(py'),
/// floor(px'))`, `(ceil(py'), ceil(px'))` with `py' = py * (K-1)/(grid_h-1)`
/// — i.e. `linspace(0, K-1, grid_h)` — and the four embeddings are blended by
/// the standard bilinear weights `((1-dh)(1-dw), (1-dh)dw, dh(1-dw), dh dw)`.
///
/// The output is permuted from `(grid_h, grid_w, hidden)` row-major into the
/// same 2x2-grouped order as `extract_patches`: for each merged-block (gy, gx)
/// in row-major, the four intra-block patches `(sy, sx) ∈ {0,1}²` are
/// consecutive — matching `view(mh, ms, mw, ms, h).permute(0, 2, 1, 3, 4)`.
pub fn fast_pos_embed_interpolate(
    pos_embed: &[f32],
    hidden: usize,
    grid_h: usize,
    grid_w: usize,
    num_grid_per_side: usize,
    merge_size: usize,
) -> Vec<f32> {
    assert!(grid_h % merge_size == 0 && grid_w % merge_size == 0);
    assert!(pos_embed.len() == num_grid_per_side * num_grid_per_side * hidden);

    // linspace(0, K-1, N) — torch semantics. With N==1 the only output is 0.0.
    fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
        if n == 0 { return Vec::new(); }
        if n == 1 { return vec![start]; }
        let step = (end - start) / (n as f32 - 1.0);
        (0..n).map(|i| start + step * i as f32).collect()
    }

    let kmax = (num_grid_per_side - 1) as f32;
    let h_idxs = linspace(0.0, kmax, grid_h);
    let w_idxs = linspace(0.0, kmax, grid_w);

    // Torch `.int()` truncates toward zero; with these non-negative values it
    // is equivalent to floor. Ceil is floor+1 clamped to K-1.
    let mut h_floor = vec![0usize; grid_h];
    let mut h_ceil  = vec![0usize; grid_h];
    let mut dh      = vec![0.0f32; grid_h];
    for (i, &v) in h_idxs.iter().enumerate() {
        let f = v as i32 as usize;
        h_floor[i] = f;
        h_ceil[i] = (f + 1).min(num_grid_per_side - 1);
        dh[i] = v - f as f32;
    }
    let mut w_floor = vec![0usize; grid_w];
    let mut w_ceil  = vec![0usize; grid_w];
    let mut dw      = vec![0.0f32; grid_w];
    for (j, &v) in w_idxs.iter().enumerate() {
        let f = v as i32 as usize;
        w_floor[j] = f;
        w_ceil[j] = (f + 1).min(num_grid_per_side - 1);
        dw[j] = v - f as f32;
    }

    let n = grid_h * grid_w;
    // (grid_h, grid_w, hidden) row-major buffer, accumulator at f32.
    let mut interp = vec![0.0f32; n * hidden];
    for py in 0..grid_h {
        let hf = h_floor[py];
        let hc = h_ceil[py];
        let dhi = dh[py];
        for px in 0..grid_w {
            let wf = w_floor[px];
            let wc = w_ceil[px];
            let dwi = dw[px];

            let w_ff = (1.0 - dhi) * (1.0 - dwi);
            let w_fc = (1.0 - dhi) * dwi;
            let w_cf = dhi * (1.0 - dwi);
            let w_cc = dhi * dwi;

            let out_off = (py * grid_w + px) * hidden;
            let r_ff = (hf * num_grid_per_side + wf) * hidden;
            let r_fc = (hf * num_grid_per_side + wc) * hidden;
            let r_cf = (hc * num_grid_per_side + wf) * hidden;
            let r_cc = (hc * num_grid_per_side + wc) * hidden;
            for d in 0..hidden {
                interp[out_off + d] =
                    w_ff * pos_embed[r_ff + d]
                  + w_fc * pos_embed[r_fc + d]
                  + w_cf * pos_embed[r_cf + d]
                  + w_cc * pos_embed[r_cc + d];
            }
        }
    }

    // Permute to 2x2 spatial-merge-grouped order:
    //   (gy, gx, sy, sx) ↦ patch row-major (py = gy*ms+sy, px = gx*ms+sx).
    let mh = grid_h / merge_size;
    let mw = grid_w / merge_size;
    let mut out = vec![0.0f32; n * hidden];
    let mut out_idx = 0usize;
    for gy in 0..mh {
        for gx in 0..mw {
            for sy in 0..merge_size {
                for sx in 0..merge_size {
                    let py = gy * merge_size + sy;
                    let px = gx * merge_size + sx;
                    let src = (py * grid_w + px) * hidden;
                    let dst = out_idx * hidden;
                    out[dst..dst + hidden].copy_from_slice(&interp[src..src + hidden]);
                    out_idx += 1;
                }
            }
        }
    }
    out
}

/// Compute per-patch cos/sin tables for the vision tower's 2D rotary, in the
/// same 2x2-grouped patch order as `extract_patches`.
///
/// HF reference: `Qwen3_5VisionModel.rot_pos_emb`. The inner rotary dim is
/// `head_dim/2`; frequencies are
///     inv_freq[i] = 1 / theta^(2i / (head_dim/2))    for i in [0, head_dim/4)
/// and the per-patch trig argument is
///     concat( row_idx * inv_freq, col_idx * inv_freq )    → (head_dim/2,)
/// where `row_idx`/`col_idx` are the un-merged (gy*ms+sy, gx*ms+sx) positions.
///
/// We only store the `head_dim/2` half — HF then concatenates `(rope, rope)`
/// along the last dim, so the trig values at `d` and `d + head_dim/2` are
/// identical. The rotary kernel reuses one scalar for both halves.
///
/// `theta` is the rotary base (HF default 10000.0); read from
/// `VisionConfig::rope_theta` at the call site.
///
/// Returns `(cos, sin)` each of length `n * (head_dim/2)`.
pub fn compute_vision_rope_cos_sin(
    grid_h: usize,
    grid_w: usize,
    head_dim: usize,
    merge_size: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim % 4 == 0, "head_dim must be divisible by 4 (got {head_dim})");
    let rot_dim = head_dim / 2;          // total rotary feature width
    let inner   = head_dim / 4;          // = rot_dim / 2 = #frequencies

    // inv_freq[i] = 1 / theta^(2i / rot_dim)  for i in [0, inner)
    let inv_freq: Vec<f32> = (0..inner)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / rot_dim as f32))
        .collect();

    let n = grid_h * grid_w;
    let mut cos_t = vec![0.0f32; n * rot_dim];
    let mut sin_t = vec![0.0f32; n * rot_dim];

    let mh = grid_h / merge_size;
    let mw = grid_w / merge_size;
    let mut out_idx = 0usize;
    for gy in 0..mh {
        for gx in 0..mw {
            for sy in 0..merge_size {
                for sx in 0..merge_size {
                    let row_idx = (gy * merge_size + sy) as f32;
                    let col_idx = (gx * merge_size + sx) as f32;
                    let row_base = out_idx * rot_dim;
                    let col_base = row_base + inner;
                    for i in 0..inner {
                        let angle_r = row_idx * inv_freq[i];
                        let angle_c = col_idx * inv_freq[i];
                        cos_t[row_base + i] = angle_r.cos();
                        sin_t[row_base + i] = angle_r.sin();
                        cos_t[col_base + i] = angle_c.cos();
                        sin_t[col_base + i] = angle_c.sin();
                    }
                    out_idx += 1;
                }
            }
        }
    }
    (cos_t, sin_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure-CPU helper checks. The Python reference values come from
    // `Qwen3_5VisionModel.fast_pos_embed_interpolate` / `rot_pos_emb` —
    // see the doc-comments above for the exact formulas.

    #[test]
    fn pos_embed_interp_identity_when_grid_matches_table() {
        // grid_h == grid_w == num_grid_per_side: every linspace point lands
        // exactly on a table entry, so the bilinear weights collapse to
        // pick a single corner and the interp output is a permutation of
        // the input table — 2x2-grouped.
        let k = 4usize;
        let hidden = 3usize;
        let merge = 2usize;
        // Table: pos_embed[r*k+c, d] = (r*k+c)*10 + d
        let mut table = vec![0.0f32; k * k * hidden];
        for r in 0..k {
            for c in 0..k {
                for d in 0..hidden {
                    table[(r * k + c) * hidden + d] = (r * k + c) as f32 * 10.0 + d as f32;
                }
            }
        }
        let out = fast_pos_embed_interpolate(&table, hidden, k, k, k, merge);
        assert_eq!(out.len(), k * k * hidden);
        // Walk 2x2-grouped: (gy, gx, sy, sx) → table[(gy*2+sy, gx*2+sx)].
        let mut idx = 0usize;
        for gy in 0..(k / merge) {
            for gx in 0..(k / merge) {
                for sy in 0..merge {
                    for sx in 0..merge {
                        let r = gy * merge + sy;
                        let c = gx * merge + sx;
                        for d in 0..hidden {
                            let want = (r * k + c) as f32 * 10.0 + d as f32;
                            let got = out[idx * hidden + d];
                            assert!((got - want).abs() < 1e-4, "idx={idx} d={d} got={got} want={want}");
                        }
                        idx += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn pos_embed_interp_bilinear_midpoint() {
        // 2x2 table → request 3x3 grid. The center sample is at the exact
        // midpoint of all four corners with weight 1/4 each.
        let k = 2usize;
        let hidden = 1usize;
        let merge = 1usize;
        let table = vec![10.0f32, 20.0f32, 30.0f32, 40.0f32]; // [(0,0)=10, (0,1)=20, (1,0)=30, (1,1)=40]
        let out = fast_pos_embed_interpolate(&table, hidden, 3, 3, k, merge);
        // 3x3 in row-major (no merge): center at out[4*hidden+0].
        assert!((out[0] - 10.0).abs() < 1e-4);         // (0,0) → (0,0)
        assert!((out[2] - 20.0).abs() < 1e-4);         // (0,2) → (0,1)
        assert!((out[6] - 30.0).abs() < 1e-4);         // (2,0) → (1,0)
        assert!((out[8] - 40.0).abs() < 1e-4);         // (2,2) → (1,1)
        assert!((out[4] - 25.0).abs() < 1e-4);         // center = mean of all 4
    }

    #[test]
    fn rope_table_concat_row_then_col() {
        // For grid=(2,2), merge=1, head_dim=8 → rot_dim=4, inner=2.
        // The token at (row=1, col=0) should have cos[:inner] = cos(1*inv_freq)
        // and cos[inner:] = cos(0*inv_freq) = 1.
        let head_dim = 8usize;
        let (cos_t, sin_t) = compute_vision_rope_cos_sin(2, 2, head_dim, 1, 10000.0);
        let inner = head_dim / 4;
        let rot = head_dim / 2;
        assert_eq!(cos_t.len(), 4 * rot);
        // Patch (row=1, col=0) is at index gy*2+gx with merge=1 → gy=1, gx=0 → idx 2.
        let base = 2 * rot;
        // Col half should be all 1 / 0 since col_idx=0.
        for i in 0..inner {
            assert!((cos_t[base + inner + i] - 1.0).abs() < 1e-6, "cos col half should be 1");
            assert!(sin_t[base + inner + i].abs() < 1e-6, "sin col half should be 0");
        }
        // Row half (row_idx=1): cos(inv_freq[0]) = cos(1.0) ≈ 0.5403
        let inv0 = 1.0f32; // 10000^0 = 1
        assert!((cos_t[base] - inv0.cos()).abs() < 1e-5);
        assert!((sin_t[base] - inv0.sin()).abs() < 1e-5);
    }

    /// Non-square grid (4×6 ≠ 6×6 table): bilinear interpolation across
    /// rows + h-major outer permutation. Catches an h/w transpose bug in
    /// the 2x2-group permute that the square cases can't see.
    #[test]
    fn pos_embed_interp_identity_non_square() {
        let k = 6usize;
        let hidden = 2usize;
        let merge = 2usize;
        let mut table = vec![0.0f32; k * k * hidden];
        for r in 0..k {
            for c in 0..k {
                for d in 0..hidden {
                    table[(r * k + c) * hidden + d] = (r * 100 + c) as f32 + (d as f32) * 0.5;
                }
            }
        }
        // grid_h=4, grid_w=6 — both ≤ K, both even, distinct so a w/h swap
        // in the permutation would produce a visibly wrong byte at a known
        // out_idx.
        let grid_h = 4usize;
        let grid_w = 6usize;
        let out = fast_pos_embed_interpolate(&table, hidden, grid_h, grid_w, k, merge);
        assert_eq!(out.len(), grid_h * grid_w * hidden);

        // Spot-check: 2x2-grouped patch at (gy=1, gx=2, sy=0, sx=1) is at:
        //   out_idx = gy * (gw * 4) + gx * 4 + sy * 2 + sx
        //   gw = grid_w / 2 = 3 → out_idx = 1*12 + 2*4 + 0*2 + 1 = 21
        // Maps to (py = gy*2+sy, px = gx*2+sx) = (2, 5).
        // linspace(0, K-1, grid_h)[2] for K=6, N=4 → 2 * 5/3 ≈ 3.333
        // linspace(0, K-1, grid_w)[5] for K=6, N=6 → 5.0 exact
        // So we sample a bilinear blend between rows 3 and 4 at column 5.
        let py_lin = 2.0f32 * 5.0 / 3.0;
        let r_floor = py_lin as i32 as usize;
        let r_ceil = (r_floor + 1).min(k - 1);
        let dh = py_lin - r_floor as f32;
        let want = (1.0 - dh) * table[(r_floor * k + 5) * hidden]
                 +        dh  * table[(r_ceil * k + 5) * hidden];
        let got = out[21 * hidden];
        assert!((got - want).abs() < 1e-4, "non-square spot check failed: got={got} want={want}");
    }

    /// Rectangular `compute_vision_rope_cos_sin`: a 2×4 grid sanity check
    /// that distinguishes the row-half vs col-half indexing — col_idx=2
    /// at head_dim=8 (rot_dim=4, inner=2) yields a deterministic table
    /// value the row-half cannot match.
    #[test]
    fn rope_table_rectangular_grid() {
        let head_dim = 8usize;
        let (cos_t, sin_t) = compute_vision_rope_cos_sin(2, 4, head_dim, 1, 10000.0);
        let inner = head_dim / 4;          // 2
        let rot = head_dim / 2;            // 4
        assert_eq!(cos_t.len(), 8 * rot);
        // Patch (row=1, col=2): gy=1, gx=2, sy=0, sx=0 → idx = 1*4 + 2 = 6
        let base = 6 * rot;
        // Row half uses row_idx=1: cos[base..base+inner] = cos(1 * inv_freq[0..inner])
        // inv_freq[0] = 1, inv_freq[1] = 10000^(-2/4) = 0.01
        let inv = [1.0f32, 0.01f32];
        for i in 0..inner {
            assert!((cos_t[base + i] - (1.0 * inv[i]).cos()).abs() < 1e-5);
            assert!((sin_t[base + i] - (1.0 * inv[i]).sin()).abs() < 1e-5);
        }
        // Col half uses col_idx=2:
        for i in 0..inner {
            assert!((cos_t[base + inner + i] - (2.0 * inv[i]).cos()).abs() < 1e-5);
            assert!((sin_t[base + inner + i] - (2.0 * inv[i]).sin()).abs() < 1e-5);
        }
    }

    /// `rope_theta` actually changes the table — defends against a future
    /// accidental hardcode regression in `compute_vision_rope_cos_sin`.
    #[test]
    fn rope_table_honors_theta_param() {
        let (c1, _) = compute_vision_rope_cos_sin(2, 2, 8, 1, 10000.0);
        let (c2, _) = compute_vision_rope_cos_sin(2, 2, 8, 1, 50000.0);
        // Some entry must differ — pick a non-row=0, non-col=0 patch so the
        // angle is non-zero on both halves.
        let any_diff = c1.iter().zip(&c2).any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "theta change must shift at least one trig value");
    }
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
    let x = linear_f16(gpu, &weights.patch_embed_w, &x_patches, &weights.patch_embed_b, h, patch_dim, n)?;
    gpu.free_tensor(x_patches)?;

    // Bilinear-interpolate the learned (K×K, h) pos_embed table down to the
    // actual (grid_h, grid_w) and reorder into 2x2-grouped patch sequence,
    // then add. HF's `fast_pos_embed_interpolate`.
    let num_grid_per_side = (config.num_position_embeddings as f64).sqrt().round() as usize;
    // Hard assertion (not debug_assert): a non-square pos_embed table is a
    // model-config malformation that silently produces wrong indexing in
    // `fast_pos_embed_interpolate` if we round to the nearest int. Fail loud
    // at vision_forward entry instead of producing garbage tokens.
    assert_eq!(
        num_grid_per_side * num_grid_per_side, config.num_position_embeddings,
        "num_position_embeddings ({}) must be a perfect square",
        config.num_position_embeddings,
    );
    let pos_embed_interp = fast_pos_embed_interpolate(
        &weights.pos_embed, h, grid_h, grid_w,
        num_grid_per_side, config.spatial_merge_size,
    );
    let pos_embed_gpu = gpu.upload_f32(&pos_embed_interp, &[n * h])?;
    gpu.add_inplace_f32(&x, &pos_embed_gpu)?;
    gpu.free_tensor(pos_embed_gpu)?;

    // Compute the 2D rotary cos/sin tables once per image and upload. The
    // kernel reads `head_dim/2` floats per token for each of cos/sin (HF's
    // `cat((rope, rope), dim=-1)` makes the two head_dim halves see the same
    // angle, so we store the half only).
    let rot_dim_half = config.head_dim / 2;
    let (rope_cos, rope_sin) = compute_vision_rope_cos_sin(
        grid_h, grid_w, config.head_dim, config.spatial_merge_size, config.rope_theta,
    );
    let rope_cos_gpu = gpu.upload_f32(&rope_cos, &[n * rot_dim_half])?;
    let rope_sin_gpu = gpu.upload_f32(&rope_sin, &[n * rot_dim_half])?;

    // Scratch buffers reused across layers
    let qkv_dim = 3 * h;

    // Stream invariant: every kernel below is enqueued on the same default
    // stream (`gpu.stream_ref()`), and per-layer scratch tensors (`tmp`,
    // `qkv`, `attn_out`, `proj`, `tmp2`, `fc1`, `fc2`) are freed within the
    // same iteration. Correctness depends on submission-order serialization:
    // pool reuse of a freed buffer is fine because the next kernel using
    // that VRAM is queued AFTER the previous one on the same stream. If
    // anyone ever refactors `rdna_compute` to use multiple streams for the
    // vision path (e.g., async memcpy on a side stream), this pattern must
    // be revisited — either add per-layer syncs or attach kernels to the
    // freeing buffer's stream. See review notes in `vision_rev_claude.md`.
    for li in 0..config.num_layers {
        let lw = &weights.layers[li];

        // LayerNorm1 → tmp
        let tmp = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.layernorm_batched(&x, &lw.norm1_w, &lw.norm1_b, &tmp, n, h, config.norm_eps)?;

        // QKV projection → [n, 3h]
        let qkv = linear_f16(gpu, &lw.qkv_w, &tmp, &lw.qkv_b, qkv_dim, h, n)?;
        gpu.free_tensor(tmp)?;

        // 2D rotary applied in-place to Q and K halves of the QKV buffer.
        gpu.apply_rope_2d_vision_f32(
            &qkv, &rope_cos_gpu, &rope_sin_gpu,
            n, h, config.num_heads, config.head_dim,
        )?;

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
    gpu.free_tensor(rope_cos_gpu)?;
    gpu.free_tensor(rope_sin_gpu)?;
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

    // Patches in `normed_data` are stored in 2x2-block-grouped order (see
    // `extract_patches`), so the 4 patches that merge into one output token
    // are CONSECUTIVE in the buffer: src indices [out_idx*4 .. out_idx*4+4].
    let mut merged = vec![0.0f32; n_merged * merge_dim];
    for my in 0..merged_h {
        for mx in 0..merged_w {
            let out_idx = my * merged_w + mx;
            for sub in 0..(sms * sms) {
                let src = out_idx * (sms * sms) + sub;
                merged[out_idx * merge_dim + sub * h..out_idx * merge_dim + sub * h + h]
                    .copy_from_slice(&normed_data[src * h..src * h + h]);
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
