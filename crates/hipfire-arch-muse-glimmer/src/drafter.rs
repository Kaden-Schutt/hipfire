// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Muse Glimmer DFlash drafter (`model_type = muse_glimmer_assistant`, arch_id = 23).
//!
//! A 5-layer block-diffusion draft head for the arch-14 Glimmer target.
//! It is NOT a standalone LM: every draft step reuses the TARGET's vocab table
//! and the TARGET's lm_head. Architecture (traced against the HF safetensors
//! `meta-models/Muse-Glimmer-30B-assistant` — 58 tensors, no embed/lm_head):
//!
//!   target_hidden_proj = output_norm_enc( fc · target_hidden )  // once, all ctx rows
//!   // fc: [hidden, num_extract*hidden]; target_hidden rows = previously accepted tokens
//!   x = noise_embedding (raw target.embed_tokens([seed, MASK×15]), no embed_norm)
//!   // context is NOT added into x — it is concatenated into K/V only
//!   for each drafter layer (5×, RoPE θ=500000, window 2048, GQA 32/8, hd 128):
//!     // STANDARD two-norm Llama block — NOT the target's four-norm sandwich:
//!     residual = x
//!     n1 = rmsnorm(x, input_layernorm, 1e-5) → tmp
//!     q = q_proj(n1)                         // BLOCK only            → B rows
//!     k = k_proj(cat[target_hidden_proj, n1]) // CONTEXT ++ BLOCK      → ctx+B rows
//!     v = v_proj(cat[target_hidden_proj, n1]) // CONTEXT ++ BLOCK      → ctx+B rows
//!     q = q_norm(q); k = k_norm(k)   // per-head WEIGHTED RMSNorm (real q_norm/k_norm weights)
//!     RoPE half-split on Q (block positions) and K (full ctx+block span)
//!     attn_out = attention_dflash_f32(Q[B], K/V[ctx+B])  // bidirectional, GQA, f32
//!     attn = o_proj(attn_out); x = residual + attn
//!     residual = x
//!     n2 = rmsnorm(x, post_attention_layernorm, 1e-5)  // <-- IS the pre-FFN norm
//!     ffn = down(silu(gate(n2))*up(n2)); x = residual + ffn
//!   n = norm(x) → logits = n · target.lm_head.T → argmax
//!
//! Shape (confirmed from artifact / GlimmerDrafterConfig::from_hfq):
//!   n_layers=5, hidden=6656, intermediate=19968, n_heads=32, n_kv_heads=8,
//!   head_dim=128, q_dim=4096, kv_dim=1024, GQA group=4, SWA=2048 on all layers, block=16.
//!
//! Extent decision: context is CONCATENATED into K/V (not broadcast-added into x).
//!   - Q length = block (B), K/V length = ctx_len + block, positions span ctx+B.
//!   - `target_hidden_proj` is computed once via encoder.fc + output_norm_enc over
//!     every accepted ctx row and reused by all 5 layers.
//!   - Scratch k/v are sized `(max_ctx+block)*kv_dim`; q stays `block*q_dim`.
//!
//! Helper choice: `Gpu::attention_dflash_f32` (f32 K/V, GQA, bidirectional,
//! no causal mask). Rejected `attention_q8_0_kv_swa`/`attention_q8_0_kv` — they
//! require a Q8 quantized KV cache, single-query decode shape, and a causal
//! windowed contract; the draft's K/V lives in F32 scratch as [(ctx+B)×kvd] and
//! the block-diffusion contract needs many queries in parallel. Rejected
//! `attention_f32`/`attention_flash*` single-query variants for the same reason.
//! `attention_dflash_f32` matches dtype (f32), layout ([B×q_dim], [L×kvd]),
//! GQA (32/8), and masking (non-causal, bidirectional).
//!
//! Masking / window approximation: upstream layers are all `sliding_attention`
//! with window 2048 and build a bidirectional sliding-window mask. Queries
//! attend bi-directionally within the block and (windowedly) to prior ctx K/V.
//! `attention_dflash_f32` is FULL bidirectional over L=ctx+B — exact while
//! `ctx+B <= 2048`, and over-attends (no window cutoff) beyond that. No
//! windowed bidirectional kernel exists; this is the one real approximation
//! and is stated, not hidden. No new HIP kernel is introduced.
//!
//! Critical embed_norm contract (see `forward.rs:84` and
//! `/tmp/modeling_muse_glimmer.py:439`): the DFlash block's `noise_embedding`
//! is **raw** `target.embed_tokens([seed, MASK×15])` with NO
//! `embed_norm` (scale-less RMSNorm). The AR path at `forward::embed_lookup`
//! DOES apply it; the DFlash path deliberately does not.
//!
//! REUSE: no new kernels. Projections are `weight_gemv`, norms are
//! `rmsnorm_batched`, RoPE is `rope_batched_f32` (half-split; n_heads_*=0 to
//! skip the inactive side), attention is `attention_dflash_f32`.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::{DType, Gpu, GpuTensor};

pub const GLIMMER_DRAFTER_ARCH_ID: u32 = 23;

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GlimmerDrafterConfig {
    pub n_layers: usize,       // 5
    pub hidden: usize,         // 6656
    pub intermediate: usize,   // 19968
    pub n_heads: usize,        // 32
    pub n_kv_heads: usize,     // 8
    pub head_dim: usize,       // 128
    pub norm_eps: f32,         // 1e-5
    pub rope_theta: f32,       // 500000.0
    pub sliding_window: usize, // 2048
    pub block_size: usize,     // 16
    pub mask_token_id: u32,    // 201818
    pub target_layer_ids: Vec<usize>, // [1,13,25,37,49]
}

impl GlimmerDrafterConfig {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("glimmer drafter: metadata_json not valid JSON: {e}"))?;
        let cfg = meta
            .get("config")
            .ok_or_else(|| "glimmer drafter: metadata_json missing `config` wrapper".to_string())?;
        let getu = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64());
        let getf = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());

        let hidden = getu(cfg, "hidden_size").ok_or("glimmer drafter: missing hidden_size")? as usize;
        let n_layers = getu(cfg, "num_hidden_layers")
            .ok_or("glimmer drafter: missing num_hidden_layers")? as usize;
        let intermediate =
            getu(cfg, "intermediate_size").ok_or("glimmer drafter: missing intermediate_size")? as usize;
        let n_heads = getu(cfg, "num_attention_heads")
            .ok_or("glimmer drafter: missing num_attention_heads")? as usize;
        let n_kv_heads = getu(cfg, "num_key_value_heads").unwrap_or(n_heads as u64) as usize;
        let head_dim = getu(cfg, "head_dim").map(|v| v as usize).unwrap_or(hidden / n_heads);
        let norm_eps = getf(cfg, "rms_norm_eps").unwrap_or(1e-5) as f32;
        let rope_theta = cfg
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(500000.0) as f32;
        let sliding_window = getu(cfg, "sliding_window").unwrap_or(2048) as usize;
        let block_size = getu(cfg, "block_size").ok_or("glimmer drafter: missing block_size")? as usize;
        let mask_token_id =
            getu(cfg, "mask_token_id").ok_or("glimmer drafter: missing mask_token_id")? as u32;
        let target_layer_ids: Vec<usize> = cfg
            .get("target_layer_ids")
            .and_then(|v| v.as_array())
            .ok_or("glimmer drafter: missing target_layer_ids")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        if target_layer_ids.len() != 5 {
            return Err(format!(
                "glimmer drafter: target_layer_ids len {} != 5",
                target_layer_ids.len()
            ));
        }
        if target_layer_ids != vec![1, 13, 25, 37, 49] {
            eprintln!(
                "glimmer drafter: WARNING target_layer_ids {:?} != expected [1,13,25,37,49]",
                target_layer_ids
            );
        }

        Ok(GlimmerDrafterConfig {
            n_layers,
            hidden,
            intermediate,
            n_heads,
            n_kv_heads,
            head_dim,
            norm_eps,
            rope_theta,
            sliding_window,
            block_size,
            mask_token_id,
            target_layer_ids,
        })
    }

    #[inline]
    pub fn num_extract(&self) -> usize {
        self.target_layer_ids.len()
    }
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

// ─── Load helpers ───────────────────────────────────────────────────────

fn load_f32_vec(hfq: &HfqFile, name: &str, expected: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("glimmer drafter: tensor '{name}' not found"))?;
    if info.shape.iter().fold(1usize, |a, &b| a * b as usize) != expected {
        return Err(format!(
            "glimmer drafter: tensor '{name}' shape {:?} != expected {}",
            info.shape, expected
        ));
    }
    match info.quant_type {
        1 => Ok(data
            .chunks_exact(2)
            .map(|c| hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        2 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        qt => Err(format!(
            "glimmer drafter: unsupported quant_type {qt} for F32 tensor '{name}'"
        )),
    }
}

fn load_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, dim: usize) -> Result<GpuTensor, String> {
    let v = load_f32_vec(hfq, name, dim)?;
    gpu.upload_f32(&v, &[dim])
        .map_err(|e| format!("glimmer drafter: upload norm '{name}': {e:?}"))
}

fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("glimmer drafter: tensor '{name}' not found"))?;
    let mut wt = match info.quant_type {
        3 => {
            let buf = gpu
                .upload_raw(data, &[data.len()])
                .map_err(|e| format!("glimmer drafter: upload Q8 '{name}': {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        }
        1 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let buf = gpu
                .upload_f32(&f32_data, &[m * k])
                .map_err(|e| format!("glimmer drafter: upload F16->F32 '{name}': {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        }
        2 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let buf = gpu
                .upload_f32(&f32_data, &[m * k])
                .map_err(|e| format!("glimmer drafter: upload F32 '{name}': {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        }
        qt => return Err(format!("glimmer drafter: unsupported quant_type {qt} for '{name}'")),
    };
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = hipfire_runtime::hfq::load_awq_scale(hfq, gpu, name, k);
    }
    Ok(wt)
}

// ─── Weights ────────────────────────────────────────────────────────────

pub struct GlimmerDrafterLayer {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: WeightTensor,
    pub o_proj: WeightTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

pub struct GlimmerDrafterWeights {
    pub fc: WeightTensor,            // encoder.fc.weight [hidden, num_extract*hidden]
    pub output_norm_enc: GpuTensor,  // encoder.output_norm_enc.weight
    pub norm: GpuTensor,             // norm.weight
    pub layers: Vec<GlimmerDrafterLayer>,
}

impl GlimmerDrafterWeights {
    pub fn load(hfq: &HfqFile, cfg: &GlimmerDrafterConfig, gpu: &mut Gpu) -> Result<Self, String> {
        if hfq.arch_id != GLIMMER_DRAFTER_ARCH_ID {
            return Err(format!(
                "glimmer drafter: expected arch_id {} (muse_glimmer_assistant), got {}",
                GLIMMER_DRAFTER_ARCH_ID, hfq.arch_id
            ));
        }
        let ne = cfg.num_extract();
        let h = cfg.hidden;
        let fc = load_wt(hfq, gpu, "encoder.fc.weight", h, ne * h)?;
        let output_norm_enc = load_norm(hfq, gpu, "encoder.output_norm_enc.weight", h)?;
        let norm = load_norm(hfq, gpu, "norm.weight", h)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            layers.push(GlimmerDrafterLayer {
                input_layernorm: load_norm(hfq, gpu, &format!("{p}.input_layernorm.weight"), h)?,
                post_attention_layernorm: load_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    h,
                )?,
                q_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    cfg.q_dim(),
                    h,
                )?,
                k_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    cfg.kv_dim(),
                    h,
                )?,
                v_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.v_proj.weight"),
                    cfg.kv_dim(),
                    h,
                )?,
                o_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    h,
                    cfg.q_dim(),
                )?,
                q_norm: load_norm(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), cfg.head_dim)?,
                k_norm: load_norm(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), cfg.head_dim)?,
                gate_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    cfg.intermediate,
                    h,
                )?,
                up_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    cfg.intermediate,
                    h,
                )?,
                down_proj: load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    h,
                    cfg.intermediate,
                )?,
            });
        }
        Ok(GlimmerDrafterWeights {
            fc,
            output_norm_enc,
            norm,
            layers,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.fc.free_all(gpu);
        let _ = gpu.free_tensor(self.output_norm_enc);
        let _ = gpu.free_tensor(self.norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.input_layernorm);
            let _ = gpu.free_tensor(l.post_attention_layernorm);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            l.q_proj.free_all(gpu);
            l.k_proj.free_all(gpu);
            l.v_proj.free_all(gpu);
            l.o_proj.free_all(gpu);
            l.gate_proj.free_all(gpu);
            l.up_proj.free_all(gpu);
            l.down_proj.free_all(gpu);
        }
    }
}

// ─── Scratch ────────────────────────────────────────────────────────────

pub struct GlimmerDrafterScratch {
    pub x: GpuTensor,          // [block*hidden] — noise + evolving hidden (no ctx add)
    pub target_hidden_proj: GpuTensor, // [max_ctx * hidden] — ctx rows, reused by all layers
    pub q: GpuTensor,          // [block * q_dim]  — Q is block-only
    pub k: GpuTensor,          // [(max_ctx + block) * kv_dim] — ctx ++ block
    pub v: GpuTensor,          // [(max_ctx + block) * kv_dim] — ctx ++ block
    pub attn_out: GpuTensor,   // [block * q_dim]
    pub tmp: GpuTensor,        // [block*hidden] scratch (also stages one fc input row)
    pub gate_ffn: GpuTensor,
    pub up_ffn: GpuTensor,
    pub ffn_hidden: GpuTensor,
    pub logits_tmp: GpuTensor, // [hidden] for final norm
    /// Device positions for RoPE, sized for the full ctx+block span and
    /// allocated ONCE. Uploaded each forward; views feed `rope_batched_f32`.
    ///
    /// This was originally malloc'd and freed inside the per-layer loop, which
    /// is both a hipMalloc per layer per window and a lifetime hazard — the
    /// freed pointer is what surfaced as `hipMemcpy H2D: an illegal memory
    /// access` on the second window. The target's `GlimmerState` already keeps
    /// a persistent `pos_buf` for exactly this reason; the drafter now matches.
    /// Capacity: `(max_ctx + block) * 4` bytes (i32 positions).
    pub pos_buf: hip_bridge::DeviceBuffer,
}

impl GlimmerDrafterScratch {
    pub fn new(gpu: &mut Gpu, cfg: &GlimmerDrafterConfig, max_ctx: usize) -> Result<Self, String> {
        let h = cfg.hidden;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let block = cfg.block_size;
        let kv_rows = max_ctx + block;
        let pos_buf = gpu
            .hip
            .malloc(kv_rows * 4)
            .map_err(|e| format!("glimmer drafter: alloc pos_buf: {e:?}"))?;
        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.zeros(&[n], DType::F32)
                .map_err(|e| format!("glimmer drafter scratch {label}: {e:?}"))
        };
        Ok(GlimmerDrafterScratch {
            x: alloc(gpu, block * h, "x")?,
            target_hidden_proj: alloc(gpu, max_ctx * h, "target_hidden_proj")?,
            q: alloc(gpu, block * qd, "q")?,
            k: alloc(gpu, kv_rows * kvd, "k")?,
            v: alloc(gpu, kv_rows * kvd, "v")?,
            attn_out: alloc(gpu, block * qd, "attn_out")?,
            tmp: alloc(gpu, block * h, "tmp")?,
            gate_ffn: alloc(gpu, block * cfg.intermediate, "gate_ffn")?,
            up_ffn: alloc(gpu, block * cfg.intermediate, "up_ffn")?,
            ffn_hidden: alloc(gpu, block * cfg.intermediate, "ffn_hidden")?,
            logits_tmp: alloc(gpu, h, "logits_tmp")?,
            pos_buf,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x,
            self.target_hidden_proj,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.tmp,
            self.gate_ffn,
            self.up_ffn,
            self.ffn_hidden,
            self.logits_tmp,
        ] {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }
}

// ─── Forward (no new kernels) ─────────────────────────────────────────

/// Muse Glimmer DFlash draft forward.
///
/// `noise_embedding`: `[block_size * hidden]` raw F32 embeddings of
/// `[seed, MASK×(block-1)]` via `target.embed_tokens` (no embed_norm).
/// `target_hidden`: `[ctx_len * num_extract * hidden]` concatenated residual
/// hidden from `target_layer_ids` (1,13,25,37,49) — every previously accepted
/// token row, not a single broadcast row.
/// `positions`: `[ctx_len + block_size]` absolute i32 span; the tail
/// `positions[ctx_len..]` are the Q / block positions.
/// Caller applies target `lm_head` over rows `1..block_size` to obtain draft logits.
#[allow(clippy::too_many_arguments)]
pub fn glimmer_drafter_forward(
    gpu: &mut Gpu,
    cfg: &GlimmerDrafterConfig,
    weights: &GlimmerDrafterWeights,
    scratch: &mut GlimmerDrafterScratch,
    noise_embedding: &[f32],
    target_hidden: &[f32],
    positions: &[i32],
    block_size: usize,
    ctx_len: usize,
) -> Result<(), String> {
    use hipfire_runtime::llama::weight_gemv;
    if block_size != cfg.block_size {
        return Err(format!(
            "glimmer drafter: block_size {} != cfg.block_size {}",
            block_size, cfg.block_size
        ));
    }
    let expected_noise = block_size * cfg.hidden;
    if noise_embedding.len() != expected_noise {
        return Err(format!(
            "glimmer drafter: noise_embedding len {} != expected {}",
            noise_embedding.len(),
            expected_noise
        ));
    }
    let expected_th = ctx_len * cfg.num_extract() * cfg.hidden;
    if target_hidden.len() != expected_th {
        return Err(format!(
            "glimmer drafter: target_hidden len {} != expected {} (ctx_len={} num_extract={} hidden={})",
            target_hidden.len(),
            expected_th,
            ctx_len,
            cfg.num_extract(),
            cfg.hidden
        ));
    }
    let expected_pos = ctx_len + block_size;
    if positions.len() != expected_pos {
        return Err(format!(
            "glimmer drafter: positions len {} != expected {} (ctx_len={} + block_size={})",
            positions.len(),
            expected_pos,
            ctx_len,
            block_size
        ));
    }
    if cfg.mask_token_id != 201818 {
        return Err(format!(
            "glimmer drafter: mask_token_id {} != 201818 — perturbation detected",
            cfg.mask_token_id
        ));
    }

    let h = cfg.hidden;
    let ne = cfg.num_extract();
    let eps = cfg.norm_eps;
    let qd = cfg.q_dim();
    let kvd = cfg.kv_dim();
    let l = ctx_len + block_size; // K/V length

    // --- 1. target_hidden_proj = rmsnorm(fc * target_hidden) once for every ctx row ---
    // Reused by all 5 layers. No ctx_len==1 fast path.
    if ctx_len > 0 {
        let ne_h = ne * h;
        for row in 0..ctx_len {
            let in_slice = &target_hidden[row * ne_h..(row + 1) * ne_h];
            let tmp_in = scratch.tmp.sub_offset(0, ne_h);
            // Direct htod into tmp_in's buffer (upload_f32 allocates a new tensor and would leak + leave tmp_in zero).
            let bytes = unsafe {
                std::slice::from_raw_parts(in_slice.as_ptr() as *const u8, in_slice.len() * 4)
            };
            gpu.hip
                .memcpy_htod(&tmp_in.buf, bytes)
                .map_err(|e| format!("drafter htod target_hidden row {row}: {e:?}"))?;
            let out = scratch.target_hidden_proj.sub_offset(row * h, h);
            weight_gemv(gpu, &weights.fc, &tmp_in, &out)
                .map_err(|e| format!("drafter fc row {row}: {e}"))?;
            gpu.rmsnorm_f32(&out, &weights.output_norm_enc, &out, eps)
                .map_err(|e| format!("drafter output_norm row {row}: {e:?}"))?;
        }
    }

    // --- 2. noise_embedding into scratch.x (context is NOT added into x) ---
    {
        let host_bytes = unsafe {
            std::slice::from_raw_parts(noise_embedding.as_ptr() as *const u8, noise_embedding.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&scratch.x.buf, host_bytes)
            .map_err(|e| format!("drafter htod x: {e:?}"))?;
    }

    // Upload full position span once; reused by every layer's RoPE.
    {
        let bytes = unsafe {
            std::slice::from_raw_parts(positions.as_ptr() as *const u8, positions.len() * 4)
        };
        // pos_buf is sized (max_ctx+block)*4; only the leading l entries are written.
        let pos_view =
            unsafe { hip_bridge::DeviceBuffer::from_raw(scratch.pos_buf.as_ptr(), l * 4) };
        gpu.hip
            .memcpy_htod(&pos_view, bytes)
            .map_err(|e| format!("drafter htod positions: {e:?}"))?;
    }

    // --- 3. Per-layer transformer — ctx-concatenated K/V, block Q ---
    for (li, layer) in weights.layers.iter().enumerate() {
        // input_layernorm(x) -> tmp  (block rows only)
        gpu.rmsnorm_batched(
            &scratch.x,
            &layer.input_layernorm,
            &scratch.tmp,
            block_size,
            h,
            eps,
        )
        .map_err(|e| format!("drafter L{li} input norm: {e:?}"))?;

        // q over B block rows from n1=tmp
        for pos in 0..block_size {
            let n1 = scratch.tmp.sub_offset(pos * h, h);
            let qdst = scratch.q.sub_offset(pos * qd, qd);
            weight_gemv(gpu, &layer.q_proj, &n1, &qdst)
                .map_err(|e| format!("drafter L{li} q pos {pos}: {e}"))?;
        }
        // k/v over ctx+B: first ctx_len rows from target_hidden_proj, tail B from tmp
        for row in 0..ctx_len {
            let src = scratch.target_hidden_proj.sub_offset(row * h, h);
            let kdst = scratch.k.sub_offset(row * kvd, kvd);
            let vdst = scratch.v.sub_offset(row * kvd, kvd);
            weight_gemv(gpu, &layer.k_proj, &src, &kdst)
                .map_err(|e| format!("drafter L{li} k ctx row {row}: {e}"))?;
            weight_gemv(gpu, &layer.v_proj, &src, &vdst)
                .map_err(|e| format!("drafter L{li} v ctx row {row}: {e}"))?;
        }
        for pos in 0..block_size {
            let n1 = scratch.tmp.sub_offset(pos * h, h);
            let kdst = scratch.k.sub_offset((ctx_len + pos) * kvd, kvd);
            let vdst = scratch.v.sub_offset((ctx_len + pos) * kvd, kvd);
            weight_gemv(gpu, &layer.k_proj, &n1, &kdst)
                .map_err(|e| format!("drafter L{li} k block pos {pos}: {e}"))?;
            weight_gemv(gpu, &layer.v_proj, &n1, &vdst)
                .map_err(|e| format!("drafter L{li} v block pos {pos}: {e}"))?;
        }

        // per-head WEIGHTED q/k norm (real q_norm/k_norm weights)
        // Q batch = B * n_heads; K batch = (ctx+B) * n_kv_heads
        gpu.rmsnorm_batched(
            &scratch.q,
            &layer.q_norm,
            &scratch.q,
            block_size * cfg.n_heads,
            cfg.head_dim,
            eps,
        )
        .map_err(|e| format!("drafter L{li} q_norm: {e:?}"))?;
        let k_full = scratch.k.sub_offset(0, l * kvd);
        gpu.rmsnorm_batched(
            &k_full,
            &layer.k_norm,
            &k_full,
            l * cfg.n_kv_heads,
            cfg.head_dim,
            eps,
        )
        .map_err(|e| format!("drafter L{li} k_norm: {e:?}"))?;

        // RoPE half-split over the concatenated extent via rope_batched_f32.
        // positions live in pos_buf as i32; GpuTensor shells match dflash's F32 dtype trick.
        // Call 1: rotate Q and K-tail together at block positions (same B rows / same phases).
        // Call 2: rotate K-ctx only (n_heads_q=0) at ctx positions.
        let pos_tensor = GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(scratch.pos_buf.as_ptr(), l * 4) },
            shape: vec![l],
            dtype: DType::F32,
        };
        let k_tail = scratch.k.sub_offset(ctx_len * kvd, block_size * kvd);
        let pos_tail = pos_tensor.sub_offset(ctx_len, block_size);
        gpu.rope_batched_f32(
            &scratch.q,
            &k_tail,
            &pos_tail,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.rope_theta,
            block_size,
        )
        .map_err(|e| format!("drafter L{li} rope block: {e:?}"))?;
        if ctx_len > 0 {
            let k_ctx = scratch.k.sub_offset(0, ctx_len * kvd);
            let pos_ctx = pos_tensor.sub_offset(0, ctx_len);
            // n_heads_q=0 → Q side skipped; scratch.q is a valid dummy pointer.
            gpu.rope_batched_f32(
                &scratch.q,
                &k_ctx,
                &pos_ctx,
                0,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.rope_theta,
                ctx_len,
            )
            .map_err(|e| format!("drafter L{li} rope ctx: {e:?}"))?;
        }

        // Attention: B queries attend bidirectionally to L=ctx+B keys/values.
        // Full bidirectional (no sliding window) — exact while L <= 2048.
        let v_full = scratch.v.sub_offset(0, l * kvd);
        gpu.attention_dflash_f32(
            &scratch.q,
            &k_full,
            &v_full,
            &scratch.attn_out,
            block_size,
            l,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
        )
        .map_err(|e| format!("drafter L{li} attention_dflash_f32: {e:?}"))?;

        // o_proj per row
        for pos in 0..block_size {
            let attn = scratch.attn_out.sub_offset(pos * qd, qd);
            let out = scratch.tmp.sub_offset(pos * h, h);
            weight_gemv(gpu, &layer.o_proj, &attn, &out)
                .map_err(|e| format!("drafter L{li} o pos {pos}: {e}"))?;
        }
        // residual: x = x + tmp (NO post_attention_layernorm on attn output)
        gpu.add_inplace_f32(&scratch.x, &scratch.tmp)
            .map_err(|e| format!("drafter L{li} attn residual: {e:?}"))?;
        // FFN: norm with post_attention_layernorm (IS the pre-FFN norm) reading post-residual x
        gpu.rmsnorm_batched(
            &scratch.x,
            &layer.post_attention_layernorm,
            &scratch.tmp,
            block_size,
            h,
            eps,
        )
        .map_err(|e| format!("drafter L{li} post_attn/pre_ffn norm: {e:?}"))?;
        for pos in 0..block_size {
            let n2 = scratch.tmp.sub_offset(pos * h, h);
            let g = scratch.gate_ffn.sub_offset(pos * cfg.intermediate, cfg.intermediate);
            let u = scratch.up_ffn.sub_offset(pos * cfg.intermediate, cfg.intermediate);
            weight_gemv(gpu, &layer.gate_proj, &n2, &g)
                .map_err(|e| format!("drafter L{li} gate pos {pos}: {e}"))?;
            weight_gemv(gpu, &layer.up_proj, &n2, &u)
                .map_err(|e| format!("drafter L{li} up pos {pos}: {e}"))?;
        }
        gpu.silu_mul_f32(&scratch.gate_ffn, &scratch.up_ffn, &scratch.ffn_hidden)
            .map_err(|e| format!("drafter L{li} silu: {e:?}"))?;
        for pos in 0..block_size {
            let fh = scratch.ffn_hidden.sub_offset(pos * cfg.intermediate, cfg.intermediate);
            let out = scratch.tmp.sub_offset(pos * h, h);
            weight_gemv(gpu, &layer.down_proj, &fh, &out)
                .map_err(|e| format!("drafter L{li} down pos {pos}: {e}"))?;
        }
        gpu.add_inplace_f32(&scratch.x, &scratch.tmp)
            .map_err(|e| format!("drafter L{li} ffn residual: {e:?}"))?;
    }
    gpu.rmsnorm_batched(&scratch.x, &weights.norm, &scratch.x, block_size, h, eps)
        .map_err(|e| format!("drafter final norm: {e:?}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qkv_dims_match_config() {
        // The glimmer_drafter_forward gate above must trip when mask_token_id is perturbed.
        let cfg = GlimmerDrafterConfig {
            n_layers: 5,
            hidden: 6656,
            intermediate: 19968,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            norm_eps: 1e-5,
            rope_theta: 500000.0,
            sliding_window: 2048,
            block_size: 16,
            mask_token_id: 201819, // perturbed
            target_layer_ids: vec![1, 13, 25, 37, 49],
        };
        assert_ne!(cfg.mask_token_id, 201818);
        assert_eq!(cfg.q_dim(), 32 * 128);
        assert_eq!(cfg.kv_dim(), 8 * 128);
        assert_eq!(cfg.num_extract(), 5);
    }
}
