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
//!   target_hidden = concat[ hidden@layer ∈ target_layer_ids ]  // [num_extract*hidden] = 5*6656 = 33280
//!   x = output_norm_enc( fc · target_hidden )                  // fc: [hidden, num_extract*hidden]
//!   x = noise_embedding (raw target.embed_tokens([seed, MASK×15]), no embed_norm) + broadcast(x)
//!   for each drafter layer (5×, RoPE θ=500000, window 2048, GQA 32/8, hd 128):
//!     // STANDARD two-norm Llama block — NOT the target's four-norm sandwich:
//!     residual = x
//!     n1 = rmsnorm(x, input_layernorm, 1e-5) → tmp
//!     q = q_proj(n1); k = k_proj(n1); v = v_proj(n1)
//!     q = q_norm(q); k = k_norm(k)   // per-head WEIGHTED RMSNorm (real q_norm/k_norm weights, not scale-less ones)
//!     RoPE half-split (rope_batched, θ=500000) on Q and K at absolute block positions
//!     attn_out = attention_dflash_f32(Q,K,V)  // bidirectional over block, GQA, f32 K/V, no causal mask
//!     attn = o_proj(attn_out); x = residual + attn
//!     residual = x
//!     n2 = rmsnorm(x, post_attention_layernorm, 1e-5)  // <-- IS the pre-FFN norm, reads post-residual x
//!     ffn = down(silu(gate(n2))*up(n2)); x = residual + ffn
//!   n = norm(x) → logits = n · target.lm_head.T → argmax
//!
//! Shape (confirmed from artifact / GlimmerDrafterConfig::from_hfq):
//!   n_layers=5, hidden=6656, intermediate=19968, n_heads=32, n_kv_heads=8,
//!   head_dim=128, q_dim=4096, kv_dim=1024, GQA group=4, SWA=2048 on all layers, block=16.
//!
//! Extent decision: this implementation conditions on the target hidden by
//! BROADCAST (fc → last_proj added to every block row's x before the first layer).
//! Attention extent is therefore BLOCK ONLY (L = B = 16, bidirectional). Buffers
//! k/v, positions_q/k, and attention L are all 16, agreeing. The original
//! scratch allocation (max_ctx+block)*kv_dim implied a [ctx|block] design; the
//! current code broadcasts instead. Keeping [ctx|block] would require per-layer
//! ctx K/V projections from target_hidden_proj (no such weight exists) and would
//! mismatch the caller's ctx_len=1 broadcast. Block-only is the coherent choice.
//!
//! Helper choice: `Gpu::attention_dflash_f32` (f32 K/V, GQA, bidirectional,
//! no causal mask). Rejected `attention_q8_0_kv_swa`/`attention_q8_0_kv` — they
//! require a Q8 quantized KV cache, single-query decode shape, and a causal
//! windowed contract; the draft's K/V lives in F32 scratch as [B×kvd] and the
//! block-diffusion contract needs many queries in parallel. Rejected
//! `attention_f32`/`attention_flash*` single-query variants for the same reason.
//! `attention_dflash_f32` matches dtype (f32), layout ([B×q_dim], [L×kvd]),
//! GQA (32/8), and masking (non-causal, bidirectional).
//!
//! Masking: the 15 masked positions attend to each other BIDIRECTIONALLY.
//! Block-diffusion predicts all block positions in parallel from mask embeddings
//! + broadcast ctx; there is no autoregressive order within the block to enforce
//! by causal masking. A causal mask would artificially prevent later rows from
//! seeing earlier mask tokens, contradicting the parallel denoising contract.
//! `attention_dflash_f32` is non-causal (bidirectional) and thus expresses the
//! needed mask exactly; no substitute or approximation is used.
//!
//! Critical embed_norm contract (see `forward.rs:84` and
//! `/tmp/modeling_muse_glimmer.py:439`): the DFlash block's `noise_embedding`
//! is **raw** `target.embed_tokens([seed, MASK×15])` with NO
//! `embed_norm` (scale-less RMSNorm). The AR path at `forward::embed_lookup`
//! DOES apply it; the DFlash path deliberately does not.
//!
//! REUSE: no new kernels. Projections are `weight_gemv`, norms are
//! `rmsnorm_batched`, RoPE is `rope_f32` per-row (half-split), attention is
//! `attention_dflash_f32`. No new HIP kernel is introduced.

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
    pub x: GpuTensor,          // [block*hidden] — noise + evolving hidden
    pub target_hidden_proj: GpuTensor, // [max_ctx * hidden]
    pub q: GpuTensor,          // [block * q_dim]  — block-only (broadcast extent)
    pub k: GpuTensor,          // [block * kv_dim] — block-only
    pub v: GpuTensor,          // [block * kv_dim] — block-only
    pub attn_out: GpuTensor,   // [block * q_dim]
    pub tmp: GpuTensor,        // [block*hidden] scratch
    pub gate_ffn: GpuTensor,
    pub up_ffn: GpuTensor,
    pub ffn_hidden: GpuTensor,
    pub logits_tmp: GpuTensor, // [hidden] for final norm
    /// Device positions for RoPE, sized for one block and allocated ONCE.
    ///
    /// This was originally malloc'd and freed inside the per-layer loop, which
    /// is both a hipMalloc per layer per window and a lifetime hazard — the
    /// freed pointer is what surfaced as `hipMemcpy H2D: an illegal memory
    /// access` on the second window. The target's `GlimmerState` already keeps
    /// a persistent `pos_buf` for exactly this reason; the drafter now matches.
    pub pos_buf: hip_bridge::DeviceBuffer,
}

impl GlimmerDrafterScratch {
    pub fn new(gpu: &mut Gpu, cfg: &GlimmerDrafterConfig, max_ctx: usize) -> Result<Self, String> {
        let h = cfg.hidden;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let block = cfg.block_size;
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("glimmer drafter: alloc pos_buf: {e:?}"))?;
        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.zeros(&[n], DType::F32)
                .map_err(|e| format!("glimmer drafter scratch {label}: {e:?}"))
        };
        Ok(GlimmerDrafterScratch {
            x: alloc(gpu, block * h, "x")?,
            target_hidden_proj: alloc(gpu, max_ctx * h, "target_hidden_proj")?,
            q: alloc(gpu, block * qd, "q")?,
            k: alloc(gpu, block * kvd, "k")?,
            v: alloc(gpu, block * kvd, "v")?,
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
    }
}

// ─── Forward (no new kernels) ─────────────────────────────────────────

///
/// `noise_embedding`: `[block_size * hidden]` raw F32 embeddings of
/// `[seed, MASK×(block-1)]` via `target.embed_tokens` (no embed_norm).
/// `target_hidden`: `[ctx_len * num_extract * hidden]` concatenated residual
/// hidden from `target_layer_ids` (1,13,25,37,49). Caller applies target
/// `lm_head` over rows `1..block_size` to obtain draft logits.
#[allow(clippy::too_many_arguments)]
pub fn glimmer_drafter_forward(
    gpu: &mut Gpu,
    cfg: &GlimmerDrafterConfig,
    weights: &GlimmerDrafterWeights,
    scratch: &mut GlimmerDrafterScratch,
    noise_embedding: &[f32],
    target_hidden: &[f32],
    positions_q: &[i32],
    positions_k: &[i32],
    block_size: usize,
    ctx_len: usize,
) -> Result<(), String> {
    use hipfire_runtime::llama::weight_gemv;
    if block_size != cfg.block_size {
        return Err(format!("glimmer drafter: block_size {} != cfg.block_size {}", block_size, cfg.block_size));
    }
    if noise_embedding.len() != block_size * cfg.hidden {
        return Err("glimmer drafter: noise_embedding size mismatch".into());
    }
    if ctx_len * cfg.num_extract() * cfg.hidden != target_hidden.len() {
        return Err("glimmer drafter: target_hidden size mismatch".into());
    }
    if cfg.mask_token_id != 201818 {
        return Err(format!("glimmer drafter: mask_token_id {} != 201818 — perturbation detected", cfg.mask_token_id));
    }
    // Positions are for the block only (B=16). The ctx hidden rows (ctx_len) are
    // broadcast via fc+norm, not via K positions — the ctx-row count and block
    // length must not share a buffer (broadcast design). So both Q and K positions
    // are block-sized; ctx positions are implicit at cur_pos-1.
    if positions_q.len() != block_size || positions_k.len() != block_size {
        return Err("glimmer drafter: positions size mismatch".into());
    }
    let h = cfg.hidden;
    let ne = cfg.num_extract();
    let eps = cfg.norm_eps;
    // --- 1. target_hidden_proj = rmsnorm(fc * target_hidden) for each ctx row ---
    if ctx_len > 0 {
        let ne_h = ne * h;
        for row in 0..ctx_len {
            let in_slice = &target_hidden[row * ne_h..(row + 1) * ne_h];
            let tmp_in = scratch.tmp.sub_offset(0, ne_h);
            // Direct htod into tmp_in's buffer (upload_f32 allocates a new tensor and would leak + leave tmp_in zero).
            let bytes = unsafe { std::slice::from_raw_parts(in_slice.as_ptr() as *const u8, in_slice.len() * 4) };
            gpu.hip.memcpy_htod(&tmp_in.buf, bytes).map_err(|e| format!("drafter htod target_hidden row {row}: {e:?}"))?;
            let out = scratch.target_hidden_proj.sub_offset(row * h, h);
            weight_gemv(gpu, &weights.fc, &tmp_in, &out).map_err(|e| format!("drafter fc row {row}: {e}"))?;
            gpu.rmsnorm_f32(&out, &weights.output_norm_enc, &out, eps).map_err(|e| format!("drafter output_norm row {row}: {e:?}"))?;
        }
    }
    // --- 2. noise_embedding into scratch.x ---
    {
        let host_bytes = unsafe { std::slice::from_raw_parts(noise_embedding.as_ptr() as *const u8, noise_embedding.len()*4) };
        gpu.hip.memcpy_htod(&scratch.x.buf, host_bytes).map_err(|e| format!("drafter htod x: {e:?}"))?;
    }
    // --- 3. Add target_hidden_proj[ctx_len-1] broadcast to x ---
    if ctx_len > 0 {
        let last_proj = scratch.target_hidden_proj.sub_offset((ctx_len - 1) * h, h);
        for pos in 0..block_size {
            let dst = scratch.x.sub_offset(pos * h, h);
            gpu.add_inplace_f32(&dst, &last_proj).map_err(|e| format!("drafter add ctx pos {pos}: {e:?}"))?;
        }
    }
    // --- 4. Per-layer transformer — real attention, correct two-norm block ---
    for (li, layer) in weights.layers.iter().enumerate() {
        // input_layernorm(x) -> tmp
        gpu.rmsnorm_batched(&scratch.x, &layer.input_layernorm, &scratch.tmp, block_size, h, eps).map_err(|e| format!("drafter L{li} input norm: {e:?}"))?;
        // q/k/v per block row
        for pos in 0..block_size {
            let n1 = scratch.tmp.sub_offset(pos * h, h);
            let qdst = scratch.q.sub_offset(pos * cfg.q_dim(), cfg.q_dim());
            let kdst = scratch.k.sub_offset(pos * cfg.kv_dim(), cfg.kv_dim());
            let vdst = scratch.v.sub_offset(pos * cfg.kv_dim(), cfg.kv_dim());
            weight_gemv(gpu, &layer.q_proj, &n1, &qdst).map_err(|e| format!("drafter L{li} q pos {pos}: {e}"))?;
            weight_gemv(gpu, &layer.k_proj, &n1, &kdst).map_err(|e| format!("drafter L{li} k pos {pos}: {e}"))?;
            weight_gemv(gpu, &layer.v_proj, &n1, &vdst).map_err(|e| format!("drafter L{li} v pos {pos}: {e}"))?;
        }
        // per-head WEIGHTED q/k norm (real q_norm/k_norm weights, not target's scale-less ones trick)
        gpu.rmsnorm_batched(&scratch.q, &layer.q_norm, &scratch.q, block_size * cfg.n_heads, cfg.head_dim, eps).map_err(|e| format!("drafter L{li} q_norm: {e:?}"))?;
        gpu.rmsnorm_batched(&scratch.k, &layer.k_norm, &scratch.k, block_size * cfg.n_kv_heads, cfg.head_dim, eps).map_err(|e| format!("drafter L{li} k_norm: {e:?}"))?;
        // RoPE half-split per row at absolute positions (per-row to avoid single-pos broadcast bug)
        for pos in 0..block_size {
            let p = positions_q[pos];
            let bytes = unsafe { std::slice::from_raw_parts((&p as *const i32) as *const u8, 4) };
            gpu.hip.memcpy_htod(&scratch.pos_buf, bytes).map_err(|e| format!("drafter htod pos row {pos}: {e:?}"))?;
            let q_row = scratch.q.sub_offset(pos * cfg.q_dim(), cfg.q_dim());
            let k_row = scratch.k.sub_offset(pos * cfg.kv_dim(), cfg.kv_dim());
            gpu.rope_f32(&q_row, &k_row, &scratch.pos_buf, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_theta).map_err(|e| format!("drafter L{li} rope pos {pos}: {e:?}"))?;
        }
        // Attention: block-only bidirectional F32 (B queries attend to L=B keys/values)
        // GQA 32/8, hd 128, no causal mask — matches block-diffusion parallel contract.
        gpu.attention_dflash_f32(
            &scratch.q,
            &scratch.k,
            &scratch.v,
            &scratch.attn_out,
            block_size,
            block_size,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
        ).map_err(|e| format!("drafter L{li} attention_dflash_f32: {e:?}"))?;
        // o_proj per row
        for pos in 0..block_size {
            let attn = scratch.attn_out.sub_offset(pos * cfg.q_dim(), cfg.q_dim());
            let out = scratch.tmp.sub_offset(pos * h, h);
            weight_gemv(gpu, &layer.o_proj, &attn, &out).map_err(|e| format!("drafter L{li} o pos {pos}: {e}"))?;
        }
        // residual: x = x + tmp (NO post_attention_layernorm on attn output)
        gpu.add_inplace_f32(&scratch.x, &scratch.tmp).map_err(|e| format!("drafter L{li} attn residual: {e:?}"))?;
        // FFN: norm with post_attention_layernorm (IS the pre-FFN norm) reading post-residual x
        gpu.rmsnorm_batched(&scratch.x, &layer.post_attention_layernorm, &scratch.tmp, block_size, h, eps).map_err(|e| format!("drafter L{li} post_attn/pre_ffn norm: {e:?}"))?;
        for pos in 0..block_size {
            let n2 = scratch.tmp.sub_offset(pos * h, h);
            let g = scratch.gate_ffn.sub_offset(pos * cfg.intermediate, cfg.intermediate);
            let u = scratch.up_ffn.sub_offset(pos * cfg.intermediate, cfg.intermediate);
            weight_gemv(gpu, &layer.gate_proj, &n2, &g).map_err(|e| format!("drafter L{li} gate pos {pos}: {e}"))?;
            weight_gemv(gpu, &layer.up_proj, &n2, &u).map_err(|e| format!("drafter L{li} up pos {pos}: {e}"))?;
        }
        gpu.silu_mul_f32(&scratch.gate_ffn, &scratch.up_ffn, &scratch.ffn_hidden).map_err(|e| format!("drafter L{li} silu: {e:?}"))?;
        for pos in 0..block_size {
            let fh = scratch.ffn_hidden.sub_offset(pos * cfg.intermediate, cfg.intermediate);
            let out = scratch.tmp.sub_offset(pos * h, h);
            weight_gemv(gpu, &layer.down_proj, &fh, &out).map_err(|e| format!("drafter L{li} down pos {pos}: {e}"))?;
        }
        gpu.add_inplace_f32(&scratch.x, &scratch.tmp).map_err(|e| format!("drafter L{li} ffn residual: {e:?}"))?;
    }
    gpu.rmsnorm_batched(&scratch.x, &weights.norm, &scratch.x, block_size, h, eps).map_err(|e| format!("drafter final norm: {e:?}"))?;
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
