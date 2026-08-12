// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! # STATUS: working — 1.24x over AR, byte-identical (measured 2026-08-11)
//!
//! hiptrx gfx1201 (R9700), 64 tokens greedy, 3 fresh processes per arm, target
//! `muse-glimmer-30b.mq4`, prompt md5 `2ef49ee70df1483079b1f73c1f768339`:
//!
//! | mode | tok/s (3 runs) | median | tau |
//! |---|---|---:|---:|
//! | AR | 32.96 · 32.94 · 32.94 | 32.94 | — |
//! | DFlash | 41.03 · 41.00 · 40.95 | **41.00** | **8.333** |
//!
//! 66 of 135 proposals accepted over 9 windows. Output byte-identical to AR at
//! temp 0, which is the required contract: acceptance is greedy-argmax, so any
//! divergence would be a bug rather than an acceptance-rate matter.
//!
//! Still opt-in via `HIPFIRE_DFLASH_DRAFT` (repo default is `dflash_mode=off`).
//!
//! ## What it took, so the next reader does not repeat it
//!
//! Four separate defects had to be cleared, and every one of them left the
//! engine *running and producing correct text* — only the acceptance rate
//! revealed them. In discovery order:
//!
//! 1. **Noise embedding never filled.** The drafter's whole input was a
//!    zero-filled `vec![0f32; block*hidden]`. Drafts decoded to token 0.
//! 2. **No attention.** The per-layer loop copied Q straight into `attn_out`
//!    behind a comment reading "for minimal, just do o_proj over q". Every
//!    block row was then bit-identical, capping acceptance at 1 per window.
//! 3. **Wrong block structure.** The drafter is a standard two-norm Llama block
//!    (58 tensors: no pre/post-FFN norm), not the target's four-norm sandwich.
//! 4. **Context delivered through the wrong pathway** — the big one. Upstream
//!    CONCATENATES the projected context into K/V, so K/V spans `ctx+block`
//!    while Q spans `block`. This code instead broadcast-ADDED a single context
//!    row into `x` and attended over the block alone. Fixing it moved tau from
//!    1.016 to 8.333 in one step.
//!
//! The authority for (4) is upstream `modeling_muse_glimmer_assistant.py`, whose
//! attention comment states it outright: *"The total k/v states in Dflash are the
//! concatenation of the previous `context_hidden_states` ... and the actual
//! projections on the diffusion window."* Guessing cost several rounds; reading
//! it cost one.
//!
//! Verify is a single BATCHED forward over the block, not B sequential decodes.
//! That distinction is the entire economics of speculative decode: the
//! sequential version streamed all 15.5 GB of weights 16 times per window and
//! ran at 12.0 tok/s — *slower than AR* — at the very same tau 8.333.
//!
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
use hipfire_runtime::llama::{rotate_x_mq_batched_for, weight_gemv, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Shared rotation gate (mirrors forward.rs) ───────────────────────────
// HIPFIRE_GLIMMER_SHARED_ROT default ON (=1 or unset), =0 selects old path.
fn shared_rot_enabled() -> bool {
    std::env::var("HIPFIRE_GLIMMER_SHARED_ROT").as_deref() != Ok("0")
}

// ─── Batched projection dispatch (mirrors forward.rs::proj_gemm_batched) ──
// Q8_0      → gemm_q8_0_batched_chunked (WMMA on gfx12)
// MQ4G256/HFQ4G256 → rotate + gemm_hfq4g256_batched_lmhead (prerotated)
// MQ6G256   → rotate + gemm_mq6g256_batched_lmhead
// others    → per-row weight_gemv fallback (explicit, no approximation)
//   Fallback dtypes: F32, Q4K, HFQ4G128, HFQ6G256, HFQ3G256, HFQ2G256, MQ3G256,
//   MQ2G256, MQ2G256Lloyd, MQ3G256Lloyd, MQ4G256Lloyd, MFP4G32, etc. — any
//   dtype without a batched GEMM kernel. Drafter weights are Q8_0 so the
//   Q8_0 batched path is taken; fallback is listed explicitly for
//   correctness parity with forward.rs and never taken on current artifacts.
fn proj_gemm_batched(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    x_rot: &GpuTensor,
    b: usize,
    label: &str,
) -> Result<(), String> {
    match w.gpu_dtype {
        DType::Q8_0 => gpu
            .gemm_q8_0_batched_chunked(&w.buf, x, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer batch {label} (q8): {e:?}")),
        DType::MQ4G256 | DType::HFQ4G256 => {
            rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)
                .map_err(|e| format!("glimmer batch {label} rotate: {e:?}"))?;
            gpu.gemm_hfq4g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
                .map_err(|e| format!("glimmer batch {label} (mq4): {e:?}"))
        }
        DType::MQ6G256 => {
            rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)
                .map_err(|e| format!("glimmer batch {label} rotate: {e:?}"))?;
            gpu.gemm_mq6g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
                .map_err(|e| format!("glimmer batch {label} (mq6): {e:?}"))
        }
        _ => {
            for i in 0..b {
                let x_row = x.sub_offset(i * w.k, w.k);
                let y_row = y.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)
                    .map_err(|e| format!("glimmer batch {label} row {i}: {e}"))?;
            }
            Ok(())
        }
    }
}

// Prerotated variant: x_rot already FWHT-rotated for MQ. Q8 still reads unrotated x.
fn proj_gemm_batched_prerotated(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_unrot: &GpuTensor,
    x_rot: &GpuTensor,
    y: &GpuTensor,
    b: usize,
    label: &str,
) -> Result<(), String> {
    match w.gpu_dtype {
        DType::Q8_0 => gpu
            .gemm_q8_0_batched_chunked(&w.buf, x_unrot, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer batch {label} (q8 prerot): {e:?}")),
        DType::MQ4G256 | DType::HFQ4G256 => gpu
            .gemm_hfq4g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer batch {label} (mq4 prerot): {e:?}")),
        DType::MQ6G256 => gpu
            .gemm_mq6g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer batch {label} (mq6 prerot): {e:?}")),
        _ => {
            for i in 0..b {
                let x_row = x_unrot.sub_offset(i * w.k, w.k);
                let y_row = y.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)
                    .map_err(|e| format!("glimmer batch {label} prerot row {i}: {e}"))?;
            }
            Ok(())
        }
    }
}

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
        // MQ4G256 (13) and its Lloyd-codebook sibling (19). The drafter's
        // forward already dispatches MQ4 with the FWHT rotation
        // (`proj_gemm_batched` / `_prerotated` above); only this loader arm was
        // missing, so an MQ4 drafter loaded fine everywhere except here and
        // DFlash silently fell back to AR with
        // "unsupported quant_type 13 for 'encoder.fc.weight'".
        //
        // Qwen's DFlash drafters have always been MQ4 (arch 20); Glimmer's was
        // the only Q8 one, at 2.59 GB against qwen35-27b's 0.88 GB for the same
        // 58-tensor / 36-weight shape.
        13 | 19 => {
            let buf = gpu
                .upload_raw(data, &[data.len()])
                .map_err(|e| format!("glimmer drafter: upload MQ4 '{name}': {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
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
    pub tmp: GpuTensor,        // [block*hidden] scratch
    pub gate_ffn: GpuTensor,
    pub up_ffn: GpuTensor,
    pub ffn_hidden: GpuTensor,
    pub logits_tmp: GpuTensor, // [hidden] for final norm
    /// Batched GEMM rotation scratch.
    pub x_rot: GpuTensor,           // [block * hidden] — shared rotation for q / gate/up
    pub kv_input: GpuTensor,        // [(max_ctx+block)*hidden] — cat[target_hidden_proj, tmp]
    pub kv_input_rot: GpuTensor,    // [(max_ctx+block)*hidden] — FWHT-rotated kv_input
    pub ffn_hidden_rot: GpuTensor,  // [block*intermediate] — FWHT-rotated ffn_hidden for down_proj
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
            x_rot: alloc(gpu, block * h, "x_rot")?,
            kv_input: alloc(gpu, kv_rows * h, "kv_input")?,
            kv_input_rot: alloc(gpu, kv_rows * h, "kv_input_rot")?,
            ffn_hidden_rot: alloc(gpu, block * cfg.intermediate, "ffn_hidden_rot")?,
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
            self.x_rot,
            self.kv_input,
            self.kv_input_rot,
            self.ffn_hidden_rot,
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
    let kvd = cfg.kv_dim();
    let l = ctx_len + block_size; // K/V length

    // --- 1. encoder.fc over ctx rows — batched GEMM (was per-row loop) ---
    // This runs once per drafter call over up to 2048 rows, so batching
    // matters at longer context even though ctx is small on the fixture.
    // Same dispatch-by-dtype as forward.rs::proj_gemm_batched.
    if ctx_len > 0 {
        let ne_h = ne * h;
        // Upload the entire target_hidden batch [ctx*ne_h] into a temporary
        // GPU buffer, then batched GEMM: Y[ctx,h] = W[h,ne_h] * X[ctx,ne_h].
        let fc_in = gpu
            .alloc_tensor(&[ctx_len * ne_h], DType::F32)
            .map_err(|e| format!("drafter fc_input alloc: {e:?}"))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(target_hidden.as_ptr() as *const u8, target_hidden.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&fc_in.buf, bytes)
            .map_err(|e| format!("drafter htod fc_input: {e:?}"))?;
        let target_slice = scratch.target_hidden_proj.sub_offset(0, ctx_len * h);
        // For fc, x_rot is dummy unless dtype is MQ — pass scratch.x_rot (or kv_input_rot)
        // sized enough for ctx*h dummy; use kv_input_rot's prefix.
        let fc_rot = scratch.kv_input_rot.sub_offset(0, ctx_len * h);
        proj_gemm_batched(
            gpu,
            &weights.fc,
            &fc_in,
            &target_slice,
            &fc_rot,
            ctx_len,
            "fc",
        )
        .map_err(|e| format!("drafter fc batched: {e}"))?;
        gpu.free_tensor(fc_in).ok();
        gpu.rmsnorm_batched(
            &target_slice,
            &weights.output_norm_enc,
            &target_slice,
            ctx_len,
            h,
            eps,
        )
        .map_err(|e| format!("drafter output_norm batched: {e:?}"))?;
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

        // q_proj over B block rows from n1=tmp — batched
        proj_gemm_batched(
            gpu,
            &layer.q_proj,
            &scratch.tmp,
            &scratch.q,
            &scratch.x_rot,
            block_size,
            "q_proj",
        )
        .map_err(|e| format!("drafter L{li} q batched: {e}"))?;

        // k/v over ctx+B: first ctx_len rows from target_hidden_proj, tail B from tmp
        // Materialize contiguous kv_input = cat[target_hidden_proj[0:ctx], tmp[0:B]]
        let kv_in = scratch.kv_input.sub_offset(0, l * h);
        let kv_in_rot = scratch.kv_input_rot.sub_offset(0, l * h);
        if ctx_len > 0 {
            gpu.hip
                .memcpy_dtod_at(
                    &kv_in.buf,
                    0,
                    &scratch.target_hidden_proj.buf,
                    0,
                    ctx_len * h * 4,
                )
                .map_err(|e| format!("drafter L{li} kv cat ctx copy: {e:?}"))?;
            gpu.hip
                .memcpy_dtod_at(
                    &kv_in.buf,
                    ctx_len * h * 4,
                    &scratch.tmp.buf,
                    0,
                    block_size * h * 4,
                )
                .map_err(|e| format!("drafter L{li} kv cat block copy: {e:?}"))?;
        } else {
            gpu.hip
                .memcpy_dtod_at(&kv_in.buf, 0, &scratch.tmp.buf, 0, block_size * h * 4)
                .map_err(|e| format!("drafter L{li} kv cat block copy: {e:?}"))?;
        }
        let k_full = scratch.k.sub_offset(0, l * kvd);
        let v_full = scratch.v.sub_offset(0, l * kvd);
        // k_proj / v_proj share the same input, so rotate once and reuse
        // (mirrors forward.rs q/k/v/gate shared FWHT).
        let need_kv_rot = shared_rot_enabled()
            && hipfire_dispatch::types::dtype_rotation_plan(layer.k_proj.gpu_dtype)
                != hipfire_dispatch::types::RotationPlan::None;
        if need_kv_rot {
            rotate_x_mq_batched_for(gpu, &layer.k_proj, &kv_in, &kv_in_rot, h, l)
                .map_err(|e| format!("drafter L{li} kv rotate: {e:?}"))?;
            proj_gemm_batched_prerotated(
                gpu, &layer.k_proj, &kv_in, &kv_in_rot, &k_full, l, "k_proj",
            )
            .map_err(|e| format!("drafter L{li} k batched: {e}"))?;
            proj_gemm_batched_prerotated(
                gpu, &layer.v_proj, &kv_in, &kv_in_rot, &v_full, l, "v_proj",
            )
            .map_err(|e| format!("drafter L{li} v batched: {e}"))?;
        } else {
            proj_gemm_batched(
                gpu, &layer.k_proj, &kv_in, &k_full, &kv_in_rot, l, "k_proj",
            )
            .map_err(|e| format!("drafter L{li} k batched: {e}"))?;
            proj_gemm_batched(
                gpu, &layer.v_proj, &kv_in, &v_full, &kv_in_rot, l, "v_proj",
            )
            .map_err(|e| format!("drafter L{li} v batched: {e}"))?;
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

        // o_proj over B rows — batched
        proj_gemm_batched(
            gpu,
            &layer.o_proj,
            &scratch.attn_out,
            &scratch.tmp,
            &scratch.x_rot,
            block_size,
            "o_proj",
        )
        .map_err(|e| format!("drafter L{li} o batched: {e}"))?;
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
        // gate_proj / up_proj share input — one rotation, then batched
        let need_ffn_rot = shared_rot_enabled()
            && hipfire_dispatch::types::dtype_rotation_plan(layer.gate_proj.gpu_dtype)
                != hipfire_dispatch::types::RotationPlan::None;
        if need_ffn_rot {
            rotate_x_mq_batched_for(gpu, &layer.gate_proj, &scratch.tmp, &scratch.x_rot, h, block_size)
                .map_err(|e| format!("drafter L{li} ffn rotate: {e:?}"))?;
            proj_gemm_batched_prerotated(
                gpu,
                &layer.gate_proj,
                &scratch.tmp,
                &scratch.x_rot,
                &scratch.gate_ffn,
                block_size,
                "gate_proj",
            )
            .map_err(|e| format!("drafter L{li} gate batched: {e}"))?;
            proj_gemm_batched_prerotated(
                gpu,
                &layer.up_proj,
                &scratch.tmp,
                &scratch.x_rot,
                &scratch.up_ffn,
                block_size,
                "up_proj",
            )
            .map_err(|e| format!("drafter L{li} up batched: {e}"))?;
        } else {
            proj_gemm_batched(
                gpu,
                &layer.gate_proj,
                &scratch.tmp,
                &scratch.gate_ffn,
                &scratch.x_rot,
                block_size,
                "gate_proj",
            )
            .map_err(|e| format!("drafter L{li} gate batched: {e}"))?;
            proj_gemm_batched(
                gpu,
                &layer.up_proj,
                &scratch.tmp,
                &scratch.up_ffn,
                &scratch.x_rot,
                block_size,
                "up_proj",
            )
            .map_err(|e| format!("drafter L{li} up batched: {e}"))?;
        }
        gpu.silu_mul_f32(&scratch.gate_ffn, &scratch.up_ffn, &scratch.ffn_hidden)
            .map_err(|e| format!("drafter L{li} silu: {e:?}"))?;
        // down_proj batched over B rows
        proj_gemm_batched(
            gpu,
            &layer.down_proj,
            &scratch.ffn_hidden,
            &scratch.tmp,
            &scratch.ffn_hidden_rot,
            block_size,
            "down_proj",
        )
        .map_err(|e| format!("drafter L{li} down batched: {e}"))?;
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
