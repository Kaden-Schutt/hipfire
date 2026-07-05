// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 weight loading. See LICENSE / NOTICE.

//! GPU-resident Gemma3 weights + the HFQ loader.
//!
//! Replicates the qwen2 loader pattern (the helpers there are crate-private)
//! adapted for the Gemma3 layout: **4 norms per layer** (`input_layernorm`,
//! `post_attention_layernorm`, `pre_feedforward_layernorm`,
//! `post_feedforward_layernorm`), **per-head `q_norm`/`k_norm`** (over
//! `head_dim`), **GeGLU** (`gate`/`up`/`down`), **tied embeddings**, and **no
//! QKV bias** (`attention_bias=false`). Norm weights ship `(1+w)`-baked from the
//! quantizer, so they load as plain F32 and need no runtime offset.
//!
//! `load_weight_tensor` covers the bring-up format set (F16 / Q8F16 / HFQ4G256
//! / HFQ4G128); extend for MQ4/MQ6 when those gemma3 artifacts ship. The
//! duplication with qwen2/qwen35/dots-ocr is intentional debt — see the
//! shared-transformer-loader cleanup in
//! `docs/plans/2026-06-19-arch-roster-feature-matrix.md`.

use hip_bridge::HipResult;
use hipfire_runtime::hfq::{load_awq_scale, HfqFile};
use hipfire_runtime::quant::f16_to_f32;
use hipfire_runtime::weights::{EmbeddingFormat, WeightTensor};
use hipfire_rdna::{DType, Gpu, GpuTensor};

use crate::config::Gemma3Config;

/// Per-layer Gemma3 weights. No biases (attention_bias=false). The four norms
/// and the two qk-norms are F32 on GPU (qk-norm shape `[head_dim]`, the rest
/// `[hidden_size]`).
pub struct Gemma3LayerWeights {
    pub input_norm: GpuTensor,     // input_layernorm.weight  [hidden]
    pub q_norm: GpuTensor,         // self_attn.q_norm.weight  [head_dim]
    pub k_norm: GpuTensor,         // self_attn.k_norm.weight  [head_dim]
    pub wq: WeightTensor,          // self_attn.q_proj.weight  [n_heads*head_dim, hidden]
    pub wk: WeightTensor,          // self_attn.k_proj.weight  [n_kv*head_dim, hidden]
    pub wv: WeightTensor,          // self_attn.v_proj.weight
    pub wo: WeightTensor,          // self_attn.o_proj.weight
    pub post_attn_norm: GpuTensor, // post_attention_layernorm.weight  [hidden]
    pub pre_ffn_norm: GpuTensor,   // pre_feedforward_layernorm.weight  [hidden]
    pub post_ffn_norm: GpuTensor,  // post_feedforward_layernorm.weight [hidden]
    pub w_gate: WeightTensor,      // mlp.gate_proj.weight  [intermediate, hidden]
    pub w_up: WeightTensor,        // mlp.up_proj.weight
    pub w_down: WeightTensor,      // mlp.down_proj.weight  [hidden, intermediate]
}

/// GPU-resident Gemma3 model weights.
pub struct Gemma3Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor, // model.norm.weight, F32
    pub output: WeightTensor,   // lm_head (tied → re-uploaded embedding bytes)
    pub layers: Vec<Gemma3LayerWeights>,
    /// Gemma3 ties lm_head to the embedding table; `output` is a separate
    /// allocation of the same bytes (GpuTensor is not Clone).
    pub tied_lm_head: bool,
}

impl Gemma3Weights {
    /// Load every tensor from `hfq` to GPU.
    pub fn load(hfq: &mut HfqFile, cfg: &Gemma3Config, gpu: &mut Gpu) -> Result<Self, String> {
        load_weights(hfq, cfg, gpu).map_err(|e| format!("gemma3: load_weights failed: {e:?}"))
    }

    /// Release every GPU buffer back to the pool. Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        free_weight_tensor(gpu, self.output);
        for l in self.layers {
            let _ = gpu.free_tensor(l.input_norm);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            free_weight_tensor(gpu, l.wq);
            free_weight_tensor(gpu, l.wk);
            free_weight_tensor(gpu, l.wv);
            free_weight_tensor(gpu, l.wo);
            let _ = gpu.free_tensor(l.post_attn_norm);
            let _ = gpu.free_tensor(l.pre_ffn_norm);
            let _ = gpu.free_tensor(l.post_ffn_norm);
            free_weight_tensor(gpu, l.w_gate);
            free_weight_tensor(gpu, l.w_up);
            free_weight_tensor(gpu, l.w_down);
        }
    }
}

/// Free-function loader; takes a borrowed `Gpu` so the `Architecture` impl can
/// pass the runtime-provided handle.
pub fn load_weights(
    hfq: &mut HfqFile,
    cfg: &Gemma3Config,
    gpu: &mut Gpu,
) -> HipResult<Gemma3Weights> {
    load_weights_prefixed(hfq, cfg, gpu, "")
}

/// Load the gemma3 text decoder with a tensor-name `prefix`. Pure-text gemma3
/// uses `""` (`model.*` / `lm_head.weight`); the gemma3 multimodal wrapper nests
/// the decoder under `"language_model."` (`language_model.model.*`).
pub fn load_weights_prefixed(
    hfq: &mut HfqFile,
    cfg: &Gemma3Config,
    gpu: &mut Gpu,
    prefix: &str,
) -> HipResult<Gemma3Weights> {
    #[cfg(unix)]
    hfq.drop_mmap();

    eprintln!("gemma3: loading token_embd...");
    let (token_embd, embd_format) = load_embed_tokens(hfq, gpu, cfg, prefix)?;

    eprintln!("gemma3: loading model.norm...");
    let output_norm = load_norm_weight_raw(
        hfq,
        gpu,
        &format!("{prefix}model.norm.weight"),
        cfg.hidden_size,
    )?;

    eprintln!(
        "gemma3: loading lm_head (tied={})...",
        cfg.tie_word_embeddings
    );
    let (output, tied_lm_head) = load_lm_head(hfq, gpu, cfg, embd_format, prefix)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        eprintln!(
            "gemma3: loading layer {}/{}...",
            i + 1,
            cfg.num_hidden_layers
        );
        hipfire_runtime::load_progress::report(
            i as u32 + 1,
            cfg.num_hidden_layers as u32,
            "weights",
        );
        layers.push(load_layer(hfq, gpu, cfg, i, prefix)?);
    }

    Ok(Gemma3Weights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        tied_lm_head,
    })
}

// ─── Per-tensor loaders (replicated from qwen2; see module doc) ──────────────

fn load_embed_tokens(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &Gemma3Config,
    prefix: &str,
) -> HipResult<(GpuTensor, EmbeddingFormat)> {
    let name = format!("{prefix}model.embed_tokens.weight");
    let (info, data) = hfq
        .tensor_data_vec(&name)
        .unwrap_or_else(|| panic!("gemma3: tensor not found: {name}"));
    match info.quant_type {
        6 => Ok((
            gpu.upload_raw(&data, &[data.len()])?,
            EmbeddingFormat::HFQ4G256,
        )),
        7 => Ok((
            gpu.upload_raw(&data, &[data.len()])?,
            EmbeddingFormat::HFQ4G128,
        )),
        3 => Ok((gpu.upload_raw(&data, &[data.len()])?, EmbeddingFormat::Q8_0)),
        1 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let buf = gpu.upload_f32(&f32_data, &[cfg.vocab_size, cfg.hidden_size])?;
            Ok((buf, EmbeddingFormat::F32))
        }
        16 => {
            // bf16 source → promote to F32 (bf16 = high 16 bits of f32; there is
            // no bf16 EmbeddingFormat, and a raw upload tagged otherwise corrupts
            // the lookup). Mirrors the F16 (type 1) arm.
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let buf = gpu.upload_f32(&f32_data, &[cfg.vocab_size, cfg.hidden_size])?;
            Ok((buf, EmbeddingFormat::F32))
        }
        qt => panic!(
            "gemma3: unsupported embedding quant_type {qt}; handled 1/3/6/7/16. \
             Extend load_embed_tokens."
        ),
    }
}

/// Load the lm_head. Gemma3 ties embeddings: re-upload the embedding bytes as a
/// separate allocation (GpuTensor is not Clone). F16 source is promoted to F32
/// (EmbeddingFormat has no F16 variant — uploading raw F16 tagged F32 corrupts
/// the matmul; see qwen2's load_lm_head doc).
fn load_lm_head(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &Gemma3Config,
    embd_format: EmbeddingFormat,
    prefix: &str,
) -> HipResult<(WeightTensor, bool)> {
    // Gemma3 is tied in every shipped config, but honor the flag.
    let name = if cfg.tie_word_embeddings {
        format!("{prefix}model.embed_tokens.weight")
    } else {
        format!("{prefix}lm_head.weight")
    };
    let (info, data) = hfq
        .tensor_data_vec(&name)
        .unwrap_or_else(|| panic!("gemma3: tensor not found for lm_head: {name}"));
    let m = cfg.vocab_size;
    let k = cfg.hidden_size;
    let mut weight = match info.quant_type {
        6 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::HFQ4G256, m, k),
        7 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::HFQ4G128, m, k),
        3 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::Q8_0, m, k),
        33 => {
            let combined = oq4_to_oq8_combined(&data, m, k);
            weight_tensor(
                gpu.upload_raw(&combined, &[combined.len()])?,
                DType::Oq8G256,
                m,
                k,
            )
        }
        34 => {
            let combined = oq4_pack_arch_combined(&data, m, k);
            weight_tensor(
                gpu.upload_raw(&combined, &[combined.len()])?,
                DType::Oq4G256,
                m,
                k,
            )
        }
        35 => {
            let combined = oq8_combined(&data, m, k);
            weight_tensor(
                gpu.upload_raw(&combined, &[combined.len()])?,
                DType::Oq8G256,
                m,
                k,
            )
        }
        37 => {
            assert_eq!(
                data.len(),
                oq4_arch_combined_len(m, k),
                "gemma3: OP4 arch-packed lm_head has invalid byte length"
            );
            weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::Oq4G256, m, k)
        }
        1 => {
            // Promote F16 → F32 on host (see doc above), unless the tied embed
            // is already a packed format (handled by the arms above).
            let _ = embd_format;
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            weight_tensor(gpu.upload_f32(&f32_data, &[m, k])?, DType::F32, m, k)
        }
        16 => {
            // bf16 → F32 (mirrors the F16 arm; bf16 = high 16 bits of f32).
            let _ = embd_format;
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            weight_tensor(gpu.upload_f32(&f32_data, &[m, k])?, DType::F32, m, k)
        }
        qt => {
            panic!("gemma3: unsupported lm_head quant_type {qt}; handled 1/3/6/7/16/33/34/35/37.")
        }
    };
    if weight.gpu_dtype.supports_awq_sidecar() {
        weight.awq_scale = load_awq_scale(hfq, gpu, &name, k);
    }
    Ok((weight, cfg.tie_word_embeddings))
}

fn load_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &Gemma3Config,
    i: usize,
    prefix: &str,
) -> HipResult<Gemma3LayerWeights> {
    let p = format!("{prefix}model.layers.{i}");
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    let input_norm = load_norm_weight_raw(
        hfq,
        gpu,
        &format!("{p}.input_layernorm.weight"),
        cfg.hidden_size,
    )?;
    // Per-head QK-norm: RMSNorm over head_dim. Bake the Q pre-scale into
    // q_norm so the attention kernel's built-in 1/√head_dim becomes Gemma's
    // 1/√query_pre_attn_scalar (no per-step scale launch). No-op when
    // q_prescale == 1.0 (query_pre_attn_scalar == head_dim, e.g. gemma3-4b).
    let q_norm = load_norm_weight_raw(
        hfq,
        gpu,
        &format!("{p}.self_attn.q_norm.weight"),
        cfg.head_dim,
    )?;
    let prescale = cfg.q_prescale();
    if (prescale - 1.0).abs() > 1e-6 {
        gpu.scale_f32(&q_norm, prescale)?;
    }
    let k_norm = load_norm_weight_raw(
        hfq,
        gpu,
        &format!("{p}.self_attn.k_norm.weight"),
        cfg.head_dim,
    )?;

    let wq = load_weight_tensor(
        hfq,
        gpu,
        &format!("{p}.self_attn.q_proj.weight"),
        q_dim,
        cfg.hidden_size,
    )?;
    let wk = load_weight_tensor(
        hfq,
        gpu,
        &format!("{p}.self_attn.k_proj.weight"),
        kv_dim,
        cfg.hidden_size,
    )?;
    let wv = load_weight_tensor(
        hfq,
        gpu,
        &format!("{p}.self_attn.v_proj.weight"),
        kv_dim,
        cfg.hidden_size,
    )?;
    let wo = load_weight_tensor(
        hfq,
        gpu,
        &format!("{p}.self_attn.o_proj.weight"),
        cfg.hidden_size,
        q_dim,
    )?;

    let post_attn_norm = load_norm_weight_raw(
        hfq,
        gpu,
        &format!("{p}.post_attention_layernorm.weight"),
        cfg.hidden_size,
    )?;
    let pre_ffn_norm = load_norm_weight_raw(
        hfq,
        gpu,
        &format!("{p}.pre_feedforward_layernorm.weight"),
        cfg.hidden_size,
    )?;
    let post_ffn_norm = load_norm_weight_raw(
        hfq,
        gpu,
        &format!("{p}.post_feedforward_layernorm.weight"),
        cfg.hidden_size,
    )?;

    let w_gate = load_weight_tensor(
        hfq,
        gpu,
        &format!("{p}.mlp.gate_proj.weight"),
        cfg.intermediate_size,
        cfg.hidden_size,
    )?;
    let w_up = load_weight_tensor(
        hfq,
        gpu,
        &format!("{p}.mlp.up_proj.weight"),
        cfg.intermediate_size,
        cfg.hidden_size,
    )?;
    let w_down = load_weight_tensor(
        hfq,
        gpu,
        &format!("{p}.mlp.down_proj.weight"),
        cfg.hidden_size,
        cfg.intermediate_size,
    )?;

    Ok(Gemma3LayerWeights {
        input_norm,
        q_norm,
        k_norm,
        wq,
        wk,
        wv,
        wo,
        post_attn_norm,
        pre_ffn_norm,
        post_ffn_norm,
        w_gate,
        w_up,
        w_down,
    })
}

/// Upload an F16/F32/BF16 norm/scalar tensor as F32 on GPU. (Gemma3 norms are
/// already `(1+w)`-baked at ingest, so this loads them verbatim.)
fn load_norm_weight_raw(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("gemma3: tensor not found: {name}"));
    let f32_data: Vec<f32> = match info.quant_type {
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
        qt => panic!("gemma3: expected F16/F32/BF16 for norm {name}, got qt={qt}"),
    };
    assert_eq!(
        f32_data.len(),
        n,
        "gemma3: norm {name} has {} elements, expected {n}",
        f32_data.len()
    );
    gpu.upload_f32(&f32_data, &[n])
}

fn weight_tensor(buf: GpuTensor, gpu_dtype: DType, m: usize, k: usize) -> WeightTensor {
    WeightTensor {
        buf,
        gpu_dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    }
}

fn free_weight_tensor(gpu: &mut Gpu, wt: WeightTensor) {
    let _ = gpu.free_tensor(wt.buf);
    if let Some(awq) = wt.awq_scale {
        let _ = gpu.free_tensor(awq);
    }
}

/// Load a linear weight to a `WeightTensor`. Bring-up format set
/// (F16 / Q8F16 / HFQ4G256 / HFQ4G128 / OP4 / OP8).
fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("gemma3: tensor not found: {name}"));
    let mut wt = match info.quant_type {
        33 => {
            let combined = oq4_to_oq8_combined(&data, m, k);
            weight_tensor(
                gpu.upload_raw(&combined, &[combined.len()])?,
                DType::Oq8G256,
                m,
                k,
            )
        }
        34 => {
            let combined = oq4_pack_arch_combined(&data, m, k);
            weight_tensor(
                gpu.upload_raw(&combined, &[combined.len()])?,
                DType::Oq4G256,
                m,
                k,
            )
        }
        35 => {
            let combined = oq8_combined(&data, m, k);
            weight_tensor(
                gpu.upload_raw(&combined, &[combined.len()])?,
                DType::Oq8G256,
                m,
                k,
            )
        }
        37 => {
            assert_eq!(
                data.len(),
                oq4_arch_combined_len(m, k),
                "gemma3: OP4 arch-packed tensor {name} has invalid byte length"
            );
            weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::Oq4G256, m, k)
        }
        // bf16 stays bf16 on GPU (the gemm/gemv families dispatch a bf16 path,
        // same as the gemma3-vl bf16 vision tower) — no F32 promotion needed. The
        // *buffer's* dtype must be BF16, not just the WeightTensor's gpu_dtype:
        // `gemm_bf16_x_bf16_wmma` asserts on the GpuTensor's dtype, and
        // `upload_raw` tags the buffer `Raw`.
        16 => {
            let mut buf = gpu.upload_raw(&data, &[data.len()])?;
            buf.dtype = DType::BF16;
            weight_tensor(buf, DType::BF16, m, k)
        }
        // All pure upload-and-tag formats (F16/Q8/HFQ*/MQ*/Qtip3G256/FP4…)
        // route through the shared canonical map in hipfire_runtime::quant;
        // the transform arms above (bf16 retag, OP4/OP8 arch-repack) stay local.
        qt => {
            let dtype = hipfire_runtime::quant::dtype_for_quant_type(qt, k).unwrap_or_else(|| {
                panic!(
                    "gemma3: unsupported linear quant_type {qt} for {name}; \
                     transform arms handle 16 (BF16), 33/34/35/37 (OP4/OP8); \
                     all pure formats come from hipfire_runtime::quant::dtype_for_quant_type."
                )
            });
            weight_tensor(gpu.upload_raw(&data, &[data.len()])?, dtype, m, k)
        }
    };
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = load_awq_scale(hfq, gpu, name, k);
    }
    Ok(wt)
}

fn sext4(nib: u8) -> i8 {
    let v = (nib & 0xf) as i8;
    if v > 7 {
        v - 16
    } else {
        v
    }
}

fn oq4_arch_combined_len(m: usize, k: usize) -> usize {
    let ng = k / 256;
    m * (k / 2) + m * ng * 4 + m * ng * (4 + 128)
}

fn oq4_pack_arch_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    // Single-sourced from hipfire-quant-format (WP-3.3): Oq4G256 = 130.
    const BLOCK: usize = hipfire_runtime::quant::QuantType::Oq4G256
        .block_bytes()
        .unwrap();
    const ILB: usize = 132;
    assert_eq!(k % GROUP, 0, "OP4 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let packed_bytes = m * (k / 2);
    let scales_bytes = m * ng * 4;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "OP4 weight byte length {} != M*ng*130 = {expect} (M={m} K={k})",
        data.len()
    );
    let mut out = vec![0u8; packed_bytes + scales_bytes + m * ng * ILB];
    let scales_base = packed_bytes;
    let il_base = packed_bytes + scales_bytes;
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let nib_dst = r * (k / 2) + g * (GROUP / 2);
            out[nib_dst..nib_dst + 128].copy_from_slice(&data[src + 2..src + BLOCK]);
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let scale_dst = scales_base + (r * ng + g) * 4;
            out[scale_dst..scale_dst + 4].copy_from_slice(&scale.to_le_bytes());
            let il_dst = il_base + (r * ng + g) * ILB;
            out[il_dst..il_dst + 4].copy_from_slice(&scale.to_le_bytes());
            out[il_dst + 4..il_dst + ILB].copy_from_slice(&data[src + 2..src + BLOCK]);
        }
    }
    out
}

fn oq4_to_oq8_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    // Single-sourced from hipfire-quant-format (WP-3.3): Oq4G256 = 130.
    const BLOCK: usize = hipfire_runtime::quant::QuantType::Oq4G256
        .block_bytes()
        .unwrap();
    assert_eq!(k % GROUP, 0, "OP4-8 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "OP4-8 weight byte length {} != M*ng*130 = {expect} (M={m} K={k})",
        data.len()
    );
    let mut combined = vec![0u8; m * k + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let dst = r * k + g * GROUP;
            for i in 0..128 {
                let byte = data[src + 2 + i];
                combined[dst + 2 * i] = sext4(byte & 0xf) as u8;
                combined[dst + 2 * i + 1] = sext4(byte >> 4) as u8;
            }
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let so = m * k + (r * ng + g) * 4;
            combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
        }
    }
    combined
}

fn oq8_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    // Single-sourced from hipfire-quant-format (WP-3.3): Oq8G256 = 258.
    const BLOCK: usize = hipfire_runtime::quant::QuantType::Oq8G256
        .block_bytes()
        .unwrap();
    assert_eq!(k % GROUP, 0, "OP8 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "OP8 weight byte length {} != M*ng*258 = {expect} (M={m} K={k})",
        data.len()
    );
    let mut combined = vec![0u8; m * k + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let dst = r * k + g * GROUP;
            combined[dst..dst + GROUP].copy_from_slice(&data[src + 2..src + BLOCK]);
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let so = m * k + (r * ng + g) * 4;
            combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
        }
    }
    combined
}
