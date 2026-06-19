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
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, EmbeddingFormat, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

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
        let _ = gpu.free_tensor(self.output.buf);
        for l in self.layers {
            let _ = gpu.free_tensor(l.input_norm);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            let _ = gpu.free_tensor(l.wq.buf);
            let _ = gpu.free_tensor(l.wk.buf);
            let _ = gpu.free_tensor(l.wv.buf);
            let _ = gpu.free_tensor(l.wo.buf);
            let _ = gpu.free_tensor(l.post_attn_norm);
            let _ = gpu.free_tensor(l.pre_ffn_norm);
            let _ = gpu.free_tensor(l.post_ffn_norm);
            let _ = gpu.free_tensor(l.w_gate.buf);
            let _ = gpu.free_tensor(l.w_up.buf);
            let _ = gpu.free_tensor(l.w_down.buf);
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
    #[cfg(unix)]
    hfq.drop_mmap();

    eprintln!("gemma3: loading token_embd...");
    let (token_embd, embd_format) = load_embed_tokens(hfq, gpu, cfg)?;

    eprintln!("gemma3: loading model.norm...");
    let output_norm = load_norm_weight_raw(hfq, gpu, "model.norm.weight", cfg.hidden_size)?;

    eprintln!(
        "gemma3: loading lm_head (tied={})...",
        cfg.tie_word_embeddings
    );
    let (output, tied_lm_head) = load_lm_head(hfq, gpu, cfg, embd_format)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        eprintln!(
            "gemma3: loading layer {}/{}...",
            i + 1,
            cfg.num_hidden_layers
        );
        layers.push(load_layer(hfq, gpu, cfg, i)?);
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
) -> HipResult<(GpuTensor, EmbeddingFormat)> {
    let name = "model.embed_tokens.weight";
    let (info, data) = hfq
        .tensor_data_vec(name)
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
        qt => panic!(
            "gemma3: unsupported embedding quant_type {qt}; handled 1/3/6/7. \
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
) -> HipResult<(WeightTensor, bool)> {
    // Gemma3 is tied in every shipped config, but honor the flag.
    let name = if cfg.tie_word_embeddings {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    };
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("gemma3: tensor not found for lm_head: {name}"));
    let m = cfg.vocab_size;
    let k = cfg.hidden_size;
    let weight = match info.quant_type {
        6 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::HFQ4G256, m, k),
        7 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::HFQ4G128, m, k),
        3 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::Q8_0, m, k),
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
        qt => panic!("gemma3: unsupported lm_head quant_type {qt}; handled 1/3/6/7."),
    };
    Ok((weight, cfg.tie_word_embeddings))
}

fn load_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    cfg: &Gemma3Config,
    i: usize,
) -> HipResult<Gemma3LayerWeights> {
    let p = format!("model.layers.{i}");
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    let input_norm = load_norm_weight_raw(
        hfq,
        gpu,
        &format!("{p}.input_layernorm.weight"),
        cfg.hidden_size,
    )?;
    // Per-head QK-norm: RMSNorm over head_dim.
    let q_norm = load_norm_weight_raw(
        hfq,
        gpu,
        &format!("{p}.self_attn.q_norm.weight"),
        cfg.head_dim,
    )?;
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

/// Load a linear weight to a `WeightTensor`. Bring-up format set
/// (F16 / Q8F16 / HFQ4G256 / HFQ4G128); extend for MQ4/MQ6.
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
    let wt = match info.quant_type {
        6 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::HFQ4G256, m, k),
        7 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::HFQ4G128, m, k),
        3 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::Q8_0, m, k),
        1 => weight_tensor(gpu.upload_raw(&data, &[data.len()])?, DType::F16, m, k),
        qt => panic!(
            "gemma3: unsupported linear quant_type {qt} for {name}; \
             handled 1 (F16), 3 (Q8F16), 6 (HFQ4G256), 7 (HFQ4G128). Extend for MQ4/MQ6."
        ),
    };
    Ok(wt)
}
