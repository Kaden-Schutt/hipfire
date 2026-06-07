// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2Moe weights + decode state.
//!
//! Mirrors the MiniMax-M2 loader (shared `WeightTensor`, `KvCache`, indexed-MoE
//! GEMV kernels). Cohere2Moe specifics handled here:
//!   * ONE `input_layernorm` per layer (the parallel block shares it between
//!     attention and the MLP — there is no post_attention_layernorm),
//!   * per-layer FFN split: layer < `first_k_dense_replace` is a dense SwiGLU
//!     MLP (`mlp.gate_proj/up_proj/down_proj`, dense_intermediate); the rest are
//!     128-expert top-8 MoE (`mlp.gate` router + `mlp.experts.{e}.*`),
//!   * NO routing bias (Cohere2Moe routing is aux-loss-free without an
//!     `e_score_correction_bias`) — a shared all-zeros bias tensor is kept so
//!     the existing bias-aware top-k kernel can be reused unchanged.
//!
//! Expert weights are byte-fused per layer into ONE packed gate_up blob + ONE
//! down blob with a device-pointer table (the minimax packed-expert layout the
//! `*_indexed` GEMV kernels expect). lm_head is tied to embed_tokens (loaded as
//! its own tensor if the HFQ carries it; the converter is responsible for the
//! tie).

use crate::config::{AttnKind, Cohere2MoeConfig};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, KvCache, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

// ───────────────────────── HFQ load helpers ─────────────────────────
// Mirror the minimax loader (those helpers are crate-private). Cohere2Moe HFQ
// files carry RAW HF tensor names, so look them up by exact name.

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("cohere2moe: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

/// Load a 1D norm vector (F16/F32) → F32 GpuTensor. Cohere2Moe uses STANDARD
/// RMSNorm (`weight * x̂`, no +1 offset — verified against `Cohere2MoeRMSNorm`,
/// which is selected because the checkpoint sets `rms_norm_eps`), so no offset
/// is baked in.
fn load_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, shape: &[usize]) -> Result<GpuTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let f32_data: Vec<f32> = match qt {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => return Err(format!("cohere2moe: expected F16/F32 norm for {name}, got qt={qt}")),
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("cohere2moe: upload norm {name}: {e:?}"))
}

/// Load a quantized 2D weight → WeightTensor, tagging gpu_dtype from quant_type.
fn load_wt(hfq: &HfqFile, gpu: &mut Gpu, name: &str, m: usize, k: usize) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("cohere2moe: load_wt {name}: {e}"))
}

/// quant_type → DType (mirrors minimax `wt_from_raw`). Uploads raw bytes and
/// tags the dtype; the forward's dtype dispatch + `rotate_x_mq` read it.
fn wt_from_raw(gpu: &mut Gpu, qt: u8, data: &[u8], m: usize, k: usize) -> Result<WeightTensor, String> {
    let dtype = match qt {
        3 => DType::Q8_0,
        6 => DType::HFQ4G256,
        8 => DType::HFQ6G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        19 => DType::MQ2G256Lloyd,
        20 => DType::MQ3G256Lloyd,
        30 => DType::MQ4G256Lloyd,
        1 => DType::F16,
        other => return Err(format!("unsupported quant_type {other}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

// ──────────────────────────── Weights ────────────────────────────

/// Dense SwiGLU MLP (the prefix dense layers, `mlp.{gate,up,down}_proj`).
pub struct DenseFfn {
    pub w1: WeightTensor, // gate_proj [dense_inter, hidden]
    pub w3: WeightTensor, // up_proj   [dense_inter, hidden]
    pub w2: WeightTensor, // down_proj [hidden, dense_inter]
}

/// Per-expert weights (representative handle into the packed per-layer blobs).
pub struct ExpertWeights {
    /// Fused gate(gate_proj)‖up(up_proj): [2*moe_inter, hidden].
    pub gate_up: WeightTensor,
    /// Down (down_proj): [hidden, moe_inter].
    pub down: WeightTensor,
}

/// Top-8 MoE FFN (`mlp.gate` router + packed experts).
pub struct MoeFfn {
    pub router: WeightTensor,               // mlp.gate.weight [n_exp, hidden]
    pub experts: Vec<ExpertWeights>,        // one representative over the packed blob
    pub expert_gate_up_ptrs: GpuTensor,     // [2*n_exp] F32 = n_exp u64 device ptrs
    pub expert_down_ptrs: GpuTensor,
}

pub enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

/// Per-layer GPU-resident weights. Cohere2Moe parallel block: a single
/// `input_norm` feeds BOTH the attention and the FFN (see `forward.rs`).
pub struct CohereLayerWeights {
    pub input_norm: GpuTensor, // input_layernorm.weight [hidden]
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub ffn: Ffn,
    /// Attention kind for this layer. SWA is deferred — see `forward.rs`; this
    /// is recorded so the windowed-KV path can branch on it later.
    pub attn_kind: AttnKind,
}

pub struct CohereWeights {
    pub embed: GpuTensor,      // model.embed_tokens.weight (Q8 raw, embedding_lookup_q8)
    pub final_norm: GpuTensor, // model.norm.weight
    pub lm_head: WeightTensor, // lm_head.weight (tied to embed in the checkpoint)
    /// Shared all-zeros routing bias [n_exp] — Cohere2Moe has no routing-bias
    /// term, but the bias-aware top-k kernel takes one; zeros make it a no-op.
    pub zero_bias: GpuTensor,
    pub layers: Vec<CohereLayerWeights>,
}

impl CohereWeights {
    pub fn load(hfq: &mut HfqFile, cfg: &Cohere2MoeConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let moe_inter = cfg.moe_intermediate_size;
        let dense_inter = cfg.dense_intermediate_size;
        let n_exp = cfg.num_experts;

        // Globals.
        let (_qt, embed_bytes) = read_tensor(hfq, "model.embed_tokens.weight")?;
        let embed = gpu
            .upload_raw(&embed_bytes, &[embed_bytes.len()])
            .map_err(|e| format!("cohere2moe: upload embed: {e:?}"))?;
        let final_norm = load_norm(hfq, gpu, "model.norm.weight", &[hidden])?;
        let lm_head = load_wt(hfq, gpu, "lm_head.weight", cfg.vocab_size, hidden)?;
        let zero_bias = gpu
            .upload_f32(&vec![0.0f32; n_exp], &[n_exp])
            .map_err(|e| format!("cohere2moe: upload zero_bias: {e:?}"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            // Parallel block → ONE norm shared by attention + MLP.
            let input_norm = load_norm(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[hidden])?;
            let wq = load_wt(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_dim, hidden)?;
            let wk = load_wt(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), kv_dim, hidden)?;
            let wv = load_wt(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), kv_dim, hidden)?;
            let wo = load_wt(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), hidden, q_dim)?;

            let ffn = if cfg.is_dense_ffn(l) {
                // Dense SwiGLU MLP (gate/up/down). Cohere `gate_proj` = SwiGLU
                // gate (w1), `up_proj` = w3, `down_proj` = w2.
                let w1 = load_wt(hfq, gpu, &format!("{p}.mlp.gate_proj.weight"), dense_inter, hidden)?;
                let w3 = load_wt(hfq, gpu, &format!("{p}.mlp.up_proj.weight"), dense_inter, hidden)?;
                let w2 = load_wt(hfq, gpu, &format!("{p}.mlp.down_proj.weight"), hidden, dense_inter)?;
                Ffn::Dense(DenseFfn { w1, w3, w2 })
            } else {
                Ffn::Moe(load_moe_layer(hfq, gpu, &p, hidden, moe_inter, n_exp)?)
            };

            layers.push(CohereLayerWeights {
                input_norm,
                wq,
                wk,
                wv,
                wo,
                ffn,
                attn_kind: cfg.attn_kind(l),
            });
        }

        Ok(CohereWeights {
            embed,
            final_norm,
            lm_head,
            zero_bias,
            layers,
        })
    }
}

/// Pack ALL experts of a MoE layer into ONE gate_up blob + ONE down blob with a
/// device-pointer table — the minimax packed-expert layout the `*_indexed` GEMV
/// kernels index by `base + e*stride`. One allocation per projection (vs the
/// per-expert hipMalloc storm that fragments VRAM).
fn load_moe_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    hidden: usize,
    moe_inter: usize,
    n_exp: usize,
) -> Result<MoeFfn, String> {
    let router = load_wt(hfq, gpu, &format!("{p}.mlp.gate.weight"), n_exp, hidden)?;

    let mut gu_combined: Vec<u8> = Vec::new();
    let mut dn_combined: Vec<u8> = Vec::new();
    let mut gu_stride = 0usize;
    let mut dn_stride = 0usize;
    let mut qt_gu = 0u8;
    let mut qt_dn = 0u8;
    for e in 0..n_exp {
        let ep = format!("{p}.mlp.experts.{e}");
        let (qt_g, gate) = read_tensor(hfq, &format!("{ep}.gate_proj.weight"))?;
        let (_qt_u, up) = read_tensor(hfq, &format!("{ep}.up_proj.weight"))?;
        let (qt_d, down) = read_tensor(hfq, &format!("{ep}.down_proj.weight"))?;
        let gu_len = gate.len() + up.len();
        if e == 0 {
            gu_stride = gu_len;
            dn_stride = down.len();
            qt_gu = qt_g;
            qt_dn = qt_d;
            gu_combined.reserve(gu_len * n_exp);
            dn_combined.reserve(down.len() * n_exp);
        } else if gu_len != gu_stride || down.len() != dn_stride {
            return Err(format!(
                "cohere2moe {p}E{e}: non-uniform expert stride (gate_up {gu_len}/{gu_stride}, down {}/{dn_stride}); packed layout requires equal-size experts",
                down.len()
            ));
        }
        gu_combined.extend_from_slice(&gate);
        gu_combined.extend_from_slice(&up);
        dn_combined.extend_from_slice(&down);
    }
    let gate_up = wt_from_raw(gpu, qt_gu, &gu_combined, 2 * moe_inter, hidden)
        .map_err(|e2| format!("cohere2moe: pack gate_up {p}: {e2}"))?;
    let down = wt_from_raw(gpu, qt_dn, &dn_combined, hidden, moe_inter)
        .map_err(|e2| format!("cohere2moe: pack down {p}: {e2}"))?;
    drop(gu_combined);
    drop(dn_combined);

    let gu_base = gate_up.buf.buf.as_ptr() as u64;
    let dn_base = down.buf.buf.as_ptr() as u64;
    let experts = vec![ExpertWeights { gate_up, down }];

    let gu_bytes: Vec<u8> = (0..n_exp)
        .flat_map(|e| (gu_base + (e * gu_stride) as u64).to_ne_bytes())
        .collect();
    let dn_bytes: Vec<u8> = (0..n_exp)
        .flat_map(|e| (dn_base + (e * dn_stride) as u64).to_ne_bytes())
        .collect();
    let expert_gate_up_ptrs = gpu
        .alloc_tensor(&[2 * n_exp], DType::F32)
        .map_err(|e| format!("cohere2moe: alloc gu_ptrs {p}: {e:?}"))?;
    let expert_down_ptrs = gpu
        .alloc_tensor(&[2 * n_exp], DType::F32)
        .map_err(|e| format!("cohere2moe: alloc dn_ptrs {p}: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
        .map_err(|e| format!("cohere2moe: htod gu_ptrs {p}: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
        .map_err(|e| format!("cohere2moe: htod dn_ptrs {p}: {e:?}"))?;

    Ok(MoeFfn {
        router,
        experts,
        expert_gate_up_ptrs,
        expert_down_ptrs,
    })
}

// ──────────────────────────── State ────────────────────────────

/// Per-decode GPU scratch + KV cache (Q8). Holds both the dense-MLP scratch
/// (used by the prefix dense layers) and the MoE scratch (used by the rest).
pub struct CohereState {
    pub kv: KvCache,
    pub pos_buf: hip_bridge::DeviceBuffer, // device i32 position scalar
    pub max_seq: usize,
    pub n_tokens: usize,

    // residual + shared norm
    pub h: GpuTensor,   // [hidden] residual stream
    pub tmp: GpuTensor, // [hidden] n = rmsnorm(h) — SHARED by attn + ffn (parallel block)

    // attention scratch
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]

    // dense-MLP scratch (prefix dense layers)
    pub dense_gate: GpuTensor, // [dense_inter]
    pub dense_up: GpuTensor,   // [dense_inter]
    pub dense_act: GpuTensor,  // [dense_inter]

    // MoE scratch
    pub ffn_x_rot: GpuTensor,     // [hidden] FWHT(n) for MQ4 experts
    pub router_logits: GpuTensor, // [n_exp]
    pub topk_indices: GpuTensor,  // [k] i32-in-F32
    pub topk_weights: GpuTensor,  // [k]
    pub gate_batch: GpuTensor,    // [k*moe_inter]
    pub up_batch: GpuTensor,      // [k*moe_inter]
    pub rot_batch: GpuTensor,     // [k*moe_inter]
    pub down_expanded: GpuTensor, // [k*hidden]

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub logits: GpuTensor,         // [vocab]
}

impl CohereState {
    pub fn new(gpu: &mut Gpu, cfg: &Cohere2MoeConfig) -> Result<Self, String> {
        // Cap the KV cache so the 500k-ctx config doesn't OOM; callers needing a
        // specific window use `new_with_max_seq`.
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(gpu: &mut Gpu, cfg: &Cohere2MoeConfig, max_seq: usize) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let moe_inter = cfg.moe_intermediate_size;
        let dense_inter = cfg.dense_intermediate_size;
        let n_exp = cfg.num_experts;
        let k = cfg.num_experts_per_tok;

        // FWHT sign LUT must exist before any rotate_x_mq / fused rotate kernel.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("cohere2moe: ensure_mq_signs: {e:?}"))?;

        let kv = KvCache::new_gpu_q8(
            gpu,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq,
        )
        .map_err(|e| format!("cohere2moe: kv cache: {e:?}"))?;
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("cohere2moe: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("cohere2moe: alloc {label}: {e:?}"))
        };

        Ok(CohereState {
            kv,
            pos_buf,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            tmp: alloc(gpu, hidden, "tmp")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            dense_gate: alloc(gpu, dense_inter, "dense_gate")?,
            dense_up: alloc(gpu, dense_inter, "dense_up")?,
            dense_act: alloc(gpu, dense_inter, "dense_act")?,
            ffn_x_rot: alloc(gpu, hidden, "ffn_x_rot")?,
            router_logits: alloc(gpu, n_exp, "router_logits")?,
            topk_indices: alloc(gpu, k, "topk_indices")?,
            topk_weights: alloc(gpu, k, "topk_weights")?,
            gate_batch: alloc(gpu, k * moe_inter, "gate_batch")?,
            up_batch: alloc(gpu, k * moe_inter, "up_batch")?,
            rot_batch: alloc(gpu, k * moe_inter, "rot_batch")?,
            down_expanded: alloc(gpu, k * hidden, "down_expanded")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
        })
    }

    pub fn reset(&mut self) {
        self.n_tokens = 0;
    }
}
