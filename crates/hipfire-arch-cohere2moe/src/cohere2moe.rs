// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cohere2-MoE (North-Mini-Code) weights + decode state.
//!
//! HFQ files carry RAW HF tensor names; the loader looks each up by exact name
//! (no rename). Mirrors the LFM2.5-MoE / MiniMax loaders (shared `WeightTensor`,
//! `KvCache`, indexed-MoE GEMV kernels) but reflects the Cohere2 structure:
//!   * ONE `input_layernorm` per layer (parallel block — feeds both attention
//!     and the FFN; there is no `post_attention_layernorm`).
//!   * No QK-norm, no attention bias.
//!   * Per-layer FFN split: the first `first_k_dense_replace` layers are dense
//!     SwiGLU MLPs (`mlp.{gate,up,down}_proj`, intermediate 3072); the rest are
//!     128-expert MoE (`mlp.experts.{j}.{gate,up,down}_proj` + `mlp.gate`), no
//!     routing bias, no shared expert.
//!   * Tied embeddings: `lm_head` reuses `model.embed_tokens.weight`.
//!
//! Expert weights ship pre-split (gate_proj/up_proj/down_proj); the loader
//! byte-fuses gate_proj‖up_proj into the per-expert `gate_up` blob the indexed
//! GEMV kernels expect. Per-expert buffers are retained (not just a packed
//! blob) so the forward can take a per-expert `weight_gemv` path for the F16
//! oracle / Q8 expert tiers (which have no indexed MoE kernel) and the indexed
//! kernels for the MQ4/MQ6 tiers.

use crate::config::{AttnKind, Cohere2MoeConfig};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, KvCache, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

// ───────────────────────── HFQ load helpers ─────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("cohere2moe: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

/// Load a 1D/raw F16/F32/Q8 vector → F32 GpuTensor with the given shape. Used
/// for the per-layer + final **RMSNorm** weights (cohere2_moe uses RMSNorm at
/// `rms_norm_eps`, NOT base Cohere2's mean-centered LayerNorm). RMSNorm has a
/// learned weight (gamma) and no bias — this loads the gamma vector.
fn load_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
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
        3 => dequant_q8_0(&data),
        _ => {
            return Err(format!(
                "cohere2moe: expected F16/F32/Q8 for {name}, got qt={qt}"
            ))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("cohere2moe: upload {name}: {e:?}"))
}

/// Minimal Q8_0 dequant (32-elem blocks: little-endian f16 scale + 32 int8).
fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for blk in data.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push((q as i8) as f32 * scale);
        }
    }
    out
}

fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("cohere2moe: load_wt {name}: {e}"))
}

/// quant_type → DType mapping (mirrors lfm2moe/minimax::wt_from_raw); uploads
/// raw bytes and tags the dtype for kernel dispatch.
fn wt_from_raw(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let dtype = match qt {
        1 => DType::F16,
        2 => DType::F32,
        16 => DType::BF16, // native bf16 reference (oracle tier)
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

/// Dense SwiGLU MLP (the `first_k_dense_replace` prefix layers; layer 0 here).
pub struct DenseFfn {
    pub gate: WeightTensor, // mlp.gate_proj [dense_inter, hidden]
    pub up: WeightTensor,   // mlp.up_proj   [dense_inter, hidden]
    pub down: WeightTensor, // mlp.down_proj [hidden, dense_inter]
}

/// One MoE expert: fused gate(gate_proj)‖up(up_proj) and down(down_proj).
pub struct ExpertWeights {
    pub gate_up: WeightTensor, // [2*moe_inter, hidden]
    pub down: WeightTensor,    // [hidden, moe_inter]
}

/// Per-layer shared ParoQuant rotation sidecars (Dir/safetensors path only).
/// All experts in a MoE layer reference these via non-owning `ParoRotation`
/// aliases, so the owner must outlive the experts — `MoeFfn` holds it.
pub struct Cohere2MoeParoSidecars {
    pub gate_up_pairs: GpuTensor,
    pub gate_up_theta: GpuTensor,
    pub gate_up_channel_scales: GpuTensor,
    pub down_pairs: GpuTensor,
    pub down_theta: GpuTensor,
    pub down_channel_scales: GpuTensor,
}

/// 128-expert MoE FFN (sigmoid selection, no bias, no shared expert).
pub struct MoeFfn {
    pub router: WeightTensor,           // mlp.gate.weight [n_exp, hidden]
    pub experts: Vec<ExpertWeights>,    // per-expert buffers (owned here)
    pub expert_gate_up_ptrs: GpuTensor, // [2*n_exp] F32 = n_exp u64 device ptrs
    pub expert_down_ptrs: GpuTensor,
    /// Owned per-layer PARO sidecars (Some on the Dir/paro path; None for HFQ).
    /// The experts' ParoRotation aliases reference these — keep them alive.
    pub paro_shared: Option<Cohere2MoeParoSidecars>,
}

pub enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

pub struct Cohere2MoeLayerWeights {
    /// The SINGLE `input_layernorm` (RMSNorm gamma, [hidden]). Feeds
    /// both the attention and FFN branches (parallel block).
    pub input_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub ffn: Ffn,
    /// full_attention (global, NoPE) vs sliding_attention (window, RoPE).
    pub attn_kind: AttnKind,
}

pub struct Cohere2MoeWeights {
    pub embed: GpuTensor,      // model.embed_tokens.weight (raw bytes)
    pub embed_dtype: DType,    // dtype of `embed` (drives the lookup path)
    pub final_norm: GpuTensor, // model.norm.weight (RMSNorm gamma)
    pub lm_head: WeightTensor, // tied = embed_tokens
    pub layers: Vec<Cohere2MoeLayerWeights>,
}

impl Cohere2MoeWeights {
    pub fn load(hfq: &mut HfqFile, cfg: &Cohere2MoeConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let dense_inter = cfg.dense_intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;

        // Globals. embed_tokens is the shared (tied) lm_head.
        let (eqt, embed_bytes) = read_tensor(hfq, "model.embed_tokens.weight")?;
        let embed = gpu
            .upload_raw(&embed_bytes, &[embed_bytes.len()])
            .map_err(|e| format!("cohere2moe: upload embed: {e:?}"))?;
        let lm_head = load_wt(
            hfq,
            gpu,
            "model.embed_tokens.weight",
            cfg.vocab_size,
            hidden,
        )?;
        let embed_dtype = lm_head.gpu_dtype;
        let final_norm = load_f32(hfq, gpu, "model.norm.weight", &[hidden])?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let input_norm = load_f32(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[hidden])?;
            let wq = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_proj.weight"),
                q_dim,
                hidden,
            )?;
            let wk = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let wv = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let wo = load_wt(
                hfq,
                gpu,
                &format!("{p}.self_attn.o_proj.weight"),
                hidden,
                q_dim,
            )?;

            let ffn = if cfg.is_dense_ffn(l) {
                let gate = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    dense_inter,
                    hidden,
                )?;
                let up = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    dense_inter,
                    hidden,
                )?;
                let down = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden,
                    dense_inter,
                )?;
                Ffn::Dense(DenseFfn { gate, up, down })
            } else {
                let router = load_wt(hfq, gpu, &format!("{p}.mlp.gate.weight"), n_exp, hidden)?;
                // Byte-fuse gate_proj‖up_proj → gate_up [2*moe_inter, hidden]; down_proj → down.
                let mut experts = Vec::with_capacity(n_exp);
                for e in 0..n_exp {
                    let ep = format!("{p}.mlp.experts.{e}");
                    let (qt_g, g) = read_tensor(hfq, &format!("{ep}.gate_proj.weight"))?;
                    let (qt_u, u) = read_tensor(hfq, &format!("{ep}.up_proj.weight"))?;
                    // gate_proj‖up_proj are byte-fused into one buffer tagged with a
                    // single dtype (qt_g); valid only when they are co-quantized. A
                    // mixed-dtype export would mis-read the up half as qt_g — refuse
                    // at load rather than serve silently-wrong inference.
                    if qt_g != qt_u {
                        return Err(format!(
                            "cohere2moe L{l}E{e}: gate/up dtype mismatch ({qt_g:?} vs {qt_u:?}) — cannot byte-fuse gate_up"
                        ));
                    }
                    let mut gate_up_bytes = g;
                    gate_up_bytes.extend_from_slice(&u);
                    let gate_up = wt_from_raw(gpu, qt_g, &gate_up_bytes, 2 * moe_inter, hidden)
                        .map_err(|e2| format!("cohere2moe: fuse gate_up L{l}E{e}: {e2}"))?;
                    let (qt_d, d) = read_tensor(hfq, &format!("{ep}.down_proj.weight"))?;
                    let down = wt_from_raw(gpu, qt_d, &d, hidden, moe_inter)
                        .map_err(|e2| format!("cohere2moe: down L{l}E{e}: {e2}"))?;
                    experts.push(ExpertWeights { gate_up, down });
                }
                // Device pointer tables for the indexed-MoE GEMV kernels: n_exp
                // u64 device addresses each, stored as [2*n_exp] F32 (8 bytes/ptr).
                let gu_bytes: Vec<u8> = experts
                    .iter()
                    .flat_map(|e| (e.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
                    .collect();
                let dn_bytes: Vec<u8> = experts
                    .iter()
                    .flat_map(|e| (e.down.buf.buf.as_ptr() as u64).to_ne_bytes())
                    .collect();
                let expert_gate_up_ptrs = gpu
                    .alloc_tensor(&[2 * n_exp], DType::F32)
                    .map_err(|e| format!("cohere2moe: alloc gu_ptrs: {e:?}"))?;
                let expert_down_ptrs = gpu
                    .alloc_tensor(&[2 * n_exp], DType::F32)
                    .map_err(|e| format!("cohere2moe: alloc dn_ptrs: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
                    .map_err(|e| format!("cohere2moe: htod gu_ptrs: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
                    .map_err(|e| format!("cohere2moe: htod dn_ptrs: {e:?}"))?;
                Ffn::Moe(MoeFfn {
                    router,
                    experts,
                    expert_gate_up_ptrs,
                    expert_down_ptrs,
                    paro_shared: None,
                })
            };

            layers.push(Cohere2MoeLayerWeights {
                input_norm,
                wq,
                wk,
                wv,
                wo,
                ffn,
                attn_kind: cfg.attn_kind(l),
            });
        }

        let _ = eqt;
        Ok(Cohere2MoeWeights {
            embed,
            embed_dtype,
            final_norm,
            lm_head,
            layers,
        })
    }
}

impl DenseFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DenseFfn { gate, up, down } = self;
        gate.free_all(gpu);
        up.free_all(gpu);
        down.free_all(gpu);
    }
}

impl ExpertWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let ExpertWeights { gate_up, down } = self;
        gate_up.free_all(gpu);
        down.free_all(gpu);
    }
}

impl MoeFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let MoeFfn {
            router,
            experts,
            expert_gate_up_ptrs,
            expert_down_ptrs,
            paro_shared,
        } = self;
        router.free_all(gpu);
        for e in experts {
            e.free_gpu(gpu);
        }
        let _ = gpu.free_tensor(expert_gate_up_ptrs);
        let _ = gpu.free_tensor(expert_down_ptrs);
        if let Some(s) = paro_shared {
            let Cohere2MoeParoSidecars {
                gate_up_pairs,
                gate_up_theta,
                gate_up_channel_scales,
                down_pairs,
                down_theta,
                down_channel_scales,
            } = s;
            let _ = gpu.free_tensor(gate_up_pairs);
            let _ = gpu.free_tensor(gate_up_theta);
            let _ = gpu.free_tensor(gate_up_channel_scales);
            let _ = gpu.free_tensor(down_pairs);
            let _ = gpu.free_tensor(down_theta);
            let _ = gpu.free_tensor(down_channel_scales);
        }
    }
}

impl Ffn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            Ffn::Dense(d) => d.free_gpu(gpu),
            Ffn::Moe(m) => m.free_gpu(gpu),
        }
    }
}

impl Cohere2MoeLayerWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Cohere2MoeLayerWeights {
            input_norm,
            wq,
            wk,
            wv,
            wo,
            ffn,
            attn_kind: _,
        } = self;
        let _ = gpu.free_tensor(input_norm);
        wq.free_all(gpu);
        wk.free_all(gpu);
        wv.free_all(gpu);
        wo.free_all(gpu);
        ffn.free_gpu(gpu);
    }
}

impl Cohere2MoeWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Cohere2MoeWeights {
            embed,
            embed_dtype: _,
            final_norm,
            lm_head,
            layers,
        } = self;
        let _ = gpu.free_tensor(embed);
        let _ = gpu.free_tensor(final_norm);
        lm_head.free_all(gpu);
        for layer in layers {
            layer.free_gpu(gpu);
        }
    }
}

// ──────────────────────────── State ────────────────────────────

/// Per-decode GPU scratch + KV cache (one slot per layer — every Cohere2 layer
/// is attention). Buffers are eager-allocated.
pub struct Cohere2MoeState {
    pub kv: KvCache,
    pub pos_buf: hip_bridge::DeviceBuffer, // device i32 position scalar
    pub max_seq: usize,
    pub n_tokens: usize,

    // residual + parallel-block normed input
    pub h: GpuTensor,      // [hidden] residual stream
    pub normed: GpuTensor, // [hidden] input_rmsnorm(h) — fed to BOTH branches

    // attention scratch
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]

    // dense-ffn scratch (layer 0)
    pub dense_gate: GpuTensor, // [dense_inter]
    pub dense_up: GpuTensor,   // [dense_inter]
    pub dense_act: GpuTensor,  // [dense_inter] silu(gate)*up

    // moe scratch
    pub ffn_x_rot: GpuTensor, // [hidden] FWHT(normed) for MQ4/MQ6 experts
    pub router_logits: GpuTensor, // [n_exp]
    pub topk_indices: GpuTensor, // [k_top] i32-in-F32
    pub topk_weights: GpuTensor, // [k_top]
    pub gate_batch: GpuTensor, // [k_top*moe_inter]
    pub up_batch: GpuTensor,  // [k_top*moe_inter]
    pub rot_batch: GpuTensor, // [k_top*moe_inter]
    pub down_expanded: GpuTensor, // [k_top*hidden]
    /// Per-expert scratch for the F16/Q8 (non-indexed) MoE path.
    pub expert_gate_up: GpuTensor, // [2*moe_inter]
    pub expert_act: GpuTensor, // [moe_inter] silu(gate)*up
    pub expert_down: GpuTensor, // [hidden]

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub logits: GpuTensor,         // [vocab]
    /// Flash-attention online-softmax partials, [n_heads · ceil(max_seq/128) ·
    /// (2+head_dim) · FLASH_PREFILL_SUBBATCH]. Enables the tiled (O(1)-LDS) flash
    /// Q8 attention used for BOTH decode and prefill — no seq-bound shared-memory
    /// ceiling, so long-context (file reads) no longer crashes the LDS-bound
    /// legacy kernel. The trailing factor IS the batched-prefill flash sub-batch
    /// size at full context (the wrapper computes sub_batch = capacity/per_pos =
    /// factor·max_tiles_alloc / max_tiles_actual): at full context that is the
    /// factor itself, so a too-small factor serializes long prefill to 1 query
    /// per launch. FLASH_PREFILL_SUBBATCH=64 keeps it batched (~68 MB for North).
    pub flash_partials: GpuTensor,
}

/// Batched-prefill flash sub-batch size at full context — the trailing factor of
/// the `flash_partials` allocation. Larger = fewer, bigger flash launches during
/// long prefill (see `flash_partials` doc); 64 ≈ 68 MB for North-Mini-Code.
const FLASH_PREFILL_SUBBATCH: usize = 64;

/// Default KV-cache window when the caller doesn't request one. North supports
/// `max_position_embeddings` = 500k, but the KV is allocated up front (~53 KB /
/// token), so we default GENEROUSLY but not to the full 500k (≈26 GB): 32k
/// (~1.7 GB) handles typical agentic long-context (large file reads) out of the
/// box. The daemon honours an explicit larger `max_seq` up to MAX_REQUESTED_SEQ
/// (512k) via `new_with_max_seq`. (The old 8k default was a bring-up-era cap.)
const DEFAULT_MAX_SEQ: usize = 32_768;

impl Cohere2MoeState {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Cohere2MoeState {
            kv,
            pos_buf,
            max_seq: _,
            n_tokens: _,
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            dense_gate,
            dense_up,
            dense_act,
            ffn_x_rot,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            expert_gate_up,
            expert_act,
            expert_down,
            final_norm_buf,
            logits,
            flash_partials,
        } = self;
        let _ = kv.free_gpu(gpu);
        let _ = gpu.hip.free(pos_buf);
        for t in [
            h,
            normed,
            fa_q,
            fa_k,
            fa_v,
            fa_attn_out,
            dense_gate,
            dense_up,
            dense_act,
            ffn_x_rot,
            router_logits,
            topk_indices,
            topk_weights,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
            expert_gate_up,
            expert_act,
            expert_down,
            final_norm_buf,
            logits,
            flash_partials,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }

    pub fn new(gpu: &mut Gpu, cfg: &Cohere2MoeConfig) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(DEFAULT_MAX_SEQ);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Cohere2MoeConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let dense_inter = cfg.dense_intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let k = cfg.num_experts_per_tok;

        // FWHT sign LUT must exist before any rotate_x_mq / fused rotate kernel.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("cohere2moe: ensure_mq_signs: {e:?}"))?;

        // One KV slot per layer (every layer is attention).
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

        Ok(Cohere2MoeState {
            kv,
            pos_buf,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            normed: alloc(gpu, hidden, "normed")?,
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
            expert_gate_up: alloc(gpu, 2 * moe_inter, "expert_gate_up")?,
            expert_act: alloc(gpu, moe_inter, "expert_act")?,
            expert_down: alloc(gpu, hidden, "expert_down")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
            flash_partials: alloc(
                gpu,
                cfg.num_attention_heads
                    * ((max_seq + 127) / 128)
                    * (2 + cfg.head_dim)
                    * FLASH_PREFILL_SUBBATCH,
                "flash_partials",
            )?,
        })
    }

    /// Reset for a fresh conversation. Rewinds the KV cursor AND zeros the KV
    /// buffers. The cursor rewind alone is sufficient for correctness (cohere2moe
    /// is pure attention with no recurrent/compressed state — unlike lfm2moe's
    /// `conv_states` — so the next cold prefill overwrites every attended slot
    /// and the stale tail is never read), but zeroing the buffers makes the reset
    /// holistic: no prior-conversation KV can survive even under a future
    /// window/LCP edge. Every daemon `reset()` call site also clears
    /// `conversation_tokens`, so a zeroed slot can never be stale-LCP-reused.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.n_tokens = 0;
        self.kv
            .clear_gpu(gpu)
            .map_err(|e| format!("cohere2moe reset: clear kv: {e:?}"))?;
        Ok(())
    }
}
