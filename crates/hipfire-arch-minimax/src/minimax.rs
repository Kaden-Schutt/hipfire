// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MiniMax-M2 config / weights / state.
//!
//! Config parses from the HFQ `metadata_json` envelope. Weights/State mirror
//! the qwen35 GQA+MoE infrastructure (shared `WeightTensor`, `KvCache`, and the
//! `gemv_hfq4g256_moe_*` indexed-expert kernels) rather than deepseek4's MLA.
//! Expert weights ship pre-split (w1/w2/w3) in the HFQ; the loader byte-fuses
//! w1‖w3 into the per-expert `gate_up` blob the indexed GEMV kernels expect.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, KvCache, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde::Deserialize;

// ───────────────────────────── Config ─────────────────────────────

/// Typed MiniMax-M2 shape constants.
#[derive(Clone, Debug)]
pub struct MiniMaxConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Expert (MoE) FFN intermediate size (HF `intermediate_size`).
    pub intermediate_size: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    /// Rotated-dim count for partial RoPE (`rotary_dim`, < head_dim).
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    /// Per-layer QK-norm on the flat q/k projection (RMSNorm pre-reshape).
    pub use_qk_norm: bool,
    /// Router uses `e_score_correction_bias` for top-k selection.
    pub use_routing_bias: bool,
    /// Router score activation; MiniMax-M2 = "sigmoid".
    pub scoring_func: String,
    /// MTP draft modules (spec-decode; 0 for the base forward / this ckpt).
    pub num_mtp_modules: usize,
}

#[derive(Deserialize)]
struct RawMiniMaxConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    intermediate_size: usize,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default = "default_rotary_dim")]
    rotary_dim: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default)]
    use_qk_norm: bool,
    #[serde(default)]
    use_routing_bias: bool,
    #[serde(default = "default_scoring")]
    scoring_func: String,
    #[serde(default)]
    num_mtp_modules: usize,
}

fn default_rotary_dim() -> usize {
    64
}
fn default_rope_theta() -> f32 {
    5_000_000.0
}
fn default_eps() -> f32 {
    1e-6
}
fn default_max_pos() -> usize {
    196_608
}
fn default_scoring() -> String {
    "sigmoid".to_string()
}

impl MiniMaxConfig {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("minimax: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "minimax: metadata_json missing `config` wrapper".to_string())?;
        let raw: RawMiniMaxConfig = serde_json::from_value(inner.clone())
            .map_err(|e| format!("minimax: parsing inner config failed: {e}"))?;
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        Ok(MiniMaxConfig {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            intermediate_size: raw.intermediate_size,
            num_local_experts: raw.num_local_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            rotary_dim: raw.rotary_dim,
            rope_theta: raw.rope_theta,
            rms_norm_eps: raw.rms_norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            use_qk_norm: raw.use_qk_norm,
            use_routing_bias: raw.use_routing_bias,
            scoring_func: raw.scoring_func,
            num_mtp_modules: raw.num_mtp_modules,
        })
    }

    /// q projection output width (n_heads * head_dim).
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    /// k/v projection output width (n_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

// ───────────────────────── HFQ load helpers ─────────────────────────
// Replicated from the qwen35 loader (those are crate-private). MiniMax HFQ
// files carry RAW HF tensor names, so we look them up by exact name.

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("minimax: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

/// Load a 1D norm vector (F16/F32) → F32 GpuTensor. MiniMax-M2 uses STANDARD
/// RMSNorm (`weight * x_normed`, no +1.0 offset — verified against
/// MiniMaxM2RMSNorm), so no offset is baked in.
fn load_norm(
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
        _ => {
            return Err(format!(
                "minimax: expected F16/F32 norm for {name}, got qt={qt}"
            ))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("minimax: upload norm {name}: {e:?}"))
}

/// Load a MiniMax AWQ shared-scale sidecar (1D F16, length k) → F32 GpuTensor.
fn load_mm_awq_scale(hfq: &HfqFile, gpu: &mut Gpu, name: &str, k: usize) -> Option<GpuTensor> {
    let (qt, data) = read_tensor(hfq, name).ok()?;
    if qt != 1 {
        return None;
    } // 1 = F16
    if data.len() != k * 2 {
        eprintln!(
            "minimax AWQ sidecar {name}: {} bytes != {} (k*2); skipping",
            data.len(),
            k * 2
        );
        return None;
    }
    let f32_data: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    gpu.upload_raw(&f32_bytes, &[f32_data.len()]).ok()
}

/// Load a quantized 2D weight → WeightTensor, tagging gpu_dtype from quant_type.
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("minimax: load_wt {name}: {e}"))
}

/// quant_type → DType mapping (subset used by MiniMax HFQ files; mirrors
/// qwen35::load_weight_tensor_raw). Uploads raw bytes and tags the dtype.
fn wt_from_raw(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
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

/// Per-layer GPU-resident weights.
pub struct MiniMaxLayerWeights {
    pub attn_norm: GpuTensor, // input_layernorm
    pub ffn_norm: GpuTensor,  // post_attention_layernorm
    pub q_norm: GpuTensor,    // [n_heads*head_dim]
    pub k_norm: GpuTensor,    // [n_kv*head_dim]
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub router: WeightTensor, // block_sparse_moe.gate.weight [n_exp, hidden]
    pub routing_bias: GpuTensor, // e_score_correction_bias [n_exp] F32
    pub experts: Vec<MiniMaxExpertWeights>,
    pub expert_gate_up_ptrs: GpuTensor, // [2*n_exp] F32 = n_exp u64 device ptrs
    pub expert_down_ptrs: GpuTensor,
}

pub struct MiniMaxExpertWeights {
    /// Fused gate(w1)‖up(w3): [2*intermediate, hidden] MQ4G256.
    pub gate_up: WeightTensor,
    /// Down (w2): [hidden, intermediate] MQ4G256.
    pub down: WeightTensor,
}

pub struct MiniMaxWeights {
    pub embed: GpuTensor, // model.embed_tokens.weight (Q8 raw, for embedding_lookup_q8)
    pub final_norm: GpuTensor, // model.norm.weight
    pub lm_head: WeightTensor, // lm_head.weight
    pub layers: Vec<MiniMaxLayerWeights>,
}

impl MiniMaxWeights {
    /// Load MiniMax weights. `shard = Some((cfg, rank))` enables **EP shard-aware
    /// loading**: each layer's experts are read from the file but ONLY the
    /// rank-owned experts are uploaded into a compact packed blob (so an 86 GB
    /// model fits across N×32 GB cards — load-then-free is impossible since the
    /// experts are one packed blob too big for a single card). Non-owned expert
    /// pointers point at a shared zeroed gate_up buffer (→ 0 contribution). The
    /// non-expert weights (embed / lm_head / attention / norms) are always loaded
    /// in full (replicated per rank). `shard = None` loads everything (single-GPU).
    pub fn load(
        hfq: &mut HfqFile,
        cfg: &MiniMaxConfig,
        gpu: &mut Gpu,
        shard: Option<(&hipfire_runtime::tp_shard::ShardConfig, usize)>,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let n_exp = cfg.num_local_experts;

        // Globals.
        let (_qt, embed_bytes) = read_tensor(hfq, "model.embed_tokens.weight")?;
        let embed = gpu
            .upload_raw(&embed_bytes, &[embed_bytes.len()])
            .map_err(|e| format!("minimax: upload embed: {e:?}"))?;
        let final_norm = load_norm(hfq, gpu, "model.norm.weight", &[hidden])?;
        let lm_head = load_wt(hfq, gpu, "lm_head.weight", cfg.vocab_size, hidden)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let attn_norm = load_norm(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[hidden])?;
            let ffn_norm = load_norm(
                hfq,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
            )?;
            let q_norm = load_norm(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), &[q_dim])?;
            let k_norm = load_norm(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), &[kv_dim])?;
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

            let router = load_wt(
                hfq,
                gpu,
                &format!("{p}.block_sparse_moe.gate.weight"),
                n_exp,
                hidden,
            )?;
            // e_score_correction_bias: [n_exp] F16 → F32 (kept F16 in HFQ).
            let routing_bias = load_norm(
                hfq,
                gpu,
                &format!("{p}.block_sparse_moe.e_score_correction_bias"),
                &[n_exp],
            )?;

            // Routed experts: pack ALL experts of this layer into ONE gate_up
            // blob + ONE down blob (deepseek4 `upload_layer_routed_experts`
            // pattern). The old code did a separate `upload_raw`/hipMalloc per
            // expert per projection — 2*n_exp tiny allocs/layer, ~31.7k total,
            // each rounded up to HIP's allocation granularity. That fragmentation
            // wasted ~20GB of VRAM, inflating mq2-lloyd's 86GB file to a ~114GB
            // resident footprint that OOM'd gfx1151's 96GB carveout. The
            // `*_indexed` GEMV kernels index experts by device pointer, so one
            // packed blob + a base+e*stride pointer table is byte- and
            // result-identical to the per-expert layout (validated against the
            // tiny oracle: gfx1151 cosine unchanged).
            let mut gu_combined: Vec<u8> = Vec::new();
            let mut dn_combined: Vec<u8> = Vec::new();
            let mut gu_stride = 0usize;
            let mut dn_stride = 0usize;
            let mut qt_gu = 0u8;
            let mut qt_dn = 0u8;
            // EP shard: only upload rank-owned experts into the compact blob.
            // `local_of_global[e]` maps a global expert id to its slot in the
            // compact (owned-only) blob, or usize::MAX if not owned by this rank.
            let owns = |e: usize| shard.map(|(s, rank)| s.owns_expert(rank, e)).unwrap_or(true);
            let mut local_of_global = vec![usize::MAX; n_exp];
            let mut n_owned = 0usize;
            for e in 0..n_exp {
                let ep = format!("{p}.block_sparse_moe.experts.{e}");
                let (qt1, w1) = read_tensor(hfq, &format!("{ep}.w1.weight"))?;
                let (_qt3, w3) = read_tensor(hfq, &format!("{ep}.w3.weight"))?;
                let (qt2, w2) = read_tensor(hfq, &format!("{ep}.w2.weight"))?;
                let gu_len = w1.len() + w3.len();
                if e == 0 {
                    gu_stride = gu_len;
                    dn_stride = w2.len();
                    qt_gu = qt1;
                    qt_dn = qt2;
                    let cap = shard.map(|(s, _)| s.experts_per_rank(n_exp)).unwrap_or(n_exp);
                    gu_combined.reserve(gu_len * cap);
                    dn_combined.reserve(w2.len() * cap);
                } else if gu_len != gu_stride || w2.len() != dn_stride {
                    return Err(format!(
                        "minimax L{l}E{e}: non-uniform expert stride (gate_up {gu_len}/{gu_stride}, down {}/{dn_stride}); packed layout requires equal-size experts",
                        w2.len()
                    ));
                }
                if owns(e) {
                    local_of_global[e] = n_owned;
                    n_owned += 1;
                    gu_combined.extend_from_slice(&w1);
                    gu_combined.extend_from_slice(&w3);
                    dn_combined.extend_from_slice(&w2);
                }
                // Non-owned: w1/w3/w2 read from the file (for stride validation)
                // then dropped — never uploaded. That is the EP memory win.
            }
            if n_owned == 0 {
                return Err(format!("minimax L{l}: shard rank owns no experts"));
            }
            // One allocation per projection. The representative `WeightTensor`'s
            // buffer IS the packed blob; its m/k describe a SINGLE expert's shape
            // (the forward's rotate_x_mq / silu_mul_rotate / dtype dispatch read
            // those + the AWQ scale, never the buffer's full extent — per-expert
            // data is reached through the pointer table below).
            let mut gate_up = wt_from_raw(gpu, qt_gu, &gu_combined, 2 * inter, hidden)
                .map_err(|e2| format!("minimax: pack gate_up L{l}: {e2}"))?;
            let mut down = wt_from_raw(gpu, qt_dn, &dn_combined, hidden, inter)
                .map_err(|e2| format!("minimax: pack down L{l}: {e2}"))?;
            drop(gu_combined);
            drop(dn_combined);
            gate_up.awq_scale = load_mm_awq_scale(
                hfq,
                gpu,
                &format!("{p}.block_sparse_moe.awq_scale_gate_up.weight"),
                hidden,
            );
            if std::env::var_os("HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ").is_some() {
                // down-AWQ harmful (shared s_down bad approx); opt-in
                down.awq_scale = load_mm_awq_scale(
                    hfq,
                    gpu,
                    &format!("{p}.block_sparse_moe.awq_scale_down.weight"),
                    inter,
                );
            }
            if gate_up.awq_scale.is_some() {
                eprintln!("minimax: AWQ scales attached at L{l} (shared per-layer)");
            }
            let gu_base = gate_up.buf.buf.as_ptr() as u64;
            let dn_base = down.buf.buf.as_ptr() as u64;
            let experts = vec![MiniMaxExpertWeights { gate_up, down }];

            // Device pointer tables: n_exp u64 device addresses, stored as
            // [2*n_exp] F32 (8 bytes/ptr). Single-GPU: base + e*stride into the
            // full packed blob. EP shard: owned e → compact-blob slot
            // (base + local*stride); non-owned e → a shared ZEROED gate_up buffer
            // (→ 0 output ⇒ 0 contribution; down ptr is irrelevant since its rot
            // input is 0, so it reuses the compact down base).
            let dummy_gu = if shard.is_some() && n_owned < n_exp {
                let z = gpu
                    .zeros(&[gu_stride / 4], DType::F32)
                    .map_err(|e| format!("minimax L{l}: zero gate_up dummy: {e:?}"))?;
                let p = z.buf.as_ptr() as u64;
                std::mem::forget(z); // leaked for model lifetime (process teardown reclaims)
                p
            } else {
                gu_base
            };
            let gu_bytes: Vec<u8> = (0..n_exp)
                .flat_map(|e| {
                    let ptr = if owns(e) {
                        gu_base + (local_of_global[e] * gu_stride) as u64
                    } else {
                        dummy_gu
                    };
                    ptr.to_ne_bytes()
                })
                .collect();
            let dn_bytes: Vec<u8> = (0..n_exp)
                .flat_map(|e| {
                    let ptr = if owns(e) {
                        dn_base + (local_of_global[e] * dn_stride) as u64
                    } else {
                        dn_base // rot input is 0 for non-owned ⇒ output 0 regardless
                    };
                    ptr.to_ne_bytes()
                })
                .collect();
            let expert_gate_up_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("minimax: alloc gu_ptrs: {e:?}"))?;
            let expert_down_ptrs = gpu
                .alloc_tensor(&[2 * n_exp], DType::F32)
                .map_err(|e| format!("minimax: alloc dn_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
                .map_err(|e| format!("minimax: htod gu_ptrs: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
                .map_err(|e| format!("minimax: htod dn_ptrs: {e:?}"))?;

            layers.push(MiniMaxLayerWeights {
                attn_norm,
                ffn_norm,
                q_norm,
                k_norm,
                wq,
                wk,
                wv,
                wo,
                router,
                routing_bias,
                experts,
                expert_gate_up_ptrs,
                expert_down_ptrs,
            });
        }

        Ok(MiniMaxWeights {
            embed,
            final_norm,
            lm_head,
            layers,
        })
    }
}

// ──────────────────────────── State ────────────────────────────

/// Per-decode GPU scratch + KV cache. Buffers are eager-allocated (the model
/// is dense in its per-token working set); the KV cache is Q8.
pub struct MiniMaxState {
    pub kv: KvCache,
    pub pos_buf: hip_bridge::DeviceBuffer, // device i32 position scalar
    /// Stable host source for the device position scalar. The hipGraph decode
    /// path captures a `memcpy_htod_auto` from these bytes; the captured node
    /// re-reads this heap-stable `Box` on every replay (see
    /// `decode_step_with_graph`). Updated host-side before each `graph_launch`.
    pub pos_host: Box<[i32]>,
    pub max_seq: usize,
    pub n_tokens: usize,
    /// hipGraph warmup gate: the first decode after a fresh load runs eager
    /// (no capture) to JIT-compile kernels + settle DPM, then the next call
    /// captures. Survives turn resets (the graph stays valid for the same
    /// model — only weight pointers + device buffers are baked, and those are
    /// stable across turns).
    pub ar_warmed_up: bool,

    // attention scratch
    pub tmp: GpuTensor,         // [hidden] rmsnorm(h)
    pub x_rot: GpuTensor,       // [hidden] FWHT scratch (unused for Q8 attn)
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]
    pub flash_partials: GpuTensor,

    // residual + embedding
    pub h: GpuTensor, // [hidden] residual stream

    // moe scratch
    pub ffn_tmp: GpuTensor,       // [hidden] rmsnorm(h)
    pub ffn_x_rot: GpuTensor,     // [hidden] FWHT(rmsnorm(h)) for MQ4 experts
    pub router_logits: GpuTensor, // [n_exp]
    pub topk_indices: GpuTensor,  // [k] i32-in-F32
    pub topk_weights: GpuTensor,  // [k]
    pub gate_batch: GpuTensor,    // [k*inter]
    pub up_batch: GpuTensor,      // [k*inter]
    pub rot_batch: GpuTensor,     // [k*inter]
    pub down_expanded: GpuTensor, // [k*hidden]

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub final_rot: GpuTensor,      // [hidden]
    pub logits: GpuTensor,         // [vocab]
}

impl MiniMaxState {
    pub fn new(gpu: &mut Gpu, cfg: &MiniMaxConfig) -> Result<Self, String> {
        // Cap the KV cache so the real 204800-ctx config doesn't OOM; callers
        // that need a specific window use `new_with_max_seq`.
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &MiniMaxConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        // `attention_q8_0_kv` (single-token decode) stages its per-head score
        // buffer in LDS sized by `max_seq`: `(max_seq + block + head_dim) * 4`
        // bytes must fit the 64 KB per-block shared-memory limit on every RDNA
        // arch, so the single-token attention launch is hard-bounded near 16K
        // context. A larger requested window blows the launch
        // (`hipModuleLaunchKernel: invalid argument` — observed serving the
        // 86 GB mq2-lloyd on gfx1151 with the daemon's default window: prefill
        // via the batched kernel succeeds, then the first decode token dies).
        // Clamp the served window here so the cache, the geometry hint, and the
        // flash-partial sizing all stay launch-valid. Proper fix = tile the
        // scores out of LDS (flash-style); tracked as a follow-up.
        const MINIMAX_ATTN_LDS_MAX_SEQ: usize = 12288;
        let max_seq = if max_seq > MINIMAX_ATTN_LDS_MAX_SEQ {
            eprintln!(
                "[minimax] requested max_seq {max_seq} exceeds the single-token \
                 attention LDS bound; clamping to {MINIMAX_ATTN_LDS_MAX_SEQ} \
                 (decode scores must fit the 64 KB per-block shared-mem limit)"
            );
            MINIMAX_ATTN_LDS_MAX_SEQ
        } else {
            max_seq
        };
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let n_exp = cfg.num_local_experts;
        let k = cfg.num_experts_per_tok;

        // FWHT sign LUT must exist before any rotate_x_mq / fused rotate kernel.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("minimax: ensure_mq_signs: {e:?}"))?;

        let kv = KvCache::new_gpu_q8(
            gpu,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq,
        )
        .map_err(|e| format!("minimax: kv cache: {e:?}"))?;
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("minimax: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("minimax: alloc {label}: {e:?}"))
        };
        // Flash-attn partials: [n_heads * max_tiles * (2+head_dim)]; max_tiles
        // bounded by ceil(max_seq/tile). Use a generous tile bound of 64.
        let max_tiles = (max_seq / 256).max(1) + 1;
        let flash_partials = alloc(
            gpu,
            cfg.num_attention_heads * max_tiles * (2 + cfg.head_dim),
            "flash_partials",
        )?;

        Ok(MiniMaxState {
            kv,
            pos_buf,
            pos_host: vec![0i32; 1].into_boxed_slice(),
            max_seq,
            n_tokens: 0,
            ar_warmed_up: false,
            tmp: alloc(gpu, hidden, "tmp")?,
            x_rot: alloc(gpu, hidden, "x_rot")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            flash_partials,
            h: alloc(gpu, hidden, "h")?,
            ffn_tmp: alloc(gpu, hidden, "ffn_tmp")?,
            ffn_x_rot: alloc(gpu, hidden, "ffn_x_rot")?,
            router_logits: alloc(gpu, n_exp, "router_logits")?,
            topk_indices: alloc(gpu, k, "topk_indices")?,
            topk_weights: alloc(gpu, k, "topk_weights")?,
            gate_batch: alloc(gpu, k * inter, "gate_batch")?,
            up_batch: alloc(gpu, k * inter, "up_batch")?,
            rot_batch: alloc(gpu, k * inter, "rot_batch")?,
            down_expanded: alloc(gpu, k * hidden, "down_expanded")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            final_rot: alloc(gpu, hidden, "final_rot")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
        })
    }

    pub fn reset(&mut self) {
        self.n_tokens = 0;
    }
}
