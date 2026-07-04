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
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::quant::f16_to_f32;
use hipfire_runtime::weights::WeightTensor;
use hipfire_rdna::{DType, Gpu, GpuTensor};
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
// ── Opus Quant (OQ4G256 / OQ8G256) on-disk → kernel-layout repack ──────────
// Mirrors lfm2moe.rs (which mirrors qwen35); the on-disk HFQM block layout is
// shared across arches, so the repack is identical. OQ4 is the W4A8 int4 format
// (`DType::Oq4G256`), OQ8 the W8A8 int8 format (`DType::Oq8G256`) — both run on
// the Opus iu8 grouped-WMMA kernel family `weight_gemv` already dispatches.
const OQ4_QT: u8 = hipfire_runtime::quant::QuantType::Oq4G256.code();
const OQ8_QT: u8 = hipfire_runtime::quant::QuantType::Oq8G256.code();
const OQPLUS_COMPACT_QT: u8 = hipfire_runtime::quant::QuantType::OqPlusCompact.code();
const OQ_GROUP: usize = 256;

/// Sign-extend a 4-bit nibble (0..15 → -8..7).
fn sext4(nib: u8) -> i8 {
    let v = (nib & 0x0f) as i8;
    if v > 7 {
        v - 16
    } else {
        v
    }
}

fn oq4_arch_combined_len(m: usize, k: usize) -> usize {
    let ng = k / OQ_GROUP;
    m * (k / 2) + m * ng * 4 + m * ng * (4 + 128)
}

/// On-disk OQ4 block [f16 scale | 128 nibbles] → arch-combined layout
/// [packed nibbles | f32 scales | interleaved f32-scale+nibble blocks].
fn pack_oq4_arch_combined(data: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    // Single-sourced from hipfire-quant-format (WP-3.3): Oq4G256 = 130.
    const BLOCK: usize = hipfire_runtime::quant::QuantType::Oq4G256
        .block_bytes()
        .unwrap();
    const INTERLEAVED_BLOCK: usize = 132;
    if k % OQ_GROUP != 0 {
        return Err(format!("OQ4G256 requires K % 256 == 0 (got K={k})"));
    }
    let ng = k / OQ_GROUP;
    let expect = m * ng * BLOCK;
    if data.len() != expect {
        return Err(format!(
            "OQ4G256 byte length {} != M*ng*130 = {expect} (M={m} K={k})",
            data.len()
        ));
    }
    let packed_bytes = m * (k / 2);
    let scales_bytes = m * ng * 4;
    let il_base = packed_bytes + scales_bytes;
    let mut out = vec![0u8; oq4_arch_combined_len(m, k)];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let split_dst = r * (k / 2) + g * (OQ_GROUP / 2);
            out[split_dst..split_dst + 128].copy_from_slice(&data[src + 2..src + BLOCK]);
            let scale_dst = packed_bytes + (r * ng + g) * 4;
            out[scale_dst..scale_dst + 4].copy_from_slice(&scale.to_le_bytes());
            let il_dst = il_base + (r * ng + g) * INTERLEAVED_BLOCK;
            out[il_dst..il_dst + 4].copy_from_slice(&scale.to_le_bytes());
            out[il_dst + 4..il_dst + INTERLEAVED_BLOCK]
                .copy_from_slice(&data[src + 2..src + BLOCK]);
        }
    }
    Ok(out)
}

/// On-disk OQ8 block [f16 scale | 256 i8] → [i8 weights | f32 scales].
fn pack_oq8(data: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    // Single-sourced from hipfire-quant-format (WP-3.3): Oq8G256 = 258.
    const BLOCK: usize = hipfire_runtime::quant::QuantType::Oq8G256
        .block_bytes()
        .unwrap();
    if k % OQ_GROUP != 0 {
        return Err(format!("OQ8G256 requires K % 256 == 0 (got K={k})"));
    }
    let ng = k / OQ_GROUP;
    let expect = m * ng * BLOCK;
    if data.len() != expect {
        return Err(format!(
            "OQ8G256 byte length {} != M*ng*258 = {expect} (M={m} K={k})",
            data.len()
        ));
    }
    let weight_bytes = m * k;
    let mut out = vec![0u8; weight_bytes + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * BLOCK;
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let dst = r * k + g * OQ_GROUP;
            out[dst..dst + OQ_GROUP].copy_from_slice(&data[src + 2..src + BLOCK]);
            let scale_dst = weight_bytes + (r * ng + g) * 4;
            out[scale_dst..scale_dst + 4].copy_from_slice(&scale.to_le_bytes());
        }
    }
    Ok(out)
}

fn upload_wt_oq(
    gpu: &mut Gpu,
    data: &[u8],
    dtype: DType,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let mut buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    buf.dtype = dtype;
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

/// Repack one expert's on-disk OQ4G256 tensor (130 B blocks `[f16 scale | 128
/// nibbles]`) into the indexed-MoE kernel block layout (132 B `[f32 scale | 128
/// nibbles]`) that `gemv_oq4g256_moe_*` reads. The 128 signed-nibble payload is
/// byte-identical; only the scale widens f16 → f32. Called per expert before
/// fusing w1/w3/w2 into the per-layer gate_up/down blobs.
fn oq4_ondisk_to_moe_blocks(data: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    // Single-sourced from hipfire-quant-format (WP-3.3): Oq4G256 = 130.
    const SRC_BLK: usize = hipfire_runtime::quant::QuantType::Oq4G256
        .block_bytes()
        .unwrap();
    const DST_BLK: usize = 132;
    if k % OQ_GROUP != 0 {
        return Err(format!("OQ4 expert requires K % 256 == 0 (got K={k})"));
    }
    let ng = k / OQ_GROUP;
    let expect = m * ng * SRC_BLK;
    if data.len() != expect {
        return Err(format!(
            "OQ4 expert byte length {} != M*ng*130 = {expect} (M={m} K={k})",
            data.len()
        ));
    }
    let mut out = vec![0u8; m * ng * DST_BLK];
    for blk in 0..(m * ng) {
        let src = blk * SRC_BLK;
        let dst = blk * DST_BLK;
        let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
        out[dst..dst + 4].copy_from_slice(&scale.to_le_bytes());
        out[dst + 4..dst + DST_BLK].copy_from_slice(&data[src + 2..src + SRC_BLK]);
    }
    Ok(out)
}

/// Expand one expert's on-disk OqPlusCompact tensor (qt=36; per group
/// `[f16 scale | 128 int4 nibbles | N_out × (u8 idx, i8 val)]`, 130 + 2·N_out B)
/// into the OQ8 indexed-MoE kernel layout (260 B `[f32 scale | 256 int8]`). The
/// int4 bulk is sign-extended into int8 and the sparse int8 outliers overlaid —
/// this is the "top-w8_frac weights → int8" tier expanded to a uniform int8
/// runtime weight `gemv_oq8g256_moe_*` reads. N_out is derived from the block
/// stride (uniform across a layer's experts at a fixed w8_frac).
fn oqplus_compact_to_moe_oq8_blocks(data: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    const DST_BLK: usize = 260; // [f32 scale | 256 int8]
    if k % OQ_GROUP != 0 {
        return Err(format!("OQ+C expert requires K % 256 == 0 (got K={k})"));
    }
    let ng = k / OQ_GROUP;
    let n_groups = m * ng;
    if n_groups == 0 || data.is_empty() || data.len() % n_groups != 0 {
        return Err(format!(
            "OQ+C expert byte length {} not divisible by n_groups {n_groups} (M={m} K={k})",
            data.len()
        ));
    }
    let block_bytes = data.len() / n_groups;
    if block_bytes < 132 || (block_bytes - 130) % 2 != 0 {
        return Err(format!(
            "OQ+C expert block_bytes {block_bytes} invalid (expected 130 + 2·N_out)"
        ));
    }
    let n_out = (block_bytes - 130) / 2;
    let mut out = vec![0u8; n_groups * DST_BLK];
    for blk in 0..n_groups {
        let src = blk * block_bytes;
        let dst = blk * DST_BLK;
        let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
        out[dst..dst + 4].copy_from_slice(&scale.to_le_bytes());
        // int4 bulk → int8 (the kernel reads bytes as signed char).
        for i in 0..128 {
            let byte = data[src + 2 + i];
            out[dst + 4 + 2 * i] = sext4(byte & 0x0f) as u8;
            out[dst + 4 + 2 * i + 1] = sext4(byte >> 4) as u8;
        }
        // Overlay the sparse int8 outliers.
        let tbl = src + 130;
        for s in 0..n_out {
            let idx = data[tbl + 2 * s] as usize;
            let val = data[tbl + 2 * s + 1];
            out[dst + 4 + idx] = val;
        }
    }
    Ok(out)
}

fn awq_scale_name(weight_name: &str) -> String {
    match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{weight_name}.awq_scale.weight"),
    }
}

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

/// Like [`load_wt`] but also attaches the OQ4 AWQ smoothing sidecar
/// (`<name>.awq_scale.weight`) when present — needed for OQ4+/OQ4++ dense
/// projections. Harmless (no-op) for formats with no sidecar.
fn load_wt_with_awq(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let mut w = load_wt(hfq, gpu, name, m, k)?;
    w.awq_scale = load_mm_awq_scale(hfq, gpu, &awq_scale_name(name), k);
    Ok(w)
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
    match qt {
        OQ4_QT => {
            return upload_wt_oq(
                gpu,
                &pack_oq4_arch_combined(data, m, k)?,
                DType::Oq4G256,
                m,
                k,
            )
        }
        OQ8_QT => return upload_wt_oq(gpu, &pack_oq8(data, m, k)?, DType::Oq8G256, m, k),
        _ => {}
    }
    // Pure (upload-and-tag) formats route through the shared canonical map in
    // hipfire_runtime::quant; the OQ arch-repack formats were handled above.
    let dtype = hipfire_runtime::quant::dtype_for_quant_type(qt, k)
        .ok_or_else(|| format!("unsupported quant_type {qt}"))?;
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
    /// Optional LQER low-rank correction for the down projection (`None` unless
    /// the .hfq carries `.w{1,3,2}.lr_u/.lr_v` sidecars). gate=w1→gate_batch,
    /// up=w3→up_batch (shared input), down=w2→down output (per-expert input).
    pub gate_lr: Option<MiniMaxLowRank>,
    pub up_lr: Option<MiniMaxLowRank>,
    pub down_lr: Option<MiniMaxLowRank>,
}

pub struct MiniMaxExpertWeights {
    /// Fused gate(w1)‖up(w3): [2*intermediate, hidden] MQ4G256.
    pub gate_up: WeightTensor,
    /// Down (w2): [hidden, intermediate] MQ4G256.
    pub down: WeightTensor,
}

/// Packed per-expert low-rank error correction (LQER) for one projection.
/// `u_data` holds all experts' U_e[m_out×r] f32 contiguously, `v_data` all
/// V_e[r×k_in]; `u_ptrs`/`v_ptrs` are the [2*n_exp] device-pointer tables the
/// indexed kernels index by routed expert id. The forward adds U_e·(V_e·x_e) to
/// the projection output. Used independently for gate (w1), up (w3), and down
/// (w2) — each on-disk tensor carries its own `.lr_u/.lr_v` sidecars.
pub struct MiniMaxLowRank {
    pub u_data: GpuTensor,
    pub v_data: GpuTensor,
    pub u_ptrs: GpuTensor,
    pub v_ptrs: GpuTensor,
    pub rank: usize,
}

impl MiniMaxLowRank {
    fn free(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.u_data);
        let _ = gpu.free_tensor(self.v_data);
        let _ = gpu.free_tensor(self.u_ptrs);
        let _ = gpu.free_tensor(self.v_ptrs);
    }
}

/// Per-projection low-rank sidecar accumulator: appends each owned expert's
/// `<base>.lr_u/.lr_v` (raw f32) into packed blobs, validating a uniform rank.
/// `rank == 0` ⇒ the .hfq carried no sidecars (correction disabled). Built once
/// per projection in the expert loop, then `finish`ed into a [`MiniMaxLowRank`].
#[derive(Default)]
struct LrAccum {
    u: Vec<u8>,
    v: Vec<u8>,
    u_stride: usize,
    v_stride: usize,
    rank: usize,
}

impl LrAccum {
    /// Read `<base>.lr_u/.lr_v` for one owned expert and append. `m_out` is the
    /// projection's output-row count (U is [m_out×r]); `cap` reserves for all
    /// owned experts. No-op if the sidecars are absent or not F32.
    fn push(&mut self, hfq: &HfqFile, base: &str, m_out: usize, cap: usize) -> Result<(), String> {
        let f32_qt = hipfire_runtime::quant::QuantType::F32.code();
        let (Ok((qtu, u)), Ok((qtv, v))) = (
            read_tensor(hfq, &format!("{base}.lr_u.weight")),
            read_tensor(hfq, &format!("{base}.lr_v.weight")),
        ) else {
            return Ok(());
        };
        if qtu != f32_qt || qtv != f32_qt {
            return Ok(());
        }
        let r = u.len() / 4 / m_out.max(1);
        if self.rank == 0 {
            self.rank = r;
            self.u_stride = u.len();
            self.v_stride = v.len();
            self.u.reserve(u.len() * cap);
            self.v.reserve(v.len() * cap);
        }
        if r != self.rank || u.len() != self.u_stride || v.len() != self.v_stride {
            return Err(format!(
                "minimax: non-uniform lr stride for {base} (u {}/{}, v {}/{})",
                u.len(),
                self.u_stride,
                v.len(),
                self.v_stride
            ));
        }
        self.u.extend_from_slice(&u);
        self.v.extend_from_slice(&v);
        Ok(())
    }

    /// Upload the packed blobs and build per-expert pointer tables (mirroring the
    /// weight tables: owned → base + local*stride, non-owned → base since their
    /// input is 0). `None` when no sidecars were collected.
    fn finish(
        self,
        gpu: &mut Gpu,
        n_exp: usize,
        local_of_global: &[usize],
        owns: &dyn Fn(usize) -> bool,
        what: &str,
    ) -> Result<Option<MiniMaxLowRank>, String> {
        if self.rank == 0 {
            return Ok(None);
        }
        let u_data = gpu
            .upload_raw(&self.u, &[self.u.len()])
            .map_err(|e| format!("minimax: upload {what} lr_u: {e:?}"))?;
        let v_data = gpu
            .upload_raw(&self.v, &[self.v.len()])
            .map_err(|e| format!("minimax: upload {what} lr_v: {e:?}"))?;
        let u_base = u_data.buf.as_ptr() as u64;
        let v_base = v_data.buf.as_ptr() as u64;
        let mk = |base: u64, stride: usize| -> Vec<u8> {
            (0..n_exp)
                .flat_map(|e| {
                    let ptr = if owns(e) {
                        base + (local_of_global[e] * stride) as u64
                    } else {
                        base
                    };
                    ptr.to_ne_bytes()
                })
                .collect()
        };
        let u_bytes = mk(u_base, self.u_stride);
        let v_bytes = mk(v_base, self.v_stride);
        let u_ptrs = gpu
            .alloc_tensor(&[2 * n_exp], DType::F32)
            .map_err(|e| format!("minimax: alloc {what} lr_u ptrs: {e:?}"))?;
        let v_ptrs = gpu
            .alloc_tensor(&[2 * n_exp], DType::F32)
            .map_err(|e| format!("minimax: alloc {what} lr_v ptrs: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&u_ptrs.buf, &u_bytes)
            .map_err(|e| format!("minimax: htod {what} lr_u ptrs: {e:?}"))?;
        gpu.hip
            .memcpy_htod(&v_ptrs.buf, &v_bytes)
            .map_err(|e| format!("minimax: htod {what} lr_v ptrs: {e:?}"))?;
        Ok(Some(MiniMaxLowRank {
            u_data,
            v_data,
            u_ptrs,
            v_ptrs,
            rank: self.rank,
        }))
    }
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
        let lm_head = load_wt_with_awq(hfq, gpu, "lm_head.weight", cfg.vocab_size, hidden)?;

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
            let wq = load_wt_with_awq(
                hfq,
                gpu,
                &format!("{p}.self_attn.q_proj.weight"),
                q_dim,
                hidden,
            )?;
            let wk = load_wt_with_awq(
                hfq,
                gpu,
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let wv = load_wt_with_awq(
                hfq,
                gpu,
                &format!("{p}.self_attn.v_proj.weight"),
                kv_dim,
                hidden,
            )?;
            let wo = load_wt_with_awq(
                hfq,
                gpu,
                &format!("{p}.self_attn.o_proj.weight"),
                hidden,
                q_dim,
            )?;

            let router = load_wt_with_awq(
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
            // Optional LQER low-rank correction, accumulated per projection from
            // the per-expert `.w{1,3,2}.lr_u/.lr_v` f32 sidecars (present only when
            // the .hfq was quantized with HIPFIRE_LOWRANK_R>0). gate=w1, up=w3,
            // down=w2; empty (rank 0) ⇒ correction disabled.
            let mut gate_acc = LrAccum::default();
            let mut up_acc = LrAccum::default();
            let mut down_acc = LrAccum::default();
            // EP shard: only upload rank-owned experts into the compact blob.
            // `local_of_global[e]` maps a global expert id to its slot in the
            // compact (owned-only) blob, or usize::MAX if not owned by this rank.
            let owns = |e: usize| {
                shard
                    .map(|(s, rank)| s.owns_expert(rank, e))
                    .unwrap_or(true)
            };
            let mut local_of_global = vec![usize::MAX; n_exp];
            let mut n_owned = 0usize;
            for e in 0..n_exp {
                let ep = format!("{p}.block_sparse_moe.experts.{e}");
                let (qt1, w1) = read_tensor(hfq, &format!("{ep}.w1.weight"))?;
                let (_qt3, w3) = read_tensor(hfq, &format!("{ep}.w3.weight"))?;
                let (qt2, w2) = read_tensor(hfq, &format!("{ep}.w2.weight"))?;
                // OQ experts ship on-disk and are repacked per expert into the
                // indexed-MoE kernel layout before fusing. OQ4G256 (130 B) → 132 B
                // int4 blocks; OqPlusCompact (qt=36) expands int4-bulk+int8-outliers
                // → 260 B int8 blocks (top-w8_frac → int8 tier). w1/w3 are
                // [inter, hidden]; w2 is [hidden, inter].
                let (w1, w3, w2) = if qt1 == OQ4_QT {
                    (
                        oq4_ondisk_to_moe_blocks(&w1, inter, hidden)?,
                        oq4_ondisk_to_moe_blocks(&w3, inter, hidden)?,
                        oq4_ondisk_to_moe_blocks(&w2, hidden, inter)?,
                    )
                } else if qt1 == OQPLUS_COMPACT_QT {
                    (
                        oqplus_compact_to_moe_oq8_blocks(&w1, inter, hidden)?,
                        oqplus_compact_to_moe_oq8_blocks(&w3, inter, hidden)?,
                        oqplus_compact_to_moe_oq8_blocks(&w2, hidden, inter)?,
                    )
                } else {
                    (w1, w3, w2)
                };
                let gu_len = w1.len() + w3.len();
                if e == 0 {
                    gu_stride = gu_len;
                    dn_stride = w2.len();
                    qt_gu = qt1;
                    qt_dn = qt2;
                    let cap = shard
                        .map(|(s, _)| s.experts_per_rank(n_exp))
                        .unwrap_or(n_exp);
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
                    // LQER low-rank sidecars per projection (gate=w1, up=w3 carry
                    // [inter×r] output rows; down=w2 carries [hidden×r]). Absent
                    // sidecars are a no-op.
                    let cap = shard
                        .map(|(s, _)| s.experts_per_rank(n_exp))
                        .unwrap_or(n_exp);
                    gate_acc.push(hfq, &format!("{ep}.w1"), inter, cap)?;
                    up_acc.push(hfq, &format!("{ep}.w3"), inter, cap)?;
                    down_acc.push(hfq, &format!("{ep}.w2"), hidden, cap)?;
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
            // OQ4 expert blobs are already in 132 B kernel layout (repacked per
            // expert above), so upload them raw as Oq4G256 — NOT through
            // wt_from_raw, whose OQ4 arm would re-run the dense arch-combined
            // repack. Other dtypes (MQ*) are byte-identical on disk → kernel.
            let mut gate_up = if qt_gu == OQ4_QT {
                upload_wt_oq(gpu, &gu_combined, DType::Oq4G256, 2 * inter, hidden)
            } else if qt_gu == OQPLUS_COMPACT_QT {
                upload_wt_oq(gpu, &gu_combined, DType::Oq8G256, 2 * inter, hidden)
            } else {
                wt_from_raw(gpu, qt_gu, &gu_combined, 2 * inter, hidden)
            }
            .map_err(|e2| format!("minimax: pack gate_up L{l}: {e2}"))?;
            let mut down = if qt_dn == OQ4_QT {
                upload_wt_oq(gpu, &dn_combined, DType::Oq4G256, hidden, inter)
            } else if qt_dn == OQPLUS_COMPACT_QT {
                upload_wt_oq(gpu, &dn_combined, DType::Oq8G256, hidden, inter)
            } else {
                wt_from_raw(gpu, qt_dn, &dn_combined, hidden, inter)
            }
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
            // Down low-rank U/V: upload the packed blobs and build per-expert
            // pointer tables mirroring `dn_bytes` (non-owned reuse the base — their
            // rotated input is 0 so the correction is 0). `None` when no sidecars.
            let gate_lr = gate_acc.finish(gpu, n_exp, &local_of_global, &owns, "gate")?;
            let up_lr = up_acc.finish(gpu, n_exp, &local_of_global, &owns, "up")?;
            let down_lr = down_acc.finish(gpu, n_exp, &local_of_global, &owns, "down")?;
            if l == 0 {
                if let Some(r) = gate_lr.as_ref().or(down_lr.as_ref()).map(|x| x.rank) {
                    eprintln!(
                        "minimax: low-rank correction active (rank {r}; gate={} up={} down={})",
                        gate_lr.is_some(),
                        up_lr.is_some(),
                        down_lr.is_some()
                    );
                }
            }
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
                gate_lr,
                up_lr,
                down_lr,
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

impl MiniMaxExpertWeights {
    /// Free this expert's packed gate_up + down buffers. `WeightTensor::free_all`
    /// also drops any AWQ sidecar and non-aliased ParoQuant rotation.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.gate_up.free_all(gpu);
        self.down.free_all(gpu);
    }
}

impl MiniMaxLayerWeights {
    pub fn free_gpu(mut self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.attn_norm);
        let _ = gpu.free_tensor(self.ffn_norm);
        let _ = gpu.free_tensor(self.q_norm);
        let _ = gpu.free_tensor(self.k_norm);
        self.wq.free_all(gpu);
        self.wk.free_all(gpu);
        self.wv.free_all(gpu);
        self.wo.free_all(gpu);
        self.router.free_all(gpu);
        let _ = gpu.free_tensor(self.routing_bias);
        for e in self.experts.drain(..) {
            e.free_gpu(gpu);
        }
        // Pointer tables (device addresses into the packed expert blobs), not the
        // blobs themselves — freed once here, no double-free with `experts`.
        let _ = gpu.free_tensor(self.expert_gate_up_ptrs);
        let _ = gpu.free_tensor(self.expert_down_ptrs);
        for lr in [self.gate_lr, self.up_lr, self.down_lr]
            .into_iter()
            .flatten()
        {
            lr.free(gpu);
        }
    }
}

impl MiniMaxWeights {
    /// Return every GPU buffer this model owns to the pool. Required because the
    /// MiniMaxWeights backend has no Drop, so `unload_model` must free explicitly
    /// or the weights (the bulk of VRAM) leak across a load/unload cycle.
    pub fn free_gpu(mut self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed);
        let _ = gpu.free_tensor(self.final_norm);
        self.lm_head.free_all(gpu);
        for l in self.layers.drain(..) {
            l.free_gpu(gpu);
        }
    }
}

impl MiniMaxState {
    /// Free the KV cache + all per-step scratch buffers + the device position
    /// scalar. Paired with `MiniMaxWeights::free_gpu` in `unload_model`.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.kv.free_gpu(gpu);
        let _ = gpu.hip.free(self.pos_buf);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.free_tensor(self.x_rot);
        let _ = gpu.free_tensor(self.fa_q);
        let _ = gpu.free_tensor(self.fa_k);
        let _ = gpu.free_tensor(self.fa_v);
        let _ = gpu.free_tensor(self.fa_attn_out);
        let _ = gpu.free_tensor(self.flash_partials);
        let _ = gpu.free_tensor(self.h);
        let _ = gpu.free_tensor(self.ffn_tmp);
        let _ = gpu.free_tensor(self.ffn_x_rot);
        let _ = gpu.free_tensor(self.router_logits);
        let _ = gpu.free_tensor(self.topk_indices);
        let _ = gpu.free_tensor(self.topk_weights);
        let _ = gpu.free_tensor(self.gate_batch);
        let _ = gpu.free_tensor(self.up_batch);
        let _ = gpu.free_tensor(self.rot_batch);
        let _ = gpu.free_tensor(self.down_expanded);
        let _ = gpu.free_tensor(self.final_norm_buf);
        let _ = gpu.free_tensor(self.final_rot);
        let _ = gpu.free_tensor(self.logits);
    }
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
