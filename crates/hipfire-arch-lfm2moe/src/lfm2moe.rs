// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5-MoE weights + decode state.
//!
//! HFQ files carry RAW HF tensor names; the loader looks each up by exact
//! name (no rename). Mirrors the MiniMax-M2 loader (shared `WeightTensor`,
//! `KvCache`, indexed-MoE GEMV kernels) and adds:
//!   * a per-layer mixer split (conv vs attention) from `layer_types`,
//!   * a per-layer FFN split (dense SwiGLU vs top-4 MoE) from `num_dense_layers`,
//!   * a rolling conv-state cache (one [hidden,(K-1)] f32 ring buffer per conv
//!     layer) — the conv analog of the KV cache.
//!
//! Expert weights ship pre-split (w1/w2/w3); the loader byte-fuses w1‖w3 into
//! the per-expert `gate_up` blob the indexed GEMV kernels expect (exactly the
//! minimax convention). lm_head is tied to embed_tokens.

use crate::config::{Lfm2MoeConfig, MixerKind};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::quant::f16_to_f32;
use hipfire_runtime::weights::WeightTensor;
use hipfire_rdna::{DType, Gpu, GpuTensor};

// ───────────────────────── HFQ load helpers ─────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("lfm2moe: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Load an LFM2 AWQ shared gate_up-scale sidecar (1D F16, length k) → F32
/// GpuTensor. Mirror of minimax `load_mm_awq_scale`. Returns None if the
/// sidecar is absent or malformed, so non-AWQ models load cleanly (the
/// attached `awq_scale` stays None and `rotate_x_mq_for` takes the plain path).
fn load_lfm2_awq_scale(hfq: &HfqFile, gpu: &mut Gpu, name: &str, k: usize) -> Option<GpuTensor> {
    let (qt, data) = read_tensor(hfq, name).ok()?;
    if qt != 1 {
        return None;
    } // 1 = F16
    if data.len() != k * 2 {
        eprintln!(
            "lfm2moe AWQ sidecar {name}: {} bytes != {} (k*2); skipping",
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

/// Load a 1D/raw F16/F32 vector → F32 GpuTensor with the given shape.
/// LFM2 uses STANDARD RMSNorm (`weight * x̂`, no +1 offset — verified against
/// Lfm2MoeRMSNorm), so no offset is baked in. Also used for the depthwise conv
/// filter ([hidden, K]) and the F32 expert_bias.
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
        16 => data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        3 => {
            // Q8_0: 32-elem blocks [f16 scale | 32 i8]. Dequant to f32.
            dequant_q8_0(&data)
        }
        _ => {
            return Err(format!(
                "lfm2moe: expected F16/F32/Q8/BF16 for {name}, got qt={qt}"
            ))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("lfm2moe: upload {name}: {e:?}"))
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
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("lfm2moe: load_wt {name}: {e}"))
}

fn awq_scale_name(weight_name: &str) -> String {
    match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{weight_name}.awq_scale.weight"),
    }
}

fn load_wt_with_awq(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let mut w = load_wt(hfq, gpu, name, m, k)?;
    let sidecar = awq_scale_name(name);
    w.awq_scale = load_lfm2_awq_scale(hfq, gpu, &sidecar, k);
    if w.awq_scale.is_some() {
        eprintln!("lfm2moe: AWQ scale attached for {name}");
    }
    Ok(w)
}

const OQ_PLUS_QT: u8 = hipfire_runtime::quant::QuantType::OqPlusG256.code();
const OQ4_QT: u8 = hipfire_runtime::quant::QuantType::Oq4G256.code();
const OQ8_QT: u8 = hipfire_runtime::quant::QuantType::Oq8G256.code();
const OQ_PLUS_COMPACT_QT: u8 = hipfire_runtime::quant::QuantType::OqPlusCompact.code();
const OQ4_ARCH_PACKED_QT: u8 = hipfire_runtime::quant::QuantType::Oq4G256ArchPacked.code();
const OQ_GROUP: usize = 256;

fn oq4_arch_combined_len(m: usize, k: usize) -> usize {
    let ng = k / OQ_GROUP;
    m * (k / 2) + m * ng * 4 + m * ng * (4 + 128)
}

fn sext4(nib: u8) -> i8 {
    let v = (nib & 0x0f) as i8;
    if v > 7 {
        v - 16
    } else {
        v
    }
}

fn pack_oq4_arch_combined(data: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    // Single-sourced from hipfire-quant-format (WP-3.3): Oq4G256 = 130.
    const BLOCK: usize = hipfire_runtime::quant::QuantType::Oq4G256
        .block_bytes()
        .unwrap();
    const INTERLEAVED_BLOCK: usize = 132; // [f32 scale][128 nibbles]
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

fn expand_oq_plus_to_oq8(data: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    // Single-sourced from hipfire-quant-format (WP-3.3): the OqPlus fixed base
    // block is the Oq4G256 geometry = 130.
    const BLOCK: usize = hipfire_runtime::quant::QuantType::Oq4G256
        .block_bytes()
        .unwrap();
    if k % OQ_GROUP != 0 {
        return Err(format!("OQPLUS requires K % 256 == 0 (got K={k})"));
    }
    let ng = k / OQ_GROUP;
    let expect = m * ng * BLOCK;
    if data.len() != expect {
        return Err(format!(
            "OQPLUS byte length {} != M*ng*130 = {expect} (M={m} K={k})",
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
            for i in 0..128 {
                let byte = data[src + 2 + i];
                out[dst + 2 * i] = sext4(byte & 0x0f) as u8;
                out[dst + 2 * i + 1] = sext4(byte >> 4) as u8;
            }
            let scale_dst = weight_bytes + (r * ng + g) * 4;
            out[scale_dst..scale_dst + 4].copy_from_slice(&scale.to_le_bytes());
        }
    }
    Ok(out)
}

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

fn expand_oq_plus_compact_to_oq8(data: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    if k % OQ_GROUP != 0 {
        return Err(format!("OQ+C requires K % 256 == 0 (got K={k})"));
    }
    let ng = k / OQ_GROUP;
    let n_groups = m * ng;
    if n_groups == 0 || data.is_empty() || data.len() % n_groups != 0 {
        return Err(format!(
            "OQ+C byte length {} not divisible by n_groups {n_groups}",
            data.len()
        ));
    }
    let block_bytes = data.len() / n_groups;
    if block_bytes < 132 || (block_bytes - 130) % 2 != 0 {
        return Err(format!(
            "OQ+C block_bytes {block_bytes} invalid (expected 130 + 2*N_out)"
        ));
    }
    let n_out = (block_bytes - 130) / 2;
    let weight_bytes = m * k;
    let mut out = vec![0u8; weight_bytes + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let src = (r * ng + g) * block_bytes;
            let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
            let dst = r * k + g * OQ_GROUP;
            for i in 0..128 {
                let byte = data[src + 2 + i];
                out[dst + 2 * i] = sext4(byte & 0x0f) as u8;
                out[dst + 2 * i + 1] = sext4(byte >> 4) as u8;
            }
            let tbl = src + 130;
            for s in 0..n_out {
                let idx = data[tbl + 2 * s] as usize;
                let val = data[tbl + 2 * s + 1];
                out[dst + idx] = val;
            }
            let scale_dst = weight_bytes + (r * ng + g) * 4;
            out[scale_dst..scale_dst + 4].copy_from_slice(&scale.to_le_bytes());
        }
    }
    Ok(out)
}

fn upload_wt_raw(
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

/// quant_type → DType mapping (mirrors minimax::wt_from_raw); uploads raw
/// bytes and tags the dtype for kernel dispatch.
fn wt_from_raw(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    match qt {
        OQ_PLUS_QT => {
            return upload_wt_raw(
                gpu,
                &expand_oq_plus_to_oq8(data, m, k)?,
                DType::Oq8G256,
                m,
                k,
            )
        }
        OQ4_QT => {
            return upload_wt_raw(
                gpu,
                &pack_oq4_arch_combined(data, m, k)?,
                DType::Oq4G256,
                m,
                k,
            )
        }
        OQ8_QT => return upload_wt_raw(gpu, &pack_oq8(data, m, k)?, DType::Oq8G256, m, k),
        OQ_PLUS_COMPACT_QT => {
            return upload_wt_raw(
                gpu,
                &expand_oq_plus_compact_to_oq8(data, m, k)?,
                DType::Oq8G256,
                m,
                k,
            )
        }
        OQ4_ARCH_PACKED_QT => {
            let expect = oq4_arch_combined_len(m, k);
            if data.len() != expect {
                return Err(format!(
                    "OQ4 arch-packed byte length {} != combined len {expect} (M={m} K={k})",
                    data.len()
                ));
            }
            return upload_wt_raw(gpu, data, DType::Oq4G256, m, k);
        }
        _ => {}
    }
    let dtype = match qt {
        // bf16 is tagged in-place here (arch-local convention); all pure
        // upload-and-tag formats route through the shared canonical map.
        16 => DType::BF16,
        _ => hipfire_runtime::quant::dtype_for_quant_type(qt, k)
            .ok_or_else(|| format!("unsupported quant_type {qt}"))?,
    };
    upload_wt_raw(gpu, data, dtype, m, k)
}

// ──────────────────────────── Weights ────────────────────────────

/// LIV short-conv mixer weights.
pub struct ConvWeights {
    /// in_proj: [3*hidden, hidden] — produces B | C_gate | x.
    pub in_proj: WeightTensor,
    /// Depthwise causal filter, squeezed [hidden, K] F32.
    pub conv_weight: GpuTensor,
    /// out_proj: [hidden, hidden].
    pub out_proj: WeightTensor,
    /// Index into the conv-state ring-buffer cache.
    pub conv_state_idx: usize,
}

/// GQA attention mixer weights (per-head QK-norm + full-dim rotate_half RoPE).
pub struct AttnWeights {
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    /// Per-head QK-norm weight, [head_dim].
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    /// Index into the KV cache.
    pub kv_idx: usize,
}

pub enum Mixer {
    Conv(ConvWeights),
    Attention(AttnWeights),
}

/// Dense SwiGLU MLP (first `num_dense_layers` layers).
pub struct DenseFfn {
    pub w1: WeightTensor, // gate [inter, hidden]
    pub w3: WeightTensor, // up   [inter, hidden]
    pub w2: WeightTensor, // down [hidden, inter]
}

/// One MoE expert: fused gate(w1)‖up(w3) and down(w2).
pub struct ExpertWeights {
    pub gate_up: WeightTensor, // [2*moe_inter, hidden]
    pub down: WeightTensor,    // [hidden, moe_inter]
}

/// Top-4 MoE FFN (sigmoid + expert_bias routing).
pub struct MoeFfn {
    pub router: WeightTensor,        // feed_forward.gate.weight [n_exp, hidden]
    pub expert_bias: GpuTensor,      // feed_forward.expert_bias [n_exp] F32
    pub experts: Vec<ExpertWeights>, // keep alive (buffers owned here)
    pub expert_gate_up_ptrs: GpuTensor, // [2*n_exp] F32 = n_exp u64 device ptrs
    pub expert_down_ptrs: GpuTensor,
}

pub enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

pub struct Lfm2MoeLayerWeights {
    pub operator_norm: GpuTensor, // pre-mixer RMSNorm
    pub ffn_norm: GpuTensor,      // pre-FFN RMSNorm
    pub mixer: Mixer,
    pub ffn: Ffn,
}

pub struct Lfm2MoeWeights {
    pub embed: GpuTensor, // model.embed_tokens.weight (raw, for embedding_lookup)
    pub embedding_norm: GpuTensor, // model.embedding_norm.weight (final norm)
    pub lm_head: WeightTensor, // tied = embed_tokens (loaded as Q8 weight)
    pub layers: Vec<Lfm2MoeLayerWeights>,
}

impl ConvWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.in_proj.free_all(gpu);
        let _ = gpu.free_tensor(self.conv_weight);
        self.out_proj.free_all(gpu);
    }
}

impl AttnWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.wq.free_all(gpu);
        self.wk.free_all(gpu);
        self.wv.free_all(gpu);
        self.wo.free_all(gpu);
        let _ = gpu.free_tensor(self.q_norm);
        let _ = gpu.free_tensor(self.k_norm);
    }
}

impl Mixer {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            Mixer::Conv(c) => c.free_gpu(gpu),
            Mixer::Attention(a) => a.free_gpu(gpu),
        }
    }
}

impl DenseFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.w1.free_all(gpu);
        self.w3.free_all(gpu);
        self.w2.free_all(gpu);
    }
}

impl ExpertWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.gate_up.free_all(gpu);
        self.down.free_all(gpu);
    }
}

impl MoeFfn {
    pub fn free_gpu(mut self, gpu: &mut Gpu) {
        self.router.free_all(gpu);
        let _ = gpu.free_tensor(self.expert_bias);
        // Experts own their own buffers (one allocation per expert), so each is
        // freed exactly once; any shared ParoQuant rotation is skipped via the
        // `is_alias` check in `WeightTensor::free_all`.
        for e in self.experts.drain(..) {
            e.free_gpu(gpu);
        }
        // Pointer tables (device addresses), not the expert blobs themselves.
        let _ = gpu.free_tensor(self.expert_gate_up_ptrs);
        let _ = gpu.free_tensor(self.expert_down_ptrs);
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

impl Lfm2MoeLayerWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.operator_norm);
        let _ = gpu.free_tensor(self.ffn_norm);
        self.mixer.free_gpu(gpu);
        self.ffn.free_gpu(gpu);
    }
}

impl Lfm2MoeWeights {
    /// Return every GPU buffer this model owns to the pool. Required because the
    /// LFM2 weights have no Drop, so `unload_model` must free explicitly or the
    /// weights (the bulk of VRAM) leak across a load/unload cycle.
    pub fn free_gpu(mut self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed);
        let _ = gpu.free_tensor(self.embedding_norm);
        self.lm_head.free_all(gpu);
        for l in self.layers.drain(..) {
            l.free_gpu(gpu);
        }
    }
}

impl Lfm2MoeWeights {
    pub fn load(hfq: &mut HfqFile, cfg: &Lfm2MoeConfig, gpu: &mut Gpu) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let head_dim = cfg.head_dim;
        let dense_inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let k_conv = cfg.conv_kernel_size;

        // Globals. embed_tokens is the shared (tied) lm_head.
        let (_eqt, embed_bytes) = read_tensor(hfq, "model.embed_tokens.weight")?;
        let embed = gpu
            .upload_raw(&embed_bytes, &[embed_bytes.len()])
            .map_err(|e| format!("lfm2moe: upload embed: {e:?}"))?;
        let embedding_norm = load_f32(hfq, gpu, "model.embedding_norm.weight", &[hidden])?;
        // lm_head: tied → reuse embed_tokens.weight as a Q8 weight tensor.
        let lm_head = load_wt(
            hfq,
            gpu,
            "model.embed_tokens.weight",
            cfg.vocab_size,
            hidden,
        )?;

        let mut conv_state_count = 0usize;
        let mut kv_count = 0usize;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            hipfire_runtime::load_progress::report(
                l as u32 + 1,
                cfg.num_hidden_layers as u32,
                "weights",
            );
            let p = format!("model.layers.{l}");
            let operator_norm =
                load_f32(hfq, gpu, &format!("{p}.operator_norm.weight"), &[hidden])?;
            let ffn_norm = load_f32(hfq, gpu, &format!("{p}.ffn_norm.weight"), &[hidden])?;

            // ── Mixer: conv vs attention ──────────────────────────────────
            let mixer = match cfg.mixer(l) {
                MixerKind::Conv => {
                    let in_proj = load_wt_with_awq(
                        hfq,
                        gpu,
                        &format!("{p}.conv.in_proj.weight"),
                        3 * hidden,
                        hidden,
                    )?;
                    // conv.conv.weight ships [hidden,1,K] → loaded flat as [hidden,K] f32.
                    let conv_weight = load_f32(
                        hfq,
                        gpu,
                        &format!("{p}.conv.conv.weight"),
                        &[hidden * k_conv],
                    )?;
                    let out_proj = load_wt_with_awq(
                        hfq,
                        gpu,
                        &format!("{p}.conv.out_proj.weight"),
                        hidden,
                        hidden,
                    )?;
                    let conv_state_idx = conv_state_count;
                    conv_state_count += 1;
                    Mixer::Conv(ConvWeights {
                        in_proj,
                        conv_weight,
                        out_proj,
                        conv_state_idx,
                    })
                }
                MixerKind::Attention => {
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
                        &format!("{p}.self_attn.out_proj.weight"),
                        hidden,
                        q_dim,
                    )?;
                    // Per-HEAD QK-norm: weight is [head_dim], applied to each head.
                    let q_norm = load_f32(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.q_layernorm.weight"),
                        &[head_dim],
                    )?;
                    let k_norm = load_f32(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.k_layernorm.weight"),
                        &[head_dim],
                    )?;
                    let kv_idx = kv_count;
                    kv_count += 1;
                    Mixer::Attention(AttnWeights {
                        wq,
                        wk,
                        wv,
                        wo,
                        q_norm,
                        k_norm,
                        kv_idx,
                    })
                }
            };

            // ── FFN: dense SwiGLU vs top-4 MoE ────────────────────────────
            let ffn = if cfg.is_dense_ffn(l) {
                let w1 = load_wt_with_awq(
                    hfq,
                    gpu,
                    &format!("{p}.feed_forward.w1.weight"),
                    dense_inter,
                    hidden,
                )?;
                let w3 = load_wt_with_awq(
                    hfq,
                    gpu,
                    &format!("{p}.feed_forward.w3.weight"),
                    dense_inter,
                    hidden,
                )?;
                let w2 = load_wt_with_awq(
                    hfq,
                    gpu,
                    &format!("{p}.feed_forward.w2.weight"),
                    hidden,
                    dense_inter,
                )?;
                Ffn::Dense(DenseFfn { w1, w3, w2 })
            } else {
                let router = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.feed_forward.gate.weight"),
                    n_exp,
                    hidden,
                )?;
                let expert_bias =
                    load_f32(hfq, gpu, &format!("{p}.feed_forward.expert_bias"), &[n_exp])?;
                // Byte-fuse w1‖w3 → gate_up [2*moe_inter, hidden]; w2 → down.
                let mut experts = Vec::with_capacity(n_exp);
                for e in 0..n_exp {
                    let ep = format!("{p}.feed_forward.experts.{e}");
                    let (qt1, w1) = read_tensor(hfq, &format!("{ep}.w1.weight"))?;
                    let (_qt3, w3) = read_tensor(hfq, &format!("{ep}.w3.weight"))?;
                    let mut gate_up_bytes = w1;
                    gate_up_bytes.extend_from_slice(&w3);
                    let mut gate_up = wt_from_raw(gpu, qt1, &gate_up_bytes, 2 * moe_inter, hidden)
                        .map_err(|e2| format!("lfm2moe: fuse gate_up L{l}E{e}: {e2}"))?;
                    let (qt2, w2) = read_tensor(hfq, &format!("{ep}.w2.weight"))?;
                    let mut down = wt_from_raw(gpu, qt2, &w2, hidden, moe_inter)
                        .map_err(|e2| format!("lfm2moe: down L{l}E{e}: {e2}"))?;
                    // AWQ scales: shared per layer, emitted once on expert 0 by the
                    // quantizer (full port of minimax 3c676d00 BOTH-projection AWQ).
                    // Attach the gate_up scale (len hidden) to expert 0's gate_up and
                    // the down scale (len moe_inter) to expert 0's down; the forward
                    // reads both from experts[0] and divides x/s in the unrotated
                    // basis via the AWQ-aware rotate_x_mq_for (gate_up input) and
                    // fused_silu_mul_rotate_mq_batched_for (post-SwiGLU intermediate
                    // for down). down (w2) is the most quant-sensitive proj, so its
                    // AWQ is the whole point. No-op on non-AWQ files.
                    if e == 0 {
                        gate_up.awq_scale = load_lfm2_awq_scale(
                            hfq,
                            gpu,
                            &format!("{p}.feed_forward.awq_scale_gate_up.weight"),
                            hidden,
                        );
                        if gate_up.awq_scale.is_some() {
                            eprintln!("lfm2moe: AWQ gate_up scale attached at {p} (expert-0 representative)");
                        }
                        down.awq_scale = load_lfm2_awq_scale(
                            hfq,
                            gpu,
                            &format!("{p}.feed_forward.awq_scale_down.weight"),
                            moe_inter,
                        );
                        if down.awq_scale.is_some() {
                            eprintln!(
                                "lfm2moe: AWQ down scale attached at {p} (expert-0 representative)"
                            );
                        }
                    }
                    experts.push(ExpertWeights { gate_up, down });
                }
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
                    .map_err(|e| format!("lfm2moe: alloc gu_ptrs: {e:?}"))?;
                let expert_down_ptrs = gpu
                    .alloc_tensor(&[2 * n_exp], DType::F32)
                    .map_err(|e| format!("lfm2moe: alloc dn_ptrs: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)
                    .map_err(|e| format!("lfm2moe: htod gu_ptrs: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)
                    .map_err(|e| format!("lfm2moe: htod dn_ptrs: {e:?}"))?;
                Ffn::Moe(MoeFfn {
                    router,
                    expert_bias,
                    experts,
                    expert_gate_up_ptrs,
                    expert_down_ptrs,
                })
            };

            layers.push(Lfm2MoeLayerWeights {
                operator_norm,
                ffn_norm,
                mixer,
                ffn,
            });
        }

        let _ = (conv_state_count, kv_count);
        Ok(Lfm2MoeWeights {
            embed,
            embedding_norm,
            lm_head,
            layers,
        })
    }
}

// ──────────────────────────── State ────────────────────────────

/// Per-decode GPU scratch + KV cache (attention layers) + conv-state cache
/// (conv layers). Buffers are eager-allocated.
pub struct Lfm2MoeState {
    pub kv: KvCache,
    /// One rolling [hidden, K-1] f32 ring buffer per conv layer (zero-init).
    pub conv_states: Vec<GpuTensor>,
    pub pos_buf: hip_bridge::DeviceBuffer, // device i32 position scalar
    /// hipGraph (HIPFIRE_LFM2_GRAPH) warmup latch: false until the first
    /// decode runs direct (so kernel JIT / lazy alloc happen outside any
    /// stream capture). Unused when the graph path is disabled.
    pub graph_warmed_up: bool,
    pub max_seq: usize,
    pub n_tokens: usize,

    // residual + shared scratch
    pub h: GpuTensor,   // [hidden] residual stream
    pub tmp: GpuTensor, // [hidden] norm output (mixer input)

    // attention scratch
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]

    // conv scratch
    pub conv_bcx: GpuTensor, // [3*hidden] in_proj output (B|C|x)
    pub conv_y: GpuTensor,   // [hidden] gated conv output (out_proj input)

    // ffn scratch
    pub ffn_tmp: GpuTensor,       // [hidden] rmsnorm(h)
    pub ffn_x_rot: GpuTensor,     // [hidden] FWHT(rmsnorm(h)) for MQ4 experts
    pub dense_gate: GpuTensor,    // [dense_inter]
    pub dense_up: GpuTensor,      // [dense_inter]
    pub dense_act: GpuTensor,     // [dense_inter] silu(gate)*up
    pub router_logits: GpuTensor, // [n_exp]
    pub topk_indices: GpuTensor,  // [k_top] i32-in-F32
    pub topk_weights: GpuTensor,  // [k_top]
    pub gate_batch: GpuTensor,    // [k_top*moe_inter]
    pub up_batch: GpuTensor,      // [k_top*moe_inter]
    pub rot_batch: GpuTensor,     // [k_top*moe_inter]
    pub down_expanded: GpuTensor, // [k_top*hidden]

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub logits: GpuTensor,         // [vocab]
}

impl Lfm2MoeState {
    /// Free the KV cache + per-conv-layer ring buffers + all per-step scratch +
    /// the device position scalar. Paired with `Lfm2MoeWeights::free_gpu` in
    /// `unload_model`.
    pub fn free_gpu(mut self, gpu: &mut Gpu) {
        self.kv.free_gpu(gpu);
        for cs in self.conv_states.drain(..) {
            let _ = gpu.free_tensor(cs);
        }
        let _ = gpu.hip.free(self.pos_buf);
        let _ = gpu.free_tensor(self.h);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.free_tensor(self.fa_q);
        let _ = gpu.free_tensor(self.fa_k);
        let _ = gpu.free_tensor(self.fa_v);
        let _ = gpu.free_tensor(self.fa_attn_out);
        let _ = gpu.free_tensor(self.conv_bcx);
        let _ = gpu.free_tensor(self.conv_y);
        let _ = gpu.free_tensor(self.ffn_tmp);
        let _ = gpu.free_tensor(self.ffn_x_rot);
        let _ = gpu.free_tensor(self.dense_gate);
        let _ = gpu.free_tensor(self.dense_up);
        let _ = gpu.free_tensor(self.dense_act);
        let _ = gpu.free_tensor(self.router_logits);
        let _ = gpu.free_tensor(self.topk_indices);
        let _ = gpu.free_tensor(self.topk_weights);
        let _ = gpu.free_tensor(self.gate_batch);
        let _ = gpu.free_tensor(self.up_batch);
        let _ = gpu.free_tensor(self.rot_batch);
        let _ = gpu.free_tensor(self.down_expanded);
        let _ = gpu.free_tensor(self.final_norm_buf);
        let _ = gpu.free_tensor(self.logits);
    }
}

impl Lfm2MoeState {
    pub fn new(gpu: &mut Gpu, cfg: &Lfm2MoeConfig) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Lfm2MoeConfig,
        max_seq: usize,
    ) -> Result<Self, String> {
        Self::new_with_physical_cap(gpu, cfg, max_seq, max_seq)
    }

    pub fn new_with_physical_cap(
        gpu: &mut Gpu,
        cfg: &Lfm2MoeConfig,
        max_seq: usize,
        physical_cap: usize,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let dense_inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let k = cfg.num_experts_per_tok;
        let k_conv = cfg.conv_kernel_size;

        // FWHT sign LUT must exist before any rotate_x_mq / fused rotate kernel.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("lfm2moe: ensure_mq_signs: {e:?}"))?;

        // KV cache: one slot per ATTENTION layer (conv layers carry no KV).
        let n_attn = cfg.num_attention_layers().max(1);
        let kv = KvCache::new_gpu_q8_capped(
            gpu,
            n_attn,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq,
            physical_cap,
        )
        .map_err(|e| format!("lfm2moe: kv cache: {e:?}"))?;

        // Conv-state cache: one [hidden,(K-1)] f32 ring buffer per CONV layer.
        let conv_hist = hidden * (k_conv - 1);
        let zeros = vec![0u8; conv_hist * 4];
        let mut conv_states = Vec::with_capacity(cfg.num_conv_layers());
        for _ in 0..cfg.num_conv_layers() {
            let cs = gpu
                .alloc_tensor(&[conv_hist], DType::F32)
                .map_err(|e| format!("lfm2moe: alloc conv_state: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&cs.buf, &zeros)
                .map_err(|e| format!("lfm2moe: zero conv_state: {e:?}"))?;
            conv_states.push(cs);
        }

        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("lfm2moe: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("lfm2moe: alloc {label}: {e:?}"))
        };

        Ok(Lfm2MoeState {
            kv,
            conv_states,
            pos_buf,
            graph_warmed_up: false,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            tmp: alloc(gpu, hidden, "tmp")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            conv_bcx: alloc(gpu, 3 * hidden, "conv_bcx")?,
            conv_y: alloc(gpu, hidden, "conv_y")?,
            ffn_tmp: alloc(gpu, hidden, "ffn_tmp")?,
            ffn_x_rot: alloc(gpu, hidden, "ffn_x_rot")?,
            dense_gate: alloc(gpu, dense_inter, "dense_gate")?,
            dense_up: alloc(gpu, dense_inter, "dense_up")?,
            dense_act: alloc(gpu, dense_inter, "dense_act")?,
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

    /// Reset for a new sequence: clear conv state and token count.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.n_tokens = 0;
        self.kv.compact_offset = 0;
        for cs in &self.conv_states {
            let zeros = vec![0u8; cs.numel() * 4];
            gpu.hip
                .memcpy_htod(&cs.buf, &zeros)
                .map_err(|e| format!("lfm2moe: reset conv_state: {e:?}"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn awq_scale_name_matches_quantizer_sidecar_convention() {
        assert_eq!(
            awq_scale_name("model.layers.0.feed_forward.w1.weight"),
            "model.layers.0.feed_forward.w1.awq_scale.weight"
        );
        assert_eq!(
            awq_scale_name("model.layers.0.conv.in_proj.weight"),
            "model.layers.0.conv.in_proj.awq_scale.weight"
        );
    }

    #[test]
    fn oq4_pack_combined_preserves_split_and_interleaved_regions() {
        let mut data = vec![0u8; 130];
        data[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0
        for i in 0..128 {
            data[2 + i] = i as u8;
        }

        let packed = pack_oq4_arch_combined(&data, 1, 256).unwrap();
        assert_eq!(packed.len(), oq4_arch_combined_len(1, 256));
        assert_eq!(&packed[0..128], &data[2..130]);
        assert_eq!(&packed[128..132], &1.0f32.to_le_bytes());
        assert_eq!(&packed[132..136], &1.0f32.to_le_bytes());
        assert_eq!(&packed[136..264], &data[2..130]);
    }

    #[test]
    fn oqplus_expands_signed_nibbles_to_oq8_layout() {
        let mut data = vec![0u8; 130];
        data[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0
        data[2] = 0x78; // low nibble -8, high nibble +7

        let expanded = expand_oq_plus_to_oq8(&data, 1, 256).unwrap();
        assert_eq!(expanded.len(), 256 + 4);
        assert_eq!(expanded[0] as i8, -8);
        assert_eq!(expanded[1] as i8, 7);
        assert_eq!(&expanded[256..260], &1.0f32.to_le_bytes());
    }

    #[test]
    fn oq8_repack_moves_scales_after_dense_int8_weights() {
        let mut data = vec![0u8; 258];
        data[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // f16 0.5
        for i in 0..256 {
            data[2 + i] = i as u8;
        }

        let packed = pack_oq8(&data, 1, 256).unwrap();
        assert_eq!(packed.len(), 256 + 4);
        assert_eq!(&packed[0..256], &data[2..258]);
        assert_eq!(&packed[256..260], &0.5f32.to_le_bytes());
    }

    #[test]
    fn oqplus_compact_overlays_outliers_after_bulk_expand() {
        let mut data = vec![0u8; 132];
        data[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0
        data[2] = 0x11;
        data[130] = 1;
        data[131] = (-5i8) as u8;

        let expanded = expand_oq_plus_compact_to_oq8(&data, 1, 256).unwrap();
        assert_eq!(expanded.len(), 256 + 4);
        assert_eq!(expanded[0] as i8, 1);
        assert_eq!(expanded[1] as i8, -5);
        assert_eq!(&expanded[256..260], &1.0f32.to_le_bytes());
    }
}
