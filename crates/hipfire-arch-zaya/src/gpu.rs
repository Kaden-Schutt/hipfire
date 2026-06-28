// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 GPU forward. Loads each big linear from the HFQ in its quant format via
//! [`LinearWeight`] (f32 / Q8 / MQ4 / MQ6) and runs the forward op-for-op against
//! `cpu.rs`, reusing `gemv_f32`/the dispatched gemv/`rmsnorm_f32`/`silu_mul_f32`
//! plus the custom CCA/EDA kernels in `rdna-compute/src/dispatch/zaya_cca.rs`.
//! Batch 1. `gpu_forward_serve` prefills the whole prompt and primes a
//! [`ZayaDecodeState`] (KV cache + conv ring + delayed value); `gpu_decode` then
//! advances one token at a time (O(1) per token). `gpu_forward_prefill` is a
//! per-block-trace variant used only by the `gpu_golden` validation example.

use crate::ZayaConfig;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_runtime::calibration::{logsumexp, topk_logits};
use hipfire_runtime::hfq::{load_awq_scale, HfqFile};
use hipfire_runtime::quant::f16_to_f32;
use hipfire_runtime::weights::WeightTensor;
use rdna_compute::{DType, Gpu, GpuTensor};

/// quant_type byte → quantized linear `DType` (None ⇒ a plain precision handled
/// by `dequant_qt`). Matches the hipfire-quantize QuantType discriminants.
fn linear_dtype(qt: u8) -> Option<DType> {
    match qt {
        3 => Some(DType::Q8_0), // Q8F16
        6 => Some(DType::HFQ4G256),
        7 => Some(DType::HFQ4G128),
        13 => Some(DType::MQ4G256), // MagnumQuant 4-bit
        15 => Some(DType::MQ6G256), // MagnumQuant 6-bit (zaya experts under --format mq4)
        _ => None,
    }
}

// ─── Opus-Quant (OQ4/OQ8) repack-on-load ─────────────────────────────────────
// The OQ formats are WMMA-gated (halo/gfx1151). Unlike the verbatim-upload quant
// formats above, the on-disk OQ block layout must be transformed to the kernel's
// combined buffer at load time. These helpers mirror `hipfire-arch-gemma3`'s
// `weights.rs` byte-for-byte so the layout cannot drift between arches.

fn sext4(nib: u8) -> i8 {
    let v = (nib & 0xf) as i8;
    if v > 7 {
        v - 16
    } else {
        v
    }
}

/// Byte length of the arch-combined OQ4 buffer (`oq4_pack_arch_combined` output /
/// `oq4_repack` quant_type 37 on-disk form): packed nibbles [M,K/2] + f32 scales
/// [M,ng] + interleaved (scale,nibbles) block [M,ng,132].
fn oq4_arch_combined_len(m: usize, k: usize) -> usize {
    let ng = k / 256;
    m * (k / 2) + m * ng * 4 + m * ng * (4 + 128)
}

/// Canonical OQ4 (qt 34): `[f16 scale][128 nibbles]` per 256-group → the arch
/// combined device layout (split nibbles + f32 scales + interleaved region for
/// the decode GEMV). Forward derives the scale/interleaved pointers via
/// `sub_offset`; feeds `gemm_oq4_grouped_wmma`.
fn oq4_pack_arch_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    const BLOCK: usize = 130; // 2 (f16 scale) + 128 nibbles
    const ILB: usize = 132; // 4 (f32 scale) + 128 nibbles, interleaved
    assert_eq!(k % GROUP, 0, "OQ4 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let packed_bytes = m * (k / 2);
    let scales_bytes = m * ng * 4;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "OQ4 weight byte length {} != M*ng*130 = {expect} (M={m} K={k})",
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

/// OQ+ / Opus-Plus W4A8 (qt 33): on-disk bytes are IDENTICAL to OQ4, but the
/// nibbles are EXPANDED to int8 and tagged Oq8G256 so the W8A8 kernel runs with
/// int8 activations (weight values stay 4-bit). Layout = `[int8 M*K | f32 scales
/// M*ng]`.
fn oq4_to_oq8_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    const BLOCK: usize = 130;
    assert_eq!(k % GROUP, 0, "OQ4→8 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "OQ4→8 weight byte length {} != M*ng*130 = {expect} (M={m} K={k})",
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

/// Canonical OQ8 (qt 35): `[f16 scale][256 int8]` per 256-group → `[int8 M*K |
/// f32 scales M*ng]` for `gemm_oq8_grouped_wmma`.
fn oq8_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    const BLOCK: usize = 258; // 2 (f16 scale) + 256 int8
    assert_eq!(k % GROUP, 0, "OQ8 requires K % 256 == 0 (got K={k})");
    let ng = k / GROUP;
    let expect = m * ng * BLOCK;
    assert_eq!(
        data.len(),
        expect,
        "OQ8 weight byte length {} != M*ng*258 = {expect} (M={m} K={k})",
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

/// Repack an OQ-family on-disk tensor to its kernel buffer + gpu_dtype, or `None`
/// for non-OQ quant_types (handled by `linear_dtype` verbatim upload / f32).
/// qt 33=OQ+ (W4A8), 34=OQ4 (W4A4), 35=OQ8 (W8A8), 37=arch-packed OQ4 (verbatim).
fn oq_repack(qt: u8, data: &[u8], m: usize, k: usize) -> Option<(Vec<u8>, DType)> {
    match qt {
        33 => Some((oq4_to_oq8_combined(data, m, k), DType::Oq8G256)),
        34 => Some((oq4_pack_arch_combined(data, m, k), DType::Oq4G256)),
        35 => Some((oq8_combined(data, m, k), DType::Oq8G256)),
        37 => {
            assert_eq!(
                data.len(),
                oq4_arch_combined_len(m, k),
                "OQ4 arch-packed byte length {} != combined len (M={m} K={k})",
                data.len()
            );
            Some((data.to_vec(), DType::Oq4G256))
        }
        _ => None,
    }
}

/// A `[out, in]` linear weight: plain f32 (bf16/f16 hfq dequant) or a quantized
/// `WeightTensor` (mq4/mq6/q8). One `gemv` entry dispatches both — mirrors
/// `hipfire_arch_nemotron::weight::LinearWeight`.
pub enum LinearWeight {
    F32(GpuTensor),
    Quant(Box<WeightTensor>),
}

impl LinearWeight {
    /// `out[m] = W · x[k]`. F32 → `gemv_f32`; Quant → the dispatched gemv
    /// (auto-rotates for MQ-family, plain for HFQ/Q8).
    pub fn gemv(&self, gpu: &mut Gpu, x: &GpuTensor, out: &GpuTensor) -> Result<(), String> {
        match self {
            LinearWeight::F32(w) => gpu
                .gemv_f32(w, x, out)
                .map_err(|e| format!("zaya f32 gemv: {e:?}")),
            LinearWeight::Quant(wt) => {
                let ctx = DispatchCtx::new(gpu);
                execute_steps(
                    gpu,
                    &ctx,
                    &[Step::Gemv {
                        w: &wt.dispatch_ref(),
                        input: GemvInput::Raw(x),
                        out,
                    }],
                )
                .map_err(|e| format!("zaya quant gemv: {e}"))
            }
        }
    }

    /// The backing GPU buffer (for activation-capture name mapping). During
    /// calibration every linear is `F32` (bf16 model), but the `Quant` arm is
    /// handled for completeness.
    fn buf(&self) -> &GpuTensor {
        match self {
            LinearWeight::F32(t) => t,
            LinearWeight::Quant(wt) => &wt.buf,
        }
    }

    /// Release the GPU storage.
    fn free(self, gpu: &mut Gpu) {
        match self {
            LinearWeight::F32(w) => {
                let _ = gpu.free_tensor(w);
            }
            LinearWeight::Quant(wt) => wt.free_all(gpu),
        }
    }

    /// Embedding-row lookup (the weight doubles as the tied embedding table).
    /// The quantizer stores `embed_tokens.weight` as Q8 for every quant format;
    /// reject anything else rather than silently mis-dequantizing.
    fn embed_lookup(
        &self,
        gpu: &mut Gpu,
        out: &GpuTensor,
        token: u32,
        dim: usize,
    ) -> Result<(), String> {
        match self {
            LinearWeight::F32(t) => gpu
                .embedding_lookup(t, out, token, dim)
                .map_err(|e| format!("zaya embed lookup: {e:?}")),
            LinearWeight::Quant(wt) => {
                if wt.gpu_dtype != DType::Q8_0 {
                    return Err(format!(
                        "zaya embed: quantized embedding must be Q8, got {:?}",
                        wt.gpu_dtype
                    ));
                }
                gpu.embedding_lookup_q8(&wt.buf, out, token, dim)
                    .map_err(|e| format!("zaya q8 embed lookup: {e:?}"))
            }
        }
    }
}

/// Load a `[m, k]` linear as quantized (`WeightTensor`) when the hfq stores it
/// quantized, else an f32 upload.
fn load_linear(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> Result<LinearWeight, String> {
    let info = hfq
        .find_tensor_info(name)
        .ok_or_else(|| format!("zaya gpu: missing tensor {name:?}"))?;
    let shape: Vec<usize> = info.shape.iter().map(|&x| x as usize).collect();
    let (m, k) = (shape[0], shape.get(1).copied().unwrap_or(1));
    let qt = info.quant_type;
    let (_, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("zaya gpu: no data for {name:?}"))?;
    // OQ-family (33/34/35/37) repacks on load; the verbatim formats upload as-is
    // (no clone — the large Q8 embedding uploads straight from `data`).
    let buf_dtype: Option<(GpuTensor, DType)> =
        if let Some((bytes, dtype)) = oq_repack(qt, &data, m, k) {
            let buf = gpu
                .upload_raw(&bytes, &[bytes.len()])
                .map_err(|e| format!("zaya upload {name}: {e:?}"))?;
            Some((buf, dtype))
        } else if let Some(dtype) = linear_dtype(qt) {
            let buf = gpu
                .upload_raw(&data, &[data.len()])
                .map_err(|e| format!("zaya upload {name}: {e:?}"))?;
            Some((buf, dtype))
        } else {
            None
        };
    if let Some((buf, dtype)) = buf_dtype {
        let mut wt = WeightTensor {
            buf,
            gpu_dtype: dtype,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };
        // The OQ `+`/`++` calibration writes a per-channel `<name>.awq_scale`
        // sidecar; the gemv (`dispatch_ref`) divides x by it before the rotation.
        if wt.gpu_dtype.supports_awq_sidecar() {
            wt.awq_scale = load_awq_scale(hfq, gpu, name, k);
        }
        Ok(LinearWeight::Quant(Box::new(wt)))
    } else {
        let f = dequant_qt(qt, &data)?;
        let g = gpu
            .upload_f32(&f, &[m, k])
            .map_err(|e| format!("zaya f32 {name}: {e:?}"))?;
        Ok(LinearWeight::F32(g))
    }
}

fn dequant_qt(qt: u8, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match qt {
        1 => Ok(bytes
            .chunks_exact(2)
            .map(|c| {
                let h = u16::from_le_bytes([c[0], c[1]]);
                let s = (h >> 15) & 1;
                let e = (h >> 10) & 0x1f;
                let m = h & 0x3ff;
                let v = match e {
                    0 => (m as f32) * 2f32.powi(-24),
                    0x1f => {
                        if m == 0 {
                            f32::INFINITY
                        } else {
                            f32::NAN
                        }
                    }
                    _ => (1.0 + (m as f32) / 1024.0) * 2f32.powi(e as i32 - 15),
                };
                if s == 1 {
                    -v
                } else {
                    v
                }
            })
            .collect()),
        2 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        other => Err(format!("zaya gpu: unsupported quant_type {other}")),
    }
}

/// Upload an HFQ tensor (bf16/f16/f32) as an f32 GpuTensor with its stored shape.
fn up(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> Result<GpuTensor, String> {
    let info = hfq
        .find_tensor_info(name)
        .ok_or_else(|| format!("zaya gpu: missing tensor {name:?}"))?;
    let shape: Vec<usize> = info.shape.iter().map(|&x| x as usize).collect();
    let qt = info.quant_type;
    let (_, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("zaya gpu: no data for {name:?}"))?;
    let f = dequant_qt(qt, &data)?;
    let shape = if shape.is_empty() {
        vec![f.len()]
    } else {
        shape
    };
    gpu.upload_f32(&f, &shape)
        .map_err(|e| format!("zaya gpu upload {name}: {e:?}"))
}

struct GpuExpert {
    gate_up: LinearWeight, // [2*moe_int, hidden]
    down: LinearWeight,    // [hidden, moe_int]
}

struct GpuLayer {
    input_ln: GpuTensor,
    post_attn_ln: GpuTensor,
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_cur: LinearWeight,
    v_del: LinearWeight,
    conv_dw_w: GpuTensor,
    conv_dw_b: GpuTensor,
    conv_gr_w: GpuTensor,
    conv_gr_b: GpuTensor,
    qk_temp: GpuTensor,
    o_proj: LinearWeight,
    // router
    down_proj_w: LinearWeight,
    down_proj_b: GpuTensor,
    router_states_scale: Option<GpuTensor>,
    rnorm_w: GpuTensor,
    fc1_w: LinearWeight,
    fc1_b: GpuTensor,
    fc2_w: LinearWeight,
    fc2_b: GpuTensor,
    out_proj_w: LinearWeight,
    balancing_biases: Vec<f32>,
    experts: Vec<GpuExpert>,
    // residual scales (each: hidden_states_scale, hidden_states_bias, residual_scale, residual_bias)
    pa_rs: [GpuTensor; 4],
    pm_rs: [GpuTensor; 4],
}

impl GpuExpert {
    fn free(self, gpu: &mut Gpu) {
        self.gate_up.free(gpu);
        self.down.free(gpu);
    }
}

impl GpuLayer {
    fn free(self, gpu: &mut Gpu) {
        for lw in [
            self.q_proj,
            self.k_proj,
            self.v_cur,
            self.v_del,
            self.o_proj,
            self.down_proj_w,
            self.fc1_w,
            self.fc2_w,
            self.out_proj_w,
        ] {
            lw.free(gpu);
        }
        for e in self.experts {
            e.free(gpu);
        }
        let mut ts = vec![
            self.input_ln,
            self.post_attn_ln,
            self.conv_dw_w,
            self.conv_dw_b,
            self.conv_gr_w,
            self.conv_gr_b,
            self.qk_temp,
            self.down_proj_b,
            self.rnorm_w,
            self.fc1_b,
            self.fc2_b,
        ];
        ts.extend(self.pa_rs);
        ts.extend(self.pm_rs);
        ts.extend(self.router_states_scale);
        for t in ts {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// All ZAYA weights resident on the GPU. Big linears (projections, experts,
/// tied lm_head/embed) keep their hfq quant format via [`LinearWeight`]; small
/// protected tensors (norms, biases, conv filters, scales) are f32.
pub struct ZayaGpuWeights {
    embed: LinearWeight,
    in_scale: GpuTensor,
    in_bias: GpuTensor,
    layers: Vec<GpuLayer>,
    norm: GpuTensor,
}

impl ZayaGpuWeights {
    /// Release every GPU buffer (weights). Consumes self.
    pub fn free(self, gpu: &mut Gpu) {
        self.embed.free(gpu);
        for l in self.layers {
            l.free(gpu);
        }
        for t in [self.in_scale, self.in_bias, self.norm] {
            let _ = gpu.free_tensor(t);
        }
    }
}

impl ZayaGpuWeights {
    pub fn load(hfq: &HfqFile, gpu: &mut Gpu, cfg: &ZayaConfig) -> Result<Self, String> {
        let embed = load_linear(hfq, gpu, "model.embed_tokens.weight")?;
        let in_scale = up(hfq, gpu, "model.input_hidden_states_scale")?;
        let in_bias = up(hfq, gpu, "model.input_hidden_states_bias")?;
        let norm = up(hfq, gpu, "model.norm.weight")?;
        let mut layers = Vec::with_capacity(cfg.num_blocks);
        for l in 0..cfg.num_blocks {
            let p = format!("model.layers.{l}");
            let qkv = format!("{p}.self_attn.qkv_proj");
            let gate = format!("{p}.mlp.gate");
            let rmlp = format!("{gate}.router_mlp");
            let rs = |gpu: &mut Gpu, base: &str| -> Result<[GpuTensor; 4], String> {
                Ok([
                    up(hfq, gpu, &format!("{base}.hidden_states_scale"))?,
                    up(hfq, gpu, &format!("{base}.hidden_states_bias"))?,
                    up(hfq, gpu, &format!("{base}.residual_scale"))?,
                    up(hfq, gpu, &format!("{base}.residual_bias"))?,
                ])
            };
            let mut experts = Vec::with_capacity(cfg.moe.num_experts);
            for e in 0..cfg.moe.num_experts {
                experts.push(GpuExpert {
                    gate_up: load_linear(
                        hfq,
                        gpu,
                        &format!("{p}.mlp.experts.{e}.gate_up_proj.weight"),
                    )?,
                    down: load_linear(hfq, gpu, &format!("{p}.mlp.experts.{e}.down_proj.weight"))?,
                });
            }
            let router_states_scale = if l == 0 {
                None
            } else {
                Some(up(hfq, gpu, &format!("{gate}.router_states_scale"))?)
            };
            // balancing_biases stays on host (top-1 done host-side).
            let (_, bb_bytes) = hfq
                .tensor_data_vec(&format!("{gate}.balancing_biases"))
                .ok_or("zaya gpu: missing balancing_biases")?;
            let bb_qt = hfq
                .find_tensor_info(&format!("{gate}.balancing_biases"))
                .unwrap()
                .quant_type;
            let balancing_biases = dequant_qt(bb_qt, &bb_bytes)?;

            layers.push(GpuLayer {
                input_ln: up(hfq, gpu, &format!("{p}.input_layernorm.weight"))?,
                post_attn_ln: up(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"))?,
                q_proj: load_linear(hfq, gpu, &format!("{qkv}.q_proj.weight"))?,
                k_proj: load_linear(hfq, gpu, &format!("{qkv}.k_proj.weight"))?,
                v_cur: load_linear(hfq, gpu, &format!("{qkv}.v_proj_current.weight"))?,
                v_del: load_linear(hfq, gpu, &format!("{qkv}.v_proj_delayed.weight"))?,
                conv_dw_w: up(hfq, gpu, &format!("{qkv}.conv_qk_depthwise.weight"))?,
                conv_dw_b: up(hfq, gpu, &format!("{qkv}.conv_qk_depthwise.bias"))?,
                conv_gr_w: up(hfq, gpu, &format!("{qkv}.conv_qk_grouped.weight"))?,
                conv_gr_b: up(hfq, gpu, &format!("{qkv}.conv_qk_grouped.bias"))?,
                qk_temp: up(hfq, gpu, &format!("{p}.self_attn.qk_norm.temp"))?,
                o_proj: load_linear(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"))?,
                down_proj_w: load_linear(hfq, gpu, &format!("{gate}.down_proj.weight"))?,
                down_proj_b: up(hfq, gpu, &format!("{gate}.down_proj.bias"))?,
                router_states_scale,
                rnorm_w: up(hfq, gpu, &format!("{rmlp}.norm.weight"))?,
                fc1_w: load_linear(hfq, gpu, &format!("{rmlp}.fc1.weight"))?,
                fc1_b: up(hfq, gpu, &format!("{rmlp}.fc1.bias"))?,
                fc2_w: load_linear(hfq, gpu, &format!("{rmlp}.fc2.weight"))?,
                fc2_b: up(hfq, gpu, &format!("{rmlp}.fc2.bias"))?,
                out_proj_w: load_linear(hfq, gpu, &format!("{rmlp}.out_proj.weight"))?,
                balancing_biases,
                experts,
                pa_rs: rs(gpu, &format!("{p}.post_attention_residual_scale"))?,
                pm_rs: rs(gpu, &format!("{p}.post_mlp_residual_scale"))?,
            });
        }
        Ok(Self {
            embed,
            in_scale,
            in_bias,
            layers,
            norm,
        })
    }
}

/// Per-block hidden states + logits, downloaded to host for golden comparison.
pub struct GpuTrace {
    pub embed_scaled: Vec<f32>,
    pub block: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
    pub seq: usize,
}

/// gemv over a sequence: `y[s,m] = x[s,k] @ w[m,k]^T` (per-token gemv loop,
/// f32 or quantized via [`LinearWeight`]).
fn gemv_seq(
    gpu: &mut Gpu,
    w: &LinearWeight,
    x: &GpuTensor,
    y: &GpuTensor,
    s: usize,
    m: usize,
    k: usize,
) -> Result<(), String> {
    for t in 0..s {
        let xt = x.sub_offset(t * k, k);
        let yt = y.sub_offset(t * m, m);
        w.gemv(gpu, &xt, &yt)?;
    }
    Ok(())
}

fn z(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    gpu.zeros(&[n], DType::F32)
        .map_err(|e| format!("zaya alloc: {e:?}"))
}

/// 2D `[rows, d]` allocation — required for tensors fed to `rmsnorm_f32`, which
/// reads `batch = shape[0]` and `n = shape.last()`.
fn z2(gpu: &mut Gpu, rows: usize, d: usize) -> Result<GpuTensor, String> {
    gpu.zeros(&[rows, d], DType::F32)
        .map_err(|e| format!("zaya alloc: {e:?}"))
}

/// Per-layer decode state: KV cache (post-rope key, composed value), the two
/// last raw qk-stream columns (conv ring), and the previous token's delayed-value
/// projection. Carried across decode steps so each step processes one token.
pub struct ZayaDecodeState {
    pub pos: usize,
    max_seq: usize,
    k_cache: Vec<GpuTensor>, // [layer] [max_seq * kvdim]
    v_cache: Vec<GpuTensor>,
    conv_ring: Vec<GpuTensor>, // [layer] [conv_ch * pad]
    delayed_v: Vec<GpuTensor>, // [layer] [v_half]
}

impl ZayaDecodeState {
    pub fn new(gpu: &mut Gpu, cfg: &ZayaConfig, max_seq: usize) -> Result<Self, String> {
        let kvdim = cfg.attn.num_kv_heads * cfg.attn.head_dim;
        let conv_ch = (cfg.attn.num_heads + cfg.attn.num_kv_heads) * cfg.attn.head_dim;
        let pad = (cfg.attn.conv_depthwise_kernel - 1) + (cfg.attn.conv_grouped_kernel - 1);
        let v_half = kvdim / 2;
        let mut k_cache = Vec::with_capacity(cfg.num_blocks);
        let mut v_cache = Vec::with_capacity(cfg.num_blocks);
        let mut conv_ring = Vec::with_capacity(cfg.num_blocks);
        let mut delayed_v = Vec::with_capacity(cfg.num_blocks);
        for _ in 0..cfg.num_blocks {
            k_cache.push(z(gpu, max_seq * kvdim)?);
            v_cache.push(z(gpu, max_seq * kvdim)?);
            conv_ring.push(z(gpu, conv_ch * pad)?);
            delayed_v.push(z(gpu, v_half)?);
        }
        Ok(Self {
            pos: 0,
            max_seq,
            k_cache,
            v_cache,
            conv_ring,
            delayed_v,
        })
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Release the KV cache + conv-ring + delayed-value buffers. Consumes self.
    pub fn free(self, gpu: &mut Gpu) {
        for v in [self.k_cache, self.v_cache, self.conv_ring, self.delayed_v] {
            for t in v {
                let _ = gpu.free_tensor(t);
            }
        }
    }
}

/// Serving forward: prefill the whole prompt, write the **last position's** logits
/// into `logits_out` (`[vocab]`), and **prime `state`** (KV cache + conv ring +
/// delayed value for every layer) so subsequent tokens go through `gpu_decode`.
#[allow(clippy::needless_range_loop)]
pub fn gpu_forward_serve(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    ids: &[u32],
    state: &mut ZayaDecodeState,
    logits_out: &GpuTensor,
) -> Result<(), String> {
    if ids.len() > state.max_seq {
        return Err(format!(
            "zaya prefill: prompt {} exceeds max_seq {}",
            ids.len(),
            state.max_seq
        ));
    }
    let s = ids.len();
    let h = cfg.hidden_size;
    let a = &cfg.attn;
    let (nq, nkv, hd) = (a.num_heads, a.num_kv_heads, a.head_dim);
    let q_dim = nq * hd;
    let k_dim = nkv * hd;
    let v_half = k_dim / 2;
    let conv_ch = q_dim + k_dim;
    let rh = cfg.moe.router_hidden_size;
    let n_route = cfg.moe.num_router_experts();
    let n_exp = cfg.moe.num_experts;
    let moe_int = cfg.moe.moe_intermediate_size;
    let eps = cfg.rms_norm_eps;
    let pad = (a.conv_depthwise_kernel - 1) + (a.conv_grouped_kernel - 1);
    let dw_len = s + pad - a.conv_depthwise_kernel + 1;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let l2_scale = (hd as f32).sqrt();

    let hidden = z2(gpu, s, h)?;
    for t in 0..s {
        let row = hidden.sub_offset(t * h, h);
        w.embed.embed_lookup(gpu, &row, ids[t], h)?;
    }
    // global input residual affine, in place.
    gpu.zaya_affine_input_f32(&hidden, &hidden, &w.in_scale, &w.in_bias, h, s * h)
        .map_err(|e| format!("{e:?}"))?;

    let normed = z2(gpu, s, h)?;
    let q = z(gpu, s * q_dim)?;
    let k = z(gpu, s * k_dim)?;
    let vcur = z(gpu, s * v_half)?;
    let vdel = z(gpu, s * v_half)?;
    let qres = z(gpu, s * nq * hd)?;
    let kres = z(gpu, s * nkv * hd)?;
    let stream = z(gpu, conv_ch * (s + pad))?;
    let dw = z(gpu, conv_ch * dw_len)?;
    let gw = z(gpu, conv_ch * s)?;
    let query = z(gpu, s * nq * hd)?;
    let key = z(gpu, s * nkv * hd)?;
    let value = z(gpu, s * nkv * hd)?;
    let ctx = z(gpu, s * q_dim)?;
    let attn_out = z(gpu, s * h)?;
    let g_res2 = z2(gpu, s, h)?;
    let rhid = z2(gpu, s, rh)?;
    let rnormed = z2(gpu, s, rh)?;
    let a1 = z2(gpu, s, rh)?;
    let a2 = z2(gpu, s, rh)?;
    let rlogits = z(gpu, s * n_route)?;
    let moe_out = z(gpu, s * h)?;
    let gate_up = z(gpu, 2 * moe_int)?;
    let act = z(gpu, moe_int)?;
    let down_t = z(gpu, h)?;
    let router_state = z(gpu, s * rh)?;

    for (li, lw) in w.layers.iter().enumerate() {
        gpu.rmsnorm_f32(&hidden, &lw.input_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.q_proj, &normed, &q, s, q_dim, h)?;
        gemv_seq(gpu, &lw.k_proj, &normed, &k, s, k_dim, h)?;
        gemv_seq(gpu, &lw.v_cur, &normed, &vcur, s, v_half, h)?;
        gemv_seq(gpu, &lw.v_del, &normed, &vdel, s, v_half, h)?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 1)
            .map_err(|e| format!("{e:?}"))?;
        gpu.fill_f32(&stream, 0.0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_stream_f32(&stream, &q, &k, s, q_dim, k_dim, pad)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &dw,
            &stream,
            &lw.conv_dw_w,
            &lw.conv_dw_b,
            conv_ch,
            conv_ch,
            a.conv_depthwise_kernel,
            s + pad,
            dw_len,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &gw,
            &dw,
            &lw.conv_gr_w,
            &lw.conv_gr_b,
            conv_ch,
            nq + nkv,
            a.conv_grouped_kernel,
            dw_len,
            s,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&query, &gw, &qres, s, nq, hd, q_dim, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&key, &gw, &kres, s, nkv, hd, q_dim, 1)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_value_compose_f32(&value, &vcur, &vdel, s, nkv, hd)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&query, None, s, nq, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&key, Some(&lw.qk_temp), s, nkv, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&query, s, nq, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&key, s, nkv, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gqa_attn_f32(&ctx, &query, &key, &value, s, nq, nkv, hd, attn_scale)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.o_proj, &ctx, &attn_out, s, h, q_dim)?;
        // Prime decode state for this layer: KV (post-rope key / composed value),
        // conv ring (last `pad` raw qk-stream columns), delayed value (last token).
        let kvdim = k_dim;
        gpu.zaya_write_at_f32(&state.k_cache[li], &key, 0, s * kvdim)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_write_at_f32(&state.v_cache[li], &value, 0, s * kvdim)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_strided_copy_f32(&state.conv_ring[li], &stream, conv_ch, pad, s + pad, s, pad)
            .map_err(|e| format!("{e:?}"))?;
        let vdel_last = vdel.sub_offset((s - 1) * v_half, v_half);
        gpu.zaya_write_at_f32(&state.delayed_v[li], &vdel_last, 0, v_half)
            .map_err(|e| format!("{e:?}"))?;
        // residual stays on-device: `hidden` is still the block input here.
        gpu.zaya_affine_residual_f32(
            &g_res2,
            &attn_out,
            &hidden,
            &lw.pa_rs[0],
            &lw.pa_rs[1],
            &lw.pa_rs[2],
            &lw.pa_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        // MoE
        gemv_seq(gpu, &lw.down_proj_w, &normed, &rhid, s, rh, h)?;
        gpu.zaya_bias_add_f32(&rhid, &lw.down_proj_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        if li != 0 {
            if let Some(scale) = lw.router_states_scale.as_ref() {
                gpu.zaya_eda_add_f32(&rhid, &router_state, scale, rh, s * rh)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        // save next router state via an on-device copy (no host round-trip).
        gpu.zaya_write_at_f32(&router_state, &rhid, 0, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc1_w, &rnormed, &a1, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a1, &lw.fc1_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a1, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc2_w, &a1, &a2, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a2, &lw.fc2_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a2, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.out_proj_w, &a2, &rlogits, s, n_route, rh)?;
        let logit_host = gpu.download_f32(&rlogits).map_err(|e| format!("{e:?}"))?;
        gpu.fill_f32(&moe_out, 0.0).map_err(|e| format!("{e:?}"))?;
        for t in 0..s {
            let row = &logit_host[t * n_route..(t + 1) * n_route];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_route];
            let mut denom = 0f32;
            for e in 0..n_route {
                probs[e] = (row[e] - maxv).exp();
                denom += probs[e];
            }
            for p in probs.iter_mut() {
                *p /= denom;
            }
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for e in 0..n_route {
                let v = probs[e] + lw.balancing_biases[e];
                if v > bestv {
                    bestv = v;
                    best = e;
                }
            }
            if best == n_exp {
                continue;
            }
            let weight = probs[best];
            let xt = normed.sub_offset(t * h, h);
            let ex = &lw.experts[best];
            ex.gate_up.gemv(gpu, &xt, &gate_up)?;
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            gpu.silu_mul_f32(&g, &u, &act)
                .map_err(|e| format!("{e:?}"))?;
            ex.down.gemv(gpu, &act, &down_t)?;
            let ot = moe_out.sub_offset(t * h, h);
            gpu.scaled_add_inplace_cpu_scalar_f32(&ot, &down_t, weight)
                .map_err(|e| format!("{e:?}"))?;
        }
        gpu.zaya_affine_residual_f32(
            &hidden,
            &moe_out,
            &g_res2,
            &lw.pm_rs[0],
            &lw.pm_rs[1],
            &lw.pm_rs[2],
            &lw.pm_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
    }

    // final norm on the last row → tied lm_head → logits_out [vocab]
    let fnorm = z2(gpu, s, h)?;
    gpu.rmsnorm_f32(&hidden, &w.norm, &fnorm, eps)
        .map_err(|e| format!("{e:?}"))?;
    let last = fnorm.sub_offset((s - 1) * h, h);
    w.embed
        .gemv(gpu, &last, logits_out)
        .map_err(|e| format!("zaya lm_head: {e}"))?;
    // Return all scratch to the pool (no DeviceBuffer Drop); reused next call.
    for t in [
        hidden,
        normed,
        q,
        k,
        vcur,
        vdel,
        qres,
        kres,
        stream,
        dw,
        gw,
        query,
        key,
        value,
        ctx,
        attn_out,
        g_res2,
        rhid,
        rnormed,
        a1,
        a2,
        rlogits,
        moe_out,
        gate_up,
        act,
        down_t,
        router_state,
        fnorm,
    ] {
        let _ = gpu.free_tensor(t);
    }
    state.pos = s;
    Ok(())
}

/// Map each dense linear's GPU buffer address → its canonical HFQ tensor name
/// (sans `.weight`), for activation capture during calibration. Routed experts
/// are intentionally excluded (LDLQ targets the dense projections; experts can
/// be captured imatrix-only in a later pass). The tied `embed` doubles as the
/// `lm_head`, keyed by `model.embed_tokens` (the name the quantizer sees).
pub fn build_capture_names(w: &ZayaGpuWeights) -> std::collections::HashMap<usize, String> {
    let mut m = std::collections::HashMap::new();
    let mut put = |lw: &LinearWeight, name: String| {
        m.insert(lw.buf().buf.as_ptr() as usize, name);
    };
    put(&w.embed, "model.embed_tokens".to_string());
    for (l, lw) in w.layers.iter().enumerate() {
        let p = format!("model.layers.{l}");
        let qkv = format!("{p}.self_attn.qkv_proj");
        let rmlp = format!("{p}.mlp.gate.router_mlp");
        put(&lw.q_proj, format!("{qkv}.q_proj"));
        put(&lw.k_proj, format!("{qkv}.k_proj"));
        put(&lw.v_cur, format!("{qkv}.v_proj_current"));
        put(&lw.v_del, format!("{qkv}.v_proj_delayed"));
        put(&lw.o_proj, format!("{p}.self_attn.o_proj"));
        put(&lw.down_proj_w, format!("{p}.mlp.gate.down_proj"));
        put(&lw.fc1_w, format!("{rmlp}.fc1"));
        put(&lw.fc2_w, format!("{rmlp}.fc2"));
        put(&lw.out_proj_w, format!("{rmlp}.out_proj"));
        // Routed experts (imatrix-only — sparse under top-1, so no full Hessian).
        for (e, ex) in lw.experts.iter().enumerate() {
            put(&ex.gate_up, format!("{p}.mlp.experts.{e}.gate_up_proj"));
            put(&ex.down, format!("{p}.mlp.experts.{e}.down_proj"));
        }
    }
    m
}

/// Calibration forward: run the full prompt and call `maybe_capture_activation`
/// before each dense gemv so the active `CalibCollector` accumulates `H = XᵀX`
/// (+ diag) keyed by the names in [`build_capture_names`]. No KV/decode state,
/// no logits output — just drives activations. The caller must set
/// `gpu.capture_names`/`gpu.active_capture` around this.
#[allow(clippy::needless_range_loop)]
/// Calibration forward over the whole prompt. Captures per-tensor Hessian +
/// imatrix via the active capture hook. When `kldref_topk` is `Some(k)`, also
/// computes the tied lm-head logits and returns the per-position `(logZ, top-k)`
/// KLDREF reference (the bf16 teacher signal for KL-divergence eval); otherwise
/// returns an empty vec and skips the lm-head gemv.
pub fn gpu_forward_calib(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    ids: &[u32],
    kldref_topk: Option<usize>,
) -> Result<Vec<(f32, Vec<(u32, f32)>)>, String> {
    let s = ids.len();
    let h = cfg.hidden_size;
    let a = &cfg.attn;
    let (nq, nkv, hd) = (a.num_heads, a.num_kv_heads, a.head_dim);
    let q_dim = nq * hd;
    let k_dim = nkv * hd;
    let v_half = k_dim / 2;
    let conv_ch = q_dim + k_dim;
    let rh = cfg.moe.router_hidden_size;
    let n_route = cfg.moe.num_router_experts();
    let n_exp = cfg.moe.num_experts;
    let moe_int = cfg.moe.moe_intermediate_size;
    let eps = cfg.rms_norm_eps;
    let pad = (a.conv_depthwise_kernel - 1) + (a.conv_grouped_kernel - 1);
    let dw_len = s + pad - a.conv_depthwise_kernel + 1;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let l2_scale = (hd as f32).sqrt();

    let hidden = z2(gpu, s, h)?;
    for t in 0..s {
        let row = hidden.sub_offset(t * h, h);
        w.embed.embed_lookup(gpu, &row, ids[t], h)?;
    }
    gpu.zaya_affine_input_f32(&hidden, &hidden, &w.in_scale, &w.in_bias, h, s * h)
        .map_err(|e| format!("{e:?}"))?;

    let normed = z2(gpu, s, h)?;
    let q = z(gpu, s * q_dim)?;
    let k = z(gpu, s * k_dim)?;
    let vcur = z(gpu, s * v_half)?;
    let vdel = z(gpu, s * v_half)?;
    let qres = z(gpu, s * nq * hd)?;
    let kres = z(gpu, s * nkv * hd)?;
    let stream = z(gpu, conv_ch * (s + pad))?;
    let dw = z(gpu, conv_ch * dw_len)?;
    let gw = z(gpu, conv_ch * s)?;
    let query = z(gpu, s * nq * hd)?;
    let key = z(gpu, s * nkv * hd)?;
    let value = z(gpu, s * nkv * hd)?;
    let ctx = z(gpu, s * q_dim)?;
    let attn_out = z(gpu, s * h)?;
    let g_res2 = z2(gpu, s, h)?;
    let rhid = z2(gpu, s, rh)?;
    let rnormed = z2(gpu, s, rh)?;
    let a1 = z2(gpu, s, rh)?;
    let a2 = z2(gpu, s, rh)?;
    let rlogits = z(gpu, s * n_route)?;
    let moe_out = z(gpu, s * h)?;
    let gate_up = z(gpu, 2 * moe_int)?;
    let act = z(gpu, moe_int)?;
    let down_t = z(gpu, h)?;
    let router_state = z(gpu, s * rh)?;

    for (li, lw) in w.layers.iter().enumerate() {
        gpu.rmsnorm_f32(&hidden, &lw.input_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        // capture the q/k/v projection inputs (all = post-input-norm hidden).
        gpu.maybe_capture_activation(lw.q_proj.buf(), &normed, s, h);
        gpu.maybe_capture_activation(lw.k_proj.buf(), &normed, s, h);
        gpu.maybe_capture_activation(lw.v_cur.buf(), &normed, s, h);
        gpu.maybe_capture_activation(lw.v_del.buf(), &normed, s, h);
        gemv_seq(gpu, &lw.q_proj, &normed, &q, s, q_dim, h)?;
        gemv_seq(gpu, &lw.k_proj, &normed, &k, s, k_dim, h)?;
        gemv_seq(gpu, &lw.v_cur, &normed, &vcur, s, v_half, h)?;
        gemv_seq(gpu, &lw.v_del, &normed, &vdel, s, v_half, h)?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 1)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_stream_f32(&stream, &q, &k, s, q_dim, k_dim, pad)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &dw,
            &stream,
            &lw.conv_dw_w,
            &lw.conv_dw_b,
            conv_ch,
            conv_ch,
            a.conv_depthwise_kernel,
            s + pad,
            dw_len,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &gw,
            &dw,
            &lw.conv_gr_w,
            &lw.conv_gr_b,
            conv_ch,
            nq + nkv,
            a.conv_grouped_kernel,
            dw_len,
            s,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&query, &gw, &qres, s, nq, hd, q_dim, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&key, &gw, &kres, s, nkv, hd, q_dim, 1)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_value_compose_f32(&value, &vcur, &vdel, s, nkv, hd)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&query, None, s, nq, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&key, Some(&lw.qk_temp), s, nkv, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&query, s, nq, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&key, s, nkv, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gqa_attn_f32(&ctx, &query, &key, &value, s, nq, nkv, hd, attn_scale)
            .map_err(|e| format!("{e:?}"))?;
        gpu.maybe_capture_activation(lw.o_proj.buf(), &ctx, s, q_dim);
        gemv_seq(gpu, &lw.o_proj, &ctx, &attn_out, s, h, q_dim)?;
        gpu.zaya_affine_residual_f32(
            &g_res2,
            &attn_out,
            &hidden,
            &lw.pa_rs[0],
            &lw.pa_rs[1],
            &lw.pa_rs[2],
            &lw.pa_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        // MoE: capture the router projections (dense); experts run but aren't captured.
        gpu.maybe_capture_activation(lw.down_proj_w.buf(), &normed, s, h);
        gemv_seq(gpu, &lw.down_proj_w, &normed, &rhid, s, rh, h)?;
        gpu.zaya_bias_add_f32(&rhid, &lw.down_proj_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        if li != 0 {
            if let Some(scale) = lw.router_states_scale.as_ref() {
                gpu.zaya_eda_add_f32(&rhid, &router_state, scale, rh, s * rh)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        gpu.zaya_write_at_f32(&router_state, &rhid, 0, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps)
            .map_err(|e| format!("{e:?}"))?;
        gpu.maybe_capture_activation(lw.fc1_w.buf(), &rnormed, s, rh);
        gemv_seq(gpu, &lw.fc1_w, &rnormed, &a1, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a1, &lw.fc1_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a1, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.maybe_capture_activation(lw.fc2_w.buf(), &a1, s, rh);
        gemv_seq(gpu, &lw.fc2_w, &a1, &a2, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a2, &lw.fc2_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a2, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.maybe_capture_activation(lw.out_proj_w.buf(), &a2, s, rh);
        gemv_seq(gpu, &lw.out_proj_w, &a2, &rlogits, s, n_route, rh)?;
        let logit_host = gpu.download_f32(&rlogits).map_err(|e| format!("{e:?}"))?;
        gpu.fill_f32(&moe_out, 0.0).map_err(|e| format!("{e:?}"))?;
        for t in 0..s {
            let row = &logit_host[t * n_route..(t + 1) * n_route];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_route];
            let mut denom = 0f32;
            for e in 0..n_route {
                probs[e] = (row[e] - maxv).exp();
                denom += probs[e];
            }
            for p in probs.iter_mut() {
                *p /= denom;
            }
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for e in 0..n_route {
                let v = probs[e] + lw.balancing_biases[e];
                if v > bestv {
                    bestv = v;
                    best = e;
                }
            }
            if best == n_exp {
                continue;
            }
            let weight = probs[best];
            let xt = normed.sub_offset(t * h, h);
            let ex = &lw.experts[best];
            gpu.maybe_capture_activation(ex.gate_up.buf(), &xt, 1, h); // no-op unless calibrating
            ex.gate_up.gemv(gpu, &xt, &gate_up)?;
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            gpu.silu_mul_f32(&g, &u, &act)
                .map_err(|e| format!("{e:?}"))?;
            gpu.maybe_capture_activation(ex.down.buf(), &act, 1, moe_int); // no-op unless calibrating
            ex.down.gemv(gpu, &act, &down_t)?;
            let ot = moe_out.sub_offset(t * h, h);
            gpu.scaled_add_inplace_cpu_scalar_f32(&ot, &down_t, weight)
                .map_err(|e| format!("{e:?}"))?;
        }
        gpu.zaya_affine_residual_f32(
            &hidden,
            &moe_out,
            &g_res2,
            &lw.pm_rs[0],
            &lw.pm_rs[1],
            &lw.pm_rs[2],
            &lw.pm_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
    }

    let fnorm = z2(gpu, s, h)?;
    gpu.rmsnorm_f32(&hidden, &w.norm, &fnorm, eps)
        .map_err(|e| format!("{e:?}"))?;
    // capture the tied lm_head input (no gemv needed for the imatrix/Hessian).
    gpu.maybe_capture_activation(w.embed.buf(), &fnorm, s, h);

    // Optional KLDREF: run the tied lm-head to get logits [s, vocab], then keep
    // only the per-position logZ + top-k (a compact bf16 reference). The full
    // [s, vocab] host buffer is dropped here, so peak host memory is one row.
    let kldref = if let Some(topk) = kldref_topk {
        let logits = z(gpu, s * cfg.vocab_size)?;
        gemv_seq(gpu, &w.embed, &fnorm, &logits, s, cfg.vocab_size, h)?;
        let host = gpu.download_f32(&logits).map_err(|e| format!("{e:?}"))?;
        let _ = gpu.free_tensor(logits);
        let v = cfg.vocab_size;
        (0..s)
            .map(|p| {
                let row = &host[p * v..(p + 1) * v];
                (logsumexp(row), topk_logits(row, topk))
            })
            .collect()
    } else {
        Vec::new()
    };

    for t in [
        hidden,
        normed,
        q,
        k,
        vcur,
        vdel,
        qres,
        kres,
        stream,
        dw,
        gw,
        query,
        key,
        value,
        ctx,
        attn_out,
        g_res2,
        rhid,
        rnormed,
        a1,
        a2,
        rlogits,
        moe_out,
        gate_up,
        act,
        down_t,
        router_state,
        fnorm,
    ] {
        let _ = gpu.free_tensor(t);
    }
    Ok(kldref)
}

/// Single-token decode at `state.pos`: O(1) per-layer compute using the KV cache,
/// conv ring, and delayed-value state. Writes the new token's logits into
/// `logits_out` and advances the state by one position.
#[allow(clippy::needless_range_loop)]
pub fn gpu_decode(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    token: u32,
    state: &mut ZayaDecodeState,
    logits_out: &GpuTensor,
) -> Result<(), String> {
    let pos = state.pos;
    if pos >= state.max_seq {
        return Err(format!(
            "zaya decode: position {pos} exceeds max_seq {}",
            state.max_seq
        ));
    }
    let h = cfg.hidden_size;
    let a = &cfg.attn;
    let (nq, nkv, hd) = (a.num_heads, a.num_kv_heads, a.head_dim);
    let q_dim = nq * hd;
    let k_dim = nkv * hd;
    let kvdim = k_dim;
    let v_half = k_dim / 2;
    let conv_ch = q_dim + k_dim;
    let rh = cfg.moe.router_hidden_size;
    let n_route = cfg.moe.num_router_experts();
    let n_exp = cfg.moe.num_experts;
    let moe_int = cfg.moe.moe_intermediate_size;
    let eps = cfg.rms_norm_eps;
    let pad = (a.conv_depthwise_kernel - 1) + (a.conv_grouped_kernel - 1);
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let l2_scale = (hd as f32).sqrt();

    let hidden = z2(gpu, 1, h)?;
    {
        let row = hidden.sub_offset(0, h);
        w.embed.embed_lookup(gpu, &row, token, h)?;
    }
    // global input residual affine, in place.
    gpu.zaya_affine_input_f32(&hidden, &hidden, &w.in_scale, &w.in_bias, h, h)
        .map_err(|e| format!("{e:?}"))?;

    // single-token scratch (s = 1)
    let normed = z2(gpu, 1, h)?;
    let q = z(gpu, q_dim)?;
    let k = z(gpu, k_dim)?;
    let vcur = z(gpu, v_half)?;
    let vdel = z(gpu, v_half)?;
    let qres = z(gpu, nq * hd)?;
    let kres = z(gpu, nkv * hd)?;
    let cur_qk = z(gpu, conv_ch)?;
    let window = z(gpu, conv_ch * (pad + 1))?;
    let dw = z(gpu, conv_ch * (pad + 1 - a.conv_depthwise_kernel + 1))?;
    let gw = z(gpu, conv_ch)?;
    let query = z(gpu, nq * hd)?;
    let key = z(gpu, nkv * hd)?;
    let value = z(gpu, nkv * hd)?;
    let ctx = z(gpu, q_dim)?;
    let attn_out = z(gpu, h)?;
    let g_res2 = z2(gpu, 1, h)?;
    let rhid = z2(gpu, 1, rh)?;
    let rnormed = z2(gpu, 1, rh)?;
    let a1 = z2(gpu, 1, rh)?;
    let a2 = z2(gpu, 1, rh)?;
    let rlogits = z(gpu, n_route)?;
    let moe_out = z(gpu, h)?;
    let gate_up = z(gpu, 2 * moe_int)?;
    let act = z(gpu, moe_int)?;
    let down_t = z(gpu, h)?;
    let router_state = z(gpu, rh)?;
    let dw_len = pad + 1 - a.conv_depthwise_kernel + 1;

    for (li, lw) in w.layers.iter().enumerate() {
        gpu.rmsnorm_f32(&hidden, &lw.input_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        lw.q_proj.gemv(gpu, &normed, &q)?;
        lw.k_proj.gemv(gpu, &normed, &k)?;
        lw.v_cur.gemv(gpu, &normed, &vcur)?;
        lw.v_del.gemv(gpu, &normed, &vdel)?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, 1, nq, nkv, hd, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, 1, nq, nkv, hd, 1)
            .map_err(|e| format!("{e:?}"))?;
        // current qk-stream column, then conv window from the ring (advances ring).
        gpu.zaya_qk_stream_f32(&cur_qk, &q, &k, 1, q_dim, k_dim, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv_window_f32(&window, &state.conv_ring[li], &cur_qk, conv_ch, pad)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &dw,
            &window,
            &lw.conv_dw_w,
            &lw.conv_dw_b,
            conv_ch,
            conv_ch,
            a.conv_depthwise_kernel,
            pad + 1,
            dw_len,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &gw,
            &dw,
            &lw.conv_gr_w,
            &lw.conv_gr_b,
            conv_ch,
            nq + nkv,
            a.conv_grouped_kernel,
            dw_len,
            1,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&query, &gw, &qres, 1, nq, hd, q_dim, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&key, &gw, &kres, 1, nkv, hd, q_dim, 1)
            .map_err(|e| format!("{e:?}"))?;
        // value: head0 = current v, head1 = previous token's delayed v; then advance.
        gpu.zaya_write_at_f32(&value, &vcur, 0, v_half)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_write_at_f32(&value, &state.delayed_v[li], v_half, v_half)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_write_at_f32(&state.delayed_v[li], &vdel, 0, v_half)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&query, None, 1, nq, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&key, Some(&lw.qk_temp), 1, nkv, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&query, 1, nq, hd, a.n_rot, a.rope_theta, pos)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&key, 1, nkv, hd, a.n_rot, a.rope_theta, pos)
            .map_err(|e| format!("{e:?}"))?;
        // append to KV cache at `pos`, then attend over 0..=pos.
        gpu.zaya_write_at_f32(&state.k_cache[li], &key, pos * kvdim, kvdim)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_write_at_f32(&state.v_cache[li], &value, pos * kvdim, kvdim)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gqa_decode_f32(
            &ctx,
            &query,
            &state.k_cache[li],
            &state.v_cache[li],
            pos,
            nq,
            nkv,
            hd,
            attn_scale,
        )
        .map_err(|e| format!("{e:?}"))?;
        lw.o_proj.gemv(gpu, &ctx, &attn_out)?;
        gpu.zaya_affine_residual_f32(
            &g_res2,
            &attn_out,
            &hidden,
            &lw.pa_rs[0],
            &lw.pa_rs[1],
            &lw.pa_rs[2],
            &lw.pa_rs[3],
            h,
            h,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        // MoE (single token)
        lw.down_proj_w.gemv(gpu, &normed, &rhid)?;
        gpu.zaya_bias_add_f32(&rhid, &lw.down_proj_b, rh, rh)
            .map_err(|e| format!("{e:?}"))?;
        if li != 0 {
            if let Some(scale) = lw.router_states_scale.as_ref() {
                gpu.zaya_eda_add_f32(&rhid, &router_state, scale, rh, rh)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        // save next router state via an on-device copy (no host round-trip).
        gpu.zaya_write_at_f32(&router_state, &rhid, 0, rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps)
            .map_err(|e| format!("{e:?}"))?;
        lw.fc1_w.gemv(gpu, &rnormed, &a1)?;
        gpu.zaya_bias_add_f32(&a1, &lw.fc1_b, rh, rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a1, rh)
            .map_err(|e| format!("{e:?}"))?;
        lw.fc2_w.gemv(gpu, &a1, &a2)?;
        gpu.zaya_bias_add_f32(&a2, &lw.fc2_b, rh, rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a2, rh)
            .map_err(|e| format!("{e:?}"))?;
        lw.out_proj_w.gemv(gpu, &a2, &rlogits)?;
        let row = gpu.download_f32(&rlogits).map_err(|e| format!("{e:?}"))?;
        let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0f32; n_route];
        let mut denom = 0f32;
        for e in 0..n_route {
            probs[e] = (row[e] - maxv).exp();
            denom += probs[e];
        }
        for p in probs.iter_mut() {
            *p /= denom;
        }
        let mut best = 0usize;
        let mut bestv = f32::NEG_INFINITY;
        for e in 0..n_route {
            let v = probs[e] + lw.balancing_biases[e];
            if v > bestv {
                bestv = v;
                best = e;
            }
        }
        gpu.fill_f32(&moe_out, 0.0).map_err(|e| format!("{e:?}"))?;
        if best != n_exp {
            let weight = probs[best];
            let ex = &lw.experts[best];
            ex.gate_up.gemv(gpu, &normed, &gate_up)?;
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            gpu.silu_mul_f32(&g, &u, &act)
                .map_err(|e| format!("{e:?}"))?;
            ex.down.gemv(gpu, &act, &down_t)?;
            gpu.scaled_add_inplace_cpu_scalar_f32(&moe_out, &down_t, weight)
                .map_err(|e| format!("{e:?}"))?;
        }
        gpu.zaya_affine_residual_f32(
            &hidden,
            &moe_out,
            &g_res2,
            &lw.pm_rs[0],
            &lw.pm_rs[1],
            &lw.pm_rs[2],
            &lw.pm_rs[3],
            h,
            h,
        )
        .map_err(|e| format!("{e:?}"))?;
    }

    let fnorm = z2(gpu, 1, h)?;
    gpu.rmsnorm_f32(&hidden, &w.norm, &fnorm, eps)
        .map_err(|e| format!("{e:?}"))?;
    w.embed
        .gemv(gpu, &fnorm, logits_out)
        .map_err(|e| format!("zaya lm_head: {e}"))?;
    // Return all scratch to the pool (no DeviceBuffer Drop); reused next decode step.
    for t in [
        hidden,
        normed,
        q,
        k,
        vcur,
        vdel,
        qres,
        kres,
        cur_qk,
        window,
        dw,
        gw,
        query,
        key,
        value,
        ctx,
        attn_out,
        g_res2,
        rhid,
        rnormed,
        a1,
        a2,
        rlogits,
        moe_out,
        gate_up,
        act,
        down_t,
        router_state,
        fnorm,
    ] {
        let _ = gpu.free_tensor(t);
    }
    state.pos = pos + 1;
    Ok(())
}

/// Run the GPU prefill forward, capturing per-block hidden states.
#[allow(clippy::needless_range_loop)]
pub fn gpu_forward_prefill(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    ids: &[u32],
) -> Result<GpuTrace, String> {
    let s = ids.len();
    let h = cfg.hidden_size;
    let a = &cfg.attn;
    let (nq, nkv, hd) = (a.num_heads, a.num_kv_heads, a.head_dim);
    let q_dim = nq * hd;
    let k_dim = nkv * hd;
    let v_half = k_dim / 2;
    let conv_ch = q_dim + k_dim;
    let rh = cfg.moe.router_hidden_size;
    let n_route = cfg.moe.num_router_experts();
    let n_exp = cfg.moe.num_experts;
    let moe_int = cfg.moe.moe_intermediate_size;
    let eps = cfg.rms_norm_eps;
    let pad = (a.conv_depthwise_kernel - 1) + (a.conv_grouped_kernel - 1);

    // ids → device i32, embed gather → input affine.
    let hidden = z2(gpu, s, h)?;
    for t in 0..s {
        let row = hidden.sub_offset(t * h, h);
        w.embed.embed_lookup(gpu, &row, ids[t], h)?;
    }
    // global input residual affine, in place.
    gpu.zaya_affine_input_f32(&hidden, &hidden, &w.in_scale, &w.in_bias, h, s * h)
        .map_err(|e| format!("{e:?}"))?;
    let trace_embed = gpu.download_f32(&hidden).map_err(|e| format!("{e:?}"))?;

    let mut block_traces = Vec::with_capacity(cfg.num_blocks);
    let mut router_state: Option<GpuTensor> = None;

    // reusable scratch
    let normed = z2(gpu, s, h)?;
    let q = z(gpu, s * q_dim)?;
    let k = z(gpu, s * k_dim)?;
    let vcur = z(gpu, s * v_half)?;
    let vdel = z(gpu, s * v_half)?;
    let qres = z(gpu, s * nq * hd)?;
    let kres = z(gpu, s * nkv * hd)?;
    let stream = z(gpu, conv_ch * (s + pad))?;
    let dw = z(gpu, conv_ch * (s + pad - a.conv_depthwise_kernel + 1))?;
    let gw = z(gpu, conv_ch * s)?;
    let query = z(gpu, s * nq * hd)?;
    let key = z(gpu, s * nkv * hd)?;
    let value = z(gpu, s * nkv * hd)?;
    let ctx = z(gpu, s * q_dim)?;
    let attn_out = z(gpu, s * h)?;
    let rhid = z2(gpu, s, rh)?;
    let rnormed = z2(gpu, s, rh)?;
    let a1 = z(gpu, s * rh)?;
    let a2 = z(gpu, s * rh)?;
    let rlogits = z(gpu, s * n_route)?;
    let moe_out = z(gpu, s * h)?;
    let gate_up = z(gpu, 2 * moe_int)?;
    let act = z(gpu, moe_int)?;
    let down_t = z(gpu, h)?;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let l2_scale = (hd as f32).sqrt();
    let dw_len = s + pad - a.conv_depthwise_kernel + 1;

    for (li, lw) in w.layers.iter().enumerate() {
        let residual = gpu.download_f32(&hidden).map_err(|e| format!("{e:?}"))?; // keep host copy for affine residual source
        let g_residual = gpu
            .upload_f32(&residual, &[s * h])
            .map_err(|e| format!("{e:?}"))?;

        gpu.rmsnorm_f32(&hidden, &lw.input_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        // CCA projections
        gemv_seq(gpu, &lw.q_proj, &normed, &q, s, q_dim, h)?;
        gemv_seq(gpu, &lw.k_proj, &normed, &k, s, k_dim, h)?;
        gemv_seq(gpu, &lw.v_cur, &normed, &vcur, s, v_half, h)?;
        gemv_seq(gpu, &lw.v_del, &normed, &vdel, s, v_half, h)?;
        // q/k residual paths
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 1)
            .map_err(|e| format!("{e:?}"))?;
        // conv stream (zero pad region) → depthwise → grouped
        gpu.fill_f32(&stream, 0.0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_stream_f32(&stream, &q, &k, s, q_dim, k_dim, pad)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &dw,
            &stream,
            &lw.conv_dw_w,
            &lw.conv_dw_b,
            conv_ch,
            conv_ch,
            a.conv_depthwise_kernel,
            s + pad,
            dw_len,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &gw,
            &dw,
            &lw.conv_gr_w,
            &lw.conv_gr_b,
            conv_ch,
            nq + nkv,
            a.conv_grouped_kernel,
            dw_len,
            s,
        )
        .map_err(|e| format!("{e:?}"))?;
        // add residuals → head-major q,k
        gpu.zaya_add_conv_residual_f32(&query, &gw, &qres, s, nq, hd, q_dim, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&key, &gw, &kres, s, nkv, hd, q_dim, 1)
            .map_err(|e| format!("{e:?}"))?;
        // value compose
        gpu.zaya_value_compose_f32(&value, &vcur, &vdel, s, nkv, hd)
            .map_err(|e| format!("{e:?}"))?;
        // qk-norm (+ temp on key), rope
        gpu.zaya_qk_l2norm_temp_f32(&query, None, s, nq, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&key, Some(&lw.qk_temp), s, nkv, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&query, s, nq, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&key, s, nkv, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        // attention + o_proj
        gpu.zaya_gqa_attn_f32(&ctx, &query, &key, &value, s, nq, nkv, hd, attn_scale)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.o_proj, &ctx, &attn_out, s, h, q_dim)?;
        // residual = post_attention_residual_scale(attn_out, residual)
        let g_res2 = z2(gpu, s, h)?;
        gpu.zaya_affine_residual_f32(
            &g_res2,
            &attn_out,
            &g_residual,
            &lw.pa_rs[0],
            &lw.pa_rs[1],
            &lw.pa_rs[2],
            &lw.pa_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        // normed = post_attention_layernorm(residual)
        gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;

        // ── MoE ──
        gemv_seq(gpu, &lw.down_proj_w, &normed, &rhid, s, rh, h)?;
        gpu.zaya_bias_add_f32(&rhid, &lw.down_proj_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        if li != 0 {
            if let (Some(scale), Some(prev)) =
                (lw.router_states_scale.as_ref(), router_state.as_ref())
            {
                gpu.zaya_eda_add_f32(&rhid, prev, scale, rh, s * rh)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        // save next router state (copy of rhid)
        let rhid_host = gpu.download_f32(&rhid).map_err(|e| format!("{e:?}"))?;
        router_state = Some(
            gpu.upload_f32(&rhid_host, &[s * rh])
                .map_err(|e| format!("{e:?}"))?,
        );
        // router_mlp
        gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc1_w, &rnormed, &a1, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a1, &lw.fc1_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a1, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc2_w, &a1, &a2, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a2, &lw.fc2_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a2, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.out_proj_w, &a2, &rlogits, s, n_route, rh)?;
        // host top-1 of (softmax + balancing_biases); MoD skip if idx==n_exp
        let logit_host = gpu.download_f32(&rlogits).map_err(|e| format!("{e:?}"))?;
        gpu.fill_f32(&moe_out, 0.0).map_err(|e| format!("{e:?}"))?;
        for t in 0..s {
            let row = &logit_host[t * n_route..(t + 1) * n_route];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_route];
            let mut denom = 0f32;
            for e in 0..n_route {
                probs[e] = (row[e] - maxv).exp();
                denom += probs[e];
            }
            for p in probs.iter_mut() {
                *p /= denom;
            }
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for e in 0..n_route {
                let v = probs[e] + lw.balancing_biases[e];
                if v > bestv {
                    bestv = v;
                    best = e;
                }
            }
            if best == n_exp {
                continue; // MoD skip
            }
            let weight = probs[best];
            let xt = normed.sub_offset(t * h, h);
            let ex = &lw.experts[best];
            ex.gate_up.gemv(gpu, &xt, &gate_up)?;
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            gpu.silu_mul_f32(&g, &u, &act)
                .map_err(|e| format!("{e:?}"))?;
            ex.down.gemv(gpu, &act, &down_t)?;
            let ot = moe_out.sub_offset(t * h, h);
            gpu.scaled_add_inplace_cpu_scalar_f32(&ot, &down_t, weight)
                .map_err(|e| format!("{e:?}"))?;
        }
        // hidden = post_mlp_residual_scale(moe_out, residual)
        gpu.zaya_affine_residual_f32(
            &hidden,
            &moe_out,
            &g_res2,
            &lw.pm_rs[0],
            &lw.pm_rs[1],
            &lw.pm_rs[2],
            &lw.pm_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        block_traces.push(gpu.download_f32(&hidden).map_err(|e| format!("{e:?}"))?);
        let _ = li;
    }

    // final norm + tied lm_head
    let fnorm = z2(gpu, s, h)?;
    gpu.rmsnorm_f32(&hidden, &w.norm, &fnorm, eps)
        .map_err(|e| format!("{e:?}"))?;
    let final_norm = gpu.download_f32(&fnorm).map_err(|e| format!("{e:?}"))?;
    let logits = z(gpu, s * cfg.vocab_size)?;
    gemv_seq(gpu, &w.embed, &fnorm, &logits, s, cfg.vocab_size, h)?;
    let logits_host = gpu.download_f32(&logits).map_err(|e| format!("{e:?}"))?;

    Ok(GpuTrace {
        embed_scaled: trace_embed,
        block: block_traces,
        final_norm,
        logits: logits_host,
        seq: s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // f16 1.0 = 0x3C00; little-endian on-disk.
    const F16_ONE: [u8; 2] = [0x00, 0x3C];

    #[test]
    fn oq8_combined_layout() {
        // m=1, k=256, ng=1. On-disk block = [f16 scale][256 int8].
        let mut data = Vec::new();
        data.extend_from_slice(&F16_ONE);
        for i in 0..256u32 {
            data.push((i as i8).wrapping_sub(8) as u8); // arbitrary signed pattern
        }
        let out = oq8_combined(&data, 1, 256);
        // Combined = [int8 weights M*K][f32 scales M*ng].
        assert_eq!(out.len(), 256 + 4);
        assert_eq!(&out[..256], &data[2..258]); // int8 copied verbatim
        assert_eq!(
            f32::from_le_bytes([out[256], out[257], out[258], out[259]]),
            1.0
        );
    }

    #[test]
    fn oq4_pack_and_oq8_expand_layout() {
        // m=1, k=256, ng=1. On-disk block = [f16 scale][128 nibbles].
        let mut data = Vec::new();
        data.extend_from_slice(&F16_ONE);
        // nibble byte 0x21 → lo=1, hi=2 (both positive, no sign-extend wrap).
        let nibbles: Vec<u8> = (0..128).map(|i| 0x21u8 ^ (i as u8)).collect();
        data.extend_from_slice(&nibbles);

        // OQ4 arch-combined: packed nibbles [k/2] + scales [ng*4] + interleaved [ng*132].
        let oq4 = oq4_pack_arch_combined(&data, 1, 256);
        assert_eq!(oq4.len(), oq4_arch_combined_len(1, 256));
        assert_eq!(&oq4[..128], nibbles.as_slice()); // packed region = raw nibbles
        let scales_base = 128;
        assert_eq!(
            f32::from_le_bytes([
                oq4[scales_base],
                oq4[scales_base + 1],
                oq4[scales_base + 2],
                oq4[scales_base + 3],
            ]),
            1.0
        );
        // Interleaved region: [f32 scale][128 nibbles].
        let il = scales_base + 4;
        assert_eq!(&oq4[il + 4..il + 132], nibbles.as_slice());

        // OQ4→OQ8 expand: each nibble sign-extended to one int8, [int8 K][f32 scale].
        let oq8 = oq4_to_oq8_combined(&data, 1, 256);
        assert_eq!(oq8.len(), 256 + 4);
        for (i, &nib_byte) in nibbles.iter().enumerate() {
            assert_eq!(oq8[2 * i] as i8, sext4(nib_byte & 0xf));
            assert_eq!(oq8[2 * i + 1] as i8, sext4(nib_byte >> 4));
        }
    }

    #[test]
    fn oq_repack_dispatch_and_dtype() {
        let mut data = Vec::new();
        data.extend_from_slice(&F16_ONE);
        data.extend_from_slice(&vec![0u8; 128]);
        assert!(matches!(
            oq_repack(34, &data, 1, 256),
            Some((_, DType::Oq4G256))
        ));
        assert!(matches!(
            oq_repack(33, &data, 1, 256),
            Some((_, DType::Oq8G256))
        ));
        // qt 35 needs the 258-byte block; 33/34 use the 130-byte block.
        let mut data8 = Vec::new();
        data8.extend_from_slice(&F16_ONE);
        data8.extend_from_slice(&vec![0u8; 256]);
        assert!(matches!(
            oq_repack(35, &data8, 1, 256),
            Some((_, DType::Oq8G256))
        ));
        // Non-OQ quant_types fall through to linear_dtype / dequant.
        assert!(oq_repack(13, &data, 1, 256).is_none());
        assert!(oq_repack(3, &data, 1, 256).is_none());
    }
}
