// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! nemotron_h weight loader: a [`hipfire_model::ModelSource`] (BF16 safetensors)
//! → host [`crate::model::NemotronWeights`] (dequantized f32). N5a.
//!
//! Tensor-name map confirmed from the Nano-4B checkpoint (see the plan doc):
//! every block has `backbone.layers.{L}.norm.weight` + a `.mixer.` namespace;
//! globals are `backbone.embeddings.weight`, `backbone.norm_f.weight`,
//! `lm_head.weight`. Block kind comes from `cfg.blocks` (the
//! `hybrid_override_pattern`), not the tensor names.

use crate::model::{HostBlock, NemotronWeights};
use crate::moe::{MoeExpertWeights, MoeWeights};
use crate::{BlockKind, NemotronHConfig};
use hipfire_model::ModelSource;
use hipfire_primitives::conv::{decode_plain_dtype_to_f32, f16_to_f32};

/// Decode a safetensors tensor's raw bytes to `Vec<f32>` per its dtype string.
fn dequant(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    decode_plain_dtype_to_f32(dtype, bytes).map_err(|e| format!("nemotron loader: {e}"))
}

/// Read tensor `name` from the source and dequantize to `Vec<f32>`.
fn get(src: &dyn ModelSource, name: &str) -> Result<Vec<f32>, String> {
    let (info, bytes) = src
        .tensor_data(name)
        .ok_or_else(|| format!("nemotron loader: missing tensor {name:?}"))?;
    dequant(&info.dtype, bytes)
}

/// Read the first available tensor from `names`.
fn get_any(src: &dyn ModelSource, names: &[&str]) -> Result<Vec<f32>, String> {
    for name in names {
        if let Some((info, bytes)) = src.tensor_data(name) {
            return dequant(&info.dtype, bytes);
        }
    }
    Err(format!("nemotron loader: missing tensor; tried {names:?}"))
}

/// Load all nemotron_h weights from `src` into host f32 [`NemotronWeights`].
pub fn load_nemotron_weights(
    src: &dyn ModelSource,
    cfg: &NemotronHConfig,
) -> Result<NemotronWeights, String> {
    let embeddings = get_any(
        src,
        &["backbone.embeddings.weight", "backbone.embedding.weight"],
    )?;
    let norm_f = get(src, "backbone.norm_f.weight")?;
    let lm_head = match get(src, "lm_head.weight") {
        Ok(w) => w,
        Err(_) if cfg.tie_word_embeddings => embeddings.clone(),
        Err(e) => return Err(e),
    };

    // Nemotron-H Mamba out-proj scaling is checkpoint-family specific. Dense
    // Nano-4B needs the GPT-style residual rescale applied at load; Nano-30B MoE
    // already matches the HF reference with stored bytes.
    let out_proj_scale = cfg.mamba_out_proj_runtime_scale();

    let mut layer_norm = Vec::with_capacity(cfg.num_layers);
    let mut blocks = Vec::with_capacity(cfg.num_layers);
    for (l, kind) in cfg.blocks.iter().enumerate() {
        let p = format!("backbone.layers.{l}");
        layer_norm.push(get(src, &format!("{p}.norm.weight"))?);
        let m = format!("{p}.mixer");
        let block = match kind {
            BlockKind::Mamba2 => HostBlock::Mamba2 {
                in_proj: get(src, &format!("{m}.in_proj.weight"))?,
                // conv1d.weight is [conv_dim, 1, K] in the checkpoint; the middle
                // dim is 1, so the flat layout is already [conv_dim, K] (c*K+k).
                conv_weight: get(src, &format!("{m}.conv1d.weight"))?,
                conv_bias: get(src, &format!("{m}.conv1d.bias"))?,
                a_log: get(src, &format!("{m}.A_log"))?,
                d: get(src, &format!("{m}.D"))?,
                dt_bias: get(src, &format!("{m}.dt_bias"))?,
                mixer_norm: get(src, &format!("{m}.norm.weight"))?,
                out_proj: {
                    let mut w = get(src, &format!("{m}.out_proj.weight"))?;
                    for v in w.iter_mut() {
                        *v *= out_proj_scale;
                    }
                    w
                },
            },
            BlockKind::Mlp => HostBlock::Mlp {
                up: get(src, &format!("{m}.up_proj.weight"))?,
                down: get(src, &format!("{m}.down_proj.weight"))?,
            },
            BlockKind::Attention => HostBlock::Attn {
                q: get(src, &format!("{m}.q_proj.weight"))?,
                k: get(src, &format!("{m}.k_proj.weight"))?,
                v: get(src, &format!("{m}.v_proj.weight"))?,
                o: get(src, &format!("{m}.o_proj.weight"))?,
            },
            BlockKind::Moe => {
                let moe = cfg
                    .moe
                    .ok_or_else(|| format!("nemotron loader: MoE block {l} has no MoE config"))?;
                let mut experts = Vec::with_capacity(moe.n_routed_experts);
                for expert_idx in 0..moe.n_routed_experts {
                    experts.push(MoeExpertWeights {
                        up: get(src, &format!("{m}.experts.{expert_idx}.up_proj.weight"))?,
                        down: get(src, &format!("{m}.experts.{expert_idx}.down_proj.weight"))?,
                    });
                }
                HostBlock::Moe(Box::new(MoeWeights {
                    router: get(src, &format!("{m}.gate.weight"))?,
                    expert_bias: get(src, &format!("{m}.gate.e_score_correction_bias"))?,
                    shared_up: get(src, &format!("{m}.shared_experts.up_proj.weight"))?,
                    shared_down: get(src, &format!("{m}.shared_experts.down_proj.weight"))?,
                    experts,
                }))
            }
        };
        blocks.push(block);
    }

    Ok(NemotronWeights {
        embeddings,
        layer_norm,
        blocks,
        norm_f,
        lm_head,
    })
}

// ── HFQ (quantized) loading ──────────────────────────────────────────────────

use crate::weight::{EmbeddingTable, LinearWeight};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weights::WeightTensor;
use rdna_compute::{DType, Gpu};

/// quant_type byte → quantized linear `DType` (None ⇒ stored as a plain
/// precision, handle via `dequant_qt`). See `qt_name` in hipfire-quantize.
fn linear_dtype(qt: u8) -> Option<DType> {
    match qt {
        3 => Some(DType::Q8_0), // Q8F16
        6 => Some(DType::HFQ4G256),
        7 => Some(DType::HFQ4G128),
        13 => Some(DType::MQ4G256),
        _ => None,
    }
}

/// Dequantize a plain-precision HFQ tensor (F16=1, F32=2, BF16=16) to f32.
fn dequant_qt(qt: u8, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match qt {
        1 => dequant("F16", bytes),
        2 => dequant("F32", bytes),
        16 => dequant("BF16", bytes),
        other => Err(format!(
            "nemotron hfq: unsupported plain quant_type {other}"
        )),
    }
}

fn hfq_tensor<'a>(hfq: &'a HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("nemotron hfq: missing tensor {name:?}"))?;
    let _ = &info.shape;
    Ok((info.quant_type, data))
}

pub fn first_hfq_tensor<'a>(hfq: &HfqFile, names: &[&'a str]) -> Result<&'a str, String> {
    for name in names {
        if hfq.find_tensor_info(name).is_some() {
            return Ok(name);
        }
    }
    Err(format!("nemotron hfq: missing tensor; tried {names:?}"))
}

/// Repack the on-disk row-major grouped OQ4 (qt=34) bytes into the arch-combined
/// `Oq4G256` layout the dispatch/gemm expect (nibbles + per-group f32 scales +
/// interleaved scale+nibble blocks). Self-contained copy of the canonical packer
/// (per the gemma3 loader convention — no cross-arch dep). `M`=rows, `K`=cols.
fn oq4_pack_arch_combined(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    const GROUP: usize = 256;
    const BLOCK: usize = 130;
    const ILB: usize = 132;
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

/// Load one linear weight as a `LinearWeight` (quantized when 4-bit/Q8, else an
/// F32 upload). `m`=out rows, `k`=in cols. The nemotron HFQ has no awq sidecars,
/// so MQ4's FWHT rotation is applied automatically by the dispatched gemv.
pub fn load_linear_hfq(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<LinearWeight, String> {
    let (qt, data) = hfq_tensor(hfq, name)?;
    // OQ4 (Opus Quant, symmetric W4): qt=34 is the row-major grouped on-disk form
    // that needs the arch-combined repack; qt=37 is already arch-packed. Both
    // dispatch as `DType::Oq4G256` (gemv via the shared dispatch, batched prefill
    // via `hipfire_runtime::weights::weight_gemm`). Mirrors the hfq.rs LLaMA loader.
    if qt == 34 || qt == 37 {
        let packed = if qt == 34 {
            oq4_pack_arch_combined(&data, m, k)
        } else {
            data
        };
        let buf = gpu
            .upload_raw(&packed, &[packed.len()])
            .map_err(|e| format!("nemotron hfq oq4 upload {name}: {e:?}"))?;
        // oq4+/oq4++ carry a per-input-channel `<name>.awq_scale.weight` sidecar
        // the dispatch applies to activations; plain oq4 has none (→ None).
        let awq_scale = hipfire_runtime::hfq::load_awq_scale(hfq, gpu, name, k);
        return Ok(LinearWeight::Quant(Box::new(WeightTensor {
            buf,
            gpu_dtype: DType::Oq4G256,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale,
        })));
    }
    if let Some(dtype) = linear_dtype(qt) {
        let buf = gpu
            .upload_raw(&data, &[data.len()])
            .map_err(|e| format!("nemotron hfq upload {name}: {e:?}"))?;
        Ok(LinearWeight::Quant(Box::new(WeightTensor {
            buf,
            gpu_dtype: dtype,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        })))
    } else {
        let f = dequant_qt(qt, &data)?;
        Ok(LinearWeight::F32(
            gpu.upload_f32(&f, &[m, k])
                .map_err(|e| format!("nemotron hfq f32 {name}: {e:?}"))?,
        ))
    }
}

/// Load a tensor stored at plain precision (BF16/F16/F32) as host f32 — for the
/// recurrence + norm tensors the quantizer keeps un-quantized.
pub fn load_f32_hfq(hfq: &HfqFile, name: &str) -> Result<Vec<f32>, String> {
    let (qt, data) = hfq_tensor(hfq, name)?;
    dequant_qt(qt, &data)
}

/// Load the embedding table: Q8 (kept quantized, looked up via
/// `embedding_lookup_q8`) or a plain-precision f32 upload.
pub fn load_embeddings_hfq(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    vocab: usize,
    hidden: usize,
) -> Result<EmbeddingTable, String> {
    let (qt, data) = hfq_tensor(hfq, name)?;
    if qt == 3 {
        let buf = gpu
            .upload_raw(&data, &[data.len()])
            .map_err(|e| format!("nemotron hfq emb {name}: {e:?}"))?;
        Ok(EmbeddingTable::Q8(buf))
    } else {
        let f = dequant_qt(qt, &data)?;
        Ok(EmbeddingTable::F32(
            gpu.upload_f32(&f, &[vocab, hidden])
                .map_err(|e| format!("nemotron hfq emb f32 {name}: {e:?}"))?,
        ))
    }
}
