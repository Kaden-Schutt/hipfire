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
use crate::{BlockKind, NemotronHConfig};
use hipfire_model::ModelSource;

/// Decode a safetensors tensor's raw bytes to `Vec<f32>` per its dtype string.
fn dequant(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        "F16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        other => Err(format!("nemotron loader: unsupported dtype {other:?}")),
    }
}

/// IEEE half → f32 (handles subnormals/inf/nan).
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = match exp {
        0 => (mant as f32) * 2f32.powi(-24),
        0x1f => {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// Read tensor `name` from the source and dequantize to `Vec<f32>`.
fn get(src: &dyn ModelSource, name: &str) -> Result<Vec<f32>, String> {
    let (info, bytes) = src
        .tensor_data(name)
        .ok_or_else(|| format!("nemotron loader: missing tensor {name:?}"))?;
    dequant(&info.dtype, bytes)
}

/// Load all nemotron_h weights from `src` into host f32 [`NemotronWeights`].
pub fn load_nemotron_weights(
    src: &dyn ModelSource,
    cfg: &NemotronHConfig,
) -> Result<NemotronWeights, String> {
    let embeddings = get(src, "backbone.embeddings.weight")?;
    let norm_f = get(src, "backbone.norm_f.weight")?;
    let lm_head = if cfg.tie_word_embeddings {
        embeddings.clone()
    } else {
        get(src, "lm_head.weight")?
    };

    // nemotron_h applies the GPT-2 residual-rescale (`_init_weights`,
    // `rescale_prenorm_residual`) to the Mamba mixer `out_proj.weight` at LOAD:
    // `out_proj /= sqrt(num_layers)`. The checkpoint stores the un-rescaled
    // weight, so we must apply it here to match the reference forward (verified
    // exact vs the HF dump: ratio == sqrt(num_layers)). Only the Mamba
    // `out_proj` is rescaled — NOT attention `o_proj` or MLP `down_proj`.
    let out_proj_scale = 1.0f32 / (cfg.num_layers as f32).sqrt();

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
