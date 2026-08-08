// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PaRo (ParoQuant) tensor-loading primitives.
//! These are arch-agnostic: any model crate that loads a ParoQuant safetensors
//! checkpoint can call these instead of reimplementing them.

use crate::llama::{f16_to_f32, ParoRotation, WeightTensor};
use crate::model_source::ModelSource;
use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── AWQ repack (sole copy; hfq.rs calls this via crate::paro) ──

/// Repack AWQ-format INT4 weights (qweight/qzeros/scales) into flat HFQ4G128
/// nibble layout expected by `gemv_hfq4g128`.
pub fn repack_awq_to_hfq4g128(
    qweight: &[u8],    // I32 raw bytes
    qzeros: &[u8],     // I32 raw bytes
    scales: &[u8],     // F16 raw bytes
    out_dim: usize,    // M (output features)
    in_dim: usize,     // K (input features)
    group_size: usize, // 128
) -> Vec<u8> {
    let groups_per_row = in_dim / group_size;
    let bytes_per_group = 8 + group_size / 2;
    let elements_per_half_group = group_size / 2;
    let bytes_per_row = groups_per_row * bytes_per_group;
    let mut out = vec![0u8; out_dim * bytes_per_row];

    // Parse qweight as &[u32] (LE)
    debug_assert_eq!(
        qweight.as_ptr() as usize % 4,
        0,
        "AWQ qweight not 4-byte aligned"
    );
    let qw: &[u32] =
        unsafe { std::slice::from_raw_parts(qweight.as_ptr() as *const u32, qweight.len() / 4) };
    // qweight shape: [in_dim, out_dim/8] → row-major
    let qw_cols = out_dim / 8;

    // Parse qzeros as &[u32]
    debug_assert_eq!(
        qzeros.as_ptr() as usize % 4,
        0,
        "AWQ qzeros not 4-byte aligned"
    );
    let qz: &[u32] =
        unsafe { std::slice::from_raw_parts(qzeros.as_ptr() as *const u32, qzeros.len() / 4) };
    // qzeros shape: [in_dim/group_size, out_dim/8]
    let qz_cols = out_dim / 8;

    // Parse scales as &[u16] (F16)
    debug_assert_eq!(
        scales.as_ptr() as usize % 2,
        0,
        "AWQ scales not 2-byte aligned"
    );
    let sc: &[u16] =
        unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u16, scales.len() / 2) };
    // scales shape: [in_dim/group_size, out_dim]

    // AWQ nibble reorder: ParoQuant packs with _AWQ_REORDER=(0,2,4,6,1,3,5,7).
    // To extract element m, use the inverse permutation:
    const AWQ_DEQUANT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

    for m in 0..out_dim {
        for g in 0..groups_per_row {
            let row_off = m * bytes_per_row + g * bytes_per_group;

            let scale_f16 = sc[g * out_dim + m];
            let scale_f32 = f16_to_f32(scale_f16);

            let zero_i32 = qz[g * qz_cols + m / 8];
            let zero_nibble = ((zero_i32 >> (AWQ_DEQUANT[m % 8] * 4)) & 0xF) as f32;
            let zero_f32 = -scale_f32 * zero_nibble;

            out[row_off..row_off + 4].copy_from_slice(&scale_f32.to_le_bytes());
            out[row_off + 4..row_off + 8].copy_from_slice(&zero_f32.to_le_bytes());

            let nibble_shift = AWQ_DEQUANT[m % 8] * 4;
            let qw_col = m / 8;
            for i in 0..elements_per_half_group {
                let in_idx0 = g * group_size + i * 2;
                let in_idx1 = in_idx0 + 1;

                let nib0 = ((qw[in_idx0 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;
                let nib1 = ((qw[in_idx1 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;

                out[row_off + 8 + i] = nib0 | (nib1 << 4);
            }
        }
    }

    out
}

// ── Prefix detection ───────────────────────────────────────────────────────────

/// Detect whether the model uses the nested `model.language_model.*` layout
/// (shisa-ai / some Qwen3.5 checkpoints) or the flat `model.*` layout.
/// Returns the prefix string to prepend to all other tensor names.
pub fn paro_text_prefix(source: &dyn ModelSource) -> HipResult<&'static str> {
    if source
        .tensor_info("model.language_model.embed_tokens.weight")
        .is_some()
    {
        Ok("model.language_model")
    } else if source.tensor_info("model.embed_tokens.weight").is_some() {
        Ok("model")
    } else {
        Err(HipError::new(0, "ParoQuant: embed_tokens.weight not found under either model.language_model. or model. layout"))
    }
}

// ── Single weight tensor loading ───────────────────────────────────────────────

/// Load a single ParoQuant weight tensor.
/// `tensor_prefix` is the fully-qualified base name without extension,
/// e.g. `"model.language_model.layers.0.self_attn.q_proj"`.
/// The function reads `.qweight`, `.qzeros`, `.scales`, `.pairs`, `.theta`,
/// and `.channel_scales` from `source`, repacks to HFQ4G128, and uploads all
/// rotation sidecars to GPU.
pub fn load_paro_weight(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    tensor_prefix: &str, // e.g. "model.language_model.layers.0.mlp.gate_proj"
    out_dim: usize,      // M
    in_dim: usize,       // K
    group_size: u32,
    krot: u8,
) -> HipResult<WeightTensor> {
    let qw_name = format!("{tensor_prefix}.qweight");
    let qz_name = format!("{tensor_prefix}.qzeros");
    let sc_name = format!("{tensor_prefix}.scales");
    let pairs_name = format!("{tensor_prefix}.pairs");
    let theta_name = format!("{tensor_prefix}.theta");
    let cs_name = format!("{tensor_prefix}.channel_scales");

    let (_, qw_data) = source
        .tensor_data(&qw_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qw_name}")))?;
    let (_, qz_data) = source
        .tensor_data(&qz_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qz_name}")))?;
    let (_, sc_data) = source
        .tensor_data(&sc_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {sc_name}")))?;

    // Repack AWQ → HFQ4G128
    let hfq_data = repack_awq_to_hfq4g128(
        qw_data,
        qz_data,
        sc_data,
        out_dim,
        in_dim,
        group_size as usize,
    );
    let mut staged = Vec::with_capacity(4);
    staged.push(gpu.upload_raw(&hfq_data, &[hfq_data.len()])?);
    #[cfg(feature = "dflash-fault-inject")]
    crate::dflash_generic::generic_dflash_allocation_boundary(
        crate::dflash_generic::GenericDflashConstructionStage::ParoWeightUpload(0),
    )
    .map_err(|e| {
        let tensor = staged.pop().expect("staged Paro weight buffer");
        let _ = gpu.free_tensor(tensor);
        HipError::new(0, &e)
    })?;

    let result = (|| -> HipResult<()> {
        // Load rotation metadata
        let (_, pairs_data) = source.tensor_data(&pairs_name).ok_or_else(|| {
            HipError::new(0, &format!("ParoQuant tensor not found: {pairs_name}"))
        })?;
        staged.push(gpu.upload_raw(pairs_data, &[pairs_data.len()])?);
        #[cfg(feature = "dflash-fault-inject")]
        crate::dflash_generic::generic_dflash_allocation_boundary(
            crate::dflash_generic::GenericDflashConstructionStage::ParoWeightUpload(1),
        )
        .map_err(|e| HipError::new(0, &e))?;

        let (_, theta_data) = source.tensor_data(&theta_name).ok_or_else(|| {
            HipError::new(0, &format!("ParoQuant tensor not found: {theta_name}"))
        })?;
        staged.push(gpu.upload_raw(theta_data, &[theta_data.len()])?);
        #[cfg(feature = "dflash-fault-inject")]
        crate::dflash_generic::generic_dflash_allocation_boundary(
            crate::dflash_generic::GenericDflashConstructionStage::ParoWeightUpload(2),
        )
        .map_err(|e| HipError::new(0, &e))?;

        let (_, cs_data) = source
            .tensor_data(&cs_name)
            .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {cs_name}")))?;
        staged.push(gpu.upload_raw(cs_data, &[cs_data.len()])?);
        #[cfg(feature = "dflash-fault-inject")]
        crate::dflash_generic::generic_dflash_allocation_boundary(
            crate::dflash_generic::GenericDflashConstructionStage::ParoWeightUpload(3),
        )
        .map_err(|e| HipError::new(0, &e))?;
        Ok(())
    })();
    if let Err(error) = result {
        for tensor in staged.into_iter().rev() {
            let _ = gpu.free_tensor(tensor);
        }
        return Err(error);
    }
    let mut tensors = staged.into_iter();
    let buf = tensors.next().expect("staged Paro weight buffer");
    let pairs = tensors.next().expect("staged Paro pairs");
    let theta = tensors.next().expect("staged Paro theta");
    let channel_scales = tensors.next().expect("staged Paro channel scales");

    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::ParoQ4G128,
        m: out_dim,
        k: in_dim,
        row_stride: 0,
        paro: Some(ParoRotation {
            pairs,
            theta,
            channel_scales,
            krot: krot as u32,
            group_size,
            is_alias: false,
        }),
        awq_scale: None,
    })
}

/// Load a weight tensor from a ParoQuant model.
/// Tries `{mp}.{prefix}.qweight` first (quantized); falls back to
/// `{mp}.{prefix}.weight` as FP16 for tensors excluded from quantization
/// (e.g. MoE router, embedding).
pub fn paro_load_wt(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    prefix: &str,
    m: usize,
    k: usize,
    gs: u32,
    kr: u8,
) -> HipResult<WeightTensor> {
    let mp = paro_text_prefix(source)?;
    let fp = format!("{mp}.{prefix}");
    if source.tensor_info(&format!("{fp}.qweight")).is_some() {
        return load_paro_weight(source, gpu, &fp, m, k, gs, kr);
    }
    load_fp16_weight_from_source(source, gpu, &format!("{fp}.weight"), m, k)
}

// ── Norm loading ───────────────────────────────────────────────────────────────

/// Load an RMSNorm weight and add `bias` to every element
/// (`1.0` for qwen3.5/gemma, `0.0` for qwen2/llama).
pub fn paro_load_norm(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
    bias: f32,
) -> HipResult<GpuTensor> {
    let mp = paro_text_prefix(source)?;
    let full = format!("{mp}.{name}");
    let (info, data) = source
        .tensor_data(&full)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {full}")))?;
    // Handles F16/BF16/F32 (raw unquantized checkpoints are commonly BF16).
    let mut v = crate::safetensors_source::source_bytes_to_f32_vec(&info.dtype, data);
    for x in &mut v {
        *x += bias;
    }
    gpu.upload_f32(&v, shape)
}

/// Load a raw F32/F16 tensor from a ParoQuant model (no norm offset).
pub fn paro_load_f32(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
) -> HipResult<GpuTensor> {
    let mp = paro_text_prefix(source)?;
    let full = format!("{mp}.{name}");
    let (info, data) = source
        .tensor_data(&full)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {full}")))?;
    // Handles F16/BF16/F32 (raw unquantized checkpoints are commonly BF16).
    let v = crate::safetensors_source::source_bytes_to_f32_vec(&info.dtype, data);
    gpu.upload_f32(&v, &[n])
}

// ── MoE shared sidecars ────────────────────────────────────────────────────────
// These remain generic (no qwen35-specific field names) so that future MoE
// architectures can reuse them if their MoE layout matches.

/// Build a non-owning `ParoRotation` whose tensor fields alias `src`'s
/// underlying GPU memory. The returned rotation must NOT outlive `src`;
/// callers store the owning sidecar struct to guarantee that.
pub fn alias_paro_rotation(
    pairs_src: &GpuTensor,
    theta_src: &GpuTensor,
    cs_src: &GpuTensor,
    krot: u32,
    group_size: u32,
) -> ParoRotation {
    let alias = |t: &GpuTensor| -> GpuTensor {
        GpuTensor {
            buf: unsafe { t.buf.alias() },
            shape: t.shape.clone(),
            dtype: t.dtype,
        }
    };
    ParoRotation {
        pairs: alias(pairs_src),
        theta: alias(theta_src),
        channel_scales: alias(cs_src),
        krot,
        group_size,
        is_alias: true,
    }
}

/// Load an FP16 weight tensor from safetensors (for excluded/unquantized layers).
pub fn load_fp16_weight_from_source(
    source: &dyn ModelSource,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = source
        .tensor_data(name)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
    // Handles F16/BF16/F32 (raw unquantized checkpoints are commonly BF16).
    let f32_data = crate::safetensors_source::source_bytes_to_f32_vec(&info.dtype, data);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4) };
    let buf = gpu.upload_raw(bytes, &[m, k])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::F32,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

// ── MoE per-expert repack ──────────────────────────────────────────────────────

/// Repack a single per-expert AWQ projection (gate, up, or down) into HFQ4G128
/// byte rows. Returns the row-major byte buffer.
pub fn paro_repack_moe_projection(
    source: &dyn ModelSource,
    full_prefix: &str,
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> HipResult<Vec<u8>> {
    let qw_name = format!("{full_prefix}.qweight");
    let qz_name = format!("{full_prefix}.qzeros");
    let sc_name = format!("{full_prefix}.scales");
    let (_, qw_data) = source
        .tensor_data(&qw_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant MoE tensor not found: {qw_name}")))?;
    let (_, qz_data) = source
        .tensor_data(&qz_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant MoE tensor not found: {qz_name}")))?;
    let (_, sc_data) = source
        .tensor_data(&sc_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant MoE tensor not found: {sc_name}")))?;
    Ok(repack_awq_to_hfq4g128(
        qw_data, qz_data, sc_data, out_dim, in_dim, group_size,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_source::{QuantConfig, TensorInfo};
    use std::collections::HashMap;
    use std::path::Path;

    struct MockSource {
        tensors: HashMap<String, TensorInfo>,
        qc: Option<QuantConfig>,
        layout: &'static str,
    }
    impl MockSource {
        fn with_layout(layout: &'static str) -> Self {
            let mut s = Self {
                tensors: Default::default(),
                qc: None,
                layout,
            };
            // plant the sentinel tensor the prefix-detector checks
            let key = format!("{layout}.embed_tokens.weight");
            s.tensors.insert(
                key.clone(),
                TensorInfo {
                    name: key,
                    dtype: "F16".into(),
                    shape: vec![1, 1],
                    quant_type: 0xFF,
                    data_offset: 0,
                    data_size: 2,
                },
            );
            s
        }
    }
    // Implement only the methods paro_text_prefix actually calls:
    impl ModelSource for MockSource {
        fn metadata_json(&self) -> &str {
            "{}"
        }
        fn arch_id(&self) -> u32 {
            5
        }
        fn quant_config(&self) -> Option<&QuantConfig> {
            self.qc.as_ref()
        }
        fn tensor_data(&self, _name: &str) -> Option<(&TensorInfo, &[u8])> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name)
        }
        fn tensor_names(&self) -> Vec<&str> {
            self.tensors.keys().map(|s| s.as_str()).collect()
        }
        fn path(&self) -> &Path {
            Path::new("/tmp/mock")
        }
    }

    #[test]
    fn paro_text_prefix_nested_layout() {
        let src = MockSource::with_layout("model.language_model");
        assert_eq!(paro_text_prefix(&src).unwrap(), "model.language_model");
    }

    #[test]
    fn paro_text_prefix_flat_layout() {
        let src = MockSource::with_layout("model");
        assert_eq!(paro_text_prefix(&src).unwrap(), "model");
    }

    #[test]
    fn paro_text_prefix_unknown_layout() {
        let src = MockSource {
            tensors: Default::default(),
            qc: None,
            layout: "unknown",
        };
        assert!(paro_text_prefix(&src).is_err());
    }
}
