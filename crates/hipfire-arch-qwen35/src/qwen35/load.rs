// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 weight loading: HFQ / ParoQuant sources, AWQ repack, packed-MQ4
//! experts, `load_weights`, and the EP sharded loader.

use super::config::f16_lm_head_mode_from_config;
use super::config::F16LmHeadMode;
use super::config::Qwen35Config;
use super::escha;
use super::escha::EschaWeightStore;
use super::forward::layers_have_mq6_moe;
use super::weights::dtype_from_quant_type;
use super::weights::mixed_expert_tag;
use super::weights::DeltaNetLayerWeights;
use super::weights::ExpertWeights;
use super::weights::FullAttnLayerWeights;
use super::weights::LayerWeights;
use super::weights::MoeFfnWeights;
use super::weights::MoeParoSidecars;
use super::weights::PackedExpertOwners;
use super::weights::PendingEpMoeFfn;
use super::weights::Qwen35EpConfigFingerprint;
use super::weights::Qwen35EpShardInfo;
use super::weights::Qwen35HfqSourceIdentity;
use super::weights::Qwen35RankSeal;
use super::weights::Qwen35Weights;
use super::weights::SharedExpertWeights;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::hfq::HfqTensorInfo;
use hipfire_runtime::hfq_parallel::read_hfq_jobs_ordered;
use hipfire_runtime::hfq_parallel::HfqReadJob;
use hipfire_runtime::llama::f16_to_f32;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::ParoRotation;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::model_load::load_weights as rt_load_weights;
use hipfire_runtime::model_load::LoadedWeights;
use hipfire_runtime::model_load::WeightSource;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::paro::paro_load_norm;
use hipfire_runtime::paro::paro_text_prefix;
use hipfire_runtime::tp_shard::ShardConfig;
use hipfire_runtime::weight_backend::dequant_norm;
use hipfire_runtime::weight_backend::dequant_weight_raw;
use hipfire_runtime::weight_backend::load_awq_scale_for;
use hipfire_runtime::weight_backend::load_embedding;
use hipfire_runtime::weight_backend::resolve_lm_head;
use hipfire_runtime::weight_backend::reupload_f16_as_f32;
use hipfire_runtime::weight_backend::HfqBackend;
use hipfire_runtime::weight_backend::ParoBackend;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

/// RMSNorm weight bias for qwen3.5/gemma-style norms: dequant computes `w + norm_bias`.
/// qwen2/llama use `0.0`. Single source of truth — referenced by the backend constructors
/// and both final-norm paths so the four former hardcoded `1.0` sites cannot drift apart.
const QWEN35_NORM_BIAS: f32 = 1.0;

const _: () = assert!(QWEN35_NORM_BIAS == 1.0);

// ─── Weight loading ─────────────────────────────────────────────────────

/// Public so gates outside the loader (e.g.
/// `examples/test_escha_dense_linear_gpu_vs_cpu.rs`) resolve names through the
/// SAME aliasing the production load uses. A gate with its own copy of this
/// would stop testing what actually runs the moment the two drifted.
pub fn qwen35_tensor_name_candidates(name: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(4);
    let mut push = |s: String| {
        if !out.iter().any(|x| x == &s) {
            out.push(s);
        }
    };

    if name == "lm_head.weight" {
        push(name.to_string());
        push("model.language_model.lm_head.weight".to_string());
        push("model.lm_head.weight".to_string());
        return out;
    }

    if name.starts_with("model.") {
        push(name.to_string());
    } else {
        push(format!("model.language_model.{name}"));
        push(format!("model.{name}"));
        push(name.to_string());
    }

    // Escha alias. A trellis-coded projection has NO `.weight` — it ships
    // `escha_code` + `escha_rin` + `escha_rout`. Offering the code under the
    // same lookup lets the ordinary `b.proj(...)` path find it and build a
    // `WeightTensor` with dtype Escha2T16/3T16; `escha_sidecars` then attaches
    // the rotation vectors.
    //
    // Ordered LAST on purpose: a checkpoint carrying both a real `.weight` and
    // a stale `escha_code` must resolve to the weight, not silently prefer a
    // code the runtime would then decode against the wrong activation.
    if let Some(stem) = name.strip_suffix(".weight") {
        let alias = format!("{stem}.escha_code");
        if alias.starts_with("model.") {
            push(alias);
        } else {
            push(format!("model.language_model.{alias}"));
            push(format!("model.{alias}"));
            push(alias);
        }
    }
    out
}

fn qwen35_tensor_data_vec<'a>(
    hfq: &'a HfqFile,
    name: &str,
) -> Option<(&'a HfqTensorInfo, Vec<u8>)> {
    for candidate in qwen35_tensor_name_candidates(name) {
        if let Some(found) = hfq.tensor_data_vec(&candidate) {
            return Some(found);
        }
    }
    None
}

/// Borrowed-first variant of [`qwen35_tensor_data_vec`]: returns the mmap
/// slice directly when the mapping is alive (dGPU loads keep it, so weight
/// uploads DMA straight out of page-cache pages with no heap staging copy);
/// falls back to the owned pread Vec on UMA (mmap dropped there). Same
/// candidate resolution.
fn qwen35_tensor_data_cow<'a>(
    hfq: &'a HfqFile,
    name: &str,
) -> Option<(&'a HfqTensorInfo, std::borrow::Cow<'a, [u8]>)> {
    for candidate in qwen35_tensor_name_candidates(name) {
        if let Some((info, data)) = hfq.tensor_data(&candidate) {
            return Some((info, std::borrow::Cow::Borrowed(data)));
        }
    }
    for candidate in qwen35_tensor_name_candidates(name) {
        if let Some((info, data)) = hfq.tensor_data_vec(&candidate) {
            return Some((info, std::borrow::Cow::Owned(data)));
        }
    }
    None
}

fn load_norm_weight(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> HipResult<GpuTensor> {
    let (info, data) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));
    dequant_norm(gpu, info.quant_type, &data, shape, QWEN35_NORM_BIAS)
}

fn load_weight_tensor_raw(
    gpu: &Gpu,
    quant_type: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    match quant_type {
        6 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        7 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        8 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        11 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        12 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        13 => {
            // MQ4-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        14 => {
            // MQ8-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ8G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        15 => {
            // MQ6-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        17 => {
            // MQ3-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        18 => {
            // MQ2-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        19 => {
            // MQ2-G256-Lloyd — 2-bit + 4-entry fp16 codebook (72 bytes/group)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        20 => {
            // MQ3-G256-Lloyd — 3-bit + 8-entry fp16 codebook (112 bytes/group)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        30 => {
            // MQ4-G256-Lloyd — 4-bit + 16-entry fp16 codebook (160 bytes/group)
            // Renumbered from qt 21 → 30 in mq4-lloyd merge to avoid HFP4G32=21 collision.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        42 | 43 => {
            // Escha-W2 trellis code, kept VERBATIM — the 2-bit/3-bit stream is
            // decoded inside the GEMV, never at load. That is the whole point
            // of the format: an 11.16 GB resident 27B instead of 22.63 GB
            // folded, at better quality (PPL 11.8654 vs 13.6957).
            //
            // Opaque raw buffer like the MQ arms above, but the resemblance
            // ends there: an escha weight is NOT self-contained. It needs its
            // `escha_rin_eff`/`escha_rout_eff` vectors and an H128 on both
            // sides of the GEMV, which is why `EschaDenseLinear` exists and
            // why the fused MQ paths (FusedQkv/FusedQkvza/gate_up) CANNOT
            // consume one — each projection needs its own rin-rotated
            // activation. A layer holding these must route through
            // `escha::escha_dense_linear_forward`.
            //
            // `m`/`k` are the logical output/input dims; the buffer length is
            // the tile-packed code, not m*k of anything.
            //
            // TILE GRID TRANSPOSED HERE, kt-major -> nt-major. Every escha
            // kernel holds one output tile column `nt` fixed and walks `kt`;
            // in the checkpoint's order consecutive `kt` are a full tile-row
            // apart (139 KB on the 27B's gate_proj), so each step is a fresh
            // 64-bit address against a cold line. Adjacent instead: measured
            // 243.9 -> 186.3 us on the decode GEMV, 24%.
            //
            // ONLY THIS (DENSE) LOADER PERMUTES. MoE experts come through
            // `escha::load_escha_moe_experts` and stay kt-major, which is why
            // the kernels take an `nt_major` flag rather than assuming a
            // layout — the dense call sites pass `true`, MoE passes `false`.
            //
            // At LOAD, not in the converter, so the payload stays verbatim
            // from upstream: no re-convert, no re-upload, no format version.
            // Whole tiles move and their contents are untouched, so every
            // decoded weight is identical. Gated on mean KLD = 0.000000.
            let permuted = escha_tiles_to_nt_major(data, m, k, quant_type)?;
            let buf = gpu.upload_raw(&permuted, &[permuted.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: if quant_type == 42 {
                    DType::Escha2T16
                } else {
                    DType::Escha3T16
                },
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        31 => {
            // MQ5-G256 — MagnumQuant FWHT-rotated 5-bit (168 bytes/group, 5.25 bpw).
            // Opaque raw buffer, same pattern as MQ4(13)/MQ6(15); the GEMV
            // dispatch FWHT-rotates x at use. AWQ sidecar attached by the
            // caller via DType::supports_awq_sidecar (already includes MQ5G256).
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ5G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        21 => {
            // HFP4G32 — E2M1 + UE8M0 g32 + FP16 row scale. See docs/quant-formats/hfp4.md.
            // K%256 — kernel constraint (gemv_hfp4g32 in dispatch.rs); refuse here so a
            // stale or externally-quantized file fails at load instead of panicking on
            // first dispatch.
            assert!(
                k % 256 == 0,
                "HFP4G32 v1 lm_head has K={k} but kernel requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        24 => {
            // MFP4G32 — HFP4G32 + offline FWHT. Drop-in MQ4 replacement; same byte
            // layout as qtype 21 with format_flags=0x05 stamped in the per-row hdr.
            assert!(
                k % 256 == 0,
                "MFP4G32 lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        32 => {
            // MFP4G32Lloyd lm_head: mfp4 rows + 32-B per-tensor fp16 codebook prefix.
            assert!(
                k % 256 == 0,
                "MFP4G32Lloyd lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        33 => {
            // MFP4G32P lm_head: mfp4+P — mfp4 rows with E4M3 per-block scale. NO prefix;
            // byte-identical layout to MFP4G32 (qt 24).
            assert!(
                k % 256 == 0,
                "MFP4G32P lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32P,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        34 => {
            // MFP4G32E8 lm_head: mfp4-E8 — mfp4+P container, NO prefix, same row_bytes;
            // per-32-block 16 E2M1 nibbles replaced by 4x32-bit E8-lattice codewords.
            assert!(
                k % 256 == 0,
                "MFP4G32E8 lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32E8,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        36 => {
            // MFP3G32E8: mfp4-E8 frame with 3-bit lattice, 13 B/blk, 3.25 bpw.
            // Drop-in cold tier for MQ3G256Lloyd (kernel tag 5).
            assert!(
                k % 256 == 0,
                "MFP3G32E8 has K={k} but FWHT requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP3G32E8,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        37 => {
            // MFP2G32E8: mfp4-E8 frame with 2-bit lattice, 9 B/blk, 2.25 bpw.
            // Drop-in cold tier for MQ2G256Lloyd (kernel tag 6).
            assert!(
                k % 256 == 0,
                "MFP2G32E8 has K={k} but FWHT requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP2G32E8,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        44 => {
            // MQ4-G256 v2 (qt=44): 136 B/group, byte-identical payload to MQ4G256
            // but per-128 fp16 scale+zero. Validate K%256 and blob length.
            if k % 256 != 0 {
                return Err(HipError::new(
                    0,
                    &format!("MQ4G256V2 has K={k} but requires K%256==0"),
                ));
            }
            let gpr = k / 256;
            let expected = m * gpr * 136;
            if data.len() != expected {
                return Err(HipError::new(
                    0,
                    &format!(
                        "MQ4G256V2 blob length mismatch: expected {expected}, got {}",
                        data.len()
                    ),
                ));
            }
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        45 => {
            // MQ4C (qt=45), pad layout: 136 B/group — ONE fp16 scale+zero dword at
            // +0 governing all 256 weights, 4 B of zero padding at +4, and the
            // 128 B nibble payload at +8, which is byte-for-byte where qt=13 puts
            // it. Same total size as qt=13; the padding is the deliberate price of
            // keeping the payload 8-byte aligned.
            //
            // Derive the stride from rdna_compute::MQ4C_GROUP_BYTES rather than a
            // literal. This site previously hardcoded 132 and rejected every valid
            // file the moment the format moved to 136 — the check was right, the
            // duplicated constant was not.
            if k % 256 != 0 {
                return Err(HipError::new(
                    0,
                    &format!("MQ4CG256 has K={k} but requires K%256==0"),
                ));
            }
            let gpr = k / 256;
            let expected = m * gpr * rdna_compute::MQ4C_GROUP_BYTES;
            if data.len() != expected {
                return Err(HipError::new(
                    0,
                    &format!(
                        "MQ4CG256 blob length mismatch: expected {expected}, got {}",
                        data.len()
                    ),
                ));
            }
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4CG256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        47 => {
            // MQ6-G256 v2 (qt=47): 200 B/group, neutral Magnum V2.
            // Header LE `[0..2)` fp16 s0, `[2..4)` fp16 z0, `[4..6)` fp16 s1,
            // `[6..8)` fp16 z1, `[8..200)` 192 B 6-bit payload (4/3 B).
            // Half 0 covers q[0..128), half 1 q[128..256); `w = q*f32(s[h])+f32(z[h])`.
            // Mirror every qt44 guard: K%256 and exact byte count fail closed.
            if k % 256 != 0 {
                return Err(HipError::new(
                    0,
                    &format!("MQ6G256V2 has K={k} but requires K%256==0"),
                ));
            }
            let gpr = k / 256;
            let expected = m * gpr * rdna_compute::MQ6G256V2_GROUP_BYTES;
            if data.len() != expected {
                return Err(HipError::new(
                    0,
                    &format!(
                        "MQ6G256V2 blob length mismatch: expected {expected}, got {}",
                        data.len()
                    ),
                ));
            }
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ6G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        48 => {
            // MQ5-G256 v2 (qt=48): 168 B/group, neutral Magnum V2.
            // Header LE `[0..2)` fp16 s0, `[2..4)` fp16 z0, `[4..6)` fp16 s1,
            // `[6..8)` fp16 z1, `[8..168)` 160 B 5-bit payload (8/5 B).
            if k % 256 != 0 {
                return Err(HipError::new(
                    0,
                    &format!("MQ5G256V2 has K={k} but requires K%256==0"),
                ));
            }
            let gpr = k / 256;
            let expected = m * gpr * rdna_compute::MQ5G256V2_GROUP_BYTES;
            if data.len() != expected {
                return Err(HipError::new(
                    0,
                    &format!(
                        "MQ5G256V2 blob length mismatch: expected {expected}, got {}",
                        data.len()
                    ),
                ));
            }
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ5G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        49 => {
            // MQ3-G256 v2 (qt=49): 104 B/group, neutral Magnum V2.
            // Header LE `[0..2)` fp16 s0, `[2..4)` fp16 z0, `[4..6)` fp16 s1,
            // `[6..8)` fp16 z1, `[8..104)` 96 B 3-bit payload (8/3 B).
            if k % 256 != 0 {
                return Err(HipError::new(
                    0,
                    &format!("MQ3G256V2 has K={k} but requires K%256==0"),
                ));
            }
            let gpr = k / 256;
            let expected = m * gpr * rdna_compute::MQ3G256V2_GROUP_BYTES;
            if data.len() != expected {
                return Err(HipError::new(
                    0,
                    &format!(
                        "MQ3G256V2 blob length mismatch: expected {expected}, got {}",
                        data.len()
                    ),
                ));
            }
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        50 => {
            // MQ2-G256 v2 (qt=50): 72 B/group, neutral Magnum V2.
            // Header LE `[0..2)` fp16 s0, `[2..4)` fp16 z0, `[4..6)` fp16 s1,
            // `[6..8)` fp16 z1, `[8..72)` 64 B 2-bit payload (4/B).
            if k % 256 != 0 {
                return Err(HipError::new(
                    0,
                    &format!("MQ2G256V2 has K={k} but requires K%256==0"),
                ));
            }
            let gpr = k / 256;
            let expected = m * gpr * rdna_compute::MQ2G256V2_GROUP_BYTES;
            if data.len() != expected {
                return Err(HipError::new(
                    0,
                    &format!(
                        "MQ2G256V2 blob length mismatch: expected {expected}, got {}",
                        data.len()
                    ),
                ));
            }
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256V2,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        38 => {
            // MQ2-G256-GL — 2-bit codes vs the TENSOR-GLOBAL codebook GL_CB2 +
            // per-block fp16 scale. 2.0625 bpw. SoA, TWO regions, no per-group
            // header (this is the first format in the tree where that is true):
            //   [0 .. m*gpr*64)                  packed 2-bit indices, 64 B/group
            //   [m*gpr*64 .. + m*gpr*2)          fp16 per-block scales
            // with gpr = k/256. Opaque raw upload — the indexed MoE GEMVs
            // (gemv_mq2g256gl_moe_{gate_up,down}_indexed) decode it and receive
            // the codebook as scalar kernel args, so nothing here needs it.
            //
            // K%256 is a HARD requirement: gpr = k/256 truncates otherwise and
            // the scale-region base M*gpr*64 shifts, which decodes to plausible
            // garbage with no error. Fail at load.
            assert!(
                k % 256 == 0,
                "MQ2G256GL has K={k} but the SoA region split requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256GL,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        39 => {
            // MQ3-G256-GL — 3-bit sibling of qt 38 (global codebook GL_CB3,
            // 8 entries). 3.0625 bpw; 96 B of indices per group then the same
            // trailing m*gpr*2 B fp16 scale region.
            assert!(
                k % 256 == 0,
                "MQ3G256GL has K={k} but the SoA region split requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256GL,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        40 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::TQ2G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        41 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::BQ1G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        3 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        1 => match f16_lm_head_mode_from_config() {
            F16LmHeadMode::Native => dequant_weight_raw(gpu, quant_type, data, m, k),
            F16LmHeadMode::F32 => {
                let f32_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
                };
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
        },
        2 => {
            // F32 — native full-precision oracle weights (qt=2). Raw f32 LE
            // bytes uploaded as-is; the engine forwards through gemv_f32 /
            // gemm_f32_batched / attention_f32. Part of the F1 native-bf16
            // reference path (no quantization).
            let buf = gpu.upload_raw(data, &[m, k])?;
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
        16 => {
            // Native BF16 storage, F32 accumulation. This is source-exact for
            // BF16 checkpoints while retaining the two-byte memory traffic;
            // the unified dispatcher routes it through GemvBf16.
            let buf = gpu.upload_raw(data, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::BF16,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        35 => {
            // MFP4G32E8SOA lm_head: mfp4-E8 SoA layout for coalesced GEMV.
            assert!(
                k % 256 == 0,
                "MFP4G32E8SOA lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32E8SOA,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        _ => dequant_weight_raw(gpu, quant_type, data, m, k),
    }
}

/// Phase A Stage A — AWQ sidecar loader for the Qwen3.5 forward path.
///
/// The .hfq quantizer emits `<weight>.awq_scale.weight` (1D F16, length K)
/// alongside MQ4G256 weights that were AWQ pre-scaled. The dispatcher in
/// `fused_rmsnorm_rotate_for_mq` / `fused_rmsnorm_rotate_mq_batched_for`
/// looks at `WeightTensor.awq_scale.is_some()` to pick the AWQ-aware
/// kernel variant. WITHOUT this loader populating the field, every MQ4
/// weight ends up with `awq_scale: None`, the dispatcher falls through
/// to the non-AWQ kernel, and the math `(W·s) · (x/s) = W·x` breaks
/// because the runtime never divides by `s` — observed KLD blowup
/// 0.6721 → 13.4893 on 0.8B Qwen3.5 before this landed.
///
/// Lookup pattern matches `hipfire_runtime::hfq::load_awq_scale`:
/// strip trailing `.weight`, append `.awq_scale.weight`. Try both the
/// `model.language_model.`-prefixed name and the bare name (the qwen35
/// crate uses prefixed names; older sidecars or tests may use either).
/// TODO(transformer-extraction): cross-arch duplicate of
/// `hipfire-arch-qwen2::qwen2::load_weight_tensor` — same name-lookup +
/// pread + AWQ-sidecar pattern, but qwen35 uses the
/// `model.language_model.` prefix (its HFQ files put text weights under
/// the VL-friendly nested name) where qwen2 uses flat `model.{...}`.
/// Pull into `hipfire_runtime::transformer::weights` with the prefix
/// as a parameter during consolidation.
pub(crate) fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
    candidates: fn(&str) -> Vec<String>,
) -> HipResult<WeightTensor> {
    // Zero-copy first: when the mmap is alive (dGPU loads keep it), DMA
    // straight from the page-cache-backed slice — no heap staging copy.
    // Pread fallback preserves UMA behavior (mmap dropped there).
    #[cfg(unix)]
    {
        let mut wt: Option<WeightTensor> = None;
        let mut matched: Option<String> = None;
        for candidate in candidates(name) {
            if let Some((info, data)) = hfq.tensor_data(&candidate) {
                let qt = info.quant_type;
                wt = Some(load_weight_tensor_raw(gpu, qt, data, m, k)?);
                matched = Some(candidate);
                break;
            }
            if let Some((info, buf)) = hfq.tensor_data_pread(&candidate) {
                let qt = info.quant_type;
                wt = Some(load_weight_tensor_raw(gpu, qt, &buf, m, k)?);
                matched = Some(candidate);
                break;
            }
        }
        let mut wt = wt.unwrap_or_else(|| panic!("tensor not found: {name}"));
        // Phase A Stage A — populate awq_scale when the dtype is on
        // the AWQ allow-list (centralized at `DType::supports_awq_sidecar`).
        // The pread call invalidates the prior pread_buf borrow, but
        // the weight bytes have already been uploaded to GPU (owned by
        // `wt.buf`) so the borrow no longer matters.
        if wt.gpu_dtype.supports_awq_sidecar() {
            if let Some(matched_name) = matched.as_deref() {
                wt.awq_scale = load_awq_scale_for(hfq, gpu, matched_name, k)
                    .or_else(|| load_awq_scale_for(hfq, gpu, name, k));
            } else {
                wt.awq_scale = load_awq_scale_for(hfq, gpu, name, k);
            }
        }
        return Ok(wt);
    }
    #[cfg(not(unix))]
    {
        let (info, data, matched_name) = {
            let mut found = None;
            for candidate in candidates(name) {
                if let Some((info, data)) = hfq.tensor_data(&candidate) {
                    found = Some((info, data, candidate));
                    break;
                }
            }
            found.unwrap_or_else(|| panic!("tensor not found: {name}"))
        };
        let mut wt = load_weight_tensor_raw(gpu, info.quant_type, data, m, k)?;
        if wt.gpu_dtype.supports_awq_sidecar() {
            wt.awq_scale = load_awq_scale_for(hfq, gpu, &matched_name, k)
                .or_else(|| load_awq_scale_for(hfq, gpu, name, k));
        }
        Ok(wt)
    }
}

/// REAP keep variant of [`load_weight_tensor`]: gather the tensor's first-axis
/// rows (one row per original expert) down to `keep` BEFORE quant decode, then
/// build the `WeightTensor` from the gathered bytes with `m = keep.len()`.
///
/// Only used for the MoE router (`mlp.gate.weight`, shape `[orig_experts, k]`)
/// under an active keep-map. `gather_rows` is exact for any row-independent
/// quant (every per-expert row is self-contained — its own scale/zero/codebook
/// live in the row), which is true for every quant_type this loader accepts.
/// `keep` MUST equal the compact slot order, and `m` MUST equal `keep.len()`.
///
/// The AWQ sidecar (when present) is indexed by `k` (the input/hidden
/// dimension), shared across all expert rows, so it is loaded UNCHANGED —
/// row selection does not touch it.
fn load_weight_tensor_keep(
    hfq: &HfqFile,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
    keep: &[u32],
) -> HipResult<WeightTensor> {
    debug_assert_eq!(
        m,
        keep.len(),
        "load_weight_tensor_keep: m ({m}) must equal keep.len() ({})",
        keep.len()
    );
    // Resolve via the shared `qwen35_tensor_data_vec` helper (same candidate
    // logic as the non-keep path; it preads + fadvise_dontneeds internally and
    // returns OWNED bytes, so the gather + AWQ-sidecar reads don't fight a
    // borrow). `orig_rows` is the on-disk first-axis length = original expert
    // count. The matched (prefixed) candidate name is resolved separately for
    // the AWQ sidecar lookup via a metadata-only existence check, since the
    // helper doesn't surface which candidate it hit.
    let (info, bytes) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));
    let quant_type = info.quant_type;
    let orig_rows = *info.shape.first().unwrap_or(&0) as usize;
    // Row-gather to the kept set. The on-disk row count is the ORIGINAL expert
    // count (= bytes.len() / rowstride); gather_rows derives it from shape[0].
    let (_new_shape, sub) = hipfire_reap::gather::gather_rows(&[orig_rows], &bytes, keep)
        .map_err(|e| HipError::new(0, &format!("qwen35: router row-gather '{name}': {e}")))?;
    let mut wt = load_weight_tensor_raw(gpu, quant_type, &sub, m, k)?;
    if wt.gpu_dtype.supports_awq_sidecar() {
        // Resolve the matched candidate name (metadata only) so the AWQ sidecar
        // is looked up under the same prefix the weight resolved to; fall back
        // to the bare `name`. Mirrors the non-keep `load_weight_tensor`.
        let matched = qwen35_tensor_name_candidates(name)
            .into_iter()
            .find(|c| hfq.find_tensor_info(c).is_some());
        wt.awq_scale = matched
            .as_deref()
            .and_then(|mn| load_awq_scale_for(hfq, gpu, mn, k))
            .or_else(|| load_awq_scale_for(hfq, gpu, name, k));
    }
    Ok(wt)
}

// ─── ParoQuant AWQ → HFQ4G128 repack ────────────────────────────────────────

/// Repack AWQ-format INT4 weights into HFQ4G128 inline layout.
///
/// AWQ layout (3 separate tensors):
///   qweight: I32 [in_dim, out_dim/8] — 8 nibbles per I32
///   qzeros:  I32 [in_dim/group_size, out_dim/8] — 8 zero-point nibbles per I32
///   scales:  F16 [in_dim/group_size, out_dim] — per-group scales
///
/// HFQ4G128 layout (per output row, one contiguous buffer):
///   For each group of 128 input elements:
///     [f32 scale (4B)][f32 zero (4B)][64B packed nibbles] = 72 bytes
///
/// Returns: Vec<u8> in HFQ4G128 format, ready for gpu.upload_raw.
///
/// SYNC: must match `repack_awq_to_hfq4g128` in
/// `crates/hipfire-runtime/src/hfq.rs`. Duplicated to avoid a cross-crate
/// dependency cycle (hipfire-arch-qwen35 -> hipfire-runtime); keep the two
/// bodies byte-identical when editing.
fn repack_awq_to_hfq4g128(
    qweight: &[u8],    // I32 raw bytes
    qzeros: &[u8],     // I32 raw bytes
    scales: &[u8],     // F16 raw bytes
    out_dim: usize,    // M (output features)
    in_dim: usize,     // K (input features)
    group_size: usize, // 128
) -> Vec<u8> {
    let groups_per_row = in_dim / group_size;
    let bytes_per_row = groups_per_row * 72;
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
            let row_off = m * bytes_per_row + g * 72;

            let scale_f16 = sc[g * out_dim + m];
            let scale_f32 = f16_to_f32(scale_f16);

            let zero_i32 = qz[g * qz_cols + m / 8];
            let zero_nibble = ((zero_i32 >> (AWQ_DEQUANT[m % 8] * 4)) & 0xF) as f32;
            let zero_f32 = -scale_f32 * zero_nibble;

            out[row_off..row_off + 4].copy_from_slice(&scale_f32.to_le_bytes());
            out[row_off + 4..row_off + 8].copy_from_slice(&zero_f32.to_le_bytes());

            let nibble_shift = AWQ_DEQUANT[m % 8] * 4;
            let qw_col = m / 8;
            for i in 0..64 {
                let in_idx0 = g * group_size + i * 2;
                let in_idx1 = in_idx0 + 1;

                let nib0 = ((qw[in_idx0 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;
                let nib1 = ((qw[in_idx1 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;

                // HFQ4G128: lo nibble = even element, hi nibble = odd element
                out[row_off + 8 + i] = nib0 | (nib1 << 4);
            }
        }
    }

    out
}

/// Load a ParoQuant-quantized weight from a SafetensorsSource.
/// Repacks AWQ INT4 → HFQ4G128 and uploads rotation metadata.
pub(crate) fn load_paroquant_weight(
    source: &dyn ModelSource,
    gpu: &Gpu,
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
    let buf = gpu.upload_raw(&hfq_data, &[hfq_data.len()])?;

    // Load rotation metadata
    let (_, pairs_data) = source
        .tensor_data(&pairs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {pairs_name}")))?;
    let (_, theta_data) = source
        .tensor_data(&theta_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {theta_name}")))?;
    let (_, cs_data) = source
        .tensor_data(&cs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {cs_name}")))?;

    let pairs = gpu.upload_raw(pairs_data, &[pairs_data.len()])?;
    let theta = gpu.upload_raw(theta_data, &[theta_data.len()])?;
    let channel_scales = gpu.upload_raw(cs_data, &[cs_data.len()])?;

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

/// Load an FP16 weight and encode it into MQ4G128 byte layout at load time.
/// Used by `paro_load_wt` for LinearAttention `in_proj_a` / `in_proj_b` weights
/// (alpha/beta) when the PARO checkpoint doesn't include them in the calibrated
/// set AND the per-arch/env gating chose the MQ4G128 path.
///
/// At decode time, the weight routes through `gemv_mq4g128_prerotated` which
/// applies FWHT-128 to the activation (via `rotate_x_mq_128_for`) before the
/// inner GEMV. Encoder applies FWHT-128 to weight with the same sign tables,
/// so the two FWHTs orthogonally cancel.

/// Load an FP16 weight tensor from safetensors (for excluded/unquantized layers).
fn load_fp16_weight_from_source(
    source: &dyn ModelSource,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (_, data) = source
        .tensor_data(name)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
    let f32_data: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
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

// ─── ParoQuant MoE expert loading (Option A — per-expert qweight, shared sidecars) ──

/// Repack a single per-expert AWQ projection (gate, up, or down) into HFQ4G128
/// byte rows. Returns the row-major byte buffer (size `out_dim * groups_per_row * 72`).
///
/// Caller is responsible for uploading the buffer to GPU (or concatenating with
/// another projection's rows before upload — gate||up fusion path).
fn paro_repack_moe_projection(
    source: &dyn ModelSource,
    full_prefix: &str, // e.g. "model.language_model.layers.0.mlp.experts.5.gate_proj"
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

/// Upload the per-layer shared PARO rotation sidecars (one tuple for gate||up,
/// one for down). All 256 experts will reference these via non-owning
/// `ParoRotation` aliases.
///
/// Shisa-ai's PARO checkpoint stores these at:
///   `model.language_model.layers.{L}.mlp.experts.{gate_up,down}_weight_{pairs,theta,channel_scales}`
fn paro_load_moe_shared_sidecars(
    source: &dyn ModelSource,
    gpu: &Gpu,
    p: &str, // e.g. "layers.0"
) -> HipResult<MoeParoSidecars> {
    let mp = paro_text_prefix(source)?;
    let base = format!("{mp}.{p}.mlp.experts");
    let load = |name: &str| -> HipResult<GpuTensor> {
        let full = format!("{base}.{name}");
        let (_, data) = source.tensor_data(&full).ok_or_else(|| {
            HipError::new(
                0,
                &format!("ParoQuant MoE shared sidecar not found: {full}"),
            )
        })?;
        gpu.upload_raw(data, &[data.len()])
    };
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant: quant_config required"))?;
    Ok(MoeParoSidecars {
        gate_up_pairs: load("gate_up_weight_pairs")?,
        gate_up_theta: load("gate_up_weight_theta")?,
        gate_up_channel_scales: load("gate_up_weight_channel_scales")?,
        down_pairs: load("down_weight_pairs")?,
        down_theta: load("down_weight_theta")?,
        down_channel_scales: load("down_weight_channel_scales")?,
        krot: qc.krot as u32,
        group_size: qc.group_size,
    })
}

/// Build a non-owning `ParoRotation` whose tensor fields alias `src`'s
/// underlying GPU memory. The returned rotation must NOT outlive `src`;
/// callers store the owning `MoeParoSidecars` in `MoeFfnWeights.paro_shared`
/// to guarantee that.
fn alias_paro_rotation(
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

/// Load the full ParoQuant MoE FFN block for one layer:
///   - dense FP16 router (`mlp.gate.weight [n_exp, hidden]`)
///   - dense FP16 shared-expert scalar gate (`mlp.shared_expert_gate.weight [1, hidden]`)
///   - shared expert (three per-projection PARO tensors: gate, up, down)
///   - 256 routed experts, each with a fused gate||up HFQ4G128 buffer + a down
///     HFQ4G128 buffer, all referencing layer-shared PARO sidecars
fn paro_load_moe_ffn(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    p: &str, // e.g. "layers.0"
    config: &Qwen35Config,
    layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let dim = config.dim;
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant MoE requires quant_config"))?;
    let gs = qc.group_size;
    let kr = qc.krot;

    let mp = paro_text_prefix(source)?;

    // ── Router (FP16 dense in shisa-ai's PARO checkpoint) ──
    // mlp.gate.weight is NOT PARO-quantized — only the expert FFN matmuls are.
    let router = load_fp16_weight_from_source(
        source,
        gpu,
        &format!("{mp}.{p}.mlp.gate.weight"),
        n_exp,
        dim,
    )?;

    // Scalar gate on the shared-expert add — also FP16 dense.
    let shared_expert_gate = load_fp16_weight_from_source(
        source,
        gpu,
        &format!("{mp}.{p}.mlp.shared_expert_gate.weight"),
        1,
        dim,
    )?;

    // ── Shared expert (its own per-projection PARO sidecars, no sharing) ──
    let shared_expert = SharedExpertWeights {
        gate: load_paroquant_weight(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.gate_proj"),
            smi,
            dim,
            gs,
            kr,
        )?,
        up: load_paroquant_weight(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.up_proj"),
            smi,
            dim,
            gs,
            kr,
        )?,
        down: load_paroquant_weight(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.down_proj"),
            dim,
            smi,
            gs,
            kr,
        )?,
    };

    // ── Routed experts ──
    // shisa-ai stores per-expert qweight/qzeros/scales but ONE shared
    // pairs/theta/channel_scales tuple per projection-group (gate||up vs down)
    // for ALL experts in the layer. Upload sidecars once, alias into each
    // expert's WeightTensor.paro.
    let shared = paro_load_moe_shared_sidecars(source, gpu, p)?;

    let groups_per_row_hidden = dim / (gs as usize); // 2048/128 = 16
    let bytes_per_row_hidden = groups_per_row_hidden * 72; // 1152
    let groups_per_row_mi = mi / (gs as usize); // 512/128 = 4
    let bytes_per_row_mi = groups_per_row_mi * 72; // 288

    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        // Per-expert prefixes (full dot-path is constructed inside the helper).
        let gate_prefix = format!("{mp}.{p}.mlp.experts.{x}.gate_proj");
        let up_prefix = format!("{mp}.{p}.mlp.experts.{x}.up_proj");
        let down_prefix = format!("{mp}.{p}.mlp.experts.{x}.down_proj");

        // Fuse gate || up at HFQ4G128 row level: each row is independent
        // (`bytes_per_row` bytes, no cross-row state), so concat works.
        // Final shape: [2*mi, dim], rows [0..mi] = gate, rows [mi..2*mi] = up.
        let gate_bytes = paro_repack_moe_projection(source, &gate_prefix, mi, dim, gs as usize)?;
        let up_bytes = paro_repack_moe_projection(source, &up_prefix, mi, dim, gs as usize)?;
        debug_assert_eq!(gate_bytes.len(), mi * bytes_per_row_hidden);
        debug_assert_eq!(up_bytes.len(), mi * bytes_per_row_hidden);
        let mut gate_up_bytes = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        gate_up_bytes.extend_from_slice(&gate_bytes);
        gate_up_bytes.extend_from_slice(&up_bytes);
        let gate_up_buf = gpu.upload_raw(&gate_up_bytes, &[gate_up_bytes.len()])?;

        let down_bytes = paro_repack_moe_projection(source, &down_prefix, dim, mi, gs as usize)?;
        debug_assert_eq!(down_bytes.len(), dim * bytes_per_row_mi);
        let down_buf = gpu.upload_raw(&down_bytes, &[down_bytes.len()])?;

        let gate_up = WeightTensor {
            buf: gate_up_buf,
            gpu_dtype: DType::ParoQ4G128,
            m: 2 * mi,
            k: dim,
            row_stride: 0,
            paro: Some(alias_paro_rotation(
                &shared.gate_up_pairs,
                &shared.gate_up_theta,
                &shared.gate_up_channel_scales,
                shared.krot,
                shared.group_size,
            )),
            awq_scale: None,
        };
        let down = WeightTensor {
            buf: down_buf,
            gpu_dtype: DType::ParoQ4G128,
            m: dim,
            k: mi,
            row_stride: 0,
            paro: Some(alias_paro_rotation(
                &shared.down_pairs,
                &shared.down_theta,
                &shared.down_channel_scales,
                shared.krot,
                shared.group_size,
            )),
            awq_scale: None,
        };
        experts.push(ExpertWeights { gate_up, down });
    }

    // ── Device-side expert pointer tables (mirrors load_moe_ffn) ──
    let mut gu_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    let mut dn_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    for e in &experts {
        gu_ptrs.push(e.gate_up.buf.buf.as_ptr() as u64);
        dn_ptrs.push(e.down.buf.buf.as_ptr() as u64);
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)?;

    Ok(MoeFfnWeights {
        router,
        experts,
        packed_expert_owners: None,
        shared_expert,
        shared_expert_gate,
        expert_gate_up_ptrs,
        expert_down_ptrs,
        // ParoQuant routed experts use shared per-layer Givens sidecars, not
        // per-expert MQ4 AWQ scales — no MoE-AWQ table.
        expert_down_awq_ptrs: None,
        // Paged/paro layers are uniform-dtype — no per-expert mixed table.
        expert_dtype_tags: None,
        layer_idx,
        expert_shape: None,
        paro_shared: Some(shared),
        global_expert_dtypes: None,
        ep_dummy_buffers: Vec::new(),
        // ParoQuant/paged, not Escha-W2.
        escha: None,
    })
}

// ─── Standard HFQ loading ───────────────────────────────────────────────────

/// Load a tensor as F32 on GPU, handling any quant type by dequanting on CPU.
fn load_any_as_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let (info, data) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));

    let f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        3 => hipfire_runtime::llama::dequantize_q8_0(&data, n),
        14 => {
            // MQ8-G256: [f16 scale][int8 × 256] = 258 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 258;
            let n_groups = data.len() / bytes_per_group;
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale_bits = data[off] as u16 | ((data[off + 1] as u16) << 8);
                let scale = hipfire_runtime::llama::f16_to_f32(scale_bits);
                let start = out.len();
                for i in 0..256 {
                    let q = data[off + 2 + i] as i8;
                    out.push(scale * q as f32);
                }
                // Inverse FWHT to recover original values
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let inv_s = 0.0625;
                for i in 0..256 {
                    group[i] *= inv_s * signs1[i];
                }
            }
            out
        }
        6 | 7 | 13 | 15 => {
            // HFQ4-G256 or G128 or MQ4-G256 or MQ6-G256 — CPU dequant
            // MQ4/MQ6 store rotated weights. For small tensors loaded here,
            // we dequant then inverse-rotate to recover the original values.
            let is_6bit = info.quant_type == 15;
            let group_size: usize =
                if info.quant_type == 6 || info.quant_type == 13 || info.quant_type == 15 {
                    256
                } else {
                    128
                };
            let bytes_per_group = if is_6bit { 200 } else { 8 + group_size / 2 };
            let n_groups = data.len() / bytes_per_group;
            let is_mq = info.quant_type == 13 || info.quant_type == 15;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let (signs1, signs2) = if is_mq {
                (
                    Some(hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256)),
                    Some(hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256)),
                )
            } else {
                (None, None)
            };
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                let start = out.len();
                if is_6bit {
                    for i in (0..group_size).step_by(4) {
                        let bo = off + 8 + (i / 4) * 3;
                        let b0 = data[bo] as u32;
                        let b1 = data[bo + 1] as u32;
                        let b2 = data[bo + 2] as u32;
                        out.push(scale * ((b0 & 0x3F) as f32) + zero);
                        out.push(scale * ((((b0 >> 6) | (b1 << 2)) & 0x3F) as f32) + zero);
                        out.push(scale * ((((b1 >> 4) | (b2 << 4)) & 0x3F) as f32) + zero);
                        out.push(scale * (((b2 >> 2) & 0x3F) as f32) + zero);
                    }
                } else {
                    for i in 0..group_size {
                        let byte_idx = i / 2;
                        let byte_val = data[off + 8 + byte_idx];
                        let nibble = if i % 2 == 0 {
                            byte_val & 0xF
                        } else {
                            byte_val >> 4
                        };
                        out.push(scale * nibble as f32 + zero);
                    }
                }
                // Inverse FWHT for MQ4/MQ6: recover original weight values
                if is_mq && group_size == 256 {
                    let s1 = signs1.as_ref().unwrap();
                    let s2 = signs2.as_ref().unwrap();
                    let group = &mut out[start..start + 256];
                    // Inverse FWHT: signs2 → butterfly → scale → signs1
                    for i in 0..256 {
                        group[i] *= s2[i];
                    }
                    let mut stride = 1;
                    while stride < 256 {
                        let mut j = 0;
                        while j < 256 {
                            for k in 0..stride {
                                let a = group[j + k];
                                let b = group[j + k + stride];
                                group[j + k] = a + b;
                                group[j + k + stride] = a - b;
                            }
                            j += stride * 2;
                        }
                        stride <<= 1;
                    }
                    let scale_inv = 0.0625; // 1/sqrt(256)
                    for i in 0..256 {
                        group[i] *= scale_inv * s1[i];
                    }
                }
            }
            out
        }
        8 => {
            // HFQ6-G256 — CPU dequant: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 200; // 8 + 192
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                // 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
                for i in (0..group_size).step_by(4) {
                    let byte_off = 8 + (i / 4) * 3;
                    let b0 = data[off + byte_off] as u32;
                    let b1 = data[off + byte_off + 1] as u32;
                    let b2 = data[off + byte_off + 2] as u32;
                    let q0 = (b0 & 0x3F) as f32;
                    let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
                    let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
                    let q3 = ((b2 >> 2) & 0x3F) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                }
            }
            out
        }
        11 => {
            // HFQ3-G256: [f32 scale][f32 zero][96B packed 3-bit] = 104 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 104;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                // 8 values per 3 bytes (matching kernel unpack)
                for chunk in 0..32 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        12 => {
            // HFQ3-G128: [f32 scale][f32 zero][48B packed 3-bit] = 56 bytes per 128 weights
            let group_size: usize = 128;
            let bytes_per_group: usize = 56;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                for chunk in 0..16 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        20 => {
            // MQ3-G256-Lloyd (qt 20, 112 B/group): 8 fp16 codebook entries + 3-bit
            // indices (cross-byte, 32 chunks × 3 bytes × 8 weights). Decode is
            // direct lookup `cb[idx]` then inverse FWHT for CPU consumers.
            let group_size: usize = 256;
            let bytes_per_group: usize = 112;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 8];
                for k in 0..8 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = hipfire_runtime::llama::f16_to_f32(bits);
                }
                let start = out.len();
                for chunk in 0..32 {
                    let bo = off + 16 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as usize;
                    let q1 = ((b0 >> 3) & 7) as usize;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as usize;
                    let q3 = ((b1 >> 1) & 7) as usize;
                    let q4 = ((b1 >> 4) & 7) as usize;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as usize;
                    let q6 = ((b2 >> 2) & 7) as usize;
                    let q7 = ((b2 >> 5) & 7) as usize;
                    out.push(cb[q0]);
                    out.push(cb[q1]);
                    out.push(cb[q2]);
                    out.push(cb[q3]);
                    out.push(cb[q4]);
                    out.push(cb[q5]);
                    out.push(cb[q6]);
                    out.push(cb[q7]);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        19 => {
            // MQ2-G256-Lloyd (qt 19, 72 B/group): 4 fp16 codebook entries + 2-bit indices.
            // Decode is direct lookup `cb[idx]`, then inverse FWHT to recover original
            // pre-rotation values for CPU consumers (DeltaNet conv1d).
            let group_size: usize = 256;
            let bytes_per_group: usize = 72;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 4];
                for k in 0..4 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = hipfire_runtime::llama::f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..64 {
                    let byte_val = data[off + 8 + i] as usize;
                    out.push(cb[byte_val & 3]);
                    out.push(cb[(byte_val >> 2) & 3]);
                    out.push(cb[(byte_val >> 4) & 3]);
                    out.push(cb[(byte_val >> 6) & 3]);
                }
                // Inverse FWHT to recover pre-rotation weights — same butterfly as the
                // MQ3/MQ2 arm below.
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        30 => {
            // MQ4-G256-Lloyd (qt 30, 160 B/group): 16 fp16 codebook entries (bytes [0..32))
            // + 4-bit packed indices (bytes [32..160), low nibble = idx[2i], high = idx[2i+1]).
            // Decode is direct lookup `cb[idx]` then inverse FWHT for CPU consumers.
            // Renumbered from qt 21 → 30 to avoid HFP4G32=21 collision.
            let group_size: usize = 256;
            let bytes_per_group: usize = 160;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 16];
                for k in 0..16 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = hipfire_runtime::llama::f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..128 {
                    let byte_val = data[off + 32 + i] as usize;
                    out.push(cb[byte_val & 0xF]);
                    out.push(cb[(byte_val >> 4) & 0xF]);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        17 | 18 => {
            // MQ3-G256 (qt 17, 104 B/group, 3-bit) or MQ2-G256 (qt 18, 72 B/group, 2-bit).
            // Both store FWHT-rotated weights — dequant then inverse-rotate to recover
            // original values for CPU consumers (e.g., DeltaNet conv1d).
            let is_mq3 = info.quant_type == 17;
            let group_size: usize = 256;
            let bytes_per_group: usize = if is_mq3 { 104 } else { 72 };
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::llama::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::llama::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                let start = out.len();
                if is_mq3 {
                    // 8 values per 3 bytes (matches gemv_hfq3g256.hip unpack).
                    for chunk in 0..32 {
                        let bo = off + 8 + chunk * 3;
                        let b0 = data[bo] as u32;
                        let b1 = data[bo + 1] as u32;
                        let b2 = data[bo + 2] as u32;
                        let q0 = (b0 & 7) as f32;
                        let q1 = ((b0 >> 3) & 7) as f32;
                        let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                        let q3 = ((b1 >> 1) & 7) as f32;
                        let q4 = ((b1 >> 4) & 7) as f32;
                        let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                        let q6 = ((b2 >> 2) & 7) as f32;
                        let q7 = ((b2 >> 5) & 7) as f32;
                        out.push(scale * q0 + zero);
                        out.push(scale * q1 + zero);
                        out.push(scale * q2 + zero);
                        out.push(scale * q3 + zero);
                        out.push(scale * q4 + zero);
                        out.push(scale * q5 + zero);
                        out.push(scale * q6 + zero);
                        out.push(scale * q7 + zero);
                    }
                } else {
                    // MQ2: 4 values per byte (matches gemv_hfq2g256.hip unpack).
                    for i in 0..64 {
                        let byte_val = data[off + 8 + i] as u32;
                        out.push(scale * ((byte_val & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 2) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 4) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 6) & 3) as f32) + zero);
                    }
                }
                // Inverse FWHT: recover original (pre-rotation) weight values.
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625; // 1/sqrt(256)
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        32 => {
            // MFP4G32Lloyd (qt 32): [32-B fp16 codebook prefix][M rows].
            // Each row = 16-B header + (K/32)*17 B blocks (UE8M0 + nibbles).
            // Recon: value = row_scale_a * 2^(block_e-127) * cb[nibble].
            // Returns rotated-domain f32 (weights stored pre-FWHT-rotated).
            let row_bytes = 16 + 17 * (n / 32);
            let m_rows = if row_bytes > 0 {
                (data.len().saturating_sub(32)) / row_bytes
            } else {
                0
            };
            let mut cb = [0.0f32; 16];
            for i in 0..16 {
                let bits = u16::from_le_bytes([data[2 * i], data[2 * i + 1]]);
                cb[i] = hipfire_runtime::llama::f16_to_f32(bits);
            }
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = 32 + r * (16 + n_blocks * 17);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * 17;
                    let block_e = data[po] as i32;
                    let scale = row_scale_a * ((block_e - 127) as f32).exp2();
                    for i in 0..16 {
                        let byte = data[po + 1 + i];
                        out[r * k_row + b * 32 + 2 * i] = scale * cb[(byte & 0x0F) as usize];
                        out[r * k_row + b * 32 + 2 * i + 1] =
                            scale * cb[((byte >> 4) & 0x0F) as usize];
                    }
                }
            }
            out
        }
        33 => {
            // MFP4G32P (qt 33): mfp4 rows (NO prefix) with E4M3 (FP8) per-block scale.
            // Recon: value = row_scale_a * e4m3_decode(scale_byte) * E2M1_LUT[nibble].
            // Returns rotated-domain f32 (weights stored pre-FWHT-rotated).
            const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            #[inline]
            fn e2m1(n: u8) -> f32 {
                let m = E2M1_MAG[(n & 0x7) as usize];
                if (n & 0x8) != 0 {
                    -m
                } else {
                    m
                }
            }
            // E4M3 (unsigned scale, bias 7, 3 mantissa) — bit-identical to the
            // quantizer `e4m3_scale_decode` and the gfx942 kernel decode.
            #[inline]
            fn e4m3(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            let row_bytes = 16 + 17 * (n / 32);
            let m_rows = if row_bytes > 0 {
                data.len() / row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = r * (16 + n_blocks * 17);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * 17;
                    let scale = row_scale_a * e4m3(data[po]);
                    for i in 0..16 {
                        let byte = data[po + 1 + i];
                        out[r * k_row + b * 32 + 2 * i] = scale * e2m1(byte & 0x0F);
                        out[r * k_row + b * 32 + 2 * i + 1] = scale * e2m1((byte >> 4) & 0x0F);
                    }
                }
            }
            out
        }
        34 => {
            // MFP4G32E8 (qt 34): mfp4+P container, NO prefix, with E8-lattice codewords.
            // Identical framing to qt 33 (E4M3 block scale, row_scale f16); per block
            // decode 4 E8 codewords instead of 32 E2M1 nibbles.
            // E8 decode — bit-identical to e8.rs::decode_index + * QUANT_STEP.
            #[inline]
            fn e4m3_e8(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            #[inline]
            fn e8_decode(idx: u32, coord: usize) -> f32 {
                // Decode a single coord of an E8 lattice point from a u32 index.
                // Matches kernel e8_decode_index exactly.
                let coset = (idx >> 31) & 1;
                let e: u32;
                if coord < 7 {
                    e = (idx >> (4 * coord as u32)) & 0xF;
                } else {
                    // coord == 7: recover from parity
                    let mut sl: u32 = 0;
                    for i in 0..7 {
                        sl += (idx >> (4 * i)) & 0xF;
                    }
                    let e7h = (idx >> 28) & 0x7;
                    let p7 = e7h << 1;
                    let lsb = (sl + p7) & 1;
                    e = p7 | lsb;
                }
                let c = (e as i32 - 7) as f32;
                if coset == 1 {
                    c + 0.5
                } else {
                    c
                }
            }
            const QUANT_STEP: f32 = 0.88;
            let row_bytes = 16 + 17 * (n / 32);
            let m_rows = if row_bytes > 0 {
                data.len() / row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = r * (16 + n_blocks * 17);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * 17;
                    let scale = row_scale_a * e4m3_e8(data[po]) * QUANT_STEP;
                    for g in 0..4usize {
                        let idx = u32::from_le_bytes([
                            data[po + 1 + g * 4],
                            data[po + 2 + g * 4],
                            data[po + 3 + g * 4],
                            data[po + 4 + g * 4],
                        ]);
                        for i in 0..8usize {
                            out[r * k_row + b * 32 + g * 8 + i] = scale * e8_decode(idx, i);
                        }
                    }
                }
            }
            out
        }
        35 => {
            // MFP4G32E8SOA (qt 35): same E8 data as qt 34 but in SoA layout.
            // Per-row: [16B hdr] + [n_blocks bytes E4M3 scales, pad16] + [n_blocks*16 bytes codewords].
            #[inline]
            fn e4m3_e8_soa(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            #[inline]
            fn e8_decode_soa(idx: u32, coord: usize) -> f32 {
                let coset = (idx >> 31) & 1;
                let e: u32;
                if coord < 7 {
                    e = (idx >> (4 * coord as u32)) & 0xF;
                } else {
                    let mut sl: u32 = 0;
                    for i in 0..7 {
                        sl += (idx >> (4 * i)) & 0xF;
                    }
                    let e7h = (idx >> 28) & 0x7;
                    let p7 = e7h << 1;
                    let lsb = (sl + p7) & 1;
                    e = p7 | lsb;
                }
                let c = (e as i32 - 7) as f32;
                if coset == 1 {
                    c + 0.5
                } else {
                    c
                }
            }
            const QUANT_STEP_SOA: f32 = 0.88;
            // Decode assuming n = k_row; figure out m_rows from total bytes.
            // n_blocks = n/32; scale_padded = ceil(n_blocks/16)*16
            let n_blocks = n / 32;
            let scale_padded = ((n_blocks + 15) >> 4) << 4;
            let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
            let m_rows = if soa_row_bytes > 0 {
                data.len() / soa_row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks2 = k_row / 32;
            let scale_padded2 = ((n_blocks2 + 15) >> 4) << 4;
            let row_bytes2 = 16 + scale_padded2 + n_blocks2 * 16;
            for r in 0..m_rows {
                let base = r * row_bytes2;
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                let scale_arr = &data[base + 16..base + 16 + n_blocks2];
                let cw_arr =
                    &data[base + 16 + scale_padded2..base + 16 + scale_padded2 + n_blocks2 * 16];
                for b in 0..n_blocks2 {
                    let scale = row_scale_a * e4m3_e8_soa(scale_arr[b]) * QUANT_STEP_SOA;
                    for g in 0..4usize {
                        let co = b * 16 + g * 4;
                        let idx = u32::from_le_bytes([
                            cw_arr[co],
                            cw_arr[co + 1],
                            cw_arr[co + 2],
                            cw_arr[co + 3],
                        ]);
                        for i in 0..8usize {
                            out[r * k_row + b * 32 + g * 8 + i] = scale * e8_decode_soa(idx, i);
                        }
                    }
                }
            }
            out
        }
        36 => {
            // MFP3G32E8 (qt 36): mfp4-E8 frame with 3-bit lattice, 13 B/blk, center 4.
            // Decode mirrors kernel mfp3_decode_index: nibbles 3b, coset bit 23, e7_high 2b@21.
            #[inline]
            fn e4m3_mfp3(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            const MFP3_BLOCK: usize = 13; // 1 + 4*3
            const MFP3_STEP: f32 = 1.8; // MSE-tuned; matches QUANT_STEP_MFP3 in e8.rs + kernels
            let row_bytes = 16 + MFP3_BLOCK * (n / 32);
            let m_rows = if row_bytes > 0 {
                data.len() / row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = r * (16 + n_blocks * MFP3_BLOCK);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * MFP3_BLOCK;
                    let scale = row_scale_a * e4m3_mfp3(data[po]) * MFP3_STEP;
                    for g in 0..4usize {
                        let cw_off = po + 1 + g * 3;
                        // 3-byte narrow read (mfp3 codeword is only 24 bits)
                        let idx: u32 = (data[cw_off] as u32)
                            | ((data[cw_off + 1] as u32) << 8)
                            | ((data[cw_off + 2] as u32) << 16);
                        // mfp3_decode_index: 3-bit nibbles, center 4, coset bit 23, e7_high @21 (2b)
                        let coset = (idx >> 23) & 1;
                        let mut e = [0u32; 8];
                        let mut sl: u32 = 0;
                        for i in 0..7 {
                            e[i] = (idx >> (3 * i as u32)) & 0x7;
                            sl += e[i];
                        }
                        let e7_high = (idx >> 21) & 0x3;
                        let p7 = e7_high << 1;
                        e[7] = p7 | ((sl + p7) & 1);
                        for i in 0..8usize {
                            let c = (e[i] as i32 - 4) as f32;
                            let coord = if coset == 1 { c + 0.5 } else { c };
                            out[r * k_row + b * 32 + g * 8 + i] = scale * coord;
                        }
                    }
                }
            }
            out
        }
        37 => {
            // MFP2G32E8 (qt 37): mfp4-E8 frame with 2-bit lattice, 9 B/blk, center 2.
            // Decode mirrors kernel mfp2_decode_index: nibbles 2b, coset bit 15, e7_high 1b@14.
            #[inline]
            fn e4m3_mfp2(b: u8) -> f32 {
                let exp = ((b >> 3) & 0xf) as i32;
                let mant = (b & 0x7) as u32;
                if exp == 0 {
                    return (2.0f32).powi(-6) * (mant as f32) / 8.0;
                }
                if exp == 0xf && mant == 7 {
                    return 448.0;
                }
                (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            }
            const MFP2_BLOCK: usize = 9; // 1 + 4*2
            const MFP2_STEP: f32 = 3.8; // MSE-tuned; matches QUANT_STEP_MFP2 in e8.rs + kernels
            let row_bytes = 16 + MFP2_BLOCK * (n / 32);
            let m_rows = if row_bytes > 0 {
                data.len() / row_bytes
            } else {
                0
            };
            let mut out = vec![0.0f32; n];
            let k_row = if m_rows > 0 { n / m_rows } else { n };
            let k_row = k_row.max(1);
            let n_blocks = k_row / 32;
            for r in 0..m_rows {
                let base = r * (16 + n_blocks * MFP2_BLOCK);
                let row_scale_a = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([
                    data[base],
                    data[base + 1],
                ]));
                for b in 0..n_blocks {
                    let po = base + 16 + b * MFP2_BLOCK;
                    let scale = row_scale_a * e4m3_mfp2(data[po]) * MFP2_STEP;
                    for g in 0..4usize {
                        let cw_off = po + 1 + g * 2;
                        // 2-byte narrow read (mfp2 codeword is only 16 bits)
                        let idx: u32 = (data[cw_off] as u32) | ((data[cw_off + 1] as u32) << 8);
                        // mfp2_decode_index: 2-bit nibbles, center 2, coset bit 15, e7_high @14 (1b)
                        let coset = (idx >> 15) & 1;
                        let mut e = [0u32; 8];
                        let mut sl: u32 = 0;
                        for i in 0..7 {
                            e[i] = (idx >> (2 * i as u32)) & 0x3;
                            sl += e[i];
                        }
                        let e7_high = (idx >> 14) & 0x1;
                        let p7 = e7_high << 1;
                        e[7] = p7 | ((sl + p7) & 1);
                        for i in 0..8usize {
                            let c = (e[i] as i32 - 2) as f32;
                            let coord = if coset == 1 { c + 0.5 } else { c };
                            out[r * k_row + b * 32 + g * 8 + i] = scale * coord;
                        }
                    }
                }
            }
            out
        }
        _ => panic!("unsupported quant_type {} for {name}", info.quant_type),
    };
    gpu.upload_f32(&f32_data[..n], &[n])
}

/// Alias for load_any_as_f32.
fn load_raw_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    load_any_as_f32(hfq, gpu, name, n)
}

/// Loud model-load diagnostic for RDNA2 (gfx1030/1031/1032). Scans the model's
/// per-tensor `quant_type` bytes and, if running on RDNA2 with any RDNA3+-only
/// dtype present, prints a clear warning listing the offending formats. RDNA2 is
/// wave32 (so the WMMA-free MQ3-Lloyd scalar kernels now resolve — W4 fix), but
/// MQ5/MQ6/HFQ6 (HasMmq, RDNA3/4 only), the WMMA-only HFP4G32-fused prefill path,
/// and the E8-lattice formats have NO validated RDNA2 kernel and are best-effort.
///
/// Additive and arch-gated to RDNA2 — no effect on RDNA3/4/CDNA loads.
fn warn_rdna2_unvalidated_dtypes(hfq: &HfqFile, gpu: &Gpu) {
    if !gpu.arch_caps.is_rdna2() {
        return;
    }
    // (quant_type byte, human label) for dtypes with NO validated RDNA2 kernel.
    //   8=HFQ6G256, 15=MQ6G256, 31=MQ5G256  → HasMmq (RDNA3/4 only)
    //   24=MFP4G32, 33=MFP4G32P             → HFP4G32-fused (WMMA-only prefill)
    //   34/35=MFP4G32E8/SOA, 36=MFP3G32E8, 37=MFP2G32E8 → E8 lattice (RDNA3/4)
    const RDNA3PLUS_ONLY: &[(u8, &str)] = &[
        (8, "HFQ6G256"),
        (15, "MQ6G256"),
        (31, "MQ5G256"),
        (24, "MFP4G32 (HFP4G32-fused)"),
        (33, "MFP4G32P (HFP4G32-fused)"),
        (34, "MFP4G32E8"),
        (35, "MFP4G32E8SOA"),
        (36, "MFP3G32E8"),
        (37, "MFP2G32E8"),
    ];
    let mut present: Vec<&str> = RDNA3PLUS_ONLY
        .iter()
        .filter(|(qt, _)| hfq.tensors().iter().any(|t| t.quant_type == *qt))
        .map(|(_, label)| *label)
        .collect();
    present.dedup();
    if present.is_empty() {
        return;
    }
    eprintln!(
        "  ⚠️  RDNA2 ({}): this model contains RDNA3+-only quant formats: {}.",
        gpu.arch,
        present.join(", ")
    );
    eprintln!(
        "      RDNA2 (gfx1030): uniform .mq4 is the validated SKU; this model's \
         MQ3-Lloyd/MQ6/E8 content is best-effort and UNVALIDATED on RDNA2."
    );
}

// TODO(transformer-extraction): the overall `load_weights` orchestration
// here (drop_mmap → embedding+tied-lm_head → norm → per-layer loop) is
// the model the Qwen2 loader at
// `hipfire-arch-qwen2::qwen2::load_weights` follows. The tied-embedding
// re-upload pattern (re-reading `embed_tokens.weight` to construct a
// second GpuTensor for the lm_head) is duplicated in both crates
// because GpuTensor is not Clone. Consolidation PR should either add
// `GpuTensor::shallow_clone()` or switch to `Arc<GpuTensor>` so tied
// embeddings stop costing 2× the embedding VRAM.

/// Attach the lm_head / tied-embed AWQ sidecar when the output dtype supports it.
/// Byte-identical no-op on current files. MUST be called AFTER `output.gpu_dtype`
/// is set (the gate reads it). See docs/plans/awq_fix_claude.md.
fn attach_lm_head_awq_sidecar(hfq: &HfqFile, gpu: &Gpu, output: &mut WeightTensor, k: usize) {
    if output.gpu_dtype.supports_awq_sidecar() {
        output.awq_scale = load_awq_scale_for(hfq, gpu, "lm_head.weight", k)
            .or_else(|| load_awq_scale_for(hfq, gpu, "model.language_model.lm_head.weight", k))
            .or_else(|| {
                load_awq_scale_for(hfq, gpu, "model.language_model.embed_tokens.weight", k)
            });
        eprintln!(
            "  lm_head AWQ sidecar: {}",
            if output.awq_scale.is_some() {
                "attached"
            } else {
                "absent (no-op)"
            }
        );
    }
}

// ── Layout (re-exported from runtime) ─────────────────────────────────────

pub use hipfire_runtime::model_load::Layout;

// ── load_weights (thin assembler over runtime orchestrator) ───────────────

/// Drive a qwen35 `WeightSource` over the device slice (runtime orchestrator),
/// then assemble `Qwen35Weights`. `pager` is always `None` here; paged-experts
/// wiring is unchanged and set by the caller post-load.
pub fn load_weights(
    source: &mut (impl WeightSource<Layer = LayerWeights>),
    devices: &mut [Gpu],
    layout: &Layout,
) -> HipResult<Qwen35Weights> {
    use std::sync::atomic::Ordering;
    use std::time::Instant;
    let t_sweep = Instant::now();
    let LoadedWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
    } = rt_load_weights(source, devices, layout)?;
    eprintln!(
        "  weight sweep: {} ms (packed-expert host-read {} ms, H2D {} ms)",
        t_sweep.elapsed().as_millis(),
        PACKED_READ_MS.load(Ordering::Relaxed),
        PACKED_UPLOAD_MS.load(Ordering::Relaxed),
    );
    Ok(Qwen35Weights {
        token_embd,
        embd_format,
        output_norm,
        output,
        moe_has_mq6: layers_have_mq6_moe(&layers),
        layers,
        pager: None,
        lm_head_aliases_embd,
        ep_shard: None,
    })
}

// ── HfqSource ─────────────────────────────────────────────────────────────

pub struct HfqSource<'a> {
    hfq: &'a mut HfqFile,
    c: &'a Qwen35Config,
}
impl<'a> HfqSource<'a> {
    pub fn new(hfq: &'a mut HfqFile, c: &'a Qwen35Config) -> Self {
        Self { hfq, c }
    }
}
impl WeightSource for HfqSource<'_> {
    type Layer = LayerWeights;

    fn n_layers(&self) -> usize {
        self.c.n_layers
    }
    fn prepare(&mut self, n_devices: usize) -> HipResult<()> {
        // Keep the mmap alive on discrete GPUs (the carrier cleared
        // `evict_page_cache` there): weight uploads DMA straight out of
        // page-cache pages with no heap staging copy — measured 11–16 GB/s
        // end-to-end vs ~4 GB/s through a pread staging buffer.
        // UMA keeps the drop — evict=true is exactly the carrier's UMA
        // signal — so cached model pages can't starve hipMalloc of RAM.
        //
        // NOTE: do NOT madvise-populate the mapping here. A blocking
        // MAP_POPULATE measured identical totals (it relocates the same
        // soft-fault work out of the upload loop into one serial stall).
        #[cfg(unix)]
        if n_devices == 1 && self.hfq.evicts_page_cache() {
            self.hfq.drop_mmap();
        }
        let _ = n_devices;
        Ok(())
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        // W4: loud RDNA2 diagnostic — flag RDNA3+-only quant formats (MQ5/MQ6/HFQ6/
        // HFP4G32-fused/E8) that have no validated RDNA2 kernel. No-op off RDNA2.
        // Fired once per load here (the first read with both hfq + gpu in scope,
        // after master's loader refactor split source from devices).
        warn_rdna2_unvalidated_dtypes(self.hfq, gpu);
        let c = self.c;
        eprintln!("  loading token_embd...");
        if c.is_vl_text {
            eprintln!(
                "  qwen3.5-vl text wrapper: mrope_interleaved={} mrope_section={:?}",
                c.mrope_interleaved, c.mrope_section
            );
        }
        let (embd_meta, embd_data) = qwen35_tensor_data_cow(self.hfq, "embed_tokens.weight")
            .expect("embed_tokens not found");
        let out = load_embedding(gpu, embd_meta.quant_type, &embd_data, c.vocab_size, c.dim)?;
        drop(embd_data);
        Ok(out)
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor> {
        eprintln!("  loading output_norm...");
        load_norm_weight(self.hfq, gpu, "norm.weight", &[self.c.dim])
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        let c = self.c;
        let hfq = &*self.hfq;
        let has_separate = qwen35_tensor_name_candidates("lm_head.weight")
            .iter()
            .any(|n| hfq.find_tensor_info(n).is_some());
        let (mut output, aliases) = resolve_lm_head(
            gpu,
            has_separate,
            can_alias,
            embd,
            embd_fmt,
            c.vocab_size,
            c.dim,
            |gpu| {
                let (lm_info, lm_data) =
                    qwen35_tensor_data_cow(hfq, "lm_head.weight").expect("lm_head present");
                load_weight_tensor_raw(gpu, lm_info.quant_type, &lm_data, c.vocab_size, c.dim)
            },
            |gpu| {
                let (embd_meta, embd_data) = qwen35_tensor_data_cow(hfq, "embed_tokens.weight")
                    .expect("embed_tokens not found");
                dequant_weight_raw(gpu, embd_meta.quant_type, &embd_data, c.vocab_size, c.dim)
            },
        )?;
        attach_lm_head_awq_sidecar(self.hfq, gpu, &mut output, c.dim);
        Ok((output, aliases))
    }

    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<LayerWeights> {
        let c = self.c;
        let is_moe = c.num_experts > 0;
        eprintln!(
            "  loading layer {layer_idx}/{} ({:?}{})...",
            c.n_layers,
            c.layer_types[layer_idx],
            if is_moe { " + MoE" } else { "" }
        );
        let p = format!("layers.{layer_idx}");
        let page = self.hfq.layer_data_range(&p);
        let lw = load_layer_into(self.hfq, c, layer_idx, &p, gpu)?;
        if let Some((start, end)) = page {
            self.hfq.drop_pages_range(start, end - start);
        }
        Ok(lw)
    }
}

// ── ParoSource ────────────────────────────────────────────────────────────

pub struct ParoSource<'a> {
    source: &'a dyn ModelSource,
    mp: &'static str,
    c: &'a Qwen35Config,
}
impl<'a> ParoSource<'a> {
    pub fn new(source: &'a dyn ModelSource, c: &'a Qwen35Config) -> HipResult<Self> {
        source
            .quant_config()
            .ok_or_else(|| HipError::new(0, "ParoQuant model must have quantization_config"))?;
        let mp = paro_text_prefix(source)?;
        Ok(Self { source, mp, c })
    }
    fn read_f16_as_f32(&self, name: &str) -> HipResult<Vec<f32>> {
        let (_, data) = self
            .source
            .tensor_data(name)
            .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
        Ok(hipfire_runtime::weight_backend::f16_bytes_to_f32(&data))
    }
}
impl WeightSource for ParoSource<'_> {
    type Layer = LayerWeights;

    fn n_layers(&self) -> usize {
        self.c.n_layers
    }

    fn prepare(&mut self, n_devices: usize) -> HipResult<()> {
        if n_devices > 1 {
            return Err(HipError::new(
                0,
                "ParoQuant multi-GPU loading is not supported (HFQ-only)",
            ));
        }
        Ok(())
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        eprintln!("  loading token_embd (ParoQuant)...");
        let f32_embd = self.read_f16_as_f32(&format!("{}.embed_tokens.weight", self.mp))?;
        let token_embd = gpu.upload_f32(&f32_embd, &[self.c.vocab_size, self.c.dim])?;
        Ok((token_embd, EmbeddingFormat::F32))
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor> {
        eprintln!("  loading output_norm...");
        paro_load_norm(
            self.source,
            gpu,
            "norm.weight",
            &[self.c.dim],
            QWEN35_NORM_BIAS,
        )
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        let mp = self.mp;
        let c = self.c;
        let source = self.source;
        let has_separate = source.tensor_data("lm_head.weight").is_some();
        resolve_lm_head(
            gpu,
            has_separate,
            can_alias,
            embd,
            embd_fmt,
            c.vocab_size,
            c.dim,
            |gpu| {
                let (_, f16) = source
                    .tensor_data("lm_head.weight")
                    .ok_or_else(|| HipError::new(0, "PARO tensor not found: lm_head.weight"))?;
                reupload_f16_as_f32(gpu, &f16, c.vocab_size, c.dim)
            },
            |gpu| {
                let embd_name = format!("{mp}.embed_tokens.weight");
                let (_, f16) = source.tensor_data(&embd_name).ok_or_else(|| {
                    HipError::new(0, &format!("PARO tensor not found: {embd_name}"))
                })?;
                reupload_f16_as_f32(gpu, &f16, c.vocab_size, c.dim)
            },
        )
    }

    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<LayerWeights> {
        let c = self.c;
        eprintln!(
            "  loading layer {layer_idx}/{} ({:?}, ParoQuant)...",
            c.n_layers, c.layer_types[layer_idx]
        );
        let mut b = qwen35_paro_backend(self.source, gpu, self.mp, layer_idx);
        let moe = |bk: &mut ParoBackend, cfg: &Qwen35Config, li: usize| {
            crate::paro_moe::paro_load_moe_ffn(
                bk.source,
                bk.gpu,
                &format!("layers.{li}"),
                cfg,
                li as u16,
            )
        };
        crate::layer_driver::load_layer(&mut b, c, layer_idx, moe)
    }
}

/// Construct an `HfqBackend` with qwen35's defaults baked in: `QWEN35_NORM_BIAS`,
/// the qwen35 tensor-name resolver, and the standard pread+awq weight reader.
fn qwen35_hfq_backend<'a>(hfq: &'a HfqFile, gpu: &'a mut Gpu, layer: usize) -> HfqBackend<'a> {
    HfqBackend {
        hfq,
        gpu,
        norm_bias: QWEN35_NORM_BIAS,
        candidates: qwen35_tensor_name_candidates,
        read_proj: load_weight_tensor,
        layer,
    }
}

/// Construct a `ParoBackend` with qwen35's `norm_bias` baked in. `mp` is the
/// text-tower prefix from `paro_text_prefix`.
fn qwen35_paro_backend<'a>(
    source: &'a dyn ModelSource,
    gpu: &'a mut Gpu,
    mp: &'static str,
    layer: usize,
) -> ParoBackend<'a> {
    ParoBackend {
        source,
        gpu,
        mp,
        layer,
        norm_bias: QWEN35_NORM_BIAS,
    }
}

/// Build one layer's `LayerWeights` on `gpu`. Extracted for `load_weights_multi`
/// so the multi-GPU loader can route each layer to its band-owning device
/// without duplicating the tensor-name table. Master's `load_weights` keeps
/// its inline body — does not consume this helper.
fn load_layer_into(
    hfq: &HfqFile,
    config: &Qwen35Config,
    layer_idx: usize,
    p: &str,
    gpu: &mut Gpu,
) -> HipResult<LayerWeights> {
    debug_assert_eq!(p, &format!("layers.{layer_idx}"));
    let mut b = qwen35_hfq_backend(hfq, gpu, layer_idx);
    let moe = |bk: &mut HfqBackend, cfg: &Qwen35Config, li: usize| {
        load_moe_ffn(bk.hfq, bk.gpu, &format!("layers.{li}"), cfg, li as u16)
    };
    crate::layer_driver::load_layer(&mut b, config, layer_idx, moe)
}

#[derive(Clone, Copy)]
enum DenseTpSlice<'a> {
    Rows(usize, usize),
    RowsMulti(&'a [(usize, usize)]),
    Cols(usize, usize),
}

const MQ4_G256_QT: u8 = 13;
const MQ4V2_G256_QT: u8 = 44;

#[inline]
fn is_dense_proj_qt(qt: u8) -> bool {
    matches!(qt, MQ4_G256_QT | MQ4V2_G256_QT)
}

fn find_qwen35_tensor<'a>(hfq: &'a HfqFile, bare: &str) -> Option<(&'a HfqTensorInfo, String)> {
    for cand in qwen35_tensor_name_candidates(bare) {
        if let Some(info) = hfq.find_tensor_info(&cand) {
            return Some((info, cand));
        }
    }
    None
}

fn expected_mq4_bytes(m: usize, k: usize) -> Option<usize> {
    if k % 256 != 0 {
        return None;
    }
    let gpr = k / 256;
    m.checked_mul(gpr)?.checked_mul(136)
}

fn validate_mq4_proj_info(
    info: &HfqTensorInfo,
    m: usize,
    k: usize,
    name: &str,
) -> Result<(), String> {
    if !is_dense_proj_qt(info.quant_type) {
        return Err(format!(
            "{name}: dense TP requires qt13 or qt44, got qt={}",
            info.quant_type
        ));
    }
    if info.group_size != 256 {
        return Err(format!(
            "{name}: group_size must be 256, got {}",
            info.group_size
        ));
    }
    if info.shape != vec![m as u32, k as u32] {
        return Err(format!(
            "{name}: shape mismatch {:?} vs [{m} {k}]",
            info.shape
        ));
    }
    let expected =
        expected_mq4_bytes(m, k).ok_or_else(|| format!("{name}: K={k} not multiple of 256"))?;
    if info.data_size != expected {
        return Err(format!(
            "{name}: blob length mismatch expected {expected}, got {}",
            info.data_size
        ));
    }
    Ok(())
}

fn validate_norm_info(info: &HfqTensorInfo, n: usize, name: &str) -> Result<(), String> {
    if info.shape != vec![n as u32] {
        return Err(format!("{name}: shape mismatch {:?} vs [{n}]", info.shape));
    }
    let product = n;
    let expected = match info.quant_type {
        1 => product.checked_mul(2),
        2 => product.checked_mul(4),
        16 => product.checked_mul(2),
        _ => None,
    };
    if let Some(exp) = expected {
        if info.data_size != exp {
            return Err(format!(
                "{name}: blob length mismatch expected {exp}, got {}",
                info.data_size
            ));
        }
    } else {
        return Err(format!(
            "{name}: unsupported norm quant_type {}",
            info.quant_type
        ));
    }
    Ok(())
}

fn validate_raw_f32_info(info: &HfqTensorInfo, n: usize, name: &str) -> Result<(), String> {
    // Canonical raw_f32 is flattened: shape product must equal n (e.g. conv1d
    // [10240,1,4] -> 40960). Retain exact byte and dtype checks.
    let product: usize = info
        .shape
        .iter()
        .try_fold(1usize, |a, &b| {
            a.checked_mul(b as usize)
                .ok_or_else(|| format!("{name}: shape product overflow {:?}", info.shape))
        })
        .map_err(|e| e)?;
    if product != n {
        return Err(format!(
            "{name}: shape product mismatch {:?} (product {product}) vs n={n}",
            info.shape
        ));
    }
    // group_size check per dtype (qt3 is Q8 group 32)
    match info.quant_type {
        3 => {
            if info.group_size != 32 {
                return Err(format!(
                    "{name}: Q8 group_size must be 32, got {}",
                    info.group_size
                ));
            }
        }
        1 | 2 | 16 => {
            if info.group_size != 0 {
                return Err(format!(
                    "{name}: group_size must be 0, got {}",
                    info.group_size
                ));
            }
        }
        _ => {}
    }
    let expected = match info.quant_type {
        1 => n.checked_mul(2),
        2 => n.checked_mul(4),
        16 => n.checked_mul(2),
        3 => {
            if n % 32 != 0 {
                return Err(format!("{name}: Q8 n={n} not multiple of 32"));
            }
            (n / 32).checked_mul(34)
        }
        _ => None,
    };
    if let Some(exp) = expected {
        if info.data_size != exp {
            return Err(format!(
                "{name}: blob length mismatch expected {exp}, got {}",
                info.data_size
            ));
        }
    } else {
        return Err(format!(
            "{name}: unsupported raw_f32 quant_type {}",
            info.quant_type
        ));
    }
    Ok(())
}

fn validate_embedding_info(
    info: &HfqTensorInfo,
    vocab: usize,
    dim: usize,
    name: &str,
) -> Result<(), String> {
    if info.shape != vec![vocab as u32, dim as u32] {
        return Err(format!(
            "{name}: shape mismatch {:?} vs [{vocab} {dim}]",
            info.shape
        ));
    }
    match info.quant_type {
        6 | 13 | 44 => {
            if info.group_size != 256 {
                return Err(format!(
                    "{name}: group_size must be 256, got {}",
                    info.group_size
                ));
            }
        }
        3 => {
            if info.group_size != 32 {
                return Err(format!(
                    "{name}: Q8 group_size must be 32, got {}",
                    info.group_size
                ));
            }
        }
        1 | 2 | 16 => {
            if info.group_size != 0 {
                return Err(format!(
                    "{name}: group_size must be 0, got {}",
                    info.group_size
                ));
            }
        }
        _ => {}
    }
    let product = vocab * dim;
    let expected = match info.quant_type {
        1 => Some(product * 2),
        2 => Some(product * 4),
        3 => {
            if dim % 32 != 0 {
                return Err(format!("{name}: Q8 dim {dim} not multiple of 32"));
            }
            Some(vocab * (dim / 32) * 34)
        }
        6 => expected_mq4_bytes(vocab, dim),
        13 => expected_mq4_bytes(vocab, dim),
        44 => expected_mq4_bytes(vocab, dim),
        16 => Some(product * 2),
        40 => {
            if dim % 128 != 0 {
                return Err(format!("{name}: TQ2 dim not multiple of 128"));
            }
            Some(vocab * (dim / 128) * 34)
        }
        41 => {
            if dim % 128 != 0 {
                return Err(format!("{name}: BQ1 dim not multiple of 128"));
            }
            Some(vocab * (dim / 128) * 18)
        }
        _ => None,
    };
    if let Some(exp) = expected {
        if info.data_size != exp {
            return Err(format!(
                "{name}: embedding blob mismatch expected {exp}, got {}",
                info.data_size
            ));
        }
    } else {
        return Err(format!(
            "{name}: unsupported embedding quant_type {}",
            info.quant_type
        ));
    }
    Ok(())
}

fn validate_awq_sidecar(hfq: &HfqFile, weight_cand: &str, k: usize) -> Result<(), String> {
    let stem = weight_cand.strip_suffix(".weight").unwrap_or(weight_cand);
    let sidecar = format!("{stem}.awq_scale.weight");
    let Some(info) = hfq.find_tensor_info(&sidecar) else {
        return Ok(());
    };
    if info.quant_type != 1 {
        return Err(format!(
            "AWQ sidecar {sidecar}: quant_type must be 1, got {}",
            info.quant_type
        ));
    }
    if info.shape != vec![k as u32] {
        return Err(format!(
            "AWQ sidecar {sidecar}: shape {:?} vs [{k}]",
            info.shape
        ));
    }
    if info.data_size != k * 2 {
        return Err(format!(
            "AWQ sidecar {sidecar}: blob {} != {}",
            info.data_size,
            k * 2
        ));
    }
    if info.group_size != 0 {
        return Err(format!(
            "AWQ sidecar {sidecar}: group_size must be 0, got {}",
            info.group_size
        ));
    }
    Ok(())
}

/// CPU-only preflight for dense TP2. Validates every tensor the rank loader
/// requires before any GPU allocation. Covers rank-independent invariants;
/// rank slicing bounds remain checked in the rank loader.
pub fn preflight_weights_dense_tp(
    hfq: &HfqFile,
    config: &Qwen35Config,
    shard: &ShardConfig,
) -> Result<(), String> {
    crate::qwen35::config::validate_dense_tp(config, shard)
        .map_err(|e| format!("preflight: {e}"))?;

    // ── Checked blob-range validation before any GPU init ──
    // HfqFile::open only bounds the index; tensor_data_pread zero-pads short
    // reads. Reject any tensor whose declared blob extends past EOF.
    {
        let identity = hfq
            .load_identity()
            .map_err(|e| format!("preflight: load_identity: {e:?}"))?;
        let file_len = identity.len as usize;
        for entry in &identity.manifest {
            let end = entry
                .data_offset
                .checked_add(entry.data_size)
                .ok_or_else(|| {
                    format!(
                        "preflight: tensor {} offset+size overflow (offset {} + size {})",
                        entry.name, entry.data_offset, entry.data_size
                    )
                })?;
            if end > file_len {
                return Err(format!(
                    "preflight: tensor {} blob out of file: offset {} + size {} = {} > file_len {}",
                    entry.name, entry.data_offset, entry.data_size, end, file_len
                ));
            }
        }
    }

    let dim = config.dim;
    let head_dim = config.head_dim;
    let n_heads = config.n_heads;
    let n_kv = config.n_kv_heads;
    let hidden = config.hidden_dim;
    let vocab = config.vocab_size;
    let n_layers = config.n_layers;
    let key_heads = config.linear_num_key_heads;
    let value_heads = config.linear_num_value_heads;
    let key_w = config.linear_key_head_dim;
    let value_w = config.linear_value_head_dim;
    let key_dim = key_heads * key_w;
    let value_dim = value_heads * value_w;
    let qkv_dim = 2 * key_dim + value_dim;
    let conv = config.conv_kernel_dim;

    let validate_proj = |bare: &str, m: usize, k: usize| -> Result<(), String> {
        // An escha-coded DENSE projection (Qwen3.8-27B) has no `.weight` at
        // all — it ships `escha_code` + `escha_rin` + `escha_rout`. Validate
        // the trio that the leaf contract makes REQUIRED (§1.4) and return;
        // the shape lives in the code tensor's own dims and is checked when
        // it is decoded, not here.
        if let Some(stem) = bare.strip_suffix(".weight") {
            let code = format!("{stem}.escha_code");
            if let Some((info, _)) = find_qwen35_tensor(hfq, &code) {
                if info.quant_type == 42 || info.quant_type == 43 {
                    for leaf in ["escha_rin_eff", "escha_rout_eff"] {
                        let n = format!("{stem}.{leaf}");
                        if find_qwen35_tensor(hfq, &n).is_none() {
                            return Err(format!(
                                "preflight: {code} is escha-coded but {n} is missing — an \
                                 incomplete escha linear must fail loudly at load, not \
                                 decode into noise"
                            ));
                        }
                    }
                    if k % 256 != 0 {
                        return Err(format!("preflight: {bare} K={k} not G256 aligned"));
                    }
                    return Ok(());
                }
            }
        }
        let (info, cand) = find_qwen35_tensor(hfq, bare)
            .ok_or_else(|| format!("preflight: missing tensor {bare}"))?;
        validate_mq4_proj_info(info, m, k, bare)?;
        if k % 256 != 0 {
            return Err(format!("preflight: {bare} K={k} not G256 aligned"));
        }
        validate_awq_sidecar(hfq, &cand, k)?;
        Ok(())
    };
    let validate_norm = |bare: &str, n: usize| -> Result<(), String> {
        let (info, _) = find_qwen35_tensor(hfq, bare)
            .ok_or_else(|| format!("preflight: missing tensor {bare}"))?;
        validate_norm_info(info, n, bare)
    };
    let validate_raw = |bare: &str, n: usize| -> Result<(), String> {
        let (info, _) = find_qwen35_tensor(hfq, bare)
            .ok_or_else(|| format!("preflight: missing tensor {bare}"))?;
        validate_raw_f32_info(info, n, bare)
    };

    {
        let (info, _) = find_qwen35_tensor(hfq, "embed_tokens.weight")
            .ok_or_else(|| "preflight: missing embed_tokens.weight".to_string())?;
        validate_embedding_info(info, vocab, dim, "embed_tokens.weight")?;
    }
    validate_norm("norm.weight", dim)?;
    if let Some((info, cand)) = find_qwen35_tensor(hfq, "lm_head.weight") {
        let ok = validate_mq4_proj_info(info, vocab, dim, "lm_head.weight")
            .or_else(|_| validate_embedding_info(info, vocab, dim, "lm_head.weight"));
        ok.map_err(|e| format!("preflight: lm_head: {e}"))?;
        validate_awq_sidecar(hfq, &cand, dim)?;
    }

    for idx in 0..n_layers {
        let lt = config.layer_types[idx];
        let p = format!("layers.{idx}");
        validate_norm(&format!("{p}.input_layernorm.weight"), dim)?;
        validate_norm(&format!("{p}.post_attention_layernorm.weight"), dim)?;
        match lt {
            crate::qwen35::config::LayerType::FullAttention => {
                let q_rows = n_heads * head_dim * 2;
                let kv_dim = n_kv * head_dim;
                let o_in = n_heads * head_dim;
                validate_proj(&format!("{p}.self_attn.q_proj.weight"), q_rows, dim)?;
                validate_proj(&format!("{p}.self_attn.k_proj.weight"), kv_dim, dim)?;
                validate_proj(&format!("{p}.self_attn.v_proj.weight"), kv_dim, dim)?;
                validate_proj(&format!("{p}.self_attn.o_proj.weight"), dim, o_in)?;
                validate_norm(&format!("{p}.self_attn.q_norm.weight"), head_dim)?;
                validate_norm(&format!("{p}.self_attn.k_norm.weight"), head_dim)?;
                validate_proj(&format!("{p}.mlp.gate_proj.weight"), hidden, dim)?;
                validate_proj(&format!("{p}.mlp.up_proj.weight"), hidden, dim)?;
                validate_proj(&format!("{p}.mlp.down_proj.weight"), dim, hidden)?;
                if o_in % 256 != 0 {
                    return Err(format!(
                        "preflight: layer {idx} o_in {o_in} not G256 aligned"
                    ));
                }
                if hidden % 256 != 0 {
                    return Err(format!(
                        "preflight: layer {idx} hidden {hidden} not G256 aligned"
                    ));
                }
            }
            crate::qwen35::config::LayerType::LinearAttention => {
                validate_proj(&format!("{p}.linear_attn.in_proj_qkv.weight"), qkv_dim, dim)?;
                validate_proj(&format!("{p}.linear_attn.in_proj_z.weight"), value_dim, dim)?;
                validate_proj(
                    &format!("{p}.linear_attn.in_proj_a.weight"),
                    value_heads,
                    dim,
                )?;
                validate_proj(
                    &format!("{p}.linear_attn.in_proj_b.weight"),
                    value_heads,
                    dim,
                )?;
                validate_raw(&format!("{p}.linear_attn.A_log"), value_heads)?;
                validate_raw(&format!("{p}.linear_attn.dt_bias"), value_heads)?;
                validate_raw(&format!("{p}.linear_attn.conv1d.weight"), qkv_dim * conv)?;
                validate_raw(
                    &format!("{p}.linear_attn.norm.weight"),
                    config.linear_value_head_dim,
                )?;
                validate_proj(&format!("{p}.linear_attn.out_proj.weight"), dim, value_dim)?;
                validate_proj(&format!("{p}.mlp.gate_proj.weight"), hidden, dim)?;
                validate_proj(&format!("{p}.mlp.up_proj.weight"), hidden, dim)?;
                validate_proj(&format!("{p}.mlp.down_proj.weight"), dim, hidden)?;
            }
        }
        if config.num_experts != 0 {
            return Err("preflight: dense TP requires num_experts=0".to_string());
        }
    }
    Ok(())
}

fn load_weight_tensor_dense_tp(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
    slice: DenseTpSlice<'_>,
) -> HipResult<WeightTensor> {
    use hipfire_runtime::tp_shard::{slice_uniform_group_cols, slice_uniform_rows};

    // Single metadata lookup for candidate/quant_type; exactly one pread for data below.
    let (info, candidate) = find_qwen35_tensor(hfq, name)
        .ok_or_else(|| HipError::new(0, &format!("TP tensor not found: {name}")))?;
    let quant_type = info.quant_type;
    let candidate = candidate.clone();
    if !is_dense_proj_qt(quant_type) {
        return Err(HipError::new(
            0,
            &format!(
                "dense TP2 requires MQ4-G256 qt13 or MQ4V2 qt44, {candidate} has qt={quant_type}"
            ),
        ));
    }
    let (bytes, new_m, new_k) = {
        let (_, data) = hfq
            .tensor_data_pread(&candidate)
            .ok_or_else(|| HipError::new(0, "TP tensor disappeared during load"))?;
        match slice {
            DenseTpSlice::Rows(start, end) => (
                slice_uniform_rows(&data, m, start, end).map_err(|e| HipError::new(0, &e))?,
                end - start,
                k,
            ),
            DenseTpSlice::RowsMulti(ranges) => {
                let mut out = Vec::new();
                for &(start, end) in ranges {
                    out.extend_from_slice(
                        &slice_uniform_rows(&data, m, start, end)
                            .map_err(|e| HipError::new(0, &e))?,
                    );
                }
                (out, ranges.iter().map(|(a, b)| b - a).sum(), k)
            }
            DenseTpSlice::Cols(start, end) => (
                slice_uniform_group_cols(&data, m, k, start, end, 256)
                    .map_err(|e| HipError::new(0, &e))?,
                m,
                end - start,
            ),
        }
    };
    let mut weight = match load_weight_tensor_raw(gpu, quant_type, &bytes, new_m, new_k) {
        Ok(w) => w,
        Err(e) => return Err(e),
    };
    let stem = candidate.strip_suffix(".weight").unwrap_or(&candidate);
    let sidecar = format!("{stem}.awq_scale.weight");
    if let Some((info, data)) = hfq.tensor_data_pread(&sidecar) {
        if info.quant_type != 1 || data.len() != k * 2 {
            weight.free_all(gpu);
            return Err(HipError::new(0, "invalid TP AWQ sidecar"));
        }
        let (start, end) = match slice {
            DenseTpSlice::Cols(start, end) => (start, end),
            _ => (0, k),
        };
        let scales: Vec<f32> = data[start * 2..end * 2]
            .chunks_exact(2)
            .map(|v| f16_to_f32(u16::from_le_bytes([v[0], v[1]])))
            .collect();
        match gpu.upload_f32(&scales, &[scales.len()]) {
            Ok(t) => weight.awq_scale = Some(t),
            Err(e) => {
                weight.free_all(gpu);
                return Err(e);
            }
        }
    }
    Ok(weight)
}

fn load_raw_f32_sliced(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    bare: &str,
    n: usize,
    ranges: &[(usize, usize)],
) -> HipResult<GpuTensor> {
    let (info, data) = qwen35_tensor_data_vec(hfq, bare)
        .ok_or_else(|| HipError::new(0, &format!("tensor not found: {bare}")))?;
    let total: Vec<f32> = match info.quant_type {
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
            .map(|c| {
                hipfire_runtime::safetensors_source::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))
            })
            .collect(),
        3 => hipfire_runtime::llama::dequantize_q8_0(&data, n),
        qt => {
            return Err(HipError::new(
                0,
                &format!("unsupported raw f32 quant {qt} for {bare}"),
            ))
        }
    };
    if total.len() != n {
        return Err(HipError::new(
            0,
            &format!("{bare} f32 len {} != {n}", total.len()),
        ));
    }
    let mut gathered = Vec::new();
    let mut total_len = 0usize;
    for &(s, e) in ranges {
        if e > n || s > e {
            return Err(HipError::new(
                0,
                &format!("invalid f32 slice {s}..{e} for {bare} n={n}"),
            ));
        }
        total_len += e - s;
        gathered.extend_from_slice(&total[s..e]);
    }
    gpu.upload_f32(&gathered, &[total_len])
}

fn gather_f32_ranges(
    gpu: &mut Gpu,
    source: &GpuTensor,
    ranges: &[(usize, usize)],
) -> HipResult<GpuTensor> {
    let source = gpu.download_f32(source)?;
    let mut gathered = Vec::new();
    for &(start, end) in ranges {
        gathered.extend_from_slice(&source[start..end]);
    }
    gpu.upload_f32(&gathered, &[gathered.len()])
}

struct DenseTpPending {
    token_embd: Option<GpuTensor>,
    embd_format: Option<EmbeddingFormat>,
    output_norm: Option<GpuTensor>,
    output: Option<WeightTensor>,
    lm_head_aliases_embd: bool,
    layers: Vec<LayerWeights>,
}

impl DenseTpPending {
    fn new() -> Self {
        Self {
            token_embd: None,
            embd_format: None,
            output_norm: None,
            output: None,
            lm_head_aliases_embd: false,
            layers: Vec::new(),
        }
    }
    fn free(self, gpu: &mut Gpu) {
        if let Some(t) = self.token_embd {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.output_norm {
            let _ = gpu.free_tensor(t);
        }
        if let Some(w) = self.output {
            // Mirror Qwen35Weights::free_gpu: when lm_head aliases token_embd,
            // output.buf is a non-owning view of token_embd.buf — do not free
            // it here (would double-free token_embd). Includes sidecar attach
            // failure semantics: if output was aliased, its buffer is never
            // freed separately; any sidecar would have been attached to the
            // alias view and is not owned separately in the aliased case.
            if !self.lm_head_aliases_embd {
                w.free_all(gpu);
            }
        }
        for layer in self.layers {
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                _ => {}
            }
        }
    }
}

/// Load one rank of a dense Qwen TP model (TP2..5). Directly constructs
/// each rank's `Qwen35Weights` from the exact `DenseTpRankLayout` ranges
/// without ever allocating the full model or any full projection on this rank.
/// Transactional: any allocation not yet published is freed on error. The
/// layout is the model-lifetime static whole-unit sharding produced by
/// `dense_tp_rank_layouts`; the loader computes all layouts CPU-only before any
/// GPU allocation and never recomputes even divisions here.
pub fn load_weights_dense_tp_rank(
    hfq: &mut HfqFile,
    config: &Qwen35Config,
    gpu: &mut Gpu,
    layout: &crate::qwen35::config::DenseTpRankLayout,
) -> HipResult<Qwen35Weights> {
    // Production callers construct every layout and run full blob preflight
    // before GPU initialization. Keep this rank-local guard for direct callers.
    let valid_range = |range: &std::ops::Range<usize>, limit: usize| {
        !range.is_empty() && range.start < range.end && range.end <= limit
    };
    if !valid_range(&layout.q_head_range, config.n_heads)
        || !valid_range(&layout.kv_head_range, config.n_kv_heads)
        || !valid_range(&layout.delta_key_head_range, config.linear_num_key_heads)
        || !valid_range(
            &layout.delta_value_head_range,
            config.linear_num_value_heads,
        )
        || !valid_range(&layout.ffn_hidden_range, config.hidden_dim)
    {
        return Err(HipError::new(0, "invalid dense TP rank layout"));
    }
    let dim = config.dim;
    let head_dim = config.head_dim;
    let q_range = layout.q_head_range.clone();
    let kv_range = layout.kv_head_range.clone();
    let kv_dim = config.n_kv_heads * head_dim;
    let q_rows = q_range.start * head_dim * 2..q_range.end * head_dim * 2;
    let kv_rows = kv_range.start * head_dim..kv_range.end * head_dim;
    let attn_cols = q_range.start * head_dim..q_range.end * head_dim;
    let ffn_range = layout.ffn_hidden_range.clone();
    let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let dn_rows = 2 * key_dim + value_dim;
    let qkv_dim = dn_rows;
    let dk_range = layout.delta_key_head_range.clone();
    let dv_range = layout.delta_value_head_range.clone();
    let key_width = config.linear_key_head_dim;
    let value_width = config.linear_value_head_dim;
    let qkv_ranges = [
        (dk_range.start * key_width, dk_range.end * key_width),
        (
            key_dim + dk_range.start * key_width,
            key_dim + dk_range.end * key_width,
        ),
        (
            2 * key_dim + dv_range.start * value_width,
            2 * key_dim + dv_range.end * value_width,
        ),
    ];
    let conv = config.conv_kernel_dim;
    let conv_ranges = [
        (qkv_ranges[0].0 * conv, qkv_ranges[0].1 * conv),
        (qkv_ranges[1].0 * conv, qkv_ranges[1].1 * conv),
        (qkv_ranges[2].0 * conv, qkv_ranges[2].1 * conv),
    ];
    let mut pending = DenseTpPending::new();
    let res: HipResult<Qwen35Weights> = (|| {
        let (embd_info, embd_data) = qwen35_tensor_data_cow(hfq, "embed_tokens.weight")
            .ok_or_else(|| HipError::new(0, "embed_tokens.weight not found"))?;
        let (token_embd, embd_format) = load_embedding(
            gpu,
            embd_info.quant_type,
            &embd_data,
            config.vocab_size,
            dim,
        )?;
        pending.token_embd = Some(token_embd);
        pending.embd_format = Some(embd_format);
        drop(embd_data);

        let output_norm = load_norm_weight(hfq, gpu, "norm.weight", &[dim])?;
        pending.output_norm = Some(output_norm);

        let has_separate = qwen35_tensor_name_candidates("lm_head.weight")
            .iter()
            .any(|n| hfq.find_tensor_info(n).is_some());
        let token_ref = pending.token_embd.as_ref().unwrap();
        let fmt = pending.embd_format.unwrap();
        let (mut output, aliases) = hipfire_runtime::weight_backend::resolve_lm_head(
            gpu,
            has_separate,
            true,
            token_ref,
            fmt,
            config.vocab_size,
            dim,
            |gpu| {
                let (info, data) = qwen35_tensor_data_cow(hfq, "lm_head.weight")
                    .ok_or_else(|| HipError::new(0, "lm_head.weight not found"))?;
                load_weight_tensor_raw(gpu, info.quant_type, &data, config.vocab_size, dim)
            },
            |gpu| {
                let (info, data) = qwen35_tensor_data_cow(hfq, "embed_tokens.weight")
                    .ok_or_else(|| HipError::new(0, "embed_tokens.weight not found"))?;
                hipfire_runtime::weight_backend::reupload_f16_as_f32(
                    gpu,
                    &data,
                    config.vocab_size,
                    dim,
                )
            },
        )?;
        attach_lm_head_awq_sidecar(hfq, gpu, &mut output, dim);
        pending.output = Some(output);
        pending.lm_head_aliases_embd = aliases;

        for layer_idx in 0..config.n_layers {
            let lt = config.layer_types[layer_idx];
            let p = format!("layers.{layer_idx}");
            let layer: LayerWeights = match lt {
                crate::qwen35::config::LayerType::FullAttention => {
                    let mut attn_norm_opt: Option<GpuTensor> = None;
                    let mut wq_opt: Option<WeightTensor> = None;
                    let mut wk_opt: Option<WeightTensor> = None;
                    let mut wv_opt: Option<WeightTensor> = None;
                    let mut wo_opt: Option<WeightTensor> = None;
                    let mut q_norm_opt: Option<GpuTensor> = None;
                    let mut k_norm_opt: Option<GpuTensor> = None;
                    let mut ffn_norm_opt: Option<GpuTensor> = None;
                    let mut w_gate_opt: Option<WeightTensor> = None;
                    let mut w_up_opt: Option<WeightTensor> = None;
                    let mut w_down_opt: Option<WeightTensor> = None;

                    let layer_res: HipResult<LayerWeights> = (|| {
                        let attn_norm = load_norm_weight(
                            hfq,
                            gpu,
                            &format!("{p}.input_layernorm.weight"),
                            &[dim],
                        )?;
                        attn_norm_opt = Some(attn_norm);
                        let wq = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.q_proj.weight"),
                            config.n_heads * head_dim * 2,
                            dim,
                            DenseTpSlice::Rows(q_rows.start, q_rows.end),
                        )?;
                        wq_opt = Some(wq);
                        let wk = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.k_proj.weight"),
                            kv_dim,
                            dim,
                            DenseTpSlice::Rows(kv_rows.start, kv_rows.end),
                        )?;
                        wk_opt = Some(wk);
                        let wv = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.v_proj.weight"),
                            kv_dim,
                            dim,
                            DenseTpSlice::Rows(kv_rows.start, kv_rows.end),
                        )?;
                        wv_opt = Some(wv);
                        let wo = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.o_proj.weight"),
                            dim,
                            config.n_heads * head_dim,
                            DenseTpSlice::Cols(attn_cols.start, attn_cols.end),
                        )?;
                        wo_opt = Some(wo);
                        let q_norm = load_norm_weight(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.q_norm.weight"),
                            &[head_dim],
                        )?;
                        q_norm_opt = Some(q_norm);
                        let k_norm = load_norm_weight(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.k_norm.weight"),
                            &[head_dim],
                        )?;
                        k_norm_opt = Some(k_norm);
                        let ffn_norm = load_norm_weight(
                            hfq,
                            gpu,
                            &format!("{p}.post_attention_layernorm.weight"),
                            &[dim],
                        )?;
                        ffn_norm_opt = Some(ffn_norm);
                        let w_gate = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.gate_proj.weight"),
                            config.hidden_dim,
                            dim,
                            DenseTpSlice::Rows(ffn_range.start, ffn_range.end),
                        )?;
                        w_gate_opt = Some(w_gate);
                        let w_up = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.up_proj.weight"),
                            config.hidden_dim,
                            dim,
                            DenseTpSlice::Rows(ffn_range.start, ffn_range.end),
                        )?;
                        w_up_opt = Some(w_up);
                        let w_down = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.down_proj.weight"),
                            dim,
                            config.hidden_dim,
                            DenseTpSlice::Cols(ffn_range.start, ffn_range.end),
                        )?;
                        w_down_opt = Some(w_down);
                        Ok(LayerWeights::FullAttn(FullAttnLayerWeights {
                            attn_norm: attn_norm_opt.take().unwrap(),
                            wq: wq_opt.take().unwrap(),
                            wk: wk_opt.take().unwrap(),
                            wv: wv_opt.take().unwrap(),
                            wo: wo_opt.take().unwrap(),
                            q_norm: q_norm_opt.take().unwrap(),
                            k_norm: k_norm_opt.take().unwrap(),
                            ffn_norm: ffn_norm_opt.take().unwrap(),
                            w_gate: w_gate_opt.take().unwrap(),
                            w_up: w_up_opt.take().unwrap(),
                            w_down: w_down_opt.take().unwrap(),
                            biases: None,
                            escha: None,
                        }))
                    })();
                    match layer_res {
                        Ok(l) => l,
                        Err(e) => {
                            if let Some(t) = attn_norm_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(w) = wq_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = wk_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = wv_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = wo_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(t) = q_norm_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(t) = k_norm_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(t) = ffn_norm_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(w) = w_gate_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = w_up_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = w_down_opt.take() {
                                w.free_all(gpu);
                            }
                            return Err(e);
                        }
                    }
                }
                crate::qwen35::config::LayerType::LinearAttention => {
                    let mut attn_norm_opt: Option<GpuTensor> = None;
                    let mut wqkv_opt: Option<WeightTensor> = None;
                    let mut wz_opt: Option<WeightTensor> = None;
                    let mut w_alpha_opt: Option<WeightTensor> = None;
                    let mut w_beta_opt: Option<WeightTensor> = None;
                    let mut a_log_opt: Option<GpuTensor> = None;
                    let mut dt_bias_opt: Option<GpuTensor> = None;
                    let mut conv_weight_opt: Option<GpuTensor> = None;
                    let mut norm_weight_opt: Option<GpuTensor> = None;
                    let mut wo_opt: Option<WeightTensor> = None;
                    let mut ffn_norm_opt: Option<GpuTensor> = None;
                    let mut w_gate_opt: Option<WeightTensor> = None;
                    let mut w_up_opt: Option<WeightTensor> = None;
                    let mut w_down_opt: Option<WeightTensor> = None;
                    let layer_res: HipResult<LayerWeights> = (|| {
                        let attn_norm = load_norm_weight(
                            hfq,
                            gpu,
                            &format!("{p}.input_layernorm.weight"),
                            &[dim],
                        )?;
                        attn_norm_opt = Some(attn_norm);
                        let wqkv = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.linear_attn.in_proj_qkv.weight"),
                            dn_rows,
                            dim,
                            DenseTpSlice::RowsMulti(&qkv_ranges),
                        )?;
                        wqkv_opt = Some(wqkv);
                        let wz = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.linear_attn.in_proj_z.weight"),
                            value_dim,
                            dim,
                            DenseTpSlice::Rows(
                                dv_range.start * value_width,
                                dv_range.end * value_width,
                            ),
                        )?;
                        wz_opt = Some(wz);
                        let w_alpha = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.linear_attn.in_proj_a.weight"),
                            config.linear_num_value_heads,
                            dim,
                            DenseTpSlice::Rows(dv_range.start, dv_range.end),
                        )?;
                        w_alpha_opt = Some(w_alpha);
                        let w_beta = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.linear_attn.in_proj_b.weight"),
                            config.linear_num_value_heads,
                            dim,
                            DenseTpSlice::Rows(dv_range.start, dv_range.end),
                        )?;
                        w_beta_opt = Some(w_beta);
                        let a_log = load_raw_f32_sliced(
                            hfq,
                            gpu,
                            &format!("{p}.linear_attn.A_log"),
                            config.linear_num_value_heads,
                            &[(dv_range.start, dv_range.end)],
                        )?;
                        a_log_opt = Some(a_log);
                        let dt_bias = load_raw_f32_sliced(
                            hfq,
                            gpu,
                            &format!("{p}.linear_attn.dt_bias"),
                            config.linear_num_value_heads,
                            &[(dv_range.start, dv_range.end)],
                        )?;
                        dt_bias_opt = Some(dt_bias);
                        let conv_weight = load_raw_f32_sliced(
                            hfq,
                            gpu,
                            &format!("{p}.linear_attn.conv1d.weight"),
                            qkv_dim * conv,
                            &conv_ranges,
                        )?;
                        conv_weight_opt = Some(conv_weight);
                        let norm_weight = load_any_as_f32(
                            hfq,
                            gpu,
                            &format!("{p}.linear_attn.norm.weight"),
                            config.linear_value_head_dim,
                        )?;
                        norm_weight_opt = Some(norm_weight);
                        let wo = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.linear_attn.out_proj.weight"),
                            dim,
                            value_dim,
                            DenseTpSlice::Cols(
                                dv_range.start * value_width,
                                dv_range.end * value_width,
                            ),
                        )?;
                        wo_opt = Some(wo);
                        let ffn_norm = load_norm_weight(
                            hfq,
                            gpu,
                            &format!("{p}.post_attention_layernorm.weight"),
                            &[dim],
                        )?;
                        ffn_norm_opt = Some(ffn_norm);
                        let w_gate = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.gate_proj.weight"),
                            config.hidden_dim,
                            dim,
                            DenseTpSlice::Rows(ffn_range.start, ffn_range.end),
                        )?;
                        w_gate_opt = Some(w_gate);
                        let w_up = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.up_proj.weight"),
                            config.hidden_dim,
                            dim,
                            DenseTpSlice::Rows(ffn_range.start, ffn_range.end),
                        )?;
                        w_up_opt = Some(w_up);
                        let w_down = load_weight_tensor_dense_tp(
                            hfq,
                            gpu,
                            &format!("{p}.mlp.down_proj.weight"),
                            dim,
                            config.hidden_dim,
                            DenseTpSlice::Cols(ffn_range.start, ffn_range.end),
                        )?;
                        w_down_opt = Some(w_down);
                        Ok(LayerWeights::DeltaNet(DeltaNetLayerWeights {
                            attn_norm: attn_norm_opt.take().unwrap(),
                            wqkv: wqkv_opt.take().unwrap(),
                            wz: wz_opt.take().unwrap(),
                            w_alpha: w_alpha_opt.take().unwrap(),
                            w_beta: w_beta_opt.take().unwrap(),
                            a_log: a_log_opt.take().unwrap(),
                            dt_bias: dt_bias_opt.take().unwrap(),
                            conv_weight: conv_weight_opt.take().unwrap(),
                            norm_weight: norm_weight_opt.take().unwrap(),
                            wo: wo_opt.take().unwrap(),
                            ffn_norm: ffn_norm_opt.take().unwrap(),
                            w_gate: w_gate_opt.take().unwrap(),
                            w_up: w_up_opt.take().unwrap(),
                            w_down: w_down_opt.take().unwrap(),
                            biases: None,
                            escha: None,
                        }))
                    })();
                    match layer_res {
                        Ok(l) => l,
                        Err(e) => {
                            if let Some(t) = attn_norm_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(w) = wqkv_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = wz_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = w_alpha_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = w_beta_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(t) = a_log_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(t) = dt_bias_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(t) = conv_weight_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(t) = norm_weight_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(w) = wo_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(t) = ffn_norm_opt.take() {
                                let _ = gpu.free_tensor(t);
                            }
                            if let Some(w) = w_gate_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = w_up_opt.take() {
                                w.free_all(gpu);
                            }
                            if let Some(w) = w_down_opt.take() {
                                w.free_all(gpu);
                            }
                            return Err(e);
                        }
                    }
                }
                _ => return Err(HipError::new(0, "dense TP does not admit MoE layers")),
            };
            pending.layers.push(layer);
        }

        let token_embd = pending.token_embd.take().unwrap();
        let embd_format = pending.embd_format.take().unwrap();
        let output_norm = pending.output_norm.take().unwrap();
        let output = pending.output.take().unwrap();
        let lm_head_aliases_embd = pending.lm_head_aliases_embd;
        let layers = std::mem::take(&mut pending.layers);
        Ok(Qwen35Weights {
            token_embd,
            embd_format,
            output_norm,
            output,
            moe_has_mq6: false,
            layers,
            pager: None,
            lm_head_aliases_embd,
            ep_shard: None,
        })
    })();
    match res {
        Ok(w) => Ok(w),
        Err(e) => {
            pending.free(gpu);
            Err(e)
        }
    }
}

thread_local! {
    /// Per-thread EP expert-shard context. When `Some((shard, rank))`,
    /// [`load_moe_ffn`] loads ONLY this rank's owned experts (streaming
    /// owned-only) and builds the `[n_exp]` global pointer tables with dummy
    /// pointers for non-owned slots — the SAME structure post-load
    /// [`shard_moe_experts`] produces, but WITHOUT the full-model load peak that
    /// OOMs a model larger than one card's VRAM. Set by the EP load driver
    /// around `load_weights`, cleared (`None`) after. `None` = full replicated
    /// load (the default for every non-EP caller).
    static EP_EXPERT_SHARD: std::cell::RefCell<Option<(ShardConfig, usize)>> =
        const { std::cell::RefCell::new(None) };
}

/// Set the per-thread EP expert-shard context consumed by `load_weights` →
/// [`load_moe_ffn`]. The EP load driver calls this with `Some((shard, rank))`
/// immediately before `load_weights` on each rank, then `None` immediately
/// after. Mirrors DeepSeek-V4's `load_weights_sharded` but threaded via TLS so
/// the 87 existing `load_weights` callers need no signature change.
pub fn set_ep_expert_shard(ctx: Option<(ShardConfig, usize)>) {
    EP_EXPERT_SHARD.with(|c| *c.borrow_mut() = ctx);
}

fn current_ep_expert_shard() -> Option<(ShardConfig, usize)> {
    EP_EXPERT_SHARD.with(|c| c.borrow().clone())
}
/// RAII guard for `EP_EXPERT_SHARD`. Sets on creation, clears on drop
/// even if the wrapped load returns an error (fail-closed TLS hygiene).
pub struct EpShardGuard;
impl EpShardGuard {
    pub fn new(shard: ShardConfig, rank: usize) -> Self {
        set_ep_expert_shard(Some((shard, rank)));
        Self
    }
}
impl Drop for EpShardGuard {
    fn drop(&mut self) {
        set_ep_expert_shard(None);
    }
}

/// EP single-rank streaming loader with RAII TLS hygiene. Wraps
/// `HfqFile` + `Qwen35Config` + `ShardConfig` and ensures
/// `EP_EXPERT_SHARD` is cleared on every exit path.
pub fn load_weights_ep_rank(
    hfq: &mut HfqFile,
    gpu: &mut Gpu,
    config: &Qwen35Config,
    shard: ShardConfig,
    rank: usize,
) -> HipResult<Qwen35Weights> {
    // ── Prevalidate before any GPU/TLS work (fail-closed, no side effects) ──
    if rank >= shard.tp_size {
        return Err(HipError::new(
            0,
            &format!(
                "EP load: rank {rank} out of range for tp_size {}",
                shard.tp_size
            ),
        ));
    }
    if shard.tp_size != 4 {
        return Err(HipError::new(
            0,
            &format!("EP load: tp_size must be exactly 4, got {}", shard.tp_size),
        ));
    }
    shard
        .validate_moe(config.num_experts)
        .map_err(|e| HipError::new(0, &format!("EP load: {e}")))?;
    // Pure-EP: replicated attention/recurrent/shared/output, exactly-one-owner
    // routed experts, PP1. No paging, no REAP, MoE must be active.
    if config.paged_experts {
        return Err(HipError::new(0, "EP load: paged_experts must be false"));
    }
    if config.reap_keep.is_some() {
        return Err(HipError::new(
            0,
            "EP load: REAP keep-map incompatible with EP",
        ));
    }
    if config.num_experts == 0 {
        return Err(HipError::new(0, "EP load: config has no routed experts"));
    }
    // Exactly one rank owns each global expert (detect missing/replicated).
    {
        let mut counts = [0usize; 4];
        for &r in &shard.expert_to_rank {
            counts[r as usize] += 1;
        }
        for (i, c) in counts.iter().enumerate() {
            if *c == 0 {
                return Err(HipError::new(
                    0,
                    &format!("EP load: rank {i} owns zero experts (missing shard)"),
                ));
            }
        }
    }
    // Pure-EP replicated-nonexpert contract: attention geometry must support
    // replication (no TP sharding of non-expert weights). Validate that the
    // shard would be valid for full-attention and DeltaNet head splits if it
    // were a TP shard — EP reuses the same head counts but forces replication.
    shard
        .validate(config.n_heads, config.n_kv_heads)
        .map_err(|e| HipError::new(0, &format!("EP load: pure-EP attention: {e}")))?;
    shard
        .validate_deltanet(config.linear_num_value_heads, config.linear_num_key_heads)
        .map_err(|e| HipError::new(0, &format!("EP load: pure-EP deltanet: {e}")))?;

    // ── Capture immutable seals before any GPU allocation ──
    let source_identity = std::sync::Arc::new(Qwen35HfqSourceIdentity::capture(&*hfq));
    let config_fingerprint = Qwen35EpConfigFingerprint::capture(config);
    let device_id = gpu.device_id;
    // ── Load with TLS sharding ──
    let _guard = EpShardGuard::new(shard.clone(), rank);
    let mut source = HfqSource::new(hfq, config);
    let layout = hipfire_runtime::model_load::Layout::single(config.n_layers);
    let mut weights = load_weights(&mut source, std::slice::from_mut(gpu), &layout)?;
    // Attach immutable provenance only after a complete successful load.
    let rank_seal = Qwen35RankSeal::capture(&weights, Some(&shard.expert_to_rank), rank);
    weights.ep_shard = Some(Qwen35EpShardInfo {
        rank: rank as u8,
        rank_count: shard.tp_size as u8,
        expert_to_rank: shard.expert_to_rank.into_boxed_slice(),
        device_id,
        source_identity,
        config_fingerprint,
        rank_seal,
    });
    Ok(weights)
}
///
/// HIPFIRE_E8_SOA_EXPERTS (cached): transpose routed E8 gate_up experts AoS->SoA at
/// load so the SoA-coalesced indexed kernel can read them. Must match the dispatch
/// flag in rdna-compute (same env). Default OFF.
fn e8_soa_experts() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_E8_SOA_EXPERTS")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

const MQ4_G256_QUANT_TYPE: u8 = 13;

struct PackedMq4ExpertSpec {
    gate_up_name: String,
    down_name: String,
}

/// Restrict expert packing to the gfx11 family where both dGPU residency and
/// gfx1151 retained-replay performance have been validated. On gfx12, placing
/// every expert view inside two large layer allocations makes the HIP/graph
/// route treat those views as one coarse allocation domain; gfx1201 tg128 fell
/// from 171 to 77 tok/s even though retained PM4 remained fast. Preserve the
/// original per-expert allocations there until a gfx12-safe packing granularity
/// is established.
fn packed_mq4_experts_supported(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_rdna3()
}

/// Pack a uniform MQ4 routed-expert layer into two GPU allocations while
/// preserving one `WeightTensor` view and one device pointer-table entry per
/// expert. Returns `None` for every non-uniform/non-MQ4 layout so mixed tiers,
/// ParoQuant, paged experts, and EP streaming retain their literal behavior.

/// Host-side byte source for a packed blob: either a borrowed mmap slice
enum HostBlob<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl AsRef<[u8]> for HostBlob<'_> {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(v) => v,
        }
    }
}

/// Cumulative packed-expert sweep timings for the current model load, in ms.
static PACKED_READ_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PACKED_UPLOAD_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Verify that `names[0..n]` are uniform-`stride` tensors laid out back-to-back
/// in file order, and return the single borrowed mmap slice covering all of
/// them. Returns `None` on any gap, overlap, stride mismatch, dropped mmap
/// (UMA), or attached overlay.
fn contiguous_tensor_span<'a>(hfq: &'a HfqFile, names: &[&str], stride: usize) -> Option<&'a [u8]> {
    if names.is_empty() || hfq.has_overlay() {
        return None;
    }
    let first = hfq.find_tensor_info(names[0])?;
    let mut expect = first.data_offset;
    for name in names {
        let info = hfq.find_tensor_info(name)?;
        if info.data_offset != expect || info.data_size != stride {
            return None;
        }
        expect = info.data_offset.checked_add(info.data_size)?;
    }
    hfq.data_range(first.data_offset, expect - first.data_offset)
}

fn try_load_packed_mq4_experts(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    expert_ids: &[usize],
    mi: usize,
    dim: usize,
) -> HipResult<Option<(Vec<ExpertWeights>, PackedExpertOwners)>> {
    use std::sync::atomic::Ordering;
    use std::time::Instant;
    if expert_ids.is_empty() {
        return Ok(None);
    }

    let mut specs = Vec::with_capacity(expert_ids.len());
    let mut gate_up_stride = None;
    let mut down_stride = None;
    for &expert_id in expert_ids {
        let gate_up_bare = format!("{p}.mlp.experts.{expert_id}.gate_up_proj.weight");
        let down_bare = format!("{p}.mlp.experts.{expert_id}.down_proj.weight");
        let Some((gate_up_name, gate_up_qt, gate_up_bytes)) =
            qwen35_tensor_name_candidates(&gate_up_bare)
                .into_iter()
                .find_map(|name| {
                    hfq.find_tensor_info(&name)
                        .map(|info| (name, info.quant_type, info.data_size))
                })
        else {
            return Ok(None);
        };
        let Some((down_name, down_qt, down_bytes)) = qwen35_tensor_name_candidates(&down_bare)
            .into_iter()
            .find_map(|name| {
                hfq.find_tensor_info(&name)
                    .map(|info| (name, info.quant_type, info.data_size))
            })
        else {
            return Ok(None);
        };
        if gate_up_qt != MQ4_G256_QUANT_TYPE || down_qt != MQ4_G256_QUANT_TYPE {
            return Ok(None);
        }
        match gate_up_stride {
            None => gate_up_stride = Some(gate_up_bytes),
            Some(stride) if stride == gate_up_bytes => {}
            Some(_) => return Ok(None),
        }
        match down_stride {
            None => down_stride = Some(down_bytes),
            Some(stride) if stride == down_bytes => {}
            Some(_) => return Ok(None),
        }
        specs.push(PackedMq4ExpertSpec {
            gate_up_name,
            down_name,
        });
    }

    let gate_up_stride = gate_up_stride.expect("non-empty packed MQ4 expert list");
    let down_stride = down_stride.expect("non-empty packed MQ4 expert list");
    // Zero-copy fast path: when every expert's tensor lies contiguous in-file
    // (expert 0's gate_up ends exactly where expert 1's begins, etc.) and the
    // mmap is alive (dGPU, no overlay), upload both blobs straight from the
    // page cache — no pread staging Vec, no heap copy. Measured dominant cost
    // of A3B MoE loads otherwise: 512 pread segments per blob into a fresh
    // heap buffer, then a second copy into hipMalloc'd VRAM.
    let gate_up_names: Vec<&str> = specs.iter().map(|s| s.gate_up_name.as_str()).collect();
    let down_names: Vec<&str> = specs.iter().map(|s| s.down_name.as_str()).collect();
    let gate_up_contig = contiguous_tensor_span(hfq, &gate_up_names, gate_up_stride);
    let down_contig = contiguous_tensor_span(hfq, &down_names, down_stride);

    let t_read = Instant::now();
    // Zero-copy only pays when pages are resident: a borrowed-slice H2D from
    // an evicted cache soft-faults serially inside the copy loop (~0.25 GB/s
    // off disk), while the parallel-pread fallback reads with multiple lanes.
    let zero_copy_ok = matches!((gate_up_contig, down_contig), (Some(_), Some(_)))
        && hfq.mostly_page_cached_memo();
    let (gate_up_host, down_host) = if zero_copy_ok {
        let (Some(gu), Some(dn)) = (gate_up_contig, down_contig) else {
            unreachable!("zero_copy_ok implies both spans resolved");
        };
        (HostBlob::Borrowed(gu), HostBlob::Borrowed(dn))
    } else if hfq.has_overlay() {
        // Overlay offsets belong to a second file; retain the overlay-aware
        // serial path exactly rather than crossing files in reader workers.
        let mut gate_up_host = vec![0u8; gate_up_stride * specs.len()];
        let mut down_host = vec![0u8; down_stride * specs.len()];
        for (slot, spec) in specs.iter().enumerate() {
            {
                let (_, bytes) = hfq.tensor_data_pread(&spec.gate_up_name).ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!(
                            "qwen35: packed MQ4 tensor disappeared: {}",
                            spec.gate_up_name
                        ),
                    )
                })?;
                if bytes.len() != gate_up_stride {
                    return Err(HipError::new(
                        0,
                        &format!(
                            "qwen35: packed MQ4 gate_up stride changed for {}: {} != {gate_up_stride}",
                            spec.gate_up_name,
                            bytes.len()
                        ),
                    ));
                }
                gate_up_host[slot * gate_up_stride..(slot + 1) * gate_up_stride]
                    .copy_from_slice(&bytes);
            }
            {
                let (_, bytes) = hfq.tensor_data_pread(&spec.down_name).ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!("qwen35: packed MQ4 tensor disappeared: {}", spec.down_name),
                    )
                })?;
                if bytes.len() != down_stride {
                    return Err(HipError::new(
                        0,
                        &format!(
                            "qwen35: packed MQ4 down stride changed for {}: {} != {down_stride}",
                            spec.down_name,
                            bytes.len()
                        ),
                    ));
                }
                down_host[slot * down_stride..(slot + 1) * down_stride].copy_from_slice(&bytes);
            }
        }
        (HostBlob::Owned(gate_up_host), HostBlob::Owned(down_host))
    } else {
        let jobs = [
            HfqReadJob::packed(
                hfq,
                format!("{p}.packed_mq4_gate_up"),
                specs.iter().map(|spec| spec.gate_up_name.as_str()),
            )
            .map_err(|e| HipError::new(0, &format!("qwen35: plan packed gate_up: {e}")))?,
            HfqReadJob::packed(
                hfq,
                format!("{p}.packed_mq4_down"),
                specs.iter().map(|spec| spec.down_name.as_str()),
            )
            .map_err(|e| HipError::new(0, &format!("qwen35: plan packed down: {e}")))?,
        ];
        let mut results = read_hfq_jobs_ordered(hfq, &jobs)
            .map_err(|e| HipError::new(0, &format!("qwen35: parallel packed expert read: {e}")))?
            .into_iter();
        (
            HostBlob::Owned(results.next().expect("two packed MQ4 jobs").data),
            HostBlob::Owned(results.next().expect("two packed MQ4 jobs").data),
        )
    };
    let trace = std::env::var_os("HIPFIRE_LOAD_TRACE").is_some();
    let zero_copy =
        matches!(gate_up_host, HostBlob::Borrowed(_)) && matches!(down_host, HostBlob::Borrowed(_));
    let t_upload = Instant::now();

    let gate_up_owner = gpu.upload_raw(gate_up_host.as_ref(), &[specs.len(), gate_up_stride])?;
    drop(gate_up_host);
    let down_owner = match gpu.upload_raw(down_host.as_ref(), &[specs.len(), down_stride]) {
        Ok(owner) => owner,
        Err(error) => {
            let _ = gpu.free_tensor(gate_up_owner);
            return Err(error);
        }
    };
    drop(down_host);
    let upload_ms = t_upload.elapsed().as_millis() as u64;
    let read_ms = t_read.elapsed().as_millis() as u64 - upload_ms;
    PACKED_READ_MS.fetch_add(read_ms, Ordering::Relaxed);
    PACKED_UPLOAD_MS.fetch_add(upload_ms, Ordering::Relaxed);
    if trace {
        eprintln!(
            "  [load-trace] packed experts: host-read {read_ms} ms, H2D {upload_ms} ms, zero-copy={zero_copy}"
        );
    }

    let mut experts = Vec::with_capacity(specs.len());
    for (slot, spec) in specs.iter().enumerate() {
        let mut gate_up = WeightTensor {
            buf: gate_up_owner.sub_offset(slot * gate_up_stride, gate_up_stride),
            gpu_dtype: DType::MQ4G256,
            m: 2 * mi,
            k: dim,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };
        gate_up.awq_scale = load_awq_scale_for(hfq, gpu, &spec.gate_up_name, dim);
        let mut down = WeightTensor {
            buf: down_owner.sub_offset(slot * down_stride, down_stride),
            gpu_dtype: DType::MQ4G256,
            m: dim,
            k: mi,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };
        down.awq_scale = load_awq_scale_for(hfq, gpu, &spec.down_name, mi);
        experts.push(ExpertWeights { gate_up, down });
    }

    Ok(Some((
        experts,
        PackedExpertOwners {
            gate_up: gate_up_owner,
            down: down_owner,
        },
    )))
}

/// AoS mfp4-E8 -> SoA byte transform (exact port of `aos_to_soa_full` in
/// bench_e8_soa_correctness). AoS row: [16B hdr][n_blocks×(1B scale + 16B cw)].
/// SoA row: [16B hdr (flag=0x06)][n_blocks scales, pad16][n_blocks×16B cw]. Same size.
fn e8_aos_to_soa(aos: &[u8], m: usize, k: usize) -> Vec<u8> {
    let n_blocks = k / 32;
    let aos_row = 16 + n_blocks * 17;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; m * soa_row];
    for r in 0..m {
        let src = &aos[r * aos_row..(r + 1) * aos_row];
        let dst = &mut out[r * soa_row..(r + 1) * soa_row];
        dst[..16].copy_from_slice(&src[..16]);
        dst[6] = 0x06; // SoA flag
        for b in 0..n_blocks {
            dst[16 + b] = src[16 + b * 17]; // E4M3 scales -> contiguous
        }
        let cw0 = 16 + scale_padded;
        for b in 0..n_blocks {
            let s = 16 + b * 17 + 1;
            let d = cw0 + b * 16;
            dst[d..d + 16].copy_from_slice(&src[s..s + 16]); // 16B codewords -> contiguous
        }
    }
    out
}

/// EP streaming-shard mode: when [`current_ep_expert_shard`] is `Some`, only the
/// rank's owned experts are read/allocated; the pointer tables are built global
/// `[n_exp]` with dummy pointers for non-owned slots (which contribute 0 to the
/// all-reduce because their gate_up is a zeroed buffer). Uniform files only —
/// graded/AWQ EP would need the full per-expert dtype map and is rejected here.
/// Escha-W2 expert storage. Production (Phase 2) is `Native` — the trellis
/// code itself, 0.25/0.375 B/weight, decoded inside the routed GEMV. The three
/// other values select DECODING stores, each of which exists to make a
/// specific measurement possible rather than to be run:
///
/// * `HIPFIRE_ESCHA_EXPERT_STORE=q8_0` (also `q8`) — Phase 1: transpose +
///   Q8_0 re-quantise, 1.0625 B/weight, 37.55 GB resident. This is the A/B arm
///   for every Phase-2 performance claim, and the arm every published Phase-1
///   number (G4's Q8_0 arm, the G5 KLD headline) was measured on. It is also
///   the only routed store that works on the per-expert HOST route, so it is
///   what `HIPFIRE_ESCHA_INDEXED=0` needs.
/// * `HIPFIRE_ESCHA_EXPERT_STORE=f16` — 2 B/weight, weight-exact, ~64 GB of
///   experts. The arm the G5 KLD reference is built with.
/// * `HIPFIRE_ESCHA_EXPERT_STORE=f32` — 4 B/weight, ~129 GB of experts on the
///   35B. Equally exact and does NOT fit; small-layer diagnostic only (the G4
///   block gate uses it).
///
/// `f16` and `f32` lose the indexed GPU-top-K path and run host-routed;
/// `native` REQUIRES it — there is no per-expert native GEMV, so an escha
/// layer that reaches the host route with this store fails loudly in
/// `GemvFamily::run_auto` (no plain GEMV exists for `RotationPlan::EschaH128`)
/// instead of running unrotated. See `qwen35/escha.rs`.
///
/// An unrecognised value falls through to production rather than erroring,
/// matching every other developer var in this loader.
fn escha_weight_store() -> EschaWeightStore {
    match hipfire_config::developer_var("HIPFIRE_ESCHA_EXPERT_STORE").as_deref() {
        Ok("f32") | Ok("F32") => EschaWeightStore::F32,
        Ok("f16") | Ok("F16") => EschaWeightStore::F16,
        Ok("q8_0") | Ok("Q8_0") | Ok("q8") | Ok("Q8") => EschaWeightStore::Q8_0,
        _ => EschaWeightStore::Native,
    }
}

pub(crate) fn load_moe_ffn(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    config: &Qwen35Config,
    layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let ep = config
        .reap_keep
        .as_ref()
        .map(|r| r.expert_plan(layer_idx as usize));
    // Detect Escha-W2 BEFORE the EP-shard block. An escha layer carries one
    // trellis code tensor per projection for all experts, not the per-expert
    // `experts.{x}.gate_up_proj.weight` tensors the EP path fishes out by
    // index — so if the EP block runs first it panics on a tensor that does
    // not exist, and the escha refusal further down is never reached. The
    // daemon sets an EP shard even on a single GPU, which is exactly how that
    // happened: `hipfire bench` panicked with
    // "tensor not found: layers.0.mlp.experts.0.gate_up_proj.weight".
    //
    // A single-rank shard splits nothing, so escha simply ignores it; only a
    // genuine multi-rank split is refused (below, and again after the router
    // load for the REAP keep-map case).
    let escha_layer = escha::layer_is_escha(hfq, p, qwen35_tensor_name_candidates);
    let ep_shard = current_ep_expert_shard();
    let ep_shard = match (escha_layer, ep_shard) {
        (true, Some((ref sc, _))) if sc.tp_size > 1 => {
            return Err(HipError::new(
                0,
                "qwen35: Escha-W2 routed experts do not support EP sharding across >1 rank                  (it re-maps experts across the per-expert tensors escha does not have)",
            ))
        }
        (true, _) => None,
        (_, other) => other,
    };
    if ep.is_some() && ep_shard.is_some() {
        return Err(HipError::new(
            0,
            "qwen35: REAP keep-map + EP sharding are mutually exclusive",
        ));
    }
    if let Some((shard, rank)) = ep_shard.clone() {
        let mut global_dtypes: Vec<(DType, DType)> = Vec::with_capacity(n_exp);
        let mut global_tags: Vec<u8> = Vec::with_capacity(n_exp);
        for slot in 0..n_exp {
            let orig = ep.as_ref().map(|e| e.src(slot)).unwrap_or(slot);
            let gate_bare = format!("{p}.mlp.experts.{orig}.gate_up_proj.weight");
            let down_bare = format!("{p}.mlp.experts.{orig}.down_proj.weight");
            let gate_info = qwen35_tensor_name_candidates(&gate_bare)
                .into_iter()
                .find_map(|name| hfq.find_tensor_info(&name).cloned())
                .ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!("qwen35: missing gate_up tensor for expert {orig} ({gate_bare})"),
                    )
                })?;
            let down_info = qwen35_tensor_name_candidates(&down_bare)
                .into_iter()
                .find_map(|name| hfq.find_tensor_info(&name).cloned())
                .ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!("qwen35: missing down tensor for expert {orig} ({down_bare})"),
                    )
                })?;
            if gate_info.shape != vec![(2 * mi) as u32, config.dim as u32] {
                return Err(HipError::new(
                    0,
                    &format!(
                        "qwen35: gate_up shape mismatch for expert {orig}: {:?} vs [{} {}]",
                        gate_info.shape,
                        2 * mi,
                        config.dim
                    ),
                ));
            }
            if down_info.shape != vec![config.dim as u32, mi as u32] {
                return Err(HipError::new(
                    0,
                    &format!(
                        "qwen35: down shape mismatch for expert {orig}: {:?} vs [{} {}]",
                        down_info.shape, config.dim, mi
                    ),
                ));
            }
            let gate_dtype = dtype_from_quant_type(gate_info.quant_type)?;
            let down_dtype = dtype_from_quant_type(down_info.quant_type)?;
            let tag = mixed_expert_tag(gate_dtype, down_dtype)?;
            let has_awq = {
                let candidates = [
                    format!("{p}.mlp.experts.{orig}.down_proj.awq_scale.weight"),
                    format!("{p}.mlp.experts.{orig}.down_proj.weight.awq_scale.weight"),
                ];
                candidates.iter().any(|n| hfq.find_tensor_info(n).is_some())
            };
            if has_awq {
                return Err(HipError::new(
                    0,
                    "AWQ MoE EP not yet supported (quantize experts without AWQ for EP serving)",
                ));
            }
            global_dtypes.push((gate_dtype, down_dtype));
            global_tags.push(tag);
        }
        for (name, m, k) in [
            (format!("{p}.mlp.gate.weight"), n_exp, config.dim),
            (
                format!("{p}.mlp.shared_expert.gate_proj.weight"),
                smi,
                config.dim,
            ),
            (
                format!("{p}.mlp.shared_expert.up_proj.weight"),
                smi,
                config.dim,
            ),
            (
                format!("{p}.mlp.shared_expert.down_proj.weight"),
                config.dim,
                smi,
            ),
            (format!("{p}.mlp.shared_expert_gate.weight"), 1, config.dim),
        ] {
            let info = qwen35_tensor_name_candidates(&name)
                .into_iter()
                .find_map(|n| hfq.find_tensor_info(&n).cloned())
                .ok_or_else(|| HipError::new(0, &format!("qwen35: missing tensor {name}")))?;
            if info.shape != vec![m as u32, k as u32]
                && !(name.contains("shared_expert_gate") && info.shape == vec![k as u32])
            {
                return Err(HipError::new(
                    0,
                    &format!(
                        "qwen35: shape mismatch for {name}: {:?} vs [{m} {k}]",
                        info.shape
                    ),
                ));
            }
        }
        if global_dtypes.len() != n_exp || global_tags.len() != n_exp {
            return Err(HipError::new(0, "qwen35: global MoE table length mismatch"));
        }
        let owned_ids: Vec<usize> = (0..n_exp)
            .filter(|&orig| shard.owns_expert(rank, orig))
            .collect();
        if owned_ids.is_empty() {
            return Err(HipError::new(
                0,
                &format!("qwen35: EP shard rank {rank} owns zero experts in layer {layer_idx}"),
            ));
        }
        let mut pending = PendingEpMoeFfn::new(layer_idx);
        pending.global_dtypes = Some(global_dtypes.clone().into_boxed_slice());
        let alloc_res: HipResult<(
            SharedExpertWeights,
            GpuTensor,
            GpuTensor,
            Option<GpuTensor>,
            Option<GpuTensor>,
        )> = (|| {
            let router = match ep.as_ref().and_then(|e| e.keep()) {
                Some(keep) => load_weight_tensor_keep(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate.weight"),
                    n_exp,
                    config.dim,
                    keep,
                )?,
                None => load_weight_tensor(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate.weight"),
                    n_exp,
                    config.dim,
                    qwen35_tensor_name_candidates,
                )?,
            };
            pending.router = Some(router);
            let gate = load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.shared_expert.gate_proj.weight"),
                smi,
                config.dim,
                qwen35_tensor_name_candidates,
            )?;
            pending.shared_gate = Some(gate);
            let up = load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.shared_expert.up_proj.weight"),
                smi,
                config.dim,
                qwen35_tensor_name_candidates,
            )?;
            pending.shared_up = Some(up);
            let down = load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.shared_expert.down_proj.weight"),
                config.dim,
                smi,
                qwen35_tensor_name_candidates,
            )?;
            pending.shared_down = Some(down);
            let scalar = load_weight_tensor(
                hfq,
                gpu,
                &format!("{p}.mlp.shared_expert_gate.weight"),
                1,
                config.dim,
                qwen35_tensor_name_candidates,
            )?;
            pending.shared_gate_scalar = Some(scalar);
            for &x in &owned_ids {
                let gate_up = load_weight_tensor(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.experts.{x}.gate_up_proj.weight"),
                    2 * mi,
                    config.dim,
                    qwen35_tensor_name_candidates,
                )?;
                let down = load_weight_tensor(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.experts.{x}.down_proj.weight"),
                    config.dim,
                    mi,
                    qwen35_tensor_name_candidates,
                )?;
                pending.experts.push(ExpertWeights { gate_up, down });
            }
            {
                use std::collections::BTreeMap;
                let mut gate_dummy_by_bytes: BTreeMap<usize, u64> = BTreeMap::new();
                let mut down_dummy_by_bytes: BTreeMap<usize, u64> = BTreeMap::new();
                for slot in 0..n_exp {
                    let orig = ep.as_ref().map(|e| e.src(slot)).unwrap_or(slot);
                    if shard.owns_expert(rank, orig) {
                        continue;
                    }
                    let gate_bare = format!("{p}.mlp.experts.{orig}.gate_up_proj.weight");
                    let down_bare = format!("{p}.mlp.experts.{orig}.down_proj.weight");
                    let gate_bytes = qwen35_tensor_name_candidates(&gate_bare)
                        .into_iter()
                        .find_map(|n| hfq.find_tensor_info(&n).map(|i| i.data_size))
                        .expect("prescan validated");
                    let down_bytes = qwen35_tensor_name_candidates(&down_bare)
                        .into_iter()
                        .find_map(|n| hfq.find_tensor_info(&n).map(|i| i.data_size))
                        .expect("prescan validated");
                    if !gate_dummy_by_bytes.contains_key(&gate_bytes) {
                        let t = gpu.zeros(&[gate_bytes / 4], DType::F32)?;
                        let ptr = t.buf.as_ptr() as u64;
                        pending.dummy_buffers.push(t);
                        gate_dummy_by_bytes.insert(gate_bytes, ptr);
                    }
                    if !down_dummy_by_bytes.contains_key(&down_bytes) {
                        let t = gpu.zeros(&[down_bytes / 4], DType::F32)?;
                        let ptr = t.buf.as_ptr() as u64;
                        pending.dummy_buffers.push(t);
                        down_dummy_by_bytes.insert(down_bytes, ptr);
                    }
                }
                let mut gu_ptrs = vec![0u64; n_exp];
                let mut dn_ptrs = vec![0u64; n_exp];
                let mut li = 0usize;
                for slot in 0..n_exp {
                    let orig = ep.as_ref().map(|e| e.src(slot)).unwrap_or(slot);
                    if shard.owns_expert(rank, orig) {
                        gu_ptrs[slot] = pending.experts[li].gate_up.buf.buf.as_ptr() as u64;
                        dn_ptrs[slot] = pending.experts[li].down.buf.buf.as_ptr() as u64;
                        li += 1;
                    } else {
                        let gate_bare = format!("{p}.mlp.experts.{orig}.gate_up_proj.weight");
                        let down_bare = format!("{p}.mlp.experts.{orig}.down_proj.weight");
                        let gate_bytes = qwen35_tensor_name_candidates(&gate_bare)
                            .into_iter()
                            .find_map(|n| hfq.find_tensor_info(&n).map(|i| i.data_size))
                            .unwrap();
                        let down_bytes = qwen35_tensor_name_candidates(&down_bare)
                            .into_iter()
                            .find_map(|n| hfq.find_tensor_info(&n).map(|i| i.data_size))
                            .unwrap();
                        gu_ptrs[slot] = *gate_dummy_by_bytes.get(&gate_bytes).unwrap();
                        dn_ptrs[slot] = *down_dummy_by_bytes.get(&down_bytes).unwrap();
                    }
                }
                let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
                let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
                let gt = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
                let dt = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
                gpu.hip.memcpy_htod(&gt.buf, &gu_bytes)?;
                gpu.hip.memcpy_htod(&dt.buf, &dn_bytes)?;
                pending.gate_up_ptrs = Some(gt);
                pending.down_ptrs = Some(dt);
            }
            let awq_ptrs: Option<GpuTensor> = None;
            let dtype_tags: Option<GpuTensor> = {
                let dtypes = pending.global_dtypes.as_ref().unwrap();
                let gate0 = dtypes[0].0;
                let down0 = dtypes[0].1;
                let mixed = dtypes.iter().any(|(g, d)| *g != gate0 || *d != down0);
                if mixed {
                    let t = gpu.alloc_tensor(&[n_exp], DType::Raw)?;
                    gpu.hip.memcpy_htod(&t.buf, &global_tags)?;
                    Some(t)
                } else {
                    None
                }
            };
            // Move ownership into pending for transactional rollback; return via take
            pending.dtype_tags = dtype_tags;
            pending.awq_ptrs = awq_ptrs;
            let shared_expert = SharedExpertWeights {
                gate: pending.shared_gate.take().unwrap(),
                up: pending.shared_up.take().unwrap(),
                down: pending.shared_down.take().unwrap(),
            };
            Ok((
                shared_expert,
                pending.gate_up_ptrs.take().unwrap(),
                pending.down_ptrs.take().unwrap(),
                pending.awq_ptrs.take(),
                pending.dtype_tags.take(),
            ))
        })();
        match alloc_res {
            Ok((shared_expert, gu_ptrs, dn_ptrs, awq_ptrs, dtype_tags)) => {
                let router = pending.router.take().expect("router");
                let scalar = pending.shared_gate_scalar.take().expect("scalar");
                let mut commit_pending = PendingEpMoeFfn::new(layer_idx);
                commit_pending.router = Some(router);
                commit_pending.shared_gate_scalar = Some(scalar);
                commit_pending.experts = pending.experts;
                commit_pending.packed_owners = pending.packed_owners;
                commit_pending.dummy_buffers = pending.dummy_buffers;
                commit_pending.global_dtypes = pending.global_dtypes;
                return Ok(commit_pending.commit(
                    shared_expert,
                    gu_ptrs,
                    dn_ptrs,
                    awq_ptrs,
                    dtype_tags,
                ));
            }
            Err(e) => return Err(pending.rollback(gpu, e)),
        }
    }
    let router = match ep.as_ref().and_then(|e| e.keep()) {
        Some(keep) => load_weight_tensor_keep(
            hfq,
            gpu,
            &format!("{p}.mlp.gate.weight"),
            n_exp,
            config.dim,
            keep,
        )?,
        None => load_weight_tensor(
            hfq,
            gpu,
            &format!("{p}.mlp.gate.weight"),
            n_exp,
            config.dim,
            qwen35_tensor_name_candidates,
        )?,
    };
    let shared_expert = SharedExpertWeights {
        gate: load_weight_tensor(
            hfq,
            gpu,
            &format!("{p}.mlp.shared_expert.gate_proj.weight"),
            smi,
            config.dim,
            qwen35_tensor_name_candidates,
        )?,
        up: load_weight_tensor(
            hfq,
            gpu,
            &format!("{p}.mlp.shared_expert.up_proj.weight"),
            smi,
            config.dim,
            qwen35_tensor_name_candidates,
        )?,
        down: load_weight_tensor(
            hfq,
            gpu,
            &format!("{p}.mlp.shared_expert.down_proj.weight"),
            config.dim,
            smi,
            qwen35_tensor_name_candidates,
        )?,
    };
    let shared_expert_gate = load_weight_tensor(
        hfq,
        gpu,
        &format!("{p}.mlp.shared_expert_gate.weight"),
        1,
        config.dim,
        qwen35_tensor_name_candidates,
    )?;
    let owns_orig = |x: usize| {
        ep_shard
            .as_ref()
            .map_or(true, |(sh, r)| sh.owns_expert(*r, x))
    };
    let expert_ids: Vec<usize> = (0..n_exp)
        .map(|slot| ep.as_ref().map(|e| e.src(slot)).unwrap_or(slot))
        .filter(|&x| owns_orig(x))
        .collect();
    // ── Escha-W2 routed experts (Task 10) ────────────────────────────────
    // An Escha-W2 layer carries ONE trellis code tensor per projection for
    // all experts, not the per-expert `experts.{x}.gate_up_proj.weight`
    // tensors every other path fishes out by index, so it bypasses both the
    // packed-MQ4 fast path and the generic per-expert loop below.
    if escha_layer && (ep_shard.is_some() || ep.is_some()) {
        return Err(HipError::new(
            0,
            "qwen35: Escha-W2 routed experts do not support EP sharding or a REAP keep-map \
             (both re-map experts across the per-expert tensors escha does not have)",
        ));
    }
    let escha_tables = if escha_layer {
        let store = escha_weight_store();
        let (experts, tables, owners) = escha::load_escha_moe_experts(
            hfq,
            gpu,
            p,
            &expert_ids,
            n_exp,
            config.dim,
            mi,
            config.num_experts_per_tok,
            store,
            qwen35_tensor_name_candidates,
        )?;
        if layer_idx == 0 {
            eprintln!(
                "  Escha-W2 routed experts: {} experts, store {store:?} ({}), {} per-expert \
                 weight buffers -> 2 layer blobs",
                experts.len(),
                match store {
                    EschaWeightStore::Native => "trellis code kept verbatim, decoded in the GEMV",
                    _ => "decoded from the trellis at load",
                },
                2 * experts.len()
            );
        }
        Some((experts, tables, owners))
    } else {
        None
    };

    let packed = if !escha_layer && ep_shard.is_none() && packed_mq4_experts_supported(gpu) {
        try_load_packed_mq4_experts(hfq, gpu, p, &expert_ids, mi, config.dim)?
    } else {
        None
    };
    let (mut experts, packed_expert_owners, escha_tables) =
        if let Some((e, t, owners)) = escha_tables {
            // Escha expert slots are views into `owners`, exactly like the packed
            // MQ4 path's — so they ride the SAME `packed_expert_owners` free path
            // (`free_moe_ffn` frees per-expert metadata only, then the two blobs).
            // Publishing them here rather than in a bespoke field is what keeps
            // teardown from double-freeing or leaking 32 GB.
            (e, Some(owners), Some(t))
        } else if let Some((experts, owners)) = packed {
            if layer_idx == 0 {
                eprintln!(
                    "  routed MQ4 expert packing: {} per-expert weight buffers -> 2 layer blobs",
                    2 * experts.len()
                );
            }
            (experts, Some(owners), None)
        } else {
            let mut experts = Vec::with_capacity(expert_ids.len());
            for x in expert_ids {
                let gate_up = load_weight_tensor(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.experts.{x}.gate_up_proj.weight"),
                    2 * mi,
                    config.dim,
                    qwen35_tensor_name_candidates,
                )?;
                let down = load_weight_tensor(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.experts.{x}.down_proj.weight"),
                    config.dim,
                    mi,
                    qwen35_tensor_name_candidates,
                )?;
                experts.push(ExpertWeights { gate_up, down });
            }
            (experts, None, None)
        };
    if e8_soa_experts() && gpu.arch_caps.is_rdna3_dgpu() && ep_shard.is_none() {
        let mut converted = 0usize;
        for ew in experts.iter_mut() {
            if ew.gate_up.gpu_dtype == DType::MFP4G32E8 {
                let (m, k) = (ew.gate_up.m, ew.gate_up.k);
                let nbytes = ew.gate_up.buf.buf.size();
                let mut aos = vec![0u8; nbytes];
                gpu.hip.memcpy_dtoh(&mut aos, &ew.gate_up.buf.buf)?;
                let soa = e8_aos_to_soa(&aos, m, k);
                if soa.len() == nbytes {
                    gpu.hip.memcpy_htod(&ew.gate_up.buf.buf, &soa)?;
                    converted += 1;
                } else if layer_idx == 0 {
                    eprintln!(
                        "  [e8-soa] SKIP: SoA size {} != AoS {} (n_blocks%16!=0) — keeping AoS",
                        soa.len(),
                        nbytes
                    );
                }
            }
        }
        if converted > 0 && layer_idx == 0 {
            eprintln!("  [e8-soa] transposed {converted} gate_up experts AoS->SoA (per layer)");
        }
    }
    let mut gu_ptrs = vec![0u64; n_exp];
    let mut dn_ptrs = vec![0u64; n_exp];
    let ep_dummy_buffers: Vec<GpuTensor> = Vec::new();
    for (e, ew) in experts.iter().enumerate() {
        gu_ptrs[e] = ew.gate_up.buf.buf.as_ptr() as u64;
        dn_ptrs[e] = ew.down.buf.buf.as_ptr() as u64;
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)?;
    let moe_awq_enabled = hipfire_config::developer_var("HIPFIRE_MOE_AWQ")
        .ok()
        .as_deref()
        != Some("0");
    let awq_present = experts
        .iter()
        .filter(|e| e.down.awq_scale.is_some())
        .count();
    let expert_down_awq_ptrs = if moe_awq_enabled && n_exp > 0 && awq_present == n_exp {
        let aw_ptrs: Vec<u64> = experts
            .iter()
            .map(|e| e.down.awq_scale.as_ref().unwrap().buf.as_ptr() as u64)
            .collect();
        let aw_bytes: Vec<u8> = aw_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
        let t = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
        gpu.hip.memcpy_htod(&t.buf, &aw_bytes)?;
        Some(t)
    } else {
        if awq_present != 0 {
            eprintln!(
                "[moe-awq] layer {layer_idx}: partial down.awq_scale coverage ({awq_present}/{n_exp}) — disabling MoE-AWQ for this layer"
            );
        }
        None
    };
    let expert_dtype_tags = if n_exp > 0 {
        let gu0 = experts[0].gate_up.gpu_dtype;
        let dn0 = experts[0].down.gpu_dtype;
        let mixed = experts.iter().any(|e| e.gate_up.gpu_dtype != gu0)
            || experts.iter().any(|e| e.down.gpu_dtype != dn0);
        if mixed
            && experts.iter().any(|e| {
                matches!(e.gate_up.gpu_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
                    || matches!(e.down.gpu_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
            })
        {
            return Err(HipError::new(
                0,
                "graded (mixed-dtype) MoE with MQ2/MQ3-G256-GL experts is not supported: the merged dtype-tag decode kernel has no GL branch. Use a UNIFORM GL file (all routed experts the same GL dtype per projection).",
            ));
        }
        if mixed {
            for e in &experts {
                mixed_expert_tag(e.gate_up.gpu_dtype, e.down.gpu_dtype).map_err(|err| {
                    HipError::new(
                        0,
                        &format!("qwen35: expert unsupported tag: {}", err.message),
                    )
                })?;
            }
            let tags: Vec<u8> = experts
                .iter()
                .map(|e| mixed_expert_tag(e.gate_up.gpu_dtype, e.down.gpu_dtype).unwrap())
                .collect();
            let t = gpu.alloc_tensor(&[n_exp], DType::Raw)?;
            gpu.hip.memcpy_htod(&t.buf, &tags)?;
            Some(t)
        } else {
            None
        }
    } else {
        None
    };
    Ok(MoeFfnWeights {
        router,
        experts,
        packed_expert_owners,
        shared_expert,
        shared_expert_gate,
        expert_gate_up_ptrs,
        expert_down_ptrs,
        expert_down_awq_ptrs,
        expert_dtype_tags,
        layer_idx,
        expert_shape: None,
        paro_shared: None,
        global_expert_dtypes: None,
        ep_dummy_buffers,
        escha: escha_tables,
    })
}

/// Transpose an escha code blob's TILE GRID from the checkpoint's
/// `[ic/16][oc/16]` (kt-major) to `[oc/16][ic/16]` (nt-major). See the call
/// site. Moves whole tiles only, so it is bit-exact by construction.
fn escha_tiles_to_nt_major(data: &[u8], m: usize, k: usize, quant_type: u8) -> HipResult<Vec<u8>> {
    let tk = if quant_type == 42 { 2usize } else { 3usize };
    if m % 16 != 0 || k % 16 != 0 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("escha code: m={m} k={k}; both must be multiples of 16"),
        ));
    }
    let (ktiles, ntiles) = (k / 16, m / 16);
    let tile_bytes = 16 * tk * 2;
    let grid = ktiles * ntiles * tile_bytes;
    if grid == 0 || data.len() % grid != 0 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "escha code: {} bytes is not a whole number of {grid}-byte grids \
                 (ktiles={ktiles} ntiles={ntiles})",
                data.len()
            ),
        ));
    }
    let mut out = vec![0u8; data.len()];
    for e in 0..(data.len() / grid) {
        let base = e * grid;
        for kt in 0..ktiles {
            for nt in 0..ntiles {
                let src = base + (kt * ntiles + nt) * tile_bytes;
                let dst = base + (nt * ktiles + kt) * tile_bytes;
                out[dst..dst + tile_bytes].copy_from_slice(&data[src..src + tile_bytes]);
            }
        }
    }
    Ok(out)
}
