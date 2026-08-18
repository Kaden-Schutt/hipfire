// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 weight loading: HFQ / ParoQuant sources, AWQ repack, packed-MQ4
//! experts, `load_weights`, and the EP sharded loader.

use super::config::f16_lm_head_mode_from_config;
use super::config::F16LmHeadMode;
use super::config::Qwen35Config;
use super::forward::layers_have_mq6_moe;
use super::weights::dtype_from_quant_type;
use super::weights::mixed_expert_tag;
use super::weights::ExpertWeights;
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

fn qwen35_tensor_name_candidates(name: &str) -> Vec<String> {
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
    // Use pread path to avoid page cache buildup on unified-memory APUs.
    #[cfg(unix)]
    {
        let mut wt: Option<WeightTensor> = None;
        let mut matched: Option<String> = None;
        for candidate in candidates(name) {
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
    let LoadedWeights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        lm_head_aliases_embd,
    } = rt_load_weights(source, devices, layout)?;
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
        #[cfg(unix)]
        if n_devices == 1 {
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
        let (embd_meta, embd_data) = qwen35_tensor_data_vec(self.hfq, "embed_tokens.weight")
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
                    qwen35_tensor_data_vec(hfq, "lm_head.weight").expect("lm_head present");
                load_weight_tensor_raw(gpu, lm_info.quant_type, &lm_data, c.vocab_size, c.dim)
            },
            |gpu| {
                let (embd_meta, embd_data) = qwen35_tensor_data_vec(hfq, "embed_tokens.weight")
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
fn try_load_packed_mq4_experts(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    expert_ids: &[usize],
    mi: usize,
    dim: usize,
) -> HipResult<Option<(Vec<ExpertWeights>, PackedExpertOwners)>> {
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
    let (gate_up_host, down_host) = if hfq.has_overlay() {
        // Overlay offsets belong to a second file; retain the overlay-aware
        // serial path exactly rather than crossing files in reader workers.
        let mut gate_up_host = Vec::with_capacity(gate_up_stride * specs.len());
        let mut down_host = Vec::with_capacity(down_stride * specs.len());
        for spec in &specs {
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
                gate_up_host.extend_from_slice(&bytes);
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
                down_host.extend_from_slice(&bytes);
            }
        }
        (gate_up_host, down_host)
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
            results.next().expect("two packed MQ4 jobs").data,
            results.next().expect("two packed MQ4 jobs").data,
        )
    };

    let gate_up_owner = gpu.upload_raw(&gate_up_host, &[specs.len(), gate_up_stride])?;
    drop(gate_up_host);
    let down_owner = match gpu.upload_raw(&down_host, &[specs.len(), down_stride]) {
        Ok(owner) => owner,
        Err(error) => {
            let _ = gpu.free_tensor(gate_up_owner);
            return Err(error);
        }
    };
    drop(down_host);

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
    let ep_shard = current_ep_expert_shard();
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
    let packed = if ep_shard.is_none() && packed_mq4_experts_supported(gpu) {
        try_load_packed_mq4_experts(hfq, gpu, p, &expert_ids, mi, config.dim)?
    } else {
        None
    };
    let (mut experts, packed_expert_owners) = if let Some((experts, owners)) = packed {
        if layer_idx == 0 {
            eprintln!(
                "  routed MQ4 expert packing: {} per-expert weight buffers -> 2 layer blobs",
                2 * experts.len()
            );
        }
        (experts, Some(owners))
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
        (experts, None)
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
    })
}
