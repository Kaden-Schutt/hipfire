// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arch-agnostic weight loading: dequant primitives, HF tensor-name resolution,
//! and a `WeightBackend` trait abstracting HFQ vs ParoQuant on-disk formats.
//! Per-arch crates build their `load_layer` schema on top of this; the only
//! arch-varying knobs are the RMSNorm `+bias` and the name-candidate resolver.

use crate::hfq::HfqFile;
use crate::llama::{f16_to_f32, EmbeddingFormat, KvCache, WeightTensor};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Widen a little-endian BF16 byte stream to F32 (lossless: bf16 is the high
/// 16 bits of an f32). Used by the qt=16 paths in dequant_weight_raw/dequant_f32.
fn widen_bf16(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

// ── HF tensor-name resolution ───────────────────────────────────────────────

/// Candidate on-disk names for a logical tensor, covering the HF nested
/// vision-wrapper layout (`model.language_model.*`), the flat layout (`model.*`),
/// and the bare name, plus the `lm_head` special-case. Layout convention only —
/// not model-specific math — so any HF text tower can share it.
pub fn hf_name_candidates(name: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(4);
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

/// Flat-only resolver for arches stored without the vision-wrapper nesting
/// (qwen2, llama). Tries `model.{name}` then bare.
pub fn flat_name_candidates(name: &str) -> Vec<String> {
    if name.starts_with("model.") {
        vec![name.to_string()]
    } else {
        vec![format!("model.{name}"), name.to_string()]
    }
}

// ── Layer-relative name builders ────────────────────────────────────────────

/// HFQ projection name: `layers.{layer}.{rel}.weight` (the backend's candidate
/// resolver then adds any layout prefix).
pub fn hfq_proj_name(layer: usize, rel: &str) -> String {
    format!("layers.{layer}.{rel}.weight")
}
/// HFQ norm / raw-f32 name: `layers.{layer}.{rel}` (rel already carries `.weight`
/// where the on-disk tensor has it, e.g. `input_layernorm.weight`).
pub fn hfq_plain_name(layer: usize, rel: &str) -> String {
    format!("layers.{layer}.{rel}")
}
/// PaRo projection base (augmentor appends `.qweight`/`.weight`): `{mp}.layers.{layer}.{rel}`.
pub fn paro_proj_name(mp: &str, layer: usize, rel: &str) -> String {
    format!("{mp}.layers.{layer}.{rel}")
}
/// PaRo norm/raw-f32 name for `paro_load_norm`/`paro_load_f32`, which prepend `mp`
/// THEMSELVES — so this is prefix-LESS: `layers.{layer}.{rel}`.
pub fn paro_plain_name(layer: usize, rel: &str) -> String {
    format!("layers.{layer}.{rel}")
}

// ── Embedding / tied-lm_head primitives ──────────────────────────────────

/// How an embedding table's on-disk bytes map to the device.
#[derive(Debug)]
pub enum EmbedPlan {
    /// Upload bytes verbatim; the lookup kernel dequantizes on the fly.
    Raw(EmbeddingFormat),
    /// Host-decode to f32 (via `dequant_f32`) then upload as F32.
    HostF32,
}

/// Pure quant_type → plan. GPU-free, unit-testable.
///
/// qt 6 → Raw(HFQ4G256), 7 → Raw(HFQ4G128), 3 → Raw(Q8_0),
/// qt 1|2|16 → HostF32, else → panic with the supported-format list.
pub fn embed_classify(quant_type: u8) -> HipResult<EmbedPlan> {
    match quant_type {
        6 => Ok(EmbedPlan::Raw(EmbeddingFormat::HFQ4G256)),
        7 => Ok(EmbedPlan::Raw(EmbeddingFormat::HFQ4G128)),
        3 => Ok(EmbedPlan::Raw(EmbeddingFormat::Q8_0)),
        1 | 2 | 16 => Ok(EmbedPlan::HostF32),
        other => Err(hip_bridge::HipError::new(
            0,
            &format!(
                "unsupported embedding quant_type {other}; \
                 handled: 1 (F16→F32), 2 (F32), 3 (Q8_0), 6 (HFQ4G256), 7 (HFQ4G128), 16 (BF16→F32). \
                 Add the format to embed_classify to support it."
            ),
        )),
    }
}

/// Load an embedding table to the device. Unifies the qwen35 and qwen2
/// hand-written matches. Returns the device tensor + its on-GPU format.
pub fn load_embedding(
    gpu: &mut Gpu,
    quant_type: u8,
    data: &[u8],
    vocab: usize,
    dim: usize,
) -> HipResult<(GpuTensor, EmbeddingFormat)> {
    match embed_classify(quant_type)? {
        EmbedPlan::Raw(fmt) => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok((buf, fmt))
        }
        EmbedPlan::HostF32 => {
            // dequant_f32 uploads with shape [n] (1D). The embedding-lookup
            // kernels compute byte offsets from token_id + dim against buf
            // directly and never read the shape, so the 1D vs 2D difference
            // is behaviorally identical.
            let buf = dequant_f32(gpu, quant_type, data, vocab * dim)?;
            Ok((buf, EmbeddingFormat::F32))
        }
    }
}

/// EmbeddingFormat → the DType tag for a tied lm_head WeightTensor.
/// Replaces both arches' inline matches. Q4K is not a valid tied format → panic.
pub fn embedding_format_dtype(fmt: EmbeddingFormat) -> DType {
    match fmt {
        EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
        EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
        EmbeddingFormat::Q8_0 => DType::Q8_0,
        EmbeddingFormat::F32 => DType::F32,
        EmbeddingFormat::Q4K => panic!("embedding_format_dtype: Q4K not valid for tied lm_head"),
    }
}

/// Load an AWQ sidecar tensor from an HFQ file.
///
/// Looks up `{stem}.awq_scale.weight` where `name` is `{stem}.weight`.
/// Returns `None` when no sidecar exists or when the sidecar has an
/// unexpected quant_type/shape.
///
/// Moved from `hipfire-arch-qwen35::qwen35::load_awq_scale_for`.
pub fn load_awq_scale_for(hfq: &HfqFile, gpu: &Gpu, name: &str, k: usize) -> Option<GpuTensor> {
    let sidecar_name = match name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{name}.awq_scale.weight"),
    };
    let (sc_info, sc_data) = hfq.tensor_data_pread(&sidecar_name)?;
    // Must be 1D F16, length K. quant_type 1 = F16.
    if sc_info.quant_type != 1 {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} has quant_type={} (expected 1=F16); skipping",
            sc_info.quant_type
        );
        return None;
    }
    if sc_info.shape.len() != 1 || sc_info.shape[0] as usize != k {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} shape mismatch ({:?} vs expected [{}]); skipping",
            sc_info.shape, k
        );
        return None;
    }
    // F16 → F32 on host so the kernel takes a plain `const float*`.
    let f32_data: Vec<f32> = sc_data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    gpu.upload_raw(&f32_bytes, &[f32_bytes.len()]).ok()
}

/// Decode a little-endian f16 byte buffer to `f32`. Pure (GPU-free). The shared
/// core of every tied-lm_head reupload across HFQ + ParoQuant arches.
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Reupload an f16 weight buffer as a device F32 `WeightTensor [m, k]`. The
/// canonical tied-lm_head / ParoQuant-output reupload — replaces three hand-rolled
/// `unsafe { from_raw_parts }` copies.
pub fn reupload_f16_as_f32(
    gpu: &mut Gpu,
    f16_bytes: &[u8],
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let f32_data = f16_bytes_to_f32(f16_bytes);
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

/// Build a tied lm_head `WeightTensor` that ALIASES the embedding device
/// buffer (`shallow_clone` — a non-owning view). The owning weights struct
/// must record the alias (`lm_head_aliases_embd` / `tied_lm_head`) so the
/// embedding buffer is freed exactly once and the view is never freed.
/// Panics via `embedding_format_dtype` on Q4K (no tied-lm_head / GEMV weight
/// path for Q4K).
pub fn tied_lm_head_alias(
    embd: &GpuTensor,
    embd_fmt: EmbeddingFormat,
    m: usize,
    k: usize,
) -> WeightTensor {
    WeightTensor {
        buf: embd.shallow_clone(),
        gpu_dtype: embedding_format_dtype(embd_fmt),
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    }
}

/// Resolve the output / lm_head weight, returning `(output, aliases_embd)`.
/// `aliases_embd == true` iff `output.buf` is a view of the embedding buffer
/// and must NOT be freed by the owning struct.
///
/// - `has_separate`        → `(load_separate(gpu)?, false)` — a distinct
///   `lm_head.weight` (or `output.weight`) tensor exists on disk.
/// - else `can_alias`      → `(tied_lm_head_alias(...), true)` — single device:
///   share the embedding buffer.
/// - else (multi-GPU)      → `(reupload_tied(gpu)?, false)` — output lives on a
///   different device than embed, so re-materialize it.
///
/// `gpu` is threaded into whichever closure runs (only one is called), so
/// neither closure needs to capture `&mut Gpu`.
pub fn resolve_lm_head<S, R>(
    gpu: &mut Gpu,
    has_separate: bool,
    can_alias: bool,
    embd: &GpuTensor,
    embd_fmt: EmbeddingFormat,
    m: usize,
    k: usize,
    load_separate: S,
    reupload_tied: R,
) -> HipResult<(WeightTensor, bool)>
where
    S: FnOnce(&mut Gpu) -> HipResult<WeightTensor>,
    R: FnOnce(&mut Gpu) -> HipResult<WeightTensor>,
{
    if has_separate {
        eprintln!("  loading output (separate lm_head)...");
        Ok((load_separate(gpu)?, false))
    } else if can_alias {
        eprintln!("  loading output (tied embeddings, aliased)...");
        Ok((tied_lm_head_alias(embd, embd_fmt, m, k), true))
    } else {
        eprintln!("  loading output (tied embeddings, reupload)...");
        Ok((reupload_tied(gpu)?, false))
    }
}

// ── Dequant primitives ───────────────────────────────────────────────

// ── Raw-passthrough quant codec registry ─────────────────────────────────────
//
// Single source of truth mapping the `.hfq` wire byte `quant_type` → compute
// `DType`, for formats whose load is a verbatim byte upload
// (`upload_raw(data, &[data.len()])`) + a dtype tag. Consumed by both
// `dequant_weight_raw` and `hfq::load_weight_tensor`. Layout facts (row_stride,
// K%256 guard) live on `DType`, not here. Formats that host-decode (qt 1, 2, 16)
// are deliberately absent — they stay explicit in their consumer.
//
// WEIGHT-DECODE ONLY. Embedding tables are NOT routed here (different output
// type + Q4K divergence); see embed_classify / load_embedding_llama.
//
// Adding a passthrough format = one row here, plus (if it has a non-trivial
// stride or K constraint) the matching arm in DType::row_stride /
// DType::requires_k_mod_256, plus (if the quantizer emits sidecars) one line in
// DType::supports_awq_sidecar.

/// One passthrough quant format: upload bytes verbatim, tag `dtype`.
pub(crate) struct RawCodec {
    pub quant_type: u8,
    pub dtype: DType,
}

/// The registry. Order is irrelevant (lookup is by quant_type); ascending for
/// readability. qt 1/2/16 are intentionally absent (host-decode, see consumers).
pub(crate) const RAW_CODECS: &[RawCodec] = &[
    RawCodec {
        quant_type: 0,
        dtype: DType::Q4F16G64,
    },
    RawCodec {
        quant_type: 3,
        dtype: DType::Q8_0,
    },
    RawCodec {
        quant_type: 4,
        dtype: DType::Q4K,
    },
    RawCodec {
        quant_type: 5,
        dtype: DType::Q8HFQ,
    },
    RawCodec {
        quant_type: 6,
        dtype: DType::HFQ4G256,
    },
    RawCodec {
        quant_type: 7,
        dtype: DType::HFQ4G128,
    },
    RawCodec {
        quant_type: 8,
        dtype: DType::HFQ6G256,
    },
    RawCodec {
        quant_type: 9,
        dtype: DType::HFQ2G256,
    },
    RawCodec {
        quant_type: 10,
        dtype: DType::HFQ2G128,
    },
    RawCodec {
        quant_type: 11,
        dtype: DType::HFQ3G256,
    },
    RawCodec {
        quant_type: 12,
        dtype: DType::HFQ3G128,
    },
    RawCodec {
        quant_type: 13,
        dtype: DType::MQ4G256,
    },
    RawCodec {
        quant_type: 14,
        dtype: DType::MQ8G256,
    },
    RawCodec {
        quant_type: 15,
        dtype: DType::MQ6G256,
    },
    RawCodec {
        quant_type: 17,
        dtype: DType::MQ3G256,
    },
    RawCodec {
        quant_type: 18,
        dtype: DType::MQ2G256,
    },
    RawCodec {
        quant_type: 19,
        dtype: DType::MQ2G256Lloyd,
    },
    RawCodec {
        quant_type: 20,
        dtype: DType::MQ3G256Lloyd,
    },
    RawCodec {
        quant_type: 21,
        dtype: DType::HFP4G32,
    },
    RawCodec {
        quant_type: 24,
        dtype: DType::MFP4G32,
    },
    RawCodec {
        quant_type: 30,
        dtype: DType::MQ4G256Lloyd,
    },
    // MQ2/MQ3-G256-GL ("global Lloyd"): 2- resp. 3-bit codes against ONE
    // tensor-global codebook plus a per-block fp16 scale, stored SoA as
    // `[M*gpr*IDX B indices][M*gpr*2 B scales]` (IDX = 64 / 96). Passthrough
    // like every other MQ*: the bytes go to the GPU verbatim and the indexed
    // MoE GEMVs decode them. The codebook is NOT in the file — the runtime
    // supplies `rdna_compute::GL_CB2` / `GL_CB3` as scalar kernel args, which
    // MUST match the quantizer's constants (see their doc comments).
    // `decode_raw_codec` enforces K%256==0 for both via
    // `DType::requires_k_mod_256` — `gpr = K/256` sets the scale-region base,
    // so a bad K silently corrupts rather than erroring.
    RawCodec {
        quant_type: 38,
        dtype: DType::MQ2G256GL,
    },
    RawCodec {
        quant_type: 39,
        dtype: DType::MQ3G256GL,
    },
    // qt 40 (MQ4-G256-GL). Its 2-bit and 3-bit siblings above were registered but
    // qt 40 was not, so `dequant_weight_raw` rejected the format with
    // "unsupported quant_type 40" at load time even though the GEMV dispatch,
    // the encoder and `DType::{row_stride, requires_k_mod_256}` all handled it.
    // A qt=40 artifact was therefore unloadable and unscoreable.
    RawCodec {
        quant_type: 40,
        dtype: DType::MQ4G256GL,
    },
    RawCodec {
        quant_type: 43,
        dtype: DType::MQ4G256SEL,
    },
];
/// Look up the passthrough codec for `quant_type`, or `None` if it is host-decode
/// (1/2/16) or genuinely unsupported.
pub(crate) fn raw_codec(quant_type: u8) -> Option<&'static RawCodec> {
    RAW_CODECS.iter().find(|c| c.quant_type == quant_type)
}

/// Decode a passthrough quant format: enforce the K%256 guard (via DType),
/// upload bytes verbatim, build the `WeightTensor` with the dtype + its
/// DType-derived row_stride. `name` is the caller context for the guard panic.
/// AWQ sidecars are attached by the caller (hfq), never here.
pub(crate) fn decode_raw_codec(
    gpu: &Gpu,
    codec: &RawCodec,
    data: &[u8],
    m: usize,
    k: usize,
    name: &str,
) -> HipResult<WeightTensor> {
    if codec.dtype.requires_k_mod_256() && k % 256 != 0 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "{:?} tensor has K={k} but kernel requires K%256==0 (caller: {name})",
                codec.dtype
            ),
        ));
    }
    if codec.dtype == DType::MQ4G256SEL {
        // AoS 132 B/group = 128 idx + 2 scale + 1 selector + 1 pad.
        // Selector 0..63 valid, 64..255 invalid — reject rather than mask.
        // Pad byte MUST be 0.
        let gpr = k / 256;
        let expected = m * gpr * 132;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MQ4G256SEL blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name})",
                    data.len()
                ),
            ));
        }
        for (gi, chunk) in data.chunks_exact(132).enumerate() {
            let sel = chunk[130];
            if sel >= 64 {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "MQ4G256SEL invalid selector {sel} at group {gi} (must be 0..63) in {name}"
                    ),
                ));
            }
            if chunk[131] != 0 {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "MQ4G256SEL pad byte non-zero {} at group {gi} in {name}",
                        chunk[131]
                    ),
                ));
            }
        }
    }
    let buf = gpu.upload_raw(data, &[data.len()])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: codec.dtype,
        m,
        k,
        row_stride: codec.dtype.row_stride(k),
        paro: None,
        awq_scale: None,
    })
}

/// Quant `data` → device `WeightTensor [m, k]`. Moved from
/// `hipfire-arch-qwen35::qwen35::load_weight_tensor_raw` (Task 2).
pub fn dequant_weight_raw(
    gpu: &Gpu,
    quant_type: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    // Host-decode formats stay explicit (NOT passthrough table rows):
    match quant_type {
        1 => {
            // F16 — keep as F16 bytes (the HFQ path host-decodes qt 1 to F32 instead;
            // this divergence is why qt 1 is not a RAW_CODECS row).
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F16,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        2 => {
            // F32 — upload as [m, k].
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
            // bf16 is the high 16 bits of an f32, so widening is lossless/exact.
            let f32_data = widen_bf16(data);
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
        other => match raw_codec(other) {
            Some(c) => decode_raw_codec(gpu, c, data, m, k, "dequant_weight_raw"),
            None => Err(hip_bridge::HipError::new(
                0,
                &format!("unsupported quant_type {other} for dequant_weight_raw"),
            )),
        },
    }
}

/// RMSNorm scale `data` → device `GpuTensor [shape]`, adding `bias` to every
/// element (`1.0` for qwen3.5/gemma, `0.0` for qwen2/llama/minimax). Moved from
/// `load_norm_weight` (Task 2), with the `+= 1.0` generalised to `+= bias`.
pub fn dequant_norm(
    gpu: &mut Gpu,
    quant_type: u8,
    data: &[u8],
    shape: &[usize],
    bias: f32,
) -> HipResult<GpuTensor> {
    let mut f32_data: Vec<f32> = match quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => widen_bf16(data),
        _ => panic!("expected F16/F32/BF16 for norm, got qt={quant_type}"),
    };
    let expected: usize = shape.iter().product();
    assert_eq!(
        f32_data.len(),
        expected,
        "dequant_norm: tensor has {} elements, expected {expected} (shape {shape:?})",
        f32_data.len()
    );
    for v in &mut f32_data {
        *v += bias;
    }
    gpu.upload_f32(&f32_data, shape)
}

/// Raw f16/f32 `data` → device `GpuTensor [n]` (no bias). Moved from
/// `load_any_as_f32` (Task 2).
/// Inverse FWHT-256 un-rotation applied per 256-element group during dequant of
/// every FWHT-rotated format (MQ4/6, MQ3, MFP4, HFQ*-rotated, codebook 19/20/30):
/// pre-multiply by `signs2`, in-place radix-2 Hadamard butterfly, then post-scale
/// by `0.0625 * signs1` (0.0625 = 1/16 = the orthonormal 1/√256 normalization).
///
/// This is attractor-critical math — a sign/normalization error here is the
/// "token soup" failure mode. It was previously inlined byte-for-byte in 6
/// dequant arms; keep it single-source so any fix lands once. `signs1`/`signs2`
/// come from `KvCache::gen_fwht_signs(42|1042, 256)` and are generated once per
/// call site outside the group loop.
fn fwht256_inplace(group: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    debug_assert_eq!(group.len(), 256);
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

pub fn dequant_f32(gpu: &mut Gpu, quant_type: u8, data: &[u8], n: usize) -> HipResult<GpuTensor> {
    let f32_data: Vec<f32> = match quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => widen_bf16(data),
        3 => crate::llama::dequantize_q8_0(data, n),
        14 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 258;
            let n_groups = data.len() / bytes_per_group;
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale_bits = data[off] as u16 | ((data[off + 1] as u16) << 8);
                let scale = f16_to_f32(scale_bits);
                let start = out.len();
                for i in 0..256 {
                    let q = data[off + 2 + i] as i8;
                    out.push(scale * q as f32);
                }
                let group = &mut out[start..start + 256];
                fwht256_inplace(group, &signs1, &signs2);
            }
            out
        }
        6 | 7 | 13 | 15 => {
            let is_6bit = quant_type == 15;
            let group_size: usize = if quant_type == 6 || quant_type == 13 || quant_type == 15 {
                256
            } else {
                128
            };
            let bytes_per_group = if is_6bit { 200 } else { 8 + group_size / 2 };
            let n_groups = data.len() / bytes_per_group;
            let is_mq = quant_type == 13 || quant_type == 15;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let (signs1, signs2) = if is_mq {
                (
                    Some(KvCache::gen_fwht_signs(42, 256)),
                    Some(KvCache::gen_fwht_signs(1042, 256)),
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
                if is_mq && group_size == 256 {
                    let s1 = signs1.as_ref().unwrap();
                    let s2 = signs2.as_ref().unwrap();
                    let group = &mut out[start..start + 256];
                    fwht256_inplace(group, s1, s2);
                }
            }
            out
        }
        8 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 200;
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
            let group_size: usize = 256;
            let bytes_per_group: usize = 112;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 8];
                for k in 0..8 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = f16_to_f32(bits);
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
                fwht256_inplace(group, &signs1, &signs2);
            }
            out
        }
        19 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 72;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 4];
                for k in 0..4 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..64 {
                    let byte_val = data[off + 8 + i] as usize;
                    out.push(cb[byte_val & 3]);
                    out.push(cb[(byte_val >> 2) & 3]);
                    out.push(cb[(byte_val >> 4) & 3]);
                    out.push(cb[(byte_val >> 6) & 3]);
                }
                let group = &mut out[start..start + 256];
                fwht256_inplace(group, &signs1, &signs2);
            }
            out
        }
        30 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 160;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 16];
                for k in 0..16 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..128 {
                    let byte_val = data[off + 32 + i] as usize;
                    out.push(cb[byte_val & 0xF]);
                    out.push(cb[(byte_val >> 4) & 0xF]);
                }
                let group = &mut out[start..start + 256];
                fwht256_inplace(group, &signs1, &signs2);
            }
            out
        }
        17 | 18 => {
            let is_mq3 = quant_type == 17;
            let group_size: usize = 256;
            let bytes_per_group: usize = if is_mq3 { 104 } else { 72 };
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
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
                    for i in 0..64 {
                        let byte_val = data[off + 8 + i] as u32;
                        out.push(scale * ((byte_val & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 2) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 4) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 6) & 3) as f32) + zero);
                    }
                }
                let group = &mut out[start..start + 256];
                fwht256_inplace(group, &signs1, &signs2);
            }
            out
        }
        _ => panic!("unsupported quant_type {quant_type} for dequant_f32"),
    };
    gpu.upload_f32(&f32_data[..n], &[n])
}

// ── WeightBackend trait ─────────────────────────────────────────────────────

use crate::augmentor::{try_augmentors, DEFAULT_AUGMENTORS};
use crate::model_source::ModelSource;
use crate::paro::{load_fp16_weight_from_source, paro_load_f32, paro_load_norm};

/// Pluggable weight-loading backend. `rel` is a layer-relative path: for `proj`
/// it carries NO file extension (the backend appends `.weight` / tries `.qweight`);
/// for `norm`/`raw_f32` it carries the on-disk suffix (e.g. `input_layernorm.weight`).
/// Set the active layer with `set_layer` before each layer's calls.
pub trait WeightBackend {
    fn set_layer(&mut self, layer: usize);
    fn proj(&mut self, rel: &str, m: usize, k: usize) -> HipResult<WeightTensor>;
    fn norm(&mut self, rel: &str, shape: &[usize]) -> HipResult<GpuTensor>;
    fn raw_f32(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor>;
    /// Load a bias vector (f32). Only qwen2 attention biases use this today.
    fn bias(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor>;
}

/// HFQ backend. `norm_bias`: `1.0` (qwen3.5/gemma) or `0.0` (qwen2/llama).
/// `candidates`: layout resolver (`hf_name_candidates` or `flat_name_candidates`).
/// `read`: the arch's pread+awq weight reader (see `HfqRead`).
pub struct HfqBackend<'a> {
    pub hfq: &'a HfqFile,
    pub gpu: &'a mut Gpu,
    pub norm_bias: f32,
    pub candidates: fn(&str) -> Vec<String>,
    pub read_proj:
        fn(&HfqFile, &Gpu, &str, usize, usize, fn(&str) -> Vec<String>) -> HipResult<WeightTensor>,
    pub layer: usize,
}

impl<'a> WeightBackend for HfqBackend<'a> {
    fn set_layer(&mut self, layer: usize) {
        self.layer = layer;
    }

    fn proj(&mut self, rel: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
        (self.read_proj)(
            self.hfq,
            self.gpu,
            &hfq_proj_name(self.layer, rel),
            m,
            k,
            self.candidates,
        )
    }
    fn norm(&mut self, rel: &str, shape: &[usize]) -> HipResult<GpuTensor> {
        let name = hfq_plain_name(self.layer, rel);
        let (info, data) = read_first(self.hfq, &name, self.candidates)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        dequant_norm(self.gpu, info.quant_type, &data, shape, self.norm_bias)
    }
    fn raw_f32(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor> {
        let name = hfq_plain_name(self.layer, rel);
        let (info, data) = read_first(self.hfq, &name, self.candidates)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        dequant_f32(self.gpu, info.quant_type, &data, n)
    }
    fn bias(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor> {
        let name = hfq_plain_name(self.layer, rel);
        let (info, data) = read_first(self.hfq, &name, self.candidates)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        let t = dequant_f32(self.gpu, info.quant_type, &data, n)?;
        assert_eq!(
            t.numel(),
            n,
            "bias {name} has {} elements, expected {n}",
            t.numel()
        );
        Ok(t)
    }
}

/// Resolve `name` via `candidates` and return the first tensor's `(info, bytes)`.
pub fn read_first(
    hfq: &HfqFile,
    name: &str,
    candidates: fn(&str) -> Vec<String>,
) -> Option<(crate::hfq::HfqTensorInfo, Vec<u8>)> {
    for c in candidates(name) {
        if let Some((info, buf)) = hfq.tensor_data_vec(&c) {
            return Some((info.clone(), buf));
        }
    }
    None
}

/// PaRo backend (augmentor chain + paro primitives) — fully arch-agnostic.
/// `mp` is the text-tower prefix from `paro_text_prefix`.
pub struct ParoBackend<'a> {
    pub source: &'a dyn ModelSource,
    pub gpu: &'a mut Gpu,
    pub mp: &'static str,
    pub layer: usize,
    /// `1.0` (qwen3.5/gemma) or `0.0` (qwen2/llama).
    pub norm_bias: f32,
}

impl<'a> WeightBackend for ParoBackend<'a> {
    fn set_layer(&mut self, layer: usize) {
        self.layer = layer;
    }

    fn proj(&mut self, rel: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
        let base = paro_proj_name(self.mp, self.layer, rel);
        match try_augmentors(self.source, &base, m, k, self.gpu, DEFAULT_AUGMENTORS)? {
            Some(t) => Ok(t),
            None => {
                load_fp16_weight_from_source(self.source, self.gpu, &format!("{base}.weight"), m, k)
            }
        }
    }
    fn norm(&mut self, rel: &str, shape: &[usize]) -> HipResult<GpuTensor> {
        paro_load_norm(
            self.source,
            self.gpu,
            &paro_plain_name(self.layer, rel),
            shape,
            self.norm_bias,
        )
    }
    fn raw_f32(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor> {
        paro_load_f32(self.source, self.gpu, &paro_plain_name(self.layer, rel), n)
    }
    fn bias(&mut self, _rel: &str, _n: usize) -> HipResult<GpuTensor> {
        Err(hip_bridge::HipError::new(
            0,
            "ParoBackend: attention biases unsupported",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `fwht256_inplace` must be bit-identical to the per-arm inlined FWHT it
    /// replaced (the version that shipped in 6 dequant arms). Pin it: run a
    /// verbatim copy of the old inline sequence and the helper on the same
    /// pseudo-random group + signs, assert exact f32 equality. A drift here is
    /// the attractor / "token soup" failure mode, so equality must be exact.
    #[test]
    fn fwht256_inplace_matches_inlined_reference() {
        // Deterministic pseudo-random inputs (no rand dep).
        let mut x: u32 = 0x1234_5678;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        let group_init: Vec<f32> = (0..256).map(|_| next()).collect();
        // signs are ±1 in practice; mirror that.
        let signs1: Vec<f32> = (0..256)
            .map(|_| if next() < 0.0 { -1.0 } else { 1.0 })
            .collect();
        let signs2: Vec<f32> = (0..256)
            .map(|_| if next() < 0.0 { -1.0 } else { 1.0 })
            .collect();

        // Reference: verbatim old inline sequence.
        let mut reference = group_init.clone();
        {
            let group = &mut reference[..];
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

        let mut got = group_init.clone();
        fwht256_inplace(&mut got, &signs1, &signs2);

        assert_eq!(got.len(), 256);
        for i in 0..256 {
            assert_eq!(
                got[i].to_bits(),
                reference[i].to_bits(),
                "fwht256_inplace diverged from inlined reference at index {i}"
            );
        }
    }

    #[test]
    fn nested_candidates_cover_both_layouts() {
        let c = hf_name_candidates("layers.0.self_attn.q_proj.weight");
        assert_eq!(
            c[0],
            "model.language_model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(c[1], "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(c[2], "layers.0.self_attn.q_proj.weight");
    }
    #[test]
    fn lm_head_special_case() {
        let c = hf_name_candidates("lm_head.weight");
        assert_eq!(c[0], "lm_head.weight");
        assert!(c.contains(&"model.language_model.lm_head.weight".to_string()));
    }
    #[test]
    fn flat_candidates_are_two() {
        assert_eq!(
            flat_name_candidates("layers.0.mlp.down_proj.weight"),
            vec![
                "model.layers.0.mlp.down_proj.weight".to_string(),
                "layers.0.mlp.down_proj.weight".to_string()
            ]
        );
    }
    #[test]
    fn name_builders() {
        assert_eq!(
            hfq_proj_name(3, "self_attn.q_proj"),
            "layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            hfq_plain_name(3, "input_layernorm.weight"),
            "layers.3.input_layernorm.weight"
        );
        assert_eq!(
            paro_proj_name("model.language_model", 0, "linear_attn.in_proj_qkv"),
            "model.language_model.layers.0.linear_attn.in_proj_qkv"
        );
        assert_eq!(
            paro_plain_name(0, "input_layernorm.weight"),
            "layers.0.input_layernorm.weight"
        );
    }

    // ── Embedding / tied-lm_head tests ──────────────────────────────────────

    #[test]
    fn embed_classify_raw_hfq4g256() {
        match embed_classify(6).unwrap() {
            EmbedPlan::Raw(EmbeddingFormat::HFQ4G256) => {}
            other => panic!("expected Raw(HFQ4G256), got {other:?}"),
        }
    }
    #[test]
    fn embed_classify_raw_hfq4g128() {
        match embed_classify(7).unwrap() {
            EmbedPlan::Raw(EmbeddingFormat::HFQ4G128) => {}
            other => panic!("expected Raw(HFQ4G128), got {other:?}"),
        }
    }
    #[test]
    fn embed_classify_raw_q8_0() {
        match embed_classify(3).unwrap() {
            EmbedPlan::Raw(EmbeddingFormat::Q8_0) => {}
            other => panic!("expected Raw(Q8_0), got {other:?}"),
        }
    }
    #[test]
    fn embed_classify_host_f32() {
        for qt in [1, 2, 16] {
            match embed_classify(qt).unwrap() {
                EmbedPlan::HostF32 => {}
                other => panic!("qt={qt}: expected HostF32, got {other:?}"),
            }
        }
    }
    #[test]
    fn embed_classify_errors_on_unknown() {
        let err = embed_classify(99).unwrap_err();
        assert!(err.message.contains("unsupported embedding quant_type"));
    }
    #[test]
    fn embedding_format_dtype_mapping() {
        assert_eq!(
            embedding_format_dtype(EmbeddingFormat::HFQ4G256),
            DType::HFQ4G256
        );
        assert_eq!(
            embedding_format_dtype(EmbeddingFormat::HFQ4G128),
            DType::HFQ4G128
        );
        assert_eq!(embedding_format_dtype(EmbeddingFormat::Q8_0), DType::Q8_0);
        assert_eq!(embedding_format_dtype(EmbeddingFormat::F32), DType::F32);
    }
    #[test]
    #[should_panic(expected = "Q4K not valid")]
    fn embedding_format_dtype_q4k_panics() {
        embedding_format_dtype(EmbeddingFormat::Q4K);
    }

    #[test]
    fn f16_bytes_to_f32_roundtrips_known_values() {
        // 1.0 = 0x3C00, 2.0 = 0x4000, -1.0 = 0xBC00 (little-endian byte pairs).
        let bytes = [0x00, 0x3C, 0x00, 0x40, 0x00, 0xBC];
        assert_eq!(f16_bytes_to_f32(&bytes), vec![1.0, 2.0, -1.0]);
    }

    #[test]
    fn tied_alias_assembles_f32_view() {
        let embd = GpuTensor::null_for_test();
        let wt = tied_lm_head_alias(&embd, EmbeddingFormat::F32, 100, 64);
        assert_eq!(wt.gpu_dtype, DType::F32);
        assert_eq!(wt.m, 100);
        assert_eq!(wt.k, 64);
        assert_eq!(wt.row_stride, 0);
        assert!(wt.paro.is_none());
        assert!(wt.awq_scale.is_none());
    }

    #[test]
    fn tied_alias_maps_hfq4g256_dtype() {
        let embd = GpuTensor::null_for_test();
        let wt = tied_lm_head_alias(&embd, EmbeddingFormat::HFQ4G256, 8, 8);
        assert_eq!(wt.gpu_dtype, DType::HFQ4G256);
    }

    #[test]
    #[should_panic(expected = "Q4K not valid for tied lm_head")]
    fn tied_alias_q4k_panics() {
        let embd = GpuTensor::null_for_test();
        let _ = tied_lm_head_alias(&embd, EmbeddingFormat::Q4K, 8, 8);
    }

    /// Pins every RAW_CODECS row against the dtype the *production* arms produced,
    /// transcribed with source citations so the oracle is independent of the table.
    /// A drift here mis-tags a quant format → "token soup"; equality must be exact.
    #[test]
    fn raw_codecs_golden_against_production_arms() {
        // (quant_type, dtype) — RHS copied from the pre-refactor arms:
        //   wb = weight_backend.rs (dequant_weight_raw), hfq = hfq.rs (load_weight_tensor)
        let expected: &[(u8, DType)] = &[
            (0, DType::Q4F16G64),      // hfq:712
            (3, DType::Q8_0),          // wb:487 / hfq:725
            (4, DType::Q4K),           // hfq:738
            (5, DType::Q8HFQ),         // hfq:754
            (6, DType::HFQ4G256),      // wb:299 / hfq:767
            (7, DType::HFQ4G128),      // wb:311 / hfq:780
            (8, DType::HFQ6G256),      // wb:323 / hfq:793
            (9, DType::HFQ2G256),      // hfq:807
            (10, DType::HFQ2G128),     // hfq:819
            (11, DType::HFQ3G256),     // wb:335 / hfq:832
            (12, DType::HFQ3G128),     // wb:347 / hfq:845
            (13, DType::MQ4G256),      // wb:359 / hfq:858
            (14, DType::MQ8G256),      // wb:371 / hfq:871
            (15, DType::MQ6G256),      // wb:383
            (17, DType::MQ3G256),      // wb:395 / hfq:884
            (18, DType::MQ2G256),      // wb:407 / hfq:897
            (19, DType::MQ2G256Lloyd), // wb:419 / hfq:910
            (20, DType::MQ3G256Lloyd), // wb:431 / hfq:923
            (21, DType::HFP4G32),      // wb:459 / hfq:944
            (24, DType::MFP4G32),      // wb:475 / hfq:963
            (30, DType::MQ4G256Lloyd), // wb:443 / hfq:978 (renumbered from 21; do not swap)
            // GL ("global Lloyd") codebook formats — MoE-routed-expert only.
            // RHS pinned against hipfire-quantize `QuantType::MQ2G256GL = 38` /
            // `MQ3G256GL = 39`; a swap here mis-decodes 64 B/group indices as
            // 96 B/group (or vice versa) → token soup, not a crash.
            (38, DType::MQ2G256GL),
            (39, DType::MQ3G256GL),
        ];
        for &(qt, dt) in expected {
            let c = raw_codec(qt).unwrap_or_else(|| panic!("no RAW_CODECS row for qt={qt}"));
            assert_eq!(c.dtype, dt, "qt={qt} dtype");
        }
        assert_eq!(
            RAW_CODECS.len(),
            expected.len(),
            "RAW_CODECS has unlisted rows"
        );
        // Host-decode formats must NOT be in the passthrough table:
        for qt in [1u8, 2, 16] {
            assert!(
                raw_codec(qt).is_none(),
                "qt={qt} is host-decode, must not be a raw codec"
            );
        }
    }

    /// No two rows may claim the same quant_type (find() is first-match-wins).
    #[test]
    fn raw_codecs_unique_quant_types() {
        for (i, a) in RAW_CODECS.iter().enumerate() {
            for b in &RAW_CODECS[i + 1..] {
                assert_ne!(
                    a.quant_type, b.quant_type,
                    "duplicate quant_type {}",
                    a.quant_type
                );
            }
        }
    }
}
