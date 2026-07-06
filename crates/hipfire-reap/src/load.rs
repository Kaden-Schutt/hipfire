//! Shared REAP keep-map row-gather loaders.
//!
//! Reads a named tensor from an [`HfqFile`] and gathers its first-axis rows (one
//! per ORIGINAL expert) down to the kept set BEFORE any dequant. Centralizes the
//! gather that was copy-pasted verbatim across the lfm2moe and minimax MoE
//! loaders (review #9) so a fix lands once; each arch keeps only its
//! `quant_type → DType` construction. Exact for any row-independent quant (every
//! per-expert row carries its own scale/zero/codebook).
//!
//! deepseek4 (multi-dim `upload_*_keep`) and qwen35 (candidate-name resolution +
//! AWQ sidecar) wrap [`crate::gather::gather_rows`] directly with arch-specific
//! tensor resolution, so they do not route through these helpers.

use crate::gather::gather_rows;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;

/// Gather a quantized 2-D weight tensor's first-axis (per-expert) rows down to
/// `keep`, returning `(quant_type, gathered_bytes)` for the arch to hand to its
/// own `wt_from_raw`. The on-disk `shape[0]` is the original expert count;
/// `gather_rows` derives the per-row stride from it and selects `keep` rows in
/// compact-slot order. `arch` prefixes the error messages.
pub fn gather_weight_rows(
    arch: &str,
    hfq: &HfqFile,
    name: &str,
    keep: &[u32],
) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("{arch}: reap tensor not found in HFQ: {name}"))?;
    let orig = *info.shape.first().unwrap_or(&0) as usize;
    let (_new_shape, sub) = gather_rows(&[orig], &data, keep)
        .map_err(|e| format!("{arch}: reap router row-gather '{name}': {e}"))?;
    Ok((info.quant_type, sub))
}

/// Gather a 1-D per-expert F16/F32 vector (e.g. the router's per-expert routing
/// bias, shape `[orig_experts]`) down to `keep` and dequantize to `f32` for the
/// arch to upload. Refuses block-packed quant: a single element is not a whole
/// quant block, so only F16 (qt 1) / F32 (qt 2) are element-gatherable. Uses the
/// SAME [`hipfire_runtime::llama::f16_to_f32`] the arches used, so the result is
/// byte-identical to the pre-refactor per-arch path.
pub fn gather_f32_vec(
    arch: &str,
    hfq: &HfqFile,
    name: &str,
    keep: &[u32],
) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("{arch}: reap tensor not found in HFQ: {name}"))?;
    // Per-element width for the row-gather. Q8_0 packs 32 elems/block, so a
    // single element is not a whole row — refuse rather than corrupt.
    let elem_bytes = match info.quant_type {
        1 => 2, // F16
        2 => 4, // F32
        other => {
            return Err(format!(
                "{arch}: reap per-expert vector {name} keep-gather needs F16/F32, got qt={other}"
            ))
        }
    };
    // 1-D vector: element count == bytes / elem width; gather selects kept elems.
    let orig = data.len() / elem_bytes;
    let (_new_shape, sub) = gather_rows(&[orig], &data, keep)
        .map_err(|e| format!("{arch}: reap per-expert vector row-gather '{name}': {e}"))?;
    Ok(match info.quant_type {
        1 => sub
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        _ => sub
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    })
}
