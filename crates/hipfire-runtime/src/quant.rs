// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Block dequantization codecs for HFQ tensor types + half/bf16 ↔ f32
//! conversions.
//!
//! Arch-agnostic numeric primitives used by the HFQ loaders. `dequant_q8f16`
//! decodes HFQ's `Q8F16` type and `dequant_q4k` its `Q4K` type — both
//! reuse the corresponding block byte layout but are native HFQ inference
//! codecs, not GGUF readers. The GGUF-only decoders (Q4_0, Q6_K, and the
//! Q4_K→Q4_F16 transcoders) were removed with the GGUF inference path — GGUF is
//! import-only, in hipfire-coexistence.

/// Dequantize an HFQ Q8F16 (int8, f16-scale) block tensor to f32.
/// Block: 2 bytes (f16 scale) + 32 bytes (32 x int8) = 34 bytes / 32 weights.
pub fn dequant_q8f16(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let block_offset = b * 34; // 2 + 32 bytes per block
        if block_offset + 34 > data.len() {
            break;
        }
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        for j in 0..32 {
            let idx = b * block_size + j;
            if idx < n {
                let val = data[block_offset + 2 + j] as i8;
                out[idx] = val as f32 * scale;
            }
        }
    }
    out
}

/// Dequantize an HFQ Oq8G256 (Opus-Quant W8A8) block tensor to **plain** f32.
/// Block: 2 bytes (f16 scale) + 256 bytes (256 signed int8) = 258 B / 256 weights.
///
/// Oq8 is FWHT-256 **rotated** at quantize time (`cpu_fwht_256(group, s1, s2)`
/// before int8 packing — see `hipfire-quantize::codecs::quantize_oq8g256`). The
/// Opus iu8 GEMM consumes the rotated weights directly (activations are rotated
/// to match), so arch loaders that feed the Opus kernel keep them rotated. This
/// helper is for callers that need the **un-rotated** weight (e.g. the gemma3-vl
/// vision tower / projector, which use a plain f32/bf16 GEMM): after `int8·scale`
/// it applies the inverse FWHT (`cpu_fwht_256(grp, s2, s1)` — signs swapped) with
/// the engine-fixed seeds (42, 1042). Oq8G256 requires K % 256 == 0, so `n` is a
/// whole number of full 256-groups (no partial group crosses the FWHT boundary).
pub fn dequant_oq8g256(data: &[u8], n: usize) -> Vec<f32> {
    use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
    const GROUP: usize = 256;
    const BLOCK: usize = 258; // 2 (f16 scale) + 256 int8
    let s1 = gen_fwht_signs(42, GROUP);
    let s2 = gen_fwht_signs(1042, GROUP);
    let nblocks = n / GROUP; // K%256==0 ⇒ n is a whole number of groups
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let off = b * BLOCK;
        if off + BLOCK > data.len() {
            break;
        }
        let scale = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let mut grp = [0.0f32; GROUP];
        for j in 0..GROUP {
            grp[j] = (data[off + 2 + j] as i8) as f32 * scale;
        }
        // Inverse rotation back to the original (un-rotated) weight basis.
        cpu_fwht_256(&mut grp, &s2, &s1);
        out[b * GROUP..b * GROUP + GROUP].copy_from_slice(&grp);
    }
    out
}

// f16↔f32 conversions are now the canonical implementations in the shared
// `hipfire-primitives` leaf (they were byte-identical copies). Re-exported here
// so the ~20 arch/loader call sites importing `hipfire_runtime::quant::*` stay
// unchanged and transitively share one implementation.
pub use hipfire_primitives::conv::{f16_to_f32, f32_to_f16};

/// Dequantize an HFQ Q4K block tensor to f32.
/// Super-block: 256 elements, 144 bytes
///   2 bytes: f16 d (super-block scale)
///   2 bytes: f16 dmin (super-block min)
///   12 bytes: scales/mins for 8 sub-blocks (6 bits each, packed)
///   128 bytes: 256 x 4-bit quantized values
pub fn dequant_q4k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 144; // 2+2+12+128
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }

        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));

        // Unpack scales and mins from 12 bytes (at off+4)
        let sc_data = &data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // First 4 sub-blocks: lower 6 bits from bytes 0-3 (scales) and 4-7 (mins)
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        // Next 4 sub-blocks: lower 4 bits from bytes 8-11, upper 2 bits from bytes 0-7
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        // Dequantize 256 values from 128 bytes of 4-bit data.
        // GGML layout: 4 groups of 64 elements. Each group has 2 sub-blocks
        // sharing 32 bytes: lower nibble → even sub-block, upper nibble → odd.
        let qdata = &data[off + 16..off + 16 + 128];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let sc_even = d * scales[sb_even] as f32;
            let m_even = dmin * mins[sb_even] as f32;
            let sc_odd = d * scales[sb_odd] as f32;
            let m_odd = dmin * mins[sb_odd] as f32;

            for l in 0..32 {
                let byte = qdata[group * 32 + l];
                let idx_even = b * block_size + group * 64 + l;
                let idx_odd = idx_even + 32;
                if idx_even < n {
                    out[idx_even] = (byte & 0x0F) as f32 * sc_even - m_even;
                }
                if idx_odd < n {
                    out[idx_odd] = ((byte >> 4) & 0x0F) as f32 * sc_odd - m_odd;
                }
            }
        }
    }
    out
}

/// Re-export the canonical on-disk byte-contract so arch loaders can reach it
/// as `hipfire_runtime::quant::QuantType` without each depending on the leaf
/// `hipfire-quant-format` crate directly.
pub use hipfire_quant_format::QuantType;

/// Canonical map from an on-disk HFQ `quant_type` byte to the GPU dispatch
/// [`DType`], for the **pure** formats: ones the loader handles as a plain
/// `upload_raw` + dtype tag with no host-side repack.
///
/// This is the single source of truth that replaces the per-arch
/// `slab_dtype_for_quant` / `dtype_for_quant` copies that drifted across
/// qwen35, minimax, lfm2, gemma3, and qwen2 (each carried a divergent subset —
/// e.g. only qwen35 mapped `31 => Qtip3G256`). Routing every loader through
/// here means a new pure format lands in all arches with one edit.
///
/// Returns `None` for:
/// - unknown codes, and
/// - formats that require a host-side transform before upload (bf16 buffer
///   retag `16`; Opus-Quant arch-repack `33/34/35/37`). Callers keep those
///   transform branches and fall through to this map for the pure cases.
///
/// `k` (the input/column dim) gates the FP4 group-32 formats, which require
/// `k % 256 == 0`.
///
/// Matches on the canonical [`QuantType`] (the shared byte-contract) rather
/// than raw integers, so the on-disk ids stay authoritative in one crate.
pub fn dtype_for_quant_type(qt: u8, k: usize) -> Option<hipfire_rdna::DType> {
    use hipfire_quant_format::QuantType as Q;
    use hipfire_rdna::DType;
    Some(match Q::from_code(qt)? {
        Q::F16 => DType::F16,
        Q::Q8F16 => DType::Q8_0,
        Q::HFQ4G256 => DType::HFQ4G256,
        Q::HFQ4G128 => DType::HFQ4G128,
        Q::HFQ6G256 => DType::HFQ6G256,
        Q::HFQ3G256 => DType::HFQ3G256,
        Q::HFQ3G128 => DType::HFQ3G128,
        Q::MQ4G256 => DType::MQ4G256,
        Q::MQ8G256 => DType::MQ8G256,
        Q::MQ6G256 => DType::MQ6G256,
        Q::MQ3G256 => DType::MQ3G256,
        Q::MQ2G256 => DType::MQ2G256,
        Q::MQ2G256Lloyd => DType::MQ2G256Lloyd,
        Q::MQ3G256Lloyd => DType::MQ3G256Lloyd,
        Q::HFP4G32 if k % 256 == 0 => DType::HFP4G32,
        Q::MFP4G32 if k % 256 == 0 => DType::MFP4G32,
        Q::MQ4G256Lloyd => DType::MQ4G256Lloyd,
        Q::Qtip3G256 => DType::Qtip3G256,
        Q::Qtip4G256 => DType::Qtip4G256,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};

    #[test]
    fn dequant_oq8g256_inverts_the_fwht_rotation() {
        // Regression: Oq8 stores FWHT-rotated weights (matches
        // hipfire-quantize::codecs::quantize_oq8g256). dequant_oq8g256 must
        // return the *un-rotated* weight, else consumers using a plain GEMM
        // (gemma3-vl vision/projector) get scrambled values (cosine ~0, which
        // flipped medgemma's ultrasound→"X-ray"). Build one block the way the
        // quantizer does and confirm recovery of the original.
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let orig: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.017 - 2.0).sin()).collect();
        let mut rot = [0.0f32; 256];
        rot.copy_from_slice(&orig);
        cpu_fwht_256(&mut rot, &s1, &s2); // rotate, as quantize_oq8g256 does
        let scale = rot.iter().fold(0.0f32, |m, &v| m.max(v.abs())) / 127.0;
        let mut block = vec![0u8; 258];
        let sf16 = f32_to_f16(scale);
        block[0] = (sf16 & 0xff) as u8;
        block[1] = (sf16 >> 8) as u8;
        for i in 0..256 {
            block[2 + i] = ((rot[i] / scale).round().clamp(-127.0, 127.0) as i8) as u8;
        }
        let deq = dequant_oq8g256(&block, 256);
        // Cosine with the ORIGINAL (un-rotated) must be ~1 (near-lossless);
        // a non-un-rotating dequant would land near 0.
        let dot: f32 = deq.iter().zip(&orig).map(|(a, b)| a * b).sum();
        let na: f32 = deq.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nb: f32 = orig.iter().map(|b| b * b).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(cos > 0.999, "dequant_oq8g256 not un-rotating: cosine={cos}");
    }
}
