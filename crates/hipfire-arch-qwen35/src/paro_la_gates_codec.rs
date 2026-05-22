// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ4G256 codec for LinearAttention gate weights not in the PARO calibrated set.
//!
//! Lever 1 (`feat/paroquant-batched-phase2-shared-expert`): some
//! LinearAttention projections (`in_proj_a`, `in_proj_b`) ship in the
//! shisa-ai PARO checkpoints without per-row Givens rotation metadata
//! and so cannot be loaded via the `ParoQ4G128` path. To keep these
//! tensors on the FWHT-rotated INT4 fast path instead of falling back
//! to Q8/F16, this module encodes their FP16 weights into the existing
//! `DType::MQ4G256` byte layout.
//!
//! Layout (per 256-element group, 136 bytes total — mirrors the
//! quantizer in `crates/hipfire-quantize/src/main.rs:573` and is what
//! `gemv_mq4g256_prerotated` reads):
//!
//!   bytes [0..4)    : f32 LE  scale  = (max - min) / 15
//!   bytes [4..8)    : f32 LE  min
//!   bytes [8..136)  : 128 packed nibbles, where byte `8 + i`
//!                     holds `lo | (hi << 4)` with
//!                     `lo = q(group[2*i])` and
//!                     `hi = q(group[2*i + 1])`,
//!                     `q(v) = clamp(round((v - min) / scale), 0, 15)`.
//!
//! Activation rotation: the encoder writes weights as-is (no FWHT). At
//! dispatch time, `rotate_x_mq_for` applies the matching FWHT-256 to
//! the activation row, so `dot(W, x) = dot(W_unrotated, x_unrotated)`
//! is preserved through the kernel. Producing MQ4G256-layout bytes is
//! sufficient — no per-row Hadamard bake-in is needed on the weight
//! side for this lever.

/// MQ4G256-encoded weight tensor ready for the
/// `gemv_mq4g256_prerotated` dispatch path.
///
/// `group_size` is always 256 for this codec; it is kept as a field
/// to mirror the shape of future codecs (e.g. MQ3G256, HFQ4G128) and
/// to make assertions explicit at call sites.
pub struct EncodedMQ4G256 {
    pub bytes: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

/// Encode a row-major FP16 weight matrix `[rows, cols]` into MQ4G256
/// layout. `cols` must be a multiple of 256.
///
/// Not yet implemented — Task 3 fills in the body. The signature is
/// frozen so Task 2's failing test pins down the API.
pub fn encode_mq4g256_from_fp16(
    _weight_fp16: &[u16],
    _rows: usize,
    _cols: usize,
) -> EncodedMQ4G256 {
    unimplemented!("encode_mq4g256_from_fp16 — Task 3 (codec impl)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::llama::f32_to_f16;

    /// One group per row: 16 rows × 256 cols = 16 groups total.
    /// At 136 bytes/group, expected encoded size is 16 * 136 = 2176 bytes.
    #[test]
    fn round_trip_basic() {
        let rows = 16usize;
        let cols = 256usize;
        let group_size = 256usize;
        let block_bytes = 136usize;

        // Fill with a deterministic ramp so the bytes aren't all-zero
        // once the encoder lands. Values stay in a small range that
        // FP16 represents exactly.
        let mut weight_fp16 = vec![0u16; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let v = ((r as f32) * 0.125) + ((c as f32) * 0.001);
                weight_fp16[r * cols + c] = f32_to_f16(v);
            }
        }

        let encoded = encode_mq4g256_from_fp16(&weight_fp16, rows, cols);

        let expected_groups = rows * (cols / group_size);
        let expected_len = expected_groups * block_bytes;
        assert_eq!(
            encoded.bytes.len(),
            expected_len,
            "MQ4G256 byte layout: {} groups × {} bytes/group",
            expected_groups,
            block_bytes
        );
        assert_eq!(encoded.rows, rows);
        assert_eq!(encoded.cols, cols);
        assert_eq!(encoded.group_size, 256);
    }
}
