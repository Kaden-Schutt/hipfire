//! MQ4G128 codec for LinearAttention in_proj_a / in_proj_b weights.
//!
//! Produces the same 72-byte-per-group layout consumed by `gemv_hfq4g128`:
//!   4-byte F32 LE scale | 4-byte F32 LE zero | 64 packed nibbles (lo=even, hi=odd)
//!
//! FWHT-128 is applied at encode using the SAME sign tables (seeds 43, 1043)
//! that `ensure_mq_signs_128` uploads to the device. This task adds the module
//! scaffolding + gating predicate. Task 9 fills in `encode_mq4g128_from_fp16`.
//!
//! Byte layout convention: bytes[0..4] = F32 LE scale, bytes[4..8] = F32 LE min_v
//! (the "zero" additive offset applied after scale multiply — matches Convention A
//! from quantize_mq4g256 in hipfire-quantize/src/main.rs:~600).
//! The kernel dequants as `scale * q + zero` (gemv_hfq4g128.hip line 37-40), so
//! storing `min_v` at zero is correct: `scale * 0 + min_v = min_v`. Nibble formula:
//! `q = round((x - min_v) * inv_scale).clamp(0, 15)`.

use hipfire_runtime::llama::f16_to_f32;

const GROUP_SIZE: usize = 128;
const BYTES_PER_GROUP: usize = 72; // 4-byte scale + 4-byte zero + 64 packed nibbles
const SIGNS1_SEED: u32 = 43;
const SIGNS2_SEED: u32 = 1043;

/// Walsh-Hadamard butterfly with pre/post sign-multiplication, scale 1/sqrt(128).
/// MUST produce bit-identical output to gemv_mq4g128.hip's mq_rotate_x_128 kernel
/// when given the same sign tables (seeds 43, 1043). T3 test already verifies this
/// for the kernel side; this CPU version is the encoder counterpart.
fn cpu_fwht_128(x: &mut [f32; GROUP_SIZE], signs1: &[f32], signs2: &[f32]) {
    debug_assert_eq!(signs1.len(), GROUP_SIZE);
    debug_assert_eq!(signs2.len(), GROUP_SIZE);
    for i in 0..GROUP_SIZE {
        x[i] *= signs1[i];
    }
    let mut stride = 1;
    while stride < GROUP_SIZE {
        let mut i = 0;
        while i < GROUP_SIZE {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    // 1/sqrt(128) = 0.0883883476...
    let scale = 1.0f32 / (GROUP_SIZE as f32).sqrt();
    for i in 0..GROUP_SIZE {
        x[i] *= scale * signs2[i];
    }
}

/// Encoded weight buffer in MQ4G128 byte layout.
pub struct EncodedMQ4G128 {
    /// Packed bytes: `rows * (cols / 128) * 72`.
    pub bytes: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

/// Encode FP16 weight (row-major) into MQ4G128 byte layout.
/// Panics if `cols % 128 != 0` or `weight_fp16.len() != rows * cols`.
///
/// Applies FWHT-128 (seeds 43, 1043) to each 128-element row group, then
/// quantizes to INT4 affine with scale and min stored per group.
pub fn encode_mq4g128_from_fp16(weight_fp16: &[u16], rows: usize, cols: usize) -> EncodedMQ4G128 {
    assert!(
        cols % GROUP_SIZE == 0,
        "encode_mq4g128_from_fp16: cols={cols} not a multiple of {GROUP_SIZE}"
    );
    assert_eq!(
        weight_fp16.len(),
        rows * cols,
        "encode_mq4g128_from_fp16: weight_fp16.len() {} != rows * cols {}",
        weight_fp16.len(),
        rows * cols
    );

    let signs1_vec = rdna_compute::gen_fwht_signs(SIGNS1_SEED, GROUP_SIZE);
    let signs2_vec = rdna_compute::gen_fwht_signs(SIGNS2_SEED, GROUP_SIZE);

    let groups_per_row = cols / GROUP_SIZE;
    let bytes_per_row = groups_per_row * BYTES_PER_GROUP;
    let mut bytes = vec![0u8; rows * bytes_per_row];

    let mut group = [0f32; GROUP_SIZE];
    for m in 0..rows {
        for g in 0..groups_per_row {
            // 1. Load FP16 → F32 into group
            for i in 0..GROUP_SIZE {
                group[i] = f16_to_f32(weight_fp16[m * cols + g * GROUP_SIZE + i]);
            }
            // 2. FWHT-128 in place (matching kernel exactly)
            cpu_fwht_128(&mut group, &signs1_vec, &signs2_vec);
            // 3. Find min/max
            let (min_v, max_v) = group.iter().fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(lo, hi), &x| (lo.min(x), hi.max(x)),
            );
            // 4. Scale + zero, with all-zeros guard
            let raw_scale = (max_v - min_v) / 15.0;
            let scale = if raw_scale == 0.0 || raw_scale.is_nan() {
                f32::EPSILON
            } else {
                raw_scale
            };
            let inv_scale = 1.0 / scale;

            // 5. Write per-group header: [scale F32 LE | min_v F32 LE]
            // Convention A: store min_v directly as the additive zero offset.
            // Kernel dequants as: w = scale * q + zero  (gemv_hfq4g128.hip:37-40)
            // So zero = min_v makes q=0 map correctly to min_v.
            let row_off = m * bytes_per_row + g * BYTES_PER_GROUP;
            bytes[row_off..row_off + 4].copy_from_slice(&scale.to_le_bytes());
            bytes[row_off + 4..row_off + 8].copy_from_slice(&min_v.to_le_bytes());

            // 6. Quantize and pack nibbles (lo nibble = even index, hi nibble = odd index)
            for i in 0..(GROUP_SIZE / 2) {
                let lo_q =
                    ((group[i * 2] - min_v) * inv_scale).round().clamp(0.0, 15.0) as u8;
                let hi_q =
                    ((group[i * 2 + 1] - min_v) * inv_scale).round().clamp(0.0, 15.0) as u8;
                bytes[row_off + 8 + i] = lo_q | (hi_q << 4);
            }
        }
    }

    EncodedMQ4G128 { bytes, rows, cols }
}

/// Decide whether to apply MQ4G128 encoding to a weight at load time.
///
/// Returns `true` only when:
/// - `prefix` ends with `linear_attn.in_proj_a` or `linear_attn.in_proj_b`, AND
/// - The arch+env gating allows.
///
/// Env var `HIPFIRE_PARO_LA_GATES_MQ4G128={0|1}` overrides arch default;
/// unset uses `rdna_compute::arch_caps::paro_la_gates_mq4g128_default(arch)`.
pub fn should_quantize_la_gate(prefix: &str, arch: &str) -> bool {
    if !(prefix.ends_with("linear_attn.in_proj_a")
        || prefix.ends_with("linear_attn.in_proj_b"))
    {
        return false;
    }
    match std::env::var("HIPFIRE_PARO_LA_GATES_MQ4G128").ok().as_deref() {
        Some("0") => return false,
        Some("1") => return true,
        Some("") | None => {}
        Some(other) => {
            eprintln!(
                "WARN: HIPFIRE_PARO_LA_GATES_MQ4G128={other:?} not recognized (expected 0|1|unset); using arch default"
            );
        }
    }
    rdna_compute::arch_caps::paro_la_gates_mq4g128_default(arch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::llama::{f16_to_f32, f32_to_f16};

    #[test]
    fn encoder_produces_expected_byte_count() {
        let rows = 16;
        let cols = 4096;
        let weight = vec![f32_to_f16(0.123); rows * cols];
        let enc = encode_mq4g128_from_fp16(&weight, rows, cols);
        assert_eq!(enc.rows, rows);
        assert_eq!(enc.cols, cols);
        assert_eq!(enc.bytes.len(), rows * (cols / 128) * 72);
    }

    #[test]
    #[should_panic(expected = "not a multiple of 128")]
    fn rejects_bad_cols() {
        encode_mq4g128_from_fp16(&vec![0u16; 100], 1, 100);
    }

    #[test]
    fn all_zeros_row_safe() {
        let enc = encode_mq4g128_from_fp16(&vec![0u16; 128], 1, 128);
        assert_eq!(enc.bytes.len(), 72);
        let scale = f32::from_le_bytes(enc.bytes[0..4].try_into().unwrap());
        assert!(scale.is_finite(), "scale should be finite, got {scale}");
    }

    /// Critical: encode → dequant → undo-FWHT (apply forward FWHT a second time) →
    /// should match the original FP16 values modulo Q4 quant error.
    ///
    /// Both the weight FWHT (applied during encode) and the activation FWHT
    /// (applied at decode time by the kernel) are the same orthogonal transform.
    /// In the dot product they cancel: (H*w)·(H*x) = w·(H^T*H*x) = w·x.
    /// We simulate this here by rotating the activation through cpu_fwht_128 and
    /// then computing dot products against the dequantized rotated weights.
    #[test]
    fn round_trip_dot_product() {
        // 16 × 256 weight (2 groups per row)
        let rows = 16;
        let cols = 256;
        let mut weight_f32 = vec![0f32; rows * cols];
        let mut weight_fp16 = vec![0u16; rows * cols];
        for m in 0..rows {
            for k in 0..cols {
                let v = ((m * 7 + k) as f32) * 0.001 - 0.5;
                weight_f32[m * cols + k] = v;
                weight_fp16[m * cols + k] = f32_to_f16(v);
            }
        }

        // Reference activation (in original basis)
        let act_f32: Vec<f32> = (0..cols).map(|k| (k as f32) * 0.01 - 1.0).collect();
        // Reference output: y[m] = sum_k W[m,k] * x[k] (original-basis matmul)
        // Use FP16-rounded weights to match what the encoder sees.
        let ref_y: Vec<f32> = (0..rows)
            .map(|m| {
                (0..cols)
                    .map(|k| f16_to_f32(weight_fp16[m * cols + k]) * act_f32[k])
                    .sum()
            })
            .collect();

        // Encode weights through MQ4G128 (applies FWHT to weights)
        let enc = encode_mq4g128_from_fp16(&weight_fp16, rows, cols);

        // Rotate the activation through FWHT-128 using the same sign tables.
        // The two FWHTs cancel orthogonally, so the dot products should match
        // the original-basis reference.
        let signs1 = rdna_compute::gen_fwht_signs(SIGNS1_SEED, GROUP_SIZE);
        let signs2 = rdna_compute::gen_fwht_signs(SIGNS2_SEED, GROUP_SIZE);
        let mut act_rot = act_f32.clone();
        for g in 0..(cols / GROUP_SIZE) {
            let mut grp: [f32; 128] = act_rot[g * 128..(g + 1) * 128].try_into().unwrap();
            cpu_fwht_128(&mut grp, &signs1, &signs2);
            act_rot[g * 128..(g + 1) * 128].copy_from_slice(&grp);
        }

        let groups_per_row = cols / 128;
        let dequant_dot = |m: usize| -> f32 {
            let mut acc = 0f32;
            for g in 0..groups_per_row {
                let off = m * groups_per_row * 72 + g * 72;
                let scale = f32::from_le_bytes(enc.bytes[off..off + 4].try_into().unwrap());
                let zero = f32::from_le_bytes(enc.bytes[off + 4..off + 8].try_into().unwrap());
                for i in 0..128 {
                    let byte = enc.bytes[off + 8 + i / 2];
                    let q = if i & 1 == 0 {
                        (byte & 0xF) as f32
                    } else {
                        (byte >> 4) as f32
                    };
                    // Convention A: w_dq = scale * q + zero  (zero = min_v)
                    let w_dq = scale * q + zero;
                    acc += w_dq * act_rot[g * 128 + i];
                }
            }
            acc
        };

        for m in 0..rows {
            let got = dequant_dot(m);
            let exp = ref_y[m];
            let err = (got - exp).abs();
            // Q4 quant has ~4-5% per-element error; integrated over 256 elements
            // with mixed signs the dot product can be off by a larger amount.
            // Tolerance: 10% relative + small absolute slack.
            let tol = exp.abs() * 0.10 + 0.5;
            assert!(
                err < tol,
                "row {m}: got={got} exp={exp} err={err} tol={tol}"
            );
        }
    }

    #[test]
    fn non_matching_prefixes() {
        // Plain mlp / attention prefixes never gate
        assert!(!should_quantize_la_gate(
            "model.layers.0.mlp.gate_proj",
            "gfx1151"
        ));
        assert!(!should_quantize_la_gate(
            "model.layers.0.linear_attn.in_proj_qkv",
            "gfx1151"
        ));
        assert!(!should_quantize_la_gate(
            "model.layers.5.linear_attn.out_proj",
            "gfx1151"
        ));
    }

    #[test]
    fn matching_prefixes_on_gfx1151() {
        // The env var may or may not be set during test runs; we don't assert on
        // the boolean outcome under matching prefixes. We just check that calling
        // the predicate with matching prefixes doesn't panic / error.
        let _ = should_quantize_la_gate("model.layers.5.linear_attn.in_proj_a", "gfx1151");
        let _ = should_quantize_la_gate("model.layers.5.linear_attn.in_proj_b", "gfx1151");
    }

    #[test]
    fn matching_prefix_on_non_gfx1151_when_env_unset() {
        // Force env-unset for this test (override Cargo's CI env if set).
        let orig = std::env::var("HIPFIRE_PARO_LA_GATES_MQ4G128").ok();
        unsafe { std::env::remove_var("HIPFIRE_PARO_LA_GATES_MQ4G128"); }
        assert!(!should_quantize_la_gate(
            "model.layers.5.linear_attn.in_proj_a",
            "gfx1100"
        ));
        assert!(!should_quantize_la_gate(
            "model.layers.5.linear_attn.in_proj_a",
            "gfx906"
        ));
        // Restore
        if let Some(v) = orig {
            unsafe { std::env::set_var("HIPFIRE_PARO_LA_GATES_MQ4G128", v); }
        }
    }
}
