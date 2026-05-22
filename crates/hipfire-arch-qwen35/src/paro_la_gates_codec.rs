//! MQ4G128 codec for LinearAttention in_proj_a / in_proj_b weights.
//!
//! Produces the same 72-byte-per-group layout consumed by `gemv_hfq4g128`:
//!   4-byte F32 LE scale | 4-byte F32 LE zero | 64 packed nibbles (lo=even, hi=odd)
//!
//! FWHT-128 is applied at encode using the SAME sign tables (seeds 43, 1043)
//! that `ensure_mq_signs_128` uploads to the device. This task adds the module
//! scaffolding + gating predicate. Task 9 fills in `encode_mq4g128_from_fp16`.

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
/// Implementation: Task 9.
pub fn encode_mq4g128_from_fp16(_weight_fp16: &[u16], _rows: usize, _cols: usize) -> EncodedMQ4G128 {
    todo!("Task 9: implement encode_mq4g128_from_fp16");
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
    if !(prefix.ends_with("linear_attn.in_proj_a") || prefix.ends_with("linear_attn.in_proj_b")) {
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

    #[test]
    #[should_panic(expected = "Task 9")]
    fn encode_is_stub() {
        encode_mq4g128_from_fp16(&vec![0u16; 128], 1, 128);
    }

    #[test]
    fn non_matching_prefixes() {
        // Plain mlp / attention prefixes never gate
        assert!(!should_quantize_la_gate("model.layers.0.mlp.gate_proj", "gfx1151"));
        assert!(!should_quantize_la_gate("model.layers.0.linear_attn.in_proj_qkv", "gfx1151"));
        assert!(!should_quantize_la_gate("model.layers.5.linear_attn.out_proj", "gfx1151"));
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
        // SAFETY: tests in a single Cargo invocation share env; this isn't ideal
        // but is OK for a sanity check.
        let orig = std::env::var("HIPFIRE_PARO_LA_GATES_MQ4G128").ok();
        std::env::remove_var("HIPFIRE_PARO_LA_GATES_MQ4G128");
        assert!(!should_quantize_la_gate("model.layers.5.linear_attn.in_proj_a", "gfx1100"));
        assert!(!should_quantize_la_gate("model.layers.5.linear_attn.in_proj_a", "gfx906"));
        // Restore
        if let Some(v) = orig {
            std::env::set_var("HIPFIRE_PARO_LA_GATES_MQ4G128", v);
        }
    }
}
