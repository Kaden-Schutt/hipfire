// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Maple-Preview onboarding: native-ternary detection and exact packing.
//!
//! `deepgrove/maple-preview` publishes DEQUANTIZED bf16 masters (40.4 GB) whose
//! values are already ternary: every linear row is `{-s_r, 0, +s_r}` with one
//! bf16 scale per OUTPUT ROW. Nothing in the published `modeling_maple.py`
//! quantizes anything — the `quantize: true` config key is dead. The structure
//! lives in the values.
//!
//! So this is a packing problem, not a quantization problem, and the bar is
//! exactness. We verify ternary-ness per ROW (a per-tensor summary would
//! happily average over a bad row) and refuse anything else, because a silent
//! lossy fallback would be indistinguishable from success right up until the
//! model generated garbage.

use crate::quant_mq::quantize_mq2g256_ternary_exact;

/// What to do with one Maple tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MapleTensorPolicy {
    /// Native-ternary linear: pack exactly into MQ2G256LloydU (qt=51).
    Ternary,
    /// Router, embeddings, lm_head, norms — measured full-precision on the
    /// published checkpoint, so they keep a higher-precision carrier.
    KeepHighPrecision,
}

/// Carrier for `lm_head.weight`.
///
/// The head is the ONE high-precision tensor where the carrier is a real
/// decode-speed decision rather than a fidelity one. It is dense over the full
/// 151,936-row vocab and is read in its entirety EVERY token: 622 MB of BF16,
/// measured at 205 GB/s — 90% of this box's 227.7 GB/s achievable ceiling and
/// 36% of the decode token (`.superpowers/sdd/maple-decode-profile.md`). It is
/// the only part of the model that is actually bandwidth-bound, so shrinking it
/// converts directly into tokens/s.
///
/// **`word_embeddings` is deliberately NOT covered by this option** even though
/// it is the same `[151936, 2048]` shape and the same 622 MB. Exactly ONE ROW of
/// it is read per token, so quantizing it would save RAM (the loader currently
/// widens it to F32 = 1.24 GB resident) and NOT bandwidth. That is a separate
/// question with a separate trade-off; conflating the two under one flag would
/// let a RAM decision silently change output quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MapleHeadQuant {
    /// Carry the head verbatim as BF16 (qt=16). The default: existing behaviour.
    #[default]
    Bf16,
    /// Q8_0 / Q8F16 (qt=3), 34 B per 32 weights = 1.0625 bpw. 331 MB.
    /// NOT FWHT-rotated — `dtype_rotation_plan(Q8_0) == None`.
    Q8,
    /// MQ4-G256-Lloyd (qt=30), 160 B per 256 weights = 5.0 bpw. 195 MB.
    /// **FWHT-rotated**: the weights are encoded against FWHT-256-rotated
    /// blocks, so the runtime MUST rotate `x` to match. See
    /// `pack_maple_head` for why the seeds are not free parameters.
    ///
    /// **DEPRECATED — use `Mq4V2`.** Measured on gfx1151 (KV bf16, 2048
    /// teacher-forced tokens): qt=44 is better on EVERY axis — mean KL 0.0744
    /// vs 0.0772, decode 165.8 vs 161.8 tok/s, and 4.25 vs 5.0 bpw. There is
    /// no workload where qt=30 is the right choice. Kept only so the packer
    /// arm and its FWHT-seed contract stay documented next to qt=44's; the
    /// CLI no longer offers it.
    Mq4,
    /// MQ4-G256 **v2** (qt=44), 136 B per 256 weights = 4.25 bpw.
    /// **FWHT-rotated**, same as `Mq4`, and the same nibble payload — but the
    /// 8 header bytes carry a SEPARATE fp16 scale/zero per 128-weight half
    /// instead of one pair governing all 256. Strictly finer quantization at a
    /// SMALLER footprint than qt=30 (4.25 vs 5.0 bpw), so it is the natural
    /// candidate if the mq4 head's accuracy cost is what rules it out.
    Mq4V2,
}

impl std::str::FromStr for MapleHeadQuant {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "bf16" | "none" => Ok(Self::Bf16),
            "q8" | "q8_0" | "q8f16" => Ok(Self::Q8),
            "mq4" | "mq4-lloyd" | "mq4g256lloyd" => Ok(Self::Mq4),
            "mq4v2" | "mq4-v2" | "mq4g256v2" => Ok(Self::Mq4V2),
            other => Err(format!(
                "unknown --head-quant {other:?} (expected bf16, q8, mq4 or mq4v2)"
            )),
        }
    }
}

impl MapleHeadQuant {
    /// Name used in the convert log and stamped into the HFQ provenance.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
            Self::Mq4 => "mq4",
            Self::Mq4V2 => "mq4v2",
        }
    }
}

/// Per-row ternary summary, for provenance and for the convert log.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TernaryRowStats {
    pub rows: usize,
    pub nonzero_frac: f64,
    pub scale_min: f32,
    pub scale_max: f32,
}

/// Route a Maple tensor by name.
///
/// The router is `mlp.gate.weight`; an expert's gate is
/// `mlp.experts.N.gate_proj.weight`. Matching on the substring "gate" would
/// conflate them, so match full suffixes only.
pub(crate) fn maple_tensor_policy(name: &str) -> MapleTensorPolicy {
    const TERNARY_SUFFIXES: &[&str] = &[
        ".q_proj.weight",
        ".k_proj.weight",
        ".v_proj.weight",
        ".o_proj.weight",
        ".gate_proj.weight",
        ".up_proj.weight",
        ".down_proj.weight",
    ];
    if TERNARY_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        MapleTensorPolicy::Ternary
    } else {
        MapleTensorPolicy::KeepHighPrecision
    }
}

/// Pack `lm_head.weight` with the requested carrier.
///
/// Returns `(bytes, quant_type, group_size)`. `Bf16` is not handled here — the
/// caller carries the source bytes verbatim rather than round-tripping them
/// through f32.
///
/// **The FWHT sign seeds are 42 and 1042 and they are NOT free parameters.**
/// `quantize_mq4g256_lloyd` rotates each 256-block by `cpu_fwht_256` before
/// fitting its codebook, and at runtime `weight_gemv` rotates `x` with the
/// signs `Scratch::ensure_mq_signs` builds — which are hardcoded
/// `gen_fwht_signs(42, 256)` / `gen_fwht_signs(1042, 256)`. FWHT is orthogonal,
/// so `<Wrot, xrot> == <W, x>` holds ONLY when both sides used the same signs.
/// A mismatch is not an error and not a crash: it is a plausible-looking
/// logit vector that is entirely wrong. Every other call site in the tree
/// (`pipeline.rs`, `pipeline_gguf.rs`, `pipeline_deepseek.rs`,
/// `reap_overlay.rs`) uses these same two seeds for G256.
///
/// `K` must be a multiple of 256 for MQ4 and of 32 for Q8. Maple's head is
/// `[151936, 2048]`, so both divide exactly and no group ever straddles a row
/// boundary — the flat-array packers are safe to use directly.
pub(crate) fn pack_maple_head(
    vals: &[f32],
    k: usize,
    how: MapleHeadQuant,
) -> Result<(Vec<u8>, crate::hfq::QuantType, u32), String> {
    use crate::hfq::QuantType;
    match how {
        MapleHeadQuant::Bf16 => Err("pack_maple_head called with Bf16".to_string()),
        MapleHeadQuant::Q8 => {
            if k % 32 != 0 {
                return Err(format!(
                    "lm_head K={k} is not a multiple of 32 (Q8_0 block)"
                ));
            }
            Ok((crate::quant_q4::quantize_q8f16(vals), QuantType::Q8F16, 32))
        }
        MapleHeadQuant::Mq4 => {
            if k % 256 != 0 {
                return Err(format!(
                    "lm_head K={k} is not a multiple of 256 (MQ4-G256 block)"
                ));
            }
            let signs1 = crate::quant_fwht::gen_fwht_signs(42, 256);
            let signs2 = crate::quant_fwht::gen_fwht_signs(1042, 256);
            Ok((
                crate::quant_mq::quantize_mq4g256_lloyd(vals, &signs1, &signs2),
                QuantType::MQ4G256Lloyd,
                256,
            ))
        }
        MapleHeadQuant::Mq4V2 => {
            if k % 256 != 0 {
                return Err(format!(
                    "lm_head K={k} is not a multiple of 256 (MQ4-G256 block)"
                ));
            }
            // Same FWHT seeds as the qt=30 arm above. They are NOT free
            // parameters: the runtime rotates `x` with signs derived from the
            // same seeds, so a mismatch here silently produces garbage logits
            // rather than a load error.
            let signs1 = crate::quant_fwht::gen_fwht_signs(42, 256);
            let signs2 = crate::quant_fwht::gen_fwht_signs(1042, 256);
            let m = vals.len() / k;
            Ok((
                crate::quant_fwht::quantize_mq4g256v2(vals, m, k, &signs1, &signs2),
                QuantType::MQ4G256V2,
                256,
            ))
        }
    }
}

/// Verify every row of a `[M, K]` row-major tensor is exactly `{-s_r, 0, +s_r}`.
///
/// Checked per row, not per tensor: rows carry independent scales, so a
/// whole-tensor `unique(|w|)` would see many values and prove nothing, while a
/// whole-tensor summary statistic would silently tolerate one dense row.
pub(crate) fn classify_ternary_rows(data: &[f32], k: usize) -> Result<TernaryRowStats, String> {
    if k == 0 || data.len() % k != 0 {
        return Err(format!("length {} is not a multiple of K={k}", data.len()));
    }
    let rows = data.len() / k;
    let mut nonzero = 0usize;
    let mut scale_min = f32::INFINITY;
    let mut scale_max = 0.0f32;

    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        let mut scale: Option<f32> = None;
        for &w in row {
            // -0.0 == 0.0 in IEEE comparison, which is what we want: the ~19%
            // of Maple weights stored as -0.0 are zeros, not a third magnitude.
            if w == 0.0 {
                continue;
            }
            if !w.is_finite() {
                return Err(format!("row {r} contains a non-finite weight ({w})"));
            }
            nonzero += 1;
            let a = w.abs();
            match scale {
                None => scale = Some(a),
                Some(s) if s == a => {}
                Some(s) => {
                    return Err(format!(
                        "row {r} is not ternary: |w| takes at least two values ({s} and {a})"
                    ))
                }
            }
        }
        if let Some(s) = scale {
            scale_min = scale_min.min(s);
            scale_max = scale_max.max(s);
        }
    }
    if !scale_min.is_finite() {
        // Every row was all-zero. Legal (a fully pruned tensor) but worth
        // reporting as 0 rather than INFINITY.
        scale_min = 0.0;
    }
    Ok(TernaryRowStats {
        rows,
        nonzero_frac: nonzero as f64 / data.len() as f64,
        scale_min,
        scale_max,
    })
}

/// Verify, then pack into MQ2G256LloydU.
///
/// `k % 256 == 0` is required so no 256-block ever straddles a row boundary —
/// if one did, it would span two different row scales and hold up to 5 distinct
/// values, and the exactness guarantee would silently depend on tensor shape.
pub(crate) fn pack_maple_tensor(
    data: &[f32],
    k: usize,
) -> Result<(Vec<u8>, TernaryRowStats), String> {
    if k % 256 != 0 {
        return Err(format!(
            "K={k} must be a multiple of 256 so no 256-block straddles two rows"
        ));
    }
    let stats = classify_ternary_rows(data, k)?;
    let bytes = quantize_mq2g256_ternary_exact(data)?;
    Ok((bytes, stats))
}

#[cfg(test)]
mod tests {
    use super::MapleTensorPolicy::*;
    use super::*;

    fn ternary_row(k: usize, s: f32, phase: usize) -> Vec<f32> {
        (0..k)
            .map(|i| match (i + phase) % 3 {
                0 => -s,
                1 => 0.0,
                _ => s,
            })
            .collect()
    }

    #[test]
    fn classify_accepts_per_row_ternary_with_a_different_scale_per_row() {
        // THE property under test is PER-ROW scaling, so the two rows must have
        // DIFFERENT scales — with equal scales this test would also pass
        // against a (wrong) per-tensor implementation and prove nothing.
        let k = 256;
        let mut data = ternary_row(k, 0.0125, 0);
        data.extend(ternary_row(k, 0.0625, 1));
        let st = classify_ternary_rows(&data, k).unwrap();
        assert_eq!(st.rows, 2);
        assert_eq!(st.scale_min, 0.0125);
        assert_eq!(st.scale_max, 0.0625);
        assert!(st.nonzero_frac > 0.6 && st.nonzero_frac < 0.7);
    }

    #[test]
    fn classify_rejects_a_single_non_ternary_row() {
        // NEGATIVE CONTROL: row 0 ternary, row 1 dense. A per-tensor check or a
        // summary statistic would tolerate this.
        let k = 256;
        let mut data = ternary_row(k, 0.02, 0);
        data.extend((0..k).map(|i| i as f32 * 0.001));
        let err = classify_ternary_rows(&data, k).unwrap_err();
        assert!(
            err.contains("row 1"),
            "must name the offending row; got: {err}"
        );
    }

    #[test]
    fn classify_rejects_a_row_with_two_magnitudes() {
        // The subtle corruption: still 3 distinct values per row, but they are
        // {-a, 0, +b} with a != b, which is NOT ternary and would silently lose
        // the larger magnitude if we only counted distinct levels.
        let k = 256;
        let mut data = ternary_row(k, 0.02, 0);
        data[7] = 0.04;
        let err = classify_ternary_rows(&data, k).unwrap_err();
        assert!(err.contains("not ternary"), "got: {err}");
    }

    #[test]
    fn classify_accepts_an_all_zero_row() {
        let k = 256;
        let data = vec![0.0f32; k];
        let st = classify_ternary_rows(&data, k).unwrap();
        assert_eq!(st.nonzero_frac, 0.0);
        assert_eq!(st.scale_min, 0.0);
    }

    #[test]
    fn pack_maple_tensor_round_trips_exactly() {
        let k = 512;
        let s = 0.024169922f32; // a real Maple row scale
        let mut data = ternary_row(k, s, 0);
        data.extend(ternary_row(k, 0.0625, 1));
        let (packed, _) = pack_maple_tensor(&data, k).unwrap();
        let recon = crate::quant_mq::dequantize_mq2g256_lloyd_u_to_f32(&packed, data.len());
        let max_err = data
            .iter()
            .zip(&recon)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(max_err, 0.0);
    }

    #[test]
    fn pack_refuses_a_k_that_would_straddle_rows() {
        // K=384 is a multiple of 128 but not 256, so block 1 would cover the
        // tail of row 0 and the head of row 1 — two different scales.
        let data = vec![0.0f32; 384 * 2];
        let err = pack_maple_tensor(&data, 384).unwrap_err();
        assert!(err.contains("multiple of 256"), "got: {err}");
    }

    /// Cross-implementation parity against REAL published Maple weights.
    ///
    /// Every other test here round-trips through OUR OWN dequantizer, so a
    /// shared misunderstanding of the 72 B/group layout would round-trip
    /// perfectly and still feed the GPU kernel garbage. This one compares our
    /// packed bytes against an independent Python implementation
    /// (`python3 -m tools.models.maple.b0_ternary_exactness_spike`) on a real tensor.
    ///
    /// Fixtures are ~4 MB and not committed. Produce them with that script,
    /// then:
    ///
    ///   MAPLE_FIXTURE_IN=/path/fixture_input.f32 \
    ///   MAPLE_FIXTURE_EXPECTED=/path/fixture_expected.bin \
    ///     cargo test -p hipfire-quantize maple_fixture_parity -- --nocapture
    ///
    /// Skips when unset so CI stays hermetic — but if MAPLE_FIXTURE_IN is set
    /// and unreadable it FAILS rather than skipping, so a typo in the path
    /// cannot masquerade as a pass.
    #[test]
    fn maple_fixture_parity_against_independent_packer() {
        let (Ok(in_path), Ok(exp_path)) = (
            std::env::var("MAPLE_FIXTURE_IN"),
            std::env::var("MAPLE_FIXTURE_EXPECTED"),
        ) else {
            eprintln!("skipped: MAPLE_FIXTURE_IN / MAPLE_FIXTURE_EXPECTED not set");
            return;
        };
        let raw = std::fs::read(&in_path)
            .unwrap_or_else(|e| panic!("MAPLE_FIXTURE_IN={in_path} unreadable: {e}"));
        let expected = std::fs::read(&exp_path)
            .unwrap_or_else(|e| panic!("MAPLE_FIXTURE_EXPECTED={exp_path} unreadable: {e}"));
        assert_eq!(raw.len() % 4, 0, "input is not a whole number of f32");
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Maple expert gate_proj/up_proj and every attention projection use
        // K=2048; down_proj uses K=512. Both are multiples of 256.
        let (packed, _) =
            pack_maple_tensor(&vals, 2048).expect("real Maple weights must be ternary");
        assert_eq!(packed.len(), expected.len(), "packed length mismatch");

        if let Some(first) = packed.iter().zip(&expected).position(|(a, b)| a != b) {
            let diffs = packed.iter().zip(&expected).filter(|(a, b)| a != b).count();
            panic!(
                "{diffs} differing bytes; first at offset {first} (block {}, \
                 byte-in-block {}): got {:#04x}, expected {:#04x}",
                first / 72,
                first % 72,
                packed[first],
                expected[first]
            );
        }

        let recon = crate::quant_mq::dequantize_mq2g256_lloyd_u_to_f32(&packed, vals.len());
        let max_err = vals
            .iter()
            .zip(&recon)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(max_err, 0.0, "packing must be value-exact on real weights");
        eprintln!(
            "parity OK: {} weights, {} packed bytes, byte-identical, max|err|=0",
            vals.len(),
            packed.len()
        );
    }

    #[test]
    fn tensor_policy_routes_published_maple_names() {
        for n in [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.7.mlp.experts.255.down_proj.weight",
            "model.layers.7.mlp.experts.255.up_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.3.self_attn.o_proj.weight",
        ] {
            assert_eq!(maple_tensor_policy(n), Ternary, "{n}");
        }
        // Measured full-precision on the published checkpoint. Note that
        // `mlp.gate.weight` is the ROUTER — conflating it with an expert's
        // `gate_proj` would try to pack a dense tensor as ternary and abort the
        // whole convert.
        for n in [
            "model.layers.0.mlp.gate.weight",
            "model.word_embeddings.weight",
            "lm_head.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.norm.weight",
        ] {
            assert_eq!(maple_tensor_policy(n), KeepHighPrecision, "{n}");
        }
    }
}
