// SPDX-License-Identifier: Apache-2.0
//! OQ+ (Opus Plus, W4A8) sim-quant for recovery-FT probes.
//!
//! Reproduces the *weight* error of the production OQ+ codec
//! (`hipfire-quantize::codecs::quantize_oq4g256` + its dequant oracle) as a
//! fp32→fp32 round-trip so a differentiable student forward carries OQ+-shaped
//! damage. Per 256-group: FWHT-256 rotate → symmetric clip-searched scale →
//! round to signed int4 (q ∈ [-7,7]) → dequant (`scale·q`) → inverse FWHT.
//!
//! Scope note: this bakes the **W4** weight-quant error only. OQ+ also int8-
//! quantizes activations at runtime (the "A8"), but the sweep in commit cf387d42
//! shows A8 adds ~negligible KLD over A16 (oq8 W8A8 0.00156 vs q8f16 W8A16
//! 0.00101) while the W8→W4 weight step is the dominant 0.15 KLD — so weight-only
//! sim-quant is the faithful first-cut recovery target. The FWHT sign basis uses
//! the production seeds (42 / 1042); any fixed orthonormal basis yields the same
//! error statistics, so this is faithful for a mechanism probe.

use hipfire_kvquant::fwht::{cpu_fwht_256, gen_fwht_signs};

const GROUP: usize = 256;

/// Symmetric clip-searched scale (mirrors codecs::symmetric_clipsearch).
fn symmetric_clipsearch(group: &[f32], qmax: f32) -> f32 {
    const CLIP_GRID: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];
    let amax = group.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let (mut best_scale, mut best_err) = (amax / qmax, f32::INFINITY);
    for &c in &CLIP_GRID {
        let scale = (c * amax / qmax).max(1e-12);
        let inv = 1.0 / scale;
        let mut err = 0.0f32;
        for &v in group.iter() {
            let q = (v * inv).round().clamp(-qmax, qmax);
            let d = v - q * scale;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_scale = scale;
        }
    }
    best_scale.max(1e-12)
}

/// OQ+ weight sim-quant: `dequant(quantize(w))` with the symmetric-int4 +
/// FWHT-256 + clip-search codec. Returns fp32 weights carrying OQ+ error.
/// `w` is flattened row-major; trailing partial groups are zero-padded to 256
/// (matching the codec) and truncated back on output.
pub fn oqplus_simquant(w: &[f32]) -> Vec<f32> {
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);
    let n = w.len();
    let n_blocks = n.div_ceil(GROUP);
    let mut out = Vec::with_capacity(n_blocks * GROUP);
    for b in 0..n_blocks {
        let start = b * GROUP;
        let end = (start + GROUP).min(n);
        let mut group = [0.0f32; GROUP];
        group[..end - start].copy_from_slice(&w[start..end]);
        // forward rotate
        cpu_fwht_256(&mut group, &signs1, &signs2);
        // symmetric int4 quant (q in [-7,7]), then dequant in the rotated domain
        let scale = symmetric_clipsearch(&group, 7.0);
        let inv = 1.0 / scale;
        for v in group.iter_mut() {
            let q = (*v * inv).round().clamp(-7.0, 7.0);
            *v = q * scale;
        }
        // inverse rotate (swap signs)
        cpu_fwht_256(&mut group, &signs2, &signs1);
        out.extend_from_slice(&group);
    }
    out.truncate(n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_lossy_but_bounded() {
        // Gaussian-ish input: error should be present but small relative to amax.
        let mut s = 1u32;
        let w: Vec<f32> = (0..1024)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
                (s as f32 / 2147483648.0 - 0.5) * 2.0
            })
            .collect();
        let q = oqplus_simquant(&w);
        assert_eq!(q.len(), w.len());
        let mse: f32 = w.iter().zip(&q).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / w.len() as f32;
        let var: f32 = w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32;
        let sqnr_db = 10.0 * (var / mse).log10();
        // 4-bit symmetric should land in a sane SQNR band, not identity, not garbage.
        assert!(sqnr_db > 8.0 && sqnr_db < 40.0, "sqnr {sqnr_db} dB out of band");
    }
}
