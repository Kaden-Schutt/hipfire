// SPDX-License-Identifier: Apache-2.0
//! A4 activation sim-quant — the activation side of the `Oq4G256` W4A4 path.
//!
//! `oqplus_quant` bakes the W4 *weight* error; this bakes the A4 *activation*
//! error so a rotation can be scored against the grid it actually deploys onto.
//! Runtime activation quant is online and cheap: per-group symmetric int4
//! (`q ∈ [-7,7]`) with an **absmax** scale (no weight-time clip-search — that is
//! a per-tensor offline luxury the per-token activation path can't afford). No
//! rotation happens here: R1 (residual) / R3 (KV) / R4 (down) are applied to the
//! activation *upstream*; this models only the int4 round-trip the rotated
//! activation then suffers.
//!
//! The point of a rotation is that it Gaussianizes the activation — spreads the
//! few high-kurtosis outlier channels (measured kurtosis > 200 in LLMs) across
//! the group so a single shared int4 scale no longer has to span an outlier and
//! clip everything else. Because an orthonormal `R` preserves per-row norm, the
//! reconstruction SNR of this round-trip *in the rotated basis* equals the
//! end-to-end activation SNR the original computation sees (see
//! [`crate::rotation::rotate_rows`]); so comparing [`snr_db`] across rotations is
//! a faithful, kernel-free measurement of a rotation's A4 quality.

/// int4 activation group width (matches `Oq4G256`).
pub const GROUP: usize = 256;

/// Per-group symmetric int4 (absmax) round-trip of a `[rows, feat]` row-major
/// activation buffer. Groups tile the `feat` dim in [`GROUP`]-wide chunks; a
/// trailing partial group uses its own absmax. Returns the dequantized fp32.
pub fn a4_simquant(x: &[f32], rows: usize, feat: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * feat);
    let mut out = vec![0.0f32; rows * feat];
    for r in 0..rows {
        let row = &x[r * feat..r * feat + feat];
        let dst = &mut out[r * feat..r * feat + feat];
        let mut g = 0;
        while g < feat {
            let end = (g + GROUP).min(feat);
            let grp = &row[g..end];
            let amax = grp.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let scale = (amax / 7.0).max(1e-12);
            let inv = 1.0 / scale;
            for (i, &v) in grp.iter().enumerate() {
                let q = (v * inv).round().clamp(-7.0, 7.0);
                dst[g + i] = q * scale;
            }
            g = end;
        }
    }
    out
}

/// Reconstruction SNR in dB: `10·log10(‖x‖² / ‖x − x̂‖²)`. Higher is better;
/// `+∞`-ish (capped) when the round-trip is lossless.
pub fn snr_db(orig: &[f32], recon: &[f32]) -> f32 {
    debug_assert_eq!(orig.len(), recon.len());
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for (&o, &r) in orig.iter().zip(recon.iter()) {
        sig += (o as f64) * (o as f64);
        let d = (o - r) as f64;
        noise += d * d;
    }
    if noise <= 0.0 {
        return 200.0;
    }
    (10.0 * (sig / noise).log10()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::{rotate_rows, Rotation};

    /// Heavy-tailed activations: a mostly-small Gaussian bulk plus a few large
    /// outlier channels — the regime int4 activation quant chokes on.
    fn heavy_tailed(rows: usize, feat: usize, seed: u64) -> Vec<f32> {
        let mut s = seed ^ 0xABCD_1234;
        let mut nxt = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // ~U(-1,1)
        };
        let mut x = vec![0.0f32; rows * feat];
        // A handful of persistent outlier channels shared across rows. The bulk
        // sits at σ≈1 (it carries real output energy, so destroying it must
        // hurt) while the outliers are ~10–14× larger — enough to inflate a
        // shared int4 group scale until the bulk rounds to zero. This is the
        // regime rotation is *for*; if the bulk were negligible (σ≪amax) you'd
        // just keep the outliers and identity would win.
        let outliers: Vec<usize> = (0..4).map(|k| (k * 61 + 7) % feat).collect();
        for r in 0..rows {
            for f in 0..feat {
                let base = nxt(); // ~U(-1,1) bulk
                x[r * feat + f] = if outliers.contains(&f) {
                    base + nxt().signum() * (10.0 + 4.0 * nxt().abs()) // 10–14× outlier
                } else {
                    base
                };
            }
        }
        x
    }

    #[test]
    fn a4_roundtrip_is_lossy_but_bounded() {
        let (rows, feat) = (4, GROUP);
        let x = heavy_tailed(rows, feat, 1);
        let q = a4_simquant(&x, rows, feat);
        let snr = snr_db(&x, &q);
        assert!(snr.is_finite() && snr > 0.0, "snr {snr} not sane");
    }

    /// Plain `y = x Wᵀ`, `x [rows,feat]`, `W [out,feat]`, both row-major.
    fn matmul_t(x: &[f32], w: &[f32], rows: usize, feat: usize, out: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * out];
        for r in 0..rows {
            for o in 0..out {
                let mut acc = 0.0f32;
                for f in 0..feat {
                    acc += x[r * feat + f] * w[o * feat + f];
                }
                y[r * out + o] = acc;
            }
        }
        y
    }

    /// The core SpinQuant claim at the A4 grid, measured *end to end* through a
    /// weight — the faithful metric. Raw-activation reconstruction SNR is a poor
    /// proxy: its Frobenius norm is dominated by the outliers, which quantize
    /// well, so it rewards the identity basis for keeping them while ignoring
    /// that the bulk channels get crushed to zero. Through a (dense) weight, that
    /// crushed bulk propagates into the output and hurts — so a Hadamard rotation,
    /// which disperses the outliers and preserves the bulk, wins on output SNR.
    ///
    /// The rotated model consumes `A4(x Rᵀ)` with weight `W Rᵀ`, i.e.
    /// `A4(xRᵀ)·(WRᵀ)ᵀ`; identity `R` recovers the unrotated `A4(x)·Wᵀ`.
    #[test]
    fn hadamard_beats_identity_end_to_end() {
        let (rows, feat, out) = (8usize, GROUP, 32usize);
        let x = heavy_tailed(rows, feat, 42);
        // Dense random weight (each output mixes all feats, so bulk loss shows).
        let mut s = 0x5151u64;
        let w: Vec<f32> = (0..out * feat)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect();
        let y_ref = matmul_t(&x, &w, rows, feat, out);
        let out_snr = |rot: &Rotation| {
            let xr = rotate_rows(&x, rot, rows); // x Rᵀ
            let xq = a4_simquant(&xr, rows, feat); // A4(x Rᵀ)
            let wr = rotate_rows(&w, rot, out); // W Rᵀ
            let yq = matmul_t(&xq, &wr, rows, feat, out); // A4(xRᵀ)(WRᵀ)ᵀ
            snr_db(&y_ref, &yq)
        };
        let s_ident = out_snr(&Rotation::identity(feat));
        let s_had = out_snr(&Rotation::hadamard(feat, 7));
        let s_rand = out_snr(&Rotation::random(feat, 7));
        println!(
            "A4 output SNR  identity={s_ident:.2} dB  hadamard={s_had:.2} dB  random={s_rand:.2} dB"
        );
        assert!(
            s_had > s_ident + 3.0,
            "hadamard {s_had:.2} not >3dB over identity {s_ident:.2}"
        );
        assert!(
            s_rand > s_ident + 3.0,
            "random {s_rand:.2} not >3dB over identity {s_ident:.2}"
        );
    }
}
