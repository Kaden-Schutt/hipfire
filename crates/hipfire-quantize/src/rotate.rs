// SPDX-License-Identifier: Apache-2.0
//! SpinQuant R1 deploy-time merge (item 2, path (a)): fold RMSNorm scales into the
//! readers and rotate the residual readers/writers by `R1 = Fᵀ M` **before** the
//! Oq4G256 quantize, so the codec's per-256-group FWHT `F` cancels the `Fᵀ` and the
//! int4 grid sees the learned rotation `M` (the +1.7 dB optimum from
//! `hipfire-train`'s `learned_r1_w4a4_probe`).
//!
//! This is the in-quantizer re-implementation of `hipfire_train::rotation::apply_r1`
//! (the quantizer can't depend on hipfire-train). The host math is identical —
//! `fold_cols` / `rotate_input` (readers, `W → W R1ᵀ`) / `rotate_output` (writers,
//! `W → R1 W`) — and the bake identity (`rotate_input(W, FᵀM)` then codec FWHT `==`
//! `rotate_input(W, M)`) is unit-tested here against the same `cpu_fwht_256` the
//! codec uses.
//!
//! The learned `M` is produced by `hipfire-train`'s `learn_r1_dump` example as a
//! `.r1` file: `b"HFR1"` + `h: u32 LE` + `h*h` f32 LE (row-major). The quantizer
//! composes `Fᵀ M` itself (codec-agnostic input), so the same `.r1` deploys through
//! any FWHT-256 Opus codec.
//!
//! Applies to **dense llama** (arch_id 0/1). The residual rotation touches every
//! q/k/v/gate/up reader (fold input/post norm, then rotate), o_proj/down_proj
//! writer (rotate), the input embedding (rotate), and the head (fold final norm,
//! rotate). Tied models are untied: a separate rotated `lm_head.weight` is emitted
//! and the runtime auto-prefers it (see `hfq.rs` loader).

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};

/// The codec's per-256-group FWHT seeds (must match `quantize_oq4g256` /
/// `quantize_oq8g256`, which use `gen_fwht_signs(42|1042, 256)`).
const FWHT_SEED1: u32 = 42;
const FWHT_SEED2: u32 = 1042;

/// The baked residual rotation `R1 = Fᵀ M` (dense `[h,h]`, row-major), where `F` is
/// the codec's per-256-group FWHT. Built once from the learned `M`; then applied to
/// every reader/writer/embedding so the codec's `F` leaves the int4 grid in `M`.
pub struct R1Plan {
    pub h: usize,
    r1: Vec<f32>,
}

impl R1Plan {
    /// Compose `R1 = Fᵀ M` from the learned rotation `m` (`[h,h]` row-major).
    pub fn from_learned_m(m: &[f32], h: usize) -> Self {
        assert_eq!(m.len(), h * h, "M must be [h,h]");
        assert_eq!(h % 256, 0, "h must be a multiple of 256 for the codec FWHT");
        let ft = block_fwht_transpose(h);
        // r1 = Fᵀ · M (row-major [h,h] · [h,h]).
        let mut r1 = vec![0.0f32; h * h];
        for i in 0..h {
            for k in 0..h {
                let a = ft[i * h + k];
                if a == 0.0 {
                    continue;
                }
                let mrow = &m[k * h..k * h + h];
                let orow = &mut r1[i * h..i * h + h];
                for (o, &mv) in orow.iter_mut().zip(mrow.iter()) {
                    *o += a * mv;
                }
            }
        }
        Self { h, r1 }
    }

    /// `max |R1 R1ᵀ − I|` — orthonormality residual (a correctness probe).
    pub fn orthonormality_error(&self) -> f32 {
        orthonormality_error(&self.r1, self.h)
    }

    /// Reader rotate on the hidden dim: `W [out,h] → W R1ᵀ`.
    /// `new[o,j] = Σ_i W[o,i] R1[j,i]`.
    pub fn rotate_reader(&self, w: &mut Vec<f32>, out: usize) {
        *w = rotate_input(w, &self.r1, out, self.h);
    }

    /// Writer rotate on the hidden output dim: `W [h,cols] → R1 W`.
    /// `new[e,c] = Σ_d R1[e,d] W[d,c]`.
    pub fn rotate_writer(&self, w: &mut Vec<f32>, cols: usize) {
        *w = rotate_output(w, &self.r1, self.h, cols);
    }
}

/// Fold a per-column RMSNorm scale into a reader: `W[o,i] *= alpha[i]`. Mirrors
/// `hipfire_train::rotation::fold_cols`. `W` is `[out, h]` row-major, `alpha` `[h]`.
pub fn fold_cols(w: &mut [f32], alpha: &[f32], out: usize, h: usize) {
    assert_eq!(w.len(), out * h);
    assert_eq!(alpha.len(), h);
    for o in 0..out {
        let row = &mut w[o * h..o * h + h];
        for (v, &a) in row.iter_mut().zip(alpha.iter()) {
            *v *= a;
        }
    }
}

/// Reader rotate `W → W Rᵀ` on the input (`h`) dim of a `[out, h]` weight.
fn rotate_input(w: &[f32], r: &[f32], out: usize, h: usize) -> Vec<f32> {
    let mut o_out = vec![0.0f32; out * h];
    for o in 0..out {
        let src = &w[o * h..o * h + h];
        let dst = &mut o_out[o * h..o * h + h];
        for (j, d) in dst.iter_mut().enumerate() {
            let rrow = &r[j * h..j * h + h];
            let mut acc = 0.0f32;
            for (s, rr) in src.iter().zip(rrow.iter()) {
                acc += s * rr;
            }
            *d = acc;
        }
    }
    o_out
}

/// Writer rotate `W → R W` on the output (`h`) dim of a `[h, cols]` weight.
fn rotate_output(w: &[f32], r: &[f32], h: usize, cols: usize) -> Vec<f32> {
    let mut o_out = vec![0.0f32; h * cols];
    for e in 0..h {
        let rrow = &r[e * h..e * h + h];
        let dst = &mut o_out[e * cols..e * cols + cols];
        for (d, &rval) in rrow.iter().enumerate() {
            if rval == 0.0 {
                continue;
            }
            let src = &w[d * cols..d * cols + cols];
            for (o, s) in dst.iter_mut().zip(src.iter()) {
                *o += rval * s;
            }
        }
    }
    o_out
}

/// Dense `Fᵀ` `[h,h]`, where `F` is the codec's block-diagonal per-256-group FWHT.
/// Column `i` of each 256-block of `F` is `cpu_fwht_256(e_i)` (so `rotate_rows(·,F)`
/// reproduces the codec's `fwht_rows`); `Fᵀ` transposes within each block.
fn block_fwht_transpose(h: usize) -> Vec<f32> {
    let (s1, s2) = (
        gen_fwht_signs(FWHT_SEED1, 256),
        gen_fwht_signs(FWHT_SEED2, 256),
    );
    let mut block = vec![0.0f32; 256 * 256]; // F block (row-major)
    for i in 0..256 {
        let mut e = [0.0f32; 256];
        e[i] = 1.0;
        cpu_fwht_256(&mut e, &s1, &s2);
        for (j, &v) in e.iter().enumerate() {
            block[j * 256 + i] = v;
        }
    }
    // Fᵀ: scatter the transposed block down the diagonal of an [h,h] matrix.
    let mut ft = vec![0.0f32; h * h];
    for b in 0..(h / 256) {
        let off = b * 256;
        for j in 0..256 {
            for i in 0..256 {
                ft[(off + i) * h + (off + j)] = block[j * 256 + i];
            }
        }
    }
    ft
}

fn orthonormality_error(r: &[f32], h: usize) -> f32 {
    let mut worst = 0.0f32;
    for i in 0..h {
        for j in 0..h {
            let mut dot = 0.0f32;
            for k in 0..h {
                dot += r[i * h + k] * r[j * h + k];
            }
            let t = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - t).abs());
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic orthonormal matrix (random-sign Hadamard, `h` power of
    /// two) to stand in for a learned M in the bake identity test.
    fn hadamard(h: usize, seed: u64) -> Vec<f32> {
        assert!(h.is_power_of_two());
        let scale = 1.0 / (h as f32).sqrt();
        let mut r = vec![0.0f32; h * h];
        for i in 0..h {
            for j in 0..h {
                let parity = (i & j).count_ones() & 1;
                r[i * h + j] = if parity == 0 { scale } else { -scale };
            }
        }
        let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
        for j in 0..h {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (state >> 63) & 1 == 1 {
                for i in 0..h {
                    r[i * h + j] = -r[i * h + j];
                }
            }
        }
        r
    }

    #[test]
    fn r1_plan_is_orthonormal() {
        let h = 256;
        let m = hadamard(h, 7);
        let plan = R1Plan::from_learned_m(&m, h);
        assert!(
            plan.orthonormality_error() < 1e-4,
            "R1=FᵀM not orthonormal: {}",
            plan.orthonormality_error()
        );
    }

    /// The bake identity: rotating a reader by `R1 = FᵀM` and THEN applying the
    /// codec's per-256-group FWHT (what `quantize_oq4g256` does to each row) equals
    /// rotating the reader by `M` directly — i.e. the codec's `F` cancels the `Fᵀ`
    /// and the int4 grid sees the learned rotation `M`. This is the quantizer-side
    /// analog of `rotation::bake_composes_to_learned_through_codec_fwht`.
    #[test]
    fn bake_cancels_codec_fwht_leaving_m() {
        let (h, out) = (256usize, 20usize);
        let m = hadamard(h, 13);
        let plan = R1Plan::from_learned_m(&m, h);

        // A deterministic reader weight [out, h].
        let mut w = vec![0.0f32; out * h];
        let mut s = 0x1234_5678u64;
        for v in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
        }

        // Path 1: rotate by R1=FᵀM, then apply the codec FWHT per 256-group per row.
        let mut w_r1 = w.clone();
        plan.rotate_reader(&mut w_r1, out);
        let (s1, s2) = (
            gen_fwht_signs(FWHT_SEED1, 256),
            gen_fwht_signs(FWHT_SEED2, 256),
        );
        for o in 0..out {
            for g in 0..(h / 256) {
                let base = o * h + g * 256;
                let mut buf = [0.0f32; 256];
                buf.copy_from_slice(&w_r1[base..base + 256]);
                cpu_fwht_256(&mut buf, &s1, &s2);
                w_r1[base..base + 256].copy_from_slice(&buf);
            }
        }

        // Path 2: rotate directly by M (the target the int4 grid should see).
        let w_m = rotate_input(&w, &m, out, h);

        let worst = w_r1
            .iter()
            .zip(&w_m)
            .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs()));
        assert!(
            worst < 1e-3,
            "bake did not cancel codec FWHT: max|Δ| {worst:.2e}"
        );
    }
}
