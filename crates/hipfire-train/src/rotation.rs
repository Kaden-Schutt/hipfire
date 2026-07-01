// SPDX-License-Identifier: Apache-2.0
//! SpinQuant R1: rotation-invariant residual-stream transform (Phase 0).
//!
//! The full-precision pre-norm transformer is *rotation-invariant*: inserting an
//! orthonormal `R` on the residual stream and its inverse at matched points
//! leaves the fp output unchanged, but rotates the basis the quantizer sees
//! (SpinQuant, arXiv:2405.16406). This module bakes such an `R1` into the fp32
//! weights of [`crate::model::LlamaModel`] so a later quantize sees the rotated
//! (better-conditioned) grid. Phase 0 only proves the invariance contract; the
//! learned optimizer (Cayley SGD) lands in a later phase.
//!
//! The transform has two moves, applied together by [`apply_r1`]:
//!
//! 1. **Fold each RMSNorm scale `α` into the following weight.** LLaMA RMSNorm is
//!    `y = (x/rms(x)) ⊙ α`; the elementwise `α` is what breaks rotation
//!    invariance (`α ⊙ (xRᵀ) ≠ (α ⊙ x) Rᵀ`). Folding `α` into the *columns* of
//!    every weight that reads the norm output (SliceGPT-style) leaves a
//!    scale-free RMSNorm `y = x/rms(x)`, which **is** rotation-equivariant
//!    (`rms(xRᵀ)=rms(x)` for orthonormal `R`).
//! 2. **Rotate residual readers and writers by `R`.** A *reader* (q/k/v/gate/up,
//!    embedding-as-input, lm_head) consumes the residual: `W → W Rᵀ` on its
//!    input (`h`) dimension. A *writer* (o_proj, down_proj) adds into the
//!    residual: `W → R W` on its output (`h`) dimension. Every block shares the
//!    one global `R` (the residual basis is model-wide).
//!
//! Result: every intermediate (q,k,v,ctx,gate,up,act) is bit-for-bit unchanged
//! and the residual stream is carried in the rotated basis `x Rᵀ`, so the logits
//! match the original up to fp reassociation.
//!
//! Tied embeddings: the input embedding needs `E Rᵀ` (rotate rows) while the
//! output head needs `α_f` folded first — incompatible in one shared matrix. So
//! [`apply_r1`] **unties** the head (materializes `lm_head` from `embed`) before
//! folding/rotating. The forward path already supports an untied `lm_head`; the
//! tied-only backward is a Phase 2 concern (learning R needs the untied head
//! grad wired anyway).

use crate::model::LlamaModel;
use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use rdna_compute::{Gpu, GpuTensor, HipResult};

/// An orthonormal `[h,h]` rotation, row-major (`r[i*h + j]`). Invariant: `R Rᵀ = I`.
#[derive(Clone)]
pub struct Rotation {
    pub h: usize,
    pub r: Vec<f32>,
}

impl Rotation {
    /// Identity rotation (`apply_r1` with this is a pure norm-scale fold — a
    /// useful control: fold alone must already be bit-exact-equivalent).
    pub fn identity(h: usize) -> Self {
        let mut r = vec![0.0f32; h * h];
        for i in 0..h {
            r[i * h + i] = 1.0;
        }
        Self { h, r }
    }

    /// A random orthonormal matrix: Gram–Schmidt on a deterministic Gaussian
    /// (Box–Muller over a splitmix64 stream). Offline `O(h³)`; fine for a bake.
    pub fn random(h: usize, seed: u64) -> Self {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_u64 = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut normal = || {
            // Box–Muller; two uniforms in (0,1].
            let u1 = ((next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
            let u2 = ((next_u64() >> 11) as f64) / (1u64 << 53) as f64;
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        };
        let mut r: Vec<f32> = (0..h * h).map(|_| normal()).collect();
        gram_schmidt_rows(&mut r, h);
        Self { h, r }
    }

    /// A random-sign normalized Hadamard (`h` must be a power of two): the
    /// QuaRot / SpinQuant *fixed* rotation (+0.9 dB tier). Sylvester construction
    /// scaled by `1/√h`, with each column flipped by a deterministic ±1 sign so
    /// the rotation is data-agnostic but not the bare Hadamard. Panics if `h` is
    /// not a power of two (the residual `h` is; sub-block Hadamards for odd dims
    /// are a later concern).
    pub fn hadamard(h: usize, seed: u64) -> Self {
        assert!(h.is_power_of_two(), "hadamard size {h} not a power of two");
        let scale = 1.0 / (h as f32).sqrt();
        let mut r = vec![0.0f32; h * h];
        for i in 0..h {
            for j in 0..h {
                // Sylvester entry sign = (-1)^popcount(i & j).
                let parity = (i & j).count_ones() & 1;
                r[i * h + j] = if parity == 0 { scale } else { -scale };
            }
        }
        // Random column signs (still orthonormal: diag(±1) is orthonormal).
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
        Self { h, r }
    }

    /// The Oq4G256 codec's per-256-group FWHT as a dense block-diagonal `[h,h]`
    /// rotation `F` (signs 42/1042 — the production seeds). `rotate_rows(x, F)`
    /// reproduces the codec's `fwht_rows`, so `F` is the rotation the deployed
    /// int4 pipeline already applies. `h % 256 == 0` required.
    pub fn block_fwht(h: usize) -> Self {
        assert_eq!(h % 256, 0, "block_fwht size {h} not a multiple of 256");
        let (s1, s2) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
        // Column i of the 256-block = cpu_fwht_256(e_i) (so rotate_rows == fwht).
        let mut block = vec![0.0f32; 256 * 256];
        for i in 0..256 {
            let mut e = [0.0f32; 256];
            e[i] = 1.0;
            cpu_fwht_256(&mut e, &s1, &s2);
            for (j, &v) in e.iter().enumerate() {
                block[j * 256 + i] = v;
            }
        }
        let mut r = vec![0.0f32; h * h];
        for b in 0..(h / 256) {
            let off = b * 256;
            for j in 0..256 {
                for i in 0..256 {
                    r[(off + j) * h + (off + i)] = block[j * 256 + i];
                }
            }
        }
        Self { h, r }
    }

    /// Transpose (= inverse, since orthonormal).
    pub fn transpose(&self) -> Self {
        let h = self.h;
        let mut r = vec![0.0f32; h * h];
        for i in 0..h {
            for j in 0..h {
                r[j * h + i] = self.r[i * h + j];
            }
        }
        Self { h, r }
    }

    /// Matrix product `self · rhs` (both `[h,h]`); orthonormal × orthonormal ⇒
    /// orthonormal.
    pub fn compose(&self, rhs: &Rotation) -> Self {
        assert_eq!(self.h, rhs.h);
        let h = self.h;
        let mut r = vec![0.0f32; h * h];
        for i in 0..h {
            for k in 0..h {
                let a = self.r[i * h + k];
                if a == 0.0 {
                    continue;
                }
                for j in 0..h {
                    r[i * h + j] += a * rhs.r[k * h + j];
                }
            }
        }
        Self { h, r }
    }

    /// Re-orthonormalize the rows in place (modified Gram–Schmidt). Cheap guard
    /// against drift when an iterative process (e.g. the approximate Cayley
    /// inverse in [`crate::learn_rotation`]) accumulates small departures from
    /// `R Rᵀ = I` over many steps.
    pub fn reorthonormalize(&mut self) {
        gram_schmidt_rows(&mut self.r, self.h);
    }

    /// `max |R Rᵀ − I|` — the orthonormality residual (a correctness probe).
    pub fn orthonormality_error(&self) -> f32 {
        let h = self.h;
        let mut worst = 0.0f32;
        for i in 0..h {
            for j in 0..h {
                let mut dot = 0.0f32;
                for k in 0..h {
                    dot += self.r[i * h + k] * self.r[j * h + k];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((dot - target).abs());
            }
        }
        worst
    }
}

/// In-place modified Gram–Schmidt orthonormalization of the `h` rows of a row-
/// major `[h,h]` matrix.
fn gram_schmidt_rows(m: &mut [f32], h: usize) {
    for i in 0..h {
        // Subtract projections onto previously-orthonormalized rows.
        for j in 0..i {
            let mut dot = 0.0f32;
            for k in 0..h {
                dot += m[i * h + k] * m[j * h + k];
            }
            for k in 0..h {
                m[i * h + k] -= dot * m[j * h + k];
            }
        }
        let mut norm = 0.0f32;
        for k in 0..h {
            norm += m[i * h + k] * m[i * h + k];
        }
        let inv = 1.0 / norm.sqrt().max(1e-20);
        for k in 0..h {
            m[i * h + k] *= inv;
        }
    }
}

/// Multiply column `i` of a row-major `[out, cols]` weight by `alpha[i]` (folds
/// an RMSNorm scale into the following reader weight).
fn fold_cols(w: &mut [f32], alpha: &[f32], out: usize, cols: usize) {
    debug_assert_eq!(alpha.len(), cols);
    for o in 0..out {
        let row = &mut w[o * cols..o * cols + cols];
        for (v, &a) in row.iter_mut().zip(alpha.iter()) {
            *v *= a;
        }
    }
}

/// Rotate the `rows` of a row-major `[rows, h]` activation buffer by `Rᵀ` (each
/// row `x → x Rᵀ`) — the residual stream as the R1-transformed model carries it.
/// Orthonormal `R` preserves per-row norm, so this is a pure basis change: the
/// SNR of a later A4 round-trip in this basis equals the end-to-end activation
/// SNR in the original basis (the SpinQuant measurement). Same op as the reader
/// weight rotate, exposed for the A4-SNR probe.
pub fn rotate_rows(x: &[f32], rot: &Rotation, rows: usize) -> Vec<f32> {
    rotate_input(x, rot, rows)
}

/// Reader rotate: `W → W Rᵀ` on the input (`h`) dimension of a `[out, h]` weight.
/// `new[o,j] = Σ_i W[o,i]·R[j,i]`.
fn rotate_input(w: &[f32], rot: &Rotation, out: usize) -> Vec<f32> {
    let h = rot.h;
    let mut o_out = vec![0.0f32; out * h];
    for o in 0..out {
        let src = &w[o * h..o * h + h];
        let dst = &mut o_out[o * h..o * h + h];
        for (j, d) in dst.iter_mut().enumerate() {
            let rrow = &rot.r[j * h..j * h + h];
            let mut acc = 0.0f32;
            for (s, rr) in src.iter().zip(rrow.iter()) {
                acc += s * rr;
            }
            *d = acc;
        }
    }
    o_out
}

/// Writer rotate: `W → R W` on the output (`h`) dimension of a `[h, cols]`
/// weight. `new[e,c] = Σ_d R[e,d]·W[d,c]`.
fn rotate_output(w: &[f32], rot: &Rotation, cols: usize) -> Vec<f32> {
    let h = rot.h;
    let mut o_out = vec![0.0f32; h * cols];
    for e in 0..h {
        let rrow = &rot.r[e * h..e * h + h];
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

/// Download → transform → re-upload, replacing `slot` and freeing the old device
/// buffer (GpuTensor has no Drop).
fn replace_tensor<F>(gpu: &mut Gpu, slot: &mut GpuTensor, f: F) -> HipResult<()>
where
    F: FnOnce(Vec<f32>) -> Vec<f32>,
{
    let host = gpu.download_f32(slot)?;
    let shape = slot.shape.clone();
    let new_host = f(host);
    debug_assert_eq!(new_host.len(), shape.iter().product::<usize>());
    let newt = gpu.upload_f32(&new_host, &shape)?;
    let old = std::mem::replace(slot, newt);
    gpu.free_tensor(old)?;
    Ok(())
}

/// Set a norm weight to all-ones (its scale has been folded into the readers).
fn set_ones(gpu: &mut Gpu, slot: &mut GpuTensor) -> HipResult<()> {
    replace_tensor(gpu, slot, |v| vec![1.0f32; v.len()])
}

/// Bake SpinQuant `R1` into `model` in place: fold every RMSNorm scale into its
/// readers, then rotate residual readers/writers/embedding/head by `R`. The fp32
/// forward is left invariant (up to fp reassociation); the residual stream is now
/// carried in the `x Rᵀ` basis. Unties the head if tied (see module docs).
///
/// `R.h` must equal the model hidden size.
pub fn apply_r1(gpu: &mut Gpu, model: &mut LlamaModel, rot: &Rotation) -> HipResult<()> {
    let h = model.dims.h;
    assert_eq!(rot.h, h, "rotation size {} != hidden {}", rot.h, h);
    let qd = model.dims.q_dim();
    let kvd = model.dims.kv_dim();
    let inter = model.dims.inter;
    let vocab = model.vocab;

    // ── Head: untie, then fold final_norm α_f into lm_head columns. ───────────
    // The input embedding and the (folded) output head diverge here, so we must
    // untie before touching either.
    if model.lm_head.is_none() {
        let embed_host = gpu.download_f32(&model.embed)?;
        let lmh = gpu.upload_f32(&embed_host, &[vocab * h])?;
        model.lm_head = Some(lmh);
    }
    let alpha_f = gpu.download_f32(&model.final_norm)?;
    {
        let lmh = model.lm_head.as_mut().expect("untied above");
        replace_tensor(gpu, lmh, |mut w| {
            fold_cols(&mut w, &alpha_f, vocab, h);
            w
        })?;
    }
    set_ones(gpu, &mut model.final_norm)?;

    // ── Per-layer: fold norms, rotate readers (input) and writers (output). ───
    for (w, _lora) in model.layers.iter_mut() {
        let a1 = gpu.download_f32(&w.norm1)?;
        for (proj, out) in [(&mut w.wq, qd), (&mut w.wk, kvd), (&mut w.wv, kvd)] {
            replace_tensor(gpu, proj, |mut m| {
                fold_cols(&mut m, &a1, out, h);
                rotate_input(&m, rot, out)
            })?;
        }
        set_ones(gpu, &mut w.norm1)?;
        replace_tensor(gpu, &mut w.wo, |m| rotate_output(&m, rot, qd))?;

        let a2 = gpu.download_f32(&w.norm2)?;
        for proj in [&mut w.wgate, &mut w.wup] {
            replace_tensor(gpu, proj, |mut m| {
                fold_cols(&mut m, &a2, inter, h);
                rotate_input(&m, rot, inter)
            })?;
        }
        set_ones(gpu, &mut w.norm2)?;
        replace_tensor(gpu, &mut w.wdown, |m| rotate_output(&m, rot, inter))?;
    }

    // ── Embedding (input writer) and head (output reader): rotate on `h`. ─────
    // Both rotate on their `h` columns (`E → E Rᵀ`, `lm_head → lm_head Rᵀ`).
    replace_tensor(gpu, &mut model.embed, |m| rotate_input(&m, rot, vocab))?;
    {
        let lmh = model.lm_head.as_mut().expect("untied above");
        replace_tensor(gpu, lmh, |m| rotate_input(&m, rot, vocab))?;
    }

    Ok(())
}

/// Bake a learned rotation `M` for deployment through the Oq4G256 codec: returns
/// `R1 = Fᵀ M` where `F` is the codec's per-256-group FWHT. Passing this to
/// [`apply_r1`] merges it into the weights; the codec then applies `F`, and
/// `F·(Fᵀ M) = M`, so the int4 grid sees the learned rotation. The result is a
/// **standard** `Oq4G256` `.hfq` — R1 is fully merged (zero runtime cost, no
/// loader changes); export is just "`apply_r1(bake_for_oq4_recipe(M))` then run
/// the normal Oq4 quantizer on the rotated weights".
pub fn bake_for_oq4_recipe(m: &Rotation) -> Rotation {
    Rotation::block_fwht(m.h).transpose().compose(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_rotation_is_orthonormal() {
        for &h in &[4usize, 8, 16, 33] {
            let rot = Rotation::random(h, 12345 + h as u64);
            let err = rot.orthonormality_error();
            assert!(err < 1e-4, "h={h} orthonormality err {err:e}");
        }
    }

    #[test]
    fn identity_is_orthonormal() {
        assert!(Rotation::identity(8).orthonormality_error() < 1e-6);
    }

    #[test]
    fn hadamard_is_orthonormal() {
        for &h in &[2usize, 4, 8, 16, 64] {
            let err = Rotation::hadamard(h, 3).orthonormality_error();
            assert!(err < 1e-5, "h={h} hadamard orthonormality err {err:e}");
        }
    }

    #[test]
    fn block_fwht_matches_cpu_fwht_and_is_orthonormal() {
        use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
        let f = Rotation::block_fwht(512);
        assert!(
            f.orthonormality_error() < 1e-4,
            "block_fwht not orthonormal"
        );
        // rotate_rows(x, F) must reproduce the codec's per-256-group fwht.
        let x: Vec<f32> = (0..2 * 512).map(|i| (i as f32 * 0.013).sin()).collect();
        let via_rot = rotate_rows(&x, &f, 2);
        let (s1, s2) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
        let mut manual = x.clone();
        for r in 0..2 {
            for seg in 0..2 {
                let base = r * 512 + seg * 256;
                let mut buf = [0.0f32; 256];
                buf.copy_from_slice(&manual[base..base + 256]);
                cpu_fwht_256(&mut buf, &s1, &s2);
                manual[base..base + 256].copy_from_slice(&buf);
            }
        }
        let worst = via_rot
            .iter()
            .zip(&manual)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst < 1e-4, "block_fwht != codec fwht: {worst:e}");
    }

    /// The export bake identity: applying `R1 = Fᵀ M` and then the codec's FWHT
    /// `F` reproduces the direct learned rotation `M` — so a standard Oq4G256
    /// quantize of the `apply_r1(bake_for_oq4_recipe(M))` weights carries the
    /// learned rotation, no loader changes.
    #[test]
    fn bake_composes_to_learned_through_codec_fwht() {
        let h = 512usize;
        let m = Rotation::random(h, 5);
        let f = Rotation::block_fwht(h);
        let r1_bake = bake_for_oq4_recipe(&m);
        assert!(
            r1_bake.orthonormality_error() < 1e-3,
            "baked R1 not orthonormal"
        );
        let x: Vec<f32> = (0..3 * h).map(|i| (i as f32 * 0.007).cos()).collect();
        // Deployed: apply R1_bake (merged into weights) then the codec FWHT.
        let deployed = rotate_rows(&rotate_rows(&x, &r1_bake, 3), &f, 3);
        // Direct: the learned rotation M.
        let direct = rotate_rows(&x, &m, 3);
        let worst = deployed
            .iter()
            .zip(&direct)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst < 1e-3, "bake identity broken: {worst:e}");
    }

    /// A reader followed by the residual rotation reproduces the pre-rotation
    /// activation: `(x Rᵀ) · (W Rᵀ)ᵀ = x Wᵀ`. This is the invariance identity the
    /// whole transform rests on, checked on tiny random data.
    #[test]
    fn reader_rotation_preserves_activation() {
        let (h, out) = (8usize, 5usize);
        let rot = Rotation::random(h, 7);
        // Random x [1,h] and W [out,h].
        let x: Vec<f32> = (0..h).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..out * h).map(|i| (i as f32 * 0.11).cos()).collect();
        // Original activation y = x Wᵀ.
        let y: Vec<f32> = (0..out)
            .map(|o| (0..h).map(|i| x[i] * w[o * h + i]).sum::<f32>())
            .collect();
        // Rotated residual x̃ = x Rᵀ  (x̃[j] = Σ_i x[i] R[j,i]).
        let xr: Vec<f32> = (0..h)
            .map(|j| (0..h).map(|i| x[i] * rot.r[j * h + i]).sum::<f32>())
            .collect();
        // Rotated reader W Rᵀ, then activation ỹ = x̃ (W Rᵀ)ᵀ.
        let wr = rotate_input(&w, &rot, out);
        let yr: Vec<f32> = (0..out)
            .map(|o| (0..h).map(|j| xr[j] * wr[o * h + j]).sum::<f32>())
            .collect();
        let worst = y
            .iter()
            .zip(&yr)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst < 1e-4, "reader-rotation mismatch {worst:e}");
    }

    /// The residual rotation applied to a writer output reproduces the rotated
    /// contribution: `R (Wᵀ c)` written by `(R W)`. Checks `writer` matches
    /// rotating the plain output.
    #[test]
    fn writer_rotation_matches_rotated_output() {
        let (h, cols) = (8usize, 5usize);
        let rot = Rotation::random(h, 9);
        let c: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.53).sin()).collect();
        let w: Vec<f32> = (0..h * cols).map(|i| (i as f32 * 0.17).cos()).collect();
        // Plain output o = W c  (o[d] = Σ_c W[d,c] c[c]), then rotate: õ = R o.
        let o: Vec<f32> = (0..h)
            .map(|d| (0..cols).map(|cc| w[d * cols + cc] * c[cc]).sum::<f32>())
            .collect();
        let or: Vec<f32> = (0..h)
            .map(|e| (0..h).map(|d| rot.r[e * h + d] * o[d]).sum::<f32>())
            .collect();
        // Rotated writer R W, then output = (R W) c.
        let wr = rotate_output(&w, &rot, cols);
        let ow: Vec<f32> = (0..h)
            .map(|e| (0..cols).map(|cc| wr[e * cols + cc] * c[cc]).sum::<f32>())
            .collect();
        let worst = or
            .iter()
            .zip(&ow)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(worst < 1e-4, "writer-rotation mismatch {worst:e}");
    }
}
