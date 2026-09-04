//! Fold an escha linear's H128 rotations and diagonals into its weight.
//!
//! An escha linear evaluates
//!
//! ```text
//!   xh  = RS * H * diag(rin) * x
//!   mid = W^T * xh
//!   y   = RS * diag(rout) * H * mid
//! ```
//!
//! Every one of those operators except `W` is linear and input-independent,
//! so they collapse into the weight:
//!
//! ```text
//!   W_eff[i][o] = RS^2 * rin_i * (H W H)[i][o] * rout_o
//! ```
//!
//! WHY THIS MATTERS: a folded escha linear is an ORDINARY dense weight. Every
//! fused path in the runtime — FusedQkv, FusedQkvza, gate_up, the batched
//! prefill arms — consumes it untouched, with no escha awareness anywhere in
//! the forward pass. That is the difference between a contained converter
//! change and threading a new weight kind through two multi-thousand-line
//! files.
//!
//! WHAT IT COSTS: the folded matrix is dense, so the 2-bit residency of the
//! trellis code is gone; what survives is escha's quantisation QUALITY, baked
//! into whatever container it is re-quantised to. This is a deliberate
//! trade, not an oversight — see the module docs on the converter flag.
//!
//! NUMERICS: folding SKIPS the reference's fp16 rounding of `xh`, so it is not
//! bit-identical to `escha_ref::expert_linear`. Measured against it on the
//! real 27B (`linear_attn.in_proj_z`, ic=5120 oc=6144): rel_rms 2.868e-4,
//! against 9.192e-5 for the runtime-transform path — same order, and the
//! difference is f32 accumulation across two Hadamard passes rather than a
//! systematic bias.

/// `1/sqrt(128)`, applied once per H128 — hence `RS*RS` in the fold.
pub const ESCHA_RS: f32 = 0.088_388_347_648;

/// Unnormalised 128-point Walsh-Hadamard, Sylvester order, in place, over
/// every contiguous 128-lane block of `x`. Byte-for-byte the same butterfly
/// as `escha_ref::h128_inplace`; duplicated rather than shared so the
/// reference stays a reference (G3 gates the GPU H128 against it, and
/// generating one from the other would make that circular).
#[inline]
fn h128_blocks(x: &mut [f32]) {
    debug_assert_eq!(x.len() % 128, 0);
    for block in x.chunks_exact_mut(128) {
        let mut h = 1;
        while h < 128 {
            let mut i = 0;
            while i < 128 {
                for j in i..i + h {
                    let (a, b) = (block[j], block[j + h]);
                    block[j] = a + b;
                    block[j + h] = a - b;
                }
                i += 2 * h;
            }
            h *= 2;
        }
    }
}

/// Blocked transpose `[rows, cols] -> [cols, rows]`.
///
/// 64x64 tiles: the naive version strides one side by a full row and thrashes
/// for anything the size of a real projection (gate_proj is 5120x17408 =
/// 89 M elements, 356 MB as f32).
fn transpose_blocked(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const T: usize = 64;
    let mut dst = vec![0.0f32; rows * cols];
    for r0 in (0..rows).step_by(T) {
        let r1 = (r0 + T).min(rows);
        for c0 in (0..cols).step_by(T) {
            let c1 = (c0 + T).min(cols);
            for r in r0..r1 {
                let srow = &src[r * cols..r * cols + cols];
                for c in c0..c1 {
                    dst[c * rows + r] = srow[c];
                }
            }
        }
    }
    dst
}

/// Fold one escha linear into an ordinary dense weight.
///
/// `w_bits` is the decoded weight as `escha_ref::reconstruct` returns it:
/// fp16 bits, IN-major `[ic, oc]` (`w[i * oc + o]`).
///
/// Returns f32 in OUT-major `[oc, ic]` — hipfire's dense convention, and
/// exactly what `quantize_mq6g256v2(data, oc, ic, ..)` and friends expect, so
/// no further shuffling is needed before quantisation.
///
/// Both Hadamard passes run over CONTIGUOUS lanes: pass A along `o` while the
/// matrix is still in-major, then one blocked transpose, then pass B along `i`
/// which is now contiguous. Doing pass B strided instead (gathering each of
/// `oc` columns at stride `oc`) is correct but cache-hostile enough to be
/// unusable across 400 projections.
pub fn fold_escha_linear(
    w_bits: &[u16],
    ic: usize,
    oc: usize,
    rin: &[f32],
    rout: &[f32],
) -> Result<Vec<f32>, String> {
    if w_bits.len() != ic * oc {
        return Err(format!(
            "fold: weight has {} elements, expected ic*oc = {}",
            w_bits.len(),
            ic * oc
        ));
    }
    if rin.len() != ic || rout.len() != oc {
        return Err(format!(
            "fold: rin {} (want {ic}), rout {} (want {oc})",
            rin.len(),
            rout.len()
        ));
    }
    if ic % 128 != 0 || oc % 128 != 0 {
        return Err(format!(
            "fold: H128 needs both dims a multiple of 128, got ic={ic} oc={oc}"
        ));
    }

    // in-major [ic, oc], f32.
    let mut w: Vec<f32> = w_bits
        .iter()
        .map(|&b| crate::float16::f16_to_f32(b))
        .collect();

    // Pass A — H along `o`, contiguous within each in-major row.
    for row in w.chunks_exact_mut(oc) {
        h128_blocks(row);
    }

    // -> out-major [oc, ic].
    let mut wt = transpose_blocked(&w, ic, oc);
    drop(w);

    // Pass B — H along `i`, now contiguous within each out-major row.
    for row in wt.chunks_exact_mut(ic) {
        h128_blocks(row);
    }

    // Diagonals and RS^2. Row `o`, column `i`.
    let rs2 = ESCHA_RS * ESCHA_RS;
    for (o, row) in wt.chunks_exact_mut(ic).enumerate() {
        let s = rs2 * rout[o];
        for (i, v) in row.iter_mut().enumerate() {
            *v *= s * rin[i];
        }
    }
    Ok(wt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// H128 is its own inverse up to a factor of 128, which is the cheapest
    /// end-to-end check that the butterfly is the right transform and not
    /// merely *a* transform.
    #[test]
    fn h128_is_an_involution_scaled_by_128() {
        let mut x: Vec<f32> = (0..256).map(|i| ((i * 37) % 91) as f32 - 45.0).collect();
        let orig = x.clone();
        h128_blocks(&mut x);
        h128_blocks(&mut x);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b * 128.0).abs() < 1e-3, "{a} vs {}", b * 128.0);
        }
    }

    /// The folded weight must reproduce the un-folded evaluation. Built on a
    /// tiny random-ish linear so the whole path is checked without a
    /// checkpoint: fold, then a plain matmul, against transform-matmul-
    /// transform done longhand.
    #[test]
    fn folded_weight_matches_the_transform_pipeline() {
        let (ic, oc) = (128usize, 256usize);
        let w_bits: Vec<u16> = (0..ic * oc)
            .map(|n| crate::float16::f32_to_f16((((n * 31) % 17) as f32 - 8.0) / 8.0))
            .collect();
        let rin: Vec<f32> = (0..ic).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
        let rout: Vec<f32> = (0..oc).map(|o| 0.5 + (o % 7) as f32 * 0.05).collect();
        let x: Vec<f32> = (0..ic).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();

        // Longhand: xh = RS*H*diag(rin)*x ; mid = W^T xh ; y = RS*diag(rout)*H*mid
        let mut xh: Vec<f32> = x.iter().zip(&rin).map(|(a, b)| a * b).collect();
        h128_blocks(&mut xh);
        for v in xh.iter_mut() {
            *v *= ESCHA_RS;
        }
        let mut mid = vec![0.0f32; oc];
        for i in 0..ic {
            let a = xh[i];
            for o in 0..oc {
                mid[o] += a * crate::float16::f16_to_f32(w_bits[i * oc + o]);
            }
        }
        h128_blocks(&mut mid);
        let want: Vec<f32> = mid
            .iter()
            .zip(&rout)
            .map(|(m, r)| m * ESCHA_RS * r)
            .collect();

        // Folded: one plain matmul on the out-major weight.
        let wf = fold_escha_linear(&w_bits, ic, oc, &rin, &rout).unwrap();
        let got: Vec<f32> = (0..oc)
            .map(|o| {
                let row = &wf[o * ic..(o + 1) * ic];
                row.iter().zip(&x).map(|(w, xi)| w * xi).sum()
            })
            .collect();

        let num: f64 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
        let rel = (num / den.max(1e-30)).sqrt();
        assert!(rel < 1e-5, "folded vs longhand rel_rms {rel:.3e}");
    }

    #[test]
    fn rejects_dims_that_are_not_h128_shaped() {
        let e = fold_escha_linear(&[0u16; 100 * 128], 100, 128, &[1.0; 100], &[1.0; 128])
            .unwrap_err();
        assert!(e.contains("multiple of 128"), "{e}");
    }
}
