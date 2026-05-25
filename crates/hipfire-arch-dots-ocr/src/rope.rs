//! 2-D spatial RoPE prep for the dots.ocr vision tower.
//!
//! Ports `VisionRotaryEmbedding` + `apply_rotary_pos_emb_vision` +
//! `get_pos_ids_by_grid` from `modeling_dots_vision.py` into a single
//! Rust CPU helper that emits per-patch `cos` / `sin` tables ready
//! for the attention kernel.
//!
//! # Why CPU?
//!
//! Per plan §2.6 / §5 phase 2: "Start with CPU pre-comp + existing
//! kernel; promote to a fused kernel only under the Δ ≥ 5% rule."
//! The tables are tiny relative to the forward pass (N_patches *
//! head_dim * 4 bytes ≈ 10 MB for our 19520-patch reference image)
//! and compute once per image, so the GPU launch overhead would
//! dominate any savings.
//!
//! # Layout (matches dots.ocr's `apply_rotary_pos_emb_vision`)
//!
//! For a patch grid of size `(grid_h, grid_w)`, the patch axis is
//! enumerated in **2×2-block-major order**: groups of
//! `spatial_merge_size × spatial_merge_size` patches are contiguous
//! in the sequence dimension. This matches the patch ordering
//! produced by [`crate::image::extract_patches`] — both sides apply
//! the same `reshape(h/sm, sm, w/sm, sm).permute(0, 2, 1, 3).flatten()`.
//!
//! Within each patch's `head_dim`-element cos/sin vector, the layout
//! is **[h-freq quarter, w-freq quarter, h-freq quarter (repeat),
//! w-freq quarter (repeat)]**:
//!
//! ```text
//! cos[i, 0           ..head_dim/4 ]  = cos(hpos[i] * inv_freq[k])  for k = 0..head_dim/4
//! cos[i, head_dim/4  ..head_dim/2 ]  = cos(wpos[i] * inv_freq[k])  for k = 0..head_dim/4
//! cos[i, head_dim/2  ..3*head_dim/4] = cos(hpos[i] * inv_freq[k])  for k = 0..head_dim/4  (repeat)
//! cos[i, 3*head_dim/4..head_dim   ]  = cos(wpos[i] * inv_freq[k])  for k = 0..head_dim/4  (repeat)
//! ```
//!
//! The repeat doubles each frequency so the standard `rotate_half`
//! transform `(x1, x2) → (-x2, x1)` (which splits the last dim at
//! `head_dim / 2`) correctly rotates pairs at indices `(d, d + head_dim/2)`
//! using the same frequency. This is the dots.ocr-specific 2-D
//! adaptation of the standard 1-D RoPE.
//!
//! `inv_freq[k] = theta ** (-2k / (head_dim/2))` for `k ∈ [0, head_dim/4)`
//! and `theta = 10000` (dots.ocr `VisionRotaryEmbedding` default).
//!
//! # Inputs
//!
//! - `grid_h`, `grid_w`: post-smart-resize patch grid dimensions
//!   (= resized_pixels / patch_size). MUST both be multiples of
//!   `spatial_merge_size`.
//! - `head_dim`: per-attention-head dimension (= embed_dim /
//!   num_attention_heads = 1536 / 12 = 128 for dots.ocr).
//! - `spatial_merge_size`: 2 for dots.ocr.
//! - `theta`: 10000.0 for dots.ocr.
//!
//! # Output
//!
//! `(cos, sin)` — two `Vec<f32>` each of length `N_patches * head_dim`,
//! laid out as `[patch_0_dim_0, ..., patch_0_dim_(head_dim-1), patch_1_dim_0, ...]`
//! (row-major, patch-major). Ready to upload to GPU and broadcast
//! across the `num_heads` axis when applied to Q/K.

/// Build 2-D RoPE cos/sin tables for the dots.ocr vision tower.
///
/// See module docs for the layout / algorithm.
///
/// # Panics
///
/// - `head_dim` is not a multiple of 4 (the layout needs four equal
///   quarters).
/// - `grid_h` or `grid_w` not a multiple of `spatial_merge_size`.
/// - `spatial_merge_size == 0`.
pub fn build_rope_2d_tables(
    grid_h: usize,
    grid_w: usize,
    head_dim: usize,
    spatial_merge_size: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(spatial_merge_size > 0, "spatial_merge_size must be > 0");
    assert_eq!(
        head_dim % 4, 0,
        "head_dim={head_dim} must be a multiple of 4 (two halves of equal h/w split)",
    );
    assert_eq!(
        grid_h % spatial_merge_size, 0,
        "grid_h={grid_h} must be a multiple of spatial_merge_size={spatial_merge_size}",
    );
    assert_eq!(
        grid_w % spatial_merge_size, 0,
        "grid_w={grid_w} must be a multiple of spatial_merge_size={spatial_merge_size}",
    );

    let n_patches = grid_h * grid_w;
    let quarter = head_dim / 4;       // = head_dim_rotary_inv_freq_len = 32 for dots.ocr
    let half = head_dim / 2;          // = 64

    // inv_freq[k] = theta^(-2k / (head_dim/2))  for k in 0..quarter.
    // The exponent denominator is (head_dim / 2) because
    // `VisionRotaryEmbedding(head_dim // 2)` passes `dim = head_dim/2`
    // and its formula is `theta^(-arange(0, dim, 2) / dim) = theta^(-2k/dim)`.
    let denom = (head_dim / 2) as f32;
    let inv_freq: Vec<f32> = (0..quarter)
        .map(|k| theta.powf(-(2.0 * k as f32) / denom))
        .collect();

    let mut cos = vec![0.0f32; n_patches * head_dim];
    let mut sin = vec![0.0f32; n_patches * head_dim];

    let outer_w = grid_w / spatial_merge_size;
    let sm = spatial_merge_size;

    // Enumerate patches in 2×2-block-major order — same as
    // image::extract_patches and the dots.ocr position-ID permute.
    // Iteration: outer_y → outer_x → inner_y → inner_x.
    let mut patch_idx = 0usize;
    for oy in 0..(grid_h / sm) {
        for ox in 0..outer_w {
            for dy in 0..sm {
                for dx in 0..sm {
                    let hpos = oy * sm + dy;
                    let wpos = ox * sm + dx;

                    let base = patch_idx * head_dim;
                    for k in 0..quarter {
                        let h_angle = hpos as f32 * inv_freq[k];
                        let w_angle = wpos as f32 * inv_freq[k];

                        let (hc, hs) = (h_angle.cos(), h_angle.sin());
                        let (wc, ws) = (w_angle.cos(), w_angle.sin());

                        // Layout per quarter: [hc, wc, hc, wc] across the head_dim.
                        cos[base + k] = hc;            // 0          ..quarter
                        cos[base + quarter + k] = wc;  // quarter    ..half
                        cos[base + half + k] = hc;     // half       ..3*quarter   (repeat)
                        cos[base + half + quarter + k] = wc;  // 3*quarter ..head_dim (repeat)

                        sin[base + k] = hs;
                        sin[base + quarter + k] = ws;
                        sin[base + half + k] = hs;
                        sin[base + half + quarter + k] = ws;
                    }
                    patch_idx += 1;
                }
            }
        }
    }
    debug_assert_eq!(patch_idx, n_patches);
    (cos, sin)
}

/// Convenience: number of patches a (grid_h, grid_w) image will produce.
pub fn n_patches(grid_h: usize, grid_w: usize) -> usize {
    grid_h * grid_w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct hpos/wpos arrays the way dots.ocr does in
    /// `get_pos_ids_by_grid`, then derive each per-patch index by
    /// integer arithmetic. Verify both approaches agree — guards
    /// against drift between our flat-loop derivation
    /// (`hpos = oy*sm+dy`) and the original
    /// `reshape(h/sm, sm, w/sm, sm).permute(0, 2, 1, 3).flatten()`
    /// formulation.
    #[test]
    fn patch_enumeration_matches_dots_ocr_reshape_permute() {
        let grid_h = 4;
        let grid_w = 6;
        let sm = 2;
        let outer_h = grid_h / sm;
        let outer_w = grid_w / sm;

        // Reproduce the dots.ocr ordering by walking the permuted
        // 4-D index space (oy, ox, dy, dx) and decoding back to
        // (hpos, wpos) the way the reshape-permute-flatten would.
        let mut want_hpos = Vec::with_capacity(grid_h * grid_w);
        let mut want_wpos = Vec::with_capacity(grid_h * grid_w);
        for oy in 0..outer_h {
            for ox in 0..outer_w {
                for dy in 0..sm {
                    for dx in 0..sm {
                        want_hpos.push(oy * sm + dy);
                        want_wpos.push(ox * sm + dx);
                    }
                }
            }
        }

        // Our build function uses the same iteration; sanity-check
        // by recovering hpos/wpos from the cos table at angle 0
        // (hpos=0 → cos=1, otherwise cos<1 for inv_freq[0]=1).
        let (cos, _sin) = build_rope_2d_tables(grid_h, grid_w, 8, sm, 10000.0);
        // For each patch we can recover (hpos, wpos) from cos[i, 0] = cos(hpos)
        // and cos[i, head_dim/4 = 2] = cos(wpos).
        let head_dim = 8;
        let quarter = head_dim / 4;
        for i in 0..(grid_h * grid_w) {
            let hpos_expected = want_hpos[i];
            let wpos_expected = want_wpos[i];

            let got_h_cos = cos[i * head_dim + 0];
            let got_w_cos = cos[i * head_dim + quarter];
            let want_h_cos = (hpos_expected as f32).cos();
            let want_w_cos = (wpos_expected as f32).cos();

            assert!(
                (got_h_cos - want_h_cos).abs() < 1e-5,
                "patch[{i}] h-cos: got {got_h_cos} want {want_h_cos} (hpos={hpos_expected})",
            );
            assert!(
                (got_w_cos - want_w_cos).abs() < 1e-5,
                "patch[{i}] w-cos: got {got_w_cos} want {want_w_cos} (wpos={wpos_expected})",
            );
        }
    }

    /// Hand-compute cos/sin for a 2×2 grid with sm=2, head_dim=8,
    /// theta=10000 and verify the table matches.
    ///
    /// Layout: 4 patches (hpos, wpos):
    ///   patch 0: (0, 0)
    ///   patch 1: (0, 1)
    ///   patch 2: (1, 0)
    ///   patch 3: (1, 1)
    ///
    /// head_dim=8 → quarter=2 → inv_freq has 2 elements:
    ///   inv_freq[0] = 10000^(-0/4) = 1.0
    ///   inv_freq[1] = 10000^(-2/4) = 1/sqrt(10000) = 1/100 = 0.01
    #[test]
    fn hand_computed_2x2_head8() {
        let (cos, sin) = build_rope_2d_tables(2, 2, 8, 2, 10000.0);
        assert_eq!(cos.len(), 4 * 8);

        // Patch 0: (hpos=0, wpos=0) → all angles = 0 → cos=1, sin=0
        let p0_cos = &cos[0..8];
        let p0_sin = &sin[0..8];
        for v in p0_cos { assert!((v - 1.0).abs() < 1e-6); }
        for v in p0_sin { assert!(v.abs() < 1e-6); }

        // Patch 2: (hpos=1, wpos=0)
        // h_angle = 1.0 * [1.0, 0.01] = [1.0, 0.01]
        // w_angle = 0.0 * [1.0, 0.01] = [0.0, 0.0]
        // cos layout: [hc[0], hc[1], wc[0], wc[1], hc[0], hc[1], wc[0], wc[1]]
        //           = [cos(1), cos(0.01), 1, 1, cos(1), cos(0.01), 1, 1]
        let p2 = &cos[16..24];
        let c1 = 1.0_f32.cos();
        let c01 = 0.01_f32.cos();
        let want_p2 = [c1, c01, 1.0, 1.0, c1, c01, 1.0, 1.0];
        for (i, (g, w)) in p2.iter().zip(want_p2.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "patch[2] cos[{i}]: got {g} want {w}");
        }

        // Patch 1: (hpos=0, wpos=1)
        // h_angle = [0, 0]
        // w_angle = [1.0, 0.01]
        let p1 = &cos[8..16];
        let want_p1 = [1.0, 1.0, c1, c01, 1.0, 1.0, c1, c01];
        for (i, (g, w)) in p1.iter().zip(want_p1.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "patch[1] cos[{i}]: got {g} want {w}");
        }

        // Patch 3: (hpos=1, wpos=1) — both h and w contribute.
        let p3_cos = &cos[24..32];
        let p3_sin = &sin[24..32];
        let s1 = 1.0_f32.sin();
        let s01 = 0.01_f32.sin();
        let want_p3_cos = [c1, c01, c1, c01, c1, c01, c1, c01];
        let want_p3_sin = [s1, s01, s1, s01, s1, s01, s1, s01];
        for (i, (g, w)) in p3_cos.iter().zip(want_p3_cos.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "patch[3] cos[{i}]: got {g} want {w}");
        }
        for (i, (g, w)) in p3_sin.iter().zip(want_p3_sin.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "patch[3] sin[{i}]: got {g} want {w}");
        }
    }

    /// 4×4 grid with sm=2, head_dim=128 (the actual dots.ocr config).
    /// Verify dimensions + that the quarter-repeat layout holds at one
    /// non-trivial patch.
    #[test]
    fn realistic_head_dim_128() {
        let grid_h = 4;
        let grid_w = 4;
        let sm = 2;
        let head_dim = 128;
        let (cos, sin) = build_rope_2d_tables(grid_h, grid_w, head_dim, sm, 10000.0);
        assert_eq!(cos.len(), grid_h * grid_w * head_dim);
        assert_eq!(sin.len(), grid_h * grid_w * head_dim);

        let quarter = head_dim / 4; // 32
        let half = head_dim / 2;     // 64

        // Patch 3 in the 4×4 grid: walk the iteration order to find it.
        // Outer_h=2, outer_w=2, sm=2:
        //   patch 0: (oy=0, ox=0, dy=0, dx=0) → hpos=0, wpos=0
        //   patch 1: (oy=0, ox=0, dy=0, dx=1) → hpos=0, wpos=1
        //   patch 2: (oy=0, ox=0, dy=1, dx=0) → hpos=1, wpos=0
        //   patch 3: (oy=0, ox=0, dy=1, dx=1) → hpos=1, wpos=1
        let i = 3;
        let base = i * head_dim;
        // Verify: cos[i, 0..quarter] equals cos[i, half..half+quarter]
        // (the h-quarter repeat). Same for w-quarter and sin.
        for k in 0..quarter {
            let a = cos[base + k];
            let b = cos[base + half + k];
            assert_eq!(a, b, "h-quarter cos repeat mismatch at k={k}");
            let a = cos[base + quarter + k];
            let b = cos[base + half + quarter + k];
            assert_eq!(a, b, "w-quarter cos repeat mismatch at k={k}");
            let a = sin[base + k];
            let b = sin[base + half + k];
            assert_eq!(a, b, "h-quarter sin repeat mismatch at k={k}");
            let a = sin[base + quarter + k];
            let b = sin[base + half + quarter + k];
            assert_eq!(a, b, "w-quarter sin repeat mismatch at k={k}");
        }
    }

    /// Sanity-check inv_freq computation against the dots.ocr formula
    /// `inv_freq[k] = theta ** (-2k / dim)` where `dim = head_dim / 2`.
    /// For head_dim=128 / theta=10000, the half-life of the highest
    /// frequency (k=0) is the seqlen at which cos(p) wraps; for the
    /// lowest (k=31), it's about 4.4M positions — both within the
    /// expected band for vision RoPE.
    #[test]
    fn inv_freq_endpoints_match_formula() {
        let (cos, _sin) = build_rope_2d_tables(2, 2, 128, 2, 10000.0);
        // Patch 2 has hpos=1 and wpos=0; cos[patch2, 0] = cos(1.0 * inv_freq[0])
        // = cos(1.0 * 1.0) = cos(1) ≈ 0.5403.
        let got = cos[2 * 128 + 0];
        let want = 1.0_f32.cos();
        assert!((got - want).abs() < 1e-6, "inv_freq[0] mismatch: got cos = {got}, want {want}");

        // cos[patch2, 31] = cos(1.0 * inv_freq[31])
        // inv_freq[31] = 10000^(-62/64) = 10000^(-0.96875)
        let inv31 = 10000.0_f32.powf(-62.0 / 64.0);
        let got = cos[2 * 128 + 31];
        let want = (1.0 * inv31).cos();
        assert!((got - want).abs() < 1e-6, "inv_freq[31] mismatch: got {got}, want {want}");
    }

    #[test]
    #[should_panic(expected = "head_dim")]
    fn rejects_non_multiple_of_4_head_dim() {
        let _ = build_rope_2d_tables(2, 2, 6, 2, 10000.0);
    }

    #[test]
    #[should_panic(expected = "grid_h")]
    fn rejects_non_sm_aligned_grid_h() {
        let _ = build_rope_2d_tables(3, 2, 8, 2, 10000.0);
    }

    #[test]
    #[should_panic(expected = "grid_w")]
    fn rejects_non_sm_aligned_grid_w() {
        let _ = build_rope_2d_tables(2, 3, 8, 2, 10000.0);
    }
}
