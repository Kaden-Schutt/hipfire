//! Image preprocessing for dots.ocr.
//!
//! Ports the algorithm in `dots_ocr/utils/image_utils.py` and the patch
//! reshape+transpose in `transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
//! into Rust. Output of [`preprocess_image`] is a CHW pixel tensor that
//! [`extract_patches`] then turns into the `flatten_patches` byte
//! sequence the dots.ocr vision tower expects on its input.
//!
//! Bring-up status (phase 2b, rev 0):
//! - [`smart_resize`] — landed. 28-divisible H/W with min/max-pixels
//!   clamp, beta scaling, and AR > 200:1 guard.
//! - [`clip_normalise`] — landed. CLIP-style mean/std normalisation,
//!   RGB; RGBA → RGB compositing onto white before normalising.
//! - [`extract_patches`] — landed. Matches the
//!   `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)` of §2.7 EXACTLY (the silent-
//!   failure trap). Unit-tested against synthetic input with a per-pixel
//!   unique tag — see `tests::extract_patches_uses_grid_block_order`.
//! - [`preprocess_image`] — landed. Top-level wrapper: load PNG/JPEG
//!   → RGBA→RGB compositing → smart_resize → triangle-filter resize
//!   → CLIP normalise (CHW f32) → extract_patches.
//!
//! Vision-tower forward (phase 2c) consumes the output of
//! [`extract_patches`] directly — no further reshape on the call site.
//!
//! # Silent-failure trap
//!
//! [`extract_patches`] MUST emit patches in (outer_row, outer_col,
//! inner_y, inner_x) order — the 2×2-grouped-block-major enumeration
//! that puts neighbouring patches in a `spatial_merge_size × spatial_merge_size`
//! tile contiguous in the output. Raster order (py, px) looks correct
//! at a single-token level but makes the merger fuse the wrong four
//! patches, producing bounding-box coordinates that are off by a
//! 2×2-tile-sized offset varying with image width. Documented in §2.7
//! of the bring-up plan. The byte-identical-vs-HF unit test is the
//! barrier here.

use std::path::Path;

// ─── Constants ────────────────────────────────────────────────────────

/// `IMAGE_FACTOR` from `dots_ocr/utils/image_utils.py`. Equal to
/// `patch_size * spatial_merge_size = 14 * 2 = 28`. Both resized H
/// and W must land on 28-multiples so the patch grid (after dividing
/// by `patch_size=14`) is then divisible by `spatial_merge_size=2`,
/// which the merger needs.
pub const IMAGE_FACTOR: usize = 28;

/// Lower bound on resized total pixels. Below this, the image is
/// upscaled along the dominant axis until it crosses the threshold.
/// `56 * 56` matches the dots.ocr source default.
pub const MIN_PIXELS: usize = 3136;

/// Upper bound on resized total pixels. Above this, the image is
/// downscaled while preserving aspect ratio. dots.ocr's published
/// value is `11_289_600` (= 3360×3360 worth of pixels), considerably
/// larger than qwen2-vl's `1_003_520` default.
pub const MAX_PIXELS: usize = 11_289_600;

/// Aspect-ratio guard. Inputs whose `max(h,w) / min(h,w)` exceeds
/// this limit are rejected at the preprocessing entry — they're
/// almost certainly OCR scan failures (one-pixel-tall strips) and
/// would land outside the resized pixel envelope anyway.
pub const MAX_RATIO: f64 = 200.0;

/// CLIP-style normalisation mean (RGB order). Pulled from
/// `preprocessor_config.json:image_mean` of the dots.ocr snapshot;
/// matches the standard CLIP / SigLIP / Qwen2-VL constants.
pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP-style normalisation std (RGB order).
pub const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Vision tower patch size (`config.json:vision_config.patch_size`).
pub const PATCH_SIZE: usize = 14;

/// Vision tower spatial merge size — number of patches per side
/// that the PatchMerger collapses into one visual token.
pub const SPATIAL_MERGE_SIZE: usize = 2;

/// dots.ocr models RGB still images, not video. `temporal_patch_size = 1`
/// means each input image goes through the tower once.
pub const TEMPORAL_PATCH_SIZE: usize = 1;

// ─── smart-resize ─────────────────────────────────────────────────────

/// Smart-resize per `dots_ocr/utils/image_utils.py:smart_resize`.
///
/// Returns the resized `(h, w)` that satisfies:
/// 1. Both dimensions are multiples of [`IMAGE_FACTOR`] (28).
/// 2. `h * w` ∈ `[MIN_PIXELS, MAX_PIXELS]`.
/// 3. Aspect ratio is preserved as closely as 28-rounding permits.
///
/// Algorithm: round each dim to the nearest 28-multiple. If the result
/// would exceed `MAX_PIXELS`, scale both dims down by
/// `sqrt(h * w / MAX_PIXELS)` then floor to 28-multiples. If below
/// `MIN_PIXELS`, scale up by `sqrt(MIN_PIXELS / (h * w))` then ceil to
/// 28-multiples.
///
/// # Errors
///
/// Returns `Err` if the input aspect ratio exceeds [`MAX_RATIO`] —
/// rejects pathological scan-failure inputs that would otherwise
/// produce a single-row visual token sequence.
pub fn smart_resize(orig_h: usize, orig_w: usize) -> Result<(usize, usize), String> {
    if orig_h == 0 || orig_w == 0 {
        return Err(format!(
            "dots-ocr: zero-dimension input image ({orig_h}×{orig_w})"
        ));
    }
    let max_dim = orig_h.max(orig_w) as f64;
    let min_dim = orig_h.min(orig_w) as f64;
    let ratio = max_dim / min_dim;
    if ratio > MAX_RATIO {
        return Err(format!(
            "dots-ocr: aspect ratio {ratio:.1} exceeds {MAX_RATIO} \
             (input {orig_h}×{orig_w}) — refusing to preprocess"
        ));
    }

    smart_resize_inner(orig_h, orig_w, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS)
}

/// Inner helper, parameterised for unit-testability. Returns the
/// resized dims clamped to `[min_pixels, max_pixels]` and rounded to
/// `factor`-multiples. Does NOT enforce the AR guard.
fn smart_resize_inner(
    orig_h: usize,
    orig_w: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    let factor_f = factor as f64;
    let h = orig_h as f64;
    let w = orig_w as f64;

    let h_round = ((h / factor_f).round() as usize) * factor;
    let w_round = ((w / factor_f).round() as usize) * factor;

    // Use u64 for pixel-count comparisons to avoid usize overflow on
    // 32-bit targets. (Defense-in-depth; callers gate dimensions.)
    let hround_wround = (h_round as u64) * (w_round as u64);

    let (h_out, w_out) = if hround_wround > max_pixels as u64 {
        // Down-scale: shrink by sqrt(h*w / max_pixels), then floor to factor.
        let hw = h * w;
        let beta = (hw / max_pixels as f64).sqrt();
        let h_scaled = factor.max(((h / beta / factor_f).floor() as usize) * factor);
        let w_scaled = factor.max(((w / beta / factor_f).floor() as usize) * factor);
        (h_scaled, w_scaled)
    } else if hround_wround < min_pixels as u64 {
        // Up-scale: grow by sqrt(min_pixels / (h*w)), then ceil to factor.
        let hw = h * w;
        let beta = (min_pixels as f64 / hw).sqrt();
        let h_scaled = factor.max(((h * beta / factor_f).ceil() as usize) * factor);
        let w_scaled = factor.max(((w * beta / factor_f).ceil() as usize) * factor);
        // Edge case from `dots_ocr/utils/image_utils.py:56-61`: the
        // upscale ceil-rounding can overshoot `max_pixels` on a very
        // skinny input (e.g. 1×100 input → upscale to cross min_pixels
        // → ceil pushes both dims to factor-multiples → product > max).
        // The Python source re-clamps with the downscale formula. For
        // dots.ocr's MIN=3136 / MAX=11_289_600 the ratio (~3600×) makes
        // this branch effectively unreachable, but we mirror the source
        // exactly to keep the byte-identity claim with HF.
        let hs_ws = (h_scaled as u64) * (w_scaled as u64);
        if hs_ws > max_pixels as u64 {
            let beta = (hs_ws as f64 / max_pixels as f64).sqrt();
            let h_re = factor.max(((h_scaled as f64 / beta / factor_f).floor() as usize) * factor);
            let w_re = factor.max(((w_scaled as f64 / beta / factor_f).floor() as usize) * factor);
            (h_re, w_re)
        } else {
            (h_scaled, w_scaled)
        }
    } else {
        (h_round, w_round)
    };

    if h_out == 0 || w_out == 0 {
        return Err(format!(
            "dots-ocr: smart_resize produced zero dimension \
             ({h_out}×{w_out}) from input {orig_h}×{orig_w}"
        ));
    }
    Ok((h_out, w_out))
}

// ─── CLIP normalisation ──────────────────────────────────────────────

/// Normalise an RGB u8 image to CHW f32 with CLIP mean/std.
///
/// Input: `rgb` is a tightly-packed `[H * W * 3]` u8 buffer in HWC
/// order (RGB). Output: `Vec<f32>` of length `3 * H * W` in CHW order
/// (channel-major, then row-major).
///
/// Formula: `out = (in / 255 - mean[c]) / std[c]` per channel `c`.
pub fn clip_normalise(rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
    assert_eq!(
        rgb.len(),
        h * w * 3,
        "clip_normalise: expected H*W*3={} bytes, got {}",
        h * w * 3,
        rgb.len()
    );
    let plane = h * w;
    let mut out = vec![0.0f32; 3 * plane];
    let inv_255 = 1.0f32 / 255.0;
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            let dst = y * w + x;
            for c in 0..3 {
                let v = rgb[src + c] as f32 * inv_255;
                out[c * plane + dst] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }
    out
}

// ─── Patch extraction (silent-failure trap) ──────────────────────────

/// Extract patches from a CHW f32 image in the dots.ocr / HF
/// `Qwen2VLImageProcessor` order — the 2×2-grouped-block-major
/// enumeration documented in §2.7 of the bring-up plan.
///
/// # Input
///
/// - `chw`: `[3 * h * w]` f32 buffer in CHW order, already
///   smart-resized so `h` and `w` are multiples of [`PATCH_SIZE`] and
///   the resulting patch grid `(h/PATCH_SIZE, w/PATCH_SIZE)` is
///   further divisible by [`SPATIAL_MERGE_SIZE`].
/// - `h`, `w`: image dimensions in pixels.
///
/// # Output
///
/// `Vec<f32>` of length `N_patches * (3 * PATCH_SIZE * PATCH_SIZE)`
/// where `N_patches = (h/PATCH_SIZE) * (w/PATCH_SIZE)`.
///
/// Patch indices in the output sequence run as:
/// ```text
///   for outer_y in 0..(grid_h / SM):
///     for outer_x in 0..(grid_w / SM):
///       for inner_y in 0..SM:
///         for inner_x in 0..SM:
///           emit patch at (py = outer_y*SM + inner_y, px = outer_x*SM + inner_x)
/// ```
/// This is what `transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)` then
/// `reshape(grid_t * grid_h * grid_w, ...)` produces in the HF
/// processor at `image_processing_qwen2_vl.py:281-295`.
///
/// Within each patch, elements are laid out (channel-major, then
/// patch_size_y, then patch_size_x) — matching the HF transpose's
/// trailing axes `(2, 1, 5, 8)`. For dots.ocr's `temporal_patch_size = 1`
/// the temporal axis collapses to one and doesn't appear in the
/// nesting.
///
/// # Panics
///
/// - `chw.len() != 3 * h * w`
/// - `h % PATCH_SIZE != 0` or `w % PATCH_SIZE != 0`
/// - `(h / PATCH_SIZE) % SPATIAL_MERGE_SIZE != 0` or
///   `(w / PATCH_SIZE) % SPATIAL_MERGE_SIZE != 0`
///
/// # TEMPORAL_PATCH_SIZE > 1 (future)
///
/// Input contract is `[3, h, w]` (single frame). For
/// `TEMPORAL_PATCH_SIZE > 1` the inner `_t` loop reads
/// `chw[c * h * w + y * w + x]` independent of `_t`, which
/// duplicates the single frame across the temporal axis. This
/// matches HF behaviour — they upstream-duplicate one frame into a
/// `[TPS, C, H, W]` tensor before the transpose at
/// `image_processing_qwen2_vl.py:267-275`. dots.ocr ships with
/// `TEMPORAL_PATCH_SIZE = 1` so the duplication path is dead code;
/// kept explicit so a future TPS>1 fork compiles unchanged.
pub fn extract_patches(chw: &[f32], h: usize, w: usize) -> Vec<f32> {
    assert_eq!(
        chw.len(),
        3 * h * w,
        "extract_patches: expected 3*{h}*{w}={} elements, got {}",
        3 * h * w,
        chw.len()
    );
    assert_eq!(
        h % PATCH_SIZE,
        0,
        "h={h} must be a multiple of PATCH_SIZE={PATCH_SIZE}"
    );
    assert_eq!(
        w % PATCH_SIZE,
        0,
        "w={w} must be a multiple of PATCH_SIZE={PATCH_SIZE}"
    );
    let grid_h = h / PATCH_SIZE;
    let grid_w = w / PATCH_SIZE;
    assert_eq!(
        grid_h % SPATIAL_MERGE_SIZE,
        0,
        "grid_h={grid_h} (from h={h}/PATCH_SIZE={PATCH_SIZE}) must be a multiple of SPATIAL_MERGE_SIZE={SPATIAL_MERGE_SIZE}"
    );
    assert_eq!(
        grid_w % SPATIAL_MERGE_SIZE,
        0,
        "grid_w={grid_w} (from w={w}/PATCH_SIZE={PATCH_SIZE}) must be a multiple of SPATIAL_MERGE_SIZE={SPATIAL_MERGE_SIZE}"
    );

    let outer_h = grid_h / SPATIAL_MERGE_SIZE;
    let outer_w = grid_w / SPATIAL_MERGE_SIZE;
    let n_patches = grid_h * grid_w;
    let patch_elems = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
    let mut out = vec![0.0f32; n_patches * patch_elems];

    let mut patch_idx = 0;
    for oy in 0..outer_h {
        for ox in 0..outer_w {
            for sy in 0..SPATIAL_MERGE_SIZE {
                for sx in 0..SPATIAL_MERGE_SIZE {
                    let py = oy * SPATIAL_MERGE_SIZE + sy;
                    let px = ox * SPATIAL_MERGE_SIZE + sx;
                    let base = patch_idx * patch_elems;
                    let mut k = 0;
                    for c in 0..3 {
                        // temporal_patch_size = 1; the loop kept explicit
                        // so the index nesting matches HF's transpose
                        // exactly. If dots.ocr ever ships a TPS>1
                        // checkpoint, this loop is correct as-is
                        // (HF duplicates the single frame across the
                        // temporal axis upstream of this transpose).
                        for _t in 0..TEMPORAL_PATCH_SIZE {
                            for dy in 0..PATCH_SIZE {
                                for dx in 0..PATCH_SIZE {
                                    let y = py * PATCH_SIZE + dy;
                                    let x = px * PATCH_SIZE + dx;
                                    let src = c * h * w + y * w + x;
                                    out[base + k] = chw[src];
                                    k += 1;
                                }
                            }
                        }
                    }
                    patch_idx += 1;
                }
            }
        }
    }
    out
}

// ─── Top-level preprocessing pipeline ────────────────────────────────

/// Preprocessed image ready for the vision tower.
///
/// `patches.len() == n_patches * (3 * PATCH_SIZE * PATCH_SIZE)`.
/// `n_patches = grid_h * grid_w`. Patches are in dots.ocr / HF
/// `Qwen2VLImageProcessor` order (see [`extract_patches`]).
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// `[N_patches, 3 * PATCH_SIZE * PATCH_SIZE]` flattened to a 1-D
    /// `Vec<f32>`.
    pub patches: Vec<f32>,
    /// Number of patches along the height dimension (post-smart-resize,
    /// before merging).
    pub grid_h: usize,
    /// Number of patches along the width dimension.
    pub grid_w: usize,
    /// Pixel dimensions after smart-resize.
    pub resized_h: usize,
    pub resized_w: usize,
}

impl PreprocessedImage {
    /// Total patches before the merger (= `grid_h * grid_w`).
    pub fn n_patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    /// Total visual tokens after the merger (`= n_patches / SM^2`).
    pub fn n_visual_tokens(&self) -> usize {
        self.n_patches() / (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE)
    }
}

/// Load an image from disk, run the full dots.ocr preprocessing
/// pipeline, and return patches ready for the vision tower.
///
/// Path can point at PNG or JPEG (the only decoders compiled in via
/// the `image` crate feature set). RGBA inputs are composited onto a
/// white background before normalisation.
pub fn preprocess_image(path: &Path) -> Result<PreprocessedImage, String> {
    let dyn_img = image::open(path)
        .map_err(|e| format!("dots-ocr: failed to open {}: {e}", path.display()))?;
    preprocess_dynamic_image(&dyn_img)
}

/// Variant of [`preprocess_image`] that decodes raw image bytes from
/// memory (e.g. a base64-decoded payload off the daemon's request).
/// The format is sniffed from the byte content.
pub fn preprocess_image_bytes(bytes: &[u8]) -> Result<PreprocessedImage, String> {
    let dyn_img = image::load_from_memory(bytes)
        .map_err(|e| format!("dots-ocr: failed to decode image bytes: {e}"))?;
    preprocess_dynamic_image(&dyn_img)
}

/// Variant of [`preprocess_image`] that takes an already-decoded
/// `DynamicImage`. Useful for callers that load image bytes from a
/// non-filesystem source (HTTP, memory, etc.).
pub fn preprocess_dynamic_image(img: &image::DynamicImage) -> Result<PreprocessedImage, String> {
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;

    let (resized_h, resized_w) = smart_resize(orig_h, orig_w)?;

    // RGBA → RGB compositing onto white background (matches HF's
    // PIL `convert("RGB")` on an RGBA source, with alpha-blended over
    // white). For non-RGBA inputs, `to_rgb8` is a straight conversion.
    let rgb_image: image::RgbImage = match img {
        image::DynamicImage::ImageRgba8(rgba) => composite_rgba_on_white(rgba),
        image::DynamicImage::ImageLumaA8(la) => {
            composite_rgba_on_white(&image::DynamicImage::ImageLumaA8(la.clone()).to_rgba8())
        }
        other => other.to_rgb8(),
    };

    let resized = image::imageops::resize(
        &rgb_image,
        resized_w as u32,
        resized_h as u32,
        // HF's Qwen2VLImageProcessor uses PIL BICUBIC for the resize step.
        // image::imageops::FilterType::CatmullRom is the closest bicubic
        // variant available in the `image` crate (cubic B-spline / Catmull-
        // Rom — small per-pixel differences vs PIL BICUBIC are expected).
        image::imageops::FilterType::CatmullRom,
    );

    let chw = clip_normalise(resized.as_raw(), resized_h, resized_w);
    let patches = extract_patches(&chw, resized_h, resized_w);

    let grid_h = resized_h / PATCH_SIZE;
    let grid_w = resized_w / PATCH_SIZE;

    Ok(PreprocessedImage {
        patches,
        grid_h,
        grid_w,
        resized_h,
        resized_w,
    })
}

/// Composite an RGBA image onto a white background, producing RGB.
/// `out_rgb = src_rgb * (alpha/255) + 255 * (1 - alpha/255)`.
fn composite_rgba_on_white(rgba: &image::RgbaImage) -> image::RgbImage {
    let mut out = image::RgbImage::new(rgba.width(), rgba.height());
    for (x, y, pix) in rgba.enumerate_pixels() {
        let a = pix[3] as u32;
        let inv_a = 255 - a;
        let blend = |c: u8| ((c as u32 * a + 255 * inv_a) / 255) as u8;
        out.put_pixel(
            x,
            y,
            image::Rgb([blend(pix[0]), blend(pix[1]), blend(pix[2])]),
        );
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_lands_on_28_multiples() {
        // A tall image well within the AR guard but on the boundary of
        // max_pixels — smart_resize must scale it down enough that
        // h*w fits and both dims stay on 28-multiples.
        let (h, w) = smart_resize(100, 3000).expect("aspect ratio 30 is well under MAX_RATIO=200");
        assert_eq!(h % IMAGE_FACTOR, 0, "h={h} is not a 28-multiple");
        assert_eq!(w % IMAGE_FACTOR, 0, "w={w} is not a 28-multiple");
        // 100×3000 = 300_000 pixels, way under MAX_PIXELS=11_289_600,
        // so smart_resize rounds to nearest 28-multiple.
        assert!(
            h * w <= MAX_PIXELS,
            "{h}×{w} = {} exceeds MAX_PIXELS",
            h * w
        );
        assert!(h * w >= MIN_PIXELS, "{h}×{w} = {} below MIN_PIXELS", h * w);
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect_ratio() {
        // 1×500 → ratio = 500, well over MAX_RATIO=200. Must error,
        // not silently produce a 28×28 or similar degenerate dim.
        let err = smart_resize(1, 500).expect_err("AR=500 must trip the MAX_RATIO=200 guard");
        assert!(
            err.contains("aspect ratio") && err.contains("200"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn smart_resize_rejects_zero_input() {
        assert!(smart_resize(0, 100).is_err());
        assert!(smart_resize(100, 0).is_err());
    }

    #[test]
    fn smart_resize_downscales_when_over_max() {
        // 8000×8000 = 64M pixels, way over 11.29M cap. Must shrink
        // to fit while preserving square aspect.
        let (h, w) = smart_resize(8000, 8000).expect("square AR=1 passes");
        assert_eq!(h, w, "square input should land on equal dims");
        assert!(h * w <= MAX_PIXELS);
        assert_eq!(h % IMAGE_FACTOR, 0);
    }

    #[test]
    fn smart_resize_upscales_when_below_min() {
        // 20×20 = 400 pixels, under MIN_PIXELS=3136. Must grow until
        // it crosses the floor.
        let (h, w) = smart_resize(20, 20).expect("square AR=1 passes");
        assert_eq!(h, w);
        assert!(h * w >= MIN_PIXELS, "{h}×{w}={} below floor", h * w);
        assert_eq!(h % IMAGE_FACTOR, 0);
    }

    #[test]
    fn clip_normalise_applies_per_channel_mean_std() {
        // 1×1 RGB pixel at (R=255, G=0, B=128). After /255:
        // R=1, G=0, B=0.50196. Normalised:
        //   R: (1.0 - 0.48145466) / 0.26862954 ≈ 1.93008
        //   G: (0.0 - 0.4578275)  / 0.26130258 ≈ -1.75215
        //   B: (0.50196 - 0.40821073) / 0.27577711 ≈ 0.33996
        let rgb = vec![255u8, 0, 128];
        let out = clip_normalise(&rgb, 1, 1);
        assert_eq!(out.len(), 3);
        // CHW layout: out[0]=R, out[1]=G, out[2]=B.
        assert!((out[0] - 1.93008).abs() < 1e-3, "R = {}", out[0]);
        assert!((out[1] - (-1.75215)).abs() < 1e-3, "G = {}", out[1]);
        assert!((out[2] - 0.33996).abs() < 1e-3, "B = {}", out[2]);
    }

    #[test]
    fn extract_patches_uses_grid_block_order() {
        // The §2.7 silent-failure trap. Build a synthetic 28×56 input
        // (= 2×4 patch grid with patch_size=14, → 1×2 outer-block
        // grid with sm=2 → 2 visual tokens after merger). Tag every
        // pixel with a unique value derived from its patch position:
        // pixel-value = (py * 1000 + px) for the patch it belongs to.
        // Then verify that output patch[i] contains the value
        // corresponding to its expected 2×2-grouped position.
        //
        // 28×56 → grid_h=2, grid_w=4, outer_h=1, outer_w=2.
        // Expected output ordering by (outer_y, outer_x, inner_y, inner_x):
        //   patch 0: (0, 0, 0, 0) → (py=0, px=0)
        //   patch 1: (0, 0, 0, 1) → (py=0, px=1)
        //   patch 2: (0, 0, 1, 0) → (py=1, px=0)
        //   patch 3: (0, 0, 1, 1) → (py=1, px=1)
        //   patch 4: (0, 1, 0, 0) → (py=0, px=2)
        //   patch 5: (0, 1, 0, 1) → (py=0, px=3)
        //   patch 6: (0, 1, 1, 0) → (py=1, px=2)
        //   patch 7: (0, 1, 1, 1) → (py=1, px=3)
        // Raster order would be: 0,1,2,3,4,5,6,7 = py=0,px=0..3 then py=1,px=0..3
        // → expected tags: 0, 1, 2, 3, 1000, 1001, 1002, 1003 (NOT what we want).
        // 2×2-grouped order gives: 0, 1, 1000, 1001, 2, 3, 1002, 1003.
        let h = 28;
        let w = 56;
        let mut chw = vec![0.0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let py = y / PATCH_SIZE;
                    let px = x / PATCH_SIZE;
                    chw[c * h * w + y * w + x] = (py as f32) * 1000.0 + (px as f32);
                }
            }
        }
        let patches = extract_patches(&chw, h, w);
        let patch_elems = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
        // 8 patches expected.
        assert_eq!(patches.len(), 8 * patch_elems);

        // Expected tag per output patch (first element of each patch
        // is its tag, since the synthetic input is per-pixel-constant
        // within a single patch).
        let expected = [
            0.0,    // (outer 0,0, inner 0,0) → py=0, px=0
            1.0,    // (outer 0,0, inner 0,1) → py=0, px=1
            1000.0, // (outer 0,0, inner 1,0) → py=1, px=0
            1001.0, // (outer 0,0, inner 1,1) → py=1, px=1
            2.0,    // (outer 0,1, inner 0,0) → py=0, px=2
            3.0,    // (outer 0,1, inner 0,1) → py=0, px=3
            1002.0, // (outer 0,1, inner 1,0) → py=1, px=2
            1003.0, // (outer 0,1, inner 1,1) → py=1, px=3
        ];
        for (i, &want) in expected.iter().enumerate() {
            let got = patches[i * patch_elems];
            let raster_would_give = if i < 4 {
                i as f32
            } else {
                1000.0 + (i - 4) as f32
            };
            assert_eq!(
                got, want,
                "patch[{i}] first element: expected {want}, got {got} \
                 (raster order would have given {raster_would_give})"
            );
        }
    }

    #[test]
    fn extract_patches_preserves_patch_interior() {
        // Synthetic input where each pixel encodes its (c, y, x)
        // position. Verify that the patch at output index 0 (which
        // should be py=0, px=0) contains the expected pixel values in
        // (c, py_inner, px_inner) order — matches the HF inner-axis
        // layout of (channel, [tps,] patch_y, patch_x).
        let h = 28;
        let w = 28; // 2×2 patch grid — minimal valid input
        let mut chw = vec![0.0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    // Pixel value = c * 10000 + y * 100 + x — unique
                    // per (c, y, x) so any reordering is detectable.
                    chw[c * h * w + y * w + x] =
                        (c as f32) * 10000.0 + (y as f32) * 100.0 + (x as f32);
                }
            }
        }
        let patches = extract_patches(&chw, h, w);
        let patch_elems = 3 * PATCH_SIZE * PATCH_SIZE;

        // Patch 0 = (oy=0, ox=0, sy=0, sx=0) = py=0, px=0.
        // Pixels in this patch: (c, y∈[0,14), x∈[0,14)).
        // Inner layout: c-major, then dy, then dx.
        let patch0 = &patches[0..patch_elems];
        let mut k = 0;
        for c in 0..3 {
            for dy in 0..PATCH_SIZE {
                for dx in 0..PATCH_SIZE {
                    let want = (c as f32) * 10000.0 + (dy as f32) * 100.0 + (dx as f32);
                    assert_eq!(
                        patch0[k], want,
                        "patch[0][{k}]: c={c} dy={dy} dx={dx} expected {want}, got {}",
                        patch0[k]
                    );
                    k += 1;
                }
            }
        }

        // Patch 3 = (oy=0, ox=0, sy=1, sx=1) = py=1, px=1.
        // Pixels: (c, y∈[14,28), x∈[14,28)).
        let patch3 = &patches[3 * patch_elems..4 * patch_elems];
        let mut k = 0;
        for c in 0..3 {
            for dy in 0..PATCH_SIZE {
                for dx in 0..PATCH_SIZE {
                    let want = (c as f32) * 10000.0
                        + ((dy + PATCH_SIZE) as f32) * 100.0
                        + ((dx + PATCH_SIZE) as f32);
                    assert_eq!(
                        patch3[k], want,
                        "patch[3][{k}]: c={c} dy={dy} dx={dx} expected {want}, got {}",
                        patch3[k]
                    );
                    k += 1;
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "must be a multiple of PATCH_SIZE")]
    fn extract_patches_panics_on_non_patch_aligned_h() {
        // 30 is not a multiple of PATCH_SIZE=14 — should panic at the
        // assert rather than silently truncate.
        let chw = vec![0.0f32; 3 * 30 * 28];
        let _ = extract_patches(&chw, 30, 28);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of SPATIAL_MERGE_SIZE")]
    fn extract_patches_panics_when_grid_not_sm_aligned() {
        // 14×14 → grid_h=1, grid_w=1 → not divisible by sm=2.
        let chw = vec![0.0f32; 3 * 14 * 14];
        let _ = extract_patches(&chw, 14, 14);
    }

    #[test]
    fn preprocessed_image_helpers() {
        let p = PreprocessedImage {
            patches: vec![],
            grid_h: 4,
            grid_w: 6,
            resized_h: 56,
            resized_w: 84,
        };
        assert_eq!(p.n_patches(), 24);
        assert_eq!(
            p.n_visual_tokens(),
            24 / (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE)
        );
    }

    #[test]
    fn composite_rgba_on_white_blends_alpha() {
        // 1×1 RGBA pixel: (R=200, G=100, B=50, A=128). With alpha
        // blending over white (255,255,255):
        //   R = (200 * 128 + 255 * 127) / 255 ≈ (25600 + 32385) / 255 ≈ 227
        //   G = (100 * 128 + 255 * 127) / 255 ≈ (12800 + 32385) / 255 ≈ 177
        //   B = ( 50 * 128 + 255 * 127) / 255 ≈ ( 6400 + 32385) / 255 ≈ 152
        let mut rgba = image::RgbaImage::new(1, 1);
        rgba.put_pixel(0, 0, image::Rgba([200, 100, 50, 128]));
        let rgb = composite_rgba_on_white(&rgba);
        let p = rgb.get_pixel(0, 0);
        assert!((p[0] as i32 - 227).abs() <= 1, "R = {} (want ~227)", p[0]);
        assert!((p[1] as i32 - 177).abs() <= 1, "G = {} (want ~177)", p[1]);
        assert!((p[2] as i32 - 152).abs() <= 1, "B = {} (want ~152)", p[2]);
    }

    #[test]
    fn composite_rgba_on_white_alpha_0_yields_white() {
        // Fully transparent pixel → white (255, 255, 255).
        let mut rgba = image::RgbaImage::new(1, 1);
        rgba.put_pixel(0, 0, image::Rgba([0, 0, 0, 0]));
        let rgb = composite_rgba_on_white(&rgba);
        let p = rgb.get_pixel(0, 0);
        assert_eq!(p[0], 255);
        assert_eq!(p[1], 255);
        assert_eq!(p[2], 255);
    }

    #[test]
    fn composite_rgba_on_white_alpha_255_preserves_color() {
        // Fully opaque → straight pass-through.
        let mut rgba = image::RgbaImage::new(1, 1);
        rgba.put_pixel(0, 0, image::Rgba([42, 99, 200, 255]));
        let rgb = composite_rgba_on_white(&rgba);
        let p = rgb.get_pixel(0, 0);
        assert_eq!(p[0], 42);
        assert_eq!(p[1], 99);
        assert_eq!(p[2], 200);
    }
}
