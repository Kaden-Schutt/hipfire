// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 nickfinease
// hipfire — see LICENSE and NOTICE in the project root.

//! Image loading and preprocessing for Qwen3.5-VL vision encoder.
//! Loads PNG/JPEG, resizes to target resolution, normalizes to [-1, 1].

use std::path::Path;

/// Maximum total pixel count, checked from format-header dimensions BEFORE
/// the pixel buffer is allocated (decompression bomb guard). 4M pixels is
/// well above `smart_resize`'s `max_pixels` target of ~1M but caps the
/// pre-decode memory request from a malicious 50000×50000 PNG.
const MAX_DIMENSION_PIXELS: usize = 4_000_000;

/// Smart resize matching HuggingFace Qwen2_5_VLImageProcessor.
///
/// `factor` MUST equal `patch_size * spatial_merge_size`. With that constraint
/// the returned (h, w) are multiples of `patch_size * sms`, which guarantees
/// (1) clean patch extraction at `patch_size` stride and (2) a patch grid
/// divisible by `sms` so the spatial merger does not silently truncate a
/// row/column. Passing any other factor (e.g. the legacy `28` from Qwen2-VL
/// when patch_size=16) yields odd patch grids on small images and a
/// merger/LM token-count mismatch downstream.
pub fn smart_resize(height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    let h_bar = ((height as f64 / factor as f64).round() as usize) * factor;
    let w_bar = ((width as f64 / factor as f64).round() as usize) * factor;
    
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        let h_bar = factor.max(((height as f64 / beta / factor as f64).floor() as usize) * factor);
        let w_bar = factor.max(((width as f64 / beta / factor as f64).floor() as usize) * factor);
        (h_bar, w_bar)
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        let h_bar = factor.max(((height as f64 * beta / factor as f64).ceil() as usize) * factor);
        let w_bar = factor.max(((width as f64 * beta / factor as f64).ceil() as usize) * factor);
        (h_bar, w_bar)
    } else {
        (h_bar, w_bar)
    }
}

/// Shared preprocessing logic that takes an already-loaded `DynamicImage`.
/// Returns (CHW data, height, width) where height and width are multiples of
/// `patch_size * spatial_merge_size`.
fn preprocess_dynamic_image(
    img: image::DynamicImage,
    patch_size: usize,
    spatial_merge_size: usize,
) -> (Vec<f32>, usize, usize) {
    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);

    let factor = patch_size * spatial_merge_size;
    let min_pixels = 56 * 56;
    let max_pixels = 14 * 14 * 4 * 1280;
    let (final_h, final_w) = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels);

    let img = img.resize_exact(
        final_w as u32,
        final_h as u32,
        image::imageops::FilterType::Triangle,
    );

    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    // CHW float in straight [R, G, B] order, normalize: pixel / 127.5 - 1.0.
    //
    // History: previously had a deliberate B<->G swap (storing as [R, B, G])
    // because pure-color PNG tests said red/green/blue were misnamed without
    // it. That diagnosis was wrong — the real cause was the (T,C,H,W) vs
    // (C,T,H,W) per-patch transpose in `extract_patches` below, which on a
    // single-color image happens to be re-fixable by any single channel
    // permutation. On natural images the two bugs compound and the swap
    // makes things strictly worse. Verified byte-identical to HF's
    // Qwen2VLImageProcessorFast on `barney_cigar.jpg` once the layout is
    // straight RGB AND extract_patches uses (C,T,H,W). See
    // benchmarks/vision/comparison-2026-05-23.md.
    let mut out = vec![0.0f32; 3 * h * w];
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let idx = y * w + x;
            out[idx] = pixel[0] as f32 / 127.5 - 1.0;             // channel 0 = R
            out[plane + idx] = pixel[1] as f32 / 127.5 - 1.0;     // channel 1 = G
            out[2 * plane + idx] = pixel[2] as f32 / 127.5 - 1.0; // channel 2 = B
        }
    }
    (out, h, w)
}

/// Load an image from a filesystem path, smart-resize, normalize.
/// Panics on failure (CLI path).
pub fn load_and_preprocess(
    path: &Path,
    patch_size: usize,
    spatial_merge_size: usize,
) -> (Vec<f32>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open image {}: {e}", path.display()));
    preprocess_dynamic_image(img, patch_size, spatial_merge_size)
}

/// Load an image from raw bytes (PNG or JPEG), smart-resize, normalize.
/// Returns `Result` so callers can surface decode errors.
///
/// Reads dimensions from the format header BEFORE decoding pixels so a
/// decompression-bomb image (e.g. 50000×50000 PNG that compresses to ~50 KB
/// but expands to multi-GB raw) is rejected before allocation. Format
/// rejection (non-PNG/JPEG) is via `ImageError::Unsupported` rather than
/// substring matching so the error surface is stable across `image` crate
/// versions.
pub fn load_and_preprocess_from_bytes(
    data: &[u8],
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(Vec<f32>, usize, usize), String> {
    let reader = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| format!("failed to read image: {e}"))?;

    let (orig_w, orig_h) = reader
        .into_dimensions()
        .map_err(map_image_err)?;
    let (orig_w, orig_h) = (orig_w as usize, orig_h as usize);
    if orig_w * orig_h > MAX_DIMENSION_PIXELS {
        return Err(format!(
            "image dimensions ({orig_w}x{orig_h}) exceed maximum ({MAX_DIMENSION_PIXELS} pixels)"
        ));
    }

    let img = image::load_from_memory(data).map_err(map_image_err)?;
    Ok(preprocess_dynamic_image(img, patch_size, spatial_merge_size))
}

fn map_image_err(e: image::ImageError) -> String {
    match e {
        image::ImageError::Unsupported(_) => {
            "unsupported image format — supported: png, jpeg".to_string()
        }
        other => format!("failed to decode image: {other}"),
    }
}

/// Extract non-overlapping patches from a CHW image.
///
/// Input: `[C, H, W]` where H and W are divisible by `patch_size` (and
/// `patch_size * spatial_merge` — currently 2, enforced by [`smart_resize`]).
/// Output: `[N, temporal_patch_size * C * patch_size * patch_size]` with
/// `N = (H/patch_size) * (W/patch_size)`, ordered to match HuggingFace
/// `Qwen2VLImageProcessorFast`:
///
///   * Patches are emitted in **2x2 spatial-merge-grouped order**: outer
///     `(gy, gx)` row-major over `(ph/2, pw/2)`, inner `(sy, sx)` row-major
///     over `(2, 2)`. This means 4 consecutive patches in the output buffer
///     form one spatial-merge output token.
///   * Per-patch layout is `(C, T, patch_h, patch_w)` flat —
///     **channel-outer, temporal-inner**.
///
/// Pre-2026-05-23 this function used row-major patch order and `(T, C, h, w)`
/// per-patch layout, both of which disagree with HF. Combined with the
/// `[R, B, G]` channel swap in [`preprocess_dynamic_image`] (also reverted in
/// that change) the vision tower received patches that were scrambled in
/// three independent dimensions — model perceived every image as "vertically
/// stretched / low-resolution / blurry". Diagnosed by element-wise diff
/// against HF reference on `barney_cigar.jpg`: rel-L1 dropped from 0.35
/// (pre-fix) to 0.002 (post-fix, residual is the resize-filter difference).
/// See `benchmarks/vision/comparison-2026-05-23.md` and `diff_dumps.py`.
pub fn extract_patches(
    chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    temporal_patch_size: usize,
) -> Vec<f32> {
    let ph = height / patch_size;
    let pw = width / patch_size;
    let n_patches = ph * pw;
    let patch_elems = temporal_patch_size * channels * patch_size * patch_size;
    let mut patches = vec![0.0f32; n_patches * patch_elems];

    let spatial_merge: usize = 2;
    assert!(
        ph % spatial_merge == 0 && pw % spatial_merge == 0,
        "patch grid {ph}x{pw} not divisible by spatial_merge={spatial_merge} — \
         smart_resize should guarantee this",
    );
    let gw = pw / spatial_merge;

    for py in 0..ph {
        for px in 0..pw {
            let gy = py / spatial_merge;
            let gx = px / spatial_merge;
            let sy = py % spatial_merge;
            let sx = px % spatial_merge;
            // 2x2-block-grouped row-major: ((gy, gx), (sy, sx)) flattened.
            let patch_out_idx = ((gy * gw + gx) * spatial_merge + sy) * spatial_merge + sx;
            let out_base = patch_out_idx * patch_elems;

            for c in 0..channels {
                for t in 0..temporal_patch_size {
                    let _ = t; // same frame duplicated for both temporal slots
                    for dy in 0..patch_size {
                        for dx in 0..patch_size {
                            let y = py * patch_size + dy;
                            let x = px * patch_size + dx;
                            let src_idx = c * height * width + y * width + x;
                            let dst_idx = out_base
                                + c * temporal_patch_size * patch_size * patch_size
                                + t * patch_size * patch_size
                                + dy * patch_size
                                + dx;
                            patches[dst_idx] = chw[src_idx];
                        }
                    }
                }
            }
        }
    }
    patches
}
