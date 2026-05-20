// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 nickfinease
// hipfire — see LICENSE and NOTICE in the project root.

//! Image loading and preprocessing for Qwen3.5-VL vision encoder.
//! Loads PNG/JPEG, resizes to target resolution, normalizes to [-1, 1].

use std::path::Path;

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

/// Load an image, smart-resize to match HuggingFace, normalize.
/// Returns (CHW data, height, width) where height and width are multiples of
/// `patch_size * spatial_merge_size`, so (h/patch_size) and (w/patch_size)
/// are both multiples of `spatial_merge_size` — required by the merger
/// downstream.
pub fn load_and_preprocess(
    path: &Path,
    patch_size: usize,
    spatial_merge_size: usize,
) -> (Vec<f32>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open image {}: {e}", path.display()));

    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);

    // factor = patch_size * spatial_merge_size matches HuggingFace
    // Qwen2_5_VLImageProcessor and guarantees the patch grid is sms-divisible.
    // Qwen3.5-VL: patch_size=16, sms=2 → factor=32. (Qwen2-VL was
    // patch_size=14, sms=2 → factor=28; the literal 28 was wrong here.)
    let factor = patch_size * spatial_merge_size;
    let min_pixels = 56 * 56;         // 3136
    let max_pixels = 14 * 14 * 4 * 1280; // 1003520
    let (final_h, final_w) = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels);

    let img = img.resize_exact(
        final_w as u32,
        final_h as u32,
        image::imageops::FilterType::Triangle,
    );

    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    // Convert to CHW float, normalize: pixel / 127.5 - 1.0
    //
    // Channel-order fix for issue #23: the vision patch_embed weights expect
    // channels in [R, B, G] layout, not [R, G, B]. Empirically confirmed by
    // feeding pure-color PNGs (R=(255,0,0), G=(0,255,0), B=(0,0,255)) through
    // the encoder with temp=0 greedy decoding:
    //
    //   input  | RGB-order (pre-fix) | R<->B swap | B<->G swap (this fix)
    //   -------+---------------------+------------+---------------------
    //   red    | "Red"   ✓           | "Green"    | "Red"   ✓
    //   green  | "Blue"  ✗           | "Blue"     | "Green" ✓
    //   blue   | "Green" ✗           | "Red"      | "Blue"  ✓
    //
    // Root cause is most likely a channel permutation in the HF patch_embed
    // weight export (input conv channels 1 and 2 appear transposed), but the
    // preprocessing swap here resolves the end-to-end symptom. See
    // crates/hipfire-runtime/tests/channel_order.rs for the pure-color test matrix.
    let mut out = vec![0.0f32; 3 * h * w];
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let idx = y * w + x;
            out[idx] = pixel[0] as f32 / 127.5 - 1.0;             // channel 0 = R
            out[plane + idx] = pixel[2] as f32 / 127.5 - 1.0;     // channel 1 = B  (was G)
            out[2 * plane + idx] = pixel[1] as f32 / 127.5 - 1.0; // channel 2 = G  (was B)
        }
    }
    (out, h, w)
}

/// Extract non-overlapping patches from a CHW image.
/// Input: [C, H, W] where H and W are divisible by patch_size.
/// For temporal_patch_size=2, duplicates the frame and interleaves.
/// Output: [N, temporal_patch_size * C * patch_size * patch_size] where N = (H/patch_size) * (W/patch_size).
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

    for py in 0..ph {
        for px in 0..pw {
            let patch_idx = py * pw + px;
            let out_base = patch_idx * patch_elems;
            // For each temporal frame (duplicated for single image)
            for t in 0..temporal_patch_size {
                let _ = t; // same frame duplicated
                for c in 0..channels {
                    for dy in 0..patch_size {
                        for dx in 0..patch_size {
                            let y = py * patch_size + dy;
                            let x = px * patch_size + dx;
                            let src_idx = c * height * width + y * width + x;
                            let dst_idx = out_base
                                + t * channels * patch_size * patch_size
                                + c * patch_size * patch_size
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
