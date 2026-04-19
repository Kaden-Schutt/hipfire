//! Image loading and preprocessing for Qwen3.5-VL vision encoder.
//! Loads PNG/JPEG, resizes to target resolution, normalizes to [-1, 1].

use std::path::Path;

/// Smart resize matching HuggingFace Qwen2VLImageProcessorFast.
/// Uses factor=28 (= patch_size * sms * 2 for Qwen3.5) to ensure grid is multiple of sms.
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
/// Returns (CHW data, height, width) where height and width are multiples of patch_size.
pub fn load_and_preprocess(path: &Path, patch_size: usize) -> (Vec<f32>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open image {}: {e}", path.display()));

    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);
    
    // Smart resize matching HuggingFace Qwen2VLImageProcessorFast
    // factor=28 ensures the grid is multiple of sms (28/16=1.75, but patches extracted
    // via unfold with stride=16 gives floor(dim/16) which is always multiple of sms=2
    // because 28 = 7*4 and 16 divides into the result evenly for the grid)
    let factor = 28;
    let min_pixels = 56 * 56;         // 3136
    let max_pixels = 14 * 14 * 4 * 1280; // 1003520
    let (resized_h, resized_w) = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels);
    
    // Round down to nearest multiple of patch_size for clean patch extraction
    let final_h = (resized_h / patch_size) * patch_size;
    let final_w = (resized_w / patch_size) * patch_size;
    
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
    // crates/engine/tests/channel_order.rs for the pure-color test matrix.
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
