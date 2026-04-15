//! Image loading and preprocessing for Qwen3.5-VL vision encoder.
//! Loads PNG/JPEG, resizes to target resolution, normalizes to [-1, 1].

use std::path::Path;

/// Load an image, resize to target_size x target_size, normalize.
/// Returns [3, target_size, target_size] in CHW order, values in [-1, 1].
pub fn load_and_preprocess(path: &Path, target_size: usize) -> Vec<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open image {}: {e}", path.display()));

    let img = img.resize_exact(
        target_size as u32,
        target_size as u32,
        image::imageops::FilterType::Triangle, // bilinear
    );

    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    // Convert to CHW float, normalize: pixel / 127.5 - 1.0
    let mut out = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                out[c * h * w + y * w + x] = pixel[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    out
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
