// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3/SigLIP image preprocessing. See LICENSE / NOTICE.

//! Decode an image → SigLIP patch tensor for `vision_forward`.
//!
//! Gemma3's image processor: resize to `image_size × image_size` (bicubic;
//! CatmullRom here), rescale `1/255`, normalize with mean/std `0.5` → `[-1, 1]`.
//! Then im2col into `[num_patches, 3·patch²]` where each patch's `3·patch²`
//! values are ordered **channel-major** `[c][kh][kw]` and patches run row-major
//! over the `grid×grid` layout — matching the flattened Conv2d `patch_embedding`
//! weight `[hidden, 3·patch²]` and `vision_forward`'s expected input.

use std::path::Path;

use crate::config::SigLipConfig;

/// Load + preprocess an image into the SigLIP patch tensor
/// `[num_patches · 3·patch²]` (row-major patches, channel-major within a patch).
pub fn preprocess_image(path: &Path, cfg: &SigLipConfig) -> Result<Vec<f32>, String> {
    let img = image::open(path).map_err(|e| format!("gemma3-vl: open image {path:?}: {e}"))?;
    let size = cfg.image_size as u32;
    // Resize to a fixed square (Gemma3 uses a fixed image_size grid).
    let img = img
        .resize_exact(size, size, image::imageops::FilterType::CatmullRom)
        .to_rgb8();

    let s = cfg.image_size;
    let ps = cfg.patch_size;
    let grid = cfg.grid_side();
    let c = cfg.num_channels; // 3
    let patch_dim = c * ps * ps;
    let n = grid * grid;

    // Normalized CHW pixels in [-1, 1]: (x/255 - 0.5) / 0.5 = x/127.5 - 1.
    let mut chw = vec![0.0f32; c * s * s];
    for y in 0..s {
        for x in 0..s {
            let px = img.get_pixel(x as u32, y as u32);
            for ch in 0..c {
                chw[ch * s * s + y * s + x] = px[ch] as f32 / 127.5 - 1.0;
            }
        }
    }

    // im2col: patch (gr, gc) → 588 values ordered [c][kh][kw].
    let mut out = vec![0.0f32; n * patch_dim];
    for gr in 0..grid {
        for gc in 0..grid {
            let pbase = (gr * grid + gc) * patch_dim;
            for ch in 0..c {
                for kh in 0..ps {
                    for kw in 0..ps {
                        let y = gr * ps + kh;
                        let x = gc * ps + kw;
                        out[pbase + ch * ps * ps + kh * ps + kw] = chw[ch * s * s + y * s + x];
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SigLipConfig {
        SigLipConfig {
            hidden_size: 1152,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            intermediate_size: 4304,
            patch_size: 14,
            image_size: 896,
            num_channels: 3,
            layer_norm_eps: 1e-6,
        }
    }

    #[test]
    fn preprocess_shape_and_range() {
        // Synthesize a tiny solid image, save as PNG, preprocess.
        let dir = std::env::temp_dir();
        let p = dir.join("gemma3vl_test_solid.png");
        let buf = image::RgbImage::from_pixel(32, 48, image::Rgb([255, 0, 128]));
        buf.save(&p).unwrap();
        let c = cfg();
        let patches = preprocess_image(&p, &c).unwrap();
        assert_eq!(
            patches.len(),
            c.num_patches() * c.num_channels * c.patch_size * c.patch_size
        );
        // 255 → +1, 0 → -1, 128 → ~0.004. All within [-1, 1].
        assert!(patches.iter().all(|&v| (-1.0001..=1.0001).contains(&v)));
        // First value is channel 0 (R=255) of patch 0 → ~+1.
        assert!((patches[0] - 1.0).abs() < 1e-3);
        let _ = std::fs::remove_file(&p);
    }
}
