//! Tests for `load_and_preprocess_from_bytes` and the decompression-bomb guard.
//!
//! No GPU needed — exercises the pure image decode + resize + normalize path
//! using synthetic images constructed in memory via the `image` crate.

use hipfire_arch_qwen35_vl::image::{load_and_preprocess_from_bytes, smart_resize};
use image::{ImageBuffer, Rgb};

fn solid_png_bytes(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(w, h, Rgb([r, g, b]));
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png).unwrap();
    buf
}

fn solid_jpeg_bytes(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(w, h, Rgb([r, g, b]));
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg).unwrap();
    buf
}

fn norm(v: u8) -> f32 {
    v as f32 / 127.5 - 1.0
}

fn channel_at(out: &[f32], h: usize, w: usize, ch: usize, x: usize, y: usize) -> f32 {
    out[ch * h * w + y * w + x]
}

#[test]
fn valid_png_round_trips() {
    let bytes = solid_png_bytes(128, 64, 32, 64, 64);
    let result = load_and_preprocess_from_bytes(&bytes, 16, 2);
    assert!(result.is_ok(), "valid PNG should decode: {result:?}");
    let (out, h, w) = result.unwrap();
    assert_eq!(out.len(), 3 * h * w);
    assert_eq!(h % (16 * 2), 0, "height should be divisible by factor=32");
    assert_eq!(w % (16 * 2), 0, "width should be divisible by factor=32");
}

#[test]
fn valid_jpeg_round_trips() {
    let bytes = solid_jpeg_bytes(200, 100, 50, 128, 128);
    let result = load_and_preprocess_from_bytes(&bytes, 16, 2);
    assert!(result.is_ok(), "valid JPEG should decode: {result:?}");
    let (out, h, w) = result.unwrap();
    assert_eq!(out.len(), 3 * h * w);
}

#[test]
fn output_matches_load_and_preprocess_for_same_input() {
    // Use a unique tempdir per test invocation so parallel runs (and CI
    // workers sharing /tmp) can't collide on the fixture file.
    let bytes = solid_png_bytes(10, 200, 50, 32, 32);
    let (from_bytes, h1, w1) = load_and_preprocess_from_bytes(&bytes, 16, 2).unwrap();

    let dir = std::env::temp_dir().join(format!(
        "hipfire-img-test-equiv-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("equiv.png");
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(32, 32, Rgb([10, 200, 50]));
    img.save(&path).unwrap();
    let (from_path, h2, w2) = hipfire_arch_qwen35_vl::image::load_and_preprocess(&path, 16, 2)
        .expect("load_and_preprocess failed");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!((h1, w1), (h2, w2), "dimensions should match between from-bytes and from-path");
    assert_eq!(from_bytes.len(), from_path.len(), "output length should match");
    for i in 0..from_bytes.len() {
        assert!(
            (from_bytes[i] - from_path[i]).abs() < 1e-5,
            "mismatch at index {i}: from_bytes={:.6} from_path={:.6}",
            from_bytes[i],
            from_path[i],
        );
    }
}

#[test]
fn channel_order_preserved_from_bytes() {
    // Use distinct values per channel so the test actually verifies
    // channel ordering — pure red (255, 0, 0) is identical under RGB
    // and BGR and would silently pass either way.
    //
    // Implementation outputs CHW in canonical (R, G, B) order. The prior
    // (R, B, G) swap was a misdiagnosis (see `image.rs` history note and
    // `benchmarks/vision/comparison-2026-05-23.md`); the real bug was the
    // patch transpose in `extract_patches`, not the channel layout.
    let bytes = solid_png_bytes(10, 100, 200, 32, 32);
    let (out, h, w) = load_and_preprocess_from_bytes(&bytes, 16, 2).unwrap();
    assert!(
        (channel_at(&out, h, w, 0, 0, 0) - norm(10)).abs() < 1e-4,
        "channel 0 should carry R=10 (got {:.4})",
        channel_at(&out, h, w, 0, 0, 0),
    );
    assert!(
        (channel_at(&out, h, w, 1, 0, 0) - norm(100)).abs() < 1e-4,
        "channel 1 should carry G=100 (got {:.4})",
        channel_at(&out, h, w, 1, 0, 0),
    );
    assert!(
        (channel_at(&out, h, w, 2, 0, 0) - norm(200)).abs() < 1e-4,
        "channel 2 should carry B=200 (got {:.4})",
        channel_at(&out, h, w, 2, 0, 0),
    );
}

#[test]
fn empty_bytes_returns_decode_error() {
    let result = load_and_preprocess_from_bytes(&[], 16, 2);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_lowercase();
    assert!(
        msg.contains("failed to decode image") || msg.contains("unsupported"),
        "empty input should produce a decode error, got: {msg}"
    );
}

#[test]
fn random_garbage_returns_decode_error() {
    let result = load_and_preprocess_from_bytes(&[0xDE, 0xAD, 0xBE, 0xEF], 16, 2);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_lowercase();
    assert!(
        msg.contains("failed to decode image") || msg.contains("unsupported"),
        "garbage bytes should produce a decode error, got: {msg}"
    );
}

#[test]
fn dimension_bomb_rejected() {
    let bytes = solid_png_bytes(0, 0, 0, 3000, 3000);
    let result = load_and_preprocess_from_bytes(&bytes, 16, 2);
    assert!(result.is_err());
    let msg = result.unwrap_err();
    assert!(
        msg.contains("exceed maximum"),
        "3000x3000 (9M) should exceed 4M ceiling, got: {msg}"
    );
}

#[test]
fn dimension_bomb_just_under_limit_passes() {
    let bytes = solid_png_bytes(0, 0, 0, 1999, 1999);
    let result = load_and_preprocess_from_bytes(&bytes, 16, 2);
    assert!(result.is_ok(), "1999x1999 (3.996M) should be under 4M ceiling: {result:?}");
}

#[test]
fn dimension_bomb_exact_boundary_passes() {
    // Exactly at the 4M limit — `>` not `>=`, so 2000×2000 = 4_000_000
    // should pass (the inclusive upper bound). Locks the boundary
    // behaviour against future drift.
    let bytes = solid_png_bytes(0, 0, 0, 2000, 2000);
    let result = load_and_preprocess_from_bytes(&bytes, 16, 2);
    assert!(result.is_ok(), "2000x2000 (4M, exactly at limit) should pass: {result:?}");
}

#[test]
fn smart_resize_divisibility() {
    let (h, w) = smart_resize(800, 600, 32, 3136, 1_003_520);
    assert_eq!(h % 32, 0, "height must be divisible by factor=32");
    assert_eq!(w % 32, 0, "width must be divisible by factor=32");
    assert_eq!(h % 16, 0, "height must be divisible by patch_size=16");
    assert_eq!(w % 16, 0, "width must be divisible by patch_size=16");
    let sms = 2;
    assert_eq!((h / 16) % sms, 0, "grid height must be divisible by spatial_merge_size");
    assert_eq!((w / 16) % sms, 0, "grid width must be divisible by spatial_merge_size");
}

#[test]
fn smart_resize_large_image_downscales() {
    let (h, w) = smart_resize(4096, 4096, 32, 3136, 1_003_520);
    assert!(h * w <= 1_003_520, "large image should be downscaled to max_pixels");
    assert!(h > 0 && w > 0);
}

#[test]
fn smart_resize_tiny_image_upscales() {
    let (h, w) = smart_resize(10, 10, 32, 3136, 1_003_520);
    assert!(h * w >= 3136, "tiny image should be upscaled to min_pixels");
    assert!(h > 0 && w > 0);
}

#[test]
fn smart_resize_normal_image_unchanged() {
    let (h, w) = smart_resize(512, 512, 32, 3136, 1_003_520);
    assert_eq!((h, w), (512, 512), "512x512 is within bounds and factor-aligned — should be unchanged");
}

#[test]
fn aspect_ratio_preserved_on_downscale() {
    let (h, w) = smart_resize(800, 600, 32, 3136, 1_003_520);
    let ratio_in = 800.0 / 600.0;
    let ratio_out = h as f64 / w as f64;
    assert!(
        (ratio_in - ratio_out).abs() < 0.05,
        "aspect ratio should be roughly preserved: in={ratio_in:.3} out={ratio_out:.3}"
    );
}

#[test]
fn zero_dimension_floors_to_factor() {
    let (h, w) = smart_resize(0, 0, 32, 3136, 1_003_520);
    assert!(h >= 32, "zero-dimension input should floor to factor");
    assert!(w >= 32, "zero-dimension input should floor to factor");
}
