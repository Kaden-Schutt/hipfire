// SPDX-License-Identifier: MIT
// Copyright (c) 2026 nickfinease
// hipfire — see LICENSE and NOTICE in the project root.

//! Regression test for the vision-encoder CHW channel ordering.
//!
//! Previously (pre-2026-05-23) this file asserted a deliberate `[R, B, G]`
//! swap that compensated for a different bug (the per-patch `(T,C,h,w)` vs
//! HF's `(C,T,h,w)` transpose in `extract_patches`). Fixing both bugs at the
//! same time required asserting the straight `[R, G, B]` layout — verified
//! byte-identical against HF's `Qwen2VLImageProcessorFast` on `barney_cigar.jpg`
//! (`benchmarks/vision/diff_dumps.py`).
//!
//! This test only locks the preprocessing contract on a single pixel — it
//! does NOT exercise the GPU. It will catch a channel permutation but cannot
//! detect the patch-ordering or per-patch layout bugs on its own: see
//! `extract_patches` for the per-patch layout assertion, and the bench at
//! `benchmarks/vision/comparison-2026-05-23.md` for the end-to-end check.

use std::path::PathBuf;

use hipfire_arch_qwen35_vl::image::load_and_preprocess;
use image::{ImageBuffer, Rgb};

/// Write a 32x32 solid-color PNG to a per-process temp path and return it.
/// PID-scoped so parallel `cargo test --test-threads N` workers (or two CI
/// jobs sharing `/tmp`) cannot collide on the same file.
fn write_solid_png(name: &str, r: u8, g: u8, b: u8) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-channel-order-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(format!("{name}.png"));
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(32, 32, Rgb([r, g, b]));
    img.save(&path).expect("write png");
    path
}

/// Normalization used by `load_and_preprocess`: u8 / 127.5 - 1.0.
fn norm(v: u8) -> f32 {
    v as f32 / 127.5 - 1.0
}

/// After loading, `out` is laid out CHW with shape [3, H, W].
/// Return the value at (channel, y=0, x=0).
fn channel_at_origin(out: &[f32], h: usize, w: usize, channel: usize) -> f32 {
    out[channel * h * w]
}

#[test]
fn pure_red_lands_in_channel_0() {
    let path = write_solid_png("red", 255, 0, 0);
    let (out, h, w) = load_and_preprocess(&path, 16, 2).expect("load_and_preprocess failed");
    assert!(
        (channel_at_origin(&out, h, w, 0) - norm(255)).abs() < 1e-5,
        "R in channel 0"
    );
    assert!(
        (channel_at_origin(&out, h, w, 1) - norm(0)).abs() < 1e-5,
        "G in channel 1 (=0)"
    );
    assert!(
        (channel_at_origin(&out, h, w, 2) - norm(0)).abs() < 1e-5,
        "B in channel 2 (=0)"
    );
}

#[test]
fn pure_green_lands_in_channel_1() {
    let path = write_solid_png("green", 0, 255, 0);
    let (out, h, w) = load_and_preprocess(&path, 16, 2).expect("load_and_preprocess failed");
    assert!(
        (channel_at_origin(&out, h, w, 0) - norm(0)).abs() < 1e-5,
        "R in channel 0 (=0)"
    );
    assert!(
        (channel_at_origin(&out, h, w, 1) - norm(255)).abs() < 1e-5,
        "G in channel 1"
    );
    assert!(
        (channel_at_origin(&out, h, w, 2) - norm(0)).abs() < 1e-5,
        "B in channel 2 (=0)"
    );
}

#[test]
fn pure_blue_lands_in_channel_2() {
    let path = write_solid_png("blue", 0, 0, 255);
    let (out, h, w) = load_and_preprocess(&path, 16, 2).expect("load_and_preprocess failed");
    assert!(
        (channel_at_origin(&out, h, w, 0) - norm(0)).abs() < 1e-5,
        "R in channel 0 (=0)"
    );
    assert!(
        (channel_at_origin(&out, h, w, 1) - norm(0)).abs() < 1e-5,
        "G in channel 1 (=0)"
    );
    assert!(
        (channel_at_origin(&out, h, w, 2) - norm(255)).abs() < 1e-5,
        "B in channel 2"
    );
}

#[test]
fn mixed_pixel_keeps_rgb_order() {
    // Distinctive per-channel values so any transposition shows up clearly.
    let path = write_solid_png("mixed", 10, 200, 50);
    let (out, h, w) = load_and_preprocess(&path, 16, 2).expect("load_and_preprocess failed");
    assert!(
        (channel_at_origin(&out, h, w, 0) - norm(10)).abs() < 1e-5,
        "R (10) in channel 0"
    );
    assert!(
        (channel_at_origin(&out, h, w, 1) - norm(200)).abs() < 1e-5,
        "G (200) in channel 1"
    );
    assert!(
        (channel_at_origin(&out, h, w, 2) - norm(50)).abs() < 1e-5,
        "B (50) in channel 2"
    );
}

/// Write a 256×256 PNG with different colors in each quadrant, returning
/// its path. 256×256 is a multiple of factor=32 (patch_size=16 ×
/// spatial_merge_size=2) and above min_pixels=3136, so smart_resize
/// keeps it at 256×256 unchanged.
/// Each quadrant uses distinct per-channel values that form a
/// unique "fingerprint" — so a spatial transpose (H↔V flip), a channel
/// swap, or any combination will corrupt at least one fingerprint.
fn write_quadrant_png(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hipfire-channel-order-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(format!("{name}.png"));
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(256, 256);
    // Quadrant colors chosen so each R,G,B triple is unique and no
    // per-channel value appears in the same position across quadrants.
    let colors = [
        (200, 30, 100), // top-left
        (50, 180, 15),  // top-right
        (10, 70, 240),  // bottom-left
        (150, 5, 60),   // bottom-right
    ];
    for (y, row) in img.rows_mut().enumerate() {
        for (x, pixel) in row.enumerate() {
            let qi = if y < 128 { 0 } else { 2 } + if x < 128 { 0 } else { 1 };
            let (r, g, b) = colors[qi];
            *pixel = Rgb([r, g, b]);
        }
    }
    img.save(&path).expect("write png");
    path
}

#[test]
fn quadrant_pixels_keep_rgb_spatial_order() {
    // Verifies channel AND spatial ordering simultaneously: each quadrant
    // of the input image has a unique (R,G,B) fingerprint, and we check
    // that the correct fingerprint lands at the correct (y,x) position
    // in each CHW channel plane.
    let path = write_quadrant_png("quadrant");
    let (out, h, w) = load_and_preprocess(&path, 16, 2).expect("load_and_preprocess failed");

    // 256×256 with patch_size=16, spatial_merge=2 → factor=32.
    // smart_resize keeps it at 256×256 (already a multiple of 32,
    // and 65536 > min_pixels=3136).
    assert_eq!(h, 256, "expected height 256, got {}", h);
    assert_eq!(w, 256, "expected width 256, got {}", w);

    // Check a pixel in each quadrant of the CHW output.
    let colors = [
        (200u8, 30u8, 100u8), // TL
        (50u8, 180u8, 15u8),  // TR
        (10u8, 70u8, 240u8),  // BL
        (150u8, 5u8, 60u8),   // BR
    ];
    let positions = [
        (64, 64),   // TL interior
        (64, 192),  // TR interior
        (192, 64),  // BL interior
        (192, 192), // BR interior
    ];
    // Note: CHW layout is [C, H, W], so pixel at (y, x) in channel c is:
    //   out[c * h * w + y * w + x]
    for (qi, &(py, px)) in positions.iter().enumerate() {
        let (r, g, b) = colors[qi];
        let pixel_offset = py * w + px;
        let channel_stride = h * w;
        let r_val = out[pixel_offset];
        let g_val = out[channel_stride + pixel_offset];
        let b_val = out[2 * channel_stride + pixel_offset];
        assert!(
            (r_val - norm(r)).abs() < 1e-4,
            "Q{} ({},{}) R: expected {} got {}",
            qi,
            py,
            px,
            norm(r),
            r_val
        );
        assert!(
            (g_val - norm(g)).abs() < 1e-4,
            "Q{} ({},{}) G: expected {} got {}",
            qi,
            py,
            px,
            norm(g),
            g_val
        );
        assert!(
            (b_val - norm(b)).abs() < 1e-4,
            "Q{} ({},{}) B: expected {} got {}",
            qi,
            py,
            px,
            norm(b),
            b_val
        );
    }
}
