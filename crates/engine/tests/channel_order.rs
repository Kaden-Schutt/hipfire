//! Regression test for the vision-encoder channel ordering fix (issue #23).
//!
//! Empirical finding: the patch_embed weights in the Qwen3.5-VL HFQ4 export
//! expect input CHW tensors in [R, B, G] order rather than [R, G, B]. Feeding
//! pure-color PNGs through the encoder with temp=0 greedy decoding confirmed
//! this — pre-fix, red worked, green→"Blue", blue→"Green"; post-fix all three
//! are identified correctly.
//!
//! This test does not exercise the GPU; it locks the preprocessing contract
//! that `load_and_preprocess` produces channel 0 = R, channel 1 = B (source
//! green pixel's blue byte is NOT used; source blue byte is), channel 2 = G.
//! If someone removes the swap, these assertions fail before any vision
//! inference runs.
//!
//! See `crates/engine/src/image.rs` for the swap itself.

use std::path::PathBuf;

use engine::image::load_and_preprocess;
use image::{ImageBuffer, Rgb};

/// Write a 32x32 solid-color PNG to a temp path and return the path.
fn write_solid_png(name: &str, r: u8, g: u8, b: u8) -> PathBuf {
    let dir = std::env::temp_dir().join("hipfire-channel-order-tests");
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
fn pure_red_preserves_red_channel() {
    // Red = (255, 0, 0). patch_size=16 is what Qwen3.5-VL uses.
    let path = write_solid_png("red", 255, 0, 0);
    let (out, h, w) = load_and_preprocess(&path, 16);
    // Channel layout post-fix: [R, B, G].
    assert!(
        (channel_at_origin(&out, h, w, 0) - norm(255)).abs() < 1e-5,
        "R channel should carry 255"
    );
    assert!(
        (channel_at_origin(&out, h, w, 1) - norm(0)).abs() < 1e-5,
        "B channel slot should carry B (0) from red pixel"
    );
    assert!(
        (channel_at_origin(&out, h, w, 2) - norm(0)).abs() < 1e-5,
        "G channel slot should carry G (0) from red pixel"
    );
}

#[test]
fn pure_green_routes_g_byte_to_channel_2() {
    // Green = (0, 255, 0). Pre-fix this was going to channel 1 and the model
    // saw it as blue. Post-fix the G byte lands in channel 2.
    let path = write_solid_png("green", 0, 255, 0);
    let (out, h, w) = load_and_preprocess(&path, 16);
    assert!(
        (channel_at_origin(&out, h, w, 0) - norm(0)).abs() < 1e-5,
        "R channel should be 0 for pure green"
    );
    assert!(
        (channel_at_origin(&out, h, w, 1) - norm(0)).abs() < 1e-5,
        "channel 1 should carry B (0) from green pixel — NOT G"
    );
    assert!(
        (channel_at_origin(&out, h, w, 2) - norm(255)).abs() < 1e-5,
        "channel 2 should carry G (255) from green pixel"
    );
}

#[test]
fn pure_blue_routes_b_byte_to_channel_1() {
    // Blue = (0, 0, 255). Pre-fix this was going to channel 2 and the model
    // saw it as green. Post-fix the B byte lands in channel 1.
    let path = write_solid_png("blue", 0, 0, 255);
    let (out, h, w) = load_and_preprocess(&path, 16);
    assert!(
        (channel_at_origin(&out, h, w, 0) - norm(0)).abs() < 1e-5,
        "R channel should be 0 for pure blue"
    );
    assert!(
        (channel_at_origin(&out, h, w, 1) - norm(255)).abs() < 1e-5,
        "channel 1 should carry B (255) from blue pixel"
    );
    assert!(
        (channel_at_origin(&out, h, w, 2) - norm(0)).abs() < 1e-5,
        "channel 2 should carry G (0) from blue pixel — NOT B"
    );
}

#[test]
fn mixed_pixel_round_trips_all_three_bytes() {
    // Distinctive per-channel values so a transposition bug would be obvious.
    let path = write_solid_png("mixed", 10, 200, 50);
    let (out, h, w) = load_and_preprocess(&path, 16);
    assert!(
        (channel_at_origin(&out, h, w, 0) - norm(10)).abs() < 1e-5,
        "R (10) should land in channel 0"
    );
    assert!(
        (channel_at_origin(&out, h, w, 1) - norm(50)).abs() < 1e-5,
        "B (50) should land in channel 1 post-fix"
    );
    assert!(
        (channel_at_origin(&out, h, w, 2) - norm(200)).abs() < 1e-5,
        "G (200) should land in channel 2 post-fix"
    );
}
