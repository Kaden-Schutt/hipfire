//! Smoke test for `Gpus` orchestrator: init_uniform layer split, peer
//! enable, boundary_copy + wait_boundary round-trip.
//!
//! Run: HIP_VISIBLE_DEVICES=0,1 cargo run -p hipfire-runtime --example gpus_smoke

use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::{DType, Gpu};

fn main() {
    println!("── Gpus::init_uniform(2, 24) ─────────────────────────────");
    let mut gpus = Gpus::init_uniform(2, 24).expect("init_uniform");
    assert_eq!(gpus.devices.len(), 2);
    assert_eq!(gpus.layer_to_device.len(), 24);
    assert_eq!(gpus.band_starts, vec![0, 12]);
    assert_eq!(gpus.output_device, 1);
    println!("  layer_to_device: {:?}", gpus.layer_to_device);
    println!("  band_starts:     {:?}", gpus.band_starts);
    println!("  output_device:   {}", gpus.output_device);

    for layer in 0..12 {
        assert_eq!(gpus.device_for_layer(layer), 0, "layer {layer} on dev 0");
    }
    for layer in 12..24 {
        assert_eq!(gpus.device_for_layer(layer), 1, "layer {layer} on dev 1");
    }
    assert!(gpus.is_band_boundary(11), "layer 11 → 12 is a boundary");
    assert!(!gpus.is_band_boundary(12), "layer 12 → 13 is not");
    assert!(!gpus.is_band_boundary(23), "last layer has no successor");

    println!("\n── allocate src/dst BEFORE enable_peer_all (ROCm gotcha) ─");
    let n_elems = 1024usize;
    let bytes = n_elems * 4;
    let pattern: Vec<f32> = (0..n_elems).map(|i| i as f32 * 0.5).collect();

    gpus.devices[0].bind_thread().expect("bind 0");
    let src_t = gpus.devices[0]
        .upload_f32(&pattern, &[n_elems])
        .expect("upload src dev 0");
    gpus.devices[1].bind_thread().expect("bind 1");
    let dst_t = gpus.devices[1]
        .zeros(&[n_elems], DType::F32)
        .expect("zeros dst dev 1");

    println!("\n── Gpus::enable_peer_all() ───────────────────────────────");
    let peer_ok = gpus.enable_peer_all().expect("enable_peer_all");
    assert!(peer_ok, "peer access should be bidirectional on 2× 7900 XTX");
    assert!(gpus.peer_access_enabled);
    println!("  peer_access_enabled = true");

    println!("\n── boundary_copy 0→1 (null-stream path) ──────────────────");
    let evt = gpus
        .boundary_copy(0, 1, &src_t.buf, &dst_t.buf, bytes)
        .expect("boundary_copy");
    assert_eq!(evt.dst_dev, 1);
    gpus.wait_boundary(evt).expect("wait_boundary");

    let dst_host = gpus.devices[1].download_f32(&dst_t).expect("download dst");
    assert_eq!(dst_host, pattern, "boundary_copy 0→1 byte-equality");
    println!("  1024 f32 elements: byte-identical after boundary_copy + wait");

    println!("\n── boundary_copy 1→0 (reverse direction, distinct pattern) ─");
    // Going through a fresh upload (rather than memset(src,0) + reverse) avoids
    // the ROCm 6.4.3 quirk where memset immediately followed by a peer read
    // silently no-ops on the same page.
    let reverse_pattern: Vec<f32> = (0..n_elems).map(|i| -(i as f32) * 0.25).collect();
    gpus.devices[1].bind_thread().expect("bind 1 for upload reverse");
    let bytes_view = unsafe {
        std::slice::from_raw_parts(reverse_pattern.as_ptr() as *const u8, bytes)
    };
    gpus.devices[1]
        .hip
        .memcpy_htod(&dst_t.buf, bytes_view)
        .expect("upload reverse pattern to dst");
    let evt = gpus
        .boundary_copy(1, 0, &dst_t.buf, &src_t.buf, bytes)
        .expect("boundary_copy reverse");
    gpus.wait_boundary(evt).expect("wait_boundary reverse");
    let src_host = gpus.devices[0].download_f32(&src_t).expect("download src");
    assert_eq!(
        src_host, reverse_pattern,
        "1→0 boundary_copy must carry reverse_pattern from dev 1 to dev 0"
    );
    println!("  reverse copy verified — dev 0 holds new pattern after 1→0");

    gpus.devices[0].free_tensor(src_t).expect("free src");
    gpus.devices[1].bind_thread().expect("bind 1 for free");
    gpus.devices[1].free_tensor(dst_t).expect("free dst");

    println!("\n── Gpus::single back-compat ──────────────────────────────");
    drop(gpus); // release dev 0/1 before re-init for single
    let solo = Gpu::init_with_device(0).expect("init solo");
    let single = Gpus::single(solo, 24);
    assert_eq!(single.devices.len(), 1);
    assert_eq!(single.layer_to_device, vec![0u8; 24]);
    assert_eq!(single.output_device, 0);
    for layer in 0..24 {
        assert_eq!(single.device_for_layer(layer), 0);
        assert!(!single.is_band_boundary(layer), "no boundaries in PP=1");
    }
    println!("  PP=1 wrap: 24 layers all on dev 0, no band boundaries");

    println!("\ngpus_smoke: PASS");
}
