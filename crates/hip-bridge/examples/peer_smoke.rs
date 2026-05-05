//! Peer-access smoke: bidirectional enable_peer_access, hipMemcpyPeer
//! byte-equality round-trip, pointer_get_attributes device verification.
//!
//! Run: HIP_VISIBLE_DEVICES=0,1 cargo run -p hip-bridge --example peer_smoke

use hip_bridge::{HipRuntime, MemoryType};

const SIZE: usize = 1 << 20; // 1 MiB

fn main() {
    println!("Loading HIP runtime via dlopen...");
    let hip = HipRuntime::load().expect("failed to load HIP runtime");

    let count = hip.device_count().expect("failed to get device count");
    println!("Visible devices: {count}");
    assert!(count >= 2, "peer_smoke requires ≥2 devices (got {count}). Set HIP_VISIBLE_DEVICES=0,1");

    for id in 0..count {
        hip.set_device(id).expect("set_device");
        let arch = hip.get_arch(id).unwrap_or_else(|_| "unknown".to_string());
        let cur = hip.current_device().expect("current_device");
        let (free, total) = hip.get_vram_info().unwrap_or((0, 0));
        println!(
            "  dev {id}: arch={arch} current_device={cur} vram={:.1}/{:.1} GiB",
            free as f64 / 1e9,
            total as f64 / 1e9,
        );
        assert_eq!(cur, id, "current_device disagrees with set_device");
    }

    // ── Probe peer access ────────────────────────────────────────
    println!("\nProbing peer access (0↔1):");
    let can_0_to_1 = hip.can_access_peer(0, 1).expect("can_access_peer 0→1");
    let can_1_to_0 = hip.can_access_peer(1, 0).expect("can_access_peer 1→0");
    println!("  can 0→1: {can_0_to_1}");
    println!("  can 1→0: {can_1_to_0}");
    if !can_0_to_1 || !can_1_to_0 {
        eprintln!("WARN: peer access not bidirectional — Stage 3 host-stage fallback path applies.");
    }

    // ── Bidirectional enable (idempotent) ────────────────────────
    if can_0_to_1 {
        hip.set_device(0).expect("bind dev 0");
        hip.enable_peer_access(1).expect("enable 0→1");
        // Second call exercises the 704→Ok translation:
        hip.enable_peer_access(1).expect("enable 0→1 (idempotent)");
        println!("Peer 0→1 enabled (idempotent).");
    }
    if can_1_to_0 {
        hip.set_device(1).expect("bind dev 1");
        hip.enable_peer_access(0).expect("enable 1→0");
        hip.enable_peer_access(0).expect("enable 1→0 (idempotent)");
        println!("Peer 1→0 enabled (idempotent).");
    }

    // ── Allocate buffers on dev 0 and dev 1 ──────────────────────
    hip.set_device(0).expect("bind dev 0");
    let buf0 = hip.malloc(SIZE).expect("malloc dev 0");
    hip.set_device(1).expect("bind dev 1");
    let buf1 = hip.malloc(SIZE).expect("malloc dev 1");

    // Write pattern from host into dev_0 buffer.
    let pattern: Vec<u8> = (0..SIZE).map(|i| ((i * 31 + 7) % 256) as u8).collect();
    hip.set_device(0).expect("bind dev 0");
    hip.memcpy_htod(&buf0, &pattern).expect("H2D dev 0");

    // ── pointer_get_attributes verifies dev 0 buffer maps to dev 0 ──
    let attr0 = hip.pointer_get_attributes(&buf0).expect("attr buf0");
    println!(
        "\nbuf0 attributes: device={} mem_type={:?} is_managed={} alloc_flags=0x{:x}",
        attr0.device,
        MemoryType::from_raw(attr0.mem_type),
        attr0.is_managed,
        attr0.allocation_flags,
    );
    assert_eq!(attr0.device, 0, "buf0 should live on device 0");
    assert_eq!(
        MemoryType::from_raw(attr0.mem_type),
        Some(MemoryType::Device),
        "buf0 should be Device-type memory"
    );

    let attr1 = hip.pointer_get_attributes(&buf1).expect("attr buf1");
    assert_eq!(attr1.device, 1, "buf1 should live on device 1");

    // ── memcpy_peer: dev_0 → dev_1 ───────────────────────────────
    let t0 = std::time::Instant::now();
    hip.memcpy_peer(&buf1, 1, &buf0, 0, SIZE).expect("memcpy_peer");
    hip.device_synchronize().expect("device_synchronize");
    let elapsed_us = t0.elapsed().as_micros();
    let mb_per_s = (SIZE as f64 / 1e6) / (elapsed_us as f64 / 1e6);
    println!(
        "memcpy_peer 0→1: {} bytes in {} µs (~{:.1} MB/s)",
        SIZE, elapsed_us, mb_per_s
    );

    // ── Download from dev_1 and verify byte-equality ─────────────
    let mut readback = vec![0u8; SIZE];
    hip.set_device(1).expect("bind dev 1");
    hip.memcpy_dtoh(&mut readback, &buf1).expect("D2H dev 1");

    if readback != pattern {
        // Find first diff for a useful failure message
        let first_diff = pattern
            .iter()
            .zip(readback.iter())
            .position(|(a, b)| a != b);
        panic!(
            "data mismatch after peer copy: first diff at byte {first_diff:?} \
             (expected={:?}, got={:?})",
            first_diff.map(|i| pattern[i]),
            first_diff.map(|i| readback[i]),
        );
    }
    println!("memcpy_peer round-trip VERIFIED — all {SIZE} bytes match.");

    // ── Async variant exercise ───────────────────────────────────
    hip.set_device(0).expect("bind dev 0");
    let stream0 = hip.stream_create().expect("stream create");
    hip.memcpy_peer_async(&buf1, 1, &buf0, 0, SIZE, &stream0)
        .expect("memcpy_peer_async");
    hip.stream_synchronize(&stream0).expect("stream_synchronize");
    println!("memcpy_peer_async on dev_0 stream: ok");
    hip.stream_destroy(stream0).expect("stream_destroy");

    // ── Cleanup ──────────────────────────────────────────────────
    hip.set_device(0).expect("bind dev 0");
    hip.free(buf0).expect("free buf0");
    hip.set_device(1).expect("bind dev 1");
    hip.free(buf1).expect("free buf1");

    println!("\npeer_smoke: PASS");
}
