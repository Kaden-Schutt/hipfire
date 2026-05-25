// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Smoke test: load HIP runtime, detect GPU, malloc/memcpy round-trip.

fn main() {
    println!("Loading HIP runtime via dlopen...");
    let hip = hip_bridge::HipRuntime::load().expect("failed to load HIP runtime");

    let count = hip.device_count().expect("failed to get device count");
    println!("Devices found: {count}");

    hip.set_device(0).expect("failed to set device");
    println!("Set device 0");

    // Allocate 4KB on GPU
    let size = 4096;
    let buf = hip.malloc(size).expect("failed to malloc");
    println!("Allocated {} bytes on GPU at {:?}", buf.size(), buf.as_ptr());

    // Write test pattern to GPU
    let src: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    hip.memcpy_htod(&buf, &src).expect("H2D copy failed");
    println!("H2D copy done");

    // Read back
    let mut dst = vec![0u8; size];
    hip.memcpy_dtoh(&mut dst, &buf).expect("D2H copy failed");
    println!("D2H copy done");

    // Verify
    assert_eq!(src, dst, "data mismatch!");
    println!("Memory round-trip VERIFIED - all {size} bytes match");

    // Cleanup
    hip.free(buf).expect("free failed");
    println!("GPU memory freed");

    println!("\nhip-bridge smoke test: PASS");
}
