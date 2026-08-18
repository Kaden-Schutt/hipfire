// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Smoke test for page-locked host staging (`hipHostMalloc`/`hipHostFree`).
//!
//! An FFI signature mistake compiles cleanly and only shows up as a runtime
//! crash or silent corruption, so this asserts a full round-trip through a
//! pinned buffer rather than merely checking that the symbols resolved.
//!
//! Pinned staging is the prerequisite for genuinely async H2D on a DISCRETE
//! GPU: a pageable source is staged through a driver bounce buffer, which both
//! halves bandwidth and serialises against compute. On an APU (gfx1151) the GPU
//! allocates from system RAM, so this path is expected to work but buy nothing
//! — correctness here, benefit only on a dGPU.

use std::ffi::c_void;

fn main() {
    let hip = hip_bridge::HipRuntime::load().expect("failed to load HIP runtime");
    hip.set_device(0).expect("failed to set device");

    if !hip.has_pinned_host_alloc() {
        println!(
            "pinned_smoke: hipHostMalloc/hipHostFree NOT available — \
                  transport will fall back to pageable staging"
        );
        println!("pinned_smoke: SKIP (not a failure)");
        return;
    }
    println!("pinned_smoke: pinned host alloc available");

    const N: usize = 8 << 20; // 8 MiB — larger than one expert role-blob (~3.4 MB)

    let host = unsafe { hip.host_malloc(N) }.expect("hipHostMalloc failed");
    println!("pinned_smoke: allocated {N} B of pinned host memory at {host:?}");

    // Fill the pinned buffer with a non-trivial pattern.
    let src: Vec<u8> = (0..N).map(|i| (i.wrapping_mul(31) % 251) as u8).collect();
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), host as *mut u8, N) };

    // Round-trip: pinned host -> device -> pageable host.
    let dev = hip.malloc(N).expect("hipMalloc failed");
    let pinned_slice = unsafe { std::slice::from_raw_parts(host as *const u8, N) };
    hip.memcpy_htod(&dev, pinned_slice)
        .expect("H2D from pinned failed");

    let mut back = vec![0u8; N];
    hip.memcpy_dtoh(&mut back, &dev).expect("D2H failed");

    assert_eq!(src, back, "pinned round-trip corrupted data");
    println!("pinned_smoke: {N} B round-trip VERIFIED byte-identical");

    hip.free(dev).expect("hipFree failed");
    unsafe { hip.host_free(host as *mut c_void) }.expect("hipHostFree failed");
    println!("pinned_smoke: freed cleanly");

    println!("\npinned_smoke: PASS");
}
