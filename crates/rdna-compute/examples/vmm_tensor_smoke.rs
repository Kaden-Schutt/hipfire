// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Verify that a VMM-backed GpuTensor grows and bypasses the hipFree pool path.
//!
//! Proves on hardware:
//! - page/chunk boundary growth preserves prior bytes
//! - grow-past-reserve fails without invalidating the allocation
//! - owner teardown releases the tracked VMM allocation
//! - unload/recreate (free then alloc) works
//! - a deterministic map failure leaves no leaked tracked allocation and is
//!   followed by a successful allocation
//!
//! Device selection (parent GPU-2 route):
//!   HIPFIRE_VMM_SMOKE_DEVICE=2 cargo run -p rdna-compute --example vmm_tensor_smoke
//! Optional knobs: HIPFIRE_VMM_CHUNK_BYTES (default 2 MiB)

use rdna_compute::{DType, Gpu};

const DEFAULT_CHUNK_BYTES: usize = 2 << 20;

fn chunk_bytes() -> usize {
    std::env::var("HIPFIRE_VMM_CHUNK_BYTES")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CHUNK_BYTES)
}

fn pattern(len: usize, mul: usize, add: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * mul + add) % 251) as u8).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = std::env::var("HIPFIRE_VMM_SMOKE_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(0);
    let chunk = chunk_bytes();
    let access = [device];

    let mut gpu = Gpu::init_with_device(device)?;
    assert_eq!(gpu.device_id, device);
    assert_eq!(gpu.vmm_allocation_count(), 0);

    // --- alloc + boundary growth preserves prior bytes ---
    let mut tensor = unsafe { gpu.alloc_vmm_tensor(&[chunk * 2], DType::Raw, chunk, &access)? };
    assert_eq!(gpu.vmm_allocation_count(), 1);
    assert_eq!(gpu.vmm_mapped_bytes(&tensor), Some(chunk));
    assert_eq!(tensor.buf.size(), chunk);
    assert!(tensor.buf.is_vmm_owner());

    let first = pattern(chunk, 1, 0);
    gpu.hip.memcpy_htod(&tensor.buf, &first)?;

    let mapped = gpu.grow_vmm_tensor(&mut tensor, chunk, &access)?;
    assert_eq!(mapped, chunk * 2);
    assert_eq!(gpu.vmm_mapped_bytes(&tensor), Some(chunk * 2));
    assert_eq!(tensor.buf.size(), chunk * 2);
    let second = pattern(chunk, 17, 3);
    gpu.hip.memcpy_htod_offset(&tensor.buf, chunk, &second)?;

    let mut readback = vec![0u8; chunk * 2];
    gpu.hip.memcpy_dtoh(&mut readback, &tensor.buf)?;
    assert_eq!(&readback[..chunk], first.as_slice());
    assert_eq!(&readback[chunk..], second.as_slice());
    println!("vmm_tensor_smoke: BOUNDARY_GROWTH PASS (mapped={mapped})");

    // --- grow past reservation fails; prior mapping + tracking intact ---
    let gran = gpu
        .vmm_granularity(&tensor)
        .expect("registered VMM tensor must expose granularity");
    let over = gran.max(1);
    let err = gpu
        .grow_vmm_tensor(&mut tensor, over, &access)
        .expect_err("grow past reserve must fail");
    assert!(
        err.to_string().contains("exceed reserve") || err.to_string().contains("VMM map"),
        "unexpected over-reserve error: {err}"
    );
    assert_eq!(gpu.vmm_allocation_count(), 1);
    assert_eq!(gpu.vmm_mapped_bytes(&tensor), Some(chunk * 2));
    assert!(tensor.buf.is_vmm_owner());
    let mut still = vec![0u8; chunk * 2];
    gpu.hip.memcpy_dtoh(&mut still, &tensor.buf)?;
    assert_eq!(still, readback);
    println!("vmm_tensor_smoke: OVER_RESERVE_FAIL PASS (allocation intact)");

    // --- borrowed alias must not free the owner ---
    let alias = tensor.shallow_clone();
    assert!(gpu.free_tensor(alias).is_err());
    assert_eq!(gpu.vmm_allocation_count(), 1);

    // --- owner teardown releases tracked allocation ---
    gpu.free_tensor(tensor)?;
    assert_eq!(gpu.vmm_allocation_count(), 0);
    println!("vmm_tensor_smoke: OWNER_TEARDOWN PASS");

    // --- unload / recreate ---
    let mut tensor2 = unsafe { gpu.alloc_vmm_tensor(&[chunk], DType::Raw, chunk, &access)? };
    assert_eq!(gpu.vmm_allocation_count(), 1);
    let p2 = pattern(chunk, 5, 9);
    gpu.hip.memcpy_htod(&tensor2.buf, &p2)?;
    let mut rb2 = vec![0u8; chunk];
    gpu.hip.memcpy_dtoh(&mut rb2, &tensor2.buf)?;
    assert_eq!(rb2, p2);
    gpu.free_tensor(tensor2)?;
    assert_eq!(gpu.vmm_allocation_count(), 0);
    println!("vmm_tensor_smoke: UNLOAD_RELOAD PASS");

    // --- deterministic allocation/map failure: no leaked tracking; next alloc ok ---
    // Non-granular initial map size is rejected before the arena is registered.
    // Fall back to mapping more than the reserved logical size when granularity == 1.
    let fail_err = if gran > 1 {
        let bad_initial = gran.saturating_sub(1).max(1);
        match unsafe { gpu.alloc_vmm_tensor(&[chunk], DType::Raw, bad_initial, &access) } {
            Ok(_) => panic!("non-granular initial map must fail"),
            Err(err) => err,
        }
    } else {
        // Reserve `chunk` but ask to map `chunk + gran` up front.
        match unsafe { gpu.alloc_vmm_tensor(&[chunk], DType::Raw, chunk + gran, &access) } {
            Ok(_) => panic!("initial map past reserve must fail"),
            Err(err) => err,
        }
    };
    assert!(
        fail_err.to_string().contains("multiple of granularity")
            || fail_err.to_string().contains("exceed reserve")
            || fail_err.to_string().contains("VMM map"),
        "unexpected deterministic failure: {fail_err}"
    );
    // Successful cleanup path must not leave a tracked/orphan arena behind.
    assert_eq!(
        gpu.vmm_allocation_count(),
        0,
        "failed alloc must not leak a tracked VMM arena"
    );
    // Follow with a successful allocation.
    let mut tensor3 = unsafe { gpu.alloc_vmm_tensor(&[chunk], DType::Raw, chunk, &access)? };
    assert_eq!(gpu.vmm_allocation_count(), 1);
    assert_eq!(gpu.vmm_mapped_bytes(&tensor3), Some(chunk));
    let p3 = pattern(chunk, 11, 2);
    gpu.hip.memcpy_htod(&tensor3.buf, &p3)?;
    let mut rb3 = vec![0u8; chunk];
    gpu.hip.memcpy_dtoh(&mut rb3, &tensor3.buf)?;
    assert_eq!(rb3, p3);
    gpu.free_tensor(tensor3)?;
    assert_eq!(gpu.vmm_allocation_count(), 0);
    println!("vmm_tensor_smoke: CLEANUP_AFTER_FAILURE PASS");

    println!("vmm_tensor_smoke: PASS");
    Ok(())
}
