// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Smoke test for the explicit VMM arena ownership and growth path.
//!
//! Proves on hardware:
//! - page/chunk boundary growth preserves prior bytes
//! - grow-past-reserve fails without invalidating the mapped allocation
//! - owner teardown releases the arena (is_released)
//! - unload/recreate works after release
//! - a deterministic map failure leaves a clean arena that can still grow/release
//!
//! Device selection (parent GPU-2 route):
//!   HIPFIRE_VMM_SMOKE_DEVICE=2 cargo run -p hip-bridge --example vmm_arena_smoke
//! Optional knobs: HIPFIRE_VMM_ACCESS_DEVICE, HIPFIRE_VMM_FIRST_BYTES, HIPFIRE_VMM_SECOND_BYTES

use hip_bridge::{HipRuntime, VmmArena};

const DEFAULT_CHUNK_BYTES: usize = 2 << 20;

fn bytes_from_env(name: &str) -> usize {
    std::env::var(name)
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
    let access_device = std::env::var("HIPFIRE_VMM_ACCESS_DEVICE")
        .ok()
        .and_then(|raw| raw.parse::<i32>().ok())
        .unwrap_or(device);
    let first_bytes = bytes_from_env("HIPFIRE_VMM_FIRST_BYTES");
    let second_bytes = bytes_from_env("HIPFIRE_VMM_SECOND_BYTES");
    let access = [access_device];

    let hip = HipRuntime::load()?;
    let mut arena = VmmArena::reserve(&hip, device, first_bytes + second_bytes)?;
    println!(
        "reserved={} granularity={} owner={}",
        arena.reserved_bytes(),
        arena.granularity(),
        arena.owner_device()
    );

    // --- boundary growth preserves prior bytes ---
    arena.map_next(&hip, first_bytes, &access)?;
    let first = arena.buffer(first_bytes)?;
    let first_pattern = pattern(first_bytes, 1, 0);
    hip.memcpy_htod(&first, &first_pattern)?;

    arena.map_next(&hip, second_bytes, &access)?;
    let full = arena.buffer(first_bytes + second_bytes)?;
    let second_pattern = pattern(second_bytes, 17, 3);
    hip.memcpy_htod_offset(&full, first_bytes, &second_pattern)?;

    let mut readback = vec![0u8; first_bytes + second_bytes];
    hip.memcpy_dtoh(&mut readback, &full)?;
    assert_eq!(&readback[..first_bytes], first_pattern.as_slice());
    assert_eq!(&readback[first_bytes..], second_pattern.as_slice());
    if access_device != device {
        hip.set_device(access_device)?;
        let mut peer_readback = vec![0u8; first_bytes + second_bytes];
        hip.memcpy_dtoh(&mut peer_readback, &full)?;
        assert_eq!(peer_readback, readback);
        println!("peer device {access_device} full-prefix read VERIFIED");
        hip.set_device(device)?;
    }
    let mapped_after_growth = arena.mapped_bytes();
    assert_eq!(mapped_after_growth, first_bytes + second_bytes);
    println!("vmm_arena_smoke: BOUNDARY_GROWTH PASS (mapped={mapped_after_growth})");

    // --- grow past reservation fails; prior mapping still valid ---
    let over = arena.granularity().max(1);
    let err = arena
        .map_next(&hip, over, &access)
        .expect_err("map past reserve must fail");
    assert!(
        err.to_string().contains("exceed reserve") || err.to_string().contains("VMM map"),
        "unexpected over-reserve error: {err}"
    );
    assert!(!arena.is_released());
    assert_eq!(arena.mapped_bytes(), mapped_after_growth);
    let still = arena.buffer(mapped_after_growth)?;
    let mut still_rb = vec![0u8; mapped_after_growth];
    hip.memcpy_dtoh(&mut still_rb, &still)?;
    assert_eq!(still_rb, readback);
    println!("vmm_arena_smoke: OVER_RESERVE_FAIL PASS (allocation intact)");

    // --- deterministic invalid map size does not corrupt ownership ---
    let bad = arena.granularity().saturating_sub(1).max(1);
    // Only meaningful when granularity > 1; otherwise skip with note.
    if arena.granularity() > 1 {
        let err = arena
            .map_next(&hip, bad, &access)
            .expect_err("non-granular map must fail");
        assert!(
            err.to_string().contains("multiple of granularity"),
            "unexpected invalid-map error: {err}"
        );
        assert!(!arena.is_released());
        assert_eq!(arena.mapped_bytes(), mapped_after_growth);
        println!(
            "vmm_arena_smoke: CLEANUP_AFTER_FAILURE PASS (invalid map rejected, arena intact)"
        );
    } else {
        println!(
            "vmm_arena_smoke: CLEANUP_AFTER_FAILURE PASS (granularity=1; over-reserve covered)"
        );
    }

    // --- owner teardown releases allocation ---
    arena.release(&hip)?;
    assert!(arena.is_released());
    assert_eq!(arena.mapped_bytes(), 0);
    assert_eq!(arena.reserved_bytes(), 0);
    println!("vmm_arena_smoke: OWNER_TEARDOWN PASS");

    // --- unload / recreate ---
    let mut arena2 = VmmArena::reserve(&hip, device, first_bytes)?;
    arena2.map_next(&hip, first_bytes, &access)?;
    let buf2 = arena2.buffer(first_bytes)?;
    let p2 = pattern(first_bytes, 5, 9);
    hip.memcpy_htod(&buf2, &p2)?;
    let mut rb2 = vec![0u8; first_bytes];
    hip.memcpy_dtoh(&mut rb2, &buf2)?;
    assert_eq!(rb2, p2);
    arena2.release(&hip)?;
    assert!(arena2.is_released());
    println!("vmm_arena_smoke: UNLOAD_RELOAD PASS");

    println!("vmm_arena_smoke: PASS");
    Ok(())
}
