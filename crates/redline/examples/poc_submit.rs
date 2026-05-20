// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Redline PoC: submit a PM4 NOP packet to the GPU compute queue.
//! This proves the full submission path works: context → IB → submit → fence.
//!
//! Usage: cargo run -p redline --example poc_submit

fn main() {
    eprintln!("=== redline PoC: compute queue submission ===\n");

    // Step 1: Open device
    let dev = redline::device::Device::open(None).expect("failed to open GPU");
    eprintln!("GPU: {} ({:.1} GB VRAM)\n", dev.info.gfx_arch, dev.info.vram_total_bytes as f64 / 1e9);

    // Step 2: Create compute queue
    let queue = redline::queue::ComputeQueue::new(&dev).expect("failed to create compute queue");

    // Step 3: Build a trivial PM4 buffer — just a NOP packet
    // PKT3_NOP (opcode 0x10): the GPU reads and discards it
    let nop_packet: [u32; 2] = [
        (3 << 30) | (0x10 << 8) | 0,  // PKT3 header: NOP, 1 dword body
        0xDEADBEEF,                     // body (ignored)
    ];

    // Upload PM4 buffer to VRAM
    let ib_buf = dev.alloc_vram(4096).expect("failed to alloc IB buffer");
    let ib_bytes: Vec<u8> = nop_packet.iter().flat_map(|d| d.to_le_bytes()).collect();
    dev.upload(&ib_buf, &ib_bytes).expect("failed to upload IB");

    eprintln!("Submitting NOP packet to compute queue...");

    // Step 4: Submit and wait
    match queue.submit_and_wait(&dev, &ib_buf, nop_packet.len() as u32, &[&ib_buf]) {
        Ok(()) => {
            eprintln!("  OK — GPU executed NOP and returned!\n");
            eprintln!("=== PASSED — full submission path works ===");
            eprintln!("Context → IB upload → cs_submit ��� fence wait → complete.");
            eprintln!("Next: dispatch a real compute kernel.");
        }
        Err(e) => {
            eprintln!("  FAILED: {e}");
            eprintln!("\nThis might fail if:");
            eprintln!("  - CsRequest struct layout doesn't match libdrm version");
            eprintln!("  - BO list handle format is wrong");
            eprintln!("  - Compute ring is not available");
            std::process::exit(1);
        }
    }

    // Cleanup
    dev.free_buffer(ib_buf).ok();
    queue.destroy(&dev);
}
