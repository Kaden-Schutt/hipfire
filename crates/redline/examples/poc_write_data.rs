#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![allow(clippy::vec_init_then_push)]

//! Test: can PM4 WRITE_DATA actually write to VRAM through our submit path?
//! If this fails, the issue is fundamental to our PM4/submit infrastructure.
//! If it passes, the issue is specifically in the compute dispatch setup.

use redline::device::Device;
use redline::queue::ComputeQueue;

fn main() {
    eprintln!("=== redline: PM4 WRITE_DATA test ===\n");

    let dev = Device::open(None).unwrap();
    let queue = ComputeQueue::new(&dev).unwrap();

    // Allocate a target buffer, fill with 0xDEAD
    let target = dev.alloc_vram(4096).unwrap();
    dev.upload(&target, &vec![0xADu8; 4096]).unwrap();

    // Build PM4: WRITE_DATA packet writes 0xCAFEBABE to target GPU VA
    // PKT3_WRITE_DATA = 0x37
    // Body: [control, dst_addr_lo, dst_addr_hi, data...]
    // Control: DST_SEL=5 (memory, async), WR_CONFIRM=1
    let control = (5u32 << 8) | (1 << 20); // dst_sel=memory(async), wr_confirm
    let hdr = |opcode: u32, ndw: u32| -> u32 {
        (3u32 << 30) | ((ndw - 1) << 16) | (opcode << 8) | (1 << 1)
    };

    let mut pm4: Vec<u32> = Vec::new();
    pm4.push(hdr(0x37, 4)); // WRITE_DATA, 4 body dwords
    pm4.push(control);
    pm4.push(target.gpu_addr as u32);
    pm4.push((target.gpu_addr >> 32) as u32);
    pm4.push(0xCAFEBABE);

    let ib = dev.alloc_vram(4096).unwrap();
    let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
    dev.upload(&ib, &ib_bytes).unwrap();

    eprintln!(
        "Submitting WRITE_DATA(0xCAFEBABE → 0x{:x})...",
        target.gpu_addr
    );
    queue
        .submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &target])
        .unwrap();

    // Read back
    let mut readback = vec![0u8; 16];
    dev.download(&target, &mut readback).unwrap();
    let val = u32::from_le_bytes([readback[0], readback[1], readback[2], readback[3]]);
    eprintln!("Read back: 0x{:08x}", val);

    if val == 0xCAFEBABE {
        eprintln!("PASSED — PM4 WRITE_DATA works! Issue is in dispatch setup.");
    } else if val == 0xADADADAD {
        eprintln!("FAILED — buffer unchanged. PM4 packets not executing.");
    } else {
        eprintln!(
            "UNEXPECTED — got 0x{:08x}, neither 0xCAFEBABE nor 0xADADADAD",
            val
        );
    }

    queue.destroy(&dev);
}
