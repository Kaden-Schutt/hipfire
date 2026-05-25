// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bisection test: dispatch a no-op compute kernel.
//! If this hangs, the issue is fundamental dispatch setup.
//! If this passes, the issue is in kernarg/memory setup.

use redline::device::Device;
use redline::hsaco::HsacoModule;
use redline::queue::ComputeQueue;

fn main() {
    eprintln!("=== redline: noop kernel dispatch test ===\n");

    // Compile the simplest possible kernel
    let hip_src = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void noop_kernel() {
    // do absolutely nothing
}
"#;
    std::fs::write("/tmp/redline_noop.hip", hip_src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3",
               "-o", "/tmp/redline_noop.hsaco", "/tmp/redline_noop.hip"])
        .output().expect("hipcc");
    if !out.status.success() {
        eprintln!("hipcc failed: {}", String::from_utf8_lossy(&out.stderr));
        std::process::exit(1);
    }

    // Parse HSACO
    let module = HsacoModule::from_file("/tmp/redline_noop.hsaco").unwrap();
    let k = &module.kernels[0];
    eprintln!("kernel: {} vgprs={} sgprs={} lds={} kernarg={} priv={}",
        k.name, k.vgpr_count(), k.sgpr_count(), k.group_segment_size,
        k.kernarg_size, k.private_segment_size);
    eprintln!("pgm_rsrc1=0x{:08x} pgm_rsrc2=0x{:08x}", k.pgm_rsrc1, k.pgm_rsrc2);
    eprintln!("kd_offset=0x{:x} code_offset=0x{:x}", k.kd_offset, k.code_offset);

    // Open GPU
    let dev = Device::open(None).unwrap();
    let queue = ComputeQueue::new(&dev).unwrap();

    // Upload kernel ELF to VRAM
    let code_buf = dev.alloc_vram(module.elf.len() as u64).unwrap();
    dev.upload(&code_buf, &module.elf).unwrap();
    let code_va = code_buf.gpu_addr + k.code_offset;
    let kd_va = code_buf.gpu_addr + k.kd_offset;
    eprintln!("code_buf base=0x{:x}", code_buf.gpu_addr);
    eprintln!("code_va=0x{:x} (aligned to 256? {})", code_va, code_va & 0xFF == 0);
    eprintln!("kd_va=0x{:x}", kd_va);

    // Decode kernel_code_properties from the KD
    let kd_off = k.kd_offset as usize;
    let kcp = u16::from_le_bytes([module.elf[kd_off + 56], module.elf[kd_off + 57]]);
    eprintln!("kernel_code_properties=0x{:04x}", kcp);
    eprintln!("  ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER: {}", kcp & 1);
    eprintln!("  ENABLE_SGPR_DISPATCH_PTR: {}", (kcp >> 1) & 1);
    eprintln!("  ENABLE_SGPR_QUEUE_PTR: {}", (kcp >> 2) & 1);
    eprintln!("  ENABLE_SGPR_KERNARG_SEGMENT_PTR: {}", (kcp >> 3) & 1);
    eprintln!("  ENABLE_SGPR_DISPATCH_ID: {}", (kcp >> 4) & 1);
    eprintln!("  ENABLE_SGPR_FLAT_SCRATCH_INIT: {}", (kcp >> 5) & 1);
    eprintln!("  ENABLE_WAVEFRONT_SIZE32: {}", (kcp >> 8) & 1);

    // Count required user SGPRs
    let mut user_sgpr_count = 0u32;
    if kcp & (1 << 0) != 0 { user_sgpr_count += 4; } // private seg buf
    if kcp & (1 << 1) != 0 { user_sgpr_count += 2; } // dispatch ptr
    if kcp & (1 << 2) != 0 { user_sgpr_count += 2; } // queue ptr
    if kcp & (1 << 3) != 0 { user_sgpr_count += 2; } // kernarg ptr
    if kcp & (1 << 4) != 0 { user_sgpr_count += 2; } // dispatch id
    if kcp & (1 << 5) != 0 { user_sgpr_count += 2; } // flat scratch init
    if kcp & (1 << 6) != 0 { user_sgpr_count += 1; } // private seg size
    eprintln!("required user SGPRs: {}", user_sgpr_count);

    // Build PM4
    let mut pm4: Vec<u32> = Vec::new();
    let hdr = |opcode: u32, ndw: u32| -> u32 {
        (3u32 << 30) | ((ndw - 1) << 16) | (opcode << 8) | (1 << 1) // SHADER_TYPE=1 (compute)
    };

    // SET_SH_REG: COMPUTE_PGM_LO/HI
    pm4.push(hdr(0x76, 3));
    pm4.push(0x020C);
    pm4.push((code_va >> 8) as u32);
    pm4.push((code_va >> 40) as u32);

    // SET_SH_REG: COMPUTE_PGM_RSRC1/RSRC2
    pm4.push(hdr(0x76, 3));
    pm4.push(0x0212);
    pm4.push(k.pgm_rsrc1);
    pm4.push(k.pgm_rsrc2);

    // SET_SH_REG: COMPUTE_PGM_RSRC3 (GFX10 required)
    pm4.push(hdr(0x76, 2));
    pm4.push(0x0228);
    pm4.push(0);

    // SET_SH_REG: COMPUTE_TMPRING_SIZE = 0
    pm4.push(hdr(0x76, 2));
    pm4.push(0x0218);
    pm4.push(0);

    // SET_SH_REG: COMPUTE_NUM_THREAD_X/Y/Z
    pm4.push(hdr(0x76, 4));
    pm4.push(0x0207);
    pm4.push(1);  // 1 thread per group (noop kernel)
    pm4.push(1);
    pm4.push(1);

    // SET_SH_REG: COMPUTE_RESOURCE_LIMITS = 0
    pm4.push(hdr(0x76, 2));
    pm4.push(0x0215);
    pm4.push(0);

    // SET_SH_REG: COMPUTE_USER_DATA — only what the kernel needs
    if user_sgpr_count > 0 {
        pm4.push(hdr(0x76, 1 + user_sgpr_count));
        pm4.push(0x0240); // COMPUTE_USER_DATA_0
        for _ in 0..user_sgpr_count {
            pm4.push(0); // all zeros (noop kernel doesn't read any)
        }
    }

    // DISPATCH_DIRECT
    // HIP on RDNA defaults to wave32 — KD flag 0 means "use default", not "wave64"
    let di = (1u32 << 0) | (1 << 2) | (1 << 15); // CS_EN | FORCE_START_AT_000 | CS_W32_EN
    pm4.push(hdr(0x15, 4));
    pm4.push(1); // 1 group
    pm4.push(1);
    pm4.push(1);
    pm4.push(di);

    eprintln!("\nPM4: {} dwords ({} bytes)", pm4.len(), pm4.len() * 4);

    // Upload and submit
    let ib_buf = dev.alloc_vram(4096).unwrap();
    let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
    dev.upload(&ib_buf, &ib_bytes).unwrap();

    eprintln!("Submitting noop dispatch...");
    match queue.submit_and_wait(&dev, &ib_buf, pm4.len() as u32, &[&ib_buf, &code_buf]) {
        Ok(()) => {
            eprintln!("\n=== PASSED — noop kernel dispatched and returned! ===");
            eprintln!("Fundamental dispatch setup is correct.");
            eprintln!("If vector_add still hangs, the issue is in kernarg/memory setup.");
        }
        Err(e) => {
            eprintln!("\n=== FAILED: {e} ===");
            eprintln!("Fundamental dispatch setup has a bug.");
            eprintln!("Check: code address alignment, RSRC1/2 values, cache state.");
        }
    }

    queue.destroy(&dev);
}
