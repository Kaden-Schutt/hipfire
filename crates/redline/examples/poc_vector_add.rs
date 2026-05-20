// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Redline PoC: vector_add compute kernel via bare libdrm_amdgpu.
//! Following ROCT Dispatch.cpp BuildIb() pattern exactly.

use redline::device::Device;
use redline::hsaco::HsacoModule;
use redline::queue::ComputeQueue;

fn main() {
    eprintln!("=== redline PoC: vector_add ===\n");

    // Compile
    let hip_src = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;
    std::fs::write("/tmp/redline_va.hip", hip_src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3", "-o", "/tmp/redline_va.hsaco", "/tmp/redline_va.hip"])
        .output().expect("hipcc");
    assert!(out.status.success(), "hipcc failed");

    // Parse
    let module = HsacoModule::from_file("/tmp/redline_va.hsaco").unwrap();
    let k = &module.kernels[0];
    eprintln!("kernel: {} vgprs={} sgprs={} lds={} kernarg={}",
        k.name, k.vgpr_count(), k.sgpr_count(), k.group_segment_size, k.kernarg_size);

    // Open GPU
    let dev = Device::open(None).unwrap();

    // Upload kernel ELF to VRAM
    let code_buf = dev.alloc_vram(module.elf.len() as u64).unwrap();
    dev.upload(&code_buf, &module.elf).unwrap();
    let code_va = code_buf.gpu_addr + k.code_offset;
    let kd_va = code_buf.gpu_addr + k.kd_offset;
    eprintln!("code_va=0x{:x} kd_va=0x{:x}", code_va, kd_va);

    // Prepare data
    let n = 256u32;
    let nbytes = (n as usize) * 4;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
    let expected: Vec<f32> = (0..n).map(|i| (i as f32) * 3.0).collect();

    let a_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let b_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let c_buf = dev.alloc_vram(nbytes as u64).unwrap();
    dev.upload(&a_buf, as_bytes(&a_data)).unwrap();
    dev.upload(&b_buf, as_bytes(&b_data)).unwrap();
    dev.upload(&c_buf, &vec![0u8; nbytes]).unwrap();

    // Kernarg: [a_ptr:u64, b_ptr:u64, c_ptr:u64, n:u32, padding...]
    let mut ka = vec![0u8; 256];
    ka[0..8].copy_from_slice(&a_buf.gpu_addr.to_le_bytes());
    ka[8..16].copy_from_slice(&b_buf.gpu_addr.to_le_bytes());
    ka[16..24].copy_from_slice(&c_buf.gpu_addr.to_le_bytes());
    ka[24..28].copy_from_slice(&n.to_le_bytes());
    let ka_buf = dev.alloc_vram(4096).unwrap();
    dev.upload(&ka_buf, &ka).unwrap();

    // Build PM4 — following ROCT Dispatch.cpp exactly
    let mut pm4: Vec<u32> = Vec::new();

    // PM4 header helper: type3, compute shader type
    let hdr = |opcode: u32, ndw: u32| -> u32 {
        (3u32 << 30) | ((ndw - 1) << 16) | (opcode << 8) | (1 << 1)
    };

    // 1. SET_SH_REG: COMPUTE_PGM_LO/HI (code entry addr >> 8)
    pm4.push(hdr(0x76, 3)); // SET_SH_REG, 3 body dwords
    pm4.push(0x020C);       // offset: COMPUTE_PGM_LO
    pm4.push((code_va >> 8) as u32);
    pm4.push((code_va >> 40) as u32);

    // 2. SET_SH_REG: COMPUTE_PGM_RSRC1 + RSRC2
    pm4.push(hdr(0x76, 3));
    pm4.push(0x0212);       // offset: COMPUTE_PGM_RSRC1
    pm4.push(k.pgm_rsrc1);
    pm4.push(k.pgm_rsrc2);

    // 2b. SET_SH_REG: COMPUTE_PGM_RSRC3 (GFX10 requires this)
    pm4.push(hdr(0x76, 2));
    pm4.push(0x0228);       // offset: COMPUTE_PGM_RSRC3
    pm4.push(0);             // SHARED_VGPR_CNT = 0

    // 3. SET_SH_REG: COMPUTE_TMPRING_SIZE = 0 (no scratch)
    pm4.push(hdr(0x76, 2));
    pm4.push(0x0218);
    pm4.push(0);

    // 4. SET_SH_REG: COMPUTE_NUM_THREAD_X/Y/Z
    pm4.push(hdr(0x76, 4));
    pm4.push(0x0207);       // offset: COMPUTE_NUM_THREAD_X
    pm4.push(256);           // threads per group X
    pm4.push(1);             // Y
    pm4.push(1);             // Z

    // 5. SET_SH_REG: COMPUTE_RESOURCE_LIMITS = 0
    pm4.push(hdr(0x76, 2));
    pm4.push(0x0215);
    pm4.push(0);

    // 6. SET_SH_REG: COMPUTE_USER_DATA_0..5
    // Layout from kernel_code_properties 0x0409:
    //   s[0:3] = private segment buffer (4 SGPRs = 0)
    //   s[4:5] = kernarg pointer
    pm4.push(hdr(0x76, 7)); // offset + 6 values
    pm4.push(0x0240);       // COMPUTE_USER_DATA_0
    pm4.push(0); pm4.push(0); pm4.push(0); pm4.push(0); // private seg buf (unused)
    pm4.push(ka_buf.gpu_addr as u32);                     // kernarg lo
    pm4.push((ka_buf.gpu_addr >> 32) as u32);              // kernarg hi

    // 7. DISPATCH_DIRECT
    let groups = (n + 255) / 256;
    // Wave32 (gfx1010 HIP default) — CS_W32_EN at bit 15
    let di = (1u32 << 0) | (1 << 2) | (1 << 3) | (1 << 15); // CS_EN | FORCE_000 | ORDER | W32
    pm4.push(hdr(0x15, 4)); // DISPATCH_DIRECT, 4 body dwords
    pm4.push(groups);
    pm4.push(1);
    pm4.push(1);
    pm4.push(di);

    eprintln!("PM4: {} dwords", pm4.len());

    // Upload IB and submit
    let ib_buf = dev.alloc_vram(4096).unwrap();
    let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
    dev.upload(&ib_buf, &ib_bytes).unwrap();

    let queue = ComputeQueue::new(&dev).unwrap();
    match queue.submit_and_wait(&dev, &ib_buf, pm4.len() as u32,
        &[&ib_buf, &code_buf, &a_buf, &b_buf, &c_buf, &ka_buf])
    {
        Ok(()) => eprintln!("GPU returned"),
        Err(e) => { eprintln!("FAILED: {e}"); std::process::exit(1); }
    }

    // Verify
    let mut c_raw = vec![0u8; nbytes];
    dev.download(&c_buf, &mut c_raw).unwrap();
    let c: &[f32] = unsafe { std::slice::from_raw_parts(c_raw.as_ptr() as *const f32, n as usize) };

    let mut bad = 0;
    for i in 0..n as usize {
        if (c[i] - expected[i]).abs() > 0.001 {
            if bad < 5 { eprintln!("  [{i}] got={} exp={}", c[i], expected[i]); }
            bad += 1;
        }
    }

    if bad == 0 {
        eprintln!("\n╔══════════════════════════════════════════════════╗");
        eprintln!("║  REDLINE: COMPUTE KERNEL EXECUTED VIA BARE DRM  ║");
        eprintln!("║  vector_add: {n} elements correct                ║");
        eprintln!("║  No HIP. No Vulkan. No ROCm. Pure libdrm.       ║");
        eprintln!("╚══════════════════════════════════════════════════╝");
    } else {
        eprintln!("{bad}/{n} wrong. c[0..8]={:?}", &c[..8]);
    }

    queue.destroy(&dev);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
