// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Redline PoC: GEMM (matrix multiply) via bare libdrm_amdgpu.
//! A real inference-relevant kernel dispatched without HIP/ROCm runtime.

use redline::device::Device;
use redline::hsaco::HsacoModule;
use redline::queue::ComputeQueue;

fn main() {
    eprintln!("=== redline PoC: GEMM f32 ===\n");

    // Compile
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3",
               "-o", "/tmp/redline_gemm.hsaco",
               "kernels/src/gemm_f32.hip"])
        .output().expect("hipcc");
    if !out.status.success() {
        eprintln!("hipcc failed:\n{}", String::from_utf8_lossy(&out.stderr));
        std::process::exit(1);
    }

    let module = HsacoModule::from_file("/tmp/redline_gemm.hsaco").unwrap();
    let k = &module.kernels[0];
    eprintln!("kernel: {} vgprs={} sgprs={} lds={} kernarg={}",
        k.name, k.vgpr_count(), k.sgpr_count(), k.group_segment_size, k.kernarg_size);
    eprintln!("pgm_rsrc1=0x{:08x} pgm_rsrc2=0x{:08x}", k.pgm_rsrc1, k.pgm_rsrc2);

    // Open GPU
    let dev = Device::open(None).unwrap();
    let queue = ComputeQueue::new(&dev).unwrap();

    // Upload ELF
    let code_buf = dev.alloc_vram(module.elf.len() as u64).unwrap();
    dev.upload(&code_buf, &module.elf).unwrap();
    let code_va = code_buf.gpu_addr + k.code_offset;

    // Test dimensions: Y = A × B^T (output M×N, inner dim K)
    // gemm_f32_batched: A[M,K], B[N,K], Y[M,N] — B is transposed
    let m = 4u32;
    let k_dim = 64u32; // K dimension (using 64 to exercise the wave reduction)
    let n = 4u32;

    // Generate test data
    let a_data: Vec<f32> = (0..m * k_dim).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..n * k_dim).map(|i| ((i % 5) as f32) * 0.1).collect();

    // CPU reference: Y[m][n] = sum_k(A[m][k] * B[n][k])
    let mut expected = vec![0.0f32; (m * n) as usize];
    for mi in 0..m as usize {
        for ni in 0..n as usize {
            let mut sum = 0.0f32;
            for ki in 0..k_dim as usize {
                sum += a_data[mi * k_dim as usize + ki] * b_data[ni * k_dim as usize + ki];
            }
            expected[mi * n as usize + ni] = sum;
        }
    }

    let a_bytes = as_bytes(&a_data);
    let b_bytes = as_bytes(&b_data);
    let y_size = (m * n * 4) as usize;

    let a_buf = dev.alloc_vram(a_bytes.len() as u64).unwrap();
    let b_buf = dev.alloc_vram(b_bytes.len() as u64).unwrap();
    let y_buf = dev.alloc_vram(y_size as u64).unwrap();
    dev.upload(&a_buf, a_bytes).unwrap();
    dev.upload(&b_buf, b_bytes).unwrap();
    dev.upload(&y_buf, &vec![0u8; y_size]).unwrap();

    // Kernarg: [A_ptr:u64, B_ptr:u64, Y_ptr:u64, M:i32, K:i32, N:i32]
    let mut ka = vec![0u8; k.kernarg_size as usize];
    ka[0..8].copy_from_slice(&a_buf.gpu_addr.to_le_bytes());
    ka[8..16].copy_from_slice(&b_buf.gpu_addr.to_le_bytes());
    ka[16..24].copy_from_slice(&y_buf.gpu_addr.to_le_bytes());
    ka[24..28].copy_from_slice(&m.to_le_bytes());
    ka[28..32].copy_from_slice(&k_dim.to_le_bytes());
    ka[32..36].copy_from_slice(&n.to_le_bytes());
    let ka_buf = dev.alloc_vram(4096).unwrap();
    dev.upload(&ka_buf, &ka).unwrap();

    // Decode kernel_code_properties
    let kd_off = k.kd_offset as usize;
    let kcp = u16::from_le_bytes([module.elf[kd_off + 56], module.elf[kd_off + 57]]);

    // Count user SGPRs
    let mut user_sgpr_count = 0u32;
    let mut user_sgpr_idx = 0u32; // where to place kernarg ptr
    if kcp & (1 << 0) != 0 { user_sgpr_count += 4; } // private seg buf
    if kcp & (1 << 1) != 0 { user_sgpr_count += 2; } // dispatch ptr
    if kcp & (1 << 2) != 0 { user_sgpr_count += 2; } // queue ptr
    user_sgpr_idx = user_sgpr_count; // kernarg ptr starts after the above
    if kcp & (1 << 3) != 0 { user_sgpr_count += 2; } // kernarg ptr
    if kcp & (1 << 4) != 0 { user_sgpr_count += 2; } // dispatch id
    if kcp & (1 << 5) != 0 { user_sgpr_count += 2; } // flat scratch init
    if kcp & (1 << 6) != 0 { user_sgpr_count += 1; } // private seg size

    eprintln!("kernel_code_properties=0x{:04x} user_sgprs={} kernarg_at_sgpr={}",
        kcp, user_sgpr_count, user_sgpr_idx);

    // Build PM4
    let mut pm4: Vec<u32> = Vec::new();
    let hdr = |opcode: u32, ndw: u32| -> u32 {
        (3u32 << 30) | ((ndw - 1) << 16) | (opcode << 8) | (1 << 1)
    };

    // PGM_LO/HI
    pm4.push(hdr(0x76, 3));
    pm4.push(0x020C);
    pm4.push((code_va >> 8) as u32);
    pm4.push((code_va >> 40) as u32);

    // PGM_RSRC1/2
    pm4.push(hdr(0x76, 3));
    pm4.push(0x0212);
    pm4.push(k.pgm_rsrc1);
    pm4.push(k.pgm_rsrc2);

    // PGM_RSRC3
    pm4.push(hdr(0x76, 2));
    pm4.push(0x0228);
    pm4.push(0);

    // TMPRING_SIZE
    pm4.push(hdr(0x76, 2));
    pm4.push(0x0218);
    pm4.push(0);

    // NUM_THREAD_X/Y/Z — __launch_bounds__(32)
    pm4.push(hdr(0x76, 4));
    pm4.push(0x0207);
    pm4.push(32); // block X = 32 (one wavefront)
    pm4.push(1);
    pm4.push(1);

    // RESOURCE_LIMITS
    pm4.push(hdr(0x76, 2));
    pm4.push(0x0215);
    pm4.push(0);

    // USER_DATA — fill all required SGPRs
    if user_sgpr_count > 0 {
        pm4.push(hdr(0x76, 1 + user_sgpr_count));
        pm4.push(0x0240); // COMPUTE_USER_DATA_0
        for i in 0..user_sgpr_count {
            if i == user_sgpr_idx {
                pm4.push(ka_buf.gpu_addr as u32); // kernarg lo
            } else if i == user_sgpr_idx + 1 {
                pm4.push((ka_buf.gpu_addr >> 32) as u32); // kernarg hi
            } else {
                pm4.push(0); // zeros for private seg buf etc.
            }
        }
    }

    // DISPATCH_DIRECT — grid=[M, N, 1]
    // HIP on RDNA defaults to wave32 — always dispatch wave32 for HIP kernels
    let di = (1u32 << 0) | (1 << 15); // CS_EN | CS_W32_EN
    pm4.push(hdr(0x15, 4));
    pm4.push(m); // groups X = M
    pm4.push(n); // groups Y = N
    pm4.push(1); // groups Z = 1
    pm4.push(di);

    eprintln!("PM4: {} dwords, grid=[{},{},1] block=[32,1,1]", pm4.len(), m, n);

    // Submit
    let ib_buf = dev.alloc_vram(4096).unwrap();
    let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
    dev.upload(&ib_buf, &ib_bytes).unwrap();

    eprintln!("Dispatching GEMM {}x{}x{} ...", m, k_dim, n);
    match queue.submit_and_wait(&dev, &ib_buf, pm4.len() as u32,
        &[&ib_buf, &code_buf, &a_buf, &b_buf, &y_buf, &ka_buf])
    {
        Ok(()) => eprintln!("GPU returned"),
        Err(e) => { eprintln!("FAILED: {e}"); std::process::exit(1); }
    }

    // Verify
    let mut y_raw = vec![0u8; y_size];
    dev.download(&y_buf, &mut y_raw).unwrap();
    let y: &[f32] = unsafe { std::slice::from_raw_parts(y_raw.as_ptr() as *const f32, (m * n) as usize) };

    let mut bad = 0;
    for i in 0..(m * n) as usize {
        let err = (y[i] - expected[i]).abs();
        let tol = expected[i].abs() * 0.01 + 0.001; // 1% relative + small absolute
        if err > tol {
            if bad < 5 { eprintln!("  [{i}] got={:.6} exp={:.6} err={:.6}", y[i], expected[i], err); }
            bad += 1;
        }
    }

    if bad == 0 {
        eprintln!("\n╔════════════════════════════════════════════════════════╗");
        eprintln!("║  REDLINE: MATMUL KERNEL EXECUTED VIA BARE DRM         ║");
        eprintln!("║  gemm_f32: {}x{}x{} = {} elements correct{} ║",
            m, k_dim, n, m*n, " ".repeat(16 - format!("{}x{}x{}", m, k_dim, n).len()));
        eprintln!("║  No HIP runtime. No Vulkan. Pure libdrm_amdgpu.       ║");
        eprintln!("╚════════════════════════════════════════════════════════╝");
    } else {
        eprintln!("{bad}/{} wrong", m*n);
        eprintln!("Y = {:?}", y);
        eprintln!("expected = {:?}", &expected);
    }

    queue.destroy(&dev);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
