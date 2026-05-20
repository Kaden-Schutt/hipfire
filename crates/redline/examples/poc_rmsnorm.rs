// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Redline: dispatch hipfire's RMSNorm kernel (uses dynamic shared memory).

use redline::device::Device;
use redline::dispatch::{DispatchQueue, KernargBuilder, Kernel, CommandBuffer};

fn main() {
    eprintln!("=== redline: hipfire RMSNorm kernel ===\n");

    let dev = Device::open(None).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();

    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3",
               "-o", "/tmp/redline_rmsnorm.hsaco", "kernels/src/rmsnorm.hip"])
        .output().expect("hipcc");
    assert!(out.status.success(), "hipcc: {}", String::from_utf8_lossy(&out.stderr));

    let module = dev.load_module_file("/tmp/redline_rmsnorm.hsaco").unwrap();
    let kernel = Kernel::find(&module, "rmsnorm_f32").expect("rmsnorm_f32 not found");
    eprintln!("kernel: {} (kernarg={}, lds={})",
        kernel.name, kernel.kernarg_size, kernel.group_segment_size);

    // Test: batch=2, dim=128, 256 threads per block
    let batch = 2u32;
    let dim = 128u32;
    let block_size = 256u32;
    let eps = 1e-5f32;

    let x_data: Vec<f32> = (0..batch * dim).map(|i| ((i as f32) - 128.0) * 0.01).collect();
    let w_data: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.001).collect();

    // CPU reference
    let mut expected = vec![0.0f32; (batch * dim) as usize];
    for b in 0..batch as usize {
        let row = &x_data[b * dim as usize..(b + 1) * dim as usize];
        let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps;
        let rms = 1.0 / ss.sqrt();
        for i in 0..dim as usize {
            expected[b * dim as usize + i] = row[i] * w_data[i] * rms;
        }
    }

    let x_buf = dev.alloc_vram((batch * dim * 4) as u64).unwrap();
    let w_buf = dev.alloc_vram((dim * 4) as u64).unwrap();
    let out_buf = dev.alloc_vram((batch * dim * 4) as u64).unwrap();
    dev.upload(&x_buf, as_bytes(&x_data)).unwrap();
    dev.upload(&w_buf, as_bytes(&w_data)).unwrap();
    dev.upload(&out_buf, &vec![0u8; (batch * dim * 4) as usize]).unwrap();

    // Kernarg: [x_ptr, weight_ptr, out_ptr, n, eps]
    let mut ka = KernargBuilder::new(32);
    ka.write_ptr(0, x_buf.gpu_addr)
      .write_ptr(8, w_buf.gpu_addr)
      .write_ptr(16, out_buf.gpu_addr)
      .write_u32(24, dim)
      .write_f32(28, eps);

    // Dynamic shared memory: block_size * sizeof(float) = 256 * 4 = 1024 bytes
    // Need to set LDS_SIZE in COMPUTE_PGM_RSRC2 for the dynamic portion.
    // LDS_SIZE field = number of 128-dword (512-byte) blocks.
    // 1024 bytes = 2 blocks of 512 bytes
    let lds_bytes = block_size * 4; // 1024 bytes for float[256]

    // Build PM4 manually since we need to override LDS_SIZE in RSRC2
    let ka_data = ka.as_bytes();
    let ka_size = kernel.kernarg_size as usize;
    let mut ka_full = vec![0u8; std::cmp::max(ka_size, ka_data.len() + 64)];
    ka_full[..ka_data.len()].copy_from_slice(ka_data);
    // Hidden args
    let hidden_off = (ka_data.len() + 7) & !7;
    if ka_size > hidden_off {
        ka_full[hidden_off..hidden_off+4].copy_from_slice(&batch.to_le_bytes());
        ka_full[hidden_off+4..hidden_off+8].copy_from_slice(&1u32.to_le_bytes());
        ka_full[hidden_off+8..hidden_off+12].copy_from_slice(&1u32.to_le_bytes());
        ka_full[hidden_off+12..hidden_off+14].copy_from_slice(&(block_size as u16).to_le_bytes());
        ka_full[hidden_off+14..hidden_off+16].copy_from_slice(&1u16.to_le_bytes());
        ka_full[hidden_off+16..hidden_off+18].copy_from_slice(&1u16.to_le_bytes());
    }
    dev.upload(dq.kernarg_buf(), &ka_full).unwrap();

    // Build command buffer with modified RSRC2 for LDS
    let mut cb = CommandBuffer::new();
    // Override LDS size: set group_segment_size for the dynamic shared memory
    cb.dispatch_with_lds(kernel, [batch, 1, 1], [block_size, 1, 1],
        dq.kernarg_buf().gpu_addr, lds_bytes);

    dq.submit(&dev, &cb,
        &[dq.kernarg_buf(), &module.code_buf, &x_buf, &w_buf, &out_buf]).unwrap();

    // Verify
    let mut out_raw = vec![0u8; (batch * dim * 4) as usize];
    dev.download(&out_buf, &mut out_raw).unwrap();
    let out: &[f32] = unsafe { std::slice::from_raw_parts(out_raw.as_ptr() as *const f32, (batch * dim) as usize) };

    let mut bad = 0;
    for i in 0..(batch * dim) as usize {
        let err = (out[i] - expected[i]).abs();
        let tol = expected[i].abs() * 0.01 + 1e-4;
        if err > tol {
            if bad < 5 { eprintln!("  [{i}] got={:.6} exp={:.6}", out[i], expected[i]); }
            bad += 1;
        }
    }

    if bad == 0 {
        eprintln!("\n╔═══════════════════════════════════════════════════════════╗");
        eprintln!("║  REDLINE: HIPFIRE RMSNorm KERNEL VIA BARE DRM            ║");
        eprintln!("║  rmsnorm_f32: batch={}, dim={}, {} elements correct    ║", batch, dim, batch*dim);
        eprintln!("║  Uses LDS (shared memory). No HIP runtime.               ║");
        eprintln!("╚═══════════════════════════════════════════════════════════╝");
    } else {
        eprintln!("{bad}/{} wrong", batch * dim);
        std::process::exit(1);
    }

    dq.destroy(&dev);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
