// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validate the clean dispatch API with vector_add and gemm_f32.

use redline::device::Device;
use redline::dispatch::{DispatchQueue, KernargBuilder, Kernel};

fn main() {
    eprintln!("=== redline dispatch API test ===\n");

    let dev = Device::open(None).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();

    // --- vector_add ---
    eprintln!("--- vector_add ---");
    let hip_va = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;
    std::fs::write("/tmp/redline_api_va.hip", hip_va).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3",
               "-o", "/tmp/redline_api_va.hsaco", "/tmp/redline_api_va.hip"])
        .output().expect("hipcc");
    assert!(out.status.success(), "hipcc: {}", String::from_utf8_lossy(&out.stderr));

    let va_mod = dev.load_module_file("/tmp/redline_api_va.hsaco").unwrap();
    let va_kernel = Kernel::find(&va_mod, "vector_add").expect("vector_add not found");
    eprintln!("loaded: {} (kernarg={})", va_kernel.name, va_kernel.kernarg_size);

    let n = 1024u32;
    let nbytes = (n as usize) * 4;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();

    let a_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let b_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let c_buf = dev.alloc_vram(nbytes as u64).unwrap();
    dev.upload(&a_buf, as_bytes(&a_data)).unwrap();
    dev.upload(&b_buf, as_bytes(&b_data)).unwrap();
    dev.upload(&c_buf, &vec![0u8; nbytes]).unwrap();

    // Explicit args only — dispatch auto-fills hidden args (block counts, group sizes)
    let mut ka = KernargBuilder::new(28); // 3 pointers (24) + 1 int (4)
    ka.write_ptr(0, a_buf.gpu_addr)
      .write_ptr(8, b_buf.gpu_addr)
      .write_ptr(16, c_buf.gpu_addr)
      .write_u32(24, n);

    let groups = (n + 255) / 256;
    dq.dispatch(&dev, va_kernel, [groups, 1, 1], [256, 1, 1],
        ka.as_bytes(), &[&va_mod.code_buf, &a_buf, &b_buf, &c_buf]).unwrap();

    let mut c_raw = vec![0u8; nbytes];
    dev.download(&c_buf, &mut c_raw).unwrap();
    let c: &[f32] = unsafe { std::slice::from_raw_parts(c_raw.as_ptr() as *const f32, n as usize) };
    let bad = (0..n as usize).filter(|&i| (c[i] - (i as f32) * 3.0).abs() > 0.001).count();
    if bad == 0 {
        eprintln!("  PASSED: {} elements correct", n);
    } else {
        eprintln!("  FAILED: {bad}/{n} wrong");
        std::process::exit(1);
    }

    // --- gemm_f32 ---
    eprintln!("\n--- gemm_f32 ---");
    let gemm_out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3",
               "-o", "/tmp/redline_api_gemm.hsaco", "kernels/src/gemm_f32.hip"])
        .output().expect("hipcc");
    assert!(gemm_out.status.success());

    let gemm_mod = dev.load_module_file("/tmp/redline_api_gemm.hsaco").unwrap();
    let gemm_kernel = Kernel::find(&gemm_mod, "gemm_f32_batched").expect("gemm not found");
    eprintln!("loaded: {} (kernarg={})", gemm_kernel.name, gemm_kernel.kernarg_size);

    let m = 8u32;
    let k_dim = 128u32;
    let nn = 8u32;
    let a_gemm: Vec<f32> = (0..m * k_dim).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b_gemm: Vec<f32> = (0..nn * k_dim).map(|i| ((i % 5) as f32) * 0.1).collect();
    let mut expected = vec![0.0f32; (m * nn) as usize];
    for mi in 0..m as usize {
        for ni in 0..nn as usize {
            for ki in 0..k_dim as usize {
                expected[mi * nn as usize + ni] += a_gemm[mi * k_dim as usize + ki] * b_gemm[ni * k_dim as usize + ki];
            }
        }
    }

    let ga = dev.alloc_vram((m * k_dim * 4) as u64).unwrap();
    let gb = dev.alloc_vram((nn * k_dim * 4) as u64).unwrap();
    let gy = dev.alloc_vram((m * nn * 4) as u64).unwrap();
    dev.upload(&ga, as_bytes(&a_gemm)).unwrap();
    dev.upload(&gb, as_bytes(&b_gemm)).unwrap();
    dev.upload(&gy, &vec![0u8; (m * nn * 4) as usize]).unwrap();

    let mut gka = KernargBuilder::new(36); // 3 pointers (24) + 3 ints (12)
    gka.write_ptr(0, ga.gpu_addr)
       .write_ptr(8, gb.gpu_addr)
       .write_ptr(16, gy.gpu_addr)
       .write_u32(24, m)
       .write_u32(28, k_dim)
       .write_u32(32, nn);

    dq.dispatch(&dev, gemm_kernel, [m, nn, 1], [32, 1, 1],
        gka.as_bytes(), &[&gemm_mod.code_buf, &ga, &gb, &gy]).unwrap();

    let mut y_raw = vec![0u8; (m * nn * 4) as usize];
    dev.download(&gy, &mut y_raw).unwrap();
    let y: &[f32] = unsafe { std::slice::from_raw_parts(y_raw.as_ptr() as *const f32, (m * nn) as usize) };
    let bad = (0..(m * nn) as usize).filter(|&i| {
        (y[i] - expected[i]).abs() > expected[i].abs() * 0.01 + 0.001
    }).count();
    if bad == 0 {
        eprintln!("  PASSED: {}x{}x{} = {} elements correct", m, k_dim, nn, m * nn);
    } else {
        eprintln!("  FAILED: {bad}/{} wrong", m * nn);
        eprintln!("  first 4 got:  {:?}", &y[..4]);
        eprintln!("  first 4 exp:  {:?}", &expected[..4]);
        std::process::exit(1);
    }

    eprintln!("\n=== All dispatch API tests PASSED ===");
    dq.destroy(&dev);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
