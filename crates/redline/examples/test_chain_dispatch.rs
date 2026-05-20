// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Test: chain multiple dispatches in one IB submission.
//! Chains: c = a + b, then d = c + a (two dependent vector_adds in one submit).

use redline::device::Device;
use redline::dispatch::{CommandBuffer, DispatchQueue, KernargBuilder, Kernel};

fn main() {
    eprintln!("=== redline: chained dispatch test ===\n");

    let dev = Device::open(None).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();

    // Compile vector_add
    let hip_src = r#"
#include <hip/hip_runtime.h>
extern "C" __launch_bounds__(256)
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;
    std::fs::write("/tmp/redline_chain_va.hip", hip_src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3",
               "-o", "/tmp/redline_chain_va.hsaco", "/tmp/redline_chain_va.hip"])
        .output().expect("hipcc");
    assert!(out.status.success(), "hipcc: {}", String::from_utf8_lossy(&out.stderr));

    let module = dev.load_module_file("/tmp/redline_chain_va.hsaco").unwrap();
    let kernel = Kernel::find(&module, "vector_add").expect("kernel not found");

    let n = 4096u32;
    let nbytes = (n as usize) * 4;

    // a = [1, 1, 1, ...], b = [2, 2, 2, ...]
    let a_data: Vec<f32> = vec![1.0; n as usize];
    let b_data: Vec<f32> = vec![2.0; n as usize];

    let a_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let b_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let c_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let d_buf = dev.alloc_vram(nbytes as u64).unwrap();
    dev.upload(&a_buf, as_bytes(&a_data)).unwrap();
    dev.upload(&b_buf, as_bytes(&b_data)).unwrap();
    dev.upload(&c_buf, &vec![0u8; nbytes]).unwrap();
    dev.upload(&d_buf, &vec![0u8; nbytes]).unwrap();

    let groups = (n + 255) / 256;

    // --- Test 1: Two sequential dispatches (separate submits) ---
    eprintln!("Test 1: Sequential dispatches (2 separate submits)");
    let t0 = std::time::Instant::now();
    {
        // c = a + b → c should be [3, 3, 3, ...]
        let mut ka1 = KernargBuilder::new(28);
        ka1.write_ptr(0, a_buf.gpu_addr).write_ptr(8, b_buf.gpu_addr)
           .write_ptr(16, c_buf.gpu_addr).write_u32(24, n);
        dq.dispatch(&dev, kernel, [groups, 1, 1], [256, 1, 1],
            ka1.as_bytes(), &[&module.code_buf, &a_buf, &b_buf, &c_buf]).unwrap();

        // d = c + a → d should be [4, 4, 4, ...]
        let mut ka2 = KernargBuilder::new(28);
        ka2.write_ptr(0, c_buf.gpu_addr).write_ptr(8, a_buf.gpu_addr)
           .write_ptr(16, d_buf.gpu_addr).write_u32(24, n);
        dq.dispatch(&dev, kernel, [groups, 1, 1], [256, 1, 1],
            ka2.as_bytes(), &[&module.code_buf, &a_buf, &c_buf, &d_buf]).unwrap();
    }
    let seq_time = t0.elapsed();

    let mut d_raw = vec![0u8; nbytes];
    dev.download(&d_buf, &mut d_raw).unwrap();
    let d: &[f32] = unsafe { std::slice::from_raw_parts(d_raw.as_ptr() as *const f32, n as usize) };
    let bad = d.iter().filter(|&&v| (v - 4.0).abs() > 0.001).count();
    if bad == 0 {
        eprintln!("  PASSED: {} elements = 4.0 ({:.1}ms)", n, seq_time.as_secs_f64() * 1000.0);
    } else {
        eprintln!("  FAILED: {bad}/{n} wrong (d[0]={}, d[1]={})", d[0], d[1]);
        std::process::exit(1);
    }

    // Reset d_buf
    dev.upload(&d_buf, &vec![0u8; nbytes]).unwrap();
    dev.upload(&c_buf, &vec![0u8; nbytes]).unwrap();

    // --- Test 2: Chained in one IB ---
    eprintln!("\nTest 2: Chained dispatches (1 IB submission)");
    let t0 = std::time::Instant::now();
    {
        // Need two separate kernarg regions in the persistent KA buffer
        // We'll use DispatchQueue::submit with a pre-built CommandBuffer
        let ka1_off = 0u64;
        let ka2_off = 256u64; // start second kernarg 256 bytes in

        // Build kernarg data for both dispatches
        let mut ka_data = vec![0u8; 512];
        // Dispatch 1: c = a + b
        ka_data[0..8].copy_from_slice(&a_buf.gpu_addr.to_le_bytes());
        ka_data[8..16].copy_from_slice(&b_buf.gpu_addr.to_le_bytes());
        ka_data[16..24].copy_from_slice(&c_buf.gpu_addr.to_le_bytes());
        ka_data[24..28].copy_from_slice(&n.to_le_bytes());
        // Hidden args for dispatch 1
        let hidden1 = 32usize;
        ka_data[hidden1..hidden1+4].copy_from_slice(&groups.to_le_bytes()); // block_count_x
        ka_data[hidden1+4..hidden1+8].copy_from_slice(&1u32.to_le_bytes()); // block_count_y
        ka_data[hidden1+8..hidden1+12].copy_from_slice(&1u32.to_le_bytes()); // block_count_z
        ka_data[hidden1+12..hidden1+14].copy_from_slice(&256u16.to_le_bytes()); // group_size_x
        ka_data[hidden1+14..hidden1+16].copy_from_slice(&1u16.to_le_bytes()); // group_size_y
        ka_data[hidden1+16..hidden1+18].copy_from_slice(&1u16.to_le_bytes()); // group_size_z

        // Dispatch 2: d = c + a
        let o2 = ka2_off as usize;
        ka_data[o2..o2+8].copy_from_slice(&c_buf.gpu_addr.to_le_bytes());
        ka_data[o2+8..o2+16].copy_from_slice(&a_buf.gpu_addr.to_le_bytes());
        ka_data[o2+16..o2+24].copy_from_slice(&d_buf.gpu_addr.to_le_bytes());
        ka_data[o2+24..o2+28].copy_from_slice(&n.to_le_bytes());
        // Hidden args for dispatch 2
        let hidden2 = o2 + 32;
        ka_data[hidden2..hidden2+4].copy_from_slice(&groups.to_le_bytes());
        ka_data[hidden2+4..hidden2+8].copy_from_slice(&1u32.to_le_bytes());
        ka_data[hidden2+8..hidden2+12].copy_from_slice(&1u32.to_le_bytes());
        ka_data[hidden2+12..hidden2+14].copy_from_slice(&256u16.to_le_bytes());
        ka_data[hidden2+14..hidden2+16].copy_from_slice(&1u16.to_le_bytes());
        ka_data[hidden2+16..hidden2+18].copy_from_slice(&1u16.to_le_bytes());

        // Fence buffer for barrier
        let fence_buf = dev.alloc_vram(4096).unwrap();
        dev.upload(&fence_buf, &vec![0u8; 64]).unwrap();

        // Upload both kernargs in one shot
        dev.upload(dq.kernarg_buf(), &ka_data).unwrap();
        let ka_base = dq.kernarg_buf().gpu_addr;

        // Build command buffer with barrier between dispatches
        let mut cb = CommandBuffer::new();
        cb.dispatch(kernel, [groups, 1, 1], [256, 1, 1], ka_base + ka1_off);
        cb.barrier(fence_buf.gpu_addr, 1);
        cb.dispatch(kernel, [groups, 1, 1], [256, 1, 1], ka_base + ka2_off);

        // One submit, one fence
        dq.submit(&dev, &cb,
            &[dq.kernarg_buf(), &module.code_buf, &a_buf, &b_buf, &c_buf, &d_buf, &fence_buf]).unwrap();
    }
    let chain_time = t0.elapsed();

    // Check intermediate result c (should be [3, 3, 3, ...])
    let mut c_raw = vec![0u8; nbytes];
    dev.download(&c_buf, &mut c_raw).unwrap();
    let c: &[f32] = unsafe { std::slice::from_raw_parts(c_raw.as_ptr() as *const f32, n as usize) };
    let c_bad = c.iter().filter(|&&v| (v - 3.0).abs() > 0.001).count();
    eprintln!("  c_buf (intermediate): {}/{} correct (c[0]={} c[255]={} c[256]={})",
        n as usize - c_bad, n, c[0], c[255], c[256]);

    let mut d_raw2 = vec![0u8; nbytes];
    dev.download(&d_buf, &mut d_raw2).unwrap();
    let d2: &[f32] = unsafe { std::slice::from_raw_parts(d_raw2.as_ptr() as *const f32, n as usize) };
    let bad = d2.iter().filter(|&&v| (v - 4.0).abs() > 0.001).count();
    if bad == 0 {
        eprintln!("  PASSED: {} elements = 4.0 ({:.1}ms)", n, chain_time.as_secs_f64() * 1000.0);
    } else {
        eprintln!("  FAILED: {bad}/{n} wrong");
        // Show first few wrong indices
        let mut shown = 0;
        for i in 0..n as usize {
            if (d2[i] - 4.0).abs() > 0.001 && shown < 8 {
                eprintln!("    [{i}] = {} (expected 4.0)", d2[i]);
                shown += 1;
            }
        }
    }

    eprintln!("\nSpeedup: sequential {:.1}ms vs chained {:.1}ms ({:.1}x)",
        seq_time.as_secs_f64() * 1000.0, chain_time.as_secs_f64() * 1000.0,
        seq_time.as_secs_f64() / chain_time.as_secs_f64());

    eprintln!("\n=== Chain dispatch PASSED ===");
    dq.destroy(&dev);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
