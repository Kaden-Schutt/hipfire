// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Proof: chained dispatch with compute barrier in a single IB submission.
//! Chain: A+B→C | barrier | C+D→E in one amdgpu_cs_submit call.
//! If E=[13,13,...], the barrier correctly serialized dependent dispatches.

use redline::device::Device;
use redline::dispatch::{CommandBuffer, FastDispatch, Kernel, KernargBuilder};

fn main() {
    eprintln!("=== redline: chained dispatch with RELEASE_MEM barrier ===\n");

    let dev = Device::open(None).unwrap();

    // Compile vector_add with __launch_bounds__
    let hip_src = r#"
#include <hip/hip_runtime.h>
extern "C" __launch_bounds__(256)
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;
    std::fs::write("/tmp/redline_chain.hip", hip_src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3",
               "-o", "/tmp/redline_chain.hsaco", "/tmp/redline_chain.hip"])
        .output().expect("hipcc");
    assert!(out.status.success());

    let module = dev.load_module_file("/tmp/redline_chain.hsaco").unwrap();
    let kernel = Kernel::find(&module, "vector_add").expect("kernel not found");

    let n = 4096u32;
    let nbytes = (n as usize) * 4;
    let groups = (n + 255) / 256;

    // A=[1,1,...], B=[2,2,...], D=[10,10,...]
    let a_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let b_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let c_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let d_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let e_buf = dev.alloc_vram(nbytes as u64).unwrap();
    dev.upload(&a_buf, &vec![1.0f32; n as usize].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    dev.upload(&b_buf, &vec![2.0f32; n as usize].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    dev.upload(&c_buf, &vec![0u8; nbytes]).unwrap();
    dev.upload(&d_buf, &vec![10.0f32; n as usize].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    dev.upload(&e_buf, &vec![0u8; nbytes]).unwrap();

    // Fence buffer for barrier
    let fence_buf = dev.alloc_vram(4096).unwrap();
    dev.upload(&fence_buf, &vec![0u8; 64]).unwrap(); // zero fence

    // FastDispatch with all buffers in persistent BO list
    let fd = FastDispatch::new(&dev, &[
        &module.code_buf, &a_buf, &b_buf, &c_buf, &d_buf, &e_buf, &fence_buf,
    ]).unwrap();

    // Build kernarg for dispatch 1: C = A + B
    let mut ka1 = KernargBuilder::new(28);
    ka1.write_ptr(0, a_buf.gpu_addr).write_ptr(8, b_buf.gpu_addr)
       .write_ptr(16, c_buf.gpu_addr).write_u32(24, n);

    // Build kernarg for dispatch 2: E = C + D
    let mut ka2 = KernargBuilder::new(28);
    ka2.write_ptr(0, c_buf.gpu_addr).write_ptr(8, d_buf.gpu_addr)
       .write_ptr(16, e_buf.gpu_addr).write_u32(24, n);

    // Upload both kernargs to different offsets in the persistent KA buffer
    // Dispatch 1 kernarg at offset 0, dispatch 2 at offset 256
    let ka1_bytes = ka1.as_bytes();
    let ka2_bytes = ka2.as_bytes();
    let mut ka_full = vec![0u8; 512];
    ka_full[..ka1_bytes.len()].copy_from_slice(ka1_bytes);
    // Hidden args for dispatch 1
    ka_full[32..36].copy_from_slice(&groups.to_le_bytes());
    ka_full[36..40].copy_from_slice(&1u32.to_le_bytes());
    ka_full[40..44].copy_from_slice(&1u32.to_le_bytes());
    ka_full[44..46].copy_from_slice(&256u16.to_le_bytes());
    ka_full[46..48].copy_from_slice(&1u16.to_le_bytes());
    ka_full[48..50].copy_from_slice(&1u16.to_le_bytes());

    ka_full[256..256 + ka2_bytes.len()].copy_from_slice(ka2_bytes);
    ka_full[288..292].copy_from_slice(&groups.to_le_bytes());
    ka_full[292..296].copy_from_slice(&1u32.to_le_bytes());
    ka_full[296..300].copy_from_slice(&1u32.to_le_bytes());
    ka_full[300..302].copy_from_slice(&256u16.to_le_bytes());
    ka_full[302..304].copy_from_slice(&1u16.to_le_bytes());
    ka_full[304..306].copy_from_slice(&1u16.to_le_bytes());

    // Upload all kernargs at once
    dev.upload(fd.ka_buf_ref(), &ka_full).unwrap();
    let ka_base = fd.ka_buf_ref().gpu_addr;

    // Build command buffer: dispatch1 → barrier → dispatch2
    let mut cb = CommandBuffer::new();
    cb.dispatch(kernel, [groups, 1, 1], [256, 1, 1], ka_base);
    cb.barrier(fence_buf.gpu_addr, 1);
    cb.dispatch(kernel, [groups, 1, 1], [256, 1, 1], ka_base + 256);

    eprintln!("IB: {} dwords ({} bytes)", cb.len_dwords(), cb.len_dwords() * 4);
    eprintln!("Submitting chained dispatch (1 ioctl)...");

    fd.submit_cmdbuf(&dev, &cb).unwrap();

    // Verify
    let mut e_raw = vec![0u8; nbytes];
    dev.download(&e_buf, &mut e_raw).unwrap();
    let e: &[f32] = unsafe { std::slice::from_raw_parts(e_raw.as_ptr() as *const f32, n as usize) };

    let bad = e.iter().filter(|&&v| (v - 13.0).abs() > 0.001).count();
    if bad == 0 {
        eprintln!("\n╔════════════════════════════════════════════════════════════╗");
        eprintln!("║  CHAINED DISPATCH: {} elements = 13.0 (1+2+10)          ║", n);
        eprintln!("║  Two dependent dispatches in ONE amdgpu_cs_submit call    ║");
        eprintln!("║  RELEASE_MEM + WAIT_REG_MEM barrier works on gfx1010!     ║");
        eprintln!("╚════════════════════════════════════════════════════════════╝");
    } else {
        eprintln!("FAILED: {bad}/{n} wrong");
        eprintln!("e[0]={} e[1]={} e[256]={} e[4095]={}", e[0], e[1], e[256], e[4095]);

        // Also check C
        let mut c_raw = vec![0u8; nbytes];
        dev.download(&c_buf, &mut c_raw).unwrap();
        let c: &[f32] = unsafe { std::slice::from_raw_parts(c_raw.as_ptr() as *const f32, n as usize) };
        eprintln!("c[0]={} (expect 3.0)", c[0]);
    }

    fd.destroy(&dev);
}
