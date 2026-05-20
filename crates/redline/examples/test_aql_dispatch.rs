// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Test: AQL user-mode dispatch via /dev/kfd.
//! Dispatches vector_add without any syscall in the hot path.

use redline::device::Device;
use redline::dispatch::{KernargBuilder, Kernel};
use redline::kfd::AqlQueue;

fn main() {
    eprintln!("=== redline: AQL user-mode dispatch test ===\n");

    let dev = Device::open(None).unwrap();
    let arch = dev.info.gfx_arch.clone();
    eprintln!("[aql-test] targeting {}", arch);

    // Create AQL queue
    let aql = match AqlQueue::new(&dev) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("FAILED to create AQL queue: {e}");
            eprintln!("This requires /dev/kfd access and KFD module loaded.");
            std::process::exit(1);
        }
    };

    // Compile vector_add
    let hip_src = r#"
#include <hip/hip_runtime.h>
extern "C" __launch_bounds__(256)
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;
    std::fs::write("/tmp/redline_aql_va.hip", hip_src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", &format!("--offload-arch={arch}"), "-O3",
               "-o", "/tmp/redline_aql_va.hsaco", "/tmp/redline_aql_va.hip"])
        .output().expect("hipcc");
    assert!(out.status.success(), "hipcc: {}", String::from_utf8_lossy(&out.stderr));

    let module = dev.load_module_file("/tmp/redline_aql_va.hsaco").unwrap();
    let kernel = Kernel::find(&module, "vector_add").expect("vector_add not found");

    // For AQL, kernel_object = kernel descriptor VA (NOT code entry VA)
    // We need to compute kd_va from the module
    let kd_va = module.code_buf.gpu_addr + kernel_kd_offset(&module, "vector_add");
    eprintln!("kernel: {} kd_va=0x{:x}", kernel.name, kd_va);

    // Prepare data
    let n = 256u32;
    let nbytes = (n as usize) * 4;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();

    let a_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let b_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let c_buf = dev.alloc_vram(nbytes as u64).unwrap();
    dev.upload(&a_buf, as_bytes(&a_data)).unwrap();
    dev.upload(&b_buf, as_bytes(&b_data)).unwrap();
    dev.upload(&c_buf, &vec![0u8; nbytes]).unwrap();

    // Kernarg — for AQL, we need the FULL kernarg with hidden args
    let mut ka = vec![0u8; kernel.kernarg_size as usize];
    ka[0..8].copy_from_slice(&a_buf.gpu_addr.to_le_bytes());
    ka[8..16].copy_from_slice(&b_buf.gpu_addr.to_le_bytes());
    ka[16..24].copy_from_slice(&c_buf.gpu_addr.to_le_bytes());
    ka[24..28].copy_from_slice(&n.to_le_bytes());
    // Hidden args at offset 32
    let groups = (n + 255) / 256;
    ka[32..36].copy_from_slice(&groups.to_le_bytes()); // block_count_x
    ka[36..40].copy_from_slice(&1u32.to_le_bytes());   // block_count_y
    ka[40..44].copy_from_slice(&1u32.to_le_bytes());   // block_count_z
    ka[44..46].copy_from_slice(&256u16.to_le_bytes());  // group_size_x
    ka[46..48].copy_from_slice(&1u16.to_le_bytes());    // group_size_y
    ka[48..50].copy_from_slice(&1u16.to_le_bytes());    // group_size_z

    // Upload kernarg to VRAM
    let ka_buf = dev.alloc_vram(4096).unwrap();
    dev.upload(&ka_buf, &ka).unwrap();

    // Dispatch via AQL (no syscall!)
    eprintln!("Dispatching via AQL queue...");
    aql.dispatch_and_wait(kd_va, [groups, 1, 1], [256, 1, 1], ka_buf.gpu_addr, 0);

    // Verify
    let mut c_raw = vec![0u8; nbytes];
    dev.download(&c_buf, &mut c_raw).unwrap();
    let c: &[f32] = unsafe { std::slice::from_raw_parts(c_raw.as_ptr() as *const f32, n as usize) };
    let bad = (0..n as usize).filter(|&i| (c[i] - (i as f32) * 3.0).abs() > 0.001).count();

    if bad == 0 {
        eprintln!("\n=== AQL DISPATCH PASSED — {} elements correct ===", n);
        eprintln!("User-mode submission: no syscall in the dispatch path!");
    } else {
        eprintln!("FAILED: {bad}/{n} wrong");
        eprintln!("c[0..8] = {:?}", &c[..8]);
    }

    aql.destroy();
}

fn kernel_kd_offset(module: &redline::dispatch::LoadedModule, name: &str) -> u64 {
    // The Kernel struct has code_va but for AQL we need kd_va.
    // We need the kd_offset from the hsaco module.
    // Re-parse to get it.
    let data = std::fs::read("/tmp/redline_aql_va.hsaco").unwrap();
    let hsaco = redline::hsaco::HsacoModule::from_bytes(data).unwrap();
    for k in &hsaco.kernels {
        if k.name == name {
            return k.kd_offset;
        }
    }
    panic!("kernel {} not found", name);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
