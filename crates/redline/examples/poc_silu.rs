// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Redline: dispatch hipfire's SiLU activation kernel through bare libdrm.
//! SiLU(x) = x / (1 + exp(-x)) — used in LLaMA, Qwen, etc.

use redline::device::Device;
use redline::dispatch::{DispatchQueue, KernargBuilder, Kernel};

fn main() {
    eprintln!("=== redline: hipfire SiLU kernel ===\n");

    let dev = Device::open(None).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();

    // Compile hipfire's actual silu kernel
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3",
               "-o", "/tmp/redline_silu.hsaco", "kernels/src/silu.hip"])
        .output().expect("hipcc");
    assert!(out.status.success(), "hipcc: {}", String::from_utf8_lossy(&out.stderr));

    let module = dev.load_module_file("/tmp/redline_silu.hsaco").unwrap();
    let kernel = Kernel::find(&module, "silu_f32").expect("silu_f32 not found");
    eprintln!("kernel: {} (kernarg={}, lds={})",
        kernel.name, kernel.kernarg_size, kernel.group_segment_size);

    let n = 4096u32;
    let nbytes = (n as usize) * 4;

    // Generate test data
    let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - 2048.0) * 0.01).collect();

    // CPU reference: silu(x) = x / (1 + exp(-x))
    let expected: Vec<f32> = x_data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();

    let x_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let out_buf = dev.alloc_vram(nbytes as u64).unwrap();
    dev.upload(&x_buf, as_bytes(&x_data)).unwrap();
    dev.upload(&out_buf, &vec![0u8; nbytes]).unwrap();

    // Kernarg: [x_ptr: u64, out_ptr: u64, n: i32]
    let mut ka = KernargBuilder::new(20); // 2 pointers (16) + 1 int (4)
    ka.write_ptr(0, x_buf.gpu_addr)
      .write_ptr(8, out_buf.gpu_addr)
      .write_u32(16, n);

    let groups = (n + 255) / 256;
    dq.dispatch(&dev, kernel, [groups, 1, 1], [256, 1, 1],
        ka.as_bytes(), &[&module.code_buf, &x_buf, &out_buf]).unwrap();

    // Verify
    let mut out_raw = vec![0u8; nbytes];
    dev.download(&out_buf, &mut out_raw).unwrap();
    let out: &[f32] = unsafe { std::slice::from_raw_parts(out_raw.as_ptr() as *const f32, n as usize) };

    let mut bad = 0;
    for i in 0..n as usize {
        let err = (out[i] - expected[i]).abs();
        let tol = expected[i].abs() * 0.001 + 1e-5;
        if err > tol {
            if bad < 5 { eprintln!("  [{i}] got={:.6} exp={:.6} err={:.6}", out[i], expected[i], err); }
            bad += 1;
        }
    }

    if bad == 0 {
        eprintln!("\n╔═══════════════════════════════════════════════════════════╗");
        eprintln!("║  REDLINE: HIPFIRE SiLU KERNEL VIA BARE DRM               ║");
        eprintln!("║  silu_f32: {} elements correct                         ║", n);
        eprintln!("║  A REAL inference kernel. No HIP runtime. Pure libdrm.    ║");
        eprintln!("╚═══════════════════════════════════════════════════════════╝");
    } else {
        eprintln!("{bad}/{n} wrong");
        std::process::exit(1);
    }

    dq.destroy(&dev);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
