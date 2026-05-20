// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Debug: test RELEASE_MEM in isolation, then chained dispatch.
use redline::device::Device;
use redline::dispatch::{CommandBuffer, FastDispatch, Kernel};

fn main() {
    let dev = Device::open(None).unwrap();

    // === Test 1: RELEASE_MEM only (no dispatch) — matches working C test ===
    eprintln!("Test 1: RELEASE_MEM only (no dispatch)...");
    let fence = dev.alloc_vram(4096).unwrap();
    dev.upload(&fence, &vec![0u8; 64]).unwrap();

    let ib = dev.alloc_vram(4096).unwrap();
    // Build PM4: just RELEASE_MEM writing 0xDEAD to fence
    let pm4: Vec<u32> = vec![
        0xC0064900,         // PACKET3(RELEASE_MEM, 6) — NO SHADER_TYPE
        0x06603514,         // DW1: event + GCR
        0x20000000,         // DW2: DATA_SEL(1)
        fence.gpu_addr as u32,  // DW3: addr lo
        (fence.gpu_addr >> 32) as u32, // DW4: addr hi
        0x0000DEAD,         // DW5: fence value
        0,                  // DW6: 0
        0,                  // DW7: 0
    ];
    let ib_bytes: Vec<u8> = pm4.iter().flat_map(|d| d.to_le_bytes()).collect();
    dev.upload(&ib, &ib_bytes).unwrap();

    let queue = redline::queue::ComputeQueue::new(&dev).unwrap();
    match queue.submit_and_wait(&dev, &ib, pm4.len() as u32, &[&ib, &fence]) {
        Ok(()) => {
            let mut fb = vec![0u8; 4];
            dev.download(&fence, &mut fb).unwrap();
            let val = u32::from_le_bytes([fb[0], fb[1], fb[2], fb[3]]);
            eprintln!("  fence=0x{:x} (expect 0xDEAD) — {}", val, if val == 0xDEAD { "OK" } else { "FAIL" });
        }
        Err(e) => eprintln!("  FAIL: {e}"),
    }

    // === Test 2: RELEASE_MEM + WAIT_REG_MEM (barrier without dispatch) ===
    eprintln!("\nTest 2: RELEASE_MEM + WAIT_REG_MEM barrier...");
    dev.upload(&fence, &vec![0u8; 64]).unwrap();
    let pm4_2: Vec<u32> = vec![
        // RELEASE_MEM
        0xC0064900, 0x06603514, 0x20000000,
        fence.gpu_addr as u32, (fence.gpu_addr >> 32) as u32,
        1, 0, 0,  // fence value = 1
        // WAIT_REG_MEM
        0xC0053C00,  // PACKET3(WAIT_REG_MEM, 5) — NO SHADER_TYPE
        0x00000013,  // MEM_SPACE=1(mem) | FUNCTION=3(equal)
        fence.gpu_addr as u32, (fence.gpu_addr >> 32) as u32,
        1,           // reference = 1
        0xFFFFFFFF,  // mask
        4,           // poll interval
    ];
    let ib_bytes_2: Vec<u8> = pm4_2.iter().flat_map(|d| d.to_le_bytes()).collect();
    dev.upload(&ib, &ib_bytes_2).unwrap();
    match queue.submit_and_wait(&dev, &ib, pm4_2.len() as u32, &[&ib, &fence]) {
        Ok(()) => eprintln!("  OK — barrier packets execute without GPU reset"),
        Err(e) => eprintln!("  FAIL: {e}"),
    }

    // === Test 3: Full chain — dispatch → barrier → dispatch ===
    eprintln!("\nTest 3: Chained dispatch with barrier...");
    let hip_src = "#include <hip/hip_runtime.h>\nextern \"C\" __launch_bounds__(256)\n__global__ void vector_add(const float* a, const float* b, float* c, int n) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) c[i] = a[i] + b[i];\n}\n";
    std::fs::write("/tmp/redline_chain.hip", hip_src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3", "-o", "/tmp/redline_chain.hsaco", "/tmp/redline_chain.hip"])
        .output().expect("hipcc");
    assert!(out.status.success());
    let module = dev.load_module_file("/tmp/redline_chain.hsaco").unwrap();
    let kernel = Kernel::find(&module, "vector_add").unwrap();

    let n = 256u32;
    let a = dev.alloc_vram(1024).unwrap();
    let b = dev.alloc_vram(1024).unwrap();
    let c = dev.alloc_vram(1024).unwrap();
    let d_buf = dev.alloc_vram(1024).unwrap();
    let e = dev.alloc_vram(1024).unwrap();
    dev.upload(&a, &f32_bytes(&vec![1.0; 256])).unwrap();
    dev.upload(&b, &f32_bytes(&vec![2.0; 256])).unwrap();
    dev.upload(&c, &vec![0u8; 1024]).unwrap();
    dev.upload(&d_buf, &f32_bytes(&vec![10.0; 256])).unwrap();
    dev.upload(&e, &vec![0u8; 1024]).unwrap();
    dev.upload(&fence, &vec![0u8; 64]).unwrap();

    let fd = FastDispatch::new(&dev, &[&module.code_buf, &a, &b, &c, &d_buf, &e, &fence]).unwrap();

    // Build kernarg
    let mut ka_data = vec![0u8; 1024];
    ka_data[0..8].copy_from_slice(&a.gpu_addr.to_le_bytes());
    ka_data[8..16].copy_from_slice(&b.gpu_addr.to_le_bytes());
    ka_data[16..24].copy_from_slice(&c.gpu_addr.to_le_bytes());
    ka_data[24..28].copy_from_slice(&n.to_le_bytes());
    wh(&mut ka_data, 32, 1, 256);
    ka_data[512..520].copy_from_slice(&c.gpu_addr.to_le_bytes());
    ka_data[520..528].copy_from_slice(&d_buf.gpu_addr.to_le_bytes());
    ka_data[528..536].copy_from_slice(&e.gpu_addr.to_le_bytes());
    ka_data[536..540].copy_from_slice(&n.to_le_bytes());
    wh(&mut ka_data, 544, 1, 256);
    dev.upload(fd.ka_buf_ref(), &ka_data).unwrap();
    let ka_base = fd.ka_buf_ref().gpu_addr;

    // Build IB with correct barrier
    let mut cb = CommandBuffer::new();
    cb.dispatch(kernel, [1, 1, 1], [256, 1, 1], ka_base);
    cb.barrier(fence.gpu_addr, 1);
    cb.dispatch(kernel, [1, 1, 1], [256, 1, 1], ka_base + 512);
    eprintln!("  IB: {} dwords", cb.len_dwords());

    // (debug: print barrier dwords omitted — cb.dwords is private)

    match fd.submit_cmdbuf(&dev, &cb) {
        Ok(()) => {
            let c_val = rf32(&dev, &c);
            let e_val = rf32(&dev, &e);
            let wrong = e_val[..n as usize].iter().filter(|&&v| (v - 13.0).abs() > 0.001).count();
            eprintln!("  c[0]={} e[0]={} e[255]={} wrong={}/{}", c_val[0], e_val[0], e_val[255], wrong, n);
            if wrong == 0 {
                eprintln!("\n=== CHAINED DISPATCH WITH BARRIER: ALL CORRECT ===");
            }
        }
        Err(e) => eprintln!("  FAIL: {e}"),
    }
    fd.destroy(&dev);
}

fn f32_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }
fn wh(d: &mut [u8], off: usize, groups: u32, block: u16) {
    d[off..off+4].copy_from_slice(&groups.to_le_bytes());
    d[off+4..off+8].copy_from_slice(&1u32.to_le_bytes());
    d[off+8..off+12].copy_from_slice(&1u32.to_le_bytes());
    d[off+12..off+14].copy_from_slice(&block.to_le_bytes());
    d[off+14..off+16].copy_from_slice(&1u16.to_le_bytes());
    d[off+16..off+18].copy_from_slice(&1u16.to_le_bytes());
}
fn rf32(dev: &Device, buf: &redline::device::GpuBuffer) -> Vec<f32> {
    let mut r = vec![0u8; buf.size as usize];
    dev.download(buf, &mut r).unwrap();
    r.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}
