// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity microbench: gemm_hfq4g256 (scalar batched) vs gemv_hfq4g256 row loop.
//! No model needed — synthesizes random HFQ4G256 weight bytes (136 B/group).
//! Usage: hfq4_gemm_parity [--m M] [--k K] [--batch N]

fn main() {
    use rdna_compute::DType;
    let argv: Vec<String> = std::env::args().collect();
    let mut m = 4096usize;
    let mut k = 3840usize;
    let mut batch = 15usize;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--m" => { m = argv[i+1].parse().unwrap(); i += 2; }
            "--k" => { k = argv[i+1].parse().unwrap(); i += 2; }
            "--batch" => { batch = argv[i+1].parse().unwrap(); i += 2; }
            o => { eprintln!("unknown {o}"); std::process::exit(1); }
        }
    }
    assert_eq!(k % 256, 0);
    let groups = k / 256;
    let row_bytes = groups * 136;

    let mut gpu = rdna_compute::Gpu::init().expect("gpu");
    eprintln!("arch={} m={m} k={k} batch={batch}", gpu.arch);

    // xorshift weight bytes + scales
    let mut s: u64 = 0x12345678;
    let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let mut a = vec![0u8; m * row_bytes];
    for r in 0..m {
        for g in 0..groups {
            let off = r * row_bytes + g * 136;
            let scale = 0.01f32 + ((next() % 1000) as f32) / 50000.0;
            let zero = -8.0f32 * scale;
            a[off..off+4].copy_from_slice(&scale.to_le_bytes());
            a[off+4..off+8].copy_from_slice(&zero.to_le_bytes());
            for b in 0..128 { a[off + 8 + b] = (next() & 0xff) as u8; }
        }
    }
    let x: Vec<f32> = (0..batch * k).map(|_| ((next() % 2000) as f32 - 1000.0) / 500.0).collect();

    let a_g = gpu.alloc_tensor(&[m * row_bytes / 4], DType::F32).expect("a");
    gpu.hip.memcpy_htod(&a_g.buf, &a).unwrap();
    let x_g = gpu.upload_f32(&x, &[batch, k]).expect("x");
    let y_gemm = gpu.alloc_tensor(&[batch, m], DType::F32).expect("y");
    let y_gemv = gpu.alloc_tensor(&[m], DType::F32).expect("yv");
    let x_row = gpu.alloc_tensor(&[k], DType::F32).expect("xr");

    gpu.gemm_hfq4g256(&a_g, &x_g, &y_gemm, m, k, batch).expect("gemm");
    let got = gpu.download_f32(&y_gemm).unwrap();

    let mut worst = 0f32; let mut worst_at = (0usize, 0usize); let mut nbad = 0usize;
    for b in 0..batch {
        gpu.hip.memcpy_dtod_at(&x_row.buf, 0, &x_g.buf, b * k * 4, k * 4).unwrap();
        gpu.gemv_hfq4g256(&a_g, &x_row, &y_gemv, m, k).expect("gemv");
        let r = gpu.download_f32(&y_gemv).unwrap();
        for row in 0..m {
            let d = (r[row] - got[b * m + row]).abs();
            let rel = d / r[row].abs().max(1.0);
            if rel > 1e-3 { nbad += 1; }
            if rel > worst { worst = rel; worst_at = (b, row); }
        }
        if b < 3 {
            eprintln!("b={b}: gemv[0..4]={:?} gemm[0..4]={:?}", &r[..4], &got[b*m..b*m+4]);
        }
    }
    println!("worst rel diff {worst:.6} at (b={}, row={}), bad(>1e-3): {nbad}/{}", worst_at.0, worst_at.1, batch * m);
    std::process::exit(if nbad > 0 { 2 } else { 0 });
}
