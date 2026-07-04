// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for the indexed-MoE low-rank correction (gemv_lowrank_moe_proj +
//! _expand) vs a CPU oracle: out[krank,row] = U_e[row,:]·(V_e·x).
//!
//!   cargo run --release -p hipfire-rdna --example parity_lowrank_moe [M K R]

use hipfire_rdna::Gpu;

fn lcgf(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}
fn up(gpu: &mut Gpu, v: &[f32]) -> hipfire_rdna::GpuTensor {
    gpu.upload_raw(
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        &[v.len()],
    )
    .unwrap()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let r: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let n_exp = 8usize;
    let k_top = 8usize;
    let mut gpu = Gpu::init().unwrap();

    let mut u: Vec<Vec<f32>> = Vec::new(); // [e] U_e[M,r]
    let mut v: Vec<Vec<f32>> = Vec::new(); // [e] V_e[r,K]
    let (mut up_ptrs, mut vp_ptrs) = (Vec::<u64>::new(), Vec::<u64>::new());
    let mut keep = Vec::new();
    for e in 0..n_exp {
        let ue = lcgf(1 + e as u32, m * r);
        let ve = lcgf(0x100 + e as u32, r * k);
        let ut = up(&mut gpu, &ue);
        let vt = up(&mut gpu, &ve);
        up_ptrs.push(ut.buf.as_ptr() as u64);
        vp_ptrs.push(vt.buf.as_ptr() as u64);
        keep.push(ut);
        keep.push(vt);
        u.push(ue);
        v.push(ve);
    }
    let upt = gpu
        .upload_raw(
            &up_ptrs
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[n_exp],
        )
        .unwrap();
    let vpt = gpu
        .upload_raw(
            &vp_ptrs
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[n_exp],
        )
        .unwrap();
    let topk: Vec<i32> = (0..k_top as i32).collect();
    let topk_t = gpu
        .upload_raw(
            &topk
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[k_top],
        )
        .unwrap();
    let x = lcgf(7, k);
    let xd = up(&mut gpu, &x);
    let t = gpu
        .upload_raw(&vec![0u8; k_top * r * 4], &[k_top, r])
        .unwrap();
    let out = gpu
        .upload_raw(&vec![0u8; k_top * m * 4], &[k_top, m])
        .unwrap();

    gpu.gemv_lowrank_moe_proj(&vpt, &topk_t, &xd, &t, r, k, k_top, 0)
        .unwrap();
    gpu.gemv_lowrank_moe_expand(&upt, &topk_t, &t, &out, m, r, k_top)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&out).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for krank in 0..k_top {
        let e = topk[krank] as usize;
        let tc: Vec<f32> = (0..r)
            .map(|i| (0..k).map(|j| v[e][i * k + j] * x[j]).sum())
            .collect();
        for row in 0..m {
            let oc: f32 = (0..r).map(|i| u[e][row * r + i] * tc[i]).sum();
            let g = got[krank * m + row];
            max_abs = max_abs.max((g - oc).abs());
            max_mag = max_mag.max(oc.abs());
        }
    }
    let tol = 1e-3 * max_mag.max(1.0);
    let pass = max_abs <= tol;
    println!(
        "parity_lowrank_moe M={m} K={k} R={r} n_exp={n_exp} k_top={k_top} on {}: max_abs={max_abs:.5} (mag={max_mag:.2}) -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
