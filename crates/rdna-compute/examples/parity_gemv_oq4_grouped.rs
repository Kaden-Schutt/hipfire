// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `gemv_oq4_grouped` (OQ4+ decode B=1 GEMV, W4A16) vs a CPU oracle.
//! The kernel unpacks the 4-bit-resident weight inline and multiplies by the
//! FULL-PRECISION f32 activation (W4A16 decode), so the reference is a direct
//! f32 dot: y[m] = Σ_g sw[m,g]·Σ_{k∈g} dequant(qw)·x[k].
//!
//!   cargo run --release -p rdna-compute --example parity_gemv_oq4_grouped [M K]

use rdna_compute::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}
fn lcgf_vals(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3584);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let group = 256usize;
    assert_eq!(k % group, 0);
    let ng = k / group;

    let mut gpu = Gpu::init().unwrap();

    // Combined weight buffer [packed nibbles M*(K/2) | f32 scales M*ng].
    let wnib = lcg(1, m * (k / 2));
    let wsc = lcgf_vals(0x11, m * ng).iter().map(|v| 0.01 + v.abs() * 0.25).collect::<Vec<_>>();
    let mut wbuf = wnib.clone();
    for s in &wsc {
        wbuf.extend_from_slice(&s.to_le_bytes());
    }
    let x: Vec<f32> = lcgf_vals(3, k);

    let wd = gpu.upload_raw(&wbuf, &[wbuf.len()]).unwrap();
    let ws = wd.sub_offset(m * (k / 2), m * ng * 4);
    let mut xbytes = Vec::with_capacity(k * 4);
    for v in &x {
        xbytes.extend_from_slice(&v.to_le_bytes());
    }
    let xd = gpu.upload_raw(&xbytes, &[1, k]).unwrap();

    let yg = gpu.upload_raw(&vec![0u8; m * 4], &[1, m]).unwrap();
    gpu.gemv_oq4_grouped(&wd, &ws, &xd, &yg, m, k, group).unwrap();
    gpu.device_synchronize().unwrap();
    let y_gemv = gpu.download_f32(&yg).unwrap();

    // CPU oracle: W4A16 dot.
    let sext = |nib: u8| -> i32 {
        let v = (nib & 0xf) as i32;
        (v << 28) >> 28
    };
    let mut y_ref = vec![0.0f32; m];
    for row in 0..m {
        let mut acc = 0.0f32;
        for g in 0..ng {
            let mut gsum = 0.0f32;
            for j in 0..group {
                let kk = g * group + j;
                let byte = wnib[row * (k / 2) + kk / 2];
                let nib = if kk & 1 == 0 { byte & 0xf } else { byte >> 4 };
                gsum += sext(nib) as f32 * x[kk];
            }
            acc += gsum * wsc[row * ng + g];
        }
        y_ref[row] = acc;
    }

    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for i in 0..m {
        max_abs = max_abs.max((y_gemv[i] - y_ref[i]).abs());
        max_mag = max_mag.max(y_ref[i].abs());
    }
    let tol = 1e-3 * max_mag.max(1.0);
    let pass = max_abs <= tol;
    println!(
        "parity_gemv_oq4_grouped (W4A16) M={m} K={k} on {}: max_abs={max_abs:.5} (mag={max_mag:.2}) -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
