// SPDX-License-Identifier: Apache-2.0
//! Decisive parity: batched (quantize_act_oq4[N] + gemm_oq4_grouped_wmma) vs
//! per-token (quantize_act_oq4[1] + gemv_oq4_grouped per row) on the SAME f32
//! activation + weight. This is exactly the batched-prefill-vs-decode contract
//! for one oq4 projection. Bit-exact expected; any drift here is the wiring bug.

use rdna_compute::{Gpu, DType};

fn lcgf(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (((s as f32 / 2_147_483_648.0) - 0.5) * 2.0).to_le_bytes()
        })
        .collect()
}
fn nib(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n).map(|_| { s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff; (s >> 13) as u8 }).collect()
}

fn main() {
    let m = 1536usize; let k = 1024usize; let group = 256usize;
    let ng = k / group;
    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() { println!("SKIP: no wmma"); return; }

    // Combined weight buffer [nibbles m*k/2 | f32 scales m*ng], Raw.
    let mut wbuf = nib(7, m * (k / 2));
    wbuf.extend_from_slice(&lcgf(8, m * ng));
    let w = gpu.upload_raw(&wbuf, &[wbuf.len()]).unwrap();
    let ws = w.sub_offset(m * (k / 2), m * ng * 4);

    // Boundary sweep: WMMA grouped GEMM vs per-row GEMV across n=5..9 with a
    // PRE-DIRTIED scratch (padding rows hold non-zero garbage). If a padding-row
    // guard is missing, valid-row outputs flip when n crosses a tile fraction.
    for &nn in &[5usize, 6, 7, 8, 9] {
        let xv: Vec<f32> = lcgf(100 + nn as u32, nn * k).chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        let xb = gpu.upload_f32(&xv, &[nn, k]).unwrap();
        // Dirty scratch sized for 16 rows so padding isn't zero.
        let dq = gpu.upload_raw(&nib(999, 16 * (k / 2)), &[16 * (k / 2)]).unwrap();
        let ds_v: Vec<f32> = (0..16 * ng).map(|i| 0.5 + (i % 7) as f32).collect();
        let ds = gpu.upload_f32(&ds_v, &[16 * ng]).unwrap();
        let yb = gpu.alloc_tensor(&[nn * m], DType::F32).unwrap();
        gpu.quantize_act_oq4(&xb, &dq, &ds, nn, k, group).unwrap();
        gpu.gemm_oq4_grouped_wmma(&w, &ws, &dq, &ds, &yb, m, k, nn, group).unwrap();
        gpu.device_synchronize().unwrap();
        let fbn = gpu.download_f32(&yb).unwrap();
        let q1 = gpu.alloc_tensor(&[k/2], DType::Raw).unwrap();
        let s1 = gpu.alloc_tensor(&[ng], DType::F32).unwrap();
        let yy = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let mut fpn = vec![0f32; nn*m];
        for r in 0..nn {
            let xr = xb.sub_offset(r*k, k);
            gpu.quantize_act_oq4(&xr, &q1, &s1, 1, k, group).unwrap();
            // Per-token W4A4 reference: the iu4 WMMA GEMM at B=1 (the decode
            // gemv_oq4_grouped is now W4A16, so it is no longer the W4A4 oracle).
            gpu.gemm_oq4_grouped_wmma(&w, &ws, &q1, &s1, &yy, m, k, 1, group).unwrap();
            gpu.device_synchronize().unwrap();
            fpn[r*m..(r+1)*m].copy_from_slice(&gpu.download_f32(&yy).unwrap());
        }
        let mut mxd=0f32; for i in 0..nn*m { mxd=mxd.max((fbn[i]-fpn[i]).abs()); }
        println!("BOUNDARY n={nn}: wmma-vs-gemv max_abs={mxd:.6} -> {}", if mxd==0.0 {"BIT-EXACT"} else {"DIFFERS"});
    }
}
