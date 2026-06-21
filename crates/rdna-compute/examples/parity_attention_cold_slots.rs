// SPDX-License-Identifier: Apache-2.0
//! Parity for `attention_cold_slots` (deferred-hierarchical KV, Phase 2b): the GPU
//! zero-LDS cold-slot decode attention vs a host reference (the same math as
//! hipfire_kvquant::ColdTier::two_tier_attend with n_hot=0 — attention over the
//! merged cold slots, GQA, all visible). f32 throughout, so expect bit-close.
//!
//!   cargo run --release -p rdna-compute --example parity_attention_cold_slots [n_slots]

use rdna_compute::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s as f32 / 2_147_483_648.0 - 0.5) * 2.0
        })
        .collect()
}

fn host_cold_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    nh: usize,
    nkv: usize,
    ns: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; nh * d];
    for hq in 0..nh {
        let kv = hq / (nh / nkv);
        let qh = &q[hq * d..hq * d + d];
        let mut logits = vec![0.0f32; ns];
        for s in 0..ns {
            let kb = (kv * ns + s) * d;
            logits[s] = (0..d).map(|i| qh[i] * k[kb + i]).sum::<f32>() * scale;
        }
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|x| (x - mx).exp()).collect();
        let z: f32 = p.iter().sum();
        for x in &mut p {
            *x /= z;
        }
        for s in 0..ns {
            let vb = (kv * ns + s) * d;
            for i in 0..d {
                out[hq * d + i] += p[s] * v[vb + i];
            }
        }
    }
    out
}

fn main() {
    let ns: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(88);
    let (nh, nkv, d) = (8usize, 2usize, 256usize); // qwen3.5-0.8b FA shape
    let scale = 1.0 / (d as f32).sqrt();

    let q = lcg(1, nh * d);
    let k = lcg(2, nkv * ns * d);
    let v = lcg(3, nkv * ns * d);

    let mut gpu = Gpu::init().unwrap();
    let qd = gpu.upload_f32(&q, &[nh, d]).unwrap();
    let kd = gpu.upload_f32(&k, &[nkv, ns, d]).unwrap();
    let vd = gpu.upload_f32(&v, &[nkv, ns, d]).unwrap();
    let od = gpu.alloc_tensor(&[nh * d], DType::F32).unwrap();
    let md = gpu.alloc_tensor(&[nh], DType::F32).unwrap();
    let ld = gpu.alloc_tensor(&[nh], DType::F32).unwrap();
    gpu.attention_cold_slots(&qd, &kd, &vd, &od, &md, &ld, nh, nkv, ns, scale, 0, 0).unwrap();
    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&od).unwrap();

    let want = host_cold_attn(&q, &k, &v, nh, nkv, ns, d, scale);
    let mut maxd = 0.0f32;
    let mut mag = 0.0f32;
    for i in 0..nh * d {
        maxd = maxd.max((got[i] - want[i]).abs());
        mag = mag.max(want[i].abs());
    }
    let tol = 2e-4 * mag.max(1.0);
    let pass = maxd <= tol;
    println!(
        "attention_cold_slots parity nh={nh} nkv={nkv} ns={ns} d={d} on {}: max_abs={maxd:.6} (mag={mag:.3}) tol={tol:.6} -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
