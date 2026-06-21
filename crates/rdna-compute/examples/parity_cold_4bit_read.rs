// SPDX-License-Identifier: Apache-2.0
//! Parity for the 4-bit-RESIDENT cold-tier read (Phase 2b sub-task 4c): the cold
//! tier stays as KVarN 4-bit records on GPU and is dequantized on-the-fly each
//! decode step via `kvarn_dequant_tile`, then attended with the channel-major
//! mode of `attention_cold_slots` (kv_layout=1) — no transpose, no f32-resident
//! storage. This is the path that actually realizes the KV storage win.
//!
//! Validates: upload ColdTier 4-bit tiles → GPU kvarn_dequant_tile → channel-major
//! attention_cold_slots, vs the CPU oracle (CPU dequant_head + plain GQA attention
//! over the same slots). The GPU dequant and CPU dequant_head decode the SAME
//! records, so this should match to f32 tolerance (rotate=false avoids the FWHT
//! basis question for v1; rotation is a later quality lever).
//!
//!   cargo run --release -p rdna-compute --example parity_cold_4bit_read [n_cold_tok]

use hipfire_kvquant::kv_compact::compact_cold_kv;
use hipfire_kvquant::kvarn::{kvarn_record_bytes, pack_kvarn_tile};
use rdna_compute::{DType, Gpu};

const NH: usize = 8;
const NKV: usize = 2;
const HD: usize = 256;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s as f32 / 2_147_483_648.0 - 0.5) * 2.0
        })
        .collect()
}

fn main() {
    let n_cold_tok: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let scale = 1.0 / (HD as f32).sqrt();
    let kv_dim = NKV * HD;

    let q = lcg(1, NH * HD);
    let ck = lcg(2, n_cold_tok * kv_dim);
    let cv = lcg(3, n_cold_tok * kv_dim);
    let importance: Vec<f32> = (0..n_cold_tok).map(|t| 0.1 + (t as f32 / n_cold_tok as f32)).collect();

    // v1 cold tier: rotate=false (no FWHT basis juggling for the GPU read).
    let cold = compact_cold_kv(&ck, &cv, n_cold_tok, NKV, HD, &importance, 0.125, 4, false);
    let nvc = cold.n_valid;
    let ns = cold.n_slots; // padded even — tile width
    let rec_bytes = kvarn_record_bytes(HD, ns);

    // ── CPU oracle: dequant_head + plain GQA attention over the nvc valid slots.
    let cold_deq: Vec<(Vec<f32>, Vec<f32>)> = (0..NKV).map(|h| cold.dequant_head(h)).collect();
    let mut oracle = vec![0.0f32; NH * HD];
    for hq in 0..NH {
        let kvh = hq / (NH / NKV);
        let (k, v) = &cold_deq[kvh];
        let qh = &q[hq * HD..hq * HD + HD];
        let mut logits = vec![0.0f32; nvc];
        for s in 0..nvc {
            logits[s] = (0..HD).map(|i| qh[i] * k[s * HD + i]).sum::<f32>() * scale;
        }
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|x| (x - mx).exp()).collect();
        let z: f32 = p.iter().sum();
        for x in &mut p {
            *x /= z;
        }
        for s in 0..nvc {
            for i in 0..HD {
                oracle[hq * HD + i] += p[s] * v[s * HD + i];
            }
        }
    }

    // ── GPU: upload the 4-bit record tiles, dequant on-GPU, channel-major attend.
    let mut gpu = Gpu::init().unwrap();
    let qd = gpu.upload_f32(&q, &[NH, HD]).unwrap();

    // Records concatenated per kv-head: [NKV × rec_bytes]. dequant → [NKV × HD × ns].
    let mut krecs = Vec::with_capacity(NKV * rec_bytes);
    let mut vrecs = Vec::with_capacity(NKV * rec_bytes);
    for h in 0..NKV {
        let kp = pack_kvarn_tile(&cold.k_tiles[h]);
        let vp = pack_kvarn_tile(&cold.v_tiles[h]);
        assert_eq!(kp.len(), rec_bytes);
        krecs.extend_from_slice(&kp);
        vrecs.extend_from_slice(&vp);
    }
    let krecs_d = gpu.upload_raw(&krecs, &[NKV * rec_bytes / 4]).unwrap();
    let vrecs_d = gpu.upload_raw(&vrecs, &[NKV * rec_bytes / 4]).unwrap();

    // Dequant into channel-major [NKV × HD × ns] f16 scratch (kvarn_dequant_tile
    // emits __half; r=HD, c=ns). 2 bytes/elem.
    let kdq = gpu.upload_raw(&vec![0u8; NKV * HD * ns * 2], &[NKV * HD * ns]).unwrap();
    let vdq = gpu.upload_raw(&vec![0u8; NKV * HD * ns * 2], &[NKV * HD * ns]).unwrap();
    gpu.kvarn_dequant_tile(&krecs_d, &kdq, NKV, HD, ns, rec_bytes).unwrap();
    gpu.kvarn_dequant_tile(&vrecs_d, &vdq, NKV, HD, ns, rec_bytes).unwrap();

    // Channel-major attention over the nvc valid slots: count = nvc, stride = ns
    // (the padded tile width), so the pad slot (when nvc is odd) is never read.
    let od = gpu.alloc_tensor(&[NH * HD], DType::F32).unwrap();
    let md = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    let ld = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    gpu.attention_cold_slots(&qd, &kdq, &vdq, &od, &md, &ld, NH, NKV, nvc, scale, 1, ns).unwrap();

    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&od).unwrap();

    let mut maxd = 0.0f32;
    let mut mag = 0.0f32;
    for i in 0..NH * HD {
        maxd = maxd.max((got[i] - oracle[i]).abs());
        mag = mag.max(oracle[i].abs());
    }
    // GPU dequant emits f16; the CPU oracle dequants to f32 — so the gap is f16
    // rounding of the cold slots (~1e-3), not an algorithmic error. f16 tolerance.
    let tol = 2.5e-3f32;
    let pass = maxd <= tol;
    println!(
        "cold 4-bit on-the-fly read (GPU f16 dequant + channel-major attend vs CPU f32 oracle) n_cold_tok={n_cold_tok} n_slots={ns} on {}: max_abs={maxd:.6} (mag={mag:.3}) tol={tol:.6} -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
