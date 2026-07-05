// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for the gfx1103 no-LDS batched attention kernels (prefill / dflash
//! verify): attention_f32_batched + attention_q8_0_kv_batched, in both causal
//! (per-row positions) and tree-bias modes. Compares against an f64 CPU
//! reference. Default runs the gfx1103 no-LDS kernels; HIPFIRE_FORCE_GENERIC=1
//! runs the generic LDS kernels — both must match the reference.

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}
fn f16_to_f32(bits: u16) -> f32 {
    let s = (bits >> 15) & 1;
    let e = (bits >> 10) & 0x1f;
    let f = bits & 0x3ff;
    let v = if e == 0 {
        (f as f32) * 2f32.powi(-24)
    } else if e == 0x1f {
        if f == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + f as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s == 1 { -v } else { v }
}

const NH: usize = 8;
const NKV: usize = 2;
const HD: usize = 128;
const MAXS: usize = 256;
const KVDIM: usize = NKV * HD;
const B: usize = 5;

fn main() {
    let forced = std::env::var("HIPFIRE_FORCE_GENERIC").is_ok();
    let mut gpu = Gpu::init().expect("gpu init");
    println!("force_generic={forced} (path: {})", if forced { "generic LDS" } else { "arch-selected" });
    let scale = 1.0f64 / (HD as f64).sqrt();
    let bph = HD / 32;
    let total_bpp = NKV * bph;

    // Per-row query, per-position KV.
    let q = lcg(0xa5a5, B * NH * HD);
    let k_src: Vec<Vec<f32>> = (0..MAXS).map(|t| lcg(0x1000 + t as u64, KVDIM)).collect();
    let v_src: Vec<Vec<f32>> = (0..MAXS).map(|t| lcg(0x9000 + t as u64, KVDIM)).collect();
    let positions: [i32; B] = [50, 120, 199, 33, 175];

    let d_q = gpu.upload_f32(&q, &[B * NH * HD]).unwrap();
    let d_out = gpu.zeros(&[B * NH * HD], DType::F32).unwrap();
    // positions tensor [B] i32
    let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let d_pos = gpu.alloc_tensor(&[B], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&d_pos.buf, &pos_bytes).unwrap();

    // Flat f32 KV
    let mut kflat = vec![0f32; MAXS * KVDIM];
    let mut vflat = vec![0f32; MAXS * KVDIM];
    for t in 0..MAXS {
        kflat[t * KVDIM..(t + 1) * KVDIM].copy_from_slice(&k_src[t]);
        vflat[t * KVDIM..(t + 1) * KVDIM].copy_from_slice(&v_src[t]);
    }
    let d_kf = gpu.upload_f32(&kflat, &[MAXS * KVDIM]).unwrap();
    let d_vf = gpu.upload_f32(&vflat, &[MAXS * KVDIM]).unwrap();

    // Q8_0 KV cache
    let d_kq = gpu.alloc_tensor(&[MAXS * KVDIM], DType::Q8_0).unwrap();
    let d_vq = gpu.alloc_tensor(&[MAXS * KVDIM], DType::Q8_0).unwrap();
    let pos_all: Vec<u8> = (0..MAXS as i32).flat_map(|p| p.to_ne_bytes()).collect();
    let pos_all_t = gpu.alloc_tensor(&[MAXS], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&pos_all_t.buf, &pos_all).unwrap();
    gpu.kv_cache_write_q8_0_batched(&d_kq, &d_kf, &pos_all_t, NKV, HD, MAXS).unwrap();
    gpu.kv_cache_write_q8_0_batched(&d_vq, &d_vf, &pos_all_t, NKV, HD, MAXS).unwrap();
    // Only positions [0, max_pos] are ever read; download just those blocks.
    let max_pos = *positions.iter().max().unwrap() as usize + 1;
    let kqb = gpu.download_raw(&d_kq, max_pos * total_bpp * 34).unwrap();
    let vqb = gpu.download_raw(&d_vq, max_pos * total_bpp * 34).unwrap();
    let deqk = |b: &[u8], t: usize, kvh: usize, d: usize| -> f64 {
        let blk = (t * total_bpp + kvh * bph + d / 32) * 34;
        f16_to_f32(u16::from_le_bytes([b[blk], b[blk + 1]])) as f64 * b[blk + 2 + d % 32] as i8 as f64
    };

    let max_ctx = *positions.iter().max().unwrap() as usize + 1;
    let mut fails = 0;

    // CPU ref for a batched attention; `bias[b]` optional per-key over the block.
    let cpu_ref = |kget: &dyn Fn(usize, usize, usize) -> f64,
                   vget: &dyn Fn(usize, usize, usize) -> f64,
                   seqfn: &dyn Fn(usize) -> usize,
                   bias: Option<&(Vec<f32>, usize, usize)>| -> Vec<f32> {
        let mut r = vec![0f32; B * NH * HD];
        for bb in 0..B {
            let sl = seqfn(bb);
            for h in 0..NH {
                let kvh = h / (NH / NKV);
                let mut sc = vec![0f64; sl];
                let mut mx = f64::MIN;
                for t in 0..sl {
                    let mut dot = 0f64;
                    for d in 0..HD { dot += q[(bb * NH + h) * HD + d] as f64 * kget(t, kvh, d); }
                    let mut s = dot * scale;
                    if let Some((bv, bstart, bcols)) = bias {
                        if t >= *bstart { s += bv[bb * bcols + (t - bstart)] as f64; }
                    }
                    sc[t] = s;
                    mx = mx.max(s);
                }
                let mut den = 0f64;
                for x in sc.iter_mut() { *x = (*x - mx).exp(); den += *x; }
                for d in 0..HD {
                    let mut acc = 0f64;
                    for t in 0..sl { acc += sc[t] * vget(t, kvh, d); }
                    r[(bb * NH + h) * HD + d] = (acc / den) as f32;
                }
            }
        }
        r
    };
    let cmp = |gpu: &Gpu, out: &hipfire_rdna::GpuTensor, refv: &[f32], name: &str, fails: &mut i32| {
        let got = gpu.download_f32(out).unwrap();
        let err = got.iter().zip(refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let ok = err < 5e-4;
        if !ok { *fails += 1; }
        println!("  {name:28} max_abs_err={err:.3e} {}", if ok { "OK" } else { "FAIL" });
    };

    let kf = |t: usize, kvh: usize, d: usize| k_src[t][kvh * HD + d] as f64;
    let vf = |t: usize, kvh: usize, d: usize| v_src[t][kvh * HD + d] as f64;
    let seq_causal = |bb: usize| positions[bb] as usize + 1;

    // ── f32 batched, causal ──
    gpu.attention_f32_batched(&d_q, &d_kf, &d_vf, &d_out, &d_pos, NH, NKV, HD, MAXS, max_ctx, B).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let r = cpu_ref(&kf, &vf, &seq_causal, None);
    cmp(&gpu, &d_out, &r, "f32_batched causal", &mut fails);

    // ── q8_0 batched, causal ──
    gpu.attention_q8_0_kv_batched(&d_q, &d_kq, &d_vq, &d_out, &d_pos, NH, NKV, HD, MAXS, max_ctx, B).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let kq = |t: usize, kvh: usize, d: usize| deqk(&kqb, t, kvh, d);
    let vq = |t: usize, kvh: usize, d: usize| deqk(&vqb, t, kvh, d);
    let r = cpu_ref(&kq, &vq, &seq_causal, None);
    cmp(&gpu, &d_out, &r, "q8_0_batched causal", &mut fails);

    // ── q8_0 batched, TREE mode (exercise tree_bias path) ──
    let block_start = 8usize;
    let block_cols = 16usize;
    let seq_tree = |_b: usize| block_start + block_cols;
    let bias_vals = lcg(0x5151, B * block_cols); // arbitrary finite per-key bias
    let d_bias = gpu.upload_f32(&bias_vals, &[B * block_cols]).unwrap();
    gpu.attention_q8_0_kv_batched_masked(
        &d_q, &d_kq, &d_vq, &d_out, &d_pos, NH, NKV, HD, MAXS, block_start + block_cols, B,
        Some(&d_bias), block_start, block_cols,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let r = cpu_ref(&kq, &vq, &seq_tree, Some(&(bias_vals.clone(), block_start, block_cols)));
    cmp(&gpu, &d_out, &r, "q8_0_batched tree", &mut fails);

    if fails == 0 { println!("OK — all batched cases within tol"); }
    else { eprintln!("PARITY FAIL — {fails} case(s)"); std::process::exit(1); }
}
