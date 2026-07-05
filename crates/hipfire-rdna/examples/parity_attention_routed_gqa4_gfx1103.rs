// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for the gfx1103 no-LDS GQA-4 and session-routed batched attention
//! kernels: attention_f32_batched_gqa4, attention_f32_routed_batched,
//! attention_q8_0_routed_batched. Each compared against an f64 CPU reference.
//! Default runs the gfx1103 no-LDS kernels; HIPFIRE_FORCE_GENERIC=1 the generic.

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
    let v = if e == 0 { (f as f32) * 2f32.powi(-24) }
        else if e == 0x1f { if f == 0 { f32::INFINITY } else { f32::NAN } }
        else { (1.0 + f as f32 / 1024.0) * 2f32.powi(e as i32 - 15) };
    if s == 1 { -v } else { v }
}

const NH: usize = 8;
const NKV: usize = 2;
const HD: usize = 128;
const MAXS: usize = 256;
const KVDIM: usize = NKV * HD;

fn main() {
    let forced = std::env::var("HIPFIRE_FORCE_GENERIC").is_ok();
    let mut gpu = Gpu::init().expect("gpu init");
    println!("force_generic={forced} (path: {})", if forced { "generic LDS" } else { "arch-selected" });
    let scale = 1.0f64 / (HD as f64).sqrt();
    let bph = HD / 32;
    let total_bpp = NKV * bph;
    let mut fails = 0;

    let q = lcg(0xa5a5, 8 * NH * HD); // up to batch 8
    let cmp = |gpu: &Gpu, out: &hipfire_rdna::GpuTensor, refv: &[f32], name: &str, fails: &mut i32| {
        let got = gpu.download_f32(out).unwrap();
        let err = got.iter().zip(refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let ok = err < 5e-4;
        if !ok { *fails += 1; }
        println!("  {name:26} max_abs_err={err:.3e} {}", if ok { "OK" } else { "FAIL" });
    };

    // ── GQA-4 (batch B4, kv_group=4) ─────────────────────────────────────
    {
        let b4 = 3usize;
        let positions: Vec<i32> = vec![40, 150, 199];
        let k = lcg(0xc3, MAXS * KVDIM);
        let v = lcg(0x96, MAXS * KVDIM);
        let dq = gpu.upload_f32(&q[..b4 * NH * HD], &[b4 * NH * HD]).unwrap();
        let dk = gpu.upload_f32(&k, &[MAXS * KVDIM]).unwrap();
        let dv = gpu.upload_f32(&v, &[MAXS * KVDIM]).unwrap();
        let dout = gpu.zeros(&[b4 * NH * HD], DType::F32).unwrap();
        let posb: Vec<u8> = positions.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let dpos = gpu.alloc_tensor(&[b4], DType::F32).unwrap();
        gpu.hip.memcpy_htod(&dpos.buf, &posb).unwrap();
        let max_ctx = *positions.iter().max().unwrap() as usize + 1;
        gpu.attention_f32_batched_gqa4(&dq, &dk, &dv, &dout, &dpos, NH, NKV, HD, max_ctx, b4).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let mut r = vec![0f32; b4 * NH * HD];
        for bb in 0..b4 {
            let sl = positions[bb] as usize + 1;
            for h in 0..NH {
                let kvh = h / (NH / NKV);
                let mut sc = vec![0f64; sl];
                let mut mx = f64::MIN;
                for t in 0..sl {
                    let mut dot = 0f64;
                    for d in 0..HD { dot += q[(bb * NH + h) * HD + d] as f64 * k[t * KVDIM + kvh * HD + d] as f64; }
                    sc[t] = dot * scale; mx = mx.max(sc[t]);
                }
                let mut den = 0f64;
                for x in sc.iter_mut() { *x = (*x - mx).exp(); den += *x; }
                for d in 0..HD {
                    let mut acc = 0f64;
                    for t in 0..sl { acc += sc[t] * v[t * KVDIM + kvh * HD + d] as f64; }
                    r[(bb * NH + h) * HD + d] = (acc / den) as f32;
                }
            }
        }
        cmp(&gpu, &dout, &r, "f32_batched_gqa4", &mut fails);
    }

    // ── routed batched (2 sessions) ──────────────────────────────────────
    // session s has its own K/V; row b reads session row_session[b].
    let n_sess = 2usize;
    let b = 4usize;
    let row_session: Vec<i32> = vec![0, 1, 1, 0];
    let positions: Vec<i32> = vec![30, 120, 199, 77];
    let max_ctx = *positions.iter().max().unwrap() as usize + 1;
    let ksess: Vec<Vec<f32>> = (0..n_sess).map(|s| lcg(0x100 + s as u64, MAXS * KVDIM)).collect();
    let vsess: Vec<Vec<f32>> = (0..n_sess).map(|s| lcg(0x200 + s as u64, MAXS * KVDIM)).collect();
    let dq = gpu.upload_f32(&q[..b * NH * HD], &[b * NH * HD]).unwrap();
    let dout = gpu.zeros(&[b * NH * HD], DType::F32).unwrap();
    let posb: Vec<u8> = positions.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dpos = gpu.alloc_tensor(&[b], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&dpos.buf, &posb).unwrap();
    let rsb: Vec<u8> = row_session.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let drs = gpu.alloc_tensor(&[b], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&drs.buf, &rsb).unwrap();

    let cpu_routed = |kget: &dyn Fn(usize, usize, usize, usize) -> f64,
                      vget: &dyn Fn(usize, usize, usize, usize) -> f64| -> Vec<f32> {
        let mut r = vec![0f32; b * NH * HD];
        for bb in 0..b {
            let s = row_session[bb] as usize;
            let sl = positions[bb] as usize + 1;
            for h in 0..NH {
                let kvh = h / (NH / NKV);
                let mut sc = vec![0f64; sl];
                let mut mx = f64::MIN;
                for t in 0..sl {
                    let mut dot = 0f64;
                    for d in 0..HD { dot += q[(bb * NH + h) * HD + d] as f64 * kget(s, t, kvh, d); }
                    sc[t] = dot * scale; mx = mx.max(sc[t]);
                }
                let mut den = 0f64;
                for x in sc.iter_mut() { *x = (*x - mx).exp(); den += *x; }
                for d in 0..HD {
                    let mut acc = 0f64;
                    for t in 0..sl { acc += sc[t] * vget(s, t, kvh, d); }
                    r[(bb * NH + h) * HD + d] = (acc / den) as f32;
                }
            }
        }
        r
    };

    // f32 routed
    {
        let dks: Vec<_> = ksess.iter().map(|k| gpu.upload_f32(k, &[MAXS * KVDIM]).unwrap()).collect();
        let dvs: Vec<_> = vsess.iter().map(|v| gpu.upload_f32(v, &[MAXS * KVDIM]).unwrap()).collect();
        let kptrs: Vec<u8> = dks.iter().flat_map(|t| (t.buf.as_ptr() as usize as u64).to_ne_bytes()).collect();
        let vptrs: Vec<u8> = dvs.iter().flat_map(|t| (t.buf.as_ptr() as usize as u64).to_ne_bytes()).collect();
        let dkp = gpu.alloc_tensor(&[n_sess * 8], DType::Raw).unwrap();
        let dvp = gpu.alloc_tensor(&[n_sess * 8], DType::Raw).unwrap();
        gpu.hip.memcpy_htod(&dkp.buf, &kptrs).unwrap();
        gpu.hip.memcpy_htod(&dvp.buf, &vptrs).unwrap();
        gpu.attention_f32_routed_batched(&dq, &dkp, &dvp, &dout, &drs, &dpos, 1, 0, NH, NKV, HD, MAXS, max_ctx, b).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let kf = |s: usize, t: usize, kvh: usize, d: usize| ksess[s][t * KVDIM + kvh * HD + d] as f64;
        let vf = |s: usize, t: usize, kvh: usize, d: usize| vsess[s][t * KVDIM + kvh * HD + d] as f64;
        let r = cpu_routed(&kf, &vf);
        cmp(&gpu, &dout, &r, "f32_routed_batched", &mut fails);
    }

    // q8_0 routed
    {
        let mut kbytes = vec![Vec::new(); n_sess];
        let mut vbytes = vec![Vec::new(); n_sess];
        let pos_all: Vec<u8> = (0..MAXS as i32).flat_map(|p| p.to_ne_bytes()).collect();
        let pos_all_t = gpu.alloc_tensor(&[MAXS], DType::F32).unwrap();
        gpu.hip.memcpy_htod(&pos_all_t.buf, &pos_all).unwrap();
        let mut dkq = Vec::new();
        let mut dvq = Vec::new();
        for s in 0..n_sess {
            let dkf = gpu.upload_f32(&ksess[s], &[MAXS * KVDIM]).unwrap();
            let dvf = gpu.upload_f32(&vsess[s], &[MAXS * KVDIM]).unwrap();
            let dk = gpu.alloc_tensor(&[MAXS * KVDIM], DType::Q8_0).unwrap();
            let dv = gpu.alloc_tensor(&[MAXS * KVDIM], DType::Q8_0).unwrap();
            gpu.kv_cache_write_q8_0_batched(&dk, &dkf, &pos_all_t, NKV, HD, MAXS).unwrap();
            gpu.kv_cache_write_q8_0_batched(&dv, &dvf, &pos_all_t, NKV, HD, MAXS).unwrap();
            kbytes[s] = gpu.download_raw(&dk, max_ctx * total_bpp * 34).unwrap();
            vbytes[s] = gpu.download_raw(&dv, max_ctx * total_bpp * 34).unwrap();
            dkq.push(dk);
            dvq.push(dv);
        }
        let kptrs: Vec<u8> = dkq.iter().flat_map(|t| (t.buf.as_ptr() as usize as u64).to_ne_bytes()).collect();
        let vptrs: Vec<u8> = dvq.iter().flat_map(|t| (t.buf.as_ptr() as usize as u64).to_ne_bytes()).collect();
        let dkp = gpu.alloc_tensor(&[n_sess * 8], DType::Raw).unwrap();
        let dvp = gpu.alloc_tensor(&[n_sess * 8], DType::Raw).unwrap();
        gpu.hip.memcpy_htod(&dkp.buf, &kptrs).unwrap();
        gpu.hip.memcpy_htod(&dvp.buf, &vptrs).unwrap();
        gpu.attention_q8_0_routed_batched(&dq, &dkp, &dvp, &dout, &drs, &dpos, 1, 0, NH, NKV, HD, MAXS, max_ctx, b).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let deq = move |bytes: &[u8], t: usize, kvh: usize, d: usize| -> f64 {
            let blk = (t * total_bpp + kvh * bph + d / 32) * 34;
            f16_to_f32(u16::from_le_bytes([bytes[blk], bytes[blk + 1]])) as f64 * bytes[blk + 2 + d % 32] as i8 as f64
        };
        let kq = |s: usize, t: usize, kvh: usize, d: usize| deq(&kbytes[s], t, kvh, d);
        let vq = |s: usize, t: usize, kvh: usize, d: usize| deq(&vbytes[s], t, kvh, d);
        let r = cpu_routed(&kq, &vq);
        cmp(&gpu, &dout, &r, "q8_0_routed_batched", &mut fails);
    }

    if fails == 0 { println!("OK — all routed/gqa4 cases within tol"); }
    else { eprintln!("PARITY FAIL — {fails} case(s)"); std::process::exit(1); }
}
