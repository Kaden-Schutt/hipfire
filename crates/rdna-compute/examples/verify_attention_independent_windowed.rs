// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Low-level gate for `attention_q8_0_kv_independent_masked_windowed`:
//   - B=1 window=0 vs existing `attention_q8_0_kv_independent`
//   - B=1 window=2048 vs existing `attention_q8_0_kv_swa`
//   - Positions 0, 1, 2047, 2048, 2049, and a long-context point
//   - B=3 mask 0b101 matches two isolated references; inactive out/KV canaries hold
//
// Exit 0 = all pass. Parent runs this; skip if no GPU.

use rdna_compute::{DType, Gpu};

const NH: usize = 8;
const NKV: usize = 2;
const HD: usize = 128;
const BLK: usize = 34;
const WINDOW: usize = 2048;
const LANE_CAP: usize = 4096;
const LONG_POS: usize = 3071;

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp_f32 == 0 {
        return sign;
    }
    if exp_f32 == 0xff {
        return sign | 0x7c00 | if mant != 0 { 1 } else { 0 };
    }
    let exp = exp_f32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn bytes_per_pos() -> usize {
    NKV * (HD / 32) * BLK
}

fn fill_lane_kv(capacity: usize, seed: u32) -> Vec<u8> {
    let bpp = bytes_per_pos();
    let mut kv = vec![0u8; capacity * bpp];
    let mut s = seed;
    let mut prng = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s
    };
    for pos in 0..capacity {
        for bi in 0..(NKV * (HD / 32)) {
            let off = pos * bpp + bi * BLK;
            let sf = 0.01 + (prng() as f32 / u32::MAX as f32) * 0.04;
            let sb = f32_to_f16_bits(sf);
            kv[off] = (sb & 0xFF) as u8;
            kv[off + 1] = (sb >> 8) as u8;
            for j in 0..32 {
                kv[off + 2 + j] = ((prng() as i32 % 255) - 127) as i8 as u8;
            }
        }
    }
    kv
}

fn make_q(seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..NH * HD)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s % 200) as f32 - 100.0) * 0.001
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> rdna_compute::GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    gpu.upload_raw(bytes, &[data.len()]).expect("upload i32")
}

fn bit_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn check_match(label: &str, got: &[f32], refv: &[f32], ok: &mut bool) {
    if bit_equal(got, refv) {
        println!("  PASS {label}: bit-exact");
        return;
    }
    let mad = max_abs_diff(got, refv);
    let mut mrel = 0.0f32;
    for (g, r) in got.iter().zip(refv) {
        let denom = r.abs().max(1e-6);
        mrel = mrel.max((g - r).abs() / denom);
    }
    let ag = argmax(got);
    let ar = argmax(refv);
    if mad < 1e-5 && ag == ar {
        println!("  PASS {label}: max_abs={mad:.3e} max_rel={mrel:.3e} argmax={ag}");
        return;
    }
    eprintln!("  FAIL {label}: max_abs={mad:.6e} max_rel={mrel:.6e} argmax got={ag} ref={ar}");
    *ok = false;
}

fn run_b1_windowed(gpu: &mut Gpu, pos: usize, window: usize, q: &[f32], kv: &[u8]) -> Vec<f32> {
    let q_t = gpu.upload_f32(q, &[NH * HD]).expect("q");
    let k_t = gpu.upload_raw(kv, &[kv.len()]).expect("k");
    let v_t = gpu.upload_raw(kv, &[kv.len()]).expect("v");
    let out = gpu.zeros(&[NH * HD], DType::F32).expect("out");
    let pos_t = upload_i32(gpu, &[pos as i32]);
    gpu.attention_q8_0_kv_independent_masked_windowed(
        &q_t,
        &k_t,
        &v_t,
        &out,
        &pos_t,
        NH,
        NKV,
        HD,
        LANE_CAP,
        pos + 1,
        1,
        0b1,
        window,
    )
    .expect("masked windowed B=1");
    gpu.download_f32(&out).expect("dl")
}

fn run_b1_independent(gpu: &mut Gpu, pos: usize, q: &[f32], kv: &[u8]) -> Vec<f32> {
    let q_t = gpu.upload_f32(q, &[NH * HD]).expect("q");
    let k_t = gpu.upload_raw(kv, &[kv.len()]).expect("k");
    let v_t = gpu.upload_raw(kv, &[kv.len()]).expect("v");
    let out = gpu.zeros(&[NH * HD], DType::F32).expect("out");
    let pos_t = upload_i32(gpu, &[pos as i32]);
    gpu.attention_q8_0_kv_independent(
        &q_t,
        &k_t,
        &v_t,
        &out,
        &pos_t,
        NH,
        NKV,
        HD,
        LANE_CAP,
        pos + 1,
        1,
    )
    .expect("independent B=1");
    gpu.download_f32(&out).expect("dl")
}

fn run_b1_swa(gpu: &mut Gpu, pos: usize, window: usize, q: &[f32], kv: &[u8]) -> Vec<f32> {
    let q_t = gpu.upload_f32(q, &[NH * HD]).expect("q");
    let k_t = gpu.upload_raw(kv, &[kv.len()]).expect("k");
    let v_t = gpu.upload_raw(kv, &[kv.len()]).expect("v");
    let out = gpu.zeros(&[NH * HD], DType::F32).expect("out");
    let pos_t = upload_i32(gpu, &[pos as i32]);
    gpu.attention_q8_0_kv_swa(
        &q_t,
        &k_t,
        &v_t,
        &out,
        &pos_t.buf,
        pos + 1,
        NH,
        NKV,
        HD,
        LANE_CAP,
        window,
    )
    .expect("swa B=1");
    gpu.download_f32(&out).expect("dl")
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };

    let max_cap = gpu.attention_q8_0_kv_independent_max_lane_capacity(HD);
    if max_cap < LANE_CAP {
        eprintln!("SKIP: device max lane capacity {max_cap} < {LANE_CAP}");
        return;
    }

    let mut ok = true;
    let q = make_q(7);
    let kv = fill_lane_kv(LANE_CAP, 42);
    let positions = [0usize, 1, 2047, 2048, 2049, LONG_POS];

    println!("=== B=1 window=0 vs attention_q8_0_kv_independent ===");
    for &pos in &positions {
        let got = run_b1_windowed(&mut gpu, pos, 0, &q, &kv);
        let refv = run_b1_independent(&mut gpu, pos, &q, &kv);
        check_match(&format!("pos={pos} window=0"), &got, &refv, &mut ok);
    }

    println!("=== B=1 window={WINDOW} vs attention_q8_0_kv_swa ===");
    for &pos in &positions {
        let got = run_b1_windowed(&mut gpu, pos, WINDOW, &q, &kv);
        let refv = run_b1_swa(&mut gpu, pos, WINDOW, &q, &kv);
        check_match(&format!("pos={pos} window={WINDOW}"), &got, &refv, &mut ok);
    }

    // B=3 mask 0b101: active lanes 0 and 2; lane 1 inactive with canaries.
    println!("=== B=3 mask=0b101 isolation + canaries ===");
    const B: usize = 3;
    let mask: u64 = 0b101;
    let pos0 = 2048usize;
    let pos2 = 2500usize;
    let q0 = make_q(11);
    let q2 = make_q(13);
    let kv0 = fill_lane_kv(LANE_CAP, 100);
    let kv2 = fill_lane_kv(LANE_CAP, 200);
    let bpp = bytes_per_pos();
    let lane_bytes = LANE_CAP * bpp;

    // Canary patterns for inactive lane 1 out + KV.
    let canary_out: Vec<f32> = (0..NH * HD).map(|i| 1234.5 + i as f32 * 0.001).collect();
    let canary_kv: Vec<u8> = (0..lane_bytes)
        .map(|i| (0xA5u8).wrapping_add(i as u8))
        .collect();

    let mut q_batch = vec![0.0f32; B * NH * HD];
    q_batch[..NH * HD].copy_from_slice(&q0);
    // hole lane 1 stays zero
    q_batch[2 * NH * HD..].copy_from_slice(&q2);

    let mut k_batch = vec![0u8; B * lane_bytes];
    let mut v_batch = vec![0u8; B * lane_bytes];
    k_batch[..lane_bytes].copy_from_slice(&kv0);
    v_batch[..lane_bytes].copy_from_slice(&kv0);
    k_batch[lane_bytes..2 * lane_bytes].copy_from_slice(&canary_kv);
    v_batch[lane_bytes..2 * lane_bytes].copy_from_slice(&canary_kv);
    k_batch[2 * lane_bytes..].copy_from_slice(&kv2);
    v_batch[2 * lane_bytes..].copy_from_slice(&kv2);

    let mut out_host = vec![0.0f32; B * NH * HD];
    out_host[NH * HD..2 * NH * HD].copy_from_slice(&canary_out);

    let q_t = gpu.upload_f32(&q_batch, &[B * NH * HD]).expect("qB");
    let k_t = gpu.upload_raw(&k_batch, &[k_batch.len()]).expect("kB");
    let v_t = gpu.upload_raw(&v_batch, &[v_batch.len()]).expect("vB");
    let out_t = gpu.upload_f32(&out_host, &[B * NH * HD]).expect("outB");
    let pos_t = upload_i32(&mut gpu, &[pos0 as i32, 0i32, pos2 as i32]);
    let max_ctx = pos0.max(pos2) + 1;

    gpu.attention_q8_0_kv_independent_masked_windowed(
        &q_t, &k_t, &v_t, &out_t, &pos_t, NH, NKV, HD, LANE_CAP, max_ctx, B, mask, WINDOW,
    )
    .expect("B=3 masked windowed");

    let out_got = gpu.download_f32(&out_t).expect("dl out");
    let k_got = {
        let mut buf = vec![0u8; k_batch.len()];
        gpu.hip.memcpy_dtoh(&mut buf, &k_t.buf).expect("dl k");
        buf
    };
    let v_got = {
        let mut buf = vec![0u8; v_batch.len()];
        gpu.hip.memcpy_dtoh(&mut buf, &v_t.buf).expect("dl v");
        buf
    };

    // Isolated references for active lanes (same window).
    let ref0 = run_b1_windowed(&mut gpu, pos0, WINDOW, &q0, &kv0);
    let ref2 = run_b1_windowed(&mut gpu, pos2, WINDOW, &q2, &kv2);
    // Also cross-check against SWA for lane 0 (shared layout at B=1).
    let swa0 = run_b1_swa(&mut gpu, pos0, WINDOW, &q0, &kv0);
    check_match(
        "lane0 vs isolated B=1 new",
        &out_got[..NH * HD],
        &ref0,
        &mut ok,
    );
    check_match("lane0 vs SWA", &out_got[..NH * HD], &swa0, &mut ok);
    check_match(
        "lane2 vs isolated B=1 new",
        &out_got[2 * NH * HD..],
        &ref2,
        &mut ok,
    );

    if out_got[NH * HD..2 * NH * HD] != canary_out[..] {
        eprintln!("  FAIL inactive lane1 output canary mutated");
        ok = false;
    } else {
        println!("  PASS inactive lane1 output canary unchanged");
    }
    if k_got[lane_bytes..2 * lane_bytes] != canary_kv[..]
        || v_got[lane_bytes..2 * lane_bytes] != canary_kv[..]
    {
        eprintln!("  FAIL inactive lane1 KV canary mutated");
        ok = false;
    } else {
        println!("  PASS inactive lane1 KV canaries unchanged");
    }

    if ok {
        println!("PASS: attention_q8_0_kv_independent_masked_windowed");
    } else {
        std::process::exit(1);
    }
}
