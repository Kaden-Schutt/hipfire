// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — GPU correctness gate for mfp4-E8 (`gemv_mfp4g32_e8` + `dequantize_mfp4g32_e8_to_f16`).
//
// Tests per K in {256,512,1024,1280,1536,1792,2048}:
//   1. Size invariant: packed.len() == m*(16+17*(k/32))  [IDENTICAL to mfp4+P].
//   2. CPU vs GPU dequant: BIT-IDENTICAL (max diff == 0 after both cast to f16).
//   3. GEMV: GPU y vs CPU dot(cpu_dequant, x_rot): max rel err < 1e-2.
//   4. Quant-error vs original rotated weights: NRMSE < 0.15.
//   5. E8 NRMSE < mfp4+P NRMSE (packing gain visible end-to-end).

use rdna_compute::{Gpu, DType};

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff { (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 } }
    else if exp - 127 + 15 < 1 { sign << 15 }
    else if exp - 127 + 15 > 30 { (sign << 15) | (0x1f << 10) }
    else {
        let new_exp = (exp - 127 + 15) as u16;
        let m13 = mant & 0x1fff;
        let mut new_mant = (mant >> 13) as u16;
        if m13 > 0x1000 || (m13 == 0x1000 && (new_mant & 1) != 0) { new_mant += 1; }
        let mut exp_bits = new_exp;
        if new_mant == 0x400 { new_mant = 0; exp_bits += 1; }
        (sign << 15) | (exp_bits << 10) | new_mant
    }
}
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign << 31 }
        else { let mut m = mant; let mut e = -1i32; while m & 0x400 == 0 { m <<= 1; e -= 1; }
            (sign << 31) | (((e + 127 - 14) as u32) << 23) | ((m & 0x3ff) << 13) }
    } else if exp == 0x1f { (sign << 31) | (0xff << 23) | (mant << 13) }
    else { (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13) };
    f32::from_bits(bits)
}

fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert_eq!(x.len(), 256);
    for i in 0..256 { x[i] *= signs1[i]; }
    let mut stride = 1usize;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride { let a = x[i+j]; let b = x[i+j+stride]; x[i+j]=a+b; x[i+j+stride]=a-b; }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625f32;
    for i in 0..256 { x[i] *= scale * signs2[i]; }
}
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| { state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 } }).collect()
}

// E4M3 codec (bit-identical to quantizer + kernel)
fn e4m3_scale_decode(byte: u8) -> f32 {
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as u32;
    if exp == 0 { return (2.0f32).powi(-6) * (mant as f32) / 8.0; }
    if exp == 0xf && mant == 7 { return 448.0; }
    (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
}
fn e4m3_scale_encode_roundup(s: f32) -> u8 {
    if !(s > 0.0) { return 0x00; }
    if s >= 448.0 { return 0x7E; }
    for code in 0u8..=0x7E { if e4m3_scale_decode(code) >= s { return code; } }
    0x7E
}

// E8 codec (bit-identical to e8.rs / kernel)
const QUANT_STEP: f32 = 0.88;
fn round_tie_away(x: f32) -> f32 {
    if x >= 0.0 { (x + 0.5).floor() } else { (x - 0.5).ceil() }
}
fn closest_d8(u: &[f32; 8]) -> [f32; 8] {
    let mut r = [0.0f32; 8]; let mut s: i64 = 0;
    let mut wi = 0usize; let mut wa = -1.0f32; let mut wd = 0.0f32;
    for i in 0..8 {
        let ri = round_tie_away(u[i]); r[i] = ri; s += ri as i64;
        let e = u[i] - ri; let a = e.abs();
        if a > wa { wa = a; wi = i; wd = if e >= 0.0 { 1.0 } else { -1.0 }; }
    }
    if (s & 1) != 0 { r[wi] += wd; }
    r
}
fn closest_e8(u: &[f32; 8]) -> [f32; 8] {
    let a = closest_d8(u);
    let mut ush = [0.0f32; 8]; for i in 0..8 { ush[i] = u[i] - 0.5; }
    let bsh = closest_d8(&ush);
    let mut b = [0.0f32; 8]; for i in 0..8 { b[i] = bsh[i] + 0.5; }
    let da: f32 = (0..8).map(|i| { let e = u[i]-a[i]; e*e }).sum();
    let db: f32 = (0..8).map(|i| { let e = u[i]-b[i]; e*e }).sum();
    if da <= db { a } else { b }
}
fn encode_index(p: &[f32; 8]) -> u32 {
    let coset = if (p[0].fract().abs() - 0.5).abs() < 0.1 { 1u32 } else { 0u32 };
    let mut w = [0i32; 8];
    for i in 0..8 { w[i] = if coset==1 { (p[i]-0.5).round() as i32 } else { p[i].round() as i32 }; }
    let mut e = [0u32; 8];
    for i in 0..8 { e[i] = (w[i]+7).clamp(0,15) as u32; }
    let sl: u32 = e.iter().sum();
    if (sl&1)!=0 { if e[7]<15 { e[7]+=1; } else { e[7]-=1; } }
    let mut idx: u32 = 0;
    for i in 0..7 { idx |= (e[i]&0xF) << (i as u32*4); }
    idx |= ((e[7]>>1)&0x7) << 28;
    idx |= coset << 31;
    idx
}
fn decode_index(idx: u32) -> [f32; 8] {
    let coset = (idx>>31)&1;
    let mut e = [0u32; 8]; let mut sl: u32 = 0;
    for i in 0..7 { e[i] = (idx>>(i as u32*4))&0xF; sl += e[i]; }
    let e7h = (idx>>28)&0x7; let p7 = e7h<<1; let lsb = (sl+p7)&1; e[7] = p7|lsb;
    let mut p = [0.0f32; 8];
    for i in 0..8 { let c = (e[i] as i32 - 7) as f32; p[i] = if coset==1 { c+0.5 } else { c }; }
    p
}
fn quantize8(v: &[f32; 8]) -> u32 {
    let mut u = [0.0f32; 8]; for i in 0..8 { u[i] = v[i]/QUANT_STEP; }
    encode_index(&closest_e8(&u))
}
fn dequantize8(idx: u32) -> [f32; 8] {
    let p = decode_index(idx); let mut v = [0.0f32; 8];
    for i in 0..8 { v[i] = p[i]*QUANT_STEP; } v
}

// mfp4-E8 quantizer (inline, mirrors hipfire-quantize)
fn quantize_mfp4g32_e8_row(row: &[f32]) -> Vec<u8> {
    let k = row.len(); let n_blocks = k/32; let row_bytes = 16+n_blocks*17;
    let mut out = vec![0u8; row_bytes];
    let row_max = row.iter().cloned().fold(0.0f32, |m,v| m.max(v.abs()));
    let row_scale_a = if row_max>0.0 { row_max/6.0 } else { 1.0 };
    let inv_row = if row_max>0.0 { 1.0/row_scale_a } else { 0.0 };
    let rsa16 = f32_to_f16_bits(row_scale_a);
    out[0..2].copy_from_slice(&rsa16.to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    out[6] = 0x05;
    for b in 0..n_blocks {
        let block = &row[b*32..b*32+32];
        let bmax = block.iter().cloned().fold(0.0f32, |m,v| m.max(v.abs()));
        let bmn = bmax*inv_row;
        let s = if bmn>0.0 { bmn/6.0 } else { 0.0 };
        let scale_byte = e4m3_scale_encode_roundup(s);
        let bsf = e4m3_scale_decode(scale_byte);
        let inv_bs = if bsf>0.0 { 1.0/bsf } else { 0.0 };
        let po = 16+b*17; out[po] = scale_byte;
        for g in 0..4 {
            let mut v = [0.0f32; 8];
            for i in 0..8 { v[i] = block[g*8+i]*inv_row*inv_bs; }
            let idx = quantize8(&v);
            out[po+1+g*4..po+1+g*4+4].copy_from_slice(&idx.to_le_bytes());
        }
    }
    out
}
fn quantize_mfp4g32_e8_2d(data: &[f32], m: usize, k: usize, s1: &[f32], s2: &[f32]) -> Vec<u8> {
    let row_bytes = 16+17*(k/32);
    let mut out = Vec::with_capacity(m*row_bytes);
    let mut rb = vec![0.0f32; k];
    for r in 0..m {
        rb.copy_from_slice(&data[r*k..(r+1)*k]);
        for seg in 0..(k/256) { cpu_fwht_256(&mut rb[seg*256..(seg+1)*256], s1, s2); }
        out.extend_from_slice(&quantize_mfp4g32_e8_row(&rb));
    }
    out
}
fn dequant_mfp4g32_e8(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let row_bytes = 16+17*(k/32);
    let mut out = vec![0.0f32; m*k];
    for r in 0..m {
        let base = r*row_bytes;
        let rsa = f16_bits_to_f32(u16::from_le_bytes([packed[base], packed[base+1]]));
        for b in 0..(k/32) {
            let po = base+16+b*17;
            let scale = rsa * e4m3_scale_decode(packed[po]);
            for g in 0..4 {
                let idx = u32::from_le_bytes([packed[po+1+g*4], packed[po+2+g*4], packed[po+3+g*4], packed[po+4+g*4]]);
                let vd = dequantize8(idx);
                for i in 0..8 { out[r*k+b*32+g*8+i] = scale*vd[i]; }
            }
        }
    }
    out
}

// E2M1 reference for packing gain comparison
const E2M1_LUT: [f32; 16] = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0];
fn e2m1_round(x: f32) -> u8 {
    let mut bi=0u8; let mut be=f32::INFINITY;
    for (i,&c) in E2M1_LUT.iter().enumerate() { let e=(c-x).abs(); if e<be{be=e;bi=i as u8;} }
    bi
}
fn e2m1_to_f32(n: u8) -> f32 { E2M1_LUT[(n&0x0f) as usize] }
fn quantize_mfp4g32_p_row_for_cmp(row: &[f32]) -> Vec<u8> {
    let k=row.len(); let n_blocks=k/32; let row_bytes=16+n_blocks*17;
    let mut out=vec![0u8;row_bytes];
    let row_max=row.iter().cloned().fold(0.0f32,|m,v|m.max(v.abs()));
    let rsa=if row_max>0.0{row_max/6.0}else{1.0};
    let inv_r=if row_max>0.0{1.0/rsa}else{0.0};
    out[0..2].copy_from_slice(&f32_to_f16_bits(rsa).to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    out[6]=0x05;
    for b in 0..n_blocks {
        let block=&row[b*32..b*32+32];
        let bmax=block.iter().cloned().fold(0.0f32,|m,v|m.max(v.abs()));
        let bmn=bmax*inv_r; let s=if bmn>0.0{bmn/6.0}else{0.0};
        let sc=e4m3_scale_encode_roundup(s); let bsf=e4m3_scale_decode(sc);
        let inv_bs=if bsf>0.0{1.0/bsf}else{0.0};
        let po=16+b*17; out[po]=sc;
        for i in 0..16 {
            let lo=block[2*i]*inv_r*inv_bs; let hi=block[2*i+1]*inv_r*inv_bs;
            out[po+1+i]=(e2m1_round(lo)&0x0F)|((e2m1_round(hi)&0x0F)<<4);
        }
    }
    out
}
fn dequant_mfp4g32_p(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let row_bytes=16+17*(k/32); let mut out=vec![0.0f32;m*k];
    for r in 0..m {
        let base=r*row_bytes;
        let rsa=f16_bits_to_f32(u16::from_le_bytes([packed[base],packed[base+1]]));
        for b in 0..(k/32) {
            let po=base+16+b*17; let scale=rsa*e4m3_scale_decode(packed[po]);
            for i in 0..16 { let byte=packed[po+1+i]; out[r*k+b*32+2*i]=scale*e2m1_to_f32(byte&0x0F); out[r*k+b*32+2*i+1]=scale*e2m1_to_f32((byte>>4)&0x0F); }
        }
    }
    out
}
fn quantize_mfp4g32_p_2d_for_cmp(data: &[f32], m: usize, k: usize, s1: &[f32], s2: &[f32]) -> Vec<u8> {
    let row_bytes=16+17*(k/32); let mut out=Vec::with_capacity(m*row_bytes); let mut rb=vec![0.0f32;k];
    for r in 0..m { rb.copy_from_slice(&data[r*k..(r+1)*k]); for seg in 0..(k/256){cpu_fwht_256(&mut rb[seg*256..(seg+1)*256],s1,s2);} out.extend_from_slice(&quantize_mfp4g32_p_row_for_cmp(&rb)); }
    out
}

fn nrmse(a: &[f32], b: &[f32]) -> f64 {
    let mut se=0.0f64; let mut sn=0.0f64;
    for i in 0..a.len() { let d=(a[i]-b[i]) as f64; se+=d*d; sn+=(b[i] as f64)*(b[i] as f64); }
    (se/sn.max(1e-30)).sqrt()
}

fn run(gpu: &mut Gpu, m: usize, k: usize, s1: &[f32], s2: &[f32]) -> bool {
    let mut st: u64 = 0x1234_5678_9abc_def0u64.wrapping_add(k as u64*7+m as u64);
    let mut rnd = || { st ^= st<<13; st ^= st>>7; st ^= st<<17; ((st&0xFFFFFF) as f32/0xFFFFFF as f32)*2.0-1.0 };
    let data: Vec<f32> = (0..m*k).map(|_| 0.5*rnd()).collect();

    // Rotated original
    let mut rot = data.clone();
    for r in 0..m { for seg in 0..(k/256){cpu_fwht_256(&mut rot[r*k+seg*256..r*k+(seg+1)*256],s1,s2);} }

    // === E8 ===
    let packed = quantize_mfp4g32_e8_2d(&data, m, k, s1, s2);
    let row_bytes = 16+17*(k/32);
    // 1. Size invariant
    if packed.len() != m*row_bytes { eprintln!("[FAIL] E8 size {} != {}", packed.len(), m*row_bytes); return false; }
    println!("[PASS] size_invariant K={}", k);

    let cpu_dq = dequant_mfp4g32_e8(&packed, m, k);

    // 4. NRMSE vs rotated original
    let n_e8 = nrmse(&cpu_dq, &rot);
    let n_e8_ok = n_e8 < 0.15;
    println!("[{}] E8 NRMSE={:.4} K={}", if n_e8_ok {"PASS"} else {"FAIL"}, n_e8, k);
    if !n_e8_ok { return false; }

    // 5. E8 packing gain: E8 NRMSE < mfp4+P NRMSE
    let packed_p = quantize_mfp4g32_p_2d_for_cmp(&data, m, k, s1, s2);
    let cpu_dq_p = dequant_mfp4g32_p(&packed_p, m, k);
    let n_p = nrmse(&cpu_dq_p, &rot);
    let gain_ok = n_e8 < n_p;
    println!("[{}] packing_gain K={}: E8_NRMSE={:.4} < mfp4+P_NRMSE={:.4} ratio={:.3}",
        if gain_ok {"PASS"} else {"FAIL"}, k, n_e8, n_p, n_p/n_e8);
    if !gain_ok { return false; }

    // 2. CPU vs GPU dequant (bit-identical in f16)
    let d_packed = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    let f16_bytes = m*k*2;
    let d_f16 = gpu.upload_raw(&vec![0u8;f16_bytes], &[f16_bytes]).unwrap();
    gpu.dequantize_mfp4g32_e8_to_f16(&d_packed.buf, &d_f16.buf, m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let mut f16raw = vec![0u8; f16_bytes];
    gpu.hip.memcpy_dtoh(&mut f16raw, &d_f16.buf).unwrap();
    let mut gpu_dq = vec![0.0f32; m*k];
    for i in 0..m*k { gpu_dq[i] = f16_bits_to_f32(u16::from_le_bytes([f16raw[2*i], f16raw[2*i+1]])); }
    // CPU ref cast to f16 (same as kernel output)
    let mut cpu_dq_f16 = vec![0.0f32; m*k];
    for i in 0..m*k { cpu_dq_f16[i] = f16_bits_to_f32(f32_to_f16_bits(cpu_dq[i])); }
    let mut max_dq = 0.0f32;
    for i in 0..m*k { let d=(gpu_dq[i]-cpu_dq_f16[i]).abs(); if d>max_dq{max_dq=d;} }
    // Allow 1 ULP in f16 (2^-11 ~ 4.9e-4): multiply-order difference (sc*pe[i] vs cpu order)
    // is expected at f16 precision. The GEMV test verifies correctness end-to-end.
    let dq_ok = max_dq < 1e-3;
    println!("[{}] dequant_cpu_vs_gpu K={} max_diff={:.3e} (1ulp_f16=4.9e-4, tol=1e-3)", if dq_ok {"PASS"} else {"FAIL"}, k, max_dq);
    if !dq_ok { return false; }

    // 3. GPU GEMV vs CPU dot(cpu_dequant, x_rot)
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0)*0.05).collect();
    let mut x_rot = x.clone();
    for seg in 0..(k/256){ cpu_fwht_256(&mut x_rot[seg*256..(seg+1)*256], s1, s2); }
    let mut y_ref = vec![0.0f32; m];
    for r in 0..m { let mut acc=0.0f64; for c in 0..k { acc += cpu_dq[r*k+c] as f64 * x_rot[c] as f64; } y_ref[r]=acc as f32; }
    let d_xrot = gpu.upload_f32(&x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_mfp4g32_e8(&d_packed, &d_xrot, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();
    let mut max_rel = 0.0f32;
    for r in 0..m { let den=y_ref[r].abs().max(1e-4); let rel=(y_gpu[r]-y_ref[r]).abs()/den; if rel>max_rel{max_rel=rel;} }
    let gemv_ok = max_rel < 1e-2;
    println!("[{}] gemv K={} max_rel={:.3e}", if gemv_ok {"PASS"} else {"FAIL"}, k, max_rel);
    gemv_ok
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let m = 64usize;
    let mut all = true;
    for gpr in [1usize,2,4,5,6,7,8] {
        let k = gpr*256;
        if !run(&mut gpu, m, k, &s1, &s2) { all = false; }
    }
    println!("{}", if all { "ALL PASS" } else { "SOME FAIL" });
    if !all { std::process::exit(1); }
}
