//! MQ family MSE harness — rotated-domain quality vs bpw.
//! Measures affine 4-bit, GL_CB4, GL_CB3, GL_CB2, GL_CB1 on real post-FWHT weights.
//! + RMS-vs-LS paired columns for every GL codebook (LS = s* = dot(w,c)/dot(c,c), fp16-rounded).
//! Template: crates/hipfire-quantize/examples/poly_fit.rs
//! Run: cargo run -q -p hipfire-quantize --example mq_family_mse --release

use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::SafetensorsFile;
use std::path::Path;

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}
fn cpu_fwht_256(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= s1[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    for i in 0..256 {
        x[i] *= 0.0625 * s2[i];
    }
}
fn cpu_fwht_1024(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert_eq!(x.len(), 1024);
    assert_eq!(s1.len(), 1024);
    assert_eq!(s2.len(), 1024);
    for i in 0..1024 {
        x[i] *= s1[i];
    }
    let mut stride = 1;
    while stride < 1024 {
        let mut i = 0;
        while i < 1024 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    for i in 0..1024 {
        x[i] *= 0.03125 * s2[i];
    }
}
fn f32_to_fp16_bits(v: f32) -> u16 {
    f32_to_f16(v)
}
fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        o => panic!("{o}"),
    }
}
const GL_CB1: [f32; 2] = [-0.7978845608028654, 0.7978845608028654];
const GL_CB2: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
const GL_CB3: [f32; 8] = [
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
];
const GL_CB4: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880, 0.6568,
    0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
];
const GL_CB35: [[f32; 2]; 128] = [
    [-3.036506, 0.557011],
    [-2.902707, -0.608592],
    [-2.486214, 1.544335],
    [-2.464717, -1.467453],
    [-2.417152, -0.023972],
    [-2.295932, 0.706263],
    [-2.162351, -0.730994],
    [-1.947877, 0.306698],
    [-1.934456, -2.409281],
    [-1.895991, -0.228143],
    [-1.885129, 1.082130],
    [-1.802912, 2.411955],
    [-1.743065, -1.124332],
    [-1.706713, -1.677137],
    [-1.685550, 1.663885],
    [-1.600927, -0.650825],
    [-1.593014, 0.659777],
    [-1.544414, 0.152975],
    [-1.450453, -0.269551],
    [-1.394326, 1.156593],
    [-1.254734, -1.360016],
    [-1.243292, -0.909463],
    [-1.220646, 0.410975],
    [-1.209120, -2.058067],
    [-1.179515, -0.024031],
    [-1.150258, 1.595281],
    [-1.142204, 0.814831],
    [-1.117422, -0.488152],
    [-1.056154, 2.141912],
    [-0.932642, 1.212272],
    [-0.907086, 0.198202],
    [-0.880944, -1.642492],
    [-0.865366, -0.820827],
    [-0.861729, -0.183285],
    [-0.853414, -2.848194],
    [-0.851236, 0.573600],
    [-0.823025, 2.881643],
    [-0.817329, -1.191992],
    [-0.718535, -0.511478],
    [-0.706018, 0.898155],
    [-0.691930, 1.711770],
    [-0.605064, 0.367565],
    [-0.590630, -2.134673],
    [-0.581430, 0.061727],
    [-0.537026, 1.304575],
    [-0.505917, -0.234563],
    [-0.483035, -0.831881],
    [-0.459100, -1.613037],
    [-0.438526, -1.205364],
    [-0.430598, 0.631694],
    [-0.378955, 2.245666],
    [-0.365405, -0.493880],
    [-0.343728, 0.964808],
    [-0.286504, 0.295790],
    [-0.256216, -0.032611],
    [-0.251191, 1.744736],
    [-0.143569, 1.333993],
    [-0.127799, -1.014336],
    [-0.117265, -0.675589],
    [-0.082967, -0.312776],
    [-0.081779, -1.424501],
    [-0.081717, 0.631896],
    [-0.079977, -1.920818],
    [-0.054644, -2.570277],
    [-0.019024, 0.970040],
    [0.013778, 0.298853],
    [0.049626, -0.021871],
    [0.130522, 2.925998],
    [0.168374, 1.710445],
    [0.168862, -0.492120],
    [0.200595, -0.825208],
    [0.220961, -1.184709],
    [0.233563, 1.253369],
    [0.236452, 2.223155],
    [0.258488, 0.521816],
    [0.279151, 0.851728],
    [0.309002, 0.157230],
    [0.315732, -1.632396],
    [0.323335, -0.213913],
    [0.440193, -2.126922],
    [0.480347, -0.565665],
    [0.533287, 1.516962],
    [0.549216, -0.924366],
    [0.555097, 0.372449],
    [0.575936, -1.307392],
    [0.584788, -0.015324],
    [0.589135, 1.098487],
    [0.597830, 0.722098],
    [0.700796, -2.967724],
    [0.707100, -0.338488],
    [0.731491, 1.954455],
    [0.819098, 0.204463],
    [0.839814, -1.646037],
    [0.843593, -0.704436],
    [0.901551, 0.549277],
    [0.917409, -1.124217],
    [0.929044, 1.410108],
    [0.943199, 0.946930],
    [0.963024, -0.094088],
    [0.990743, 2.622817],
    [0.994562, -2.242123],
    [1.106773, -0.427862],
    [1.152424, 0.263105],
    [1.241635, -0.816511],
    [1.263196, 0.701724],
    [1.272961, 1.810202],
    [1.312531, -1.273730],
    [1.318652, 1.197619],
    [1.347748, -0.085024],
    [1.350326, -1.776562],
    [1.508685, 0.354619],
    [1.548087, -0.466790],
    [1.668620, 0.866390],
    [1.706376, -0.913996],
    [1.778556, -0.034955],
    [1.791853, 1.454724],
    [1.797822, -2.373145],
    [1.810372, 2.283694],
    [1.863575, -1.481466],
    [1.939231, 0.442107],
    [2.074944, -0.428416],
    [2.232601, 0.910838],
    [2.283314, -0.947652],
    [2.356225, 0.114156],
    [2.557220, 1.613256],
    [2.653772, -1.641341],
    [2.899535, -0.451693],
    [2.944135, 0.598941],
];

fn mse_gl(blocks: &[Vec<f32>], scales: &[f32], cb: &[f32]) -> f64 {
    let mut sse = 0.0f64;
    let mut n = 0usize;
    for (grp, &sc) in blocks.iter().zip(scales.iter()) {
        if sc == 0.0 {
            for &v in grp {
                sse += (v as f64) * (v as f64);
            }
            n += grp.len();
            continue;
        }
        let inv = 1.0 / sc;
        for &v in grp {
            let z = v * inv;
            let mut best = cb[0];
            let mut bd = (z - cb[0]).abs();
            for &c in cb.iter().skip(1) {
                let d = (z - c).abs();
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            let recon = sc * best;
            let e = v as f64 - recon as f64;
            sse += e * e;
        }
        n += grp.len();
    }
    sse / n as f64
}
fn mse_affine(blocks: &[Vec<f32>]) -> f64 {
    let mut sse = 0.0f64;
    let mut n = 0usize;
    for grp in blocks {
        let min = grp.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = grp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let sc = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv = if range > 0.0 { 1.0 / sc } else { 0.0 };
        for &v in grp {
            let q = ((v - min) * inv + 0.5).clamp(0.0, 15.0) as u8;
            let recon = min + q as f32 * sc;
            let e = v as f64 - recon as f64;
            sse += e * e;
        }
        n += grp.len();
    }
    sse / n as f64
}

// --- LS-aware metrics ---

#[derive(Clone, Copy)]
struct Metrics {
    mse: f64,
    retained: f64,
    norm_ratio: f64,
}

fn metrics_scalar(blocks: &[Vec<f32>], cb: &[f32], use_ls: bool, two_pass: bool) -> Metrics {
    let mut sse = 0.0f64;
    let mut dot = 0.0f64;
    let mut ss_w = 0.0f64;
    let mut ss_h = 0.0f64;
    let mut n = 0usize;
    for grp in blocks {
        let g = grp.len();
        // RMS seed
        let ss: f64 = grp.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let rms = (ss / g as f64).sqrt() as f32;
        let sbits0 = f32_to_fp16_bits(rms);
        let scale0 = f16_to_f32(sbits0);
        let inv0 = if scale0 > 0.0 { 1.0 / scale0 } else { 0.0 };
        let mut codes: Vec<u8> = Vec::with_capacity(g);
        for &v in grp.iter() {
            let z = v * inv0;
            let mut best = 0usize;
            let mut bd = (z - cb[0]).abs();
            for (k, &c) in cb.iter().enumerate().skip(1) {
                let d = (z - c).abs();
                if d < bd {
                    bd = d;
                    best = k;
                }
            }
            codes.push(best as u8);
        }
        let mut scale = scale0;
        let mut sbits = sbits0;
        let mut final_codes = codes.clone();
        if use_ls && scale0 != 0.0 {
            let mut dot_wc: f64 = 0.0;
            let mut dot_cc: f64 = 0.0;
            for i in 0..g {
                let c = cb[codes[i] as usize] as f64;
                dot_wc += grp[i] as f64 * c;
                dot_cc += c * c;
            }
            if dot_cc != 0.0 {
                let s_star = (dot_wc / dot_cc) as f32;
                if s_star.is_finite() && s_star > 0.0 {
                    let sbits1 = f32_to_fp16_bits(s_star);
                    let scale1 = f16_to_f32(sbits1);
                    if scale1 != 0.0 {
                        if sbits1 != sbits0 {
                            let inv1 = 1.0 / scale1;
                            for i in 0..g {
                                let z = grp[i] * inv1;
                                let mut best = 0usize;
                                let mut bd = (z - cb[0]).abs();
                                for (k, &c) in cb.iter().enumerate().skip(1) {
                                    let d = (z - c).abs();
                                    if d < bd {
                                        bd = d;
                                        best = k;
                                    }
                                }
                                final_codes[i] = best as u8;
                            }
                        }
                        scale = scale1;
                        sbits = sbits1;
                        if two_pass {
                            let mut dot_wc2: f64 = 0.0;
                            let mut dot_cc2: f64 = 0.0;
                            for i in 0..g {
                                let c = cb[final_codes[i] as usize] as f64;
                                dot_wc2 += grp[i] as f64 * c;
                                dot_cc2 += c * c;
                            }
                            if dot_cc2 != 0.0 {
                                let s_star2 = (dot_wc2 / dot_cc2) as f32;
                                if s_star2.is_finite() && s_star2 > 0.0 {
                                    let sbits2 = f32_to_fp16_bits(s_star2);
                                    let scale2 = f16_to_f32(sbits2);
                                    if scale2 != 0.0 {
                                        if sbits2 != sbits {
                                            let inv2 = 1.0 / scale2;
                                            for i in 0..g {
                                                let z = grp[i] * inv2;
                                                let mut best = 0usize;
                                                let mut bd = (z - cb[0]).abs();
                                                for (k, &c) in cb.iter().enumerate().skip(1) {
                                                    let d = (z - c).abs();
                                                    if d < bd {
                                                        bd = d;
                                                        best = k;
                                                    }
                                                }
                                                final_codes[i] = best as u8;
                                            }
                                        }
                                        scale = scale2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for i in 0..g {
            let recon = scale * cb[final_codes[i] as usize];
            let w = grp[i] as f64;
            let r = recon as f64;
            let e = w - r;
            sse += e * e;
            dot += w * r;
            ss_w += w * w;
            ss_h += r * r;
        }
        n += g;
    }
    let retained = if ss_w > 0.0 { dot / ss_w } else { 0.0 };
    let norm_ratio = if ss_w > 0.0 {
        (ss_h / ss_w).sqrt()
    } else {
        0.0
    };
    Metrics {
        mse: sse / n as f64,
        retained,
        norm_ratio,
    }
}

fn metrics_vq35(blocks: &[Vec<f32>], use_ls: bool, two_pass: bool) -> Metrics {
    let mut sse = 0.0f64;
    let mut dot = 0.0f64;
    let mut ss_w = 0.0f64;
    let mut ss_h = 0.0f64;
    let mut n = 0usize;
    for grp in blocks {
        let g = grp.len();
        assert_eq!(g, 256);
        let ss: f64 = grp.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let rms = (ss / 256.0).sqrt() as f32;
        let sbits0 = f32_to_fp16_bits(rms);
        let scale0 = f16_to_f32(sbits0);
        let inv0 = if scale0 > 0.0 { 1.0 / scale0 } else { 0.0 };
        let mut codes = [0u8; 128];
        for p in 0..128 {
            let z0 = grp[2 * p] * inv0;
            let z1 = grp[2 * p + 1] * inv0;
            let mut best = 0usize;
            let mut bd = (z0 - GL_CB35[0][0]).powi(2) + (z1 - GL_CB35[0][1]).powi(2);
            for (j, c) in GL_CB35.iter().enumerate().skip(1) {
                let d = (z0 - c[0]).powi(2) + (z1 - c[1]).powi(2);
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            codes[p] = best as u8;
        }
        let mut scale = scale0;
        let mut sbits = sbits0;
        let mut final_codes = codes;
        if use_ls && scale0 != 0.0 {
            let mut dot_wc: f64 = 0.0;
            let mut dot_cc: f64 = 0.0;
            for p in 0..128 {
                let c = GL_CB35[codes[p] as usize];
                dot_wc += grp[2 * p] as f64 * c[0] as f64 + grp[2 * p + 1] as f64 * c[1] as f64;
                dot_cc += (c[0] as f64) * (c[0] as f64) + (c[1] as f64) * (c[1] as f64);
            }
            if dot_cc != 0.0 {
                let s_star = (dot_wc / dot_cc) as f32;
                if s_star.is_finite() && s_star > 0.0 {
                    let sbits1 = f32_to_fp16_bits(s_star);
                    let scale1 = f16_to_f32(sbits1);
                    if scale1 != 0.0 {
                        if sbits1 != sbits0 {
                            let inv1 = 1.0 / scale1;
                            for p in 0..128 {
                                let z0 = grp[2 * p] * inv1;
                                let z1 = grp[2 * p + 1] * inv1;
                                let mut best = 0usize;
                                let mut bd =
                                    (z0 - GL_CB35[0][0]).powi(2) + (z1 - GL_CB35[0][1]).powi(2);
                                for (j, c) in GL_CB35.iter().enumerate().skip(1) {
                                    let d = (z0 - c[0]).powi(2) + (z1 - c[1]).powi(2);
                                    if d < bd {
                                        bd = d;
                                        best = j;
                                    }
                                }
                                final_codes[p] = best as u8;
                            }
                        }
                        scale = scale1;
                        sbits = sbits1;
                        if two_pass {
                            let mut dot_wc2: f64 = 0.0;
                            let mut dot_cc2: f64 = 0.0;
                            for p in 0..128 {
                                let c = GL_CB35[final_codes[p] as usize];
                                dot_wc2 += grp[2 * p] as f64 * c[0] as f64
                                    + grp[2 * p + 1] as f64 * c[1] as f64;
                                dot_cc2 +=
                                    (c[0] as f64) * (c[0] as f64) + (c[1] as f64) * (c[1] as f64);
                            }
                            if dot_cc2 != 0.0 {
                                let s_star2 = (dot_wc2 / dot_cc2) as f32;
                                if s_star2.is_finite() && s_star2 > 0.0 {
                                    let sbits2 = f32_to_fp16_bits(s_star2);
                                    let scale2 = f16_to_f32(sbits2);
                                    if scale2 != 0.0 {
                                        if sbits2 != sbits {
                                            let inv2 = 1.0 / scale2;
                                            for p in 0..128 {
                                                let z0 = grp[2 * p] * inv2;
                                                let z1 = grp[2 * p + 1] * inv2;
                                                let mut best = 0usize;
                                                let mut bd = (z0 - GL_CB35[0][0]).powi(2)
                                                    + (z1 - GL_CB35[0][1]).powi(2);
                                                for (j, c) in GL_CB35.iter().enumerate().skip(1) {
                                                    let d =
                                                        (z0 - c[0]).powi(2) + (z1 - c[1]).powi(2);
                                                    if d < bd {
                                                        bd = d;
                                                        best = j;
                                                    }
                                                }
                                                final_codes[p] = best as u8;
                                            }
                                        }
                                        scale = scale2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for p in 0..128 {
            let c = GL_CB35[final_codes[p] as usize];
            let r0 = scale * c[0];
            let r1 = scale * c[1];
            let w0 = grp[2 * p] as f64;
            let w1 = grp[2 * p + 1] as f64;
            let e0 = w0 - r0 as f64;
            let e1 = w1 - r1 as f64;
            sse += e0 * e0 + e1 * e1;
            dot += w0 * r0 as f64 + w1 * r1 as f64;
            ss_w += w0 * w0 + w1 * w1;
            ss_h += (r0 as f64) * (r0 as f64) + (r1 as f64) * (r1 as f64);
        }
        n += g;
    }
    let retained = if ss_w > 0.0 { dot / ss_w } else { 0.0 };
    let norm_ratio = if ss_w > 0.0 {
        (ss_h / ss_w).sqrt()
    } else {
        0.0
    };
    Metrics {
        mse: sse / n as f64,
        retained,
        norm_ratio,
    }
}

fn collect_blocks_256(
    f32d: &[f32],
    m: usize,
    k: usize,
    s1: &[f32],
    s2: &[f32],
    budget: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let gpr = k / 256;
    let mut groups = Vec::new();
    let mut scales = Vec::new();
    let mut cnt = 0;
    for row in 0..m {
        for g in 0..gpr {
            if cnt >= budget {
                break;
            }
            let start = row * k + g * 256;
            let mut grp = [0.0f32; 256];
            grp.copy_from_slice(&f32d[start..start + 256]);
            cpu_fwht_256(&mut grp, s1, s2);
            let ss: f64 = grp.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let rms = (ss / 256.0).sqrt() as f32;
            let sc = f16_to_f32(f32_to_fp16_bits(rms));
            groups.push(grp.to_vec());
            scales.push(sc);
            cnt += 1;
        }
        if cnt >= budget {
            break;
        }
    }
    (groups, scales)
}
fn collect_blocks_1024(
    f32d: &[f32],
    m: usize,
    k: usize,
    s1: &[f32],
    s2: &[f32],
    budget_groups: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    // budget in 1024-groups; for 256 budget we need 1/4 as many 1024 groups to cover same weight count.
    let gpr = k / 1024;
    let mut groups = Vec::new();
    let mut scales = Vec::new();
    let mut cnt = 0;
    for row in 0..m {
        for g in 0..gpr {
            if cnt >= budget_groups {
                break;
            }
            let start = row * k + g * 1024;
            let mut grp = [0.0f32; 1024];
            grp.copy_from_slice(&f32d[start..start + 1024]);
            cpu_fwht_1024(&mut grp, s1, s2);
            let ss: f64 = grp.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let rms = (ss / 1024.0).sqrt() as f32;
            let sc = f16_to_f32(f32_to_fp16_bits(rms));
            groups.push(grp.to_vec());
            scales.push(sc);
            cnt += 1;
        }
        if cnt >= budget_groups {
            break;
        }
    }
    (groups, scales)
}
fn load_tensor(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let idx_bytes = std::fs::read(dir.join("model.safetensors.index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
    let shard = idx["weight_map"][name].as_str().unwrap();
    let sf = SafetensorsFile::open(&dir.join(shard)).unwrap();
    let (meta, data) = sf.tensor_data(name).unwrap();
    (to_f32(data, &meta.dtype), meta.shape.clone())
}
fn main() {
    let dir = Path::new("/home/kaden/models/Qwen3.8-27B");
    let s1_256 = gen_fwht_signs(42, 256);
    let s2_256 = gen_fwht_signs(1042, 256);
    let s1_1024 = gen_fwht_signs(42, 1024);
    let s2_1024 = gen_fwht_signs(1042, 1024);
    let targets: Vec<(&str, &str)> = vec![
        (
            "early linear_attn out_proj (layer 0)",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
        ),
        (
            "mid mlp down_proj (layer 20)",
            "model.language_model.layers.20.mlp.down_proj.weight",
        ),
        (
            "late mlp gate_proj (layer 40)",
            "model.language_model.layers.40.mlp.gate_proj.weight",
        ),
    ];
    let budget_256 = 4096usize;
    // For 1024 grouping, use 1024 groups budget (= same total weights as 4096*256 = 1,048,576 weights => 1024 groups)
    let budget_1024 = 1024usize;
    struct Rec {
        name: &'static str,
        blocks_256: Vec<Vec<f32>>,
        scales_256: Vec<f32>,
        blocks_1024: Vec<Vec<f32>>,
        scales_1024: Vec<f32>,
    }
    let mut recs = Vec::new();
    for (label, name) in &targets {
        println!("Loading {} : {}", label, name);
        let (f32d, shape) = load_tensor(dir, name);
        let m = shape[0];
        let k = shape[1];
        println!("  shape {:?} (m={m} k={k})", shape);
        let (b256, s256) = collect_blocks_256(&f32d, m, k, &s1_256, &s2_256, budget_256);
        let (b1024, s1024) = collect_blocks_1024(&f32d, m, k, &s1_1024, &s2_1024, budget_1024);
        println!("  256-groups {} 1024-groups {}", b256.len(), b1024.len());
        recs.push(Rec {
            name: *label,
            blocks_256: b256,
            scales_256: s256,
            blocks_1024: b1024,
            scales_1024: s1024,
        });
    }
    let mut all_b256 = Vec::new();
    let mut all_s256 = Vec::new();
    let mut all_b1024 = Vec::new();
    let mut all_s1024 = Vec::new();
    for r in &recs {
        all_b256.extend(r.blocks_256.clone());
        all_s256.extend(r.scales_256.clone());
        all_b1024.extend(r.blocks_1024.clone());
        all_s1024.extend(r.scales_1024.clone());
    }
    println!(
        "\nCombined {} x256 groups, {} x1024 groups",
        all_b256.len(),
        all_b1024.len()
    );
    let mse_aff = mse_affine(&all_b256);
    let mse_cb4_rms = mse_gl(&all_b256, &all_s256, &GL_CB4);
    let mse_cb3_rms = mse_gl(&all_b256, &all_s256, &GL_CB3);
    let mse_cb2_rms = mse_gl(&all_b256, &all_s256, &GL_CB2);
    let mse_cb1_rms = mse_gl(&all_b1024, &all_s1024, &GL_CB1);
    // --- Baseline RMS table (kept for self-validation) ---
    println!("\n=== Combined MSE (rotated domain, fp16 RMS) ===");
    println!(" affine (4-bit uniform) : {:.8e}", mse_aff);
    println!(
        " GL_CB4 (4-bit)         : {:.8e}  gain {:.2}% vs affine",
        mse_cb4_rms,
        100.0 * (1.0 - mse_cb4_rms / mse_aff)
    );
    println!(
        " GL_CB3 (3-bit)         : {:.8e}  vs GL_CB4 {:.2}%  vs affine {:.2}%",
        mse_cb3_rms,
        100.0 * (mse_cb3_rms / mse_cb4_rms - 1.0),
        100.0 * (1.0 - mse_cb3_rms / mse_aff)
    );
    println!(
        " GL_CB2 (2-bit)         : {:.8e}  vs GL_CB4 {:.2}%  vs affine {:.2}%",
        mse_cb2_rms,
        100.0 * (mse_cb2_rms / mse_cb4_rms - 1.0),
        100.0 * (1.0 - mse_cb2_rms / mse_aff)
    );
    println!(
        " GL_CB1 (1-bit)         : {:.8e}  vs GL_CB4 {:.2}%  vs affine {:.2}%",
        mse_cb1_rms,
        100.0 * (mse_cb1_rms / mse_cb4_rms - 1.0),
        100.0 * (1.0 - mse_cb1_rms / mse_aff)
    );
    // Per-tensor RMS table
    println!("\n=== Per-tensor MSE (RMS) ===");
    println!(
        "{:<35} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "tensor", "affine", "GL_CB4", "GL_CB3", "GL_CB2", "GL_CB1"
    );
    for r in &recs {
        let a = mse_affine(&r.blocks_256);
        let c4 = mse_gl(&r.blocks_256, &r.scales_256, &GL_CB4);
        let c3 = mse_gl(&r.blocks_256, &r.scales_256, &GL_CB3);
        let c2 = mse_gl(&r.blocks_256, &r.scales_256, &GL_CB2);
        let c1 = mse_gl(&r.blocks_1024, &r.scales_1024, &GL_CB1);
        println!(
            "{:<35} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e}",
            r.name, a, c4, c3, c2, c1
        );
    }

    let m_cb4_rms = metrics_scalar(&all_b256, &GL_CB4, false, false);
    let m_cb4_ls = metrics_scalar(&all_b256, &GL_CB4, true, false);
    let m_cb4_ls2 = metrics_scalar(&all_b256, &GL_CB4, true, true);
    let m_cb3_rms = metrics_scalar(&all_b256, &GL_CB3, false, false);
    let m_cb3_ls = metrics_scalar(&all_b256, &GL_CB3, true, false);
    let m_cb3_ls2 = metrics_scalar(&all_b256, &GL_CB3, true, true);
    let m_cb2_rms = metrics_scalar(&all_b256, &GL_CB2, false, false);
    let m_cb2_ls = metrics_scalar(&all_b256, &GL_CB2, true, false);
    let m_cb2_ls2 = metrics_scalar(&all_b256, &GL_CB2, true, true);
    let m_cb1_rms = metrics_scalar(&all_b1024, &GL_CB1, false, false);
    let m_cb1_ls = metrics_scalar(&all_b1024, &GL_CB1, true, false);
    let m_cb1_ls2 = metrics_scalar(&all_b1024, &GL_CB1, true, true);
    let m_cb35_rms = metrics_vq35(&all_b256, false, false);
    let m_cb35_ls = metrics_vq35(&all_b256, true, false);
    let m_cb35_ls2 = metrics_vq35(&all_b256, true, true);
    // Paired RMS-vs-LS table
    println!("\n=== Paired RMS vs LS (one refinement pass, fp16-rounded s*) ===");
    println!(
        "{:<8} {:>4} {:>14} {:>14} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "cb",
        "bits",
        "MSE(RMS)",
        "MSE(LS)",
        "gain%",
        "ret(RMS)",
        "ret(LS)",
        "|wh|/|w|(RMS)",
        "|wh|/|w|(LS)"
    );
    let rows = vec![
        ("GL_CB1", 1, m_cb1_rms, m_cb1_ls),
        ("GL_CB2", 2, m_cb2_rms, m_cb2_ls),
        ("GL_CB3", 3, m_cb3_rms, m_cb3_ls),
        ("GL_CB4", 4, m_cb4_rms, m_cb4_ls),
        ("GL_CB35", 3, m_cb35_rms, m_cb35_ls), // 7b per pair ~3.5b per weight, label bits approx
    ];
    for (name, bits, rms, ls) in rows {
        let gain = 100.0 * (1.0 - ls.mse / rms.mse);
        println!(
            "{:<8} {:>4} {:>14.8e} {:>14.8e} {:>7.3}% {:>10.5} {:>10.5} {:>10.5} {:>10.5}",
            name,
            bits,
            rms.mse,
            ls.mse,
            gain,
            rms.retained,
            ls.retained,
            rms.norm_ratio,
            ls.norm_ratio
        );
    }

    // Explicit report table requested in spec
    println!("\n=== Report table: codebook, bits, MSE(RMS), MSE(LS), relative improvement, retained before, retained after ===");
    println!(
        "{:<8} {:>4} {:>14} {:>14} {:>8} {:>14} {:>14}",
        "cb", "bits", "MSE_RMS", "MSE_LS", "impr%", "retained_RMS", "retained_LS"
    );
    for (name, bits, rms, ls) in vec![
        ("GL_CB4", 4, m_cb4_rms, m_cb4_ls),
        ("GL_CB3", 3, m_cb3_rms, m_cb3_ls),
        ("GL_CB2", 2, m_cb2_rms, m_cb2_ls),
        ("GL_CB1", 1, m_cb1_rms, m_cb1_ls),
    ] {
        let impr = 100.0 * (1.0 - ls.mse / rms.mse);
        println!(
            "{:<8} {:>4} {:>14.8e} {:>14.8e} {:>7.3}% {:>14.5} {:>14.5}",
            name, bits, rms.mse, ls.mse, impr, rms.retained, ls.retained
        );
    }
    println!(" GL_CB35 (VQ 7b/pair ~3.5b/w) : MSE_RMS {:.8e} MSE_LS {:.8e} impr {:.3}% retained {:.5}->{:.5}", m_cb35_rms.mse, m_cb35_ls.mse, 100.0*(1.0 - m_cb35_ls.mse/m_cb35_rms.mse), m_cb35_rms.retained, m_cb35_ls.retained);

    // Shrinkage visibility
    println!("\n=== Shrinkage: retained = dot(w,w_hat)/dot(w,w), norm = |w_hat|/|w| ===");
    for (name, rms, ls) in vec![
        ("GL_CB4", m_cb4_rms, m_cb4_ls),
        ("GL_CB3", m_cb3_rms, m_cb3_ls),
        ("GL_CB2", m_cb2_rms, m_cb2_ls),
        ("GL_CB1", m_cb1_rms, m_cb1_ls),
        ("GL_CB35", m_cb35_rms, m_cb35_ls),
    ] {
        println!(
            " {}: retained RMS {:.5} -> LS {:.5} (delta {:+.5}), norm RMS {:.5} -> LS {:.5}",
            name,
            rms.retained,
            ls.retained,
            ls.retained - rms.retained,
            rms.norm_ratio,
            ls.norm_ratio
        );
    }

    // Second pass delta
    println!("\n=== Second refinement pass delta (LS1 vs LS2) ===");
    for (name, ls1, ls2) in vec![
        ("GL_CB4", m_cb4_ls, m_cb4_ls2),
        ("GL_CB3", m_cb3_ls, m_cb3_ls2),
        ("GL_CB2", m_cb2_ls, m_cb2_ls2),
        ("GL_CB1", m_cb1_ls, m_cb1_ls2),
        ("GL_CB35", m_cb35_ls, m_cb35_ls2),
    ] {
        let d = 100.0 * (1.0 - ls2.mse / ls1.mse);
        println!(
            " {}: LS1 {:.8e} LS2 {:.8e}  LS2 gain over LS1 {:.4}%",
            name, ls1.mse, ls2.mse, d
        );
    }

    // Family table with bpw (RMS baseline kept)
    let baseline_aff = mse_aff;
    let baseline_cb4 = mse_cb4_rms;
    println!("\n=== Family table (group, bytes/group, bpw, MSE_RMS, vs GL_CB4, vs affine) ===");
    println!(
        "{:<10} {:>6} {:>10} {:>8} {:>14} {:>12} {:>12}",
        "format", "group", "bytes/grp", "bpw", "MSE", "vs GL_CB4", "vs affine"
    );
    let rows: Vec<(&str, usize, usize, f64, f64)> = vec![
        ("mq1", 1024, 130, 1.015625, mse_cb1_rms),
        ("mq2gl", 256, 66, 2.0625, mse_cb2_rms),
        ("mq3gl", 256, 98, 3.0625, mse_cb3_rms),
        ("mq35gl", 256, 114, 3.5625, m_cb35_rms.mse),
        ("affine", 256, 136, 4.25, mse_aff),
    ];
    for (fmt, grp, bytes, bpw, mse) in rows {
        let vs_cb4 = if baseline_cb4 > 0.0 {
            mse / baseline_cb4 - 1.0
        } else {
            0.0
        };
        let vs_aff = if baseline_aff > 0.0 {
            mse / baseline_aff - 1.0
        } else {
            0.0
        };
        println!(
            "{:<10} {:>6} {:>10} {:>8.4} {:>14.8e} {:>11.2}% {:>11.2}%",
            fmt,
            grp,
            bytes,
            bpw,
            mse,
            vs_cb4 * 100.0,
            vs_aff * 100.0
        );
    }
    println!(
        "\nBaseline references: affine {:.8e}  GL_CB4 {:.8e}",
        baseline_aff, baseline_cb4
    );
    println!("Expected baselines (poly_fit): affine 1.85588400e-06  GL_CB4 1.47499897e-06");
    // Flag ordering check
    println!(
        "\n=== Ordering check (lower bpw should NOT beat higher bpw if codebook is correct) ==="
    );
    let ordered = vec![
        ("mq1", m_cb1_rms.mse),
        ("mq2gl", m_cb2_rms.mse),
        ("mq3gl", m_cb3_rms.mse),
        ("mq35gl", m_cb35_rms.mse),
    ];
    for w in ordered.windows(2) {
        let (low_fmt, low_mse) = w[0];
        let (high_fmt, high_mse) = w[1];
        if low_mse < high_mse {
            println!(
                "  FLAG: {} (lower bpw) MSE {:.8e} < {} {:.8e} — indicates bug",
                low_fmt, low_mse, high_fmt, high_mse
            );
        } else {
            println!(
                "  OK: {} {:.8e} >= {} {:.8e}",
                low_fmt, low_mse, high_fmt, high_mse
            );
        }
    }
    // Bytes-per-group unchanged confirmation
    println!("\n=== Bytes per group (on-disk layout UNCHANGED) ===");
    println!(" mq1    : 130 B / 1024  = 1.015625 bpw  (128 idx +2 scale)");
    println!(" mq2gl  :  66 B / 256   = 2.0625  bpw  (64 idx +2 scale)");
    println!(" mq3gl  :  98 B / 256   = 3.0625  bpw  (96 idx +2 scale)");
    println!(" mq35gl : 114 B / 256   = 3.5625  bpw  (112 idx +2 scale)");
    println!(" affine : 136 B / 256   = 4.25    bpw  (128 nibbles +8 hdr)");
    println!(" Layout SoA: [m*gpr*IDX][m*gpr*2 scale] — same bytes per group, only VALUE in scale field changes.");
    // Lane alignment report
    println!("\n=== Lane alignment (group * bits / 32) ===");
    println!(" mq1     : 1024*1/32 = 32 bits = 4 B = 1 u32/lane  => exactly aligned, no unroll");
    println!(
        " mq2gl   : 256*2/32 = 16 bits = 2 B = 0.5 register => needs K2 unroll to fill register"
    );
    println!(" mq3gl   : 256*3/32 = 24 bits = 3 B => needs K4 unroll (existing MQ3 kernels are K4-unrolled)");
    println!(" affine  : 256*4/32 = 32 bits = 4 B = 1 u32/lane => exactly aligned, no unroll");
    // LS exposure note
    println!("\n=== LS scale exposure ===");
    println!(" RMS path kept reachable via gl_encode_block_rms / gl_encode_block_vq35_rms in src/main.rs (and via use_ls=false in this harness).");
    println!(" Production path gl_encode_block / gl_encode_block_vq35 / quantize_mq1g1024gl now use LS one-pass (s* -> fp16 -> re-encode).");
}
