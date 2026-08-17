//! MQ family MSE harness — rotated-domain quality vs bpw.
//! Measures affine 4-bit, GL_CB4, GL_CB3, GL_CB2, GL_CB1 on real post-FWHT weights.
//! Template: crates/hipfire-quantize/examples/poly_fit.rs
//! Run: cargo run -q -p hipfire-quantize --example mq_family_mse --release

use std::path::Path;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::SafetensorsFile;

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(1103515245)
                .wrapping_add(12345)
                & 0x7fffffff;
            if (state >> 16) & 1 == 1 { 1.0 } else { -1.0 }
        })
        .collect()
}
fn cpu_fwht_256(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 { x[i] *= s1[i]; }
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
    for i in 0..256 { x[i] *= 0.0625 * s2[i]; }
}
fn cpu_fwht_1024(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert_eq!(x.len(), 1024);
    assert_eq!(s1.len(), 1024);
    assert_eq!(s2.len(), 1024);
    for i in 0..1024 { x[i] *= s1[i]; }
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
    for i in 0..1024 { x[i] *= 0.03125 * s2[i]; }
}
fn f32_to_fp16_bits(v: f32) -> u16 { f32_to_f16(v) }
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
const GL_CB3: [f32; 8] = [-2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520];
const GL_CB4: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284,
    0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
];

fn mse_gl(blocks: &[Vec<f32>], scales: &[f32], cb: &[f32]) -> f64 {
    let mut sse = 0.0f64;
    let mut n = 0usize;
    for (grp, &sc) in blocks.iter().zip(scales.iter()) {
        if sc == 0.0 {
            for &v in grp { sse += (v as f64) * (v as f64); }
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
                if d < bd { bd = d; best = c; }
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
fn collect_blocks_256(f32d: &[f32], m: usize, k: usize, s1: &[f32], s2: &[f32], budget: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let gpr = k / 256;
    let mut groups = Vec::new();
    let mut scales = Vec::new();
    let mut cnt = 0;
    for row in 0..m {
        for g in 0..gpr {
            if cnt >= budget { break; }
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
        if cnt >= budget { break; }
    }
    (groups, scales)
}
fn collect_blocks_1024(f32d: &[f32], m: usize, k: usize, s1: &[f32], s2: &[f32], budget_groups: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    // budget in 1024-groups; for 256 budget we need 1/4 as many 1024 groups to cover same weight count.
    let gpr = k / 1024;
    let mut groups = Vec::new();
    let mut scales = Vec::new();
    let mut cnt = 0;
    for row in 0..m {
        for g in 0..gpr {
            if cnt >= budget_groups { break; }
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
        if cnt >= budget_groups { break; }
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
        ("early linear_attn out_proj (layer 0)", "model.language_model.layers.0.linear_attn.out_proj.weight"),
        ("mid mlp down_proj (layer 20)", "model.language_model.layers.20.mlp.down_proj.weight"),
        ("late mlp gate_proj (layer 40)", "model.language_model.layers.40.mlp.gate_proj.weight"),
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
        recs.push(Rec { name: *label, blocks_256: b256, scales_256: s256, blocks_1024: b1024, scales_1024: s1024 });
    }
    let mut all_b256 = Vec::new(); let mut all_s256 = Vec::new();
    let mut all_b1024 = Vec::new(); let mut all_s1024 = Vec::new();
    for r in &recs {
        all_b256.extend(r.blocks_256.clone());
        all_s256.extend(r.scales_256.clone());
        all_b1024.extend(r.blocks_1024.clone());
        all_s1024.extend(r.scales_1024.clone());
    }
    println!("\nCombined {} x256 groups, {} x1024 groups", all_b256.len(), all_b1024.len());
    let mse_aff = mse_affine(&all_b256);
    let mse_cb4 = mse_gl(&all_b256, &all_s256, &GL_CB4);
    let mse_cb3 = mse_gl(&all_b256, &all_s256, &GL_CB3);
    let mse_cb2 = mse_gl(&all_b256, &all_s256, &GL_CB2);
    let mse_cb1 = mse_gl(&all_b1024, &all_s1024, &GL_CB1);
    println!("\n=== Combined MSE (rotated domain, fp16 RMS) ===");
    println!(" affine (4-bit uniform) : {:.8e}", mse_aff);
    println!(" GL_CB4 (4-bit)         : {:.8e}  gain {:.2}% vs affine", mse_cb4, 100.0 * (1.0 - mse_cb4 / mse_aff));
    println!(" GL_CB3 (3-bit)         : {:.8e}  vs GL_CB4 {:.2}%  vs affine {:.2}%", mse_cb3, 100.0 * (mse_cb3 / mse_cb4 - 1.0), 100.0 * (1.0 - mse_cb3 / mse_aff));
    println!(" GL_CB2 (2-bit)         : {:.8e}  vs GL_CB4 {:.2}%  vs affine {:.2}%", mse_cb2, 100.0 * (mse_cb2 / mse_cb4 - 1.0), 100.0 * (1.0 - mse_cb2 / mse_aff));
    println!(" GL_CB1 (1-bit)         : {:.8e}  vs GL_CB4 {:.2}%  vs affine {:.2}%", mse_cb1, 100.0 * (mse_cb1 / mse_cb4 - 1.0), 100.0 * (1.0 - mse_cb1 / mse_aff));

    // Per-tensor table
    println!("\n=== Per-tensor MSE ===");
    println!("{:<35} {:>12} {:>12} {:>12} {:>12} {:>12}", "tensor", "affine", "GL_CB4", "GL_CB3", "GL_CB2", "GL_CB1");
    for r in &recs {
        let a = mse_affine(&r.blocks_256);
        let c4 = mse_gl(&r.blocks_256, &r.scales_256, &GL_CB4);
        let c3 = mse_gl(&r.blocks_256, &r.scales_256, &GL_CB3);
        let c2 = mse_gl(&r.blocks_256, &r.scales_256, &GL_CB2);
        let c1 = mse_gl(&r.blocks_1024, &r.scales_1024, &GL_CB1);
        println!("{:<35} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e}", r.name, a, c4, c3, c2, c1);
    }

    // Family table with bpw
    let baseline_aff = mse_aff;
    let baseline_cb4 = mse_cb4;
    println!("\n=== Family table (group, bytes/group, bpw, MSE, vs GL_CB4, vs affine) ===");
    println!("{:<10} {:>6} {:>10} {:>8} {:>14} {:>12} {:>12}", "format", "group", "bytes/grp", "bpw", "MSE", "vs GL_CB4", "vs affine");
    let rows: Vec<(&str, usize, usize, f64, f64)> = vec![
        ("mq1", 1024, 130, 1.015625, mse_cb1),
        ("mq2gl", 256, 66, 2.0625, mse_cb2),
        ("mq3gl", 256, 98, 3.0625, mse_cb3),
        ("mq4gl", 256, 130, 4.0625, mse_cb4),
        ("affine", 256, 136, 4.25, mse_aff), // reference: uniform 4-bit with f32 scale/zero 136B
    ];
    // Sort by bpw ascending already
    for (fmt, grp, bytes, bpw, mse) in rows {
        let vs_cb4 = if baseline_cb4 > 0.0 { mse / baseline_cb4 - 1.0 } else { 0.0 };
        let vs_aff = if baseline_aff > 0.0 { mse / baseline_aff - 1.0 } else { 0.0 };
        println!("{:<10} {:>6} {:>10} {:>8.4} {:>14.8e} {:>11.2}% {:>11.2}%", fmt, grp, bytes, bpw, mse, vs_cb4 * 100.0, vs_aff * 100.0);
    }
    println!("\nBaseline references: affine {:.8e}  GL_CB4 {:.8e}", baseline_aff, baseline_cb4);
    println!("Expected baselines (poly_fit): affine 1.85588400e-06  GL_CB4 1.47499897e-06");
    // Flag ordering check
    println!("\n=== Ordering check (lower bpw should NOT beat higher bpw if codebook is correct) ===");
    let ordered = vec![("mq1", mse_cb1), ("mq2gl", mse_cb2), ("mq3gl", mse_cb3), ("mq4gl", mse_cb4)];
    for w in ordered.windows(2) {
        let (low_fmt, low_mse) = w[0];
        let (high_fmt, high_mse) = w[1];
        if low_mse < high_mse {
            println!("  FLAG: {} (lower bpw) MSE {:.8e} < {} {:.8e} — indicates bug, not real result", low_fmt, low_mse, high_fmt, high_mse);
        } else {
            println!("  OK: {} {:.8e} >= {} {:.8e}", low_fmt, low_mse, high_fmt, high_mse);
        }
    }
    // Lane alignment report
    println!("\n=== Lane alignment (group * bits / 32) ===");
    println!(" mq1     : 1024*1/32 = 32 bits = 4 B = 1 u32/lane  => exactly aligned, no unroll");
    println!(" mq2gl   : 256*2/32 = 16 bits = 2 B = 0.5 register => needs K2 unroll to fill register");
    println!(" mq3gl   : 256*3/32 = 24 bits = 3 B => needs K4 unroll (existing MQ3 kernels are K4-unrolled)");
    println!(" mq4gl   : 256*4/32 = 32 bits = 4 B = 1 u32/lane => exactly aligned, no unroll");
    println!(" affine  : same as mq4gl (4 bits, G256, 32 bits/lane)");
}
