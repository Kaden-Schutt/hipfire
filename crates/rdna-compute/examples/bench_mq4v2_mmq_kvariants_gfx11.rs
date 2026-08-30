// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
//! Candidate K-axis variants parity + timing microbench for MQ4V2 MMQ.
//! Gfx11 only (gfx1100/gfx1151). Runs on hipx via HipxMeasure.

use std::ffi::c_void;
use rdna_compute::Gpu;
const CANDIDATE_SRC: &str = include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_kvariants.hip");
const PROD_SRC: &str = include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq.hip");

const GROUP: usize = 256;
const GROUP_BYTES: usize = 136;
const HALF: usize = 128;
const MMQ_X: usize = 128;
const MMQ_Y: usize = 128;
const MMQ_TILE_Y_K: usize = 36;
const MMQ_TILE_X_K: usize = 76;
const REL_RMS_LIMIT: f64 = 0.002;
const CANARY: f32 = 7.654_321;
const WARMUP: usize = 32;
const MEASURED: usize = 100;

#[derive(Clone, Copy)]
struct Candidate {
    label: &'static str,
    generic: &'static str,
    full_add: &'static str,
    full_set: &'static str,
    is_ksplit: bool,
}

const CANDIDATES: &[Candidate] = &[
    Candidate { label: "k2", generic: "gemm_mq4g256v2_residual_mmq_k2", full_add: "gemm_mq4g256v2_residual_mmq_k2_full_add", full_set: "gemm_mq4g256v2_residual_mmq_k2_full_set", is_ksplit: false },
    Candidate { label: "k4", generic: "gemm_mq4g256v2_residual_mmq_k4", full_add: "gemm_mq4g256v2_residual_mmq_k4_full_add", full_set: "gemm_mq4g256v2_residual_mmq_k4_full_set", is_ksplit: false },
    Candidate { label: "ksplit", generic: "gemm_mq4g256v2_residual_mmq_ksplit", full_add: "gemm_mq4g256v2_residual_mmq_ksplit_full_add", full_set: "gemm_mq4g256v2_residual_mmq_ksplit_full_set", is_ksplit: true },
];

const CANDIDATE_MODULE: &str = "gemm_mq4g256v2_residual_mmq_kvariants";

fn prng(i: usize, salt: u32) -> f32 {
    let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(salt as u64);
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    x = x.wrapping_mul(0x2545F4914F6CDD1D);
    ((x >> 32) as u32 as f64 / u32::MAX as f64) as f32
}
fn half_from_f32(v: f32) -> u16 { rdna_compute::kv_slots::half_from_f32(v) }

fn build_disjoint_halves(m: usize, k: usize) -> Vec<f32> {
    let groups = k / GROUP;
    let mut w = vec![0f32; m * k];
    for r in 0..m {
        for g in 0..groups {
            let k0 = g * GROUP;
            let s0 = prng(r * 100000 + g * 2, 0xA11CE) * 2.0 - 1.0;
            let z0 = prng(r * 100000 + g * 2, 0xBEEF) * 2.0 - 1.0;
            let s1 = 96.0 + prng(r * 100000 + g * 2 + 1, 0xA11CE) * 64.0;
            let z1 = 96.0 + prng(r * 100000 + g * 2 + 1, 0xBEEF) * 64.0;
            for t in 0..HALF { w[r * k + k0 + t] = s0 * (prng(r * k + k0 + t, 0x1234) * 2.0 - 1.0) * 6.0 + z0; }
            for t in HALF..GROUP { w[r * k + k0 + t] = s1 * (prng(r * k + k0 + t, 0x1234) * 2.0 - 1.0) * 6.0 + z1; }
        }
    }
    w
}
fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    let groups = k / GROUP;
    let mut blob = vec![0u8; m * groups * GROUP_BYTES];
    for r in 0..m {
        for g in 0..groups {
            let k0 = g * GROUP;
            let base = (r * groups + g) * GROUP_BYTES;
            let mut amax0: f32 = 0.0;
            let mut amax1: f32 = 0.0;
            for t in 0..HALF { amax0 = amax0.max(w[r * k + k0 + t].abs()); }
            for t in HALF..GROUP { amax1 = amax1.max(w[r * k + k0 + t].abs()); }
            let sc0 = if amax0 < 1e-6 { 1.0 } else { amax0 / 7.5 };
            let sc1 = if amax1 < 1e-6 { 1.0 } else { amax1 / 7.5 };
            let s0h = half_from_f32(sc0);
            let z0h = half_from_f32(0.0);
            let s1h = half_from_f32(sc1);
            let z1h = half_from_f32(0.0);
            let hs0 = (z0h as u32) << 16 | s0h as u32;
            let hs1 = (z1h as u32) << 16 | s1h as u32;
            blob[base..base+4].copy_from_slice(&hs0.to_le_bytes());
            blob[base+4..base+8].copy_from_slice(&hs1.to_le_bytes());
            for t in 0..GROUP {
                let v = w[r * k + k0 + t];
                let sc = if t < HALF { sc0 } else { sc1 };
                let q = ((v / sc).round().clamp(0.0, 15.0)) as u8;
                let byte_idx = base + 8 + t / 2;
                if t % 2 == 0 { blob[byte_idx] = (blob[byte_idx] & 0xF0) | (q & 0x0F); }
                else { blob[byte_idx] = (blob[byte_idx] & 0x0F) | ((q & 0x0F) << 4); }
            }
        }
    }
    blob
}
fn is_finite(v: &[f32]) -> bool { v.iter().all(|x| x.is_finite()) }
fn variance(v: &[f32]) -> f64 {
    if v.is_empty() { return 0.0; }
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    v.iter().map(|x| { let d = *x as f64 - mean; d * d }).sum::<f64>() / v.len() as f64
}
fn rel_rms(got: &[f32], want: &[f32]) -> f64 {
    let mut num = 0.0; let mut den = 0.0;
    for (a,b) in got.iter().zip(want.iter()) { let d = *a as f64 - *b as f64; num += d*d; den += (*b as f64)*(*b as f64); }
    if den == 0.0 { return if num==0.0 {0.0} else {f64::INFINITY}; }
    (num/den).sqrt()
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b.iter()).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max) }

fn launch_candidate(
    gpu: &mut Gpu,
    cand: Candidate,
    a_raw: &rdna_compute::GpuTensor,
    xq_ptr: *mut c_void,
    y: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    n: usize,
    add: bool,
) -> Result<(), String> {
    let full = m % 128 == 0 && n % 128 == 0;
    let kernel_name = match (full, add) {
        (true, true) => cand.full_add,
        (true, false) => cand.full_set,
        (false, _) => cand.generic,
    };
    gpu.ensure_kernel_public(CANDIDATE_MODULE, CANDIDATE_SRC, kernel_name)
        .map_err(|e| format!("ensure {kernel_name}: {e:?}"))?;
    let shared = ((MMQ_X * MMQ_TILE_Y_K + MMQ_Y * MMQ_TILE_X_K) * std::mem::size_of::<i32>()) as u32;
    let row_tiles = m.div_ceil(MMQ_Y) as u32;
    let col_tiles = n.div_ceil(MMQ_X) as u32;
    let grid_z = if cand.is_ksplit { 4 } else { 1 };
    let grid = [row_tiles, col_tiles, grid_z];
    let block = [32u32, 8, 1];
    let mut blob = hip_bridge::KernargBlob::new();
    blob.push_ptr(a_raw.buf.as_ptr() as *mut c_void);
    blob.push_ptr(xq_ptr);
    blob.push_ptr(y.buf.as_ptr() as *mut c_void);
    blob.push_i32(m as i32);
    blob.push_i32(k as i32);
    blob.push_i32(n as i32);
    blob.push_i32(i32::from(add));
    let mut blob_bytes = blob.as_bytes().to_vec();
    gpu.launch_kernel_blob(kernel_name, grid, block, shared, &mut blob_bytes)
        .map_err(|e| format!("launch {kernel_name}: {e:?}"))?;
    Ok(())
}

fn time_prod_vs_candidate(
    gpu: &mut Gpu,
    a_raw: &rdna_compute::GpuTensor,
    x_host: &[f32],
    y_init_host: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (f64, Vec<(&'static str, f64)>) {
    let d_x = gpu.upload_f32(x_host, &[n * k]).expect("upload x");
    gpu.hip.device_synchronize().unwrap();
    let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).expect("quant");
    gpu.hip.device_synchronize().unwrap();
    let y_len = n * m;

    let d_y_prod = gpu.upload_f32(y_init_host, &[y_len+1]).expect("upload y prod");
    for _ in 0..WARMUP {
        gpu.gemm_mq4g256v2_mmq_set_prequant(a_raw, xq, &d_y_prod, m, k, n).expect("prod warmup");
        gpu.hip.device_synchronize().unwrap();
    }
    let mut prod_times = Vec::with_capacity(MEASURED);
    for _ in 0..MEASURED {
        let t0 = std::time::Instant::now();
        gpu.gemm_mq4g256v2_mmq_set_prequant(a_raw, xq, &d_y_prod, m, k, n).expect("prod");
        gpu.hip.device_synchronize().unwrap();
        prod_times.push(t0.elapsed().as_secs_f64()*1e6);
    }
    prod_times.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let prod_med = prod_times[prod_times.len()/2];
    let _ = gpu.free_tensor(d_y_prod);

    let mut cand_times = Vec::new();
    for cand in CANDIDATES {
        let d_y = gpu.upload_f32(y_init_host, &[y_len+1]).expect("upload y cand");
        for _ in 0..WARMUP {
            launch_candidate(gpu, *cand, a_raw, xq, &d_y, m, k, n, false).expect("cand warmup");
            gpu.hip.device_synchronize().unwrap();
        }
        let mut times = Vec::with_capacity(MEASURED);
        for _ in 0..MEASURED {
            let t0 = std::time::Instant::now();
            launch_candidate(gpu, *cand, a_raw, xq, &d_y, m, k, n, false).expect("cand launch");
            gpu.hip.device_synchronize().unwrap();
            times.push(t0.elapsed().as_secs_f64()*1e6);
        }
        times.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let med = times[times.len()/2];
        cand_times.push((cand.label, med));
        let _ = gpu.free_tensor(d_y);
    }
    let _ = gpu.free_tensor(d_x);
    (prod_med, cand_times)
}

fn run_parity(gpu: &mut Gpu, d_a: &rdna_compute::GpuTensor, m: usize, k: usize, n: usize, add: bool) -> bool {
    let label = if add {"mmq-add"} else {"mmq-set"};
    let x_host: Vec<f32> = (0..n*k).map(|i| prng(i, 0xC0FFEE00) *2.0-1.0).collect();
    let d_x = gpu.upload_f32(&x_host, &[n*k]).expect("upload x");
    gpu.hip.device_synchronize().unwrap();
    let y_init: Vec<f32> = {
        let mut v = vec![0.0f32; n*m +1];
        for i in 0..n*m { v[i] = if add { prng(i, 0xBEEF1234)*2.0-1.0+0.5 } else {0.0}; }
        v[n*m]=CANARY;
        v
    };
    let d_y_ref = gpu.upload_f32(&y_init, &[n*m+1]).expect("yref");
    let xq = gpu.ensure_q8_1_mmq_x(&d_x, n, k).expect("q8");
    gpu.hip.device_synchronize().unwrap();
    let res = if add { gpu.gemm_mq4g256v2_mmq_add_prequant(d_a, xq, &d_y_ref, m, k, n) } else { gpu.gemm_mq4g256v2_mmq_set_prequant(d_a, xq, &d_y_ref, m, k, n) };
    if let Err(e)=res { eprintln!("  base {} N={} failed {:?}", label, n, e); let _=gpu.free_tensor(d_x); let _=gpu.free_tensor(d_y_ref); return false; }
    gpu.hip.device_synchronize().unwrap();
    let y_ref_full = gpu.download_f32(&d_y_ref).expect("dl ref");
    let canary_ref_ok = y_ref_full.len()==n*m+1 && y_ref_full[n*m].to_bits()==CANARY.to_bits();
    let y_ref = &y_ref_full[..n*m];
    if !is_finite(y_ref) || variance(y_ref)<=1e-12 { eprintln!("  base invalid"); let _=gpu.free_tensor(d_x); let _=gpu.free_tensor(d_y_ref); return false; }
    let _ = canary_ref_ok;
    let mut ok_all = true;
    for cand in CANDIDATES {
        let d_y_cand = gpu.upload_f32(&y_init, &[n*m+1]).expect("ycand");
        let xq2 = gpu.ensure_q8_1_mmq_x(&d_x, n, k).expect("q8b");
        gpu.hip.device_synchronize().unwrap();
        let res2 = launch_candidate(gpu, *cand, d_a, xq2, &d_y_cand, m, k, n, add);
        if let Err(e)=res2 { eprintln!("  {} {} failed {:?}", cand.label, label, e); let _=gpu.free_tensor(d_y_cand); ok_all=false; continue; }
        gpu.hip.device_synchronize().unwrap();
        let y_cand_full = gpu.download_f32(&d_y_cand).expect("dl cand");
        let canary_ok = y_cand_full.len()==n*m+1 && y_cand_full[n*m].to_bits()==CANARY.to_bits();
        let y_cand = &y_cand_full[..n*m.min(y_cand_full.len())];
        if y_cand.iter().zip(y_init[..n*m].iter()).all(|(a,b)| a.to_bits()==b.to_bits()) {
            eprintln!("  STUB {} {} wrote nothing", cand.label, label); ok_all=false;
            let _=gpu.free_tensor(d_y_cand); continue;
        }
        let rr = rel_rms(y_cand, y_ref);
        let finite = is_finite(y_cand);
        let ok = rr <= REL_RMS_LIMIT && finite && canary_ok;
        eprintln!("  parity {} vs prod {} N={} rel_rms={:.5} finite={} canary={} -> {}", cand.label, label, n, rr, finite, canary_ok, if ok{"PASS"} else{"FAIL"});
        if !ok { ok_all=false; }
        println!("{{\"kind\":\"parity\",\"variant\":\"{}\",\"mode\":\"{}\",\"m\":{},\"k\":{},\"n\":{},\"rel_rms\":{},\"max_abs\":{},\"pass\":{}}}", cand.label, label, m,k,n, rr, max_abs_diff(y_cand,y_ref), ok);
        let _=gpu.free_tensor(d_y_cand);
    }
    let _=gpu.free_tensor(d_x);
    let _=gpu.free_tensor(d_y_ref);
    ok_all
}

fn main() {
    let mut gpu = match Gpu::init() { Ok(g)=>g, Err(e)=>{ eprintln!("SKIP no GPU {e}"); return; } };
    let arch = gpu.arch.clone();
    if !matches!(arch.as_str(), "gfx1100"|"gfx1151") {
        eprintln!("SKIP arch {arch} not gfx11"); return;
    }
    eprintln!("arch {arch} bench_mq4v2_mmq_kvariants parity+timing");
    println!("{{\"arch\":\"{arch}\",\"kind\":\"meta\",\"commit\":\"{}\",\"binary\":\"bench_mq4v2_mmq_kvariants\"}}", env!("CARGO_PKG_VERSION"));

    let m_small = 128; let k_small = 512;
    let w_small = build_disjoint_halves(m_small, k_small);
    let blob_small = pack_mq4g256v2(&w_small, m_small, k_small);
    let d_a_small = gpu.upload_raw(&blob_small, &[blob_small.len()]).expect("upload A small");
    let mut all_ok = true;
    for &n in &[128usize, 512usize] {
        for &add in &[false, true] {
            if !run_parity(&mut gpu, &d_a_small, m_small, k_small, n, add) { all_ok=false; }
        }
    }
    let _=gpu.free_tensor(d_a_small);
    if !all_ok { eprintln!("parity FAIL"); std::process::exit(1); }
    eprintln!("parity PASS");

    let shapes: &[(usize,usize,usize,&str)] = &[
        (17408, 512, 5120, "gate_up 17408x5120 N=512"),
        (17408, 2048, 5120, "gate_up 17408x5120 N=2048"),
        (5120, 512, 17408, "down 5120x17408 N=512"),
        (5120, 2048, 17408, "down 5120x17408 N=2048"),
    ];
    for &(m,n,k,label) in shapes {
        eprintln!("\n=== timing {label} M={m} N={n} K={k} ===");
        let w: Vec<f32> = (0..m*k).map(|i| prng(i, 0xCAFEBABE)*2.0-1.0).collect();
        let blob = pack_mq4g256v2(&w, m, k);
        let d_a = gpu.upload_raw(&blob, &[blob.len()]).expect("upload A");
        let x_host: Vec<f32> = (0..n*k).map(|i| prng(i, 0xDEADBEEF)*2.0-1.0).collect();
        let y_init = vec![0.0f32; n*m+1];
        let (prod_us, cand_times) = time_prod_vs_candidate(&mut gpu, &d_a, &x_host, &y_init, m, k, n);
        eprintln!("  prod median {:.1} us", prod_us);
        for (cand_label, us) in &cand_times {
            let delta_pct = if prod_us>0.0 { (us - prod_us)/prod_us*100.0 } else {0.0};
            eprintln!("  {} median {:.1} us delta {:+.2}%", cand_label, us, delta_pct);
            println!("{{\"kind\":\"timing\",\"shape\":\"{label}\",\"m\":{m},\"k\":{k},\"n\":{n},\"variant\":\"{cand_label}\",\"prod_us\":{prod_us},\"cand_us\":{us},\"delta_pct\":{delta_pct}}}");
        }
        println!("{{\"kind\":\"timing\",\"shape\":\"{label}\",\"m\":{m},\"k\":{k},\"n\":{n},\"variant\":\"prod\",\"prod_us\":{prod_us},\"cand_us\":{prod_us},\"delta_pct\":0.0}}");
        let _=gpu.free_tensor(d_a);
    }
    eprintln!("\nDONE bench_mq4v2_mmq_kvariants");
}
