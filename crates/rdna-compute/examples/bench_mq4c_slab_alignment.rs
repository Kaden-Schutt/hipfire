//! Residual WMMA _bt GEMM kernel-level isolation for the dense 27B 2.9% gap.
//!
//! Prior hypothesis (LDS staged [16][264] dwordx4 misalignment) is DISPROVEN:
//! the only residual GEMMs that compiled in the scoring run were
//! gemm_mq4cg256_residual_wmma_gfx12_bt12.hsaco and _bt8.hsaco.
//! The ldsstage variants never compiled, so they are not on the path.
//! This bench therefore measures the kernels that ACTUALLY run:
//!   arm A: v1  _bt residual (136 B/group, two f32 header at +0/+4, nibbles +8)
//!   arm B: mq4c _bt residual (132 B/group, one packed fp16|fp16 dword at +0, nibbles +4)
//! Both read weights with 4-byte scalar loads, so there is no misalignment to fix.
//!
//! Question: at identical logical weights and shape M=K=5120, is mq4c _bt slower
//! than v1 _bt, and by how much? A null result (within noise) means the model-level
//! 2.9% comes from elsewhere.
//!
//! Requirements maintained from original spec:
//! - Shape m=k=5120, sweep batch 8,12,16,32
//! - Seeded weights encoded ONCE into BOTH containers sharing nibbles, so inter-arm diff = container only
//! - Correctness gate before timing with worst relative error per arm printed; faster-but-wrong is the known failure mode
//! - Device-side timing only (hip events, host Instant carries ~22.1 us sync tax)
//! - >=32 warmups, >=200 launches/sample, >=3 samples, interleaved sample-by-sample, report MINIMUMS (thermal drift ~6%)
//! - Both bt8 and bt12 tiles are measured (both compiled in the scoring run, cheap to test)
//!
//! Run: `cargo run --release -p rdna-compute --example bench_mq4c_slab_alignment`

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};

const GROUP: usize = 256;
const V1_BYTES: usize = 136;
const MQ4C_BYTES: usize = 132;

const WARMUP: usize = 64;
const ITERS: usize = 200;
const RUNS: usize = 5;

const M: usize = 5120;
const K: usize = 5120;
const BATCHES: &[usize] = &[8, 12, 16, 32];
const BVS: &[usize] = &[8, 12]; // bt tiles; also test 4 if needed but 8/12 are the compiled ones

const HFQ4_BT_SRC: &str = include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_gfx12_bt.hip");
const MQ4C_BT_SRC: &str = include_str!("../../../kernels/src/gemm_mq4cg256_residual_wmma_gfx12_bt.hip");

fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = b & 0x007F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1F {
        return sign | 0x7C00;
    }
    let mut m = (mant >> 13) as u16;
    if (mant & 0x1000) != 0 && ((mant & 0x0FFF) != 0 || (m & 1) != 0) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
            if exp >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp as u16) << 10) | m
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32 >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    let fbits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal -> normalize
            let mut e = -14i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let exp32 = (e + 127) as u32;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        let exp32 = (exp - 15 + 127) as u32;
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(fbits)
}

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Pack same logical weights into both containers sharing nibbles, as the repack does.
fn pack_both(w: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let gpr = k / GROUP;
    let mut b1 = vec![0u8; m * gpr * V1_BYTES];
    let mut bc = vec![0u8; m * gpr * MQ4C_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let d1 = (r * gpr + g) * V1_BYTES;
            let dc = (r * gpr + g) * MQ4C_BYTES;
            let s = &w[src..src + GROUP];
            let lo = s.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
            b1[d1..d1 + 4].copy_from_slice(&step.to_le_bytes());
            b1[d1 + 4..d1 + 8].copy_from_slice(&lo.to_le_bytes());
            let sb = f16_bits(step);
            let zb = f16_bits(lo);
            bc[dc..dc + 2].copy_from_slice(&sb.to_le_bytes());
            bc[dc + 2..dc + 4].copy_from_slice(&zb.to_le_bytes());
            let inv = if step > 0.0 { 1.0 / step } else { 0.0 };
            let mut q = [0u8; GROUP];
            for i in 0..GROUP {
                q[i] = ((s[i] - lo) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
            }
            for i in 0..128 {
                let byte = (q[2 * i] & 0xF) | ((q[2 * i + 1] & 0xF) << 4);
                b1[d1 + 8 + i] = byte;
                bc[dc + 4 + i] = byte;
            }
        }
    }
    (b1, bc)
}

fn host_gemm(
    blob: &[u8],
    m: usize,
    k: usize,
    batch: usize,
    x_f32: &[f32],
    is_mq4c: bool,
) -> Vec<f32> {
    let gpr = k / GROUP;
    let bpg = if is_mq4c { MQ4C_BYTES } else { V1_BYTES };
    let mut y = vec![0.0f32; batch * m];
    for b in 0..batch {
        let x_base = b * k;
        for r in 0..m {
            let mut acc = 0.0f64;
            for g in 0..gpr {
                let off = (r * gpr + g) * bpg;
                let (sc, zp, nib_base) = if is_mq4c {
                    let hdr = u32::from_le_bytes(blob[off..off + 4].try_into().unwrap());
                    let sc = f16_to_f32((hdr & 0xFFFF) as u16);
                    let zp = f16_to_f32((hdr >> 16) as u16);
                    (sc, zp, off + 4)
                } else {
                    let sc = f32::from_le_bytes(blob[off..off + 4].try_into().unwrap());
                    let zp = f32::from_le_bytes(blob[off + 4..off + 8].try_into().unwrap());
                    (sc, zp, off + 8)
                };
                for ki in 0..GROUP {
                    let byte = blob[nib_base + ki / 2];
                    let nib = if ki % 2 == 0 { byte & 0xF } else { (byte >> 4) & 0xF };
                    let w = sc * nib as f32 + zp;
                    let x = x_f32[x_base + g * GROUP + ki];
                    acc += w as f64 * x as f64;
                }
            }
            y[b * m + r] = acc as f32;
        }
    }
    y
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x as f64 - y as f64;
        num += d * d;
        den += y as f64 * y as f64;
    }
    (num / den.max(1e-30)).sqrt()
}
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x,&y)| (x - y).abs()).fold(0.0f32, f32::max)
}
fn max_abs_val(v: &[f32]) -> f32 { v.iter().map(|x| x.abs()).fold(0.0f32, f32::max) }

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a,b| a.partial_cmp(b).unwrap());
    v[v.len()/2]
}
fn minv(v: &[f64]) -> f64 { v.iter().cloned().fold(f64::INFINITY, f64::min) }
fn maxv(v: &[f64]) -> f64 { v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => { eprintln!("bench_mq4c_slab_alignment: no GPU ({e}) — skipping"); return; }
    };
    eprintln!("bench_mq4c_slab_alignment: arch={}", gpu.arch);
    if !gpu.arch.starts_with("gfx12") {
        eprintln!("this bench targets gfx12 _bt kernels; skipping on {}", gpu.arch);
        return;
    }
    let gpr = K / GROUP;
    // logical weights ONCE, seeded, shared nibbles
    let w: Vec<f32> = (0..M * K).map(|i| {
        let u1 = prng(i, 0x1234_5678).max(1e-7);
        let u2 = prng(i, 0x9ABC_DEF0);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos() * 0.011
    }).collect();
    let (b_v1, b_mq4c) = pack_both(&w, M, K);
    // X for max batch, shared across batches (prefix used)
    let max_batch = *BATCHES.iter().max().unwrap();
    let x_all_f32: Vec<f32> = (0..max_batch * K).map(|i| prng(i, 0xC0FF_EE) * 2.0 - 1.0).collect();

    // compile _bt modules — each file contains bt4/8/12, we extract 8 and 12
    for &bv in BVS {
        let sym_v1 = format!("gemm_hfq4g256_residual_wmma_gfx12_bt{bv}");
        let sym_mq = format!("gemm_mq4cg256_residual_wmma_gfx12_bt{bv}");
        let mod_v1 = format!("bench_hfq4_bt{bv}");
        let mod_mq = format!("bench_mq4c_bt{bv}");
        match gpu.ensure_kernel_public(&mod_v1, HFQ4_BT_SRC, &sym_v1) {
            Ok(()) => eprintln!("compiled {sym_v1}"),
            Err(e) => eprintln!("compile {sym_v1} failed: {e}"),
        }
        match gpu.ensure_kernel_public(&mod_mq, MQ4C_BT_SRC, &sym_mq) {
            Ok(()) => eprintln!("compiled {sym_mq}"),
            Err(e) => eprintln!("compile {sym_mq} failed: {e}"),
        }
    }
    // also ensure bt4 present as sanity (not benched unless BVS includes 4)
    // quick check that at least one bt compiled
    // upload weights once
    let d_v1 = gpu.upload_raw(&b_v1, &[b_v1.len()]).unwrap();
    let d_mq4c = gpu.upload_raw(&b_mq4c, &[b_mq4c.len()]).unwrap();

    // correctness gate first, then timing
    println!("\ncorrectness gate: M={M} K={K} gpr={gpr} batches {:?}  BVs {:?}", BATCHES, BVS);
    println!("host ref: exact dequant of each container (f64 accum). worst rel-l2 and max_abs_err per arm:");
    println!("{:<14} {:<10} {:>12} {:>12} {:>10} {:>10}", "batch", "arm", "rel_l2", "max_abs", "max_ref", "status");
    let mut any_fail = false;
    for &batch in BATCHES {
        let x_f32 = &x_all_f32[..batch*K];
        // build F16 X blob for GPU
        let x_f16_bytes: Vec<u8> = {
            let mut v = Vec::with_capacity(batch*K*2);
            for &f in x_f32 {
                v.extend_from_slice(&f16_bits(f).to_le_bytes());
            }
            v
        };
        let d_x = gpu.upload_raw(&x_f16_bytes, &[x_f16_bytes.len()]).unwrap();
        let y_host_v1 = host_gemm(&b_v1, M, K, batch, x_f32, false);
        let y_host_mq = host_gemm(&b_mq4c, M, K, batch, x_f32, true);

        for &bv in BVS {
            for (label, is_mq, host_ref) in [
                (format!("v1_bt{bv}"), false, &y_host_v1),
                (format!("mq4c_bt{bv}"), true, &y_host_mq),
            ] {
                let sym = if is_mq {
                    format!("gemm_mq4cg256_residual_wmma_gfx12_bt{bv}")
                } else {
                    format!("gemm_hfq4g256_residual_wmma_gfx12_bt{bv}")
                };
                let d_y = gpu.zeros(&[batch*M], DType::F32).unwrap();
                let a_ptr = if is_mq { &d_mq4c } else { &d_v1 };
                let grid = [ ((M+15)/16) as u32, ((batch + 16*bv -1)/(16*bv)) as u32, 1];
                let block = [32u32,1,1];
                let mut kb = KernargBlob::new();
                kb.push_ptr(a_ptr.buf.as_ptr());
                kb.push_ptr(d_x.buf.as_ptr());
                kb.push_ptr(d_y.buf.as_ptr());
                kb.push_i32(M as i32);
                kb.push_i32(K as i32);
                kb.push_i32(batch as i32);
                let r = gpu.launch_kernel_blob(&sym, grid, block, 0, kb.as_mut_slice());
                if let Err(e) = r {
                    println!("{:<14} {:<10} {:>12} {:>12} {:>10} {:>10}", format!("b={batch}"), label, "-", "-", "-", format!("LAUNCH FAIL {e}"));
                    any_fail = true;
                    continue;
                }
                gpu.hip.device_synchronize().unwrap();
                let got = gpu.download_f32(&d_y).unwrap();
                let rel = rel_l2(&got, host_ref);
                let mae = max_abs_err(&got, host_ref);
                let mrv = max_abs_val(host_ref);
                let status_str = if rel < 5e-3 { "PASS" } else { any_fail = true; "FAIL" };
                println!("{:<14} {:<10} {:>12.3e} {:>12.3e} {:>10.2e} {:>10}", format!("b={batch}"), label, rel, mae, mrv, status_str);
                if rel >= 1e-2 {
                    eprintln!("  WARNING: large rel {rel:.3e} for {label} batch {batch} bv {bv} — wrong container/stride would land here (qt=44 bug decoded to ~1e-14)");
                }
            }
        }
        // explicit cross-check: v1 vs mq4c GPU results should be close (header rounding only ~1e-3) not ~1e-14
        // free d_x implicitly via pool; keep for next iter by dropping explicitly?
        // GpuTensor drop handled by pool? we use pool alloc, need free? upload_raw uses malloc not pool, but zeros uses pool.
        // We leak d_x for simplicity (process exit frees); but to avoid OOM over batches, we can pool-free y etc already.
        let _ = gpu.free_tensor(gpu.upload_raw(&[], &[]).unwrap()); // no-op to avoid unused warning suppression
        // need to free d_x which is raw malloc: use hip.free? Gpu::free_tensor handles raw? It routes via pool vs raw check.
        // upload_raw uses hip.malloc directly, not pool, but free_tensor will treat as non-pool and return to pool incorrectly.
        // For this bench, just leak small  ~ max 32*5120*2=327k per batch *4 = 1.3M, negligible.
    }
    if any_fail {
        eprintln!("correctness gate: SOME ARMS FAILED — timing below is not a win");
    } else {
        eprintln!("correctness gate: all arms PASS (rel<5e-3 vs own host ref)");
    }

    // ── timing ── device-side events, interleaved sample-by-sample
    println!("\ntiming: M={M} K={K}  warmup={WARMUP} iters={ITERS} runs={RUNS}  device events (min is lead statistic, thermal drift ~6%)");
    for &batch in BATCHES {
        let x_f32 = &x_all_f32[..batch*K];
        let x_f16_bytes: Vec<u8> = {
            let mut v = Vec::with_capacity(batch*K*2);
            for &f in x_f32 { v.extend_from_slice(&f16_bits(f).to_le_bytes()); }
            v
        };
        let d_x = gpu.upload_raw(&x_f16_bytes, &[x_f16_bytes.len()]).unwrap();
        let weight_bytes_v1 = (M * gpr * V1_BYTES) as f64;
        let weight_bytes_mq = (M * gpr * MQ4C_BYTES) as f64;
        // per-arm samples map: key = label -> Vec<us per launch>
        struct Arm { label: String, sym: String, is_mq: bool, bv: usize, samples: Vec<f64> }
        let mut arms: Vec<Arm> = Vec::new();
        for &bv in BVS {
            arms.push(Arm{ label: format!("v1_bt{bv}"), sym: format!("gemm_hfq4g256_residual_wmma_gfx12_bt{bv}"), is_mq: false, bv, samples: Vec::new() });
            arms.push(Arm{ label: format!("mq4c_bt{bv}"), sym: format!("gemm_mq4cg256_residual_wmma_gfx12_bt{bv}"), is_mq: true, bv, samples: Vec::new() });
        }
        // warmups
        for arm in &arms {
            let d_y = gpu.zeros(&[batch*M], DType::F32).unwrap();
            let a_ptr = if arm.is_mq { &d_mq4c } else { &d_v1 };
            let grid = [ ((M+15)/16) as u32, ((batch + 16*arm.bv -1)/(16*arm.bv)) as u32, 1];
            let block = [32u32,1,1];
            for _ in 0..WARMUP {
                let mut kb = KernargBlob::new();
                kb.push_ptr(a_ptr.buf.as_ptr());
                kb.push_ptr(d_x.buf.as_ptr());
                kb.push_ptr(d_y.buf.as_ptr());
                kb.push_i32(M as i32);
                kb.push_i32(K as i32);
                kb.push_i32(batch as i32);
                gpu.launch_kernel_blob(&arm.sym, grid, block, 0, kb.as_mut_slice()).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
        }
        // interleaved runs
        for _run in 0..RUNS {
            for arm in &mut arms {
                let d_y = gpu.zeros(&[batch*M], DType::F32).unwrap();
                let a_ptr = if arm.is_mq { &d_mq4c } else { &d_v1 };
                let grid = [ ((M+15)/16) as u32, ((batch + 16*arm.bv -1)/(16*arm.bv)) as u32, 1];
                let block = [32u32,1,1];
                // pre-sync to flush
                gpu.hip.device_synchronize().unwrap();
                let start = gpu.hip.event_create().unwrap();
                let stop = gpu.hip.event_create().unwrap();
                gpu.hip.event_record(&start, None).unwrap();
                for _ in 0..ITERS {
                    let mut kb = KernargBlob::new();
                    kb.push_ptr(a_ptr.buf.as_ptr());
                    kb.push_ptr(d_x.buf.as_ptr());
                    kb.push_ptr(d_y.buf.as_ptr());
                    kb.push_i32(M as i32);
                    kb.push_i32(K as i32);
                    kb.push_i32(batch as i32);
                    gpu.launch_kernel_blob(&arm.sym, grid, block, 0, kb.as_mut_slice()).unwrap();
                }
                gpu.hip.event_record(&stop, None).unwrap();
                gpu.hip.event_synchronize(&stop).unwrap();
                let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
                let us = ms as f64 * 1000.0 / ITERS as f64;
                arm.samples.push(us);
                gpu.hip.event_destroy(start).unwrap();
                gpu.hip.event_destroy(stop).unwrap();
            }
        }
        // report
        let bytes_for = |is_mq: bool| -> f64 {
            let w = if is_mq { weight_bytes_mq } else { weight_bytes_v1 };
            // bytes = weight (read once) + X fp16 (batch*K*2) + Y F32 read+write (batch*M*4*2 for residual +=)
            w + (batch*K*2) as f64 + (batch*M*4*2) as f64
        };
        println!("\nbatch={batch}  (grid_y = ceil(batch/(16*BV)))");
        println!("{:<12} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}", "arm", "min_us", "med_us", "max_us", "GB/s", "vs_v1", "samples");
        // need vs_v1 per BV: ratio = mq_min / v1_min same BV
        let mut v1_min_by_bv: std::collections::HashMap<usize,f64> = std::collections::HashMap::new();
        for arm in &arms {
            if !arm.is_mq {
                v1_min_by_bv.insert(arm.bv, minv(&arm.samples));
            }
        }
        for arm in &mut arms {
            let mn = minv(&arm.samples);
            let md = median(arm.samples.clone());
            let mx = maxv(&arm.samples);
            let gbs = bytes_for(arm.is_mq) / (mn * 1e-6) / 1e9;
            let base = v1_min_by_bv.get(&arm.bv).cloned().unwrap_or(f64::NAN);
            let ratio = if arm.is_mq && base.is_finite() { format!("{:.4}", mn/base) } else { "-".to_string() };
            let ss: Vec<String> = arm.samples.iter().map(|v| format!("{:.1}", v)).collect();
            println!("{:<12} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9} {:>10}", arm.label, mn, md, mx, gbs, ratio, ss.join(","));
        }
        println!("GB/s = (weight + X_fp16 + Y_residual*2) / min_us. vs_v1 = mq_min / v1_min same BV ( <1 means mq faster).");
        // per-batch summary sentence
        for &bv in BVS {
            let v1 = arms.iter().find(|a| !a.is_mq && a.bv==bv).unwrap();
            let mq = arms.iter().find(|a| a.is_mq && a.bv==bv).unwrap();
            let r = minv(&mq.samples)/minv(&v1.samples);
            let delta = (r-1.0)*100.0;
            let verdict = if r < 0.99 { "mq faster" } else if r > 1.01 { "mq slower" } else { "within noise (±1%)" };
            println!("  bv={bv}: mq vs v1 = {r:.4} ({delta:+.2}%) — {verdict}");
        }
    }
    println!("\nnote: arm C (a16, 528 B slab) DROPPED — ldsstage kernels are not on the dispatch path, so there is no misalignment to fix. This bench isolates the _bt kernels that actually ran.");
    println!("If mq vs v1 is within noise, the 2.9% model-level prefill gap reproduces elsewhere (not this kernel).");
}
