// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ4V2 residual split-K LDS parity + timing sweep on exact gfx1100.
//!
//! Loads REAL weight bytes for layer-0 out_proj (M=5120,K=6144) and down_proj
//! (M=5120,K=17408) from qwen3.8-27b.mq4, random finite F32 X at N=1,8,16,
//! identical nonzero Y init on both arms. Reference: the historical base
//! `gemm_mq4g256v2_residual_wmma` forced via capture_mode (skips the
//! production ksplit tier). Each runnable kw in {2,4,8} (skipped when
//! (K/256) % kw != 0, by kernel-design contract): relL2, max-abs, finite
//! check, then timing (32 warmups, 200 launches/sample, 3 samples interleaved
//! arm-by-arm, min+median). Exit nonzero on any relL2 > 1e-5 or non-finite.
//! Split-K changes fp32 association order, so bit-exactness is NOT required.
//! On any other arch the harness SKIPs cleanly (exit 0, no GPU work).

use rdna_compute::{DType, Gpu};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const MODEL_DEFAULT: &str = "/home/kaden/.hipfire/models/qwen3.8-27b.mq4";
const NS: [usize; 3] = [1, 8, 16];
const KWS: [usize; 3] = [2, 4, 8];
const WARMUP: usize = 32;
const LAUNCHES: usize = 200;
const SAMPLES: usize = 3;

struct HfqTensor {
    name: String,
    shape: Vec<u32>,
    data_off: usize,
    data_len: usize,
}

fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ])
}

/// Minimal HFQ index parse mirroring HfqFile::open_at_offset (hfq.rs:445+).
fn parse_hfq_index(path: &std::path::Path) -> (String, Vec<HfqTensor>) {
    let canon = std::fs::canonicalize(path)
        .unwrap_or_else(|e| panic!("canonicalize {}: {e}", path.display()));
    let mut f = File::open(&canon).expect("open hfq");
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr).expect("read hfq header");
    assert_eq!(&hdr[0..4], b"HFQM", "not an HFQ container");
    let n_tensors = u32le(&hdr[12..16]) as usize;
    let metadata_offset = u64le(&hdr[16..24]) as usize;
    let data_offset = u64le(&hdr[24..32]) as usize;
    let region_len = data_offset - metadata_offset;
    let mut region = vec![0u8; region_len];
    f.seek(SeekFrom::Start(metadata_offset as u64)).unwrap();
    f.read_exact(&mut region).expect("read hfq meta+index");
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    let mut json_end = 0usize;
    for (i, &b) in region.iter().enumerate() {
        if esc {
            esc = false;
            continue;
        }
        if b == b'\\' && in_str {
            esc = true;
            continue;
        }
        if b == b'"' {
            in_str = !in_str;
            continue;
        }
        if !in_str {
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
    }
    assert!(json_end > 0, "metadata JSON not brace-terminated");
    let mut pos = json_end;
    let idx_n = u32le(&region[pos..pos + 4]) as usize;
    assert_eq!(idx_n, n_tensors, "index count != header count");
    pos += 4;
    let mut tensors = Vec::with_capacity(n_tensors);
    let mut cum = data_offset;
    for _ in 0..n_tensors {
        let nl = u16::from_le_bytes([region[pos], region[pos + 1]]) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&region[pos..pos + nl]).to_string();
        pos += nl;
        pos += 1; // qt
        let nd = region[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(nd);
        for _ in 0..nd {
            shape.push(u32le(&region[pos..pos + 4]));
            pos += 4;
        }
        pos += 4; // group_size
        let data_len = u64le(&region[pos..pos + 8]) as usize;
        pos += 8;
        tensors.push(HfqTensor { name, shape, data_off: cum, data_len });
        cum += data_len;
    }
    (canon.display().to_string(), tensors)
}

fn find_tensor<'a>(tensors: &'a [HfqTensor], suffix: &str) -> &'a HfqTensor {
    tensors
        .iter()
        .find(|t| t.name.ends_with(suffix))
        .unwrap_or_else(|| panic!("tensor not found: *{suffix}"))
}

fn read_tensor_bytes(path: &str, t: &HfqTensor) -> Vec<u8> {
    let mut f = File::open(path).expect("reopen hfq for payload");
    f.seek(SeekFrom::Start(t.data_off as u64)).unwrap();
    let mut buf = vec![0u8; t.data_len];
    f.read_exact(&mut buf).expect("read tensor payload");
    buf
}

fn is_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn variance(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        if num == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (num / den).sqrt()
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn random_f32(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut st = seed | 1;
    (0..n)
        .map(|_| {
            let r = (xorshift64(&mut st) >> 11) as f64 / (u64::MAX >> 11) as f64;
            (lo + (r as f32) * (hi - lo)).clamp(lo, hi)
        })
        .collect()
}

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().unwrap();
}

fn htod_f32(gpu: &Gpu, t: &rdna_compute::GpuTensor, v: &[f32]) {
    gpu.hip
        .memcpy_htod(
            &t.buf,
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) },
        )
        .expect("htod f32");
    sync(gpu);
}

/// Time LAUNCHES launches of `launch` (device-sync around, per-launch us).
fn time_batch(gpu: &mut Gpu, launch: &mut dyn FnMut(&mut Gpu)) -> f64 {
    sync(gpu);
    let t0 = std::time::Instant::now();
    for _ in 0..LAUNCHES {
        launch(gpu);
    }
    sync(gpu);
    t0.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    let arch = gpu.arch.clone();
    if !(gpu.arch_caps.is_gfx1100() && arch == "gfx1100") {
        eprintln!("SKIP: arch {arch} is not exact gfx1100 — harness requires gfx1100 only");
        return;
    }
    eprintln!("arch {arch} confirmed exact gfx1100 — running residual ksplit parity (Y+=W@X)");
    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some — harness requires no capture");
        return;
    }

    let model_arg = std::env::args().nth(1);
    let model_path = std::path::PathBuf::from(model_arg.as_deref().unwrap_or(MODEL_DEFAULT));
    let (canon, tensors) = parse_hfq_index(&model_path);
    eprintln!("model: {canon}");

    struct Proj {
        label: &'static str,
        suffix: &'static str,
        m: usize,
        k: usize,
    }
    let projs = [
        Proj { label: "out_proj", suffix: "layers.0.linear_attn.out_proj.weight", m: 5120, k: 6144 },
        Proj { label: "down_proj", suffix: "layers.0.mlp.down_proj.weight", m: 5120, k: 17408 },
    ];

    println!(
        "{:>10} {:>3} {:>4} {:>12} {:>12} {:>7} {:>10} {:>10}",
        "proj", "N", "kw", "relL2", "maxAbs", "finite", "min_us", "med_us"
    );

    let mut all_ok = true;
    for p in &projs {
        let t = find_tensor(&tensors, p.suffix);
        let m = t.shape[0] as usize;
        let k = t.shape[1] as usize;
        assert_eq!(m, p.m, "{}: M {m} != {}", p.label, p.m);
        assert_eq!(k, p.k, "{}: K {k} != {}", p.label, p.k);
        let expect = m * (k / 256) * 136;
        assert_eq!(t.data_len, expect, "{}: size {} != {expect}", p.label, t.data_len);
        eprintln!("{}: {} M={m} K={k} bytes={}", p.label, t.name, t.data_len);
        let payload = read_tensor_bytes(&canon, t);
        let d_a = gpu.upload_raw(&payload, &[m, k]).expect("upload weights");
        let g = k / 256;
        let runnable: Vec<usize> =
            KWS.iter().copied().filter(|&kw| g >= kw && g % kw == 0).collect();

        for &n in &NS {
            let x_host = random_f32(n * k, 0x1234_9E37 + k as u64, -1.0, 1.0);
            // Identical nonzero Y init on both arms (fused Y += W@X).
            let y_init = random_f32(n * m, 0xBEEF_1234 + n as u64, -0.5, 1.5);
            let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
            htod_f32(&gpu, &d_x, &x_host);

            // Reference: historical base kernel, production ksplit skipped
            // via capture_mode for this launch only.
            let d_y_ref = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y ref");
            htod_f32(&gpu, &d_y_ref, &y_init);
            let saved_capture = gpu.graphs.capture_mode;
            gpu.graphs.capture_mode = true;
            sync(&gpu);
            let r = gpu.gemm_mq4g256v2_residual_wmma(&d_a, &d_x, &d_y_ref, m, k, n);
            gpu.graphs.capture_mode = saved_capture;
            r.expect("base gemm_mq4g256v2_residual_wmma failed");
            sync(&gpu);
            let y_ref = gpu.download_f32(&d_y_ref).expect("download ref");
            assert!(is_finite(&y_ref), "ref not finite {} N={n}", p.label);
            assert!(variance(&y_ref) > 1e-12, "ref degenerate {} N={n}", p.label);

            // One Y tensor per runnable kw arm (kept for the timing phase).
            let mut arms: Vec<(usize, rdna_compute::GpuTensor)> = Vec::new();
            let mut par: Vec<(usize, f64, f32, bool)> = Vec::new();
            for &kw in &KWS {
                if !runnable.contains(&kw) {
                    println!(
                        "{:>10} {:>3} {:>4}  SKIP (K/256={g} not divisible by kw={kw})",
                        p.label, n, kw
                    );
                    continue;
                }
                let d_y = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y");
                htod_f32(&gpu, &d_y, &y_init);
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds(&d_a, &d_x, &d_y, m, k, n, kw)
                    .unwrap_or_else(|e| panic!("ksplit kw={kw} launch failed: {e:?}"));
                sync(&gpu);
                let y_got = gpu.download_f32(&d_y).expect("download ksplit");
                let finite = is_finite(&y_got);
                let r2 = rel_l2(&y_got, &y_ref);
                let ma = max_abs_diff(&y_got, &y_ref);
                let ok = finite && r2 <= 1e-5;
                if !ok {
                    all_ok = false;
                    eprintln!("  FAIL parity {} N={n} kw={kw}: relL2={r2:.3e} maxAbs={ma:.3e} finite={finite}", p.label);
                }
                arms.push((kw, d_y));
                par.push((kw, r2, ma, finite));
            }

            // Warmups per arm (right kw), then SAMPLES interleaved arm-by-arm.
            for (i, (_, d_y)) in arms.iter().enumerate() {
                let kw = par[i].0;
                htod_f32(&gpu, d_y, &y_init);
                for _ in 0..WARMUP {
                    gpu.gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds(&d_a, &d_x, d_y, m, k, n, kw)
                        .unwrap();
                }
            }
            sync(&gpu);
            let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(SAMPLES); arms.len()];
            for _ in 0..SAMPLES {
                for (i, (_, d_y)) in arms.iter().enumerate() {
                    let kw = par[i].0;
                    htod_f32(&gpu, d_y, &y_init);
                    samples[i].push(time_batch(&mut gpu, &mut |gm: &mut Gpu| {
                        gm.gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds(&d_a, &d_x, d_y, m, k, n, kw)
                            .unwrap()
                    }));
                }
            }
            for (i, (kw, r2, ma, finite)) in par.iter().enumerate() {
                let mut us = samples[i].clone();
                us.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = median(us.clone());
                let ok = *finite && *r2 <= 1e-5;
                let status = if ok { "OK" } else { "FAIL" };
                println!(
                    "{:>10} {:>3} {:>4} {:>12.3e} {:>12.3e} {:>7} {:>10.1} {:>10.1} [{status}]",
                    p.label, n, kw, r2, ma, finite, us[0], med
                );
            }
        }
    }

    if all_ok {
        eprintln!("\nPASS: every runnable (proj, N, kw) relL2<=1e-5, all finite, Y+=W@X preserved");
    } else {
        eprintln!("\nFAIL: one or more parity checks violated relL2<=1e-5 or finiteness");
        std::process::exit(1);
    }
}
