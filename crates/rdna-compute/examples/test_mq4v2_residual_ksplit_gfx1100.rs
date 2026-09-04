// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ4V2 residual split-K LDS parity + timing sweep on exact gfx1100, plus the
//! LDS-staged (gfx12-port) `ldsstage` arm.
//!
//! Loads REAL weight bytes for layer-0 out_proj (M=5120,K=6144) and down_proj
//! (M=5120,K=17408) from qwen3.8-27b.mq4, random finite F32 X at N=1,8,16,
//! identical nonzero Y init on both arms. Reference: the historical base
//! `gemm_mq4g256v2_residual_wmma` forced via the residual_ksplit_off kill
//! switch (the tier is capture-safe, so capture_mode no longer diverts it).
//! (K/256) % kw != 0, by kernel-design contract): relL2, max-abs, finite
//! check, then timing (32 warmups, 200 launches/sample, 3 samples interleaved
//! arm-by-arm, min+median). Exit nonzero on any relL2(ks, base) > 5e-5 or
//! non-finite. Split-K changes fp32 association order, so bit-exactness is
//! NOT required. The `ldsstage` arm (kw column prints `lds`, requires
//! K % 512 == 0) runs the same gate and the same timing discipline against
//! the same base reference and f64 floor.
//! The `pf2`/`pf4` arms (exp/ks4-prefetch) run the depth-2/depth-4
//! software-prefetched ks4 variants wherever kw=4 is runnable and gate EXACT
//! bitwise equality with the ks4 kw=4 single-launch output (any nonzero
//! bitwise-diff count fails the run), plus the same relL2/timing columns.
//!
//! Association-floor documentation: for each (shape, N) the harness also
//! builds an f64 host reference — real weights dequantized with the exact
//! kernel formula (dual fp16 headers, kt<8 -> s0/z0 else s1/z1, nibble
//! unpacking, sc*nibble+zp), X rounded to fp16 exactly as the
//! `convert_f32_to_f16` staging kernel does (hardware cvt = RN-even), Y init
//! exact, accumulation in f64 ascending-K order — and prints
//! relL2(base,f64) next to relL2(ks,base) and relL2(ks,f64), proving the
//! split-K delta is the fp32 association floor and ks is no farther from
//! truth than base is. Caveat [INFERENCE]: the reference evaluates the
//! dequant sc*nibble+zp in f64 while the kernel folds it in fp16; that
//! f16-rounding (~2^-11 rel on weights) is common to base and ks alike, so
//! it cannot bias the ks-vs-base comparison that the gate rests on.
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
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
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
        tensors.push(HfqTensor {
            name,
            shape,
            data_off: cum,
            data_len,
        });
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

/// Parity tolerance: split-K reassociation noise floor is 1.1e-5..3.5e-5
/// (measured), so the gate sits at 5e-5, not 1e-5.
const PARITY_TOL: f64 = 5e-5;
/// Absolute-diff threshold for the argmax-support statistic: fraction of
/// output elements whose |ks - base| exceeds this.
const BIG_DIFF: f64 = 1e-3;

/// f32 -> IEEE binary16 bits, round-to-nearest-even. Mirrors the hardware
/// cvt used by the `convert_f32_to_f16` X-staging kernel (NOT the
/// round-toward-zero `half_from_f32` test helper). X here is finite in
/// [-1, 1]; inf/nan map to the inf pattern and never occur (asserted).
fn f32_to_f16_bits_rne(v: f32) -> u16 {
    debug_assert!(v.is_finite());
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return s | 0x7c00; // inf/nan input: pin to inf (unreachable here)
    }
    if e == 0 {
        return s; // f32 subnormal << f16 min subnormal: underflows to zero
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return s | 0x7c00; // overflow to inf (unreachable for |X| <= 1)
    }
    if e16 >= 1 {
        // Normal f16: round 23-bit mantissa to 10 bits, RN-even.
        let half = (m >> 13) as u16;
        let rest = m & 0x1fff;
        let round_up = rest > 0x1000 || (rest == 0x1000 && (half & 1) == 1);
        let mut h = half + round_up as u16;
        let mut e16 = e16;
        if h == 0x400 {
            h = 0;
            e16 += 1;
        }
        if e16 >= 31 {
            return s | 0x7c00;
        }
        return s | ((e16 as u16) << 10) | (h & 0x3ff);
    }
    // Subnormal f16 (e16 <= 0): h = round(m32 * 2^-sh), RN-even, u64 math so
    // large shifts cannot panic. e in 1..=112 here, so sh = 126 - e >= 14.
    let m32 = (1u64 << 23) | m as u64;
    let sh = (126 - e) as u32;
    let (q, r) = if sh >= 64 {
        (0u64, m32)
    } else if sh == 0 {
        (m32, 0)
    } else {
        (m32 >> sh, m32 & ((1u64 << sh) - 1))
    };
    let half_bit = if sh == 0 || sh > 64 {
        0
    } else {
        1u64 << (sh - 1)
    };
    let round_up = if sh == 0 {
        false
    } else if sh > 64 {
        m32 != 0
    } else {
        r > half_bit || (r == half_bit && (q & 1) == 1)
    };
    let h = q + round_up as u64;
    if h >= 0x400 {
        s | (1u16 << 10) // rounded up into the smallest normal
    } else {
        s | (h as u16)
    }
}

/// IEEE binary16 bits -> f64, exact.
fn f16_to_f64(bits: u16) -> f64 {
    let s = ((bits >> 15) & 1) as f64;
    let e = ((bits >> 10) & 0x1f) as i32;
    let m = (bits & 0x3ff) as f64;
    let v = if e == 0 {
        m * 2f64.powi(-24)
    } else if e == 31 {
        f64::INFINITY // unreachable: weights/X headers are finite
    } else {
        (m + 1024.0) * 2f64.powi(e - 15 - 10)
    };
    if s == 0.0 {
        v
    } else {
        -v
    }
}

fn rel_l2_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        num += d * d;
        den += y * y;
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

fn rms_f64(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

/// f64 host truth for one (weights, X, Y-init) triple, layout col*M+row.
///
/// Dequant mirrors `gemm_mq4g256v2_residual_wmma.hip` exactly: per row, per
/// 136 B group, dual fp16 headers (kt<8 -> s0/z0 from gp+0, else s1/z1 from
/// gp+4), nibble unpacking (kt*16+i, pk0/pk1 at gp+8+k_off/2), weight =
/// sc*nibble+zp evaluated in f64. X is f16-rounded (RN-even, as the staging
/// kernel does), Y init is exact, accumulation is f64 in ascending-K order.
fn host_f64_ref(
    payload: &[u8],
    x_host: &[f32],
    y_init: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f64> {
    assert_eq!(x_host.len(), n * k);
    assert_eq!(y_init.len(), n * m);
    let g = k / 256;
    // X through the same f16 rounding the device staging kernel applies.
    let xr: Vec<f64> = x_host
        .iter()
        .map(|&v| f16_to_f64(f32_to_f16_bits_rne(v)))
        .collect();
    let mut y: Vec<f64> = y_init.iter().map(|&v| v as f64).collect();
    let mut w256 = [0f64; 256];
    for r in 0..m {
        let row_base = r * g * 136;
        for gg in 0..g {
            let gp = row_base + gg * 136;
            let ha = u32le(&payload[gp..gp + 4]);
            let hb = u32le(&payload[gp + 4..gp + 8]);
            let sc0 = f16_to_f64((ha & 0xffff) as u16);
            let zp0 = f16_to_f64((ha >> 16) as u16);
            let sc1 = f16_to_f64((hb & 0xffff) as u16);
            let zp1 = f16_to_f64((hb >> 16) as u16);
            for kt in 0..16 {
                let (sc, zp) = if kt < 8 { (sc0, zp0) } else { (sc1, zp1) };
                let k_off = kt * 16;
                let pk0 = u32le(&payload[gp + 8 + k_off / 2..gp + 12 + k_off / 2]);
                let pk1 = u32le(&payload[gp + 12 + k_off / 2..gp + 16 + k_off / 2]);
                for i in 0..16 {
                    let pk = if i < 8 { pk0 } else { pk1 };
                    let nib = ((pk >> ((i % 8) * 4)) & 0xf) as f64;
                    w256[kt * 16 + i] = sc * nib + zp;
                }
            }
            for col in 0..n {
                let xrow = &xr[col * k + gg * 256..col * k + gg * 256 + 256];
                let mut s = 0f64;
                for i in 0..256 {
                    s += w256[i] * xrow[i];
                }
                y[col * m + r] += s;
            }
        }
    }
    y
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
        .memcpy_htod(&t.buf, unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
        })
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
        Proj {
            label: "out_proj",
            suffix: "layers.0.linear_attn.out_proj.weight",
            m: 5120,
            k: 6144,
        },
        Proj {
            label: "down_proj",
            suffix: "layers.0.mlp.down_proj.weight",
            m: 5120,
            k: 17408,
        },
    ];

    println!(
        "{:>10} {:>3} {:>4} {:>12} {:>12} {:>7} {:>12} {:>12} {:>12} {:>12} {:>9} {:>10} {:>10}",
        "proj",
        "N",
        "kw",
        "r(ks,base)",
        "maxAbs",
        "finite",
        "r(base,f64)",
        "r(ks,f64)",
        "mx(ks,f64)",
        "rmsRef",
        "fr>1e-3",
        "min_us",
        "med_us"
    );

    let mut all_ok = true;
    for p in &projs {
        let t = find_tensor(&tensors, p.suffix);
        let m = t.shape[0] as usize;
        let k = t.shape[1] as usize;
        assert_eq!(m, p.m, "{}: M {m} != {}", p.label, p.m);
        assert_eq!(k, p.k, "{}: K {k} != {}", p.label, p.k);
        let expect = m * (k / 256) * 136;
        assert_eq!(
            t.data_len, expect,
            "{}: size {} != {expect}",
            p.label, t.data_len
        );
        eprintln!("{}: {} M={m} K={k} bytes={}", p.label, t.name, t.data_len);
        let payload = read_tensor_bytes(&canon, t);
        let d_a = gpu.upload_raw(&payload, &[m, k]).expect("upload weights");
        let g = k / 256;
        let runnable: Vec<usize> = KWS
            .iter()
            .copied()
            .filter(|&kw| g >= kw && g % kw == 0)
            .collect();

        for &n in &NS {
            let x_host = random_f32(n * k, 0x1234_9E37 + k as u64, -1.0, 1.0);
            // Identical nonzero Y init on both arms (fused Y += W@X).
            let y_init = random_f32(n * m, 0xBEEF_1234 + n as u64, -0.5, 1.5);
            let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
            htod_f32(&gpu, &d_x, &x_host);

            // Reference: historical base kernel. The ksplit tier is
            // capture-safe now, so capture_mode no longer forces the base;
            // force it via the HIPFIRE_RESIDUAL_KSPLIT_OFF kill switch
            // (flags Arc swap for this launch only).
            let d_y_ref = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y ref");
            htod_f32(&gpu, &d_y_ref, &y_init);
            let saved_flags = gpu.flags.clone();
            gpu.flags = std::sync::Arc::new(rdna_compute::FeatureFlags {
                residual_ksplit_off: true,
                ..(*saved_flags).clone()
            });
            sync(&gpu);
            let r = gpu.gemm_mq4g256v2_residual_wmma(&d_a, &d_x, &d_y_ref, m, k, n);
            gpu.flags = saved_flags;
            r.expect("base gemm_mq4g256v2_residual_wmma failed");
            sync(&gpu);
            let y_ref = gpu.download_f32(&d_y_ref).expect("download ref");
            assert!(is_finite(&y_ref), "ref not finite {} N={n}", p.label);
            assert!(variance(&y_ref) > 1e-12, "ref degenerate {} N={n}", p.label);
            // Association floor: f64 truth for this exact (weights, X, Y-init).
            let t_f64 = std::time::Instant::now();
            let y_f64 = host_f64_ref(&payload, &x_host, &y_init, m, k, n);
            let f64_ms = t_f64.elapsed().as_secs_f64() * 1e3;
            let y_ref64: Vec<f64> = y_ref.iter().map(|&v| v as f64).collect();
            let r_base_f64 = rel_l2_f64(&y_ref64, &y_f64);
            let rms_ref = rms_f64(&y_f64);
            let ma_base_f64 = y_ref64
                .iter()
                .zip(y_f64.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            eprintln!("  f64 truth {} N={n}: relL2(base,f64)={r_base_f64:.3e} maxAbs(base,f64)={ma_base_f64:.3e} rmsRef={rms_ref:.3e} ({f64_ms:.0} ms host)", p.label);

            // One Y tensor per runnable kw arm (kept for the timing phase).
            let mut arms: Vec<(usize, rdna_compute::GpuTensor)> = Vec::new();
            let mut par: Vec<(usize, f64, f32, bool, f64, f64, f64)> = Vec::new();
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
                let y_got64: Vec<f64> = y_got.iter().map(|&v| v as f64).collect();
                let r_ks_f64 = rel_l2_f64(&y_got64, &y_f64);
                let ma_ks_f64 = y_got64
                    .iter()
                    .zip(y_f64.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f64, f64::max);
                let bigfrac = y_got
                    .iter()
                    .zip(y_ref.iter())
                    .filter(|(a, b)| (**a - **b).abs() as f64 > BIG_DIFF)
                    .count() as f64
                    / y_got.len() as f64;
                let ok = finite && r2 <= PARITY_TOL;
                if !ok {
                    all_ok = false;
                    eprintln!("  FAIL parity {} N={n} kw={kw}: relL2(ks,base)={r2:.3e} maxAbs={ma:.3e} finite={finite}", p.label);
                }
                arms.push((kw, d_y));
                par.push((kw, r2, ma, finite, r_ks_f64, ma_ks_f64, bigfrac));
            }

            // Warmups per arm (right kw), then SAMPLES interleaved arm-by-arm.
            for (i, (_, d_y)) in arms.iter().enumerate() {
                let kw = par[i].0;
                htod_f32(&gpu, d_y, &y_init);
                for _ in 0..WARMUP {
                    gpu.gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds(
                        &d_a, &d_x, d_y, m, k, n, kw,
                    )
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
                        gm.gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds(
                            &d_a, &d_x, d_y, m, k, n, kw,
                        )
                        .unwrap()
                    }));
                }
            }
            for (i, (kw, r2, ma, finite, r_ks_f64, ma_ks_f64, bigfrac)) in par.iter().enumerate() {
                let mut us = samples[i].clone();
                us.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = median(us.clone());
                let ok = *finite && *r2 <= PARITY_TOL;
                let status = if ok { "OK" } else { "FAIL" };
                println!(
                    "{:>10} {:>3} {:>4} {:>12.3e} {:>12.3e} {:>7} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e} {:>9.2e} {:>10.1} {:>10.1} [{status}]",
                    p.label, n, kw, r2, ma, finite, r_base_f64, r_ks_f64, ma_ks_f64, rms_ref, bigfrac, us[0], med
                );
            }
            // LDS-stage arm (gfx1100 port of the gfx12 ldsstage design): same
            // f64 floor + relL2 <= 5e-5 gate, same timing discipline (32
            // warmups, 200 launches/sample, 3 samples, min+median).
            if k % 512 == 0 {
                let d_y_lds = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y lds");
                htod_f32(&gpu, &d_y_lds, &y_init);
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage(&d_a, &d_x, &d_y_lds, m, k, n)
                    .unwrap_or_else(|e| panic!("ldsstage launch failed: {e:?}"));
                sync(&gpu);
                let y_lds = gpu.download_f32(&d_y_lds).expect("download ldsstage");
                let finite_lds = is_finite(&y_lds);
                let r_lds_base = rel_l2(&y_lds, &y_ref);
                let ma_lds = max_abs_diff(&y_lds, &y_ref);
                let y_lds64: Vec<f64> = y_lds.iter().map(|&v| v as f64).collect();
                let r_lds_f64 = rel_l2_f64(&y_lds64, &y_f64);
                let ma_lds_f64 = y_lds64
                    .iter()
                    .zip(y_f64.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f64, f64::max);
                let bigfrac_lds = y_lds
                    .iter()
                    .zip(y_ref.iter())
                    .filter(|(a, b)| (**a - **b).abs() as f64 > BIG_DIFF)
                    .count() as f64
                    / y_lds.len() as f64;
                let ok_lds = finite_lds && r_lds_base <= PARITY_TOL;
                if !ok_lds {
                    all_ok = false;
                    eprintln!("  FAIL parity {} N={n} ldsstage: relL2(lds,base)={r_lds_base:.3e} maxAbs={ma_lds:.3e} finite={finite_lds}", p.label);
                }
                htod_f32(&gpu, &d_y_lds, &y_init);
                for _ in 0..WARMUP {
                    gpu.gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage(
                        &d_a, &d_x, &d_y_lds, m, k, n,
                    )
                    .unwrap();
                }
                sync(&gpu);
                let mut us_lds: Vec<f64> = Vec::with_capacity(SAMPLES);
                for _ in 0..SAMPLES {
                    htod_f32(&gpu, &d_y_lds, &y_init);
                    us_lds.push(time_batch(&mut gpu, &mut |gm: &mut Gpu| {
                        gm.gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage(
                            &d_a, &d_x, &d_y_lds, m, k, n,
                        )
                        .unwrap()
                    }));
                }
                us_lds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med_lds = median(us_lds.clone());
                let status_lds = if ok_lds { "OK" } else { "FAIL" };
                println!(
                    "{:>10} {:>3} {:>4} {:>12.3e} {:>12.3e} {:>7} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e} {:>9.2e} {:>10.1} {:>10.1} [{status_lds}]",
                    p.label, n, "lds", r_lds_base, ma_lds, finite_lds, r_base_f64, r_lds_f64, ma_lds_f64, rms_ref, bigfrac_lds, us_lds[0], med_lds
                );
            } else {
                println!(
                    "{:>10} {:>3} {:>4}  SKIP (K % 512 != 0, ldsstage requires K % 512 == 0)",
                    p.label, n, "lds"
                );
            }
            // Prefetch arms (exp/ks4-prefetch, KILL EXPERIMENT): depth-2
            // `ks4_pf_lds` and depth-4 `ks4_pf4_lds` must be BIT-IDENTICAL to
            // ks4 kw=4 (same dequant/WMMA/reduce order — exact u32 equality,
            // not relL2). Fresh ks4 single-launch reference per (proj, N),
            // same timing discipline (32 warmups, 200 launches/sample,
            // 3 samples, min+median) against the same base reference.
            if runnable.contains(&4) {
                let d_y_ksref = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y ksref");
                htod_f32(&gpu, &d_y_ksref, &y_init);
                gpu.gemm_mq4g256v2_residual_wmma_gfx1100_ksplit_lds(&d_a, &d_x, &d_y_ksref, m, k, n, 4)
                    .expect("ks4 kw=4 ref launch failed");
                sync(&gpu);
                let y_ksref = gpu.download_f32(&d_y_ksref).expect("download ks4 ref");
                for &depth in &[2u8, 4u8] {
                    let tag = if depth == 2 { "pf2" } else { "pf4" };
                    let d_y_pf = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y pf");
                    htod_f32(&gpu, &d_y_pf, &y_init);
                    gpu.gemm_mq4g256v2_residual_wmma_gfx1100_ks4_pf_lds(&d_a, &d_x, &d_y_pf, m, k, n, depth)
                        .unwrap_or_else(|e| panic!("{tag} launch failed: {e:?}"));
                    sync(&gpu);
                    let y_pf = gpu.download_f32(&d_y_pf).expect("download pf");
                    let finite_pf = is_finite(&y_pf);
                    let ndiff = y_pf
                        .iter()
                        .zip(y_ksref.iter())
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    eprintln!("  pf-exact {} N={n} {tag}: bitwise_diffs_vs_ks4={ndiff}/{} finite={finite_pf}", p.label, y_pf.len());
                    let r_pf_base = rel_l2(&y_pf, &y_ref);
                    let ma_pf = max_abs_diff(&y_pf, &y_ref);
                    let y_pf64: Vec<f64> = y_pf.iter().map(|&v| v as f64).collect();
                    let r_pf_f64 = rel_l2_f64(&y_pf64, &y_f64);
                    let ma_pf_f64 = y_pf64
                        .iter()
                        .zip(y_f64.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f64, f64::max);
                    let bigfrac_pf = y_pf
                        .iter()
                        .zip(y_ref.iter())
                        .filter(|(a, b)| (**a - **b).abs() as f64 > BIG_DIFF)
                        .count() as f64
                        / y_pf.len() as f64;
                    let ok_pf = finite_pf && ndiff == 0 && r_pf_base <= PARITY_TOL;
                    if !ok_pf {
                        all_ok = false;
                        eprintln!("  FAIL parity {} N={n} {tag}: ndiff={ndiff} relL2(pf,base)={r_pf_base:.3e} maxAbs={ma_pf:.3e} finite={finite_pf}", p.label);
                    }
                    htod_f32(&gpu, &d_y_pf, &y_init);
                    for _ in 0..WARMUP {
                        gpu.gemm_mq4g256v2_residual_wmma_gfx1100_ks4_pf_lds(
                            &d_a, &d_x, &d_y_pf, m, k, n, depth,
                        )
                        .unwrap();
                    }
                    sync(&gpu);
                    let mut us_pf: Vec<f64> = Vec::with_capacity(SAMPLES);
                    for _ in 0..SAMPLES {
                        htod_f32(&gpu, &d_y_pf, &y_init);
                        us_pf.push(time_batch(&mut gpu, &mut |gm: &mut Gpu| {
                            gm.gemm_mq4g256v2_residual_wmma_gfx1100_ks4_pf_lds(
                                &d_a, &d_x, &d_y_pf, m, k, n, depth,
                            )
                            .unwrap()
                        }));
                    }
                    us_pf.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let med_pf = median(us_pf.clone());
                    let status_pf = if ok_pf { "OK" } else { "FAIL" };
                    println!(
                        "{:>10} {:>3} {:>4} {:>12.3e} {:>12.3e} {:>7} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e} {:>9.2e} {:>10.1} {:>10.1} [{status_pf}]",
                        p.label, n, tag, r_pf_base, ma_pf, finite_pf, r_base_f64, r_pf_f64, ma_pf_f64, rms_ref, bigfrac_pf, us_pf[0], med_pf
                    );
                }
            } else {
                println!(
                    "{:>10} {:>3} {:>4}  SKIP (kw=4 not runnable, pf requires (K/256)%4==0)",
                    p.label, n, "pf2/pf4"
                );
            }
        }
    }
    if all_ok {
        eprintln!("\nPASS: every runnable (proj, N, kw) relL2(ks,base)<=5e-5, ldsstage relL2(lds,base)<=5e-5, pf2/pf4 bit-identical to ks4 kw=4, all finite, Y+=W@X preserved");
    } else {
        eprintln!("\nFAIL: one or more parity checks violated relL2<=5e-5, pf bit-exactness, or finiteness");
        std::process::exit(1);
    }
}
