//! Deterministic CPU-vs-GPU parity for MQ4G256V2 (qt=44) fused prefill
//! WMMA paths — the gfx11 slice (`gemm_qkv`, `gemm_qkvza`, `gemm_gate_up`) —
//! plus N=1 scalar-decode oracles for `fused_qkv_hfq4g256_mq4v2`,
//! `fused_qkvza_hfq4g256_mq4v2`, and `fused_gate_up_hfq4g256_mq4v2`.
//!
//! ## Why this harness exists
//!
//! Mirrors `mq4v2_residual_parity` but covers the three fused families that
//! were until this slice gfx12-only: `gemm_qkv_mq4g256v2_wmma`,
//! `gemm_qkvza_mq4g256v2_wmma`, `gemm_gate_up_mq4g256v2_wmma`.
//!
//! Same discriminating construction: each 256-weight group's halves occupy
//! disjoint ranges ([-1,1] vs [96,160]) so a half-select bug reconstructs
//! ~0 instead of ~128 and fails by >100% relative error. Fused families use
//! independently salted weight blobs per constituent so pointer swaps or
//! repeated-A mistakes cannot pass. Every output carries one trailing f32
//! canary beyond the logical `n*m` that production must not clobber.
//! Shapes sweep K=256/512 (1 and 2 groups/row) and N=16/64 (one and four
//! 16-batch tiles).
//!
//! Each shape is exercised via the public arch-aware dispatchers
//! `gemm_qkv_hfq4g256_mq4v2`, `gemm_qkvza_hfq4g256_mq4v2`,
//! `gemm_gate_up_hfq4g256_mq4v2` which select `has_wmma_w32_gfx12` (RDNA4) or
//! `has_wmma_w32` (RDNA3/3.5). Tolerances mirror the residual harness:
//! rel-RMS <5% against exact-dequant fp32 ref, bug baseline at least 10x worse.
//!
//! Scalar N=1 oracles: independently salted qt44 blobs at K=256 with unequal
//! Ms per family. Reference is standalone `gemv_mq4g256v2` on the same x;
//! fused outputs must match those GPU refs elementwise (`to_bits()` equality)
//! and all canaries must survive. Catches V1 single-header decode,
//! pointer/output swaps, and tail overwrite.
//! Overall pass requires every independent output comparison and every canary.
//!
//! Run: `cargo run --release -p hipfire-runtime --example mq4v2_fused_parity`

use half::f16;
use rdna_compute::Gpu;

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;
/// Trailing f32 sentinel past each logical Y buffer; must survive the kernel.
const CANARY_F32: f32 = 12345.6789;

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Independently salted half-discriminating weights. Distinct `salt` per fused
/// constituent so pointer swaps / repeated-A cannot pass parity.
fn build_disjoint_halves_salted(m: usize, k: usize, salt: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let s = salt
                .wrapping_add((r as u32).wrapping_mul(7919))
                .wrapping_add((g as u32).wrapping_mul(104729));
            for i in 0..HALF {
                w[base + i] = prng(i, s) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, s ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut q = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let s = &w[src + off..src + off + HALF];
                let lo = s.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let sb = if hi == lo {
                    0u16
                } else {
                    f16::from_f32(step).to_bits()
                };
                let zb = f16::from_f32(lo).to_bits();
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&sb.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&zb.to_le_bytes());
                let st = f16::from_bits(sb).to_f32();
                let z = f16::from_bits(zb).to_f32();
                if st == 0.0 {
                    continue;
                }
                let inv = 1.0 / st;
                for i in 0..HALF {
                    q[off + i] = ((s[i] - z) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
                }
            }
            for i in 0..HALF {
                blob[dst + 8 + i] = (q[2 * i] & 0xF) | ((q[2 * i + 1] & 0xF) << 4);
            }
        }
    }
    blob
}

fn decode_w(blob: &[u8], m: usize, k: usize) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..gpr {
            let src = (r * gpr + g) * GROUP_BYTES;
            let h_a = u32::from_le_bytes([blob[src], blob[src + 1], blob[src + 2], blob[src + 3]]);
            let h_b =
                u32::from_le_bytes([blob[src + 4], blob[src + 5], blob[src + 6], blob[src + 7]]);
            let sc0 = f16::from_bits((h_a & 0xFFFF) as u16).to_f32();
            let zp0 = f16::from_bits((h_a >> 16) as u16).to_f32();
            let sc1 = f16::from_bits((h_b & 0xFFFF) as u16).to_f32();
            let zp1 = f16::from_bits((h_b >> 16) as u16).to_f32();
            for kt in 0..16 {
                let (sc, zp) = if kt < 8 { (sc0, zp0) } else { (sc1, zp1) };
                let k_off = kt * 16;
                for lane in 0..16 {
                    let byte = blob[src + 8 + k_off / 2 + lane / 2];
                    let nib = if lane % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let wk = g * GROUP + k_off + lane;
                    w[r * k + wk] = sc * (nib as f32) + zp;
                }
            }
        }
    }
    w
}

fn decode_w_single_header(blob: &[u8], m: usize, k: usize) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..gpr {
            let src = (r * gpr + g) * GROUP_BYTES;
            let h_a = u32::from_le_bytes([blob[src], blob[src + 1], blob[src + 2], blob[src + 3]]);
            let sc0 = f16::from_bits((h_a & 0xFFFF) as u16).to_f32();
            let zp0 = f16::from_bits((h_a >> 16) as u16).to_f32();
            for kt in 0..16 {
                let k_off = kt * 16;
                for lane in 0..16 {
                    let byte = blob[src + 8 + k_off / 2 + lane / 2];
                    let nib = if lane % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let wk = g * GROUP + k_off + lane;
                    w[r * k + wk] = sc0 * (nib as f32) + zp0;
                }
            }
        }
    }
    w
}

fn ref_gemm_ow(w: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    // Y [n,m] row-major = W[m,k] * X[n,k]^T   (overwrite semantics y=acc)
    let mut out = vec![0.0f64; n * m];
    for nn in 0..n {
        for mm in 0..m {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += w[mm * k + kk] as f64 * x[nn * k + kk] as f64;
            }
            out[nn * m + mm] = acc;
        }
    }
    out
}

fn rel_err(got: &[f32], want: &[f64]) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut wi = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1.0);
        let e = ((g as f64 - w).abs()) / denom;
        if e > worst {
            worst = e;
            wi = i;
        }
    }
    (worst, wi)
}

fn rel_rms(got: &[f32], want: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        num += (g as f64 - w).powi(2);
        den += w * w;
    }
    (num / den.max(1e-30)).sqrt()
}

fn upload_y_canary(gpu: &mut Gpu, n: usize, m: usize) -> (rdna_compute::GpuTensor, usize) {
    let y_len = n * m + 1;
    let mut y_host = vec![0.0f32; y_len];
    y_host[y_len - 1] = CANARY_F32;
    let d_y = gpu.upload_f32(&y_host, &[y_len]).unwrap();
    (d_y, y_len)
}

fn download_y_canary(
    gpu: &Gpu,
    d_y: &rdna_compute::GpuTensor,
    y_len: usize,
    n: usize,
    m: usize,
) -> (Vec<f32>, bool) {
    let got_all = gpu.download_f32(d_y).unwrap();
    assert_eq!(
        got_all.len(),
        y_len,
        "download len {} != {}",
        got_all.len(),
        y_len
    );
    let canary_ok = (got_all[y_len - 1] - CANARY_F32).abs() < 1e-6;
    (got_all[..n * m].to_vec(), canary_ok)
}

/// V1 single-header bug baseline rel-RMS for one N=1 matrix (fixture gate).
fn v1_header_bug_rel(blob: &[u8], x: &[f32], m: usize, k: usize) -> f64 {
    let ok = decode_w(blob, m, k);
    let bug = decode_w_single_header(blob, m, k);
    let want = ref_gemm_ow(&ok, x, m, k, 1);
    let bad: Vec<f32> = ref_gemm_ow(&bug, x, m, k, 1)
        .iter()
        .map(|&v| v as f32)
        .collect();
    rel_rms(&bad, &want)
}

/// Pack a salted qt44 blob and upload it.
fn pack_upload_mq4v2(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    salt: u32,
) -> (Vec<u8>, rdna_compute::GpuTensor) {
    let w = build_disjoint_halves_salted(m, k, salt);
    let blob = pack_mq4g256v2(&w, m, k);
    let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    (blob, d_a)
}

/// Standalone `gemv_mq4g256v2` N=1 reference on the same x / weight blob.
fn gemv_mq4v2_ref_n1(
    gpu: &mut Gpu,
    d_a: &rdna_compute::GpuTensor,
    d_x: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    tag: &str,
) -> Vec<f32> {
    let (d_y, _) = upload_y_canary(gpu, 1, m);
    gpu.gemv_mq4g256v2(d_a, d_x, &d_y, m, k)
        .unwrap_or_else(|e| panic!("scalar gemv ref {tag}: {e}"));
    gpu.hip.device_synchronize().unwrap();
    gpu.download_f32(&d_y).unwrap()[..m].to_vec()
}

/// Require elementwise `to_bits()` equality and an intact trailing canary.
fn record_scalar_exact(
    failures: &mut Vec<String>,
    family: &str,
    part: &str,
    k: usize,
    got: &[f32],
    refer: &[f32],
    canary_ok: bool,
) -> (bool, f64, f64, usize) {
    let want: Vec<f64> = refer.iter().map(|&v| v as f64).collect();
    let exact = got
        .iter()
        .zip(refer.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let rms = rel_rms(got, &want);
    let (worst, wi) = rel_err(got, &want);
    if !exact {
        failures.push(format!(
            "scalar_{family}/{part} K={k} N=1 bit mismatch rms {rms:.3e}"
        ));
    }
    if !canary_ok {
        failures.push(format!("scalar_{family}/{part} K={k} N=1 canary clobbered"));
    }
    (exact, rms, worst, wi)
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("mq4v2_fused_parity: no GPU ({e}) — skipping");
            return;
        }
    };
    eprintln!(
        "mq4v2_fused_parity: arch={} has_wmma_w32={} has_wmma_w32_gfx12={}",
        gpu.arch,
        gpu.arch_caps.has_wmma_w32(),
        gpu.arch_caps.has_wmma_w32_gfx12()
    );

    let mut failures = Vec::new();

    // Shapes covering K16 tiles and batch tiles; N and K drive the dispatch grid.
    let k_vals = [256usize, 512];
    let n_vals = [16usize, 64];

    // ── gate_up: gate_m=16, up_m=16 (total 32) ──
    let gate_m = 16usize;
    let up_m = 16usize;
    for &k in &k_vals {
        for &n in &n_vals {
            let w_gate = build_disjoint_halves_salted(gate_m, k, 0x6A7E_0001);
            let w_up = build_disjoint_halves_salted(up_m, k, 0x6A7E_0002);
            let blob_gate = pack_mq4g256v2(&w_gate, gate_m, k);
            let blob_up = pack_mq4g256v2(&w_up, up_m, k);
            let w_gate_ok = decode_w(&blob_gate, gate_m, k);
            let w_up_ok = decode_w(&blob_up, up_m, k);
            let w_gate_bug = decode_w_single_header(&blob_gate, gate_m, k);
            let w_up_bug = decode_w_single_header(&blob_up, up_m, k);
            let x: Vec<f32> = (0..n * k)
                .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
                .collect();
            let want_gate = ref_gemm_ow(&w_gate_ok, &x, gate_m, k, n);
            let want_up = ref_gemm_ow(&w_up_ok, &x, up_m, k, n);
            let want_gate_bug = ref_gemm_ow(&w_gate_bug, &x, gate_m, k, n);
            let want_up_bug = ref_gemm_ow(&w_up_bug, &x, up_m, k, n);
            let bug_gate = {
                let b: Vec<f32> = want_gate_bug.iter().map(|&v| v as f32).collect();
                rel_rms(&b, &want_gate)
            };
            let bug_up = {
                let b: Vec<f32> = want_up_bug.iter().map(|&v| v as f32).collect();
                rel_rms(&b, &want_up)
            };
            let bug_rel = bug_gate.max(bug_up);
            eprintln!("gate_up K={k} N={n}: bug baseline rel {bug_rel:.3e} (gate {bug_gate:.3e} up {bug_up:.3e})");
            assert!(
                bug_rel > 0.5,
                "gate_up fixture not discriminating K={k} N={n}: bug {bug_rel:.3e}"
            );

            let d_a_gate = gpu.upload_raw(&blob_gate, &[blob_gate.len()]).unwrap();
            let d_a_up = gpu.upload_raw(&blob_up, &[blob_up.len()]).unwrap();
            let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
            let (d_y_gate, y_gate_len) = upload_y_canary(&mut gpu, n, gate_m);
            let (d_y_up, y_up_len) = upload_y_canary(&mut gpu, n, up_m);
            gpu.gemm_gate_up_hfq4g256_mq4v2(
                &d_a_gate, &d_a_up, &d_x, &d_y_gate, &d_y_up, gate_m, up_m, k, n,
            )
            .unwrap_or_else(|e| panic!("gate_up launch K={k} N={n}: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let (got_gate, c_gate) = download_y_canary(&gpu, &d_y_gate, y_gate_len, n, gate_m);
            let (got_up, c_up) = download_y_canary(&gpu, &d_y_up, y_up_len, n, up_m);
            let rms_gate = rel_rms(&got_gate, &want_gate);
            let rms_up = rel_rms(&got_up, &want_up);
            let rms = rms_gate.max(rms_up);
            let canary_ok = c_gate && c_up;
            let status = if rms < 0.05 && canary_ok {
                "ok"
            } else {
                "FAIL"
            };
            let (worst_g, wi_g) = rel_err(&got_gate, &want_gate);
            let (worst_u, wi_u) = rel_err(&got_up, &want_up);
            eprintln!(
                "  gate_up K={k} N={n}: rel-rms {rms:.3e} gate {rms_gate:.3e} (worst {worst_g:.3e}@{wi_g}) up {rms_up:.3e} (worst {worst_u:.3e}@{wi_u}) canary gate={c_gate} up={c_up} {status} bug {bug_rel:.3e}"
            );
            if rms_gate >= 0.05 {
                failures.push(format!("gate_up/gate K={k} N={n} rms {rms_gate:.3e}"));
            }
            if rms_up >= 0.05 {
                failures.push(format!("gate_up/up K={k} N={n} rms {rms_up:.3e}"));
            }
            if !c_gate {
                failures.push(format!("gate_up/gate K={k} N={n} canary clobbered"));
            }
            if !c_up {
                failures.push(format!("gate_up/up K={k} N={n} canary clobbered"));
            }
            assert!(rms < bug_rel * 0.1, "half1 decoded with half0 at gate_up K={k} N={n}: rms {rms:.3e} not below bug {bug_rel:.3e}");
        }
    }

    // ── qkv: q=16, k=16, v=16 (total 48) ──
    let q_m = 16usize;
    let k_m = 16usize;
    let v_m = 16usize;
    for &k in &k_vals {
        for &n in &n_vals {
            let w_q = build_disjoint_halves_salted(q_m, k, 0x9CB0_0001);
            let w_k = build_disjoint_halves_salted(k_m, k, 0x9CB0_0002);
            let w_v = build_disjoint_halves_salted(v_m, k, 0x9CB0_0003);
            let blob_q = pack_mq4g256v2(&w_q, q_m, k);
            let blob_k = pack_mq4g256v2(&w_k, k_m, k);
            let blob_v = pack_mq4g256v2(&w_v, v_m, k);
            let w_q_ok = decode_w(&blob_q, q_m, k);
            let w_k_ok = decode_w(&blob_k, k_m, k);
            let w_v_ok = decode_w(&blob_v, v_m, k);
            let w_q_bug = decode_w_single_header(&blob_q, q_m, k);
            let w_k_bug = decode_w_single_header(&blob_k, k_m, k);
            let w_v_bug = decode_w_single_header(&blob_v, v_m, k);
            let x: Vec<f32> = (0..n * k)
                .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
                .collect();
            let want_q = ref_gemm_ow(&w_q_ok, &x, q_m, k, n);
            let want_k = ref_gemm_ow(&w_k_ok, &x, k_m, k, n);
            let want_v = ref_gemm_ow(&w_v_ok, &x, v_m, k, n);
            let bug_q = {
                let b: Vec<f32> = ref_gemm_ow(&w_q_bug, &x, q_m, k, n)
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                rel_rms(&b, &want_q)
            };
            let bug_k = {
                let b: Vec<f32> = ref_gemm_ow(&w_k_bug, &x, k_m, k, n)
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                rel_rms(&b, &want_k)
            };
            let bug_v = {
                let b: Vec<f32> = ref_gemm_ow(&w_v_bug, &x, v_m, k, n)
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                rel_rms(&b, &want_v)
            };
            let bug_rel = bug_q.max(bug_k).max(bug_v);
            eprintln!("qkv K={k} N={n}: bug baseline rel {bug_rel:.3e} (q {bug_q:.3e} k {bug_k:.3e} v {bug_v:.3e})");
            assert!(
                bug_rel > 0.5,
                "qkv fixture not discriminating K={k} N={n}: bug {bug_rel:.3e}"
            );

            let d_a_q = gpu.upload_raw(&blob_q, &[blob_q.len()]).unwrap();
            let d_a_k = gpu.upload_raw(&blob_k, &[blob_k.len()]).unwrap();
            let d_a_v = gpu.upload_raw(&blob_v, &[blob_v.len()]).unwrap();
            let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
            let (d_y_q, y_q_len) = upload_y_canary(&mut gpu, n, q_m);
            let (d_y_k, y_k_len) = upload_y_canary(&mut gpu, n, k_m);
            let (d_y_v, y_v_len) = upload_y_canary(&mut gpu, n, v_m);
            gpu.gemm_qkv_hfq4g256_mq4v2(
                &d_a_q, &d_a_k, &d_a_v, &d_x, &d_y_q, &d_y_k, &d_y_v, q_m, k_m, v_m, k, n,
            )
            .unwrap_or_else(|e| panic!("qkv launch K={k} N={n}: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let (got_q, c_q) = download_y_canary(&gpu, &d_y_q, y_q_len, n, q_m);
            let (got_k, c_k) = download_y_canary(&gpu, &d_y_k, y_k_len, n, k_m);
            let (got_v, c_v) = download_y_canary(&gpu, &d_y_v, y_v_len, n, v_m);
            let rms_q = rel_rms(&got_q, &want_q);
            let rms_k = rel_rms(&got_k, &want_k);
            let rms_v = rel_rms(&got_v, &want_v);
            let rms = rms_q.max(rms_k).max(rms_v);
            let canary_ok = c_q && c_k && c_v;
            let status = if rms < 0.05 && canary_ok {
                "ok"
            } else {
                "FAIL"
            };
            eprintln!(
                "  qkv K={k} N={n}: rel-rms {rms:.3e} q {rms_q:.3e} k {rms_k:.3e} v {rms_v:.3e} canary q={c_q} k={c_k} v={c_v} {status} bug {bug_rel:.3e}"
            );
            if rms_q >= 0.05 {
                failures.push(format!("qkv/q K={k} N={n} rms {rms_q:.3e}"));
            }
            if rms_k >= 0.05 {
                failures.push(format!("qkv/k K={k} N={n} rms {rms_k:.3e}"));
            }
            if rms_v >= 0.05 {
                failures.push(format!("qkv/v K={k} N={n} rms {rms_v:.3e}"));
            }
            if !c_q {
                failures.push(format!("qkv/q K={k} N={n} canary clobbered"));
            }
            if !c_k {
                failures.push(format!("qkv/k K={k} N={n} canary clobbered"));
            }
            if !c_v {
                failures.push(format!("qkv/v K={k} N={n} canary clobbered"));
            }
            assert!(rms < bug_rel * 0.1, "half1 decoded with half0 at qkv K={k} N={n}: rms {rms:.3e} not below bug {bug_rel:.3e}");
        }
    }

    // ── qkvza: qkv=16, z=16, beta=16, alpha=16 (total 64) ──
    let qkv_m = 16usize;
    let z_m = 16usize;
    let beta_m = 16usize;
    let alpha_m = 16usize;
    for &k in &k_vals {
        for &n in &n_vals {
            let w_qkv = build_disjoint_halves_salted(qkv_m, k, 0x2A1A_0001);
            let w_z = build_disjoint_halves_salted(z_m, k, 0x2A1A_0002);
            let w_beta = build_disjoint_halves_salted(beta_m, k, 0x2A1A_0003);
            let w_alpha = build_disjoint_halves_salted(alpha_m, k, 0x2A1A_0004);
            let blob_qkv = pack_mq4g256v2(&w_qkv, qkv_m, k);
            let blob_z = pack_mq4g256v2(&w_z, z_m, k);
            let blob_beta = pack_mq4g256v2(&w_beta, beta_m, k);
            let blob_alpha = pack_mq4g256v2(&w_alpha, alpha_m, k);
            let w_qkv_ok = decode_w(&blob_qkv, qkv_m, k);
            let w_z_ok = decode_w(&blob_z, z_m, k);
            let w_beta_ok = decode_w(&blob_beta, beta_m, k);
            let w_alpha_ok = decode_w(&blob_alpha, alpha_m, k);
            let w_qkv_bug = decode_w_single_header(&blob_qkv, qkv_m, k);
            let w_z_bug = decode_w_single_header(&blob_z, z_m, k);
            let w_beta_bug = decode_w_single_header(&blob_beta, beta_m, k);
            let w_alpha_bug = decode_w_single_header(&blob_alpha, alpha_m, k);
            let x: Vec<f32> = (0..n * k)
                .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
                .collect();
            let want_qkv = ref_gemm_ow(&w_qkv_ok, &x, qkv_m, k, n);
            let want_z = ref_gemm_ow(&w_z_ok, &x, z_m, k, n);
            let want_beta = ref_gemm_ow(&w_beta_ok, &x, beta_m, k, n);
            let want_alpha = ref_gemm_ow(&w_alpha_ok, &x, alpha_m, k, n);
            let bug_qkv = {
                let b: Vec<f32> = ref_gemm_ow(&w_qkv_bug, &x, qkv_m, k, n)
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                rel_rms(&b, &want_qkv)
            };
            let bug_z = {
                let b: Vec<f32> = ref_gemm_ow(&w_z_bug, &x, z_m, k, n)
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                rel_rms(&b, &want_z)
            };
            let bug_beta = {
                let b: Vec<f32> = ref_gemm_ow(&w_beta_bug, &x, beta_m, k, n)
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                rel_rms(&b, &want_beta)
            };
            let bug_alpha = {
                let b: Vec<f32> = ref_gemm_ow(&w_alpha_bug, &x, alpha_m, k, n)
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                rel_rms(&b, &want_alpha)
            };
            let bug_rel = bug_qkv.max(bug_z).max(bug_beta).max(bug_alpha);
            eprintln!("qkvza K={k} N={n}: bug baseline rel {bug_rel:.3e} (qkv {bug_qkv:.3e} z {bug_z:.3e} beta {bug_beta:.3e} alpha {bug_alpha:.3e})");
            assert!(
                bug_rel > 0.5,
                "qkvza fixture not discriminating K={k} N={n}: bug {bug_rel:.3e}"
            );

            let d_a_qkv = gpu.upload_raw(&blob_qkv, &[blob_qkv.len()]).unwrap();
            let d_a_z = gpu.upload_raw(&blob_z, &[blob_z.len()]).unwrap();
            let d_a_beta = gpu.upload_raw(&blob_beta, &[blob_beta.len()]).unwrap();
            let d_a_alpha = gpu.upload_raw(&blob_alpha, &[blob_alpha.len()]).unwrap();
            let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
            let (d_y_qkv, y_qkv_len) = upload_y_canary(&mut gpu, n, qkv_m);
            let (d_y_z, y_z_len) = upload_y_canary(&mut gpu, n, z_m);
            let (d_y_beta, y_beta_len) = upload_y_canary(&mut gpu, n, beta_m);
            let (d_y_alpha, y_alpha_len) = upload_y_canary(&mut gpu, n, alpha_m);
            gpu.gemm_qkvza_hfq4g256_mq4v2(
                &d_a_qkv, &d_a_z, &d_a_beta, &d_a_alpha, &d_x, &d_y_qkv, &d_y_z, &d_y_beta,
                &d_y_alpha, qkv_m, z_m, beta_m, alpha_m, k, n,
            )
            .unwrap_or_else(|e| panic!("qkvza launch K={k} N={n}: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let (got_qkv, c_qkv) = download_y_canary(&gpu, &d_y_qkv, y_qkv_len, n, qkv_m);
            let (got_z, c_z) = download_y_canary(&gpu, &d_y_z, y_z_len, n, z_m);
            let (got_beta, c_beta) = download_y_canary(&gpu, &d_y_beta, y_beta_len, n, beta_m);
            let (got_alpha, c_alpha) = download_y_canary(&gpu, &d_y_alpha, y_alpha_len, n, alpha_m);
            let rms_qkv = rel_rms(&got_qkv, &want_qkv);
            let rms_z = rel_rms(&got_z, &want_z);
            let rms_beta = rel_rms(&got_beta, &want_beta);
            let rms_alpha = rel_rms(&got_alpha, &want_alpha);
            let rms = rms_qkv.max(rms_z).max(rms_beta).max(rms_alpha);
            let canary_ok = c_qkv && c_z && c_beta && c_alpha;
            let status = if rms < 0.05 && canary_ok {
                "ok"
            } else {
                "FAIL"
            };
            eprintln!(
                "  qkvza K={k} N={n}: rel-rms {rms:.3e} qkv {rms_qkv:.3e} z {rms_z:.3e} beta {rms_beta:.3e} alpha {rms_alpha:.3e} canary qkv={c_qkv} z={c_z} beta={c_beta} alpha={c_alpha} {status} bug {bug_rel:.3e}"
            );
            if rms_qkv >= 0.05 {
                failures.push(format!("qkvza/qkv K={k} N={n} rms {rms_qkv:.3e}"));
            }
            if rms_z >= 0.05 {
                failures.push(format!("qkvza/z K={k} N={n} rms {rms_z:.3e}"));
            }
            if rms_beta >= 0.05 {
                failures.push(format!("qkvza/beta K={k} N={n} rms {rms_beta:.3e}"));
            }
            if rms_alpha >= 0.05 {
                failures.push(format!("qkvza/alpha K={k} N={n} rms {rms_alpha:.3e}"));
            }
            if !c_qkv {
                failures.push(format!("qkvza/qkv K={k} N={n} canary clobbered"));
            }
            if !c_z {
                failures.push(format!("qkvza/z K={k} N={n} canary clobbered"));
            }
            if !c_beta {
                failures.push(format!("qkvza/beta K={k} N={n} canary clobbered"));
            }
            if !c_alpha {
                failures.push(format!("qkvza/alpha K={k} N={n} canary clobbered"));
            }
            assert!(rms < bug_rel * 0.1, "half1 decoded with half0 at qkvza K={k} N={n}: rms {rms:.3e} not below bug {bug_rel:.3e}");
        }
    }

    // ── scalar N=1 oracles: QKV / QKVZA / GateUp vs standalone gemv_mq4g256v2 ──
    // Unequal Ms so each gid routing range is non-empty and a swapped y
    // pointer cannot hide behind equal-length buffers. Exact `to_bits()` only.
    let n1 = 1usize;
    let k1 = 256usize;

    // scalar fused_qkvza
    {
        let ms = [32usize, 16, 8, 24];
        let salts = [0x5CA1_0001u32, 0x5CA1_0002, 0x5CA1_0003, 0x5CA1_0004];
        let parts = ["qkv", "z", "beta", "alpha"];
        let x: Vec<f32> = (0..n1 * k1)
            .map(|i| prng(i, 0xC0FF_EE01) * 2.0 - 1.0)
            .collect();
        let mut blobs = Vec::with_capacity(4);
        let mut d_as = Vec::with_capacity(4);
        for i in 0..4 {
            let (blob, d_a) = pack_upload_mq4v2(&mut gpu, ms[i], k1, salts[i]);
            blobs.push(blob);
            d_as.push(d_a);
        }
        let bug_rels: Vec<f64> = (0..4)
            .map(|i| v1_header_bug_rel(&blobs[i], &x, ms[i], k1))
            .collect();
        let bug_rel = bug_rels.iter().cloned().fold(0.0f64, f64::max);
        eprintln!(
            "scalar_fused_qkvza K={k1} N={n1} Ms=({},{},{},{}): bug baseline rel {bug_rel:.3e} (qkv {:.3e} z {:.3e} beta {:.3e} alpha {:.3e})",
            ms[0], ms[1], ms[2], ms[3], bug_rels[0], bug_rels[1], bug_rels[2], bug_rels[3]
        );
        assert!(
            bug_rel > 0.5,
            "scalar_fused_qkvza fixture not discriminating K={k1} N={n1}: bug {bug_rel:.3e}"
        );
        let d_x = gpu.upload_f32(&x, &[n1 * k1]).unwrap();
        let refs: Vec<Vec<f32>> = (0..4)
            .map(|i| gemv_mq4v2_ref_n1(&mut gpu, &d_as[i], &d_x, ms[i], k1, parts[i]))
            .collect();
        let mut d_ys = Vec::with_capacity(4);
        let mut y_lens = Vec::with_capacity(4);
        for &m in &ms {
            let (d_y, y_len) = upload_y_canary(&mut gpu, n1, m);
            d_ys.push(d_y);
            y_lens.push(y_len);
        }
        gpu.fused_qkvza_hfq4g256_mq4v2(
            &d_as[0], &d_as[1], &d_as[2], &d_as[3], &d_x, &d_ys[0], &d_ys[1], &d_ys[2], &d_ys[3],
            ms[0], ms[1], ms[2], ms[3], k1,
        )
        .unwrap_or_else(|e| panic!("scalar fused_qkvza_mq4v2 launch: {e}"));
        gpu.hip.device_synchronize().unwrap();
        let mut rms_max = 0.0f64;
        let mut detail = String::new();
        let mut canary_ok = true;
        let mut all_exact = true;
        for i in 0..4 {
            let (got, c_ok) = download_y_canary(&gpu, &d_ys[i], y_lens[i], n1, ms[i]);
            let (exact, rms, worst, wi) = record_scalar_exact(
                &mut failures,
                "fused_qkvza",
                parts[i],
                k1,
                &got,
                &refs[i],
                c_ok,
            );
            rms_max = rms_max.max(rms);
            all_exact &= exact;
            canary_ok &= c_ok;
            detail.push_str(&format!(
                " {} {rms:.3e} exact={exact} (worst {worst:.3e}@{wi}) canary={c_ok}",
                parts[i]
            ));
        }
        let status = if all_exact && canary_ok { "ok" } else { "FAIL" };
        eprintln!(
            "  scalar_fused_qkvza K={k1} N={n1}: rel-rms {rms_max:.3e}{detail} {status} bug {bug_rel:.3e}"
        );
        assert!(
            rms_max < bug_rel * 0.1,
            "half1 decoded with half0 at scalar_fused_qkvza K={k1} N={n1}: rms {rms_max:.3e} not below bug {bug_rel:.3e}"
        );
    }

    // scalar fused_qkv
    {
        let ms = [40usize, 24, 12];
        let salts = [0x7E4D_0001u32, 0x7E4D_0002, 0x7E4D_0003];
        let parts = ["q", "k", "v"];
        let x: Vec<f32> = (0..n1 * k1)
            .map(|i| prng(i, 0xC0FF_EE02) * 2.0 - 1.0)
            .collect();
        let mut blobs = Vec::with_capacity(3);
        let mut d_as = Vec::with_capacity(3);
        for i in 0..3 {
            let (blob, d_a) = pack_upload_mq4v2(&mut gpu, ms[i], k1, salts[i]);
            blobs.push(blob);
            d_as.push(d_a);
        }
        let bug_rels: Vec<f64> = (0..3)
            .map(|i| v1_header_bug_rel(&blobs[i], &x, ms[i], k1))
            .collect();
        let bug_rel = bug_rels.iter().cloned().fold(0.0f64, f64::max);
        eprintln!(
            "scalar_fused_qkv K={k1} N={n1} Ms=({},{},{}): bug baseline rel {bug_rel:.3e} (q {:.3e} k {:.3e} v {:.3e})",
            ms[0], ms[1], ms[2], bug_rels[0], bug_rels[1], bug_rels[2]
        );
        assert!(
            bug_rel > 0.5,
            "scalar_fused_qkv fixture not discriminating K={k1} N={n1}: bug {bug_rel:.3e}"
        );
        let d_x = gpu.upload_f32(&x, &[n1 * k1]).unwrap();
        let refs: Vec<Vec<f32>> = (0..3)
            .map(|i| gemv_mq4v2_ref_n1(&mut gpu, &d_as[i], &d_x, ms[i], k1, parts[i]))
            .collect();
        let mut d_ys = Vec::with_capacity(3);
        let mut y_lens = Vec::with_capacity(3);
        for &m in &ms {
            let (d_y, y_len) = upload_y_canary(&mut gpu, n1, m);
            d_ys.push(d_y);
            y_lens.push(y_len);
        }
        gpu.fused_qkv_hfq4g256_mq4v2(
            &d_as[0], &d_as[1], &d_as[2], &d_x, &d_ys[0], &d_ys[1], &d_ys[2], ms[0], ms[1], ms[2],
            k1,
        )
        .unwrap_or_else(|e| panic!("scalar fused_qkv_mq4v2 launch: {e}"));
        gpu.hip.device_synchronize().unwrap();
        let mut rms_max = 0.0f64;
        let mut detail = String::new();
        let mut canary_ok = true;
        let mut all_exact = true;
        for i in 0..3 {
            let (got, c_ok) = download_y_canary(&gpu, &d_ys[i], y_lens[i], n1, ms[i]);
            let (exact, rms, worst, wi) = record_scalar_exact(
                &mut failures,
                "fused_qkv",
                parts[i],
                k1,
                &got,
                &refs[i],
                c_ok,
            );
            rms_max = rms_max.max(rms);
            all_exact &= exact;
            canary_ok &= c_ok;
            detail.push_str(&format!(
                " {} {rms:.3e} exact={exact} (worst {worst:.3e}@{wi}) canary={c_ok}",
                parts[i]
            ));
        }
        let status = if all_exact && canary_ok { "ok" } else { "FAIL" };
        eprintln!(
            "  scalar_fused_qkv K={k1} N={n1}: rel-rms {rms_max:.3e}{detail} {status} bug {bug_rel:.3e}"
        );
        assert!(
            rms_max < bug_rel * 0.1,
            "half1 decoded with half0 at scalar_fused_qkv K={k1} N={n1}: rms {rms_max:.3e} not below bug {bug_rel:.3e}"
        );
    }

    // scalar fused_gate_up
    {
        let ms = [48usize, 20];
        let salts = [0x81A0_0001u32, 0x81A0_0002];
        let parts = ["gate", "up"];
        let x: Vec<f32> = (0..n1 * k1)
            .map(|i| prng(i, 0xC0FF_EE03) * 2.0 - 1.0)
            .collect();
        let mut blobs = Vec::with_capacity(2);
        let mut d_as = Vec::with_capacity(2);
        for i in 0..2 {
            let (blob, d_a) = pack_upload_mq4v2(&mut gpu, ms[i], k1, salts[i]);
            blobs.push(blob);
            d_as.push(d_a);
        }
        let bug_rels: Vec<f64> = (0..2)
            .map(|i| v1_header_bug_rel(&blobs[i], &x, ms[i], k1))
            .collect();
        let bug_rel = bug_rels.iter().cloned().fold(0.0f64, f64::max);
        eprintln!(
            "scalar_fused_gate_up K={k1} N={n1} Ms=({},{}): bug baseline rel {bug_rel:.3e} (gate {:.3e} up {:.3e})",
            ms[0], ms[1], bug_rels[0], bug_rels[1]
        );
        assert!(
            bug_rel > 0.5,
            "scalar_fused_gate_up fixture not discriminating K={k1} N={n1}: bug {bug_rel:.3e}"
        );
        let d_x = gpu.upload_f32(&x, &[n1 * k1]).unwrap();
        let refs: Vec<Vec<f32>> = (0..2)
            .map(|i| gemv_mq4v2_ref_n1(&mut gpu, &d_as[i], &d_x, ms[i], k1, parts[i]))
            .collect();
        let mut d_ys = Vec::with_capacity(2);
        let mut y_lens = Vec::with_capacity(2);
        for &m in &ms {
            let (d_y, y_len) = upload_y_canary(&mut gpu, n1, m);
            d_ys.push(d_y);
            y_lens.push(y_len);
        }
        gpu.fused_gate_up_hfq4g256_mq4v2(
            &d_as[0], &d_as[1], &d_x, &d_ys[0], &d_ys[1], ms[0], ms[1], k1,
        )
        .unwrap_or_else(|e| panic!("scalar fused_gate_up_mq4v2 launch: {e}"));
        gpu.hip.device_synchronize().unwrap();
        let mut rms_max = 0.0f64;
        let mut detail = String::new();
        let mut canary_ok = true;
        let mut all_exact = true;
        for i in 0..2 {
            let (got, c_ok) = download_y_canary(&gpu, &d_ys[i], y_lens[i], n1, ms[i]);
            let (exact, rms, worst, wi) = record_scalar_exact(
                &mut failures,
                "fused_gate_up",
                parts[i],
                k1,
                &got,
                &refs[i],
                c_ok,
            );
            rms_max = rms_max.max(rms);
            all_exact &= exact;
            canary_ok &= c_ok;
            detail.push_str(&format!(
                " {} {rms:.3e} exact={exact} (worst {worst:.3e}@{wi}) canary={c_ok}",
                parts[i]
            ));
        }
        let status = if all_exact && canary_ok { "ok" } else { "FAIL" };
        eprintln!(
            "  scalar_fused_gate_up K={k1} N={n1}: rel-rms {rms_max:.3e}{detail} {status} bug {bug_rel:.3e}"
        );
        assert!(
            rms_max < bug_rel * 0.1,
            "half1 decoded with half0 at scalar_fused_gate_up K={k1} N={n1}: rms {rms_max:.3e} not below bug {bug_rel:.3e}"
        );
    }

    // ── Ornith qt44 shared-down fuse: M=2048 K=512 residual_sigmoid_scaled ──
    // Reference: sigmoid_f32 → plain gemv_mq4g256v2 into temp → scaled_add.
    // Candidate: gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2 (four-row A/B).
    // R1 also launched for three-way bit equality. Exact gfx1100|gfx1201 only;
    // skip elsewhere. 16 tail canaries.
    {
        let arch_ok = gpu.arch_caps.is_gfx1100() || gpu.arch_caps.is_gfx1201();
        if !arch_ok {
            eprintln!(
                "shared_down_fused residual_sigmoid_scaled_k512_r2: skip (arch={} not gfx1100|gfx1201)",
                gpu.arch
            );
        } else {
            const M: usize = 2048;
            const K: usize = 512;
            const N_CANARY: usize = 16;
            let w = build_disjoint_halves_salted(M, K, 0x51C0_D0F1);
            let blob = pack_mq4g256v2(&w, M, K);
            let bug_rel = {
                let x_probe: Vec<f32> = (0..K).map(|i| prng(i, 0xC0FF_EE51) * 2.0 - 1.0).collect();
                v1_header_bug_rel(&blob, &x_probe, M, K)
            };
            eprintln!(
                "shared_down_fused residual_sigmoid_scaled_k512_r2 M={M} K={K}: V1-header bug baseline rel {bug_rel:.3e}"
            );
            assert!(
                bug_rel > 0.5,
                "shared_down_fused fixture not discriminating: bug {bug_rel:.3e}"
            );

            let x: Vec<f32> = (0..K).map(|i| prng(i, 0xC0FF_EE52) * 2.0 - 1.0).collect();
            // Nontrivial pre-sigmoid scalar (not 0 → gate≠0.5).
            let c_host = [1.25f32];
            // Nonzero residual seed.
            let mut y_seed = vec![0.0f32; M + N_CANARY];
            for i in 0..M {
                y_seed[i] = prng(i, 0x5E51_D00D) * 0.5 + 0.25;
            }
            for i in 0..N_CANARY {
                y_seed[M + i] = CANARY_F32 + i as f32 * 0.001;
            }

            let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
            let d_x = gpu.upload_f32(&x, &[K]).unwrap();
            let d_c = gpu.upload_f32(&c_host, &[1]).unwrap();

            // Reference route.
            let d_y_ref = gpu.upload_f32(&y_seed[..M], &[M]).unwrap();
            let d_tmp = gpu.upload_f32(&vec![0.0f32; M], &[M]).unwrap();
            let d_c_ref = gpu.upload_f32(&c_host, &[1]).unwrap();
            gpu.sigmoid_f32(&d_c_ref)
                .unwrap_or_else(|e| panic!("ref sigmoid_f32: {e}"));
            gpu.gemv_mq4g256v2(&d_a, &d_x, &d_tmp, M, K)
                .unwrap_or_else(|e| panic!("ref gemv_mq4g256v2: {e}"));
            gpu.scaled_add_inplace_gpu_scalar_f32(&d_y_ref, &d_tmp, &d_c_ref)
                .unwrap_or_else(|e| panic!("ref scaled_add: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let ref_all = gpu.download_f32(&d_y_ref).unwrap();

            // Explicit R1 run (default product path).
            let d_y_r1 = gpu.upload_f32(&y_seed, &[M + N_CANARY]).unwrap();
            gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512(&d_a, &d_x, &d_y_r1, &d_c, M, K)
                .unwrap_or_else(|e| panic!("R1 fused launch: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let r1_all = gpu.download_f32(&d_y_r1).unwrap();

            // Candidate R2 fused route (does not mutate c).
            let d_y_cand = gpu.upload_f32(&y_seed, &[M + N_CANARY]).unwrap();
            gpu.gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2(&d_a, &d_x, &d_y_cand, &d_c, M, K)
                .unwrap_or_else(|e| panic!("candidate R2 fused launch: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let cand_all = gpu.download_f32(&d_y_cand).unwrap();

            let ref_y = &ref_all[..M];
            let r1_y = &r1_all[..M];
            let cand_y = &cand_all[..M];
            let mut first_mismatch: Option<(usize, f32, f32, f32)> = None;
            let exact = ref_y
                .iter()
                .zip(r1_y.iter())
                .zip(cand_y.iter())
                .enumerate()
                .all(|(i, ((&a, &b), &c))| {
                    let ok = a.to_bits() == b.to_bits() && b.to_bits() == c.to_bits();
                    if !ok && first_mismatch.is_none() {
                        first_mismatch = Some((i, a, b, c));
                    }
                    ok
                });
            // Only the fused candidates receive padded output. The production
            // reference scaled-add sizes its grid from y, so padding that y
            // would deliberately make it read beyond the M-element temp.
            let canary_ok = (0..N_CANARY).all(|i| {
                let want = CANARY_F32 + i as f32 * 0.001;
                cand_all[M + i].to_bits() == want.to_bits()
                    && r1_all[M + i].to_bits() == want.to_bits()
            });
            // Scalar must be bit-identical on the fused path (not mutated).
            let c_after = gpu.download_f32(&d_c).unwrap();
            let c_intact = c_after[0].to_bits() == c_host[0].to_bits();

            let want64: Vec<f64> = ref_y.iter().map(|&v| v as f64).collect();
            let rms = rel_rms(cand_y, &want64);
            let (worst, wi) = rel_err(cand_y, &want64);
            let status = if exact && canary_ok && c_intact {
                "ok"
            } else {
                "FAIL"
            };
            eprintln!(
                "  shared_down_fused residual_sigmoid_scaled_k512_r2: exact={exact} rms {rms:.3e} (worst {worst:.3e}@{wi}) canary={canary_ok} c_intact={c_intact} {status} bug {bug_rel:.3e}"
            );
            if let Some((row, a, b, c)) = first_mismatch {
                eprintln!(
                    "  first mismatch row={row} row%4={} ref={a:?} r1={b:?} r2={c:?}",
                    row % 4
                );
            }
            if !exact {
                failures.push(format!(
                    "shared_down_fused residual_sigmoid_scaled_k512_r2 bit mismatch rms {rms:.3e}"
                ));
            }
            if !canary_ok {
                failures.push(
                    "shared_down_fused residual_sigmoid_scaled_k512_r2 canary clobbered".into(),
                );
            }
            if !c_intact {
                failures.push(
                    "shared_down_fused residual_sigmoid_scaled_k512_r2 mutated pre-sigmoid scalar"
                        .into(),
                );
            }
            assert!(
                rms < bug_rel * 0.1,
                "half1 decoded with half0 at shared_down_fused: rms {rms:.3e} not below bug {bug_rel:.3e}"
            );
        }
    }

    if failures.is_empty() {
        eprintln!("\nmq4v2_fused_parity: PASS — all fused families half-select/pointer-swap/canary correct at K=256/512, N=16/64; scalar fused_qkv / fused_qkvza / fused_gate_up N=1 match standalone gemv_mq4g256v2; residual_sigmoid_scaled_k512_r2 bit-exact vs R1 and reference on gfx1100|gfx1201");
    } else {
        eprintln!("\nmq4v2_fused_parity: FAIL — {:?}", failures);
        std::process::exit(1);
    }
}
