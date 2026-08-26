// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ{2,3,5,6}V2 qkv BT8 parity on exact gfx1201 — generic weight-reuse gate.
//!
//! Synthetic V2 writer: group_bytes = 8 + 32*bits, deliberately disjoint dual
//! FP16 header halves, 256 integer codes packed contiguously LSB-first.
//! Routing-sensitive shapes q=40/k=32/v=48, K=512, deterministic nondegenerate
//! F32 X. On exact gfx1201, for bits∈{2,3,5,6} and N∈{128,256}:
//!   1) run exact base `gemm_qkv_mq{bits}g256v2_wmma_gfx12` as the oracle,
//!      forcing `capture_mode` so production batch-tile policy is bypassed,
//!   2) run direct `gemm_qkv_mqv2_wmma_gfx1201_bt8(bits, …)`,
//!  and compare q/k/v separately for raw `f32::to_bits` equality on every
//!  element plus finite/nondegenerate (24 projection checks total).
//! On any other arch the harness SKIPs cleanly (exit 0, no GPU work).

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu};

const GROUP: usize = 256;
const HALF: usize = 128;
const BITS: [u8; 4] = [2, 3, 5, 6];
const NS: [usize; 2] = [128, 256];

fn group_bytes(bits: u8) -> usize {
    8 + 32 * bits as usize
}

fn max_q(bits: u8) -> u8 {
    match bits {
        2 => 3,
        3 => 7,
        5 => 31,
        6 => 63,
        _ => unreachable!("bits must be 2/3/5/6"),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
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

/// Synthetic V2 packer: dual FP16 headers + contiguous LSB-first codes.
fn pack_mqv2(bits: u8, w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert!(matches!(bits, 2 | 3 | 5 | 6));
    assert_eq!(k % GROUP, 0, "k must be multiple of 256");
    assert_eq!(w.len(), m * k);
    let gb = group_bytes(bits);
    let mq = max_q(bits) as f32;
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * gb];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * gb;
            let mut s_bits = [0u16; 2];
            let mut z_bits = [0u16; 2];
            let mut s_rt = [0.0f32; 2];
            let mut z_rt = [0.0f32; 2];
            let mut degenerate = [false; 2];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / mq } else { 0.0 };
                let sb = if hi == lo { 0u16 } else { half_from_f32(step) };
                let zb = half_from_f32(lo);
                s_bits[h] = sb;
                z_bits[h] = zb;
                s_rt[h] = f16_to_f32(sb);
                z_rt[h] = f16_to_f32(zb);
                degenerate[h] = hi == lo || step == 0.0 || s_rt[h] == 0.0;
            }
            // Dual-half header: s0 z0 s1 z1 little-endian (deliberately disjoint).
            blob[dst..dst + 2].copy_from_slice(&s_bits[0].to_le_bytes());
            blob[dst + 2..dst + 4].copy_from_slice(&z_bits[0].to_le_bytes());
            blob[dst + 4..dst + 6].copy_from_slice(&s_bits[1].to_le_bytes());
            blob[dst + 6..dst + 8].copy_from_slice(&z_bits[1].to_le_bytes());

            let mut q = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                if degenerate[h] {
                    continue;
                }
                let inv = 1.0 / s_rt[h];
                let z = z_rt[h];
                for i in 0..HALF {
                    let v = w[src + off + i];
                    let qq = ((v - z) * inv + 0.5).floor().clamp(0.0, mq) as u8;
                    q[off + i] = qq;
                }
            }

            // Contiguous LSB-first pack of all 256 integer codes across payload bytes.
            let payload = &mut blob[dst + 8..dst + gb];
            match bits {
                2 => {
                    for i in 0..64 {
                        let mut b = 0u8;
                        for j in 0..4 {
                            b |= (q[4 * i + j] & 3) << (j * 2);
                        }
                        payload[i] = b;
                    }
                }
                3 => {
                    for chunk in 0..32 {
                        let ci = chunk * 8;
                        let qq = [
                            q[ci] & 7,
                            q[ci + 1] & 7,
                            q[ci + 2] & 7,
                            q[ci + 3] & 7,
                            q[ci + 4] & 7,
                            q[ci + 5] & 7,
                            q[ci + 6] & 7,
                            q[ci + 7] & 7,
                        ];
                        let b0 = (qq[0] & 7) | ((qq[1] & 7) << 3) | ((qq[2] & 3) << 6);
                        let b1 = ((qq[2] >> 2) & 1)
                            | ((qq[3] & 7) << 1)
                            | ((qq[4] & 7) << 4)
                            | ((qq[5] & 1) << 7);
                        let b2 = ((qq[5] >> 1) & 3) | ((qq[6] & 7) << 2) | ((qq[7] & 7) << 5);
                        let bo = chunk * 3;
                        payload[bo] = b0;
                        payload[bo + 1] = b1;
                        payload[bo + 2] = b2;
                    }
                }
                5 => {
                    for i in (0..256).step_by(8) {
                        let bo = (i / 8) * 5;
                        let q0 = q[i] & 31;
                        let q1 = q[i + 1] & 31;
                        let q2 = q[i + 2] & 31;
                        let q3 = q[i + 3] & 31;
                        let q4 = q[i + 4] & 31;
                        let q5 = q[i + 5] & 31;
                        let q6 = q[i + 6] & 31;
                        let q7 = q[i + 7] & 31;
                        payload[bo] = q0 | (q1 << 5);
                        payload[bo + 1] = (q1 >> 3) | (q2 << 2) | (q3 << 7);
                        payload[bo + 2] = (q3 >> 1) | (q4 << 4);
                        payload[bo + 3] = (q4 >> 4) | (q5 << 1) | (q6 << 6);
                        payload[bo + 4] = (q6 >> 2) | (q7 << 3);
                    }
                }
                6 => {
                    for i in (0..256).step_by(4) {
                        let bo = (i / 4) * 3;
                        let q0 = q[i] & 63;
                        let q1 = q[i + 1] & 63;
                        let q2 = q[i + 2] & 63;
                        let q3 = q[i + 3] & 63;
                        payload[bo] = q0 | (q1 << 6);
                        payload[bo + 1] = (q1 >> 2) | (q2 << 4);
                        payload[bo + 2] = (q2 >> 4) | (q3 << 2);
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    blob
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn check_proj(label: &str, bits: u8, n: usize, got: &[f32], want: &[f32]) -> bool {
    let finite = is_finite(got) && is_finite(want);
    let var_got = variance(got);
    let var_want = variance(want);
    let nondeg = var_got > 1e-12 && var_want > 1e-12;
    let bit_eq = got.len() == want.len()
        && got
            .iter()
            .zip(want.iter())
            .all(|(g, w)| g.to_bits() == w.to_bits());
    let m = max_abs_diff(got, want);
    let ok = bit_eq && finite && nondeg;
    let status = if ok { "PASS" } else { "FAIL" };
    eprintln!(
        "  [mq{bits}v2 {label} N={n}] bitEq={bit_eq} finite={finite} nondeg={nondeg} (var {var_got:.3e}/{var_want:.3e}) maxAbs={m:.3e} [{status}]"
    );
    if !ok {
        if !bit_eq {
            let mism = got
                .iter()
                .zip(want.iter())
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            eprintln!(
                "    GATE FAIL: raw-bit equality violated — {mism} element(s) differ by to_bits()"
            );
        }
        if !finite {
            eprintln!("    non-finite values detected");
        }
        if !nondeg {
            eprintln!("    degenerate (near-zero variance)");
        }
    }
    ok
}

fn build_disjoint_halves_seeded(m: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32 ^ seed;
            for i in 0..HALF {
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

fn fill_sentinel(gpu: &Gpu, buf: &rdna_compute::GpuTensor, sen: f32, len: usize) {
    let host = vec![sen; len];
    gpu.hip
        .memcpy_htod(&buf.buf, unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
        })
        .expect("htod sentinel");
}

fn launch_base(
    gpu: &mut Gpu,
    bits: u8,
    d_q: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_x: &rdna_compute::GpuTensor,
    d_y_q: &rdna_compute::GpuTensor,
    d_y_k: &rdna_compute::GpuTensor,
    d_y_v: &rdna_compute::GpuTensor,
    q_m: usize,
    k_m: usize,
    v_m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    let r = match bits {
        2 => gpu.gemm_qkv_mq2g256v2_wmma_gfx12(
            d_q, d_k, d_v, d_x, d_y_q, d_y_k, d_y_v, q_m, k_m, v_m, k, n,
        ),
        3 => gpu.gemm_qkv_mq3g256v2_wmma_gfx12(
            d_q, d_k, d_v, d_x, d_y_q, d_y_k, d_y_v, q_m, k_m, v_m, k, n,
        ),
        5 => gpu.gemm_qkv_mq5g256v2_wmma_gfx12(
            d_q, d_k, d_v, d_x, d_y_q, d_y_k, d_y_v, q_m, k_m, v_m, k, n,
        ),
        6 => gpu.gemm_qkv_mq6g256v2_wmma_gfx12(
            d_q, d_k, d_v, d_x, d_y_q, d_y_k, d_y_v, q_m, k_m, v_m, k, n,
        ),
        _ => return Err(format!("unsupported bits {bits}")),
    };
    r.map_err(|e| format!("{e:?}"))
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
    let is_gfx1201 = gpu.arch_caps.is_gfx1201() && arch == "gfx1201";
    if !is_gfx1201 {
        eprintln!("SKIP: arch {arch} is not exact gfx1201 — harness requires gfx1201 only");
        return;
    }
    eprintln!("arch {arch} confirmed exact gfx1201 — running MQV2 qkv BT8 parity");

    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some — BT harness requires no capture");
        return;
    }

    // Tail projection sizes crossing stacked routing regions (not 16-aligned).
    let q_m: usize = 40;
    let k_m: usize = 32;
    let v_m: usize = 48;
    let k: usize = 512;
    assert_eq!(k % 256, 0);
    let total_m = q_m + k_m + v_m;
    eprintln!("q_m={q_m} k_m={k_m} v_m={v_m} total={total_m} K={k}");
    eprintln!(
        "arms: bits={BITS:?} N={NS:?} → {} projection checks",
        BITS.len() * NS.len() * 3
    );

    let sentinel_q = f32::from_bits(0x7FC0_0001);
    let sentinel_k = f32::from_bits(0x7FC0_0002);
    let sentinel_v = f32::from_bits(0x7FC0_0003);
    let mut all_ok = true;
    let mut checks = 0usize;

    for &bits in &BITS {
        let gb = group_bytes(bits);
        eprintln!("\n######## bits={bits} group_bytes={gb} ########");

        let w_q = build_disjoint_halves_seeded(q_m, k, 0x1111_2222 ^ (bits as u32) << 16);
        let w_k = build_disjoint_halves_seeded(k_m, k, 0x3333_4444 ^ (bits as u32) << 16);
        let w_v = build_disjoint_halves_seeded(v_m, k, 0x5555_6666 ^ (bits as u32) << 16);
        let blob_q = pack_mqv2(bits, &w_q, q_m, k);
        let blob_k = pack_mqv2(bits, &w_k, k_m, k);
        let blob_v = pack_mqv2(bits, &w_v, v_m, k);
        assert_eq!(blob_q.len(), q_m * (k / GROUP) * gb);
        assert_eq!(blob_k.len(), k_m * (k / GROUP) * gb);
        assert_eq!(blob_v.len(), v_m * (k / GROUP) * gb);
        // Header halves must diverge on at least one group (fixture discrimination).
        {
            let s0 = u16::from_le_bytes([blob_q[0], blob_q[1]]);
            let z0 = u16::from_le_bytes([blob_q[2], blob_q[3]]);
            let s1 = u16::from_le_bytes([blob_q[4], blob_q[5]]);
            let z1 = u16::from_le_bytes([blob_q[6], blob_q[7]]);
            assert!(
                s0 != s1 || z0 != z1,
                "mq{bits}v2 fixture headers not disjoint"
            );
        }
        eprintln!(
            "packed q {} B, k {} B, v {} B",
            blob_q.len(),
            blob_k.len(),
            blob_v.len()
        );

        let d_q = gpu.upload_raw(&blob_q, &[blob_q.len()]).expect("upload q");
        let d_k = gpu.upload_raw(&blob_k, &[blob_k.len()]).expect("upload k");
        let d_v = gpu.upload_raw(&blob_v, &[blob_v.len()]).expect("upload v");

        for &n in &NS {
            eprintln!("\n=== mq{bits}v2 N={n} ===");
            let x_host: Vec<f32> = (0..n * k)
                .map(|i| prng(i, 0xC0FF_EE00 ^ (bits as u32)) * 2.0 - 1.0)
                .collect();

            let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
            let d_y_q_ref = gpu.alloc_tensor(&[n * q_m], DType::F32).expect("y_q ref");
            let d_y_k_ref = gpu.alloc_tensor(&[n * k_m], DType::F32).expect("y_k ref");
            let d_y_v_ref = gpu.alloc_tensor(&[n * v_m], DType::F32).expect("y_v ref");

            gpu.hip
                .memcpy_htod(&d_x.buf, unsafe {
                    std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4)
                })
                .expect("htod x");
            gpu.hip.device_synchronize().expect("sync x");

            fill_sentinel(&gpu, &d_y_q_ref, sentinel_q, n * q_m);
            fill_sentinel(&gpu, &d_y_k_ref, sentinel_k, n * k_m);
            fill_sentinel(&gpu, &d_y_v_ref, sentinel_v, n * v_m);
            gpu.hip.device_synchronize().expect("sync sentinel ref");

            // Force the historical gfx12 base path: production batch-tile
            // policy deliberately bypasses graph capture.
            let saved_capture = gpu.graphs.capture_mode;
            gpu.graphs.capture_mode = true;
            let t_ref = std::time::Instant::now();
            let base_res = launch_base(
                &mut gpu, bits, &d_q, &d_k, &d_v, &d_x, &d_y_q_ref, &d_y_k_ref, &d_y_v_ref, q_m,
                k_m, v_m, k, n,
            );
            gpu.graphs.capture_mode = saved_capture;
            if let Err(e) = base_res {
                eprintln!("  base gemm_qkv_mq{bits}g256v2_wmma_gfx12 FAILED N={n}: {e}");
                all_ok = false;
                continue;
            }
            gpu.hip.device_synchronize().expect("sync ref");
            let ref_us = t_ref.elapsed().as_secs_f64() * 1e6;
            eprintln!("  gfx12 base mq{bits}v2 N={n} done in {ref_us:.1} us");

            let y_q_ref = gpu.download_f32(&d_y_q_ref).expect("download q ref");
            let y_k_ref = gpu.download_f32(&d_y_k_ref).expect("download k ref");
            let y_v_ref = gpu.download_f32(&d_y_v_ref).expect("download v ref");
            assert_eq!(y_q_ref.len(), n * q_m);
            assert_eq!(y_k_ref.len(), n * k_m);
            assert_eq!(y_v_ref.len(), n * v_m);

            for (label, data, sen) in [
                ("q_ref", &y_q_ref, sentinel_q),
                ("k_ref", &y_k_ref, sentinel_k),
                ("v_ref", &y_v_ref, sentinel_v),
            ] {
                let cnt = data.iter().filter(|x| x.to_bits() == sen.to_bits()).count();
                assert_eq!(cnt, 0, "ref {label} still contains sentinel NaN N={n}");
            }
            assert!(is_finite(&y_q_ref) && is_finite(&y_k_ref) && is_finite(&y_v_ref));
            assert!(
                variance(&y_q_ref) > 1e-12
                    && variance(&y_k_ref) > 1e-12
                    && variance(&y_v_ref) > 1e-12
            );

            let d_y_q_bt = gpu.alloc_tensor(&[n * q_m], DType::F32).expect("y_q bt");
            let d_y_k_bt = gpu.alloc_tensor(&[n * k_m], DType::F32).expect("y_k bt");
            let d_y_v_bt = gpu.alloc_tensor(&[n * v_m], DType::F32).expect("y_v bt");
            fill_sentinel(&gpu, &d_y_q_bt, sentinel_q, n * q_m);
            fill_sentinel(&gpu, &d_y_k_bt, sentinel_k, n * k_m);
            fill_sentinel(&gpu, &d_y_v_bt, sentinel_v, n * v_m);
            gpu.hip.device_synchronize().expect("sync sentinel bt");

            let t_bt = std::time::Instant::now();
            let res = gpu.gemm_qkv_mqv2_wmma_gfx1201_bt8(
                bits, &d_q, &d_k, &d_v, &d_x, &d_y_q_bt, &d_y_k_bt, &d_y_v_bt, q_m, k_m, v_m, k, n,
            );
            if let Err(e) = res {
                eprintln!("  candidate gemm_qkv_mqv2_wmma_gfx1201_bt8 FAILED N={n}: {e:?}");
                all_ok = false;
                continue;
            }
            gpu.hip.device_synchronize().expect("sync bt");
            let bt_us = t_bt.elapsed().as_secs_f64() * 1e6;
            eprintln!("  gfx1201 bt8 mq{bits}v2 N={n} done in {bt_us:.1} us");

            let y_q_bt = gpu.download_f32(&d_y_q_bt).expect("download q bt");
            let y_k_bt = gpu.download_f32(&d_y_k_bt).expect("download k bt");
            let y_v_bt = gpu.download_f32(&d_y_v_bt).expect("download v bt");

            for (label, data, sen) in [
                ("q_bt", &y_q_bt, sentinel_q),
                ("k_bt", &y_k_bt, sentinel_k),
                ("v_bt", &y_v_bt, sentinel_v),
            ] {
                let cnt = data.iter().filter(|x| x.to_bits() == sen.to_bits()).count();
                if cnt != 0 {
                    eprintln!("  bt8 {label} still contains {cnt} sentinel NaN(s) N={n}");
                    all_ok = false;
                }
            }

            let ok_q = check_proj("q", bits, n, &y_q_bt, &y_q_ref);
            let ok_k = check_proj("k", bits, n, &y_k_bt, &y_k_ref);
            let ok_v = check_proj("v", bits, n, &y_v_bt, &y_v_ref);
            checks += 3;
            if !ok_q || !ok_k || !ok_v {
                eprintln!("  mq{bits}v2 BT8 N={n} FAILED");
                all_ok = false;
            } else {
                eprintln!("  mq{bits}v2 BT8 N={n} OK (q/k/v raw-bit)");
            }
        }
    }

    eprintln!("\nprojection checks reported: {checks} (expect 24)");
    if all_ok && checks == 24 {
        eprintln!(
            "\nPASS: all bits={{2,3,5,6}} × N={{128,256}} × {{q,k,v}} raw-bit equal vs gfx12 base; finite/nondegenerate"
        );
    } else {
        eprintln!(
            "\nFAIL: one or more mqV2 qkv BT8 parity checks violated raw-bit equality or finite/nondegenerate"
        );
        std::process::exit(1);
    }
}
