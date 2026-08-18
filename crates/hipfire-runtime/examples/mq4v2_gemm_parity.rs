//! v1-vs-v2 cross-check for the **WMMA GEMM** path (qt=13 vs qt=44).
//!
//! `mq4v2_parity` verifies the decode GEMV against a host oracle. It does NOT
//! cover the WMMA prefill GEMMs, which are what `--scoring-mode prefill` actually
//! runs, and those are where MQ4 v2 currently fails: with the v2 GEMMs finally
//! executing (8 v2 modules compiled), WT2 KLD came back 16.705139 against a
//! 0.043776 baseline.
//!
//! ## Why cross-check instead of a host reference
//!
//! Replicating a WMMA kernel on the host means reproducing fp16 activation
//! conversion, 16x16 tiling, and accumulation order — a reference that is itself
//! more likely to be wrong than the kernel. Instead: encode ONE set of weights
//! into BOTH containers and run each through its OWN GEMM with the same `x`.
//!
//! Both paths then share every stage except the 8 header bytes and their decode.
//! v1 quantizes with one affine grid per 256 weights; v2 with one per 128. v2 is
//! therefore slightly MORE accurate, so agreement should sit at the scale of
//! 4-bit quantization noise. A systematic blow-up isolates the v2 header decode —
//! in practice the half-select predicate, which the spec calls out as "the single
//! highest-risk detail in the port" because a wrong one "compiles, runs, and
//! silently applies the wrong scale to half of every tensor."
//!
//! ## Why sweep batch size
//!
//! The residual launcher picks among `ldsstage`, `bt4`/`bt8`/`bt12`, and the plain
//! body by batch size and flags. Scoring compiled `_bt8` and `_bt12`, so the BT
//! bodies are live — and BT is b-transposed, which changes the nibble addressing
//! the half-select must be derived from. Sweeping batch size tells us WHICH body
//! is wrong rather than just that something is.
//!
//! Run: `cargo run --release -p hipfire-runtime --example mq4v2_gemm_parity`

use half::f16;
use rdna_compute::Gpu;

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Realistic post-FWHT weights: roughly Gaussian, sigma ~0.011 as measured on the
/// Qwen3.8-27B parent. Deliberately NOT the disjoint-halves fixture -- here both
/// containers must be individually reasonable so their outputs are comparable.
fn build_weights(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for (i, v) in w.iter_mut().enumerate() {
        // Box-Muller from two deterministic uniforms.
        let u1 = prng(i, 0x1234_5678).max(1e-7);
        let u2 = prng(i, 0x9ABC_DEF0);
        *v = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos() * 0.011;
    }
    w
}

/// qt=13 / HFQ4 container: `[0..4) f32 scale, [4..8) f32 zero` over all 256.
fn pack_v1(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let s = &w[src..src + GROUP];
            let lo = s.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
            blob[dst..dst + 4].copy_from_slice(&step.to_le_bytes());
            blob[dst + 4..dst + 8].copy_from_slice(&lo.to_le_bytes());
            let inv = if step > 0.0 { 1.0 / step } else { 0.0 };
            let mut q = [0u8; GROUP];
            for i in 0..GROUP {
                q[i] = ((s[i] - lo) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
            }
            for i in 0..HALF {
                blob[dst + 8 + i] = (q[2 * i] & 0xF) | ((q[2 * i + 1] & 0xF) << 4);
            }
        }
    }
    blob
}

/// qt=44 / HFQ4-v2 container: fp16 scale+zero per 128, halves at [0..4) and [4..8).
fn pack_v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
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
                let sb = if hi == lo { 0u16 } else { f16::from_f32(step).to_bits() };
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

/// Exact dequant of a blob per its container, then y = W·x in f64. Used only to
/// bound how far apart v1 and v2 SHOULD be, so the pass threshold is derived from
/// the data rather than guessed.
fn ref_gemm(blob: &[u8], x: &[f32], m: usize, k: usize, batch: usize, v2: bool) -> Vec<f64> {
    let gpr = k / GROUP;
    let mut y = vec![0.0f64; batch * m];
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GROUP_BYTES;
            let (mut sc, mut zp) = ([0.0f32; 2], [0.0f32; 2]);
            if v2 {
                for h in 0..2 {
                    let s = u16::from_le_bytes([blob[dst + h * 4], blob[dst + h * 4 + 1]]);
                    let z = u16::from_le_bytes([blob[dst + h * 4 + 2], blob[dst + h * 4 + 3]]);
                    sc[h] = f16::from_bits(s).to_f32();
                    zp[h] = f16::from_bits(z).to_f32();
                }
            } else {
                let s = f32::from_le_bytes([
                    blob[dst], blob[dst + 1], blob[dst + 2], blob[dst + 3],
                ]);
                let z = f32::from_le_bytes([
                    blob[dst + 4], blob[dst + 5], blob[dst + 6], blob[dst + 7],
                ]);
                sc = [s, s];
                zp = [z, z];
            }
            for i in 0..HALF {
                let byte = blob[dst + 8 + i];
                for (nib, idx) in [((byte & 0xF) as f32, 2 * i), ((byte >> 4) as f32, 2 * i + 1)] {
                    let h = idx / HALF;
                    let wv = (zp[h] + sc[h] * nib) as f64;
                    let col = g * GROUP + idx;
                    for b in 0..batch {
                        y[b * m + r] += wv * x[b * k + col] as f64;
                    }
                }
            }
        }
    }
    y
}

fn rel_rms(a: &[f32], b: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        num += ((x as f64) - y) * ((x as f64) - y);
        den += y * y;
    }
    (num / den.max(1e-30)).sqrt()
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("mq4v2_gemm_parity: no GPU ({e}) — skipping");
            return;
        }
    };
    eprintln!("mq4v2_gemm_parity: arch={}", gpu.arch);

    let (m, k) = (256usize, 512usize);
    let w = build_weights(m, k);
    let b1 = pack_v1(&w, m, k);
    let b2 = pack_v2(&w, m, k);

    // Batch sweep: 1 exercises the small/plain body, 8/12/16 push into the BT
    // bodies (_bt8 / _bt12 compiled during scoring), 32 goes past them.
    let mut failures = Vec::new();
    for &batch in &[1usize, 8, 12, 16, 32] {
        let x: Vec<f32> = (0..batch * k)
            .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
            .collect();

        let want_v1 = ref_gemm(&b1, &x, m, k, batch, false);
        let want_v2 = ref_gemm(&b2, &x, m, k, batch, true);

        let run = |gpu: &mut Gpu, blob: &[u8], v2: bool| -> Vec<f32> {
            let d_a = gpu.upload_raw(blob, &[blob.len()]).unwrap();
            let d_x = gpu.upload_f32(&x, &[batch * k]).unwrap();
            let d_y = gpu.zeros(&[batch * m], rdna_compute::DType::F32).unwrap();
            if v2 {
                gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2(&d_a, &d_x, &d_y, m, k, batch)
                    .expect("v2 residual wmma launch");
            } else {
                gpu.gemm_hfq4g256_residual_wmma_gfx12(&d_a, &d_x, &d_y, m, k, batch)
                    .expect("v1 residual wmma launch");
            }
            gpu.hip.device_synchronize().unwrap();
            gpu.download_f32(&d_y).unwrap()
        };

        let got_v1 = run(&mut gpu, &b1, false);
        let got_v2 = run(&mut gpu, &b2, true);

        // v1-vs-own-reference calibrates the WMMA fp16 error floor for this shape;
        // v2 must not be dramatically worse against ITS own reference.
        let e1 = rel_rms(&got_v1, &want_v1);
        let e2 = rel_rms(&got_v2, &want_v2);
        let verdict = if e2 <= (e1 * 4.0).max(0.05) { "ok" } else { "FAIL" };
        eprintln!(
            "residual batch {batch:>3}: v1 rel-rms {e1:.4e}   v2 rel-rms {e2:.4e}   ratio {:.2}x   {verdict}",
            e2 / e1.max(1e-30)
        );
        if verdict == "FAIL" {
            failures.push((batch, e1, e2));
        }
    }

    // ── The fused multi-output GEMMs ────────────────────────────────────────
    //
    // These are the rest of the live v2 set. `gemm_qkvza` carries NINETEEN header
    // sites -- more than a third of every v2 site in the tree -- so it has by far
    // the highest prior probability of a bad half-select. Same cross-check: one
    // set of weights, both containers, each through its own kernel, each compared
    // against its OWN exact dequant.
    let batch = 12usize; // in the BT band that scoring actually compiled
    let x: Vec<f32> = (0..batch * k)
        .map(|i| prng(i, 0xBEEF_0001) * 2.0 - 1.0)
        .collect();
    let d_x = gpu.upload_f32(&x, &[batch * k]).unwrap();

    // Two-output gate_up.
    {
        let wg = build_weights(m, k);
        let wu = build_weights(m, k);
        let (g1, u1) = (pack_v1(&wg, m, k), pack_v1(&wu, m, k));
        let (g2, u2) = (pack_v2(&wg, m, k), pack_v2(&wu, m, k));
        let mut worst = 0.0f64;
        for (tag, bg, bu, v2) in [("v1", &g1, &u1, false), ("v2", &g2, &u2, true)] {
            let d_g = gpu.upload_raw(bg, &[bg.len()]).unwrap();
            let d_u = gpu.upload_raw(bu, &[bu.len()]).unwrap();
            let y_g = gpu.zeros(&[batch * m], rdna_compute::DType::F32).unwrap();
            let y_u = gpu.zeros(&[batch * m], rdna_compute::DType::F32).unwrap();
            if v2 {
                gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2(&d_g, &d_u, &d_x, &y_g, &y_u, m, m, k, batch).expect("v2 gate_up");
            } else {
                gpu.gemm_gate_up_hfq4g256_wmma_gfx12(&d_g, &d_u, &d_x, &y_g, &y_u, m, m, k, batch).expect("v1 gate_up");
            }
            gpu.hip.device_synchronize().unwrap();
            let eg = rel_rms(&gpu.download_f32(&y_g).unwrap(), &ref_gemm(bg, &x, m, k, batch, v2));
            let eu = rel_rms(&gpu.download_f32(&y_u).unwrap(), &ref_gemm(bu, &x, m, k, batch, v2));
            eprintln!("gate_up {tag}: gate rel-rms {eg:.4e}  up rel-rms {eu:.4e}");
            if v2 {
                worst = eg.max(eu);
            }
        }
        if worst > 0.05 {
            failures.push((1000, 0.0, worst));
            eprintln!("  -> gate_up v2 FAIL (rel-rms {worst:.4e})");
        }
    }

    // Four-output qkvza.
    {
        let ws: Vec<Vec<f32>> = (0..4).map(|_| build_weights(m, k)).collect();
        let p1: Vec<Vec<u8>> = ws.iter().map(|w| pack_v1(w, m, k)).collect();
        let p2: Vec<Vec<u8>> = ws.iter().map(|w| pack_v2(w, m, k)).collect();
        let mut worst = 0.0f64;
        for (tag, p, v2) in [("v1", &p1, false), ("v2", &p2, true)] {
            let d: Vec<_> = p.iter().map(|b| gpu.upload_raw(b, &[b.len()]).unwrap()).collect();
            let y: Vec<_> = (0..4).map(|_| gpu.zeros(&[batch * m], rdna_compute::DType::F32).unwrap()).collect();
            if v2 {
                gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2(&d[0], &d[1], &d[2], &d[3], &d_x, &y[0], &y[1], &y[2], &y[3], m, m, m, m, k, batch).expect("v2 qkvza");
            } else {
                gpu.gemm_qkvza_hfq4g256_wmma_gfx12(&d[0], &d[1], &d[2], &d[3], &d_x, &y[0], &y[1], &y[2], &y[3], m, m, m, m, k, batch).expect("v1 qkvza");
            }
            gpu.hip.device_synchronize().unwrap();
            let errs: Vec<f64> = (0..4)
                .map(|i| rel_rms(&gpu.download_f32(&y[i]).unwrap(), &ref_gemm(&p[i], &x, m, k, batch, v2)))
                .collect();
            eprintln!("qkvza  {tag}: qkv {:.4e}  z {:.4e}  beta {:.4e}  alpha {:.4e}", errs[0], errs[1], errs[2], errs[3]);
            if v2 {
                worst = errs.iter().cloned().fold(0.0f64, f64::max);
            }
        }
        if worst > 0.05 {
            failures.push((2000, 0.0, worst));
            eprintln!("  -> qkvza v2 FAIL (rel-rms {worst:.4e})");
        }
    }

    if failures.is_empty() {
        eprintln!("\nmq4v2_gemm_parity: PASS — every live v2 WMMA GEMM matches its own exact dequant");
    } else {
        eprintln!("\nmq4v2_gemm_parity: FAIL — codes {:?}", failures.iter().map(|f| f.0).collect::<Vec<_>>());
        eprintln!("(1000 = gate_up, 2000 = qkvza, otherwise the residual batch size)");
        eprintln!("The v1 row is the WMMA fp16 error floor; a v2 row far above it means that");
        eprintln!("kernel mis-decodes its own header. Each body has its OWN nibble addressing,");
        eprintln!("so its half-select predicate must be derived from that addressing, never copied.");
        std::process::exit(1);
    }
}
