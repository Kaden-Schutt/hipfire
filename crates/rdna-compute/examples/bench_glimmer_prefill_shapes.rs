// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Architecture-safe attribution oracle for Muse Glimmer's prefill projection path.
//!
//! Compares the production gfx1100 route (`gemm_hfq4g256_batched_lmhead`, which
//! is overwrite-=`=` semantics and internally zeros Y) against:
//!   - `gemm_hfq4g256_residual` with correctly zeroed accumulation (Y+=W@X)
//!   - fused `gate+up` (`gemm_gate_up_hfq4g256`) where applicable
//!   - Muse-owned `gemm_hfq4g256_residual_muse` (gfx1201 only, bt=12, B%192==0)
//!   - Muse-owned `gemm_hfq4g256_residual_muse_gfx1100` gate_proj BT sweep
//!     (gfx1100 only, exact M=19968 K=6656, bt ∈ {4,6,8,12,16}, full-tile B)
//!   - Muse-owned `gemm_hfq4g256_residual_muse_gfx1100_cb` LDS-codebook BT
//!     candidate (gfx1100 only, exact M=19968 K=6656, bt ∈ {4,6,12}, full-tile B)
//!   - Muse-owned `gemm_hfq4g256_residual_muse_gfx1100_lds_g256` LDS X-group
//!     candidate (gfx1100 only, exact M=19968 K=6656, batch%96==0)
//!   - Muse-owned `gemm_hfq4g256_residual_muse_gfx1100_rm` row-reuse RM×BV
//!     sweep (gfx1100 only, exact M=19968 K=6656, rm ∈ {2,3,4,6}, RM*BV=12,
//!     full-tile B divisible by 16*BV)
//!   - Muse-owned `gemm_hfq4g256_residual_muse_gfx1100_rm_pk` packed-half2
//!     dequant row-reuse (gfx1100 only, exact M=19968 K=6656, rm ∈ {1,2}
//!     → BV 12/6, full-tile B)
//!   - Muse-owned `gemm_hfq4g256_residual_muse_gfx1100_rm2_pipe` RM2/BV6
//!     two-slot X-fragment pipeline (gfx1100 only, exact M=19968 K=6656 B=192;
//!     scalar + packed-half2 symbols)
//!
//! For each exact Muse shape and B=128/192/256 reports time, TFLOP/s,
//! bitdiff, max_abs, max_rel versus the production baseline. The gfx1201 Muse
//! sibling is called only on gfx1201 and eligible tile widths; the gfx1100
//! gate_proj BT/codebook/row-reuse/pipe candidates are called only when
//! `arch_caps.is_gfx1100()` so the example completes on other arches without
//! binding those kernels.
//!
//! Dominant FLOPs: gate/up/down are 82.4% of layer params, attention
//! projections 17.6% — the table is weighted accordingly.
//!
//! Usage: bench_glimmer_prefill_shapes [B ...]
//!   default B = 128 192 256

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn build_hfq4g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    let bpr = gpr * 136;
    let mut out = vec![0u8; m * bpr];
    let mix = |x: u64| {
        let h = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
    };
    let s0 = seed as u64;
    for row in 0..m {
        for g in 0..gpr {
            let off = row * bpr + g * 136;
            let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
            let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
            let scale = 0.01 + (((r1 as u32) % 4001) as f32) * 1e-5;
            let zero = (((r2 as u32) % 1500) as f32) * 1e-4 - 0.075;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte_i in 0..128 {
                let r = mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (byte_i as u64));
                out[off + 8 + byte_i] = (r & 0xff) as u8;
            }
        }
    }
    out
}

fn correctness_stats(a: &[f32], b: &[f32]) -> (usize, f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut bitdiff = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            bitdiff += 1;
        }
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = x.abs().max(1e-7);
        let rel = d / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    (bitdiff, max_abs, max_rel)
}

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let batches: Vec<usize> = {
        let a: Vec<usize> = std::env::args()
            .skip(1)
            .filter_map(|s| s.parse().ok())
            .collect();
        if a.is_empty() {
            vec![128, 192, 256]
        } else {
            a
        }
    };

    let dim = 6656usize;
    let ffn = 19968usize;
    // Exact Muse shapes. M rows, K cols (HFQ4G256, k%256==0).
    let shapes: Vec<(&str, usize, usize)> = vec![
        ("q_proj", 4096, dim),
        ("k_proj", 256, dim),
        ("v_proj", 256, dim),
        ("attn_gate", 4096, dim),
        ("o_proj", dim, 4096),
        ("gate_proj", ffn, dim),
        ("up_proj", ffn, dim),
        ("down_proj", dim, ffn),
    ];

    let mut gpu = Gpu::init().expect("GPU init");
    let arch = gpu.arch_caps.arch().to_string();
    let is_gfx1201 = arch.starts_with("gfx1201");
    let is_gfx1100 = gpu.arch_caps.is_gfx1100();
    println!(
        "arch: {}  is_gfx1201={}  is_gfx1100={}",
        arch, is_gfx1201, is_gfx1100
    );
    println!("shapes: dim={} ffn={}  B={:?}", dim, ffn, batches);
    println!("dominant FLOP family: gate/up/down 82.4%, attention 17.6%, o_proj单独");

    // Sustained DPM warmup — preserve. A 2-iter warm-up reads ~40% low while
    // DPM ramps, corrupting cross-B comparisons. Drive real work ~1.5 s.
    {
        let (wm, wk, wb) = (19968usize, 6656usize, 256usize);
        let ww = gpu
            .upload_raw(&build_hfq4g256(wm, wk, 0x5C), &[wm, wk])
            .expect("warm w");
        let wxv: Vec<f32> = (0..wb * wk).map(|i| ((i % 61) as f32 - 30.0) * 0.01).collect();
        let wx = gpu.upload_f32(&wxv, &[wb, wk]).expect("warm x");
        let wy = gpu.alloc_tensor(&[wb, wm], DType::F32).expect("warm y");
        let t = Instant::now();
        while t.elapsed().as_secs_f64() < 1.5 {
            for _ in 0..8 {
                let _ = gpu.gemm_hfq4g256_residual(&ww, &wx, &wy, wm, wk, wb);
            }
            let _ = gpu.hip.device_synchronize();
        }
        let _ = gpu.free_tensor(wy);
        let _ = gpu.free_tensor(wx);
        let _ = gpu.free_tensor(ww);
        println!("(warm-up complete)");
    }

    // Header for the attribution table. Each logical row is one (shape,B)
    // baseline; candidate columns are printed as sibling rows so the
    // correctness/speed oracle is a single grep-able table.
    println!(
        "{:<12} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "shape", "B", "kernel", "ms", "TFLOP/s", "bitdiff", "max_abs", "max_rel", "vs_base"
    );
    println!("{}", "-".repeat(110));

    // Reusable gate/up weights for the fused measurement — avoid huge
    // duplicate allocations by uploading once and reusing across B.
    let gate_w_raw = build_hfq4g256(ffn, dim, 0xB1);
    let up_w_raw = build_hfq4g256(ffn, dim, 0xB2);
    let gate_w = gpu
        .upload_raw(&gate_w_raw, &[ffn, dim])
        .expect("gate w");
    let up_w = gpu.upload_raw(&up_w_raw, &[ffn, dim]).expect("up w");
    // Drop host copies after upload (keep device only) to avoid resident duplicate.
    drop(gate_w_raw);
    drop(up_w_raw);

    for &b in &batches {
        // Shared X for this batch width and K=dim (most shapes). Reuse allocation
        // per b to avoid per-shape X uploads where K matches; shapes with K=4096
        // or 19968 need their own X, handled below via separate upload when k != dim.
        let x_dim_f32: Vec<f32> = (0..b * dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let x_dim = gpu
            .upload_f32(&x_dim_f32, &[b, dim])
            .expect("x dim");

        // Fused gate+up: one call producing both outputs from shared x.
        // Compare each output against its separate batched_lmhead baseline.
        {
            let yg = gpu.alloc_tensor(&[b, ffn], DType::F32).expect("yg");
            let yu = gpu.alloc_tensor(&[b, ffn], DType::F32).expect("yu");
            let y_gate_base = gpu.alloc_tensor(&[b, ffn], DType::F32).expect("y gate base");
            let y_up_base = gpu.alloc_tensor(&[b, ffn], DType::F32).expect("y up base");

            // Warmup fused (2 iters)
            for _ in 0..2 {
                let _ = gpu.gemm_gate_up_hfq4g256(&gate_w, &up_w, &x_dim, &yg, &yu, ffn, ffn, dim, b);
            }
            let _ = gpu.hip.device_synchronize();

            let iters = if b >= 512 { 6 } else { 12 };
            let mut reps: Vec<f64> = Vec::new();
            for _ in 0..3 {
                let t0 = Instant::now();
                for _ in 0..iters {
                    gpu.gemm_gate_up_hfq4g256(&gate_w, &up_w, &x_dim, &yg, &yu, ffn, ffn, dim, b)
                        .expect("fused gate_up");
                }
                let _ = gpu.hip.device_synchronize();
                reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
            }
            let ms_fused = median_ms(reps.clone());
            let tflops_fused =
                2.0 * (ffn as f64 + ffn as f64) * (dim as f64) * (b as f64) / (ms_fused / 1000.0) / 1e12;

            // Baseline: two separate batched_lmhead calls (production path) for reference.
            // Time them together as the cost Glimmer pays today (gate then up).
            for _ in 0..2 {
                let _ = gpu.gemm_hfq4g256_batched_lmhead(&gate_w, &x_dim, &y_gate_base, ffn, dim, b);
                let _ = gpu.gemm_hfq4g256_batched_lmhead(&up_w, &x_dim, &y_up_base, ffn, dim, b);
            }
            let _ = gpu.hip.device_synchronize();
            let mut base_reps: Vec<f64> = Vec::new();
            for _ in 0..3 {
                let t0 = Instant::now();
                for _ in 0..iters {
                    gpu.gemm_hfq4g256_batched_lmhead(&gate_w, &x_dim, &y_gate_base, ffn, dim, b)
                        .expect("gate base");
                    gpu.gemm_hfq4g256_batched_lmhead(&up_w, &x_dim, &y_up_base, ffn, dim, b)
                        .expect("up base");
                }
                let _ = gpu.hip.device_synchronize();
                base_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
            }
            let ms_base = median_ms(base_reps);
            let tflops_base =
                2.0 * (ffn as f64 + ffn as f64) * (dim as f64) * (b as f64) / (ms_base / 1000.0) / 1e12;

            // Correctness: fused outputs vs separate baseline outputs.
            let _ = gpu.gemm_gate_up_hfq4g256(&gate_w, &up_w, &x_dim, &yg, &yu, ffn, ffn, dim, b)
                .expect("fused for dl");
            let _ = gpu.hip.device_synchronize();
            let _ = gpu.gemm_hfq4g256_batched_lmhead(&gate_w, &x_dim, &y_gate_base, ffn, dim, b)
                .expect("gate base dl");
            let _ = gpu.gemm_hfq4g256_batched_lmhead(&up_w, &x_dim, &y_up_base, ffn, dim, b)
                .expect("up base dl");
            let _ = gpu.hip.device_synchronize();
            let fused_g = gpu.download_f32(&yg).expect("dl fused g");
            let fused_u = gpu.download_f32(&yu).expect("dl fused u");
            let base_g = gpu.download_f32(&y_gate_base).expect("dl base g");
            let base_u = gpu.download_f32(&y_up_base).expect("dl base u");
            let (bdiff_g, maxabs_g, maxrel_g) = correctness_stats(&base_g, &fused_g);
            let (bdiff_u, maxabs_u, maxrel_u) = correctness_stats(&base_u, &fused_u);
            // Report the worse of gate/up for the fused family.
            let bdiff = bdiff_g + bdiff_u;
            let maxabs = maxabs_g.max(maxabs_u);
            let maxrel = maxrel_g.max(maxrel_u);

            let vs = 100.0 * (ms_base / ms_fused - 1.0);
            println!(
                "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%",
                "gate+up", b, "FUSED", ms_fused, tflops_fused, bdiff, maxabs, maxrel, vs
            );
            println!(
                "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9} {:>9} {:>9}",
                "gate+up", b, "2x_batched", ms_base, tflops_base, "-", "-", "-", "-"
            );
            // Also report per-output detail at verbose level as two sub-rows
            println!(
                "  {:<10} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9.2e} {:>9.2e} {:>9}",
                "gate", b, "fused_vs_base", "-", "-", bdiff_g, maxabs_g, maxrel_g, ""
            );
            println!(
                "  {:<10} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9.2e} {:>9.2e} {:>9}",
                "up", b, "fused_vs_base", "-", "-", bdiff_u, maxabs_u, maxrel_u, ""
            );

            let _ = gpu.free_tensor(y_up_base);
            let _ = gpu.free_tensor(y_gate_base);
            let _ = gpu.free_tensor(yu);
            let _ = gpu.free_tensor(yg);
        }

        for (label, m, k) in &shapes {
            if *k == dim {
                let x = &x_dim;
                // Select backing weight: reuse gate_w/up_w for those labels to avoid
                // re-uploading 19968x6656 (~17.6 MB compressed) per B.
                let w: rdna_compute::GpuTensor;
                let w_ref: &rdna_compute::GpuTensor;
                // Temporary owned weight for non-gate/up shapes; freed after this shape.
                let mut owned_w: Option<rdna_compute::GpuTensor> = None;
                if *label == "gate_proj" {
                    w_ref = &gate_w;
                } else if *label == "up_proj" {
                    w_ref = &up_w;
                } else {
                    let wr = build_hfq4g256(*m, *k, 0xA7);
                    w = gpu.upload_raw(&wr, &[*m, *k]).expect("w");
                    owned_w = Some(w);
                    w_ref = owned_w.as_ref().unwrap();
                }

                let y_base = gpu.alloc_tensor(&[b, *m], DType::F32).expect("y_base");
                let y_resid = gpu.alloc_tensor(&[b, *m], DType::F32).expect("y_resid");

                // ---- Baseline: production gemm_hfq4g256_batched_lmhead (overwrite) ----
                for _ in 0..2 {
                    let _ = gpu.gemm_hfq4g256_batched_lmhead(w_ref, x, &y_base, *m, *k, b);
                }
                let _ = gpu.hip.device_synchronize();
                let iters = if b >= 512 { 6 } else { 12 };
                let mut base_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        gpu.gemm_hfq4g256_batched_lmhead(w_ref, x, &y_base, *m, *k, b)
                            .expect("batched");
                    }
                    let _ = gpu.hip.device_synchronize();
                    base_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_base = median_ms(base_reps);
                let tflops_base =
                    2.0 * (*m as f64) * (*k as f64) * (b as f64) / (ms_base / 1000.0) / 1e12;

                // ---- Candidate: gemm_hfq4g256_residual with correctly zeroed Y (accumulate) ----
                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y_resid.buf, 0, b * *m * 4);
                    let _ = gpu.gemm_hfq4g256_residual(w_ref, x, &y_resid, *m, *k, b);
                }
                let _ = gpu.hip.device_synchronize();
                let mut resid_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y_resid.buf, 0, b * *m * 4);
                        gpu.gemm_hfq4g256_residual(w_ref, x, &y_resid, *m, *k, b)
                            .expect("residual");
                    }
                    let _ = gpu.hip.device_synchronize();
                    resid_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_resid = median_ms(resid_reps);
                let tflops_resid =
                    2.0 * (*m as f64) * (*k as f64) * (b as f64) / (ms_resid / 1000.0) / 1e12;

                // Correctness: baseline vs residual (both should be bit-identical; residual
                // needs zeroed Y, batched zeros internally). Report bitdiff etc.
                let _ = gpu.gemm_hfq4g256_batched_lmhead(w_ref, x, &y_base, *m, *k, b)
                    .expect("base for dl");
                let _ = gpu.hip.memset(&y_resid.buf, 0, b * *m * 4);
                gpu.gemm_hfq4g256_residual(w_ref, x, &y_resid, *m, *k, b)
                    .expect("resid for dl");
                let _ = gpu.hip.device_synchronize();
                let base_host = gpu.download_f32(&y_base).expect("dl base");
                let resid_host = gpu.download_f32(&y_resid).expect("dl resid");
                let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &resid_host);
                let vs = 100.0 * (ms_base / ms_resid - 1.0);

                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9} {:>9} {:>9}",
                    label, b, "batched", ms_base, tflops_base, "-", "-", "-", "-"
                );
                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%",
                    label, b, "residual+zero", ms_resid, tflops_resid, bdiff, maxabs, maxrel, vs
                );

                // ---- Muse-owned gfx12 sibling: only on gfx1201 and bt=12 (B%192==0) ----
                // The kernel is instantiated only for bt=12 (16*12=192). Earlier bt=8/4
                // variants were removed after measurement (no win / spill). It
                // returns Ok(false) when ineligible rather than launching, so we
                // must not treat that as an error, and must not call it at all on
                // gfx1100/gfx1151 (architecture-safe gate).
                let muse_eligible = is_gfx1201 && b % 192 == 0;
                if muse_eligible {
                    let bt = 12usize;
                    // Use a separate y buffer for muse so we don't disturb the
                    // baseline/residual correctness pair; still only one extra alloc
                    // per shape (avoid huge duplicate allocations by reusing y_resid's
                    // storage size — allocate a fresh tensor of same shape).
                    let y_muse = gpu.alloc_tensor(&[b, *m], DType::F32).expect("y_muse");
                    // Warmup muse (2 iters, discard)
                    for _ in 0..2 {
                        let _ = gpu.hip.memset(&y_muse.buf, 0, b * *m * 4);
                        let _ = gpu.gemm_hfq4g256_residual_muse(w_ref, x, &y_muse, *m, *k, b, bt);
                    }
                    let _ = gpu.hip.device_synchronize();

                    // Correctness vs baseline
                    let _ = gpu.hip.memset(&y_muse.buf, 0, b * *m * 4);
                    let used = gpu
                        .gemm_hfq4g256_residual_muse(w_ref, x, &y_muse, *m, *k, b, bt)
                        .expect("muse");
                    let _ = gpu.hip.device_synchronize();
                    if used {
                        let muse_host = gpu.download_f32(&y_muse).expect("dl muse");
                        // baseline already downloaded as base_host above, but re-download
                        // after ensuring y_base still holds baseline result.
                        let (bdiff_m, maxabs_m, maxrel_m) = correctness_stats(&base_host, &muse_host);
                        // Timed muse
                        let mut muse_reps: Vec<f64> = Vec::new();
                        for _ in 0..3 {
                            let t0 = Instant::now();
                            for _ in 0..iters {
                                let _ = gpu.hip.memset(&y_muse.buf, 0, b * *m * 4);
                                gpu.gemm_hfq4g256_residual_muse(w_ref, x, &y_muse, *m, *k, b, bt)
                                    .expect("muse bench");
                            }
                            let _ = gpu.hip.device_synchronize();
                            muse_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                        }
                        let ms_muse = median_ms(muse_reps);
                        let tflops_muse = 2.0 * (*m as f64) * (*k as f64) * (b as f64)
                            / (ms_muse / 1000.0)
                            / 1e12;
                        let vs_muse = 100.0 * (ms_base / ms_muse - 1.0);
                        println!(
                            "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  bt={}",
                            label, b, "muse_bt12", ms_muse, tflops_muse, bdiff_m, maxabs_m, maxrel_m, vs_muse, bt
                        );
                    } else {
                        println!(
                            "{:<12} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
                            label, b, "muse_bt12", "-", "-", "inelig", "-", "-", "-"
                        );
                    }
                    let _ = gpu.free_tensor(y_muse);
                } else if is_gfx1201 {
                    // On gfx1201 but B not multiple of 192 -> not eligible, report skip
                    // without calling the kernel (avoids Ok(false) path confusion).
                    println!(
                        "{:<12} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
                        label, b, "muse_bt12", "-", "-", "skip", "-", "-", "-"
                    );
                } else {
                    // On gfx1100/gfx1151 — must not call muse at all. Report arch-skip.
                    println!(
                        "{:<12} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
                        label, b, "muse_bt12", "-", "-", "arch_skip", "-", "-", "-"
                    );
                }

                let _ = gpu.free_tensor(y_resid);
                let _ = gpu.free_tensor(y_base);
                if let Some(wt) = owned_w {
                    let _ = gpu.free_tensor(wt);
                }
            } else {
                // k != dim path: o_proj (6656x4096) and down_proj (6656x19968).
                // Allocate K-specific X and weight for this shape.
                let xv: Vec<f32> = (0..b * *k)
                    .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                    .collect();
                let xk = gpu.upload_f32(&xv, &[b, *k]).expect("xk");
                let wr = build_hfq4g256(*m, *k, 0xA7);
                let w = gpu.upload_raw(&wr, &[*m, *k]).expect("w");
                drop(wr);
                let y_base = gpu.alloc_tensor(&[b, *m], DType::F32).expect("y_base");
                let y_resid = gpu.alloc_tensor(&[b, *m], DType::F32).expect("y_resid");

                for _ in 0..2 {
                    let _ = gpu.gemm_hfq4g256_batched_lmhead(&w, &xk, &y_base, *m, *k, b);
                }
                let _ = gpu.hip.device_synchronize();
                let iters = if b >= 512 { 6 } else { 12 };
                let mut base_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        gpu.gemm_hfq4g256_batched_lmhead(&w, &xk, &y_base, *m, *k, b)
                            .expect("batched");
                    }
                    let _ = gpu.hip.device_synchronize();
                    base_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_base = median_ms(base_reps);
                let tflops_base =
                    2.0 * (*m as f64) * (*k as f64) * (b as f64) / (ms_base / 1000.0) / 1e12;

                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y_resid.buf, 0, b * *m * 4);
                    let _ = gpu.gemm_hfq4g256_residual(&w, &xk, &y_resid, *m, *k, b);
                }
                let _ = gpu.hip.device_synchronize();
                let mut resid_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y_resid.buf, 0, b * *m * 4);
                        gpu.gemm_hfq4g256_residual(&w, &xk, &y_resid, *m, *k, b)
                            .expect("residual");
                    }
                    let _ = gpu.hip.device_synchronize();
                    resid_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_resid = median_ms(resid_reps);
                let tflops_resid =
                    2.0 * (*m as f64) * (*k as f64) * (b as f64) / (ms_resid / 1000.0) / 1e12;

                let _ = gpu.gemm_hfq4g256_batched_lmhead(&w, &xk, &y_base, *m, *k, b)
                    .expect("base dl");
                let _ = gpu.hip.memset(&y_resid.buf, 0, b * *m * 4);
                gpu.gemm_hfq4g256_residual(&w, &xk, &y_resid, *m, *k, b)
                    .expect("resid dl");
                let _ = gpu.hip.device_synchronize();
                let base_host = gpu.download_f32(&y_base).expect("dl base");
                let resid_host = gpu.download_f32(&y_resid).expect("dl resid");
                let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &resid_host);
                let vs = 100.0 * (ms_base / ms_resid - 1.0);

                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9} {:>9} {:>9}",
                    label, b, "batched", ms_base, tflops_base, "-", "-", "-", "-"
                );
                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%",
                    label, b, "residual+zero", ms_resid, tflops_resid, bdiff, maxabs, maxrel, vs
                );

                let muse_eligible = is_gfx1201 && b % 192 == 0;
                if muse_eligible {
                    let bt = 12usize;
                    let y_muse = gpu.alloc_tensor(&[b, *m], DType::F32).expect("y_muse");
                    for _ in 0..2 {
                        let _ = gpu.hip.memset(&y_muse.buf, 0, b * *m * 4);
                        let _ = gpu.gemm_hfq4g256_residual_muse(&w, &xk, &y_muse, *m, *k, b, bt);
                    }
                    let _ = gpu.hip.device_synchronize();
                    let _ = gpu.hip.memset(&y_muse.buf, 0, b * *m * 4);
                    let used = gpu
                        .gemm_hfq4g256_residual_muse(&w, &xk, &y_muse, *m, *k, b, bt)
                        .expect("muse");
                    let _ = gpu.hip.device_synchronize();
                    if used {
                        let muse_host = gpu.download_f32(&y_muse).expect("dl muse");
                        let (bdiff_m, maxabs_m, maxrel_m) = correctness_stats(&base_host, &muse_host);
                        let mut muse_reps: Vec<f64> = Vec::new();
                        for _ in 0..3 {
                            let t0 = Instant::now();
                            for _ in 0..iters {
                                let _ = gpu.hip.memset(&y_muse.buf, 0, b * *m * 4);
                                gpu.gemm_hfq4g256_residual_muse(&w, &xk, &y_muse, *m, *k, b, bt)
                                    .expect("muse bench");
                            }
                            let _ = gpu.hip.device_synchronize();
                            muse_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                        }
                        let ms_muse = median_ms(muse_reps);
                        let tflops_muse = 2.0 * (*m as f64) * (*k as f64) * (b as f64)
                            / (ms_muse / 1000.0)
                            / 1e12;
                        let vs_muse = 100.0 * (ms_base / ms_muse - 1.0);
                        println!(
                            "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  bt={}",
                            label, b, "muse_bt12", ms_muse, tflops_muse, bdiff_m, maxabs_m, maxrel_m, vs_muse, bt
                        );
                    } else {
                        println!(
                            "{:<12} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
                            label, b, "muse_bt12", "-", "-", "inelig", "-", "-", "-"
                        );
                    }
                    let _ = gpu.free_tensor(y_muse);
                } else if is_gfx1201 {
                    println!(
                        "{:<12} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
                        label, b, "muse_bt12", "-", "-", "skip", "-", "-", "-"
                    );
                } else {
                    println!(
                        "{:<12} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
                        label, b, "muse_bt12", "-", "-", "arch_skip", "-", "-", "-"
                    );
                }
                // Ksplit sweep for exact o_proj (6656x4096) and down_proj
                // (6656x19968) — deterministic K-split phase1 with BT tiling
                // to K_SPLITS=4 scratch then existing finalize. Full-tile only
                // (batch % (16*bt) == 0); bt 4/6/12. Exact M=6656,
                // K∈{4096,19968}. Compared to same fresh batched_lmhead baseline
                // (which zeros internally). Candidate Y zeroed before each
                // residual API call. Two warmups, three medians, same iters.
                // Skips Ok(false) and never calls on non-gfx1100.
                if is_gfx1100 {
                    let y_ks = gpu.alloc_tensor(&[b, *m], DType::F32).expect("y_ks");
                    const G11_KS_BTS: [usize; 3] = [4, 6, 12];
                    let flops = 2.0 * (*m as f64) * (*k as f64) * (b as f64);
                    for &bt in &G11_KS_BTS {
                        let _ = gpu.hip.memset(&y_ks.buf, 0, b * *m * 4);
                        let used = gpu
                            .gemm_hfq4g256_residual_muse_gfx1100_ksplit(
                                &w, &xk, &y_ks, *m, *k, b, bt,
                            )
                            .expect("muse_g11_ks probe");
                        let _ = gpu.hip.device_synchronize();
                        if !used {
                            continue;
                        }

                        for _ in 0..2 {
                            let _ = gpu.hip.memset(&y_ks.buf, 0, b * *m * 4);
                            let _ = gpu.gemm_hfq4g256_residual_muse_gfx1100_ksplit(
                                &w, &xk, &y_ks, *m, *k, b, bt,
                            );
                        }
                        let _ = gpu.hip.device_synchronize();

                        let _ = gpu.hip.memset(&y_ks.buf, 0, b * *m * 4);
                        let ok = gpu
                            .gemm_hfq4g256_residual_muse_gfx1100_ksplit(
                                &w, &xk, &y_ks, *m, *k, b, bt,
                            )
                            .expect("muse_g11_ks dl");
                        let _ = gpu.hip.device_synchronize();
                        if !ok {
                            continue;
                        }
                        let cand_host = gpu.download_f32(&y_ks).expect("dl muse_g11_ks");
                        let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &cand_host);

                        let mut cand_reps: Vec<f64> = Vec::new();
                        for _ in 0..3 {
                            let t0 = Instant::now();
                            for _ in 0..iters {
                                let _ = gpu.hip.memset(&y_ks.buf, 0, b * *m * 4);
                                gpu.gemm_hfq4g256_residual_muse_gfx1100_ksplit(
                                    &w, &xk, &y_ks, *m, *k, b, bt,
                                )
                                .expect("muse_g11_ks bench");
                            }
                            let _ = gpu.hip.device_synchronize();
                            cand_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                        }
                        let ms_cand = median_ms(cand_reps);
                        let tflops_cand = flops / (ms_cand / 1000.0) / 1e12;
                        let vs = 100.0 * (ms_base / ms_cand - 1.0);
                        let klabel = format!("muse_g11_ks{}", bt);
                        println!(
                            "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  bt={}",
                            label, b, klabel, ms_cand, tflops_cand, bdiff, maxabs, maxrel, vs, bt
                        );
                    }
                    let _ = gpu.free_tensor(y_ks);
                }

                let _ = gpu.free_tensor(y_resid);
                let _ = gpu.free_tensor(y_base);
                let _ = gpu.free_tensor(w);
                let _ = gpu.free_tensor(xk);
            }
        }

        // ---- Muse-owned gfx1100 gate_proj BT sweep (exact M=19968, K=6656) ----
        // Candidate widths for measurement only. Host returns Ok(false) when
        // ineligible (wrong arch/shape/bt/full-tile); never call on non-gfx1100.
        // Each arm includes exactly one required zero: the production wrapper
        // zeros internally; the residual candidate is zeroed by this caller.
        if is_gfx1100 {
            let m_gate = ffn;
            let k_gate = dim;
            let y_base = gpu
                .alloc_tensor(&[b, m_gate], DType::F32)
                .expect("y_gate_g11_base");
            let y_cand = gpu
                .alloc_tensor(&[b, m_gate], DType::F32)
                .expect("y_gate_g11_cand");
            let iters = if b >= 512 { 6 } else { 12 };
            let flops = 2.0 * (m_gate as f64) * (k_gate as f64) * (b as f64);

            // Fresh production baseline. batched_lmhead zeros Y internally.
            for _ in 0..2 {
                let _ = gpu.gemm_hfq4g256_batched_lmhead(
                    &gate_w, &x_dim, &y_base, m_gate, k_gate, b,
                );
            }
            let _ = gpu.hip.device_synchronize();
            let mut base_reps: Vec<f64> = Vec::new();
            for _ in 0..3 {
                let t0 = Instant::now();
                for _ in 0..iters {
                    gpu.gemm_hfq4g256_batched_lmhead(
                        &gate_w, &x_dim, &y_base, m_gate, k_gate, b,
                    )
                    .expect("gate g11 base");
                }
                let _ = gpu.hip.device_synchronize();
                base_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
            }
            let ms_base = median_ms(base_reps);
            let tflops_base = flops / (ms_base / 1000.0) / 1e12;

            // Reference output for bitdiff (fresh launch after timed baseline).
            let _ = gpu.hip.memset(&y_base.buf, 0, b * m_gate * 4);
            gpu.gemm_hfq4g256_batched_lmhead(&gate_w, &x_dim, &y_base, m_gate, k_gate, b)
                .expect("gate g11 base dl");
            let _ = gpu.hip.device_synchronize();
            let base_host = gpu.download_f32(&y_base).expect("dl gate g11 base");

            println!(
                "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9} {:>9} {:>9}",
                "gate_proj", b, "g11_batched", ms_base, tflops_base, "-", "-", "-", "-"
            );

            const G11_BTS: [usize; 5] = [4, 6, 8, 12, 16];
            for &bt in &G11_BTS {
                // Probe eligibility without treating Ok(false) as failure.
                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let used = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, bt,
                    )
                    .expect("muse_g11 probe");
                let _ = gpu.hip.device_synchronize();
                if !used {
                    continue;
                }

                // Warmup eligible width (2 iters).
                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                    let _ = gpu.gemm_hfq4g256_residual_muse_gfx1100(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, bt,
                    );
                }
                let _ = gpu.hip.device_synchronize();

                // Correctness vs fresh production gate baseline.
                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let ok = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, bt,
                    )
                    .expect("muse_g11 dl");
                let _ = gpu.hip.device_synchronize();
                if !ok {
                    continue;
                }
                let cand_host = gpu.download_f32(&y_cand).expect("dl muse_g11");
                let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &cand_host);

                let mut cand_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                        gpu.gemm_hfq4g256_residual_muse_gfx1100(
                            &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, bt,
                        )
                        .expect("muse_g11 bench");
                    }
                    let _ = gpu.hip.device_synchronize();
                    cand_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_cand = median_ms(cand_reps);
                let tflops_cand = flops / (ms_cand / 1000.0) / 1e12;
                let vs = 100.0 * (ms_base / ms_cand - 1.0);
                let klabel = format!("muse_g11_bt{}", bt);
                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  bt={}",
                    "gate_proj",
                    b,
                    klabel,
                    ms_cand,
                    tflops_cand,
                    bdiff,
                    maxabs,
                    maxrel,
                    vs,
                    bt
                );
            }

            // Second Muse-owned gfx1100 candidate: LDS HFQ affine codebook.
            // Same shape gate and timing contract as muse_g11_bt*; bt ∈ {4,6,12}.
            // Reuses the already-fresh production base_host (no extra baseline zero).
            const G11_CB_BTS: [usize; 3] = [4, 6, 12];
            for &bt in &G11_CB_BTS {
                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let used = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_cb(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, bt,
                    )
                    .expect("muse_g11_cb probe");
                let _ = gpu.hip.device_synchronize();
                if !used {
                    continue;
                }

                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                    let _ = gpu.gemm_hfq4g256_residual_muse_gfx1100_cb(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, bt,
                    );
                }
                let _ = gpu.hip.device_synchronize();

                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let ok = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_cb(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, bt,
                    )
                    .expect("muse_g11_cb dl");
                let _ = gpu.hip.device_synchronize();
                if !ok {
                    continue;
                }
                let cand_host = gpu.download_f32(&y_cand).expect("dl muse_g11_cb");
                let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &cand_host);

                let mut cand_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                        gpu.gemm_hfq4g256_residual_muse_gfx1100_cb(
                            &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, bt,
                        )
                        .expect("muse_g11_cb bench");
                    }
                    let _ = gpu.hip.device_synchronize();
                    cand_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_cand = median_ms(cand_reps);
                let tflops_cand = flops / (ms_cand / 1000.0) / 1e12;
                let vs = 100.0 * (ms_base / ms_cand - 1.0);
                let klabel = format!("muse_g11_cb{}", bt);
                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  bt={}",
                    "gate_proj",
                    b,
                    klabel,
                    ms_cand,
                    tflops_cand,
                    bdiff,
                    maxabs,
                    maxrel,
                    vs,
                    bt
                );
            }
            // Multiwave gate sweep: groups multiple identical BT4 row waves into
            // one block to expose activation-cache reuse/scheduling for exact
            // gate/up. Full-tile only (batch % 64 == 0); waves 2/4/8. Exact
            // M=19968 K=6656. Grid [ceil(ceil(M/16)/waves), batch/64,1], block
            // [32*waves,1,1]. Y+=, caller zeros. Compared to same fresh
            // g11_batched baseline (which zeros internally). Two warmups, three
            // medians, same iters. Skips Ok(false) and never calls on non-gfx1100.
            const G11_MW_WAVES: [usize; 3] = [2, 4, 8];
            for &waves in &G11_MW_WAVES {
                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let used = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_mw(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, waves,
                    )
                    .expect("muse_g11_mw probe");
                let _ = gpu.hip.device_synchronize();
                if !used {
                    continue;
                }

                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                    let _ = gpu.gemm_hfq4g256_residual_muse_gfx1100_mw(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, waves,
                    );
                }
                let _ = gpu.hip.device_synchronize();

                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let ok = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_mw(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, waves,
                    )
                    .expect("muse_g11_mw dl");
                let _ = gpu.hip.device_synchronize();
                if !ok {
                    continue;
                }
                let cand_host = gpu.download_f32(&y_cand).expect("dl muse_g11_mw");
                let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &cand_host);

                let mut cand_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                        gpu.gemm_hfq4g256_residual_muse_gfx1100_mw(
                            &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, waves,
                        )
                        .expect("muse_g11_mw bench");
                    }
                    let _ = gpu.hip.device_synchronize();
                    cand_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_cand = median_ms(cand_reps);
                let tflops_cand = flops / (ms_cand / 1000.0) / 1e12;
                let vs = 100.0 * (ms_base / ms_cand - 1.0);
                let klabel = format!("muse_g11_mw{}", waves);
                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  waves={}",
                    "gate_proj",
                    b,
                    klabel,
                    ms_cand,
                    tflops_cand,
                    bdiff,
                    maxabs,
                    maxrel,
                    vs,
                    waves
                );
            }

            // LDS-staged X group (96 batch cols) reused across 8 row waves.
            // Exact M=19968 K=6656; batch%96==0. Y+=, caller zeros. Compared to
            // same fresh g11_batched baseline. Two warmups, three medians, same
            // iters. Skips Ok(false) and never calls on non-gfx1100.
            {
                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let used = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_lds_g256(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b,
                    )
                    .expect("muse_g11_lds probe");
                let _ = gpu.hip.device_synchronize();
                if used {
                    for _ in 0..2 {
                        let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                        let _ = gpu.gemm_hfq4g256_residual_muse_gfx1100_lds_g256(
                            &gate_w, &x_dim, &y_cand, m_gate, k_gate, b,
                        );
                    }
                    let _ = gpu.hip.device_synchronize();

                    let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                    let ok = gpu
                        .gemm_hfq4g256_residual_muse_gfx1100_lds_g256(
                            &gate_w, &x_dim, &y_cand, m_gate, k_gate, b,
                        )
                        .expect("muse_g11_lds dl");
                    let _ = gpu.hip.device_synchronize();
                    if ok {
                        let cand_host = gpu.download_f32(&y_cand).expect("dl muse_g11_lds");
                        let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &cand_host);

                        let mut cand_reps: Vec<f64> = Vec::new();
                        for _ in 0..3 {
                            let t0 = Instant::now();
                            for _ in 0..iters {
                                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                                gpu.gemm_hfq4g256_residual_muse_gfx1100_lds_g256(
                                    &gate_w, &x_dim, &y_cand, m_gate, k_gate, b,
                                )
                                .expect("muse_g11_lds bench");
                            }
                            let _ = gpu.hip.device_synchronize();
                            cand_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                        }
                        let ms_cand = median_ms(cand_reps);
                        let tflops_cand = flops / (ms_cand / 1000.0) / 1e12;
                        let vs = 100.0 * (ms_base / ms_cand - 1.0);
                        println!(
                            "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%",
                            "gate_proj",
                            b,
                            "muse_g11_lds",
                            ms_cand,
                            tflops_cand,
                            bdiff,
                            maxabs,
                            maxrel,
                            vs
                        );
                    }
                }
            }

            // Row-reuse RM×BV family: each wave owns RM row tiles × BV batch
            // tiles with RM*BV=12. Loads each B half16 once per batch tile and
            // reuses across RM WMMA calls. Exact M=19968 K=6656; full-tile only
            // (batch % (16*BV) == 0). Y+=, caller zeros. Compared to same fresh
            // g11_batched baseline. Two warmups, three medians, same iters.
            // Skips Ok(false) and never calls on non-gfx1100.
            const G11_RMS: [usize; 4] = [2, 3, 4, 6];
            for &rm in &G11_RMS {
                let bv = 12 / rm;
                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let used = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_rm(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                    )
                    .expect("muse_g11_rm probe");
                let _ = gpu.hip.device_synchronize();
                if !used {
                    continue;
                }

                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                    let _ = gpu.gemm_hfq4g256_residual_muse_gfx1100_rm(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                    );
                }
                let _ = gpu.hip.device_synchronize();

                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let ok = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_rm(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                    )
                    .expect("muse_g11_rm dl");
                let _ = gpu.hip.device_synchronize();
                if !ok {
                    continue;
                }
                let cand_host = gpu.download_f32(&y_cand).expect("dl muse_g11_rm");
                let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &cand_host);

                let mut cand_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                        gpu.gemm_hfq4g256_residual_muse_gfx1100_rm(
                            &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                        )
                        .expect("muse_g11_rm bench");
                    }
                    let _ = gpu.hip.device_synchronize();
                    cand_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_cand = median_ms(cand_reps);
                let tflops_cand = flops / (ms_cand / 1000.0) / 1e12;
                let vs = 100.0 * (ms_base / ms_cand - 1.0);
                let klabel = format!("muse_g11_rm{}x{}", rm, bv);
                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  rm={} bv={}",
                    "gate_proj",
                    b,
                    klabel,
                    ms_cand,
                    tflops_cand,
                    bdiff,
                    maxabs,
                    maxrel,
                    vs,
                    rm,
                    bv
                );
            }

            // Half-broadcast row-reuse: rm ∈ {2,4} only (bv 6/3). Lanes 0..15
            // load/dequant A; paired lanes receive exact f16 bit patterns via
            // wave32 shuffle/bpermute. Same shape/tiling guards and Y+= zero
            // contract as muse_g11_rm*. Compared to same fresh g11_batched
            // baseline. Two warmups, three medians, same iters.
            // Skips Ok(false) and never calls on non-gfx1100.
            const G11_RM_HBS: [usize; 2] = [2, 4];
            for &rm in &G11_RM_HBS {
                let bv = 12 / rm;
                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let used = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_rm_hb(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                    )
                    .expect("muse_g11_rm_hb probe");
                let _ = gpu.hip.device_synchronize();
                if !used {
                    continue;
                }

                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                    let _ = gpu.gemm_hfq4g256_residual_muse_gfx1100_rm_hb(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                    );
                }
                let _ = gpu.hip.device_synchronize();

                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let ok = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_rm_hb(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                    )
                    .expect("muse_g11_rm_hb dl");
                let _ = gpu.hip.device_synchronize();
                if !ok {
                    continue;
                }
                let cand_host = gpu.download_f32(&y_cand).expect("dl muse_g11_rm_hb");
                let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &cand_host);

                let mut cand_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                        gpu.gemm_hfq4g256_residual_muse_gfx1100_rm_hb(
                            &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                        )
                        .expect("muse_g11_rm_hb bench");
                    }
                    let _ = gpu.hip.device_synchronize();
                    cand_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_cand = median_ms(cand_reps);
                let tflops_cand = flops / (ms_cand / 1000.0) / 1e12;
                let vs = 100.0 * (ms_base / ms_cand - 1.0);
                let klabel = format!("muse_g11_rm{}x{}_hb", rm, bv);
                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  rm={} bv={}",
                    "gate_proj",
                    b,
                    klabel,
                    ms_cand,
                    tflops_cand,
                    bdiff,
                    maxabs,
                    maxrel,
                    vs,
                    rm,
                    bv
                );
            }

            // Packed-half2 dequant row-reuse: rm ∈ {1,2} only (bv 12/6).
            // A fragment uses eight __half2 values (16 nibbles) via __hfma2;
            // bit-oracle authoritative (fused FMA may diverge from scalar).
            // Same shape/tiling guards and Y+= zero contract as muse_g11_rm*.
            // Compared to same fresh g11_batched baseline. Two warmups, three
            // medians, same iters. Skips Ok(false) and never calls on non-gfx1100.
            const G11_RM_PKS: [usize; 2] = [1, 2];
            for &rm in &G11_RM_PKS {
                let bv = 12 / rm;
                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let used = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_rm_pk(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                    )
                    .expect("muse_g11_rm_pk probe");
                let _ = gpu.hip.device_synchronize();
                if !used {
                    continue;
                }

                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                    let _ = gpu.gemm_hfq4g256_residual_muse_gfx1100_rm_pk(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                    );
                }
                let _ = gpu.hip.device_synchronize();

                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let ok = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_rm_pk(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                    )
                    .expect("muse_g11_rm_pk dl");
                let _ = gpu.hip.device_synchronize();
                if !ok {
                    continue;
                }
                let cand_host = gpu.download_f32(&y_cand).expect("dl muse_g11_rm_pk");
                let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &cand_host);

                let mut cand_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                        gpu.gemm_hfq4g256_residual_muse_gfx1100_rm_pk(
                            &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, rm,
                        )
                        .expect("muse_g11_rm_pk bench");
                    }
                    let _ = gpu.hip.device_synchronize();
                    cand_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_cand = median_ms(cand_reps);
                let tflops_cand = flops / (ms_cand / 1000.0) / 1e12;
                let vs = 100.0 * (ms_base / ms_cand - 1.0);
                let klabel = format!("muse_g11_rm{}x{}_pk", rm, bv);
                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  rm={} bv={}",
                    "gate_proj",
                    b,
                    klabel,
                    ms_cand,
                    tflops_cand,
                    bdiff,
                    maxabs,
                    maxrel,
                    vs,
                    rm,
                    bv
                );
            }

            // RM2/BV6 two-slot X-fragment software pipeline (K2): scalar and
            // packed-half2. Host fail-closed unless gfx1100 + exact M/K/B192.
            // Gate shape is timed (zero/probe/2 warmup/fresh correctness/median3).
            // Additionally one fresh up_w correctness probe per symbol prints an
            // indented up bitdiff line; timing stays on gate. Never called outside
            // the is_gfx1100 block above.
            for (packed, klabel) in [
                (false, "rm2_pipe_scalar"),
                (true, "rm2_pipe_pk2"),
            ] {
                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let used = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_rm2_pipe(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, packed,
                    )
                    .expect("muse_g11_rm2_pipe probe");
                let _ = gpu.hip.device_synchronize();
                if !used {
                    continue;
                }

                for _ in 0..2 {
                    let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                    let _ = gpu.gemm_hfq4g256_residual_muse_gfx1100_rm2_pipe(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, packed,
                    );
                }
                let _ = gpu.hip.device_synchronize();

                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let ok = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_rm2_pipe(
                        &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, packed,
                    )
                    .expect("muse_g11_rm2_pipe dl");
                let _ = gpu.hip.device_synchronize();
                if !ok {
                    continue;
                }
                let cand_host = gpu.download_f32(&y_cand).expect("dl muse_g11_rm2_pipe");
                let (bdiff, maxabs, maxrel) = correctness_stats(&base_host, &cand_host);

                let mut cand_reps: Vec<f64> = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                        gpu.gemm_hfq4g256_residual_muse_gfx1100_rm2_pipe(
                            &gate_w, &x_dim, &y_cand, m_gate, k_gate, b, packed,
                        )
                        .expect("muse_g11_rm2_pipe bench");
                    }
                    let _ = gpu.hip.device_synchronize();
                    cand_reps.push(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
                }
                let ms_cand = median_ms(cand_reps);
                let tflops_cand = flops / (ms_cand / 1000.0) / 1e12;
                let vs = 100.0 * (ms_base / ms_cand - 1.0);
                println!(
                    "{:<12} {:>4} {:<14} {:>9.3} {:>9.2} {:>9} {:>9.2e} {:>9.2e} {:>+8.1}%  rm=2 bv=6",
                    "gate_proj",
                    b,
                    klabel,
                    ms_cand,
                    tflops_cand,
                    bdiff,
                    maxabs,
                    maxrel,
                    vs,
                );

                // Fresh up production base + one residual pipe correctness probe.
                // Timing remains gate-only; this is bit-exact evidence only.
                let _ = gpu.hip.memset(&y_base.buf, 0, b * m_gate * 4);
                gpu.gemm_hfq4g256_batched_lmhead(&up_w, &x_dim, &y_base, m_gate, k_gate, b)
                    .expect("up g11 pipe base dl");
                let _ = gpu.hip.device_synchronize();
                let up_base_host = gpu.download_f32(&y_base).expect("dl up g11 pipe base");

                let _ = gpu.hip.memset(&y_cand.buf, 0, b * m_gate * 4);
                let up_ok = gpu
                    .gemm_hfq4g256_residual_muse_gfx1100_rm2_pipe(
                        &up_w, &x_dim, &y_cand, m_gate, k_gate, b, packed,
                    )
                    .expect("muse_g11_rm2_pipe up probe");
                let _ = gpu.hip.device_synchronize();
                if up_ok {
                    let up_cand_host = gpu.download_f32(&y_cand).expect("dl muse_g11_rm2_pipe up");
                    let (ubdiff, umaxabs, umaxrel) =
                        correctness_stats(&up_base_host, &up_cand_host);
                    println!(
                        "  {:<10} {:>4} {:<14} {:>9} {:>9} {:>9} {:>9.2e} {:>9.2e} {:>9}",
                        "up", b, klabel, "-", "-", ubdiff, umaxabs, umaxrel, ""
                    );
                }
            }

            let _ = gpu.free_tensor(y_cand);
            let _ = gpu.free_tensor(y_base);
        }

        let _ = gpu.free_tensor(x_dim);
    }

    let _ = gpu.free_tensor(up_w);
    let _ = gpu.free_tensor(gate_w);

    println!("\nAttribution summary:");
    println!("  gate/up/down (ffn 19968) dominate 82.4% of layer FLOPs — they explain the gap.");
    println!("  o_proj (6656x4096) is next; attention q/k/v/gate are 17.6% combined.");
    println!("  candidate oracle: batched (production overwrite) vs residual+zero must be bitdiff=0");
    println!("  fused gate+up must be bitdiff=0 per output vs 2x batched; muse_bt12 must be bitdiff=0 vs batched");
    println!("  on gfx1100/gfx1151 muse rows report arch_skip and do not call the gfx12 kernel");
    println!("  on gfx1100 only: gate_proj muse_g11_bt{{4,6,8,12,16}}, muse_g11_cb{{4,6,12}}, muse_g11_mw{{2,4,8}}, muse_g11_lds, muse_g11_rm{{2,3,4,6}}x{{6,4,3,2}}, muse_g11_rm{{2,4}}x{{6,3}}_hb, muse_g11_rm{{1,2}}x{{12,6}}_pk, rm2_pipe_scalar/pk2 (B192) rows vs g11_batched (zeroed); skipped if Ok(false); pipe also prints indented up bitdiff");
    println!("  on non-gfx1100 the muse_gfx1100 APIs are never called");
}
