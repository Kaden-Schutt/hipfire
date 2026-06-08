// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Per-kernel gradient checks for the DFlash trainer backward kernels.
// Each backward is checked against a central finite difference of a linear
// probe loss L = Σ a·out (f64-summed on host so the fd signal isn't swamped by
// f32 reduction roundoff). Isolating each kernel localizes any bug precisely.

use rdna_compute::{DType, Gpu, GpuTensor};

fn frand(seed: usize) -> f32 {
    ((seed.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0
}
fn loss(a: &[f32], out: &[f32]) -> f64 {
    a.iter().zip(out).map(|(x, y)| *x as f64 * *y as f64).sum()
}
fn samples(n: usize) -> Vec<usize> {
    let cnt = 16.min(n);
    (0..cnt).map(|i| (i * 2654435761) % n).collect()
}
fn report(name: &str, pairs: &[(f64, f64)]) -> f64 {
    let mut worst = 0f64;
    for (fd, an) in pairs {
        let den = fd.abs().max(*an).max(1e-4);
        worst = worst.max((fd - an).abs() / den);
    }
    println!("  {name} (fd): max rel err = {worst:.3e}  (n={})", pairs.len());
    worst
}
// Decisive check: GPU grad vs independent f64 host-analytic grad (no fd noise).
fn report_host(name: &str, gpu: &[f32], host: &[f64]) -> f64 {
    let mut worst = 0f64;
    for (g, h) in gpu.iter().zip(host) {
        let den = (*g as f64).abs().max(h.abs()).max(1e-4);
        worst = worst.max((*g as f64 - h).abs() / den);
    }
    println!("  {name} (host-f64): max rel err = {worst:.3e}", );
    worst
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let eps = 1e-3f64;
    let mut worst = 0f64;

    // ---------------- linear (dX, dW) ----------------
    {
        let (m, k, n) = (8usize, 64usize, 48usize);
        let x: Vec<f32> = (0..m * k).map(|i| frand(i + 1)).collect();
        let w: Vec<f32> = (0..n * k).map(|i| frand(i + 5000)).collect();
        let a: Vec<f32> = (0..m * n).map(|i| frand(i + 9000)).collect();
        let wg = gpu.upload_f32(&w, &[n, k]).unwrap();
        let xg = gpu.upload_f32(&x, &[m, k]).unwrap();
        let fwd = |g: &mut Gpu, xv: &[f32], wv: &[f32]| -> Vec<f32> {
            let xt = g.upload_f32(xv, &[m, k]).unwrap();
            let wt = g.upload_f32(wv, &[n, k]).unwrap();
            let yt = g.zeros(&[m, n], DType::F32).unwrap();
            g.linear_fwd_f32(&xt, &wt, &yt, m, k, n).unwrap();
            g.download_f32(&yt).unwrap()
        };
        let dyg = gpu.upload_f32(&a, &[m, n]).unwrap();
        let dxg = gpu.zeros(&[m, k], DType::F32).unwrap();
        let dwg = gpu.zeros(&[n, k], DType::F32).unwrap();
        gpu.linear_bwd_dx_f32(&dyg, &wg, &dxg, m, k, n).unwrap();
        gpu.linear_bwd_dw_f32(&dyg, &xg, &dwg, m, k, n).unwrap();
        let dx = gpu.download_f32(&dxg).unwrap();
        let dw = gpu.download_f32(&dwg).unwrap();
        let mut pdx = vec![];
        for t in samples(m * k) {
            let mut xp = x.clone(); xp[t] += eps as f32;
            let mut xm = x.clone(); xm[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &xp, &w)) - loss(&a, &fwd(&mut gpu, &xm, &w))) / (2.0 * eps);
            pdx.push((fd, dx[t] as f64));
        }
        let mut pdw = vec![];
        for t in samples(n * k) {
            let mut wp = w.clone(); wp[t] += eps as f32;
            let mut wm = w.clone(); wm[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &x, &wp)) - loss(&a, &fwd(&mut gpu, &x, &wm))) / (2.0 * eps);
            pdw.push((fd, dw[t] as f64));
        }
        report("linear dX", &pdx);
        report("linear dW", &pdw);
        worst = worst.max(report_host("linear dX", &dx, &{
            let mut h = vec![0f64; m * k];
            for mm in 0..m { for kk in 0..k {
                let mut s = 0f64; for nn in 0..n { s += a[mm * n + nn] as f64 * w[nn * k + kk] as f64; }
                h[mm * k + kk] = s;
            }} h
        }));
        worst = worst.max(report_host("linear dW", &dw, &{
            let mut h = vec![0f64; n * k];
            for nn in 0..n { for kk in 0..k {
                let mut s = 0f64; for mm in 0..m { s += a[mm * n + nn] as f64 * x[mm * k + kk] as f64; }
                h[nn * k + kk] = s;
            }} h
        }));
    }

    // ---------------- rmsnorm (dX, dG) ----------------
    {
        let (rows, n) = (6usize, 64usize);
        let x: Vec<f32> = (0..rows * n).map(|i| frand(i + 11)).collect();
        let g: Vec<f32> = (0..n).map(|i| 1.0 + 0.1 * frand(i + 77)).collect();
        let a: Vec<f32> = (0..rows * n).map(|i| frand(i + 33)).collect();
        let epsn = 1e-5f32;
        let fwd = |gp: &mut Gpu, xv: &[f32], gv: &[f32]| -> Vec<f32> {
            let xt = gp.upload_f32(xv, &[rows, n]).unwrap();
            let gt = gp.upload_f32(gv, &[n]).unwrap();
            let yt = gp.zeros(&[rows, n], DType::F32).unwrap();
            gp.rmsnorm_batched(&xt, &gt, &yt, rows, n, epsn).unwrap();
            gp.download_f32(&yt).unwrap()
        };
        let xg = gpu.upload_f32(&x, &[rows, n]).unwrap();
        let gg = gpu.upload_f32(&g, &[n]).unwrap();
        let dyg = gpu.upload_f32(&a, &[rows, n]).unwrap();
        let dxg = gpu.zeros(&[rows, n], DType::F32).unwrap();
        let dgg = gpu.zeros(&[n], DType::F32).unwrap();
        gpu.rmsnorm_bwd_f32(&xg, &gg, &dyg, &dxg, &dgg, rows, n, epsn).unwrap();
        let dx = gpu.download_f32(&dxg).unwrap();
        let dg = gpu.download_f32(&dgg).unwrap();
        let mut pdx = vec![];
        for t in samples(rows * n) {
            let mut xp = x.clone(); xp[t] += eps as f32;
            let mut xm = x.clone(); xm[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &xp, &g)) - loss(&a, &fwd(&mut gpu, &xm, &g))) / (2.0 * eps);
            pdx.push((fd, dx[t] as f64));
        }
        let mut pdg = vec![];
        for t in samples(n) {
            let mut gp = g.clone(); gp[t] += eps as f32;
            let mut gm = g.clone(); gm[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &x, &gp)) - loss(&a, &fwd(&mut gpu, &x, &gm))) / (2.0 * eps);
            pdg.push((fd, dg[t] as f64));
        }
        worst = worst.max(report("rmsnorm dX", &pdx));
        worst = worst.max(report("rmsnorm dG", &pdg));
    }

    // ---------------- silu_mul (dG, dU) ----------------
    {
        let n = 128usize;
        let g: Vec<f32> = (0..n).map(|i| frand(i + 5)).collect();
        let u: Vec<f32> = (0..n).map(|i| frand(i + 6000)).collect();
        let a: Vec<f32> = (0..n).map(|i| frand(i + 700)).collect();
        let fwd = |gp: &mut Gpu, gv: &[f32], uv: &[f32]| -> Vec<f32> {
            let gt = gp.upload_f32(gv, &[n]).unwrap();
            let ut = gp.upload_f32(uv, &[n]).unwrap();
            let ot = gp.zeros(&[n], DType::F32).unwrap();
            gp.silu_mul_f32(&gt, &ut, &ot).unwrap();
            gp.download_f32(&ot).unwrap()
        };
        let gg = gpu.upload_f32(&g, &[n]).unwrap();
        let ug = gpu.upload_f32(&u, &[n]).unwrap();
        let dag = gpu.upload_f32(&a, &[n]).unwrap();
        let dgg = gpu.zeros(&[n], DType::F32).unwrap();
        let dug = gpu.zeros(&[n], DType::F32).unwrap();
        gpu.silu_mul_bwd_f32(&gg, &ug, &dag, &dgg, &dug, n).unwrap();
        let dg = gpu.download_f32(&dgg).unwrap();
        let du = gpu.download_f32(&dug).unwrap();
        let mut pdg = vec![];
        for t in samples(n) {
            let mut gp = g.clone(); gp[t] += eps as f32;
            let mut gm = g.clone(); gm[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &gp, &u)) - loss(&a, &fwd(&mut gpu, &gm, &u))) / (2.0 * eps);
            pdg.push((fd, dg[t] as f64));
        }
        let mut pdu = vec![];
        for t in samples(n) {
            let mut up = u.clone(); up[t] += eps as f32;
            let mut um = u.clone(); um[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &g, &up)) - loss(&a, &fwd(&mut gpu, &g, &um))) / (2.0 * eps);
            pdu.push((fd, du[t] as f64));
        }
        worst = worst.max(report("silu_mul dG", &pdg));
        worst = worst.max(report("silu_mul dU", &pdu));
    }

    // ---------------- rope (dX via R^T) ----------------
    {
        let (rows, nh, hd) = (5usize, 4usize, 64usize);
        let n = rows * nh * hd;
        let pos: Vec<i32> = (0..rows).map(|i| (i + 3) as i32).collect();
        let theta = 1.0e6f32;
        let x: Vec<f32> = (0..n).map(|i| frand(i + 9)).collect();
        let a: Vec<f32> = (0..n).map(|i| frand(i + 4242)).collect();
        let pos_t = {
            let t = gpu.alloc_tensor(&[rows], DType::F32).unwrap();
            let b: Vec<u8> = pos.iter().flat_map(|p| p.to_le_bytes()).collect();
            gpu.hip.memcpy_htod(&t.buf, &b).unwrap(); t
        };
        // forward: rope a fresh copy (rope_batched is in place, q only via nh,0)
        let fwd = |gp: &mut Gpu, xv: &[f32]| -> Vec<f32> {
            let xt = gp.upload_f32(xv, &[rows, nh * hd]).unwrap();
            gp.rope_batched_f32(&xt, &xt, &pos_t, nh, 0, hd, theta, rows).unwrap();
            gp.download_f32(&xt).unwrap()
        };
        // analytic dX = R^T a
        let dxg = gpu.upload_f32(&a, &[rows, nh * hd]).unwrap();
        gpu.rope_rows_bwd_f32(&dxg, &pos_t, nh, hd, theta, rows).unwrap();
        let dx = gpu.download_f32(&dxg).unwrap();
        let mut pdx = vec![];
        for t in samples(n) {
            let mut xp = x.clone(); xp[t] += eps as f32;
            let mut xm = x.clone(); xm[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &xp)) - loss(&a, &fwd(&mut gpu, &xm))) / (2.0 * eps);
            pdx.push((fd, dx[t] as f64));
        }
        worst = worst.max(report("rope dX", &pdx));
    }

    // ---------------- attn (dQ, dK, dV) ----------------
    {
        let (b, n_ctx, nh, nkv, hd) = (4usize, 6usize, 4usize, 2usize, 64usize);
        let l = n_ctx + b;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..b * nh * hd).map(|i| frand(i + 2)).collect();
        let k: Vec<f32> = (0..l * nkv * hd).map(|i| frand(i + 40_000)).collect();
        let v: Vec<f32> = (0..l * nkv * hd).map(|i| frand(i + 80_000)).collect();
        let a: Vec<f32> = (0..b * nh * hd).map(|i| frand(i + 12_345)).collect();
        let fwd = |gp: &mut Gpu, qv: &[f32], kv: &[f32], vv: &[f32]| -> Vec<f32> {
            let qt = gp.upload_f32(qv, &[b, nh, hd]).unwrap();
            let kt = gp.upload_f32(kv, &[l, nkv, hd]).unwrap();
            let vt = gp.upload_f32(vv, &[l, nkv, hd]).unwrap();
            let ot = gp.zeros(&[b, nh, hd], DType::F32).unwrap();
            gp.attn_block_ctx_f32(&qt, &kt, &vt, &ot, b, l, nh, nkv, hd, scale).unwrap();
            gp.download_f32(&ot).unwrap()
        };
        let qg = gpu.upload_f32(&q, &[b, nh, hd]).unwrap();
        let kg = gpu.upload_f32(&k, &[l, nkv, hd]).unwrap();
        let vg = gpu.upload_f32(&v, &[l, nkv, hd]).unwrap();
        let dog = gpu.upload_f32(&a, &[b, nh, hd]).unwrap();
        let dqg = gpu.zeros(&[b, nh, hd], DType::F32).unwrap();
        let dkg = gpu.zeros(&[l, nkv, hd], DType::F32).unwrap();
        let dvg = gpu.zeros(&[l, nkv, hd], DType::F32).unwrap();
        gpu.attn_block_ctx_bwd_f32(&qg, &kg, &vg, &dog, &dqg, &dkg, &dvg, b, l, nh, nkv, hd, scale).unwrap();
        let dq = gpu.download_f32(&dqg).unwrap();
        let dk = gpu.download_f32(&dkg).unwrap();
        let dv = gpu.download_f32(&dvg).unwrap();
        let mut pq = vec![];
        for t in samples(b * nh * hd) {
            let mut p = q.clone(); p[t] += eps as f32;
            let mut mm = q.clone(); mm[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &p, &k, &v)) - loss(&a, &fwd(&mut gpu, &mm, &k, &v))) / (2.0 * eps);
            pq.push((fd, dq[t] as f64));
        }
        let mut pk = vec![];
        for t in samples(l * nkv * hd) {
            let mut p = k.clone(); p[t] += eps as f32;
            let mut mm = k.clone(); mm[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &q, &p, &v)) - loss(&a, &fwd(&mut gpu, &q, &mm, &v))) / (2.0 * eps);
            pk.push((fd, dk[t] as f64));
        }
        let mut pv = vec![];
        for t in samples(l * nkv * hd) {
            let mut p = v.clone(); p[t] += eps as f32;
            let mut mm = v.clone(); mm[t] -= eps as f32;
            let fd = (loss(&a, &fwd(&mut gpu, &q, &k, &p)) - loss(&a, &fwd(&mut gpu, &q, &k, &mm))) / (2.0 * eps);
            pv.push((fd, dv[t] as f64));
        }
        report("attn dQ", &pq);
        report("attn dK", &pk);
        report("attn dV", &pv);
        // decisive: independent f64 host-analytic backward (no fd noise)
        let group = nh / nkv;
        let mut hq = vec![0f64; b * nh * hd];
        let mut hk = vec![0f64; l * nkv * hd];
        let mut hv = vec![0f64; l * nkv * hd];
        for i in 0..b {
            for h in 0..nh {
                let g = h / group;
                let mut s = vec![0f64; l];
                let mut mx = f64::NEG_INFINITY;
                for j in 0..l {
                    let mut d = 0f64;
                    for dd in 0..hd { d += q[(i * nh + h) * hd + dd] as f64 * k[(j * nkv + g) * hd + dd] as f64; }
                    s[j] = d * scale as f64;
                    if s[j] > mx { mx = s[j]; }
                }
                let mut z = 0f64;
                for j in 0..l { s[j] = (s[j] - mx).exp(); z += s[j]; }
                for j in 0..l { s[j] /= z; } // p_j
                // dp_j and D
                let mut dp = vec![0f64; l];
                let mut dd = 0f64;
                for j in 0..l {
                    let mut t = 0f64;
                    for d in 0..hd { t += a[(i * nh + h) * hd + d] as f64 * v[(j * nkv + g) * hd + d] as f64; }
                    dp[j] = t;
                    dd += s[j] * t;
                    for d in 0..hd { hv[(j * nkv + g) * hd + d] += s[j] * a[(i * nh + h) * hd + d] as f64; }
                }
                for j in 0..l {
                    let ds = s[j] * (dp[j] - dd);
                    let dsc = ds * scale as f64;
                    for d in 0..hd {
                        hq[(i * nh + h) * hd + d] += dsc * k[(j * nkv + g) * hd + d] as f64;
                        hk[(j * nkv + g) * hd + d] += dsc * q[(i * nh + h) * hd + d] as f64;
                    }
                }
            }
        }
        worst = worst.max(report_host("attn dQ", &dq, &hq));
        worst = worst.max(report_host("attn dK", &dk, &hk));
        worst = worst.max(report_host("attn dV", &dv, &hv));
    }

    if worst < 1e-2 {
        println!("dflash_train BACKWARD kernels: PASS (worst rel err {worst:.3e})");
    } else {
        println!("dflash_train BACKWARD kernels: FAIL (worst rel err {worst:.3e})");
        std::process::exit(1);
    }
}
