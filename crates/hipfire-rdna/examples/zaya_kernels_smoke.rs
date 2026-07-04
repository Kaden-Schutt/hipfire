// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// GPU smoke test for the ZAYA1 custom no-LDS kernels: validates grouped/valid
// causal conv1d, per-head L2 qk-norm+temp, the residual-scale affine, and exact
// GELU against tiny CPU references. Run: cargo run -p hipfire-rdna --example
// zaya_kernels_smoke

use hipfire_rdna::{DType, Gpu};

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");

    // ── 1. valid grouped causal conv1d ────────────────────────────────────────
    // channels=4, groups=2 (in_per_group=2), kernel=2, in_len=5 -> out_len=4.
    let (channels, groups, kernel, in_len) = (4usize, 2usize, 2usize, 5usize);
    let out_len = in_len - kernel + 1;
    let per = channels / groups;
    let input: Vec<f32> = (0..channels * in_len)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..channels * per * kernel)
        .map(|i| 0.05 * (i as f32) + 0.2)
        .collect();
    let bias: Vec<f32> = (0..channels).map(|c| 0.01 * c as f32).collect();
    // CPU ref
    let mut cpu = vec![0f32; channels * out_len];
    for c in 0..channels {
        let base = (c / per) * per;
        for t in 0..out_len {
            let mut acc = bias[c];
            for j in 0..per {
                for k in 0..kernel {
                    acc += weight[(c * per + j) * kernel + k] * input[(base + j) * in_len + t + k];
                }
            }
            cpu[c * out_len + t] = acc;
        }
    }
    let g_in = gpu.upload_f32(&input, &[channels * in_len]).unwrap();
    let g_w = gpu.upload_f32(&weight, &[weight.len()]).unwrap();
    let g_b = gpu.upload_f32(&bias, &[channels]).unwrap();
    let g_out = gpu.zeros(&[channels * out_len], DType::F32).unwrap();
    gpu.zaya_conv1d_valid_f32(
        &g_out, &g_in, &g_w, &g_b, channels, groups, kernel, in_len, out_len,
    )
    .unwrap();
    let got = gpu.download_f32(&g_out).unwrap();
    println!("conv1d_valid : maxdiff={:.3e}", maxdiff(&cpu, &got));

    // ── 2. L2 qk-norm + per-head temp ─────────────────────────────────────────
    let (s, heads, hd) = (3usize, 2usize, 4usize);
    let scale = (hd as f32).sqrt();
    let xin: Vec<f32> = (0..s * heads * hd).map(|i| 0.3 * i as f32 - 2.0).collect();
    let temp = vec![1.5f32, 0.5f32];
    let mut cpu2 = xin.clone();
    for t in 0..s {
        for h in 0..heads {
            let row = &mut cpu2[(t * heads + h) * hd..(t * heads + h + 1) * hd];
            let nrm: f32 = row
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
                .max(f32::EPSILON);
            let inv = scale / nrm;
            for v in row.iter_mut() {
                *v = *v * inv * temp[h];
            }
        }
    }
    let g_x = gpu.upload_f32(&xin, &[xin.len()]).unwrap();
    let g_t = gpu.upload_f32(&temp, &[heads]).unwrap();
    gpu.zaya_qk_l2norm_temp_f32(&g_x, Some(&g_t), s, heads, hd, scale, f32::EPSILON)
        .unwrap();
    let got2 = gpu.download_f32(&g_x).unwrap();
    println!("qk_l2norm    : maxdiff={:.3e}", maxdiff(&cpu2, &got2));

    // ── 3. residual-scale affine ──────────────────────────────────────────────
    let d = 4usize;
    let n = 3 * d;
    let hh: Vec<f32> = (0..n).map(|i| 0.2 * i as f32).collect();
    let rr: Vec<f32> = (0..n).map(|i| -0.1 * i as f32 + 1.0).collect();
    let hs: Vec<f32> = (0..d).map(|c| 1.0 + 0.1 * c as f32).collect();
    let hb: Vec<f32> = (0..d).map(|c| 0.05 * c as f32).collect();
    let rs: Vec<f32> = (0..d).map(|c| 0.9 - 0.05 * c as f32).collect();
    let rb: Vec<f32> = (0..d).map(|c| -0.02 * c as f32).collect();
    let mut cpu3 = vec![0f32; n];
    for i in 0..n {
        let c = i % d;
        cpu3[i] = (hh[i] + hb[c]) * hs[c] + (rr[i] + rb[c]) * rs[c];
    }
    let (gh, gr) = (
        gpu.upload_f32(&hh, &[n]).unwrap(),
        gpu.upload_f32(&rr, &[n]).unwrap(),
    );
    let (ghs, ghb) = (
        gpu.upload_f32(&hs, &[d]).unwrap(),
        gpu.upload_f32(&hb, &[d]).unwrap(),
    );
    let (grs, grb) = (
        gpu.upload_f32(&rs, &[d]).unwrap(),
        gpu.upload_f32(&rb, &[d]).unwrap(),
    );
    let go = gpu.zeros(&[n], DType::F32).unwrap();
    gpu.zaya_affine_residual_f32(&go, &gh, &gr, &ghs, &ghb, &grs, &grb, d, n)
        .unwrap();
    let got3 = gpu.download_f32(&go).unwrap();
    println!("affine_resid : maxdiff={:.3e}", maxdiff(&cpu3, &got3));

    // ── 4. exact GELU ─────────────────────────────────────────────────────────
    let gx: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
    let erf = |x: f32| {
        let s = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t)
                * (-x * x).exp();
        s * y
    };
    let cpu4: Vec<f32> = gx
        .iter()
        .map(|&v| 0.5 * v * (1.0 + erf(v * std::f32::consts::FRAC_1_SQRT_2)))
        .collect();
    let g4 = gpu.upload_f32(&gx, &[gx.len()]).unwrap();
    gpu.zaya_gelu_exact_f32(&g4, gx.len()).unwrap();
    let got4 = gpu.download_f32(&g4).unwrap();
    println!("gelu_exact   : maxdiff={:.3e}", maxdiff(&cpu4, &got4));

    let mut diffs = vec![
        ("conv1d_valid", maxdiff(&cpu, &got)),
        ("qk_l2norm", maxdiff(&cpu2, &got2)),
        ("affine_resid", maxdiff(&cpu3, &got3)),
        ("gelu_exact", maxdiff(&cpu4, &got4)),
    ];

    // ── 5. embed gather ───────────────────────────────────────────────────────
    {
        let (vocab, hidden, sq) = (5usize, 6usize, 3usize);
        let embed: Vec<f32> = (0..vocab * hidden).map(|i| 0.1 * i as f32).collect();
        let ids = [2i32, 0, 4];
        let mut cpu = vec![0f32; sq * hidden];
        for t in 0..sq {
            for c in 0..hidden {
                cpu[t * hidden + c] = embed[ids[t] as usize * hidden + c];
            }
        }
        let g_e = gpu.upload_f32(&embed, &[vocab, hidden]).unwrap();
        let id_bytes: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
        let g_ids = gpu.upload_raw(&id_bytes, &[sq]).unwrap();
        let out = gpu.zeros(&[sq * hidden], DType::F32).unwrap();
        gpu.zaya_embed_gather_f32(&out, &g_e, &g_ids, hidden, sq * hidden)
            .unwrap();
        diffs.push((
            "embed_gather",
            maxdiff(&cpu, &gpu.download_f32(&out).unwrap()),
        ));
    }

    // ── 6. value compose + qk_residual + add_conv_residual + rope + gqa_attn ───
    {
        let (sq, nq, nkv, hd) = (3usize, 4usize, 2usize, 4usize);
        let groups = nq / nkv;
        // value compose
        let vcur: Vec<f32> = (0..sq * hd).map(|i| 0.2 * i as f32 - 1.0).collect();
        let vdel: Vec<f32> = (0..sq * hd).map(|i| -0.1 * i as f32 + 0.5).collect();
        let mut cval = vec![0f32; sq * nkv * hd];
        for t in 0..sq {
            for d in 0..hd {
                cval[(t * nkv) * hd + d] = vcur[t * hd + d];
                cval[(t * nkv + 1) * hd + d] = if t == 0 { 0.0 } else { vdel[(t - 1) * hd + d] };
            }
        }
        let gvc = gpu.upload_f32(&vcur, &[vcur.len()]).unwrap();
        let gvd = gpu.upload_f32(&vdel, &[vdel.len()]).unwrap();
        let gval = gpu.zeros(&[sq * nkv * hd], DType::F32).unwrap();
        gpu.zaya_value_compose_f32(&gval, &gvc, &gvd, sq, nkv, hd)
            .unwrap();
        diffs.push((
            "value_compose",
            maxdiff(&cval, &gpu.download_f32(&gval).unwrap()),
        ));

        // qk_residual
        let q: Vec<f32> = (0..sq * nq * hd).map(|i| 0.1 * i as f32).collect();
        let k: Vec<f32> = (0..sq * nkv * hd).map(|i| 0.05 * i as f32 - 0.3).collect();
        let mut qres = vec![0f32; sq * nq * hd];
        let mut kres = vec![0f32; sq * nkv * hd];
        for t in 0..sq {
            for head in 0..nq {
                let kh = head / groups;
                for d in 0..hd {
                    qres[(t * nq + head) * hd + d] =
                        (q[t * nq * hd + head * hd + d] + k[t * nkv * hd + kh * hd + d]) * 0.5;
                }
            }
        }
        for t in 0..sq {
            for kh in 0..nkv {
                for d in 0..hd {
                    let mut acc = 0f32;
                    for g in 0..groups {
                        acc += qres[(t * nq + kh * groups + g) * hd + d];
                    }
                    kres[(t * nkv + kh) * hd + d] = acc / groups as f32;
                }
            }
        }
        let gq = gpu.upload_f32(&q, &[q.len()]).unwrap();
        let gk = gpu.upload_f32(&k, &[k.len()]).unwrap();
        let gqr = gpu.zeros(&[sq * nq * hd], DType::F32).unwrap();
        let gkr = gpu.zeros(&[sq * nkv * hd], DType::F32).unwrap();
        gpu.zaya_qk_residual_f32(&gqr, &gkr, &gq, &gk, sq, nq, nkv, hd, 0)
            .unwrap();
        gpu.zaya_qk_residual_f32(&gqr, &gkr, &gq, &gk, sq, nq, nkv, hd, 1)
            .unwrap();
        diffs.push((
            "qk_residual_q",
            maxdiff(&qres, &gpu.download_f32(&gqr).unwrap()),
        ));
        diffs.push((
            "qk_residual_k",
            maxdiff(&kres, &gpu.download_f32(&gkr).unwrap()),
        ));

        // gqa_attn (no rope/norm for the math check; just causal softmax)
        let scaling = 1.0 / (hd as f32).sqrt();
        let qq: Vec<f32> = (0..sq * nq * hd).map(|i| 0.07 * i as f32 - 0.5).collect();
        let kk: Vec<f32> = (0..sq * nkv * hd).map(|i| 0.03 * i as f32).collect();
        let vv: Vec<f32> = (0..sq * nkv * hd).map(|i| 0.02 * i as f32 - 0.1).collect();
        let mut catt = vec![0f32; sq * nq * hd];
        for head in 0..nq {
            let kh = head / groups;
            for i in 0..sq {
                let mut sc = vec![0f32; i + 1];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i {
                    let mut dot = 0f32;
                    for d in 0..hd {
                        dot += qq[(i * nq + head) * hd + d] * kk[(j * nkv + kh) * hd + d];
                    }
                    sc[j] = dot * scaling;
                    mx = mx.max(sc[j]);
                }
                let mut den = 0f32;
                for x in sc.iter_mut() {
                    *x = (*x - mx).exp();
                    den += *x;
                }
                for j in 0..=i {
                    let p = sc[j] / den;
                    for d in 0..hd {
                        catt[(i * nq + head) * hd + d] += p * vv[(j * nkv + kh) * hd + d];
                    }
                }
            }
        }
        let gqq = gpu.upload_f32(&qq, &[qq.len()]).unwrap();
        let gkk = gpu.upload_f32(&kk, &[kk.len()]).unwrap();
        let gvv = gpu.upload_f32(&vv, &[vv.len()]).unwrap();
        let gca = gpu.zeros(&[sq * nq * hd], DType::F32).unwrap();
        gpu.zaya_gqa_attn_f32(&gca, &gqq, &gkk, &gvv, sq, nq, nkv, hd, scaling)
            .unwrap();
        diffs.push(("gqa_attn", maxdiff(&catt, &gpu.download_f32(&gca).unwrap())));
    }

    println!();
    let mut worst = 0f32;
    for (name, d) in &diffs {
        println!("{name:<16} maxdiff={d:.3e}");
        worst = worst.max(*d);
    }
    println!(
        "\nworst maxdiff = {worst:.3e}  {}",
        if worst < 1e-4 { "PASS" } else { "FAIL" }
    );
}
