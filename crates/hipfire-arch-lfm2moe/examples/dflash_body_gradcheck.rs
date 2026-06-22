// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash spike — step 4.2: full reverse-mode backward of the block-parallel
// LFM2 draft body (conv + GQA-with-KV-injection + dense SwiGLU), assembled from
// the validated forward/backward kernels, and gradient-checked end-to-end
// against central finite differences on a tiny toy config (the gate before any
// training run). Validates grads w.r.t. body input, the injected context
// feature, and a sampling of every weight class.
//
// Parameterized config (the real LFM2.5-350M is the same code with prod dims).

use rdna_compute::{DType, Gpu, GpuTensor};

#[derive(Clone)]
struct Cfg {
    d: usize,
    is_attn: Vec<bool>,
    nh: usize,
    nkv: usize,
    hd: usize,
    conv_k: usize,
    inter: usize,
    theta: f32,
    eps: f32,
}
impl Cfg {
    fn n_layers(&self) -> usize { self.is_attn.len() }
    fn qd(&self) -> usize { self.nh * self.hd }
    fn kvd(&self) -> usize { self.nkv * self.hd }
}

struct LW {
    op_norm: GpuTensor,
    ffn_norm: GpuTensor,
    in_proj: Option<GpuTensor>,
    conv_w: Option<GpuTensor>,
    out_proj: Option<GpuTensor>,
    wq: Option<GpuTensor>,
    wk: Option<GpuTensor>,
    wv: Option<GpuTensor>,
    wo: Option<GpuTensor>,
    q_norm: Option<GpuTensor>,
    k_norm: Option<GpuTensor>,
    w1: GpuTensor,
    w3: GpuTensor,
    w2: GpuTensor,
}

#[derive(Default)]
struct LT {
    h_in: Option<GpuTensor>,
    xn: Option<GpuTensor>,
    h_mid: Option<GpuTensor>,
    fnorm: Option<GpuTensor>,
    g: Option<GpuTensor>,
    u: Option<GpuTensor>,
    act: Option<GpuTensor>,
    bcx: Option<GpuTensor>,
    cy: Option<GpuTensor>,
    q0: Option<GpuTensor>,
    kfull0: Option<GpuTensor>,
    qr: Option<GpuTensor>,
    kr: Option<GpuTensor>,
    vfull: Option<GpuTensor>,
    attn_out: Option<GpuTensor>,
}

fn frand(seed: usize) -> f32 {
    ((seed.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0
}
fn up(gpu: &mut Gpu, v: &[f32], shape: &[usize]) -> GpuTensor { gpu.upload_f32(v, shape).unwrap() }
fn zeros(gpu: &mut Gpu, n: usize) -> GpuTensor { gpu.zeros(&[n], DType::F32).unwrap() }
fn lin(gpu: &mut Gpu, x: &GpuTensor, w: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let y = gpu.zeros(&[m, n], DType::F32).unwrap();
    gpu.linear_fwd_f32(x, w, &y, m, k, n).unwrap();
    y
}
fn lin_dx(gpu: &mut Gpu, dy: &GpuTensor, w: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let dx = gpu.zeros(&[m, k], DType::F32).unwrap();
    gpu.linear_bwd_dx_f32(dy, w, &dx, m, k, n).unwrap();
    dx
}
fn lin_dw(gpu: &mut Gpu, dy: &GpuTensor, x: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let dw = gpu.zeros(&[n, k], DType::F32).unwrap();
    gpu.linear_bwd_dw_f32(dy, x, &dw, m, k, n).unwrap();
    dw
}
fn addv(gpu: &mut Gpu, a: &GpuTensor, b: &GpuTensor) { gpu.add_inplace_f32(a, b).unwrap(); }
fn add_new(gpu: &mut Gpu, a: &GpuTensor, b: &GpuTensor, n: usize) -> GpuTensor {
    let c = gpu.zeros(&[n], DType::F32).unwrap();
    gpu.add_f32(a, b, &c).unwrap();
    c
}
fn dup(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> GpuTensor {
    let c = gpu.zeros(&[n], DType::F32).unwrap();
    gpu.add_inplace_f32(&c, t).unwrap();
    c
}
fn upload_pos(gpu: &mut Gpu, pos: &[i32]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[pos.len()], DType::F32).unwrap();
    let bytes: Vec<u8> = pos.iter().flat_map(|p| p.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).unwrap();
    t
}

// ---- f64 host forward (decisive fd oracle: removes the f32 GPU-forward noise
//      that swamps fd on small-magnitude gradient components) ----
fn matt(x: &[f64], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    // y[i,o] = Σ_k x[i,k]·w[o,k]   (w is [n,k] row-major, PyTorch Linear layout)
    let mut y = vec![0f64; m * n];
    for i in 0..m { for o in 0..n {
        let mut s = 0f64;
        for kk in 0..k { s += x[i * k + kk] * w[o * k + kk] as f64; }
        y[i * n + o] = s;
    }}
    y
}
fn rmsnorm_rows(x: &[f64], g: &[f32], rows: usize, n: usize, eps: f64) -> Vec<f64> {
    let mut y = vec![0f64; rows * n];
    for r in 0..rows {
        let mut q = 0f64;
        for i in 0..n { let v = x[r * n + i]; q += v * v; }
        let inv = 1.0 / (q / n as f64 + eps).sqrt();
        for i in 0..n { y[r * n + i] = g[i] as f64 * x[r * n + i] * inv; }
    }
    y
}
fn qknorm(x: &mut [f64], g: &[f32], rows: usize, heads: usize, hd: usize, eps: f64) {
    // per (row, head) rmsnorm over hd
    for rh in 0..rows * heads {
        let base = rh * hd;
        let mut q = 0f64;
        for d in 0..hd { let v = x[base + d]; q += v * v; }
        let inv = 1.0 / (q / hd as f64 + eps).sqrt();
        for d in 0..hd { x[base + d] = g[d] as f64 * x[base + d] * inv; }
    }
}
fn rope_rows(x: &mut [f64], pos: &[i32], rows: usize, heads: usize, hd: usize, theta: f64) {
    let half = hd / 2;
    for r in 0..rows {
        let p = pos[r] as f64;
        for h in 0..heads {
            let base = (r * heads + h) * hd;
            for i in 0..half {
                let freq = 1.0 / theta.powf((2 * i) as f64 / hd as f64);
                let (c, s) = (( p * freq).cos(), (p * freq).sin());
                let (a0, a1) = (x[base + i], x[base + i + half]);
                x[base + i] = a0 * c - a1 * s;
                x[base + i + half] = a0 * s + a1 * c;
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn host_forward(cfg: &Cfg, hw: &[HW], body_in: &[f32], ctx: &[f32], blk: &[i32], full: &[i32], b: usize, n_ctx: usize) -> Vec<f64> {
    let (d, nh, nkv, hd, qd, kvd) = (cfg.d, cfg.nh, cfg.nkv, cfg.hd, cfg.qd(), cfg.kvd());
    let k = cfg.conv_k; let l = n_ctx + b; let eps = cfg.eps as f64; let theta = cfg.theta as f64;
    let scale = 1.0 / (hd as f64).sqrt();
    let group = nh / nkv;
    let ctxf: Vec<f64> = ctx.iter().map(|&v| v as f64).collect();
    let mut h: Vec<f64> = body_in.iter().map(|&v| v as f64).collect();
    for (li, hwl) in hw.iter().enumerate() {
        let xn = rmsnorm_rows(&h, &hwl.op_norm, b, d, eps);
        let mixer_out: Vec<f64> = if cfg.is_attn[li] {
            let mut q = matt(&xn, &hwl.wq, b, d, qd);
            let mut kf = vec![0f64; l * kvd];
            let mut vf = vec![0f64; l * kvd];
            for ci in 0..n_ctx {
                let krow = matt(&ctxf[ci * d..(ci + 1) * d], &hwl.wk, 1, d, kvd);
                let vrow = matt(&ctxf[ci * d..(ci + 1) * d], &hwl.wv, 1, d, kvd);
                kf[ci * kvd..(ci + 1) * kvd].copy_from_slice(&krow);
                vf[ci * kvd..(ci + 1) * kvd].copy_from_slice(&vrow);
            }
            for bi in 0..b {
                let row = n_ctx + bi;
                let krow = matt(&xn[bi * d..(bi + 1) * d], &hwl.wk, 1, d, kvd);
                let vrow = matt(&xn[bi * d..(bi + 1) * d], &hwl.wv, 1, d, kvd);
                kf[row * kvd..(row + 1) * kvd].copy_from_slice(&krow);
                vf[row * kvd..(row + 1) * kvd].copy_from_slice(&vrow);
            }
            qknorm(&mut q, &hwl.q_norm, b, nh, hd, eps);
            qknorm(&mut kf, &hwl.k_norm, l, nkv, hd, eps);
            rope_rows(&mut q, blk, b, nh, hd, theta);
            rope_rows(&mut kf, full, l, nkv, hd, theta);
            let mut ao = vec![0f64; b * qd];
            for i in 0..b { for head in 0..nh {
                let g = head / group;
                let mut sc = vec![0f64; l];
                let mut mx = f64::NEG_INFINITY;
                for j in 0..l {
                    let mut dot = 0f64;
                    for dd in 0..hd { dot += q[(i * nh + head) * hd + dd] * kf[(j * nkv + g) * hd + dd]; }
                    sc[j] = dot * scale;
                    if sc[j] > mx { mx = sc[j]; }
                }
                let mut z = 0f64;
                for j in 0..l { sc[j] = (sc[j] - mx).exp(); z += sc[j]; }
                for dd in 0..hd {
                    let mut acc = 0f64;
                    for j in 0..l { acc += sc[j] * vf[(j * nkv + g) * hd + dd]; }
                    ao[(i * nh + head) * hd + dd] = acc / z;
                }
            }}
            matt(&ao, &hwl.wo, b, qd, d)
        } else {
            let bcx = matt(&xn, &hwl.in_proj, b, d, 3 * d);
            let mut cy = vec![0f64; b * d];
            for j in 0..b { for c in 0..d {
                let cg = bcx[j * 3 * d + d + c];
                let mut acc = 0f64;
                for t in 0..k {
                    let p = j as isize - (k as isize - 1) + t as isize;
                    if p >= 0 {
                        let pp = p as usize;
                        let bx = bcx[pp * 3 * d + c] * bcx[pp * 3 * d + 2 * d + c];
                        acc += bx * hwl.conv_w[c * k + t] as f64;
                    }
                }
                cy[j * d + c] = cg * acc;
            }}
            matt(&cy, &hwl.out_proj, b, d, d)
        };
        let mut h_mid = vec![0f64; b * d];
        for i in 0..b * d { h_mid[i] = h[i] + mixer_out[i]; }
        let fnorm = rmsnorm_rows(&h_mid, &hwl.ffn_norm, b, d, eps);
        let gg = matt(&fnorm, &hwl.w1, b, d, cfg.inter);
        let uu = matt(&fnorm, &hwl.w3, b, d, cfg.inter);
        let mut act = vec![0f64; b * cfg.inter];
        for i in 0..b * cfg.inter { let g = gg[i]; act[i] = (g / (1.0 + (-g).exp())) * uu[i]; }
        let fo = matt(&act, &hwl.w2, b, cfg.inter, d);
        for i in 0..b * d { h[i] = h_mid[i] + fo[i]; }
    }
    h
}

#[allow(clippy::too_many_arguments)]
fn forward(
    gpu: &mut Gpu, cfg: &Cfg, w: &[LW], h_in0: &GpuTensor, ctx: &GpuTensor,
    block_pos: &GpuTensor, full_pos: &GpuTensor, conv_state: &GpuTensor, b: usize, n_ctx: usize,
) -> (GpuTensor, Vec<LT>) {
    let (d, nh, nkv, hd, qd, kvd) = (cfg.d, cfg.nh, cfg.nkv, cfg.hd, cfg.qd(), cfg.kvd());
    let l = n_ctx + b;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut tape: Vec<LT> = Vec::new();
    // h carried as a fresh tensor each layer
    let mut h = dup(gpu, h_in0, b * d); // copy
    for li in 0..cfg.n_layers() {
        let lw = &w[li];
        let mut t = LT::default();
        let h_in = h;
        let xn = gpu.zeros(&[b, d], DType::F32).unwrap();
        gpu.rmsnorm_batched(&h_in, &lw.op_norm, &xn, b, d, cfg.eps).unwrap();

        let mixer_out;
        if cfg.is_attn[li] {
            let q0 = lin(gpu, &xn, lw.wq.as_ref().unwrap(), b, d, qd);
            let kfull0 = gpu.zeros(&[l, kvd], DType::F32).unwrap();
            let vfull = gpu.zeros(&[l, kvd], DType::F32).unwrap();
            if n_ctx > 0 {
                gpu.linear_fwd_f32(ctx, lw.wk.as_ref().unwrap(), &kfull0.sub_offset(0, n_ctx * kvd), n_ctx, d, kvd).unwrap();
                gpu.linear_fwd_f32(ctx, lw.wv.as_ref().unwrap(), &vfull.sub_offset(0, n_ctx * kvd), n_ctx, d, kvd).unwrap();
            }
            gpu.linear_fwd_f32(&xn, lw.wk.as_ref().unwrap(), &kfull0.sub_offset(n_ctx * kvd, b * kvd), b, d, kvd).unwrap();
            gpu.linear_fwd_f32(&xn, lw.wv.as_ref().unwrap(), &vfull.sub_offset(n_ctx * kvd, b * kvd), b, d, kvd).unwrap();
            // qk-norm into fresh buffers (preserve pre-norm q0/kfull0 for bwd)
            let qn = gpu.zeros(&[b, qd], DType::F32).unwrap();
            gpu.rmsnorm_batched(&q0, lw.q_norm.as_ref().unwrap(), &qn, b * nh, hd, cfg.eps).unwrap();
            let kn = gpu.zeros(&[l, kvd], DType::F32).unwrap();
            gpu.rmsnorm_batched(&kfull0, lw.k_norm.as_ref().unwrap(), &kn, l * nkv, hd, cfg.eps).unwrap();
            // rope in place on qn/kn -> qr/kr
            gpu.rope_batched_f32(&qn, &qn, block_pos, nh, 0, hd, cfg.theta, b).unwrap();
            gpu.rope_batched_f32(&kn, &kn, full_pos, 0, nkv, hd, cfg.theta, l).unwrap();
            let attn_out = gpu.zeros(&[b, qd], DType::F32).unwrap();
            gpu.attn_block_ctx_f32(&qn, &kn, &vfull, &attn_out, b, l, nh, nkv, hd, scale).unwrap();
            mixer_out = lin(gpu, &attn_out, lw.wo.as_ref().unwrap(), b, qd, d);
            t.q0 = Some(q0); t.kfull0 = Some(kfull0); t.qr = Some(qn); t.kr = Some(kn);
            t.vfull = Some(vfull); t.attn_out = Some(attn_out);
        } else {
            let bcx = lin(gpu, &xn, lw.in_proj.as_ref().unwrap(), b, d, 3 * d);
            let cy = gpu.zeros(&[b, d], DType::F32).unwrap();
            gpu.conv1d_gated_batched_f32(&bcx, conv_state, lw.conv_w.as_ref().unwrap(), &cy, b, d, cfg.conv_k).unwrap();
            mixer_out = lin(gpu, &cy, lw.out_proj.as_ref().unwrap(), b, d, d);
            t.bcx = Some(bcx); t.cy = Some(cy);
        }
        let h_mid = add_new(gpu, &h_in, &mixer_out, b * d);

        let fnorm = gpu.zeros(&[b, d], DType::F32).unwrap();
        gpu.rmsnorm_batched(&h_mid, &lw.ffn_norm, &fnorm, b, d, cfg.eps).unwrap();
        let g = lin(gpu, &fnorm, &lw.w1, b, d, cfg.inter);
        let u = lin(gpu, &fnorm, &lw.w3, b, d, cfg.inter);
        let act = gpu.zeros(&[b, cfg.inter], DType::F32).unwrap();
        gpu.silu_mul_f32(&g, &u, &act).unwrap();
        let fo = lin(gpu, &act, &lw.w2, b, cfg.inter, d);
        let h_out = add_new(gpu, &h_mid, &fo, b * d);

        t.h_in = Some(h_in); t.xn = Some(xn); t.h_mid = Some(h_mid); t.fnorm = Some(fnorm);
        t.g = Some(g); t.u = Some(u); t.act = Some(act);
        tape.push(t);
        h = h_out;
    }
    (h, tape)
}

#[derive(Default)]
struct Grads {
    // sampled weight grads keyed by (layer, class)
    d_wq: Vec<Option<Vec<f32>>>,
    d_in_proj: Vec<Option<Vec<f32>>>,
    d_w1: Vec<Option<Vec<f32>>>,
    d_op_norm: Vec<Option<Vec<f32>>>,
}

#[allow(clippy::too_many_arguments)]
fn backward(
    gpu: &mut Gpu, cfg: &Cfg, w: &[LW], tape: &[LT], dh_out: &GpuTensor, ctx: &GpuTensor,
    block_pos: &GpuTensor, full_pos: &GpuTensor, conv_state: &GpuTensor, b: usize, n_ctx: usize,
) -> (Vec<f32>, Vec<f32>, Grads) {
    let (d, nh, nkv, hd, qd, kvd) = (cfg.d, cfg.nh, cfg.nkv, cfg.hd, cfg.qd(), cfg.kvd());
    let l = n_ctx + b;
    let scale = 1.0 / (hd as f32).sqrt();
    let nl = cfg.n_layers();
    let mut g = Grads { d_wq: vec![None; nl], d_in_proj: vec![None; nl], d_w1: vec![None; nl], d_op_norm: vec![None; nl] };
    let d_ctx = zeros(gpu, n_ctx.max(1) * d); // accumulator across attn layers
    let mut dh = dup(gpu, dh_out, b * d); // copy

    for li in (0..nl).rev() {
        let lw = &w[li];
        let t = &tape[li];
        // ---- FFN backward ----
        let dfo = &dh; // residual: grad to fo == grad to h_out
        let dact = lin_dx(gpu, dfo, &lw.w2, b, cfg.inter, d);
        let _dw2 = lin_dw(gpu, dfo, t.act.as_ref().unwrap(), b, cfg.inter, d);
        let dg = zeros(gpu, b * cfg.inter);
        let du = zeros(gpu, b * cfg.inter);
        gpu.silu_mul_bwd_f32(t.g.as_ref().unwrap(), t.u.as_ref().unwrap(), &dact, &dg, &du, b * cfg.inter).unwrap();
        let dfnorm = lin_dx(gpu, &dg, &lw.w1, b, d, cfg.inter);
        let dfnorm_u = lin_dx(gpu, &du, &lw.w3, b, d, cfg.inter);
        addv(gpu, &dfnorm, &dfnorm_u);
        let dw1 = lin_dw(gpu, &dg, t.fnorm.as_ref().unwrap(), b, d, cfg.inter);
        let _dw3 = lin_dw(gpu, &du, t.fnorm.as_ref().unwrap(), b, d, cfg.inter);
        // fnorm = rmsnorm(h_mid, ffn_norm)
        let dhmid_n = zeros(gpu, b * d);
        let dffn_norm = zeros(gpu, d);
        gpu.rmsnorm_bwd_f32(t.h_mid.as_ref().unwrap(), &lw.ffn_norm, &dfnorm, &dhmid_n, &dffn_norm, b, d, cfg.eps).unwrap();
        // dh_mid = dh (residual) + dhmid_n
        let dh_mid = add_new(gpu, &dh, &dhmid_n, b * d);

        // ---- mixer backward ----
        let dmix = &dh_mid; // grad to mixer_out == grad to h_mid (residual)
        let d_xn;
        let mut d_op_norm_save: Option<Vec<f32>> = None;
        let mut d_inproj_save: Option<Vec<f32>> = None;
        let mut d_wq_save: Option<Vec<f32>> = None;
        if cfg.is_attn[li] {
            let d_attn_out = lin_dx(gpu, dmix, lw.wo.as_ref().unwrap(), b, qd, d);
            let _dwo = lin_dw(gpu, dmix, t.attn_out.as_ref().unwrap(), b, qd, d);
            // attn backward (dk/dv zeroed; atomic-accumulated)
            let dqr = gpu.zeros(&[b, qd], DType::F32).unwrap();
            let dkr = zeros(gpu, l * kvd);
            let dvfull = zeros(gpu, l * kvd);
            gpu.attn_block_ctx_bwd_f32(t.qr.as_ref().unwrap(), t.kr.as_ref().unwrap(), t.vfull.as_ref().unwrap(),
                &d_attn_out, &dqr, &dkr, &dvfull, b, l, nh, nkv, hd, scale).unwrap();
            // rope backward (in place)
            gpu.rope_rows_bwd_f32(&dqr, block_pos, nh, hd, cfg.theta, b).unwrap();
            gpu.rope_rows_bwd_f32(&dkr, full_pos, nkv, hd, cfg.theta, l).unwrap();
            // qk-norm backward
            let dq0 = zeros(gpu, b * qd);
            let dq_norm = zeros(gpu, hd);
            gpu.rmsnorm_bwd_f32(t.q0.as_ref().unwrap(), lw.q_norm.as_ref().unwrap(), &dqr, &dq0, &dq_norm, b * nh, hd, cfg.eps).unwrap();
            let dkfull0 = zeros(gpu, l * kvd);
            let dk_norm = zeros(gpu, hd);
            gpu.rmsnorm_bwd_f32(t.kfull0.as_ref().unwrap(), lw.k_norm.as_ref().unwrap(), &dkr, &dkfull0, &dk_norm, l * nkv, hd, cfg.eps).unwrap();
            // split dkfull0/dvfull -> ctx rows [0,n_ctx), block rows [n_ctx,l)
            let dk_blk = dkfull0.sub_offset(n_ctx * kvd, b * kvd);
            let dv_blk = dvfull.sub_offset(n_ctx * kvd, b * kvd);
            // d_xn from wq, wk(block), wv(block)
            let dxn = lin_dx(gpu, &dq0, lw.wq.as_ref().unwrap(), b, d, qd);
            let dxn_k = lin_dx(gpu, &dk_blk, lw.wk.as_ref().unwrap(), b, d, kvd);
            let dxn_v = lin_dx(gpu, &dv_blk, lw.wv.as_ref().unwrap(), b, d, kvd);
            addv(gpu, &dxn, &dxn_k);
            addv(gpu, &dxn, &dxn_v);
            d_xn = dxn;
            // dWq
            let dwq = lin_dw(gpu, &dq0, t.xn.as_ref().unwrap(), b, d, qd);
            d_wq_save = Some(gpu.download_f32(&dwq).unwrap());
            // d_ctx from wk(ctx), wv(ctx)
            if n_ctx > 0 {
                let dk_ctx = dkfull0.sub_offset(0, n_ctx * kvd);
                let dv_ctx = dvfull.sub_offset(0, n_ctx * kvd);
                let dctx_k = lin_dx(gpu, &dk_ctx, lw.wk.as_ref().unwrap(), n_ctx, d, kvd);
                let dctx_v = lin_dx(gpu, &dv_ctx, lw.wv.as_ref().unwrap(), n_ctx, d, kvd);
                addv(gpu, &d_ctx, &dctx_k);
                addv(gpu, &d_ctx, &dctx_v);
            }
        } else {
            let d_cy = lin_dx(gpu, dmix, lw.out_proj.as_ref().unwrap(), b, d, d);
            let _dwout = lin_dw(gpu, dmix, t.cy.as_ref().unwrap(), b, d, d);
            // conv backward
            let d_bcx = zeros(gpu, b * 3 * d);
            let d_conv_w = zeros(gpu, d * cfg.conv_k);
            let d_state = zeros(gpu, d * (cfg.conv_k - 1));
            let d_conv_scratch = zeros(gpu, b * d);
            gpu.conv1d_gated_batched_bwd(t.bcx.as_ref().unwrap(), conv_state, lw.conv_w.as_ref().unwrap(),
                &d_cy, &d_bcx, &d_conv_w, &d_state, &d_conv_scratch, b, d, cfg.conv_k).unwrap();
            let dxn = lin_dx(gpu, &d_bcx, lw.in_proj.as_ref().unwrap(), b, d, 3 * d);
            let dinproj = lin_dw(gpu, &d_bcx, t.xn.as_ref().unwrap(), b, d, 3 * d);
            d_inproj_save = Some(gpu.download_f32(&dinproj).unwrap());
            d_xn = dxn;
        }
        // op_norm backward: xn = rmsnorm(h_in, op_norm)
        let dhin_n = zeros(gpu, b * d);
        let dop_norm = zeros(gpu, d);
        gpu.rmsnorm_bwd_f32(t.h_in.as_ref().unwrap(), &lw.op_norm, &d_xn, &dhin_n, &dop_norm, b, d, cfg.eps).unwrap();
        d_op_norm_save = Some(gpu.download_f32(&dop_norm).unwrap());
        let dh_in = add_new(gpu, &dh_mid, &dhin_n, b * d);

        // sampled weight grads
        g.d_w1[li] = Some(gpu.download_f32(&dw1).unwrap());
        g.d_op_norm[li] = d_op_norm_save;
        g.d_wq[li] = d_wq_save;
        g.d_in_proj[li] = d_inproj_save;

        dh = dh_in;
    }
    let d_body_in = gpu.download_f32(&dh).unwrap();
    let d_ctx_v = if n_ctx > 0 { gpu.download_f32(&d_ctx).unwrap() } else { vec![] };
    (d_body_in, d_ctx_v, g)
}

// host weights (so fd can perturb on the host then re-upload)
#[derive(Clone)]
struct HW {
    op_norm: Vec<f32>, ffn_norm: Vec<f32>,
    in_proj: Vec<f32>, conv_w: Vec<f32>, out_proj: Vec<f32>,
    wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>, q_norm: Vec<f32>, k_norm: Vec<f32>,
    w1: Vec<f32>, w3: Vec<f32>, w2: Vec<f32>, is_attn: bool,
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    // tiny toy: 2 layers (conv then attn), small dims for fast fd
    let cfg = Cfg { d: 32, is_attn: vec![false, true], nh: 4, nkv: 2, hd: 8, conv_k: 3, inter: 48, theta: 1.0e6, eps: 1e-5 };
    let (d, nh, nkv, hd, qd, kvd) = (cfg.d, cfg.nh, cfg.nkv, cfg.hd, cfg.qd(), cfg.kvd());
    let b = 4usize;
    let n_ctx = 3usize;
    let _l = n_ctx + b;

    let mk = |li: usize, is_attn: bool| -> HW {
        let s = li * 100_003;
        let ws = 1.0 / (d as f32).sqrt();
        HW {
            op_norm: (0..d).map(|i| 1.0 + 0.1 * frand(i + s)).collect(),
            ffn_norm: (0..d).map(|i| 1.0 + 0.1 * frand(i + s + 7)).collect(),
            in_proj: (0..3 * d * d).map(|i| frand(i + s + 100) * ws).collect(),
            conv_w: (0..d * cfg.conv_k).map(|i| frand(i + s + 200) * 0.3).collect(),
            out_proj: (0..d * d).map(|i| frand(i + s + 300) * ws).collect(),
            wq: (0..qd * d).map(|i| frand(i + s + 400) * ws).collect(),
            wk: (0..kvd * d).map(|i| frand(i + s + 500) * ws).collect(),
            wv: (0..kvd * d).map(|i| frand(i + s + 600) * ws).collect(),
            wo: (0..d * qd).map(|i| frand(i + s + 700) * (1.0 / (qd as f32).sqrt())).collect(),
            q_norm: (0..hd).map(|i| 1.0 + 0.1 * frand(i + s + 11)).collect(),
            k_norm: (0..hd).map(|i| 1.0 + 0.1 * frand(i + s + 13)).collect(),
            w1: (0..cfg.inter * d).map(|i| frand(i + s + 800) * ws).collect(),
            w3: (0..cfg.inter * d).map(|i| frand(i + s + 900) * ws).collect(),
            w2: (0..d * cfg.inter).map(|i| frand(i + s + 1000) * (1.0 / (cfg.inter as f32).sqrt())).collect(),
            is_attn,
        }
    };
    let hw: Vec<HW> = cfg.is_attn.iter().enumerate().map(|(li, &a)| mk(li, a)).collect();

    let body_in: Vec<f32> = (0..b * d).map(|i| 0.5 * frand(i + 3)).collect();
    let ctx_v: Vec<f32> = (0..n_ctx * d).map(|i| 0.5 * frand(i + 8000)).collect();
    let aprobe: Vec<f32> = (0..b * d).map(|i| frand(i + 555)).collect();

    let block_pos: Vec<i32> = (0..b).map(|i| (n_ctx + i) as i32).collect();
    let mut full: Vec<i32> = (0..n_ctx).map(|i| i as i32).collect();
    full.extend(block_pos.iter().copied());

    // build GPU weights from host (inline; no nested gpu-borrowing closure)
    let build = |gpu: &mut Gpu, hw: &[HW]| -> Vec<LW> {
        let mut out = Vec::new();
        for h in hw {
            let op_norm = up(gpu, &h.op_norm, &[d]);
            let ffn_norm = up(gpu, &h.ffn_norm, &[d]);
            let (in_proj, conv_w, out_proj) = if !h.is_attn {
                (Some(up(gpu, &h.in_proj, &[3 * d, d])), Some(up(gpu, &h.conv_w, &[d, cfg.conv_k])), Some(up(gpu, &h.out_proj, &[d, d])))
            } else { (None, None, None) };
            let (wq, wk, wv, wo, q_norm, k_norm) = if h.is_attn {
                (Some(up(gpu, &h.wq, &[qd, d])), Some(up(gpu, &h.wk, &[kvd, d])), Some(up(gpu, &h.wv, &[kvd, d])),
                 Some(up(gpu, &h.wo, &[d, qd])), Some(up(gpu, &h.q_norm, &[hd])), Some(up(gpu, &h.k_norm, &[hd])))
            } else { (None, None, None, None, None, None) };
            let w1 = up(gpu, &h.w1, &[cfg.inter, d]);
            let w3 = up(gpu, &h.w3, &[cfg.inter, d]);
            let w2 = up(gpu, &h.w2, &[d, cfg.inter]);
            out.push(LW { op_norm, ffn_norm, in_proj, conv_w, out_proj, wq, wk, wv, wo, q_norm, k_norm, w1, w3, w2 });
        }
        out
    };

    // analytic backward
    let block_pos_g = upload_pos(&mut gpu, &block_pos);
    let full_pos_g = upload_pos(&mut gpu, &full);
    let conv_state = gpu.zeros(&[d, cfg.conv_k - 1], DType::F32).unwrap();
    let weights = build(&mut gpu, &hw);
    let h0 = up(&mut gpu, &body_in, &[b, d]);
    let ctxg = up(&mut gpu, &ctx_v, &[n_ctx, d]);
    let (_hout, tape) = forward(&mut gpu, &cfg, &weights, &h0, &ctxg, &block_pos_g, &full_pos_g, &conv_state, b, n_ctx);
    let dh_out = up(&mut gpu, &aprobe, &[b, d]);
    let (d_body_in, d_ctx, grads) = backward(&mut gpu, &cfg, &weights, &tape, &dh_out, &ctxg, &block_pos_g, &full_pos_g, &conv_state, b, n_ctx);

    // fd helper: L(perturbed)
    let _run_loss = |gpu: &mut Gpu, hw: &[HW], bin: &[f32], cv: &[f32]| -> f64 {
        let wts = build(gpu, hw);
        let h0 = up(gpu, bin, &[b, d]);
        let ctxg = up(gpu, cv, &[n_ctx, d]);
        let (hout, _t) = forward(gpu, &cfg, &wts, &h0, &ctxg, &block_pos_g, &full_pos_g, &conv_state, b, n_ctx);
        let ho = gpu.download_f32(&hout).unwrap();
        aprobe.iter().zip(&ho).map(|(x, y)| *x as f64 * *y as f64).sum()
    };
    // fd loss via the f64 host oracle (no f32 GPU-forward noise)
    let host_loss = |hw: &[HW], bin: &[f32], cv: &[f32]| -> f64 {
        let ho = host_forward(&cfg, hw, bin, cv, &block_pos, &full, b, n_ctx);
        aprobe.iter().zip(&ho).map(|(x, y)| *x as f64 * y).sum()
    };
    let eps = 1e-3f64;
    let samp = |n: usize| -> Vec<usize> { let c = 12.min(n); (0..c).map(|i| (i * 2654435761) % n).collect() };
    let check = |gpu: &mut Gpu, name: &str, analytic: &[f32], fd_fn: &dyn Fn(&mut Gpu, usize) -> f64, n: usize| -> f64 {
        let mut w = 0f64;
        for t in samp(n) {
            let fd = fd_fn(gpu, t);
            let an = analytic[t] as f64;
            let den = fd.abs().max(an.abs()).max(1e-3);
            w = w.max((fd - an).abs() / den);
        }
        println!("  {name}: max rel err = {w:.3e}  (n={})", samp(n).len());
        w
    };
    let mut worst = 0f64;

    worst = worst.max(check(&mut gpu, "d_body_in", &d_body_in, &|_g, t| {
        let mut p = body_in.clone(); p[t] += eps as f32;
        let mut m = body_in.clone(); m[t] -= eps as f32;
        (host_loss(&hw, &p, &ctx_v) - host_loss(&hw, &m, &ctx_v)) / (2.0 * eps)
    }, b * d));
    worst = worst.max(check(&mut gpu, "d_ctx", &d_ctx, &|_g, t| {
        let mut p = ctx_v.clone(); p[t] += eps as f32;
        let mut m = ctx_v.clone(); m[t] -= eps as f32;
        (host_loss(&hw, &body_in, &p) - host_loss(&hw, &body_in, &m)) / (2.0 * eps)
    }, n_ctx * d));
    if let Some(dw1) = &grads.d_w1[0] {
        worst = worst.max(check(&mut gpu, "d_w1[L0]", dw1, &|_g, t| {
            let mut hp = hw.clone(); hp[0].w1[t] += eps as f32;
            let mut hm = hw.clone(); hm[0].w1[t] -= eps as f32;
            (host_loss(&hp, &body_in, &ctx_v) - host_loss(&hm, &body_in, &ctx_v)) / (2.0 * eps)
        }, cfg.inter * d));
    }
    if let Some(dip) = &grads.d_in_proj[0] {
        worst = worst.max(check(&mut gpu, "d_in_proj[L0]", dip, &|_g, t| {
            let mut hp = hw.clone(); hp[0].in_proj[t] += eps as f32;
            let mut hm = hw.clone(); hm[0].in_proj[t] -= eps as f32;
            (host_loss(&hp, &body_in, &ctx_v) - host_loss(&hm, &body_in, &ctx_v)) / (2.0 * eps)
        }, 3 * d * d));
    }
    if let Some(dwq) = &grads.d_wq[1] {
        worst = worst.max(check(&mut gpu, "d_wq[L1]", dwq, &|_g, t| {
            let mut hp = hw.clone(); hp[1].wq[t] += eps as f32;
            let mut hm = hw.clone(); hm[1].wq[t] -= eps as f32;
            (host_loss(&hp, &body_in, &ctx_v) - host_loss(&hm, &body_in, &ctx_v)) / (2.0 * eps)
        }, qd * d));
    }
    if let Some(don) = &grads.d_op_norm[1] {
        worst = worst.max(check(&mut gpu, "d_op_norm[L1]", don, &|_g, t| {
            let mut hp = hw.clone(); hp[1].op_norm[t] += eps as f32;
            let mut hm = hw.clone(); hm[1].op_norm[t] -= eps as f32;
            (host_loss(&hp, &body_in, &ctx_v) - host_loss(&hm, &body_in, &ctx_v)) / (2.0 * eps)
        }, d));
    }

    if worst < 2e-2 {
        println!("dflash_body_gradcheck: PASS (worst rel err {worst:.3e})");
    } else {
        println!("dflash_body_gradcheck: FAIL (worst rel err {worst:.3e})");
        std::process::exit(1);
    }
}
