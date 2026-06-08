// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash spike — step 5: block-diffusion loss + Adam + driver, validated by an
// OVERFIT test. Assembles the full draft model (token -> frozen target embed ->
// in_proj -> LFM2 body w/ GQA KV-injection -> final_norm -> out_proj -> frozen
// target lm_head -> logits), block-diffusion position-weighted CE, full
// GPU-resident reverse-mode backward (all kernels validated in steps 3b/4), and
// AdamW. If the loop can drive a fixed tiny batch's loss to ~0, the entire
// trainer (forward + loss + backward + optimizer + driver) is wired correctly.

use rdna_compute::{DType, Gpu, GpuTensor};

#[derive(Clone)]
struct Cfg {
    d: usize, is_attn: Vec<bool>, nh: usize, nkv: usize, hd: usize,
    conv_k: usize, inter: usize, theta: f32, eps: f32,
    d_tgt: usize, vocab: usize, n_tgt_layers: usize,
}
impl Cfg {
    fn n_layers(&self) -> usize { self.is_attn.len() }
    fn qd(&self) -> usize { self.nh * self.hd }
    fn kvd(&self) -> usize { self.nkv * self.hd }
}

// Per-layer trainable tensors (also used to hold per-layer GRADS — same shape).
struct LW {
    op_norm: GpuTensor, ffn_norm: GpuTensor,
    in_proj: Option<GpuTensor>, conv_w: Option<GpuTensor>, out_proj: Option<GpuTensor>,
    wq: Option<GpuTensor>, wk: Option<GpuTensor>, wv: Option<GpuTensor>, wo: Option<GpuTensor>,
    q_norm: Option<GpuTensor>, k_norm: Option<GpuTensor>,
    w1: GpuTensor, w3: GpuTensor, w2: GpuTensor,
}
// Whole trainable net (layers + vocab/context adapters). Used for weights, grads,
// and Adam m/v (all same shape).
struct Net {
    layers: Vec<LW>,
    in_proj_v: GpuTensor,  // [d, d_tgt]
    out_proj_v: GpuTensor, // [d_tgt, d]
    fc: GpuTensor,         // [d, n_tgt_layers*d_tgt]
    final_norm: GpuTensor, // [d]
}
fn lw_tensors(lw: &LW) -> Vec<&GpuTensor> {
    let mut v = vec![&lw.op_norm, &lw.ffn_norm];
    for o in [&lw.in_proj, &lw.conv_w, &lw.out_proj, &lw.wq, &lw.wk, &lw.wv, &lw.wo, &lw.q_norm, &lw.k_norm] {
        if let Some(t) = o { v.push(t); }
    }
    v.push(&lw.w1); v.push(&lw.w3); v.push(&lw.w2);
    v
}
fn net_tensors(net: &Net) -> Vec<&GpuTensor> {
    let mut v = Vec::new();
    for l in &net.layers { v.extend(lw_tensors(l)); }
    v.push(&net.in_proj_v); v.push(&net.out_proj_v); v.push(&net.fc); v.push(&net.final_norm);
    v
}

#[derive(Default)]
struct LT {
    h_in: Option<GpuTensor>, xn: Option<GpuTensor>, h_mid: Option<GpuTensor>, fnorm: Option<GpuTensor>,
    g: Option<GpuTensor>, u: Option<GpuTensor>, act: Option<GpuTensor>,
    bcx: Option<GpuTensor>, cy: Option<GpuTensor>,
    q0: Option<GpuTensor>, kfull0: Option<GpuTensor>, qr: Option<GpuTensor>, kr: Option<GpuTensor>,
    vfull: Option<GpuTensor>, attn_out: Option<GpuTensor>,
}

fn frand(seed: usize) -> f32 { ((seed.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0 }
fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }
fn zt(gpu: &mut Gpu, n: usize) -> GpuTensor { gpu.zeros(&[n], DType::F32).unwrap() }
fn lin(gpu: &mut Gpu, x: &GpuTensor, w: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let y = gpu.zeros(&[m, n], DType::F32).unwrap(); gpu.linear_fwd_f32(x, w, &y, m, k, n).unwrap(); y
}
fn lin_dx(gpu: &mut Gpu, dy: &GpuTensor, w: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let dx = gpu.zeros(&[m, k], DType::F32).unwrap(); gpu.linear_bwd_dx_f32(dy, w, &dx, m, k, n).unwrap(); dx
}
fn lin_dw(gpu: &mut Gpu, dy: &GpuTensor, x: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let dw = gpu.zeros(&[n, k], DType::F32).unwrap(); gpu.linear_bwd_dw_f32(dy, x, &dw, m, k, n).unwrap(); dw
}
fn addv(gpu: &mut Gpu, a: &GpuTensor, b: &GpuTensor) { gpu.add_inplace_f32(a, b).unwrap(); }
fn add_new(gpu: &mut Gpu, a: &GpuTensor, b: &GpuTensor, n: usize) -> GpuTensor {
    let c = gpu.zeros(&[n], DType::F32).unwrap(); gpu.add_f32(a, b, &c).unwrap(); c
}
fn dup(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> GpuTensor {
    let c = gpu.zeros(&[n], DType::F32).unwrap(); gpu.add_inplace_f32(&c, t).unwrap(); c
}
fn upos(gpu: &mut Gpu, pos: &[i32]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[pos.len()], DType::F32).unwrap();
    let bytes: Vec<u8> = pos.iter().flat_map(|p| p.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).unwrap(); t
}
fn ui32(gpu: &mut Gpu, vals: &[i32]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[vals.len()], DType::F32).unwrap();
    let bytes: Vec<u8> = vals.iter().flat_map(|p| p.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).unwrap(); t
}

// ---------------- body forward (saves tape) ----------------
#[allow(clippy::too_many_arguments)]
fn body_forward(gpu: &mut Gpu, cfg: &Cfg, w: &[LW], h_in0: &GpuTensor, ctx: &GpuTensor,
    block_pos: &GpuTensor, full_pos: &GpuTensor, conv_state: &GpuTensor, b: usize, n_ctx: usize) -> (GpuTensor, Vec<LT>) {
    let (d, nh, nkv, hd, qd, kvd) = (cfg.d, cfg.nh, cfg.nkv, cfg.hd, cfg.qd(), cfg.kvd());
    let l = n_ctx + b; let scale = 1.0 / (hd as f32).sqrt();
    let mut tape = Vec::new();
    let mut h = dup(gpu, h_in0, b * d);
    for li in 0..cfg.n_layers() {
        let lw = &w[li]; let mut t = LT::default();
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
            let qn = gpu.zeros(&[b, qd], DType::F32).unwrap();
            gpu.rmsnorm_batched(&q0, lw.q_norm.as_ref().unwrap(), &qn, b * nh, hd, cfg.eps).unwrap();
            let kn = gpu.zeros(&[l, kvd], DType::F32).unwrap();
            gpu.rmsnorm_batched(&kfull0, lw.k_norm.as_ref().unwrap(), &kn, l * nkv, hd, cfg.eps).unwrap();
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
        tape.push(t); h = h_out;
    }
    (h, tape)
}

// ---------------- body backward (full GPU grads) ----------------
#[allow(clippy::too_many_arguments)]
fn body_backward(gpu: &mut Gpu, cfg: &Cfg, w: &[LW], tape: &[LT], dh_out: &GpuTensor, ctx: &GpuTensor,
    block_pos: &GpuTensor, full_pos: &GpuTensor, conv_state: &GpuTensor, b: usize, n_ctx: usize)
    -> (GpuTensor, GpuTensor, Vec<LW>) {
    let (d, nh, nkv, hd, qd, kvd) = (cfg.d, cfg.nh, cfg.nkv, cfg.hd, cfg.qd(), cfg.kvd());
    let l = n_ctx + b; let scale = 1.0 / (hd as f32).sqrt(); let nl = cfg.n_layers();
    let mut glayers: Vec<Option<LW>> = (0..nl).map(|_| None).collect();
    let d_ctx = zt(gpu, n_ctx.max(1) * d);
    let mut dh = dup(gpu, dh_out, b * d);
    for li in (0..nl).rev() {
        let lw = &w[li]; let t = &tape[li];
        // FFN bwd
        let dact = lin_dx(gpu, &dh, &lw.w2, b, cfg.inter, d);
        let dw2 = lin_dw(gpu, &dh, t.act.as_ref().unwrap(), b, cfg.inter, d);
        let dg = zt(gpu, b * cfg.inter); let du = zt(gpu, b * cfg.inter);
        gpu.silu_mul_bwd_f32(t.g.as_ref().unwrap(), t.u.as_ref().unwrap(), &dact, &dg, &du, b * cfg.inter).unwrap();
        let dfnorm = lin_dx(gpu, &dg, &lw.w1, b, d, cfg.inter);
        let dfnorm_u = lin_dx(gpu, &du, &lw.w3, b, d, cfg.inter);
        addv(gpu, &dfnorm, &dfnorm_u);
        let dw1 = lin_dw(gpu, &dg, t.fnorm.as_ref().unwrap(), b, d, cfg.inter);
        let dw3 = lin_dw(gpu, &du, t.fnorm.as_ref().unwrap(), b, d, cfg.inter);
        let dhmid_n = zt(gpu, b * d); let dffn_norm = zt(gpu, d);
        gpu.rmsnorm_bwd_f32(t.h_mid.as_ref().unwrap(), &lw.ffn_norm, &dfnorm, &dhmid_n, &dffn_norm, b, d, cfg.eps).unwrap();
        let dh_mid = add_new(gpu, &dh, &dhmid_n, b * d);
        // mixer bwd
        let d_xn;
        let (mut g_inproj, mut g_convw, mut g_outproj) = (None, None, None);
        let (mut g_wq, mut g_wk, mut g_wv, mut g_wo, mut g_qn, mut g_kn) = (None, None, None, None, None, None);
        if cfg.is_attn[li] {
            let d_attn_out = lin_dx(gpu, &dh_mid, lw.wo.as_ref().unwrap(), b, qd, d);
            let dwo = lin_dw(gpu, &dh_mid, t.attn_out.as_ref().unwrap(), b, qd, d);
            let dqr = gpu.zeros(&[b, qd], DType::F32).unwrap();
            let dkr = zt(gpu, l * kvd); let dvfull = zt(gpu, l * kvd);
            gpu.attn_block_ctx_bwd_f32(t.qr.as_ref().unwrap(), t.kr.as_ref().unwrap(), t.vfull.as_ref().unwrap(),
                &d_attn_out, &dqr, &dkr, &dvfull, b, l, nh, nkv, hd, scale).unwrap();
            gpu.rope_rows_bwd_f32(&dqr, block_pos, nh, hd, cfg.theta, b).unwrap();
            gpu.rope_rows_bwd_f32(&dkr, full_pos, nkv, hd, cfg.theta, l).unwrap();
            let dq0 = zt(gpu, b * qd); let dq_norm = zt(gpu, hd);
            gpu.rmsnorm_bwd_f32(t.q0.as_ref().unwrap(), lw.q_norm.as_ref().unwrap(), &dqr, &dq0, &dq_norm, b * nh, hd, cfg.eps).unwrap();
            let dkfull0 = zt(gpu, l * kvd); let dk_norm = zt(gpu, hd);
            gpu.rmsnorm_bwd_f32(t.kfull0.as_ref().unwrap(), lw.k_norm.as_ref().unwrap(), &dkr, &dkfull0, &dk_norm, l * nkv, hd, cfg.eps).unwrap();
            let dk_blk = dkfull0.sub_offset(n_ctx * kvd, b * kvd);
            let dv_blk = dvfull.sub_offset(n_ctx * kvd, b * kvd);
            let dxn = lin_dx(gpu, &dq0, lw.wq.as_ref().unwrap(), b, d, qd);
            let dxn_k = lin_dx(gpu, &dk_blk, lw.wk.as_ref().unwrap(), b, d, kvd);
            let dxn_v = lin_dx(gpu, &dv_blk, lw.wv.as_ref().unwrap(), b, d, kvd);
            addv(gpu, &dxn, &dxn_k); addv(gpu, &dxn, &dxn_v);
            d_xn = dxn;
            let dwq = lin_dw(gpu, &dq0, t.xn.as_ref().unwrap(), b, d, qd);
            let dwk = lin_dw(gpu, &dk_blk, t.xn.as_ref().unwrap(), b, d, kvd);
            let dwv = lin_dw(gpu, &dv_blk, t.xn.as_ref().unwrap(), b, d, kvd);
            if n_ctx > 0 {
                let dk_ctx = dkfull0.sub_offset(0, n_ctx * kvd);
                let dv_ctx = dvfull.sub_offset(0, n_ctx * kvd);
                let dctx_k = lin_dx(gpu, &dk_ctx, lw.wk.as_ref().unwrap(), n_ctx, d, kvd);
                let dctx_v = lin_dx(gpu, &dv_ctx, lw.wv.as_ref().unwrap(), n_ctx, d, kvd);
                addv(gpu, &d_ctx, &dctx_k); addv(gpu, &d_ctx, &dctx_v);
                let dwk_c = lin_dw(gpu, &dk_ctx, ctx, n_ctx, d, kvd);
                let dwv_c = lin_dw(gpu, &dv_ctx, ctx, n_ctx, d, kvd);
                addv(gpu, &dwk, &dwk_c); addv(gpu, &dwv, &dwv_c);
            }
            g_wq = Some(dwq); g_wk = Some(dwk); g_wv = Some(dwv); g_wo = Some(dwo);
            g_qn = Some(dq_norm); g_kn = Some(dk_norm);
        } else {
            let d_cy = lin_dx(gpu, &dh_mid, lw.out_proj.as_ref().unwrap(), b, d, d);
            let dwout = lin_dw(gpu, &dh_mid, t.cy.as_ref().unwrap(), b, d, d);
            let d_bcx = zt(gpu, b * 3 * d); let d_conv_w = zt(gpu, d * cfg.conv_k);
            let d_state = zt(gpu, d * (cfg.conv_k - 1)); let d_scratch = zt(gpu, b * d);
            gpu.conv1d_gated_batched_bwd(t.bcx.as_ref().unwrap(), conv_state, lw.conv_w.as_ref().unwrap(),
                &d_cy, &d_bcx, &d_conv_w, &d_state, &d_scratch, b, d, cfg.conv_k).unwrap();
            let dxn = lin_dx(gpu, &d_bcx, lw.in_proj.as_ref().unwrap(), b, d, 3 * d);
            let dinproj = lin_dw(gpu, &d_bcx, t.xn.as_ref().unwrap(), b, d, 3 * d);
            d_xn = dxn;
            g_inproj = Some(dinproj); g_convw = Some(d_conv_w); g_outproj = Some(dwout);
        }
        let dhin_n = zt(gpu, b * d); let dop_norm = zt(gpu, d);
        gpu.rmsnorm_bwd_f32(t.h_in.as_ref().unwrap(), &lw.op_norm, &d_xn, &dhin_n, &dop_norm, b, d, cfg.eps).unwrap();
        let dh_in = add_new(gpu, &dh_mid, &dhin_n, b * d);
        glayers[li] = Some(LW {
            op_norm: dop_norm, ffn_norm: dffn_norm,
            in_proj: g_inproj, conv_w: g_convw, out_proj: g_outproj,
            wq: g_wq, wk: g_wk, wv: g_wv, wo: g_wo, q_norm: g_qn, k_norm: g_kn,
            w1: dw1, w3: dw3, w2: dw2,
        });
        dh = dh_in;
    }
    (dh, d_ctx, glayers.into_iter().map(|o| o.unwrap()).collect())
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let cfg = Cfg {
        d: 32, is_attn: vec![false, true], nh: 4, nkv: 2, hd: 8, conv_k: 3, inter: 48,
        theta: 1.0e6, eps: 1e-5, d_tgt: 24, vocab: 48, n_tgt_layers: 5,
    };
    let (d, d_tgt, vocab) = (cfg.d, cfg.d_tgt, cfg.vocab);
    let fc_in = cfg.n_tgt_layers * d_tgt;
    let b = 4usize; let n_ctx = 3usize;
    let block_pos: Vec<i32> = (0..b).map(|i| (n_ctx + i) as i32).collect();
    let mut full: Vec<i32> = (0..n_ctx).map(|i| i as i32).collect();
    full.extend(block_pos.iter().copied());
    let block_pos_g = upos(&mut gpu, &block_pos);
    let full_pos_g = upos(&mut gpu, &full);
    let conv_state = gpu.zeros(&[d, cfg.conv_k - 1], DType::F32).unwrap();

    // ---- weights (random init) ----
    let ws = 1.0 / (d as f32).sqrt();
    let mklw = |gpu: &mut Gpu, li: usize, is_attn: bool| -> LW {
        let s = li * 100_003; let (nh, nkv, hd, qd, kvd) = (cfg.nh, cfg.nkv, cfg.hd, cfg.qd(), cfg.kvd());
        let norm = |gpu: &mut Gpu, n: usize, seed: usize| { let v: Vec<f32> = (0..n).map(|i| 1.0 + 0.1 * frand(i + seed)).collect(); up(gpu, &v, &[n]) };
        let rnd = |gpu: &mut Gpu, rows: usize, cols: usize, seed: usize, sc: f32| { let v: Vec<f32> = (0..rows * cols).map(|i| frand(i + seed) * sc).collect(); up(gpu, &v, &[rows, cols]) };
        LW {
            op_norm: norm(gpu, d, s), ffn_norm: norm(gpu, d, s + 7),
            in_proj: if !is_attn { Some(rnd(gpu, 3 * d, d, s + 100, ws)) } else { None },
            conv_w: if !is_attn { Some(rnd(gpu, d, cfg.conv_k, s + 200, 0.3)) } else { None },
            out_proj: if !is_attn { Some(rnd(gpu, d, d, s + 300, ws)) } else { None },
            wq: if is_attn { Some(rnd(gpu, qd, d, s + 400, ws)) } else { None },
            wk: if is_attn { Some(rnd(gpu, kvd, d, s + 500, ws)) } else { None },
            wv: if is_attn { Some(rnd(gpu, kvd, d, s + 600, ws)) } else { None },
            wo: if is_attn { Some(rnd(gpu, d, qd, s + 700, 1.0 / (qd as f32).sqrt())) } else { None },
            q_norm: if is_attn { Some(norm(gpu, hd, s + 11)) } else { None },
            k_norm: if is_attn { Some(norm(gpu, hd, s + 13)) } else { None },
            w1: rnd(gpu, cfg.inter, d, s + 800, ws),
            w3: rnd(gpu, cfg.inter, d, s + 900, ws),
            w2: rnd(gpu, d, cfg.inter, s + 1000, 1.0 / (cfg.inter as f32).sqrt()),
        }
    };
    let layers: Vec<LW> = cfg.is_attn.iter().enumerate().map(|(li, &a)| mklw(&mut gpu, li, a)).collect();
    let rndv = |gpu: &mut Gpu, rows: usize, cols: usize, seed: usize, sc: f32| { let v: Vec<f32> = (0..rows * cols).map(|i| frand(i + seed) * sc).collect(); up(gpu, &v, &[rows, cols]) };
    let net = Net {
        layers,
        in_proj_v: rndv(&mut gpu, d, d_tgt, 7_001, 1.0 / (d_tgt as f32).sqrt()),
        out_proj_v: rndv(&mut gpu, d_tgt, d, 7_002, ws),
        fc: rndv(&mut gpu, d, fc_in, 7_003, 1.0 / (fc_in as f32).sqrt()),
        final_norm: { let v: Vec<f32> = (0..d).map(|i| 1.0 + 0.1 * frand(i + 7_004)).collect(); up(&mut gpu, &v, &[d]) },
    };

    // ---- frozen target embed + lm_head ----
    let embed: Vec<f32> = (0..vocab * d_tgt).map(|i| frand(i + 50_000) * 0.5).collect();
    let lm_head_v: Vec<f32> = (0..vocab * d_tgt).map(|i| frand(i + 60_000) * (1.0 / (d_tgt as f32).sqrt())).collect();
    let lm_head_g = up(&mut gpu, &lm_head_v, &[vocab, d_tgt]);

    // ---- fixed training batch (block diffusion) ----
    // target_hiddens [n_ctx, fc_in], block input tokens (all masked = 0),
    // target tokens, position weights w_k = exp(-(k-1)/gamma).
    let tgt_hiddens: Vec<f32> = (0..n_ctx * fc_in).map(|i| 0.4 * frand(i + 80_000)).collect();
    let block_tokens: Vec<i32> = vec![0; b]; // masked
    let targets: Vec<i32> = (0..b).map(|i| ((i * 7 + 3) % vocab) as i32).collect();
    let gamma = 4.0f32;
    let weights: Vec<f32> = (0..b).map(|k| (-(k as f32) / gamma).exp()).collect();
    let tgt_hiddens_g = up(&mut gpu, &tgt_hiddens, &[n_ctx, fc_in]);
    let targets_g = ui32(&mut gpu, &targets);
    let weights_g = up(&mut gpu, &weights, &[b]);

    // ---- Adam state (m, v = zeros, same shapes as net) ----
    let zeros_like = |gpu: &mut Gpu, src: &Net| -> Net {
        let mk = |gpu: &mut Gpu, t: &GpuTensor| gpu.zeros(&[t.numel()], DType::F32).unwrap();
        let layers = src.layers.iter().map(|l| LW {
            op_norm: mk(gpu, &l.op_norm), ffn_norm: mk(gpu, &l.ffn_norm),
            in_proj: l.in_proj.as_ref().map(|t| mk(gpu, t)), conv_w: l.conv_w.as_ref().map(|t| mk(gpu, t)),
            out_proj: l.out_proj.as_ref().map(|t| mk(gpu, t)),
            wq: l.wq.as_ref().map(|t| mk(gpu, t)), wk: l.wk.as_ref().map(|t| mk(gpu, t)),
            wv: l.wv.as_ref().map(|t| mk(gpu, t)), wo: l.wo.as_ref().map(|t| mk(gpu, t)),
            q_norm: l.q_norm.as_ref().map(|t| mk(gpu, t)), k_norm: l.k_norm.as_ref().map(|t| mk(gpu, t)),
            w1: mk(gpu, &l.w1), w3: mk(gpu, &l.w3), w2: mk(gpu, &l.w2),
        }).collect();
        Net { layers, in_proj_v: mk(gpu, &src.in_proj_v), out_proj_v: mk(gpu, &src.out_proj_v), fc: mk(gpu, &src.fc), final_norm: mk(gpu, &src.final_norm) }
    };
    let m_state = zeros_like(&mut gpu, &net);
    let v_state = zeros_like(&mut gpu, &net);

    // ---- training loop (overfit) ----
    let (lr, b1, b2, eps_a, wd) = (5e-3f32, 0.9f32, 0.999f32, 1e-8f32, 0.0f32);
    let steps = 400usize;
    println!("overfit: vocab={vocab} B={b} n_ctx={n_ctx}; ln(vocab)={:.3}", (vocab as f32).ln());
    let mut first_loss = 0f32;
    for step in 1..=steps {
        // embed gather (host)
        let mut embed_in = vec![0f32; b * d_tgt];
        for i in 0..b { let tk = block_tokens[i] as usize; embed_in[i * d_tgt..(i + 1) * d_tgt].copy_from_slice(&embed[tk * d_tgt..(tk + 1) * d_tgt]); }
        let embed_in_g = up(&mut gpu, &embed_in, &[b, d_tgt]);
        // forward
        let body_in = lin(&mut gpu, &embed_in_g, &net.in_proj_v, b, d_tgt, d);
        let ctx = lin(&mut gpu, &tgt_hiddens_g, &net.fc, n_ctx, fc_in, d);
        let (body_out, tape) = body_forward(&mut gpu, &cfg, &net.layers, &body_in, &ctx, &block_pos_g, &full_pos_g, &conv_state, b, n_ctx);
        let fn2 = gpu.zeros(&[b, d], DType::F32).unwrap();
        gpu.rmsnorm_batched(&body_out, &net.final_norm, &fn2, b, d, cfg.eps).unwrap();
        let out = lin(&mut gpu, &fn2, &net.out_proj_v, b, d, d_tgt);
        let logits = lin(&mut gpu, &out, &lm_head_g, b, d_tgt, vocab);
        // loss + dlogits
        let dlogits = gpu.zeros(&[b, vocab], DType::F32).unwrap();
        let loss_t = gpu.zeros(&[b], DType::F32).unwrap();
        gpu.ce_loss_bwd_f32(&logits, &targets_g, &weights_g, &dlogits, &loss_t, b, vocab).unwrap();
        let loss_v = gpu.download_f32(&loss_t).unwrap();
        let loss: f32 = loss_v.iter().sum::<f32>() / weights.iter().sum::<f32>();
        if step == 1 { first_loss = loss; }
        if step % 40 == 0 || step == 1 { println!("  step {step:4}  loss = {loss:.5}"); }
        // backward (head)
        let d_out = lin_dx(&mut gpu, &dlogits, &lm_head_g, b, d_tgt, vocab);
        let d_fn2 = lin_dx(&mut gpu, &d_out, &net.out_proj_v, b, d, d_tgt);
        let g_out_proj_v = lin_dw(&mut gpu, &d_out, &fn2, b, d, d_tgt);
        let d_body_out = zt(&mut gpu, b * d); let g_final_norm = zt(&mut gpu, d);
        gpu.rmsnorm_bwd_f32(&body_out, &net.final_norm, &d_fn2, &d_body_out, &g_final_norm, b, d, cfg.eps).unwrap();
        let (d_body_in, d_ctx, glayers) = body_backward(&mut gpu, &cfg, &net.layers, &tape, &d_body_out, &ctx, &block_pos_g, &full_pos_g, &conv_state, b, n_ctx);
        let g_in_proj_v = lin_dw(&mut gpu, &d_body_in, &embed_in_g, b, d_tgt, d);
        let g_fc = lin_dw(&mut gpu, &d_ctx, &tgt_hiddens_g, n_ctx, fc_in, d);
        let grad = Net { layers: glayers, in_proj_v: g_in_proj_v, out_proj_v: g_out_proj_v, fc: g_fc, final_norm: g_final_norm };

        // Adam step over all params
        let bc1 = 1.0 / (1.0 - b1.powi(step as i32));
        let bc2 = 1.0 / (1.0 - b2.powi(step as i32));
        let ps = net_tensors(&net); let gs = net_tensors(&grad); let ms = net_tensors(&m_state); let vs = net_tensors(&v_state);
        for i in 0..ps.len() {
            let n = ps[i].numel();
            gpu.adam_step_f32(ps[i], gs[i], ms[i], vs[i], lr, b1, b2, eps_a, wd, bc1, bc2, n).unwrap();
        }
    }
    // final loss
    let mut embed_in = vec![0f32; b * d_tgt];
    for i in 0..b { let tk = block_tokens[i] as usize; embed_in[i * d_tgt..(i + 1) * d_tgt].copy_from_slice(&embed[tk * d_tgt..(tk + 1) * d_tgt]); }
    let embed_in_g = up(&mut gpu, &embed_in, &[b, d_tgt]);
    let body_in = lin(&mut gpu, &embed_in_g, &net.in_proj_v, b, d_tgt, d);
    let ctx = lin(&mut gpu, &tgt_hiddens_g, &net.fc, n_ctx, fc_in, d);
    let (body_out, _tp) = body_forward(&mut gpu, &cfg, &net.layers, &body_in, &ctx, &block_pos_g, &full_pos_g, &conv_state, b, n_ctx);
    let fn2 = gpu.zeros(&[b, d], DType::F32).unwrap();
    gpu.rmsnorm_batched(&body_out, &net.final_norm, &fn2, b, d, cfg.eps).unwrap();
    let out = lin(&mut gpu, &fn2, &net.out_proj_v, b, d, d_tgt);
    let logits = lin(&mut gpu, &out, &lm_head_g, b, d_tgt, vocab);
    let lg = gpu.download_f32(&logits).unwrap();
    let mut correct = 0;
    for i in 0..b {
        let row = &lg[i * vocab..(i + 1) * vocab];
        let am = row.iter().enumerate().max_by(|a, c| a.1.partial_cmp(c.1).unwrap()).unwrap().0;
        if am as i32 == targets[i] { correct += 1; }
    }
    let dl = gpu.zeros(&[b, vocab], DType::F32).unwrap(); let lt = gpu.zeros(&[b], DType::F32).unwrap();
    gpu.ce_loss_bwd_f32(&logits, &targets_g, &weights_g, &dl, &lt, b, vocab).unwrap();
    let final_loss: f32 = gpu.download_f32(&lt).unwrap().iter().sum::<f32>() / weights.iter().sum::<f32>();
    println!("final loss = {final_loss:.5} (from {first_loss:.5}); argmax-correct {correct}/{b}");
    if final_loss < 0.05 && correct == b {
        println!("dflash_train_overfit: PASS (loss collapsed, batch memorized)");
    } else {
        println!("dflash_train_overfit: FAIL (loss {final_loss:.4}, correct {correct}/{b})");
        std::process::exit(1);
    }
}
