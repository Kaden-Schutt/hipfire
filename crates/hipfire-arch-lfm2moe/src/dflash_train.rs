// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! hipfire-native DFlash trainer (spike): block-parallel LFM2 draft body forward
//! with GQA target-context KV-injection + full hand-coded reverse-mode backward
//! + block-diffusion CE loss + AdamW, plus a bf16-safetensors warm-start loader.
//! All fp32. Kernels live in `kernels/src/{conv1d_gated_batched,dflash_train}.hip`
//! and are validated (per-kernel host-analytic, full-body gradcheck vs f64 oracle,
//! overfit loss collapse). See `docs/plans/dflash-lfm2-350m-spike.md`.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

#[derive(Clone)]
pub struct Cfg {
    pub d: usize,
    pub is_attn: Vec<bool>,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub conv_k: usize,
    pub inter: usize,
    pub theta: f32,
    pub eps: f32,
    pub d_tgt: usize,
    pub vocab: usize,
    pub n_tgt_layers: usize,
}
impl Cfg {
    pub fn n_layers(&self) -> usize { self.is_attn.len() }
    pub fn qd(&self) -> usize { self.nh * self.hd }
    pub fn kvd(&self) -> usize { self.nkv * self.hd }
    /// Real LFM2.5-350M-Base dims; `is_attn` per config.json layer_types.
    pub fn lfm2_350m(d_tgt: usize, vocab: usize, n_tgt_layers: usize) -> Self {
        let attn = [2usize, 5, 8, 10, 12, 14];
        Cfg {
            d: 1024,
            is_attn: (0..16).map(|i| attn.contains(&i)).collect(),
            nh: 16, nkv: 8, hd: 64, conv_k: 3, inter: 4608,
            theta: 1.0e6, eps: 1e-5, d_tgt, vocab, n_tgt_layers,
        }
    }
}

/// Per-layer trainable tensors (also holds per-layer GRADS — same shapes).
pub struct LW {
    pub op_norm: GpuTensor, pub ffn_norm: GpuTensor,
    pub in_proj: Option<GpuTensor>, pub conv_w: Option<GpuTensor>, pub out_proj: Option<GpuTensor>,
    pub wq: Option<GpuTensor>, pub wk: Option<GpuTensor>, pub wv: Option<GpuTensor>, pub wo: Option<GpuTensor>,
    pub q_norm: Option<GpuTensor>, pub k_norm: Option<GpuTensor>,
    pub w_c: Option<GpuTensor>, // conv-gate-injection [d,d] (conv layers; None = off)
    pub w1: GpuTensor, pub w3: GpuTensor, pub w2: GpuTensor,
}
/// Whole trainable net (body layers + vocab/context adapters). Reused for
/// weights, grads, and Adam m/v (all same shape) via `tensors()`.
pub struct Net {
    pub layers: Vec<LW>,
    pub in_proj_v: GpuTensor,  // [d, d_tgt]
    pub out_proj_v: GpuTensor, // [d_tgt, d]
    pub fc: GpuTensor,         // [d, n_tgt_layers*d_tgt]
    pub final_norm: GpuTensor, // [d]
}
pub fn lw_tensors(lw: &LW) -> Vec<&GpuTensor> {
    let mut v = vec![&lw.op_norm, &lw.ffn_norm];
    for o in [&lw.in_proj, &lw.conv_w, &lw.out_proj, &lw.wq, &lw.wk, &lw.wv, &lw.wo, &lw.q_norm, &lw.k_norm, &lw.w_c] {
        if let Some(t) = o { v.push(t); }
    }
    v.push(&lw.w1); v.push(&lw.w3); v.push(&lw.w2);
    v
}
pub fn net_tensors(net: &Net) -> Vec<&GpuTensor> {
    let mut v = Vec::new();
    for l in &net.layers { v.extend(lw_tensors(l)); }
    v.push(&net.in_proj_v); v.push(&net.out_proj_v); v.push(&net.fc); v.push(&net.final_norm);
    v
}

#[derive(Default)]
pub struct LT {
    pub h_in: Option<GpuTensor>, pub xn: Option<GpuTensor>, pub h_mid: Option<GpuTensor>, pub fnorm: Option<GpuTensor>,
    pub g: Option<GpuTensor>, pub u: Option<GpuTensor>, pub act: Option<GpuTensor>,
    pub bcx: Option<GpuTensor>, pub cy: Option<GpuTensor>,
    pub q0: Option<GpuTensor>, pub kfull0: Option<GpuTensor>, pub qr: Option<GpuTensor>, pub kr: Option<GpuTensor>,
    pub vfull: Option<GpuTensor>, pub attn_out: Option<GpuTensor>,
}

// ---- small GPU helpers ----
pub fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }
pub fn zt(gpu: &mut Gpu, n: usize) -> GpuTensor { gpu.zeros(&[n], DType::F32).unwrap() }
pub fn lin(gpu: &mut Gpu, x: &GpuTensor, w: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let y = gpu.zeros(&[m, n], DType::F32).unwrap(); gpu.linear_fwd_f32(x, w, &y, m, k, n).unwrap(); y
}
pub fn lin_dx(gpu: &mut Gpu, dy: &GpuTensor, w: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let dx = gpu.zeros(&[m, k], DType::F32).unwrap();
    let dx_mfma = std::env::var("HIPFIRE_DFLASH_DX_MFMA").ok().as_deref() == Some("1");
    if dflash_use_mfma() && dx_mfma && m <= 16 {
        gpu.linear_bwd_dx_mfma_f32(dy, w, &dx, m, k, n).unwrap();
    } else {
        gpu.linear_bwd_dx_f32(dy, w, &dx, m, k, n).unwrap();
    }
    dx
}
/// dW backward dispatch: MFMA path (gfx942) when HIPFIRE_DFLASH_MFMA=1 and the
/// batch (reduction) fits the 16-wide MFMA k-dim; else the portable naive kernel.
pub fn dflash_use_mfma() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("HIPFIRE_DFLASH_MFMA").ok().as_deref() == Some("1"))
}
pub fn lin_dw(gpu: &mut Gpu, dy: &GpuTensor, x: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
    let dw = gpu.zeros(&[n, k], DType::F32).unwrap();
    if dflash_use_mfma() {
        gpu.linear_bwd_dw_mfma_f32(dy, x, &dw, m, k, n).unwrap();
    } else {
        gpu.linear_bwd_dw_f32(dy, x, &dw, m, k, n).unwrap();
    }
    dw
}
pub fn addv(gpu: &mut Gpu, a: &GpuTensor, b: &GpuTensor) { gpu.add_inplace_f32(a, b).unwrap(); }
pub fn add_new(gpu: &mut Gpu, a: &GpuTensor, b: &GpuTensor, n: usize) -> GpuTensor {
    let c = gpu.zeros(&[n], DType::F32).unwrap(); gpu.add_f32(a, b, &c).unwrap(); c
}
pub fn dup(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> GpuTensor {
    let c = gpu.zeros(&[n], DType::F32).unwrap(); gpu.add_inplace_f32(&c, t).unwrap(); c
}
pub fn upos(gpu: &mut Gpu, pos: &[i32]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[pos.len()], DType::F32).unwrap();
    let bytes: Vec<u8> = pos.iter().flat_map(|p| p.to_le_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).unwrap(); t
}
pub fn ui32(gpu: &mut Gpu, vals: &[i32]) -> GpuTensor { upos(gpu, vals) }

// ---- bf16 safetensors reader (self-contained) ----
pub struct StFile {
    bytes: Vec<u8>,
    // name -> (dtype, shape, data_start, data_end) absolute byte offsets
    map: std::collections::HashMap<String, (String, Vec<usize>, usize, usize)>,
}
impl StFile {
    pub fn open(path: &Path) -> std::io::Result<StFile> {
        let bytes = std::fs::read(path)?;
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let base = 8 + hlen;
        let mut map = std::collections::HashMap::new();
        if let serde_json::Value::Object(obj) = hdr {
            for (name, meta) in obj {
                if name == "__metadata__" { continue; }
                let dtype = meta["dtype"].as_str().unwrap_or("BF16").to_string();
                let shape: Vec<usize> = meta["shape"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()).unwrap_or_default();
                let off = meta["data_offsets"].as_array().unwrap();
                let s = off[0].as_u64().unwrap() as usize;
                let e = off[1].as_u64().unwrap() as usize;
                map.insert(name, (dtype, shape, base + s, base + e));
            }
        }
        Ok(StFile { bytes, map })
    }
    pub fn has(&self, name: &str) -> bool { self.map.contains_key(name) }
    /// Read a tensor by name as f32 (supports BF16 / F16 / F32).
    pub fn f32(&self, name: &str) -> Vec<f32> {
        let (dtype, _shape, s, e) = self.map.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        let raw = &self.bytes[*s..*e];
        match dtype.as_str() {
            "BF16" => raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
            "F16" => raw.chunks_exact(2).map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            d => panic!("unsupported dtype {d} for {name}"),
        }
    }
}
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1; let exp = (h >> 10) & 0x1f; let mant = h & 0x3ff;
    let f = if exp == 0 {
        if mant == 0 { (sign as u32) << 31 } else {
            let mut e = -14i32; let mut m = mant as u32;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            m &= 0x3ff; (((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)) as u32
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(f)
}

// ---- LFM2.5-350M warm-start loader ----
/// Load the real LFM2.5-350M body weights (bf16 safetensors) into a `Vec<LW>`
/// at fp32, plus the `embedding_norm` warm-start for `final_norm`. Adapters
/// (in_proj_v / out_proj_v / fc) are NOT loaded (fresh-init by the caller).
pub fn load_lfm2_warmstart(gpu: &mut Gpu, cfg: &Cfg, st_path: &Path) -> std::io::Result<(Vec<LW>, GpuTensor)> {
    let st = StFile::open(st_path)?;
    let g = |gpu: &mut Gpu, st: &StFile, name: &str, sh: &[usize]| -> GpuTensor { up(gpu, &st.f32(name), sh) };
    let (d, qd, kvd, k) = (cfg.d, cfg.qd(), cfg.kvd(), cfg.conv_k);
    let mut layers = Vec::new();
    for li in 0..cfg.n_layers() {
        let p = format!("model.layers.{li}");
        let op_norm = g(gpu, &st, &format!("{p}.operator_norm.weight"), &[d]);
        let ffn_norm = g(gpu, &st, &format!("{p}.ffn_norm.weight"), &[d]);
        let w1 = g(gpu, &st, &format!("{p}.feed_forward.w1.weight"), &[cfg.inter, d]);
        let w3 = g(gpu, &st, &format!("{p}.feed_forward.w3.weight"), &[cfg.inter, d]);
        let w2 = g(gpu, &st, &format!("{p}.feed_forward.w2.weight"), &[d, cfg.inter]);
        let lw = if cfg.is_attn[li] {
            LW {
                op_norm, ffn_norm, in_proj: None, conv_w: None, out_proj: None,
                wq: Some(g(gpu, &st, &format!("{p}.self_attn.q_proj.weight"), &[qd, d])),
                wk: Some(g(gpu, &st, &format!("{p}.self_attn.k_proj.weight"), &[kvd, d])),
                wv: Some(g(gpu, &st, &format!("{p}.self_attn.v_proj.weight"), &[kvd, d])),
                wo: Some(g(gpu, &st, &format!("{p}.self_attn.out_proj.weight"), &[d, qd])),
                q_norm: Some(g(gpu, &st, &format!("{p}.self_attn.q_layernorm.weight"), &[cfg.hd])),
                k_norm: Some(g(gpu, &st, &format!("{p}.self_attn.k_layernorm.weight"), &[cfg.hd])),
                w_c: None,
                w1, w3, w2,
            }
        } else {
            LW {
                op_norm, ffn_norm,
                in_proj: Some(g(gpu, &st, &format!("{p}.conv.in_proj.weight"), &[3 * d, d])),
                conv_w: Some(g(gpu, &st, &format!("{p}.conv.conv.weight"), &[d, k])), // [d,1,k] flat == [d,k]
                out_proj: Some(g(gpu, &st, &format!("{p}.conv.out_proj.weight"), &[d, d])),
                wq: None, wk: None, wv: None, wo: None, q_norm: None, k_norm: None,
                w_c: None,
                w1, w3, w2,
            }
        };
        layers.push(lw);
    }
    let final_norm = g(gpu, &st, "model.embedding_norm.weight", &[d]);
    Ok((layers, final_norm))
}

// ---------------- body forward (saves tape) ----------------
#[allow(clippy::too_many_arguments)]
pub fn body_forward(gpu: &mut Gpu, cfg: &Cfg, w: &[LW], h_in0: &GpuTensor, ctx: &GpuTensor,
    block_pos: &GpuTensor, full_pos: &GpuTensor, conv_state: &GpuTensor, b: usize, n_ctx: usize) -> (GpuTensor, Vec<LT>) {
    let (d, nh, nkv, hd, qd, kvd) = (cfg.d, cfg.nh, cfg.nkv, cfg.hd, cfg.qd(), cfg.kvd());
    let l = n_ctx + b; let scale = 1.0 / (hd as f32).sqrt();
    let mut tape = Vec::new();
    // conv-gate-injection: pool the target context (SUM) once if any conv
    // layer carries W_c. inj = W_c . ctx_pooled added into the conv C_gate.
    let ctx_pooled = if n_ctx > 0 && w.iter().any(|l| l.w_c.is_some()) {
        let cp = gpu.zeros(&[d], DType::F32).unwrap();
        gpu.colsum_strided_f32(ctx, &cp, n_ctx, d, 0, d, 1.0).unwrap();
        Some(cp)
    } else { None };
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
            if let (Some(wc), Some(cp)) = (lw.w_c.as_ref(), ctx_pooled.as_ref()) {
                let inj = lin(gpu, cp, wc, 1, d, d); // [1,d] = W_c . ctx_pooled
                gpu.cgate_add_f32(&bcx, &inj, b, d).unwrap();
            }
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
pub fn body_backward(gpu: &mut Gpu, cfg: &Cfg, w: &[LW], tape: &[LT], dh_out: &GpuTensor, ctx: &GpuTensor,
    block_pos: &GpuTensor, full_pos: &GpuTensor, conv_state: &GpuTensor, b: usize, n_ctx: usize)
    -> (GpuTensor, GpuTensor, Vec<LW>) {
    let (d, nh, nkv, hd, qd, kvd) = (cfg.d, cfg.nh, cfg.nkv, cfg.hd, cfg.qd(), cfg.kvd());
    let l = n_ctx + b; let scale = 1.0 / (hd as f32).sqrt(); let nl = cfg.n_layers();
    let mut glayers: Vec<Option<LW>> = (0..nl).map(|_| None).collect();
    let d_ctx = zt(gpu, n_ctx.max(1) * d);
    let conv_inject = n_ctx > 0 && w.iter().any(|l| l.w_c.is_some());
    let ctx_pooled = if conv_inject {
        let cp = gpu.zeros(&[d], DType::F32).unwrap();
        gpu.colsum_strided_f32(ctx, &cp, n_ctx, d, 0, d, 1.0).unwrap();
        Some(cp)
    } else { None };
    let d_ctx_pooled = if conv_inject { Some(zt(gpu, d)) } else { None };
    let mut dh = dup(gpu, dh_out, b * d);
    for li in (0..nl).rev() {
        let lw = &w[li]; let t = &tape[li];
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
        let d_xn;
        let (mut g_inproj, mut g_convw, mut g_outproj, mut g_wc) = (None, None, None, None);
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
            // conv-gate-injection backward: d_inj = colsum of d_bcx C_gate slice
            if let (Some(wc), Some(cp), Some(dcp)) = (lw.w_c.as_ref(), ctx_pooled.as_ref(), d_ctx_pooled.as_ref()) {
                let d_inj = zt(gpu, d);
                gpu.colsum_strided_f32(&d_bcx, &d_inj, b, 3 * d, d, d, 1.0).unwrap();
                let dwc = lin_dw(gpu, &d_inj, cp, 1, d, d); // [d,d]
                let dcp_l = lin_dx(gpu, &d_inj, wc, 1, d, d); // [1,d]
                addv(gpu, dcp, &dcp_l);
                g_wc = Some(dwc);
            }
            g_inproj = Some(dinproj); g_convw = Some(d_conv_w); g_outproj = Some(dwout);
        }
        let dhin_n = zt(gpu, b * d); let dop_norm = zt(gpu, d);
        gpu.rmsnorm_bwd_f32(t.h_in.as_ref().unwrap(), &lw.op_norm, &d_xn, &dhin_n, &dop_norm, b, d, cfg.eps).unwrap();
        let dh_in = add_new(gpu, &dh_mid, &dhin_n, b * d);
        glayers[li] = Some(LW {
            op_norm: dop_norm, ffn_norm: dffn_norm,
            in_proj: g_inproj, conv_w: g_convw, out_proj: g_outproj,
            wq: g_wq, wk: g_wk, wv: g_wv, wo: g_wo, q_norm: g_qn, k_norm: g_kn,
            w_c: g_wc,
            w1: dw1, w3: dw3, w2: dw2,
        });
        dh = dh_in;
    }
    if let Some(dcp) = d_ctx_pooled.as_ref() {
        gpu.bias_add_f32(&d_ctx, dcp, n_ctx, d).unwrap();
    }
    (dh, d_ctx, glayers.into_iter().map(|o| o.unwrap()).collect())
}

// ---- DFNET checkpoint loader (mirror of dflash_train_run::save_net) ----
/// Load a trained drafter checkpoint (DFNET container) into a `Net`. Shapes are
/// derived from `cfg` + tensor name. Pairs with the saver in
/// `examples/dflash_train_run.rs`.
pub fn load_net(gpu: &mut Gpu, cfg: &Cfg, path: &Path) -> std::io::Result<Net> {
    use std::collections::HashMap;
    let bytes = std::fs::read(path)?;
    assert_eq!(&bytes[0..8], b"DFNET\0\0\0", "bad DFNET magic");
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    // 9 cfg ints follow (d,n_layers,nh,nkv,hd,conv_k,inter,d_tgt,vocab) — skip.
    let mut off = 12 + 9 * 4;
    let mut map: HashMap<String, Vec<f32>> = HashMap::new();
    for _ in 0..n {
        let nl = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize; off += 4;
        let name = String::from_utf8(bytes[off..off + nl].to_vec()).unwrap(); off += nl;
        let numel = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize; off += 4;
        let data: Vec<f32> = bytes[off..off + numel * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        off += numel * 4;
        map.insert(name, data);
    }
    let (d, qd, kvd, k, inter, d_tgt) = (cfg.d, cfg.qd(), cfg.kvd(), cfg.conv_k, cfg.inter, cfg.d_tgt);
    let fc_in = cfg.n_tgt_layers * d_tgt;
    let g = |gpu: &mut Gpu, map: &HashMap<String, Vec<f32>>, name: &str, sh: &[usize]| -> GpuTensor {
        let v = map.get(name).unwrap_or_else(|| panic!("DFNET missing {name}"));
        up(gpu, v, sh)
    };
    let mut layers = Vec::new();
    for li in 0..cfg.n_layers() {
        let pfx = format!("layers.{li}");
        let op_norm = g(gpu, &map, &format!("{pfx}.op_norm"), &[d]);
        let ffn_norm = g(gpu, &map, &format!("{pfx}.ffn_norm"), &[d]);
        let w1 = g(gpu, &map, &format!("{pfx}.w1"), &[inter, d]);
        let w3 = g(gpu, &map, &format!("{pfx}.w3"), &[inter, d]);
        let w2 = g(gpu, &map, &format!("{pfx}.w2"), &[d, inter]);
        let lw = if cfg.is_attn[li] {
            LW { op_norm, ffn_norm, in_proj: None, conv_w: None, out_proj: None,
                wq: Some(g(gpu, &map, &format!("{pfx}.wq"), &[qd, d])),
                wk: Some(g(gpu, &map, &format!("{pfx}.wk"), &[kvd, d])),
                wv: Some(g(gpu, &map, &format!("{pfx}.wv"), &[kvd, d])),
                wo: Some(g(gpu, &map, &format!("{pfx}.wo"), &[d, qd])),
                q_norm: Some(g(gpu, &map, &format!("{pfx}.q_norm"), &[cfg.hd])),
                k_norm: Some(g(gpu, &map, &format!("{pfx}.k_norm"), &[cfg.hd])),
                w_c: None, w1, w3, w2 }
        } else {
            LW { op_norm, ffn_norm,
                in_proj: Some(g(gpu, &map, &format!("{pfx}.in_proj"), &[3 * d, d])),
                conv_w: Some(g(gpu, &map, &format!("{pfx}.conv_w"), &[d, k])),
                out_proj: Some(g(gpu, &map, &format!("{pfx}.out_proj"), &[d, d])),
                wq: None, wk: None, wv: None, wo: None, q_norm: None, k_norm: None,
                w_c: { let nm = format!("{pfx}.w_c"); if map.contains_key(&nm) { Some(g(gpu, &map, &nm, &[d, d])) } else { None } },
                w1, w3, w2 }
        };
        layers.push(lw);
    }
    Ok(Net {
        layers,
        in_proj_v: g(gpu, &map, "in_proj_v", &[d, d_tgt]),
        out_proj_v: g(gpu, &map, "out_proj_v", &[d_tgt, d]),
        fc: g(gpu, &map, "fc", &[d, fc_in]),
        final_norm: g(gpu, &map, "final_norm", &[d]),
    })
}
