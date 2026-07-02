// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dispatch wrappers for the ZAYA1 CCA + EDA/MoD custom kernels
//! (`kernels/src/zaya_cca.hip`). All are register-only (no LDS). Faithful to
//! `crates/hipfire-arch-zaya/src/cpu.rs`.

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

const BLOCK: u32 = 256;

impl Gpu {
    fn zaya_ensure(&mut self, func: &str) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("zaya_cca", kernels::ZAYA_CCA_SRC, func)
    }

    fn zaya_launch(
        &self,
        func: &str,
        threads: usize,
        params: &mut Vec<*mut c_void>,
    ) -> HipResult<()> {
        let grid = (threads as u32).div_ceil(BLOCK);
        let f = &self.functions[func];
        unsafe {
            self.hip
                .launch_kernel(f, [grid, 1, 1], [BLOCK, 1, 1], 0, self.stream_ref(), params)
        }
    }

    /// Embedding gather: `out[t,i] = embed[ids[t]*hidden + i]`. `ids` is an i32
    /// device buffer of length `s`; `n = s*hidden`.
    pub fn zaya_embed_gather_f32(
        &mut self,
        out: &GpuTensor,
        embed: &GpuTensor,
        ids: &GpuTensor,
        hidden: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_embed_gather_f32")?;
        let (op, ep, ip) = (out.buf.as_ptr(), embed.buf.as_ptr(), ids.buf.as_ptr());
        let (hi, ni) = (hidden as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_embed_gather_f32", n, &mut p)
    }

    /// Broadcast bias add: `x[i] += bias[i % d]`, `n = s*d`.
    pub fn zaya_bias_add_f32(
        &mut self,
        x: &GpuTensor,
        bias: &GpuTensor,
        d: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_bias_add_f32")?;
        let (xp, bp) = (x.buf.as_ptr(), bias.buf.as_ptr());
        let (di, ni) = (d as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_bias_add_f32", n, &mut p)
    }

    /// Global input residual affine on the embeddings:
    /// `out[i] = (x[i]+bias[c])*scale[c]`, `c = i % d`, `n = seq*d`.
    pub fn zaya_affine_input_f32(
        &mut self,
        out: &GpuTensor,
        x: &GpuTensor,
        scale: &GpuTensor,
        bias: &GpuTensor,
        d: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_affine_input_f32")?;
        let (op, xp, sp, bp) = (
            out.buf.as_ptr(),
            x.buf.as_ptr(),
            scale.buf.as_ptr(),
            bias.buf.as_ptr(),
        );
        let (di, ni) = (d as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_affine_input_f32", n, &mut p)
    }

    /// Learned residual merge:
    /// `out[i] = (h[i]+hb[c])*hs[c] + (res[i]+rb[c])*rs[c]`, `c = i % d`.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_affine_residual_f32(
        &mut self,
        out: &GpuTensor,
        h: &GpuTensor,
        res: &GpuTensor,
        hs: &GpuTensor,
        hb: &GpuTensor,
        rs: &GpuTensor,
        rb: &GpuTensor,
        d: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_affine_residual_f32")?;
        let (op, hp, rp) = (out.buf.as_ptr(), h.buf.as_ptr(), res.buf.as_ptr());
        let (hsp, hbp, rsp, rbp) = (
            hs.buf.as_ptr(),
            hb.buf.as_ptr(),
            rs.buf.as_ptr(),
            rb.buf.as_ptr(),
        );
        let (di, ni) = (d as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &hp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &hsp as *const _ as *mut c_void,
            &hbp as *const _ as *mut c_void,
            &rsp as *const _ as *mut c_void,
            &rbp as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_affine_residual_f32", n, &mut p)
    }

    /// Scatter q `[s,q_dim]` and k `[s,k_dim]` into a left-padded channel-major
    /// stream `[q_dim+k_dim, pad+s]` (pad region pre-zeroed by the caller).
    pub fn zaya_qk_stream_f32(
        &mut self,
        stream: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        s: usize,
        q_dim: usize,
        k_dim: usize,
        pad: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_qk_stream_f32")?;
        let (stp, qp, kp) = (stream.buf.as_ptr(), q.buf.as_ptr(), k.buf.as_ptr());
        let (si, qd, kd, pd) = (s as i32, q_dim as i32, k_dim as i32, pad as i32);
        let mut p: Vec<*mut c_void> = vec![
            &stp as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &qd as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &pd as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_qk_stream_f32", (q_dim + k_dim) * s, &mut p)
    }

    /// VALID causal grouped conv1d over channel-major input `[channels, in_len]`
    /// → `[channels, out_len]`, `out_len = in_len - kernel + 1`. weight layout
    /// `[channels, channels/groups, kernel]`.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_conv1d_valid_f32(
        &mut self,
        out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        channels: usize,
        groups: usize,
        kernel: usize,
        in_len: usize,
        out_len: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_conv1d_valid_f32")?;
        let (op, ip, wp, bp) = (
            out.buf.as_ptr(),
            input.buf.as_ptr(),
            weight.buf.as_ptr(),
            bias.buf.as_ptr(),
        );
        let (ch, gr, kn, il, ol) = (
            channels as i32,
            groups as i32,
            kernel as i32,
            in_len as i32,
            out_len as i32,
        );
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ch as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &kn as *const _ as *mut c_void,
            &il as *const _ as *mut c_void,
            &ol as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_conv1d_valid_f32", channels * out_len, &mut p)
    }

    /// q/k residual paths. `mode = 0` writes `query_res [s,nq,hd]`; `mode = 1`
    /// writes `key_res [s,nkv,hd]` (mean over groups of query_res).
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_qk_residual_f32(
        &mut self,
        query_res: &GpuTensor,
        key_res: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        s: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        mode: i32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_qk_residual_f32")?;
        let (qrp, krp, qp, kp) = (
            query_res.buf.as_ptr(),
            key_res.buf.as_ptr(),
            q.buf.as_ptr(),
            k.buf.as_ptr(),
        );
        let (si, nqi, nkvi, hdi) = (s as i32, nq as i32, nkv as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &qrp as *const _ as *mut c_void,
            &krp as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &mode as *const _ as *mut c_void,
        ];
        let threads = if mode == 0 { s * nq * hd } else { s * nkv * hd };
        self.zaya_launch("zaya_qk_residual_f32", threads, &mut p)
    }

    /// Add conv output (channel-major `[conv_ch, s]`) to a residual path, emit
    /// head-major `[s, heads, hd]`. `mode = 0` query (conv ch = head*hd+d),
    /// `mode = 1` key (conv ch = q_dim + head*hd+d).
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_add_conv_residual_f32(
        &mut self,
        out: &GpuTensor,
        conv: &GpuTensor,
        res: &GpuTensor,
        s: usize,
        heads: usize,
        hd: usize,
        q_dim: usize,
        mode: i32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_add_conv_residual_f32")?;
        let (op, cp, rp) = (out.buf.as_ptr(), conv.buf.as_ptr(), res.buf.as_ptr());
        let (si, hi, hdi, qd) = (s as i32, heads as i32, hd as i32, q_dim as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &qd as *const _ as *mut c_void,
            &mode as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_add_conv_residual_f32", s * heads * hd, &mut p)
    }

    /// Compositional value: head 0 = current `v_cur[s,hd]`, head 1 = previous
    /// token's `v_del` (0 at t==0). Writes `value [s, nkv, hd]`.
    pub fn zaya_value_compose_f32(
        &mut self,
        value: &GpuTensor,
        v_cur: &GpuTensor,
        v_del: &GpuTensor,
        s: usize,
        nkv: usize,
        hd: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_value_compose_f32")?;
        let (vp, cp, dp) = (value.buf.as_ptr(), v_cur.buf.as_ptr(), v_del.buf.as_ptr());
        let (si, nkvi, hdi) = (s as i32, nkv as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &vp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_value_compose_f32", s * hd, &mut p)
    }

    /// Per `(token, head)` L2-normalize rows to `scale` (= sqrt(head_dim)); if
    /// `temp` is `Some`, multiply each head by `temp[head]` (key path).
    pub fn zaya_qk_l2norm_temp_f32(
        &mut self,
        x: &GpuTensor,
        temp: Option<&GpuTensor>,
        s: usize,
        heads: usize,
        hd: usize,
        scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_qk_l2norm_temp_f32")?;
        let xp = x.buf.as_ptr();
        // Dummy pointer (x) when no temp; `has_temp` gates the read.
        let tp = temp.map_or(xp, |t| t.buf.as_ptr());
        let has_temp: i32 = temp.is_some() as i32;
        let (si, hi, hdi) = (s as i32, heads as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &has_temp as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_qk_l2norm_temp_f32", s * heads, &mut p)
    }

    /// Exact erf-GELU in place over `n` elements.
    pub fn zaya_gelu_exact_f32(&mut self, x: &GpuTensor, n: usize) -> HipResult<()> {
        self.zaya_ensure("zaya_gelu_exact_f32")?;
        let xp = x.buf.as_ptr();
        let ni = n as i32;
        let mut p: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_gelu_exact_f32", n, &mut p)
    }

    /// Partial rotary (half-split) at position = `pos_base + token index`, over
    /// `x [s,heads,hd]`. `pos_base=0` for prefill; current pos for decode.
    pub fn zaya_rope_partial_f32(
        &mut self,
        x: &GpuTensor,
        s: usize,
        heads: usize,
        hd: usize,
        n_rot: usize,
        theta: f32,
        pos_base: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_rope_partial_f32")?;
        let xp = x.buf.as_ptr();
        let (si, hi, hdi, nr, pb) = (
            s as i32,
            heads as i32,
            hd as i32,
            n_rot as i32,
            pos_base as i32,
        );
        let mut p: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &hi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &nr as *const _ as *mut c_void,
            &theta as *const _ as *mut c_void,
            &pb as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_rope_partial_f32", s * heads, &mut p)
    }

    /// Strided row copy: `dst[r*dst_stride + j] = src[r*src_stride + src_off + j]`.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_strided_copy_f32(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        rows: usize,
        dst_stride: usize,
        src_stride: usize,
        src_off: usize,
        len: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_strided_copy_f32")?;
        let (dp, sp) = (dst.buf.as_ptr(), src.buf.as_ptr());
        let (r, ds, ss, so, l) = (
            rows as i32,
            dst_stride as i32,
            src_stride as i32,
            src_off as i32,
            len as i32,
        );
        let mut p: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &r as *const _ as *mut c_void,
            &ds as *const _ as *mut c_void,
            &ss as *const _ as *mut c_void,
            &so as *const _ as *mut c_void,
            &l as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_strided_copy_f32", rows * len, &mut p)
    }

    /// Decode conv window build + ring advance: `window [conv_ch, pad+1]` = `[ring | cur]`.
    pub fn zaya_conv_window_f32(
        &mut self,
        window: &GpuTensor,
        ring: &GpuTensor,
        cur: &GpuTensor,
        conv_ch: usize,
        pad: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_conv_window_f32")?;
        let (wp, rp, cp) = (window.buf.as_ptr(), ring.buf.as_ptr(), cur.buf.as_ptr());
        let (cc, pd) = (conv_ch as i32, pad as i32);
        let mut p: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &cc as *const _ as *mut c_void,
            &pd as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_conv_window_f32", conv_ch, &mut p)
    }

    /// Copy `src[0..n]` into `dst[offset..offset+n]` (KV-cache append).
    pub fn zaya_write_at_f32(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        offset: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_write_at_f32")?;
        let (dp, sp) = (dst.buf.as_ptr(), src.buf.as_ptr());
        let (off, ni) = (offset as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &off as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_write_at_f32", n, &mut p)
    }

    /// Single-token GQA decode attention over the KV cache (positions 0..=pos).
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_gqa_decode_f32(
        &mut self,
        out: &GpuTensor,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        pos: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        scaling: f32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_gqa_decode_f32")?;
        let (op, qp, kp, vp) = (
            out.buf.as_ptr(),
            q.buf.as_ptr(),
            k_cache.buf.as_ptr(),
            v_cache.buf.as_ptr(),
        );
        let (posi, nqi, nkvi, hdi) = (pos as i32, nq as i32, nkv as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &posi as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &scaling as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_gqa_decode_f32", nq, &mut p)
    }

    /// Simple GQA causal attention (bring-up). q `[s,nq,hd]`, k/v `[s,nkv,hd]`
    /// head-major; out `[s,nq,hd]`.
    #[allow(clippy::too_many_arguments)]
    pub fn zaya_gqa_attn_f32(
        &mut self,
        out: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        s: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        scaling: f32,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_gqa_attn_f32")?;
        let (op, qp, kp, vp) = (
            out.buf.as_ptr(),
            q.buf.as_ptr(),
            k.buf.as_ptr(),
            v.buf.as_ptr(),
        );
        let (si, nqi, nkvi, hdi) = (s as i32, nq as i32, nkv as i32, hd as i32);
        let mut p: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &si as *const _ as *mut c_void,
            &nqi as *const _ as *mut c_void,
            &nkvi as *const _ as *mut c_void,
            &hdi as *const _ as *mut c_void,
            &scaling as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_gqa_attn_f32", s * nq, &mut p)
    }

    /// EDA cross-layer state add: `router_hidden[i] += prev[i] * scale[i%rh]`.
    pub fn zaya_eda_add_f32(
        &mut self,
        router_hidden: &GpuTensor,
        prev: &GpuTensor,
        scale: &GpuTensor,
        rh: usize,
        n: usize,
    ) -> HipResult<()> {
        self.zaya_ensure("zaya_eda_add_f32")?;
        let (rp, pp, sp) = (
            router_hidden.buf.as_ptr(),
            prev.buf.as_ptr(),
            scale.buf.as_ptr(),
        );
        let (rhi, ni) = (rh as i32, n as i32);
        let mut p: Vec<*mut c_void> = vec![
            &rp as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &rhi as *const _ as *mut c_void,
            &ni as *const _ as *mut c_void,
        ];
        self.zaya_launch("zaya_eda_add_f32", n, &mut p)
    }
}
