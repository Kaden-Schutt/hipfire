// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Mamba-2 mixer **block** decode step — CPU reference (the N3 math oracle).
//!
//! Composes the four validated primitives into the full nemotron_h Mamba-2 mixer
//! forward for one token, so the GPU block forward (which chains the matching HIP
//! kernels: `gemv` → `conv1d_bias_silu_decode_f32` → `mamba2_ssd_decode_f32` →
//! `mamba2_gated_norm_f32` → `gemv`) can be validated gpu-vs-cpu against a
//! readable baseline. Decode-only (single token); chunked prefill lands at N6.
//!
//! Pipeline (confirmed from `modeling_nemotron_h.py`):
//! ```text
//!   proj = in_proj @ hidden              # [d_inner + conv_dim + num_heads]
//!   z, xBC, dt_raw = split(proj)         # gate | conv-input | dt (d_mlp=0)
//!   xBC = silu(conv1d(xBC) + conv_bias)  # depthwise causal K=4 over conv_dim
//!   x, B, C = split(xBC)                 # [d_inner] | [n_groups*ssm] | [..]
//!   y = ssd_decode(x, B, C, dt_raw)      # selective scan, updates ssm_state
//!   y = rmsnorm_gated(y, z)              # gate-then-group-RMSNorm, group=960
//!   out = out_proj @ y                   # [hidden_size]
//! ```

use crate::ssd::{ssd_decode_step, SsdParams};

/// Static shapes for one Mamba-2 mixer block.
#[derive(Clone, Debug)]
pub struct Mamba2Dims {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub state_size: usize,
    pub n_groups: usize,
    pub conv_kernel: usize,
    pub rms_norm_eps: f32,
    pub dt_min: f32,
    pub dt_max: f32,
}

impl Mamba2Dims {
    /// `num_heads * head_dim` — the SSM inner width (NOT expand*hidden).
    pub fn d_inner(&self) -> usize {
        self.num_heads * self.head_dim
    }
    /// `d_inner + 2 * n_groups * state_size` — the conv1d channel count (xBC).
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.n_groups * self.state_size
    }
    /// `d_inner + conv_dim + num_heads` — the in_proj output width.
    pub fn projection_size(&self) -> usize {
        self.d_inner() + self.conv_dim() + self.num_heads
    }
    /// `d_inner / n_groups` — the RMSNormGated group width (= 960 for Nano-4B).
    pub fn norm_group_size(&self) -> usize {
        self.d_inner() / self.n_groups
    }
}

/// Block weights (row-major, no proj bias — `mamba_proj_bias=false`).
pub struct Mamba2BlockWeights<'a> {
    /// `[projection_size, hidden_size]`.
    pub in_proj: &'a [f32],
    /// `[conv_dim, conv_kernel]` (per-channel filter, newest at `k=K-1`).
    pub conv_weight: &'a [f32],
    /// `[conv_dim]` (`use_conv_bias=true`).
    pub conv_bias: &'a [f32],
    /// Per-head `A_log`, `D`, `dt_bias` — each `[num_heads]`.
    pub a_log: &'a [f32],
    pub d: &'a [f32],
    pub dt_bias: &'a [f32],
    /// RMSNormGated weight `[d_inner]`.
    pub norm_weight: &'a [f32],
    /// `[hidden_size, d_inner]`.
    pub out_proj: &'a [f32],
}

/// Per-sequence recurrent state for one Mamba-2 block.
#[derive(Clone, Debug)]
pub struct Mamba2BlockState {
    /// `[conv_dim * (conv_kernel - 1)]` rolling conv history (newest-1 at t=0).
    pub conv_state: Vec<f32>,
    /// `[num_heads * head_dim * state_size]` SSM state.
    pub ssm_state: Vec<f32>,
}

impl Mamba2BlockState {
    pub fn zeros(dims: &Mamba2Dims) -> Self {
        Self {
            conv_state: vec![0.0; dims.conv_dim() * (dims.conv_kernel - 1)],
            ssm_state: vec![0.0; dims.num_heads * dims.head_dim * dims.state_size],
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Row-major matvec: `out[i] = Σ_j w[i*in + j] * x[j]`, `w` is `[out, in]`.
fn matvec(w: &[f32], x: &[f32], out: usize, n_in: usize, dst: &mut [f32]) {
    debug_assert_eq!(w.len(), out * n_in);
    debug_assert_eq!(x.len(), n_in);
    debug_assert_eq!(dst.len(), out);
    for i in 0..out {
        let row = &w[i * n_in..i * n_in + n_in];
        let mut acc = 0.0f32;
        for j in 0..n_in {
            acc += row[j] * x[j];
        }
        dst[i] = acc;
    }
}

/// One Mamba-2 mixer block decode step. Returns the `[hidden_size]` mixer output;
/// updates `state.conv_state` and `state.ssm_state` in place.
pub fn mamba2_block_decode_step(
    dims: &Mamba2Dims,
    w: &Mamba2BlockWeights,
    state: &mut Mamba2BlockState,
    hidden: &[f32],
) -> Vec<f32> {
    let d_inner = dims.d_inner();
    let conv_dim = dims.conv_dim();
    let num_heads = dims.num_heads;
    let nss = dims.n_groups * dims.state_size;
    debug_assert_eq!(hidden.len(), dims.hidden_size);

    // 1. in_proj → [z | xBC | dt]
    let mut proj = vec![0.0f32; dims.projection_size()];
    matvec(
        w.in_proj,
        hidden,
        dims.projection_size(),
        dims.hidden_size,
        &mut proj,
    );
    let z = &proj[0..d_inner];
    let xbc_in = &proj[d_inner..d_inner + conv_dim];
    let dt_raw = &proj[d_inner + conv_dim..d_inner + conv_dim + num_heads];

    // 2. depthwise causal conv1d (K) + bias + SiLU over conv_dim channels.
    let k = dims.conv_kernel;
    let hist = k - 1;
    let mut xbc_act = vec![0.0f32; conv_dim];
    for c in 0..conv_dim {
        let sbase = c * hist;
        let mut acc = w.conv_bias[c];
        // window = [history(hist) ..., current], weight[c*k + t], t newest at k-1.
        for t in 0..hist {
            acc += state.conv_state[sbase + t] * w.conv_weight[c * k + t];
        }
        acc += xbc_in[c] * w.conv_weight[c * k + (k - 1)];
        xbc_act[c] = silu(acc);
        // roll ring buffer: drop oldest, append current input.
        for t in 0..hist.saturating_sub(1) {
            state.conv_state[sbase + t] = state.conv_state[sbase + t + 1];
        }
        if hist > 0 {
            state.conv_state[sbase + (hist - 1)] = xbc_in[c];
        }
    }

    // 3. split activated xBC → x | B | C
    let x = &xbc_act[0..d_inner];
    let b = &xbc_act[d_inner..d_inner + nss];
    let c = &xbc_act[d_inner + nss..d_inner + 2 * nss];

    // 4. SSD selective scan → y [d_inner]
    let ssd_params = SsdParams {
        num_heads,
        head_dim: dims.head_dim,
        state_size: dims.state_size,
        n_groups: dims.n_groups,
        dt_min: dims.dt_min,
        dt_max: dims.dt_max,
        a_log: w.a_log.to_vec(),
        d: w.d.to_vec(),
        dt_bias: w.dt_bias.to_vec(),
    };
    let mut y = vec![0.0f32; d_inner];
    ssd_decode_step(&ssd_params, &mut state.ssm_state, x, b, c, dt_raw, &mut y);

    // 5. RMSNormGated: gate FIRST (y*silu(z)), then group-RMSNorm.
    let gs = dims.norm_group_size();
    let n_norm_groups = d_inner / gs;
    let mut y_norm = vec![0.0f32; d_inner];
    for g in 0..n_norm_groups {
        let base = g * gs;
        let mut ss = 0.0f32;
        for i in 0..gs {
            let gated = y[base + i] * silu(z[base + i]);
            ss += gated * gated;
        }
        let inv = 1.0f32 / (ss / gs as f32 + dims.rms_norm_eps).sqrt();
        for i in 0..gs {
            let gated = y[base + i] * silu(z[base + i]);
            y_norm[base + i] = gated * inv * w.norm_weight[base + i];
        }
    }

    // 6. out_proj → [hidden_size]
    let mut out = vec![0.0f32; dims.hidden_size];
    matvec(w.out_proj, &y_norm, dims.hidden_size, d_inner, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nano_like_dims() -> Mamba2Dims {
        // Tiny but structurally faithful: 2 heads × 3 head_dim, 4 state, 2 groups.
        Mamba2Dims {
            hidden_size: 5,
            num_heads: 2,
            head_dim: 3,
            state_size: 4,
            n_groups: 2,
            conv_kernel: 4,
            rms_norm_eps: 1e-5,
            dt_min: 0.0,
            dt_max: f32::INFINITY,
        }
    }

    #[test]
    fn dims_match_nano4b() {
        let d = Mamba2Dims {
            hidden_size: 3136,
            num_heads: 96,
            head_dim: 80,
            state_size: 128,
            n_groups: 8,
            conv_kernel: 4,
            rms_norm_eps: 1e-5,
            dt_min: 0.0,
            dt_max: f32::INFINITY,
        };
        assert_eq!(d.d_inner(), 7680);
        assert_eq!(d.conv_dim(), 9728);
        assert_eq!(d.projection_size(), 7680 + 9728 + 96);
        assert_eq!(d.norm_group_size(), 960);
    }

    #[test]
    fn zero_weights_give_zero_output() {
        let d = nano_like_dims();
        let mut st = Mamba2BlockState::zeros(&d);
        let w = Mamba2BlockWeights {
            in_proj: &vec![0.0; d.projection_size() * d.hidden_size],
            conv_weight: &vec![0.0; d.conv_dim() * d.conv_kernel],
            conv_bias: &vec![0.0; d.conv_dim()],
            a_log: &vec![0.0; d.num_heads],
            d: &vec![0.0; d.num_heads],
            dt_bias: &vec![0.0; d.num_heads],
            norm_weight: &vec![0.0; d.d_inner()],
            out_proj: &vec![0.0; d.hidden_size * d.d_inner()],
        };
        let out = mamba2_block_decode_step(&d, &w, &mut st, &vec![1.0; d.hidden_size]);
        assert!(out.iter().all(|&v| v == 0.0), "zero out_proj → zero output");
    }

    #[test]
    fn output_is_finite_and_state_advances() {
        let d = nano_like_dims();
        let mut st = Mamba2BlockState::zeros(&d);
        // deterministic pseudo-random weights
        let mut seed = 0x2545F491u32;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
        };
        let in_proj: Vec<f32> = (0..d.projection_size() * d.hidden_size)
            .map(|_| rng())
            .collect();
        let conv_weight: Vec<f32> = (0..d.conv_dim() * d.conv_kernel).map(|_| rng()).collect();
        let conv_bias: Vec<f32> = (0..d.conv_dim()).map(|_| rng()).collect();
        let a_log: Vec<f32> = (0..d.num_heads).map(|_| rng()).collect();
        let dd: Vec<f32> = (0..d.num_heads).map(|_| rng()).collect();
        let dt_bias: Vec<f32> = (0..d.num_heads).map(|_| rng()).collect();
        let norm_weight: Vec<f32> = (0..d.d_inner()).map(|_| 1.0 + rng()).collect();
        let out_proj: Vec<f32> = (0..d.hidden_size * d.d_inner()).map(|_| rng()).collect();
        let w = Mamba2BlockWeights {
            in_proj: &in_proj,
            conv_weight: &conv_weight,
            conv_bias: &conv_bias,
            a_log: &a_log,
            d: &dd,
            dt_bias: &dt_bias,
            norm_weight: &norm_weight,
            out_proj: &out_proj,
        };
        let hidden = vec![0.3; d.hidden_size];
        let out1 = mamba2_block_decode_step(&d, &w, &mut st, &hidden);
        assert!(out1.iter().all(|v| v.is_finite()));
        // SSM state should be non-zero after a step with non-trivial input.
        assert!(
            st.ssm_state.iter().any(|&s| s != 0.0),
            "ssm state must advance"
        );
        // a second step uses the advanced conv/ssm state (no panic, finite).
        let out2 = mamba2_block_decode_step(&d, &w, &mut st, &hidden);
        assert!(out2.iter().all(|v| v.is_finite()));
    }
}
