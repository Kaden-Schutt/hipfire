// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Mamba-2 SSD (selective state-space duality) recurrence — **CPU reference**.
//!
//! This is the math oracle for the GPU SSD kernel (N1): a small, exact, pure-Rust
//! implementation of one selective-scan *decode* step (single token), used in
//! gpu-vs-cpu tests so a sign/scaling error in the HIP kernel surfaces against a
//! readable baseline instead of as silent attractor garbage.
//!
//! Per head `h`, with per-head SSM state `state[head][p][n]`
//! (`p` ∈ head_dim, `n` ∈ state_size):
//! ```text
//!   dt   = softplus(dt_raw[head] + dt_bias[head])        # > 0
//!   dt   = clamp(dt, time_step_min, time_step_max)        # nemotron bounds
//!   A    = -exp(A_log[head])                              # < 0 (decay)
//!   dA   = exp(dt * A)                                    # ∈ (0, 1)
//!   h[p][n] = dA * h[p][n] + (dt * B[n]) * x[p]           # B⊗x outer product
//!   y[p]    = Σ_n C[n] * h[p][n] + D[head] * x[p]         # read-out + skip
//! ```
//! `B`/`C` are shared across the heads in a group (`n_groups`): head `h` uses
//! group `h / (num_heads / n_groups)`.
//!
//! **VERIFY vs HF** at N5: the exact `dt` activation (softplus + bias + the
//! `time_step_floor`/min/max clamp order) and the `A = -exp(A_log)` sign are the
//! Mamba-2 conventions; cross-check against `modeling_nemotron_h.py` /
//! `mamba_ssm` before trusting the kernel for coherence.

/// Shapes + per-head scalar params for one Mamba-2 mixer (decode step).
#[derive(Clone, Debug)]
pub struct SsdParams {
    pub num_heads: usize,
    pub head_dim: usize,
    pub state_size: usize,
    pub n_groups: usize,
    /// `time_step_min` / `time_step_max` clamp bounds on `dt`.
    pub dt_min: f32,
    pub dt_max: f32,
    /// Per-head `A_log` (so `A = -exp(A_log)`), length `num_heads`.
    pub a_log: Vec<f32>,
    /// Per-head `D` skip scalar, length `num_heads`.
    pub d: Vec<f32>,
    /// Per-head `dt_bias`, length `num_heads`.
    pub dt_bias: Vec<f32>,
}

impl SsdParams {
    /// B/C group for a head. HF expands groups with `B.repeat(num_heads //
    /// n_groups)` — a **tile**, so head `h` uses group `h % n_groups` (NOT the
    /// interleave `h / (num_heads/n_groups)`). Verified exact vs the HF dump.
    fn group_of(&self, head: usize) -> usize {
        head % self.n_groups
    }
}

#[inline]
fn softplus(x: f32) -> f32 {
    // numerically-stable log(1+exp(x))
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// One Mamba-2 SSD decode step (single token), in place on `state`.
///
/// - `state`: `[num_heads * head_dim * state_size]`, updated in place.
/// - `x`: `[num_heads * head_dim]` (the conv'd, SiLU'd input).
/// - `b`, `c`: `[n_groups * state_size]` (per-group B/C).
/// - `dt_raw`: `[num_heads]` (pre-activation).
/// - `y` (out): `[num_heads * head_dim]`.
pub fn ssd_decode_step(
    p: &SsdParams,
    state: &mut [f32],
    x: &[f32],
    b: &[f32],
    c: &[f32],
    dt_raw: &[f32],
    y: &mut [f32],
) {
    let (h, dh, n, g) = (p.num_heads, p.head_dim, p.state_size, p.n_groups);
    debug_assert_eq!(state.len(), h * dh * n);
    debug_assert_eq!(x.len(), h * dh);
    debug_assert_eq!(b.len(), g * n);
    debug_assert_eq!(c.len(), g * n);
    debug_assert_eq!(dt_raw.len(), h);
    debug_assert_eq!(y.len(), h * dh);

    for head in 0..h {
        let grp = p.group_of(head);
        let a = -(p.a_log[head].exp());
        let dt = {
            let v = softplus(dt_raw[head] + p.dt_bias[head]);
            v.clamp(p.dt_min, p.dt_max)
        };
        let da = (dt * a).exp();
        let b_grp = &b[grp * n..grp * n + n];
        let c_grp = &c[grp * n..grp * n + n];
        for pp in 0..dh {
            let xp = x[head * dh + pp];
            let dbx_scale = dt * xp;
            let srow = &mut state[(head * dh + pp) * n..(head * dh + pp) * n + n];
            let mut acc = 0.0f32;
            for nn in 0..n {
                // h = dA*h + (dt*B)*x
                srow[nn] = da * srow[nn] + dbx_scale * b_grp[nn];
                // y += C*h
                acc += c_grp[nn] * srow[nn];
            }
            y[head * dh + pp] = acc + p.d[head] * xp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(h: usize, dh: usize, n: usize, g: usize) -> SsdParams {
        SsdParams {
            num_heads: h,
            head_dim: dh,
            state_size: n,
            n_groups: g,
            dt_min: 0.0,
            dt_max: f32::INFINITY,
            a_log: vec![0.0; h],
            d: vec![0.0; h],
            dt_bias: vec![0.0; h],
        }
    }

    #[test]
    fn d_skip_only_when_state_dead() {
        // A_log -> very large so A very negative; but simpler: B=0 so no state
        // accumulation, D=1 -> y == x exactly (pure skip).
        let mut p = params(2, 3, 4, 1);
        p.d = vec![1.0, 1.0];
        let mut state = vec![0.0; 2 * 3 * 4];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![0.0; 4]; // B=0 -> no state input
        let c = vec![1.0; 4];
        let dt_raw = vec![0.0; 2];
        let mut y = vec![0.0; 6];
        ssd_decode_step(&p, &mut state, &x, &b, &c, &dt_raw, &mut y);
        assert_eq!(y, x, "B=0,D=1 -> y must equal x (skip only)");
        assert!(state.iter().all(|&s| s == 0.0), "B=0 -> state stays zero");
    }

    #[test]
    fn single_step_matches_hand_computation() {
        // 1 head, head_dim 1, state 2, 1 group. A_log=0 -> A=-1.
        // dt_raw=0, dt_bias=0 -> dt=softplus(0)=ln(2). da=exp(ln2 * -1)=0.5.
        // x=[2], B=[1,1], C=[1,1], D=0, state init [0,0].
        // h[n] = 0.5*0 + (ln2 * 2) * 1 = 2*ln2 each.
        // y = C·h = 2*(2*ln2) = 4*ln2.
        let p = params(1, 1, 2, 1);
        let mut state = vec![0.0; 2];
        let x = vec![2.0];
        let b = vec![1.0, 1.0];
        let c = vec![1.0, 1.0];
        let dt_raw = vec![0.0];
        let mut y = vec![0.0];
        ssd_decode_step(&p, &mut state, &x, &b, &c, &dt_raw, &mut y);
        let ln2 = 2.0f32.ln();
        let expected_h = ln2 * 2.0; // dt*B*x with dt=ln2, B=1, x=2
        assert!((state[0] - expected_h).abs() < 1e-5);
        assert!((state[1] - expected_h).abs() < 1e-5);
        assert!((y[0] - 4.0 * ln2).abs() < 1e-5);
    }

    #[test]
    fn decay_applies_on_second_step() {
        // Two steps with x=0 on the second: state should decay by da, y -> C·(da*h).
        let p = params(1, 1, 1, 1);
        let mut state = vec![0.0; 1];
        let b = vec![1.0];
        let c = vec![1.0];
        let mut y = vec![0.0];
        // step 1: x=1 -> h = dt*1*1 = ln2; da=0.5
        ssd_decode_step(&p, &mut state, &[1.0], &b, &c, &[0.0], &mut y);
        let ln2 = 2.0f32.ln();
        assert!((state[0] - ln2).abs() < 1e-5);
        // step 2: x=0 -> h = 0.5*ln2 + 0 ; y = h
        ssd_decode_step(&p, &mut state, &[0.0], &b, &c, &[0.0], &mut y);
        assert!((state[0] - 0.5 * ln2).abs() < 1e-5);
        assert!((y[0] - 0.5 * ln2).abs() < 1e-5);
    }
}
