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
    /// B/C group for a head — the **interleave** `h / (num_heads/n_groups)`,
    /// matching mamba-ssm's fast path (`mamba_chunk_scan_combined` /
    /// `selective_state_update`) and HF's *decode* path. (HF's *prefill*
    /// torch_forward uses `B.repeat` → `h % n_groups`, which is inconsistent
    /// with its own decode path — a torch-fallback bug; we follow the fast path
    /// since hipfire is a decode-style implementation and that's the trained
    /// convention.)
    fn group_of(&self, head: usize) -> usize {
        head / (self.num_heads / self.n_groups)
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

/// Sequential multi-token reference (N6): run the decode recurrence over
/// `seq_len` tokens, emitting every per-position output. This is the **ground
/// truth** the chunked prefill — both the CPU [`ssd_chunked`] decomposition and
/// the eventual GPU chunk-scan kernel — must match. `state` is updated to the
/// post-sequence recurrent state (so prefill can hand off to decode).
///
/// Layout: `x_seq` `[seq_len * num_heads*head_dim]`, `b_seq`/`c_seq`
/// `[seq_len * n_groups*state_size]`, `dt_seq` `[seq_len * num_heads]`,
/// `y_seq` (out) `[seq_len * num_heads*head_dim]` — all position-major.
#[allow(clippy::too_many_arguments)]
pub fn ssd_sequence(
    p: &SsdParams,
    state: &mut [f32],
    x_seq: &[f32],
    b_seq: &[f32],
    c_seq: &[f32],
    dt_seq: &[f32],
    y_seq: &mut [f32],
) {
    let (h, dh, n, g) = (p.num_heads, p.head_dim, p.state_size, p.n_groups);
    let xd = h * dh;
    let bd = g * n;
    let seq = x_seq.len() / xd;
    for t in 0..seq {
        ssd_decode_step(
            p,
            state,
            &x_seq[t * xd..t * xd + xd],
            &b_seq[t * bd..t * bd + bd],
            &c_seq[t * bd..t * bd + bd],
            &dt_seq[t * h..t * h + h],
            &mut y_seq[t * xd..t * xd + xd],
        );
    }
}

/// Chunked-SSD scan (N6) — the **parallel-friendly decomposition** of
/// [`ssd_sequence`], processing the sequence in chunks of `chunk_size`. Within a
/// chunk the recurrence unrolls into (per head, per `head_dim` channel `p`):
/// ```text
///   y_t[p] = exp(S_t)·(C_t · h_in[p])                      # inter-chunk (state)
///          + Σ_{s≤t} exp(S_t − S_s)·dt_s·(C_t · B_s)·x_s[p] # intra-chunk (L⊙G)
///          + D·x_t[p]                                       # skip
///   h_out[p][n] = exp(S_{L-1})·h_in[p][n]
///               + Σ_s exp(S_{L-1} − S_s)·dt_s·B_s[n]·x_s[p] # carry to next chunk
/// ```
/// where `S_t = Σ_{r≤t} dt_r·A` is the cumulative log-decay (A = −exp(A_log)).
/// Using `exp(S_t − S_s)` (a difference of cumulative SUMS) rather than a product
/// of `dA` keeps the lower-triangular decay matrix numerically stable for long
/// chunks. Mathematically identical to the sequential scan; f32 reassociation
/// makes it match within ~1e-4 (the gpu-vs-cpu validation bar). The GPU kernel
/// computes the `C·B` Gram and `L` decay matrices as matmuls; this CPU form is
/// the explicit double-loop oracle.
#[allow(clippy::too_many_arguments)]
pub fn ssd_chunked(
    p: &SsdParams,
    state: &mut [f32],
    x_seq: &[f32],
    b_seq: &[f32],
    c_seq: &[f32],
    dt_seq: &[f32],
    y_seq: &mut [f32],
    chunk_size: usize,
) {
    let (h, dh, n, g) = (p.num_heads, p.head_dim, p.state_size, p.n_groups);
    let xd = h * dh;
    let bd = g * n;
    let seq = x_seq.len() / xd;

    let mut t0 = 0;
    while t0 < seq {
        let chunk = chunk_size.min(seq - t0);
        for head in 0..h {
            let grp = p.group_of(head);
            let a = -(p.a_log[head].exp());
            // per-position dt and inclusive cumulative log-decay S_t = Σ_{r≤t} dt_r·A.
            let mut dt = vec![0.0f32; chunk];
            let mut s_cum = vec![0.0f32; chunk];
            let mut run = 0.0f32;
            for ti in 0..chunk {
                let v = softplus(dt_seq[(t0 + ti) * h + head] + p.dt_bias[head])
                    .clamp(p.dt_min, p.dt_max);
                dt[ti] = v;
                run += v * a;
                s_cum[ti] = run;
            }
            let s_end = s_cum[chunk - 1];
            for pp in 0..dh {
                let base = (head * dh + pp) * n;
                // h_in for this (head, channel) — read for all outputs, updated last.
                let h_in: Vec<f32> = state[base..base + n].to_vec();

                for ti in 0..chunk {
                    let c_t = &c_seq[(t0 + ti) * bd + grp * n..(t0 + ti) * bd + grp * n + n];
                    // inter-chunk state term: exp(S_t)·(C_t · h_in)
                    let mut ch = 0.0f32;
                    for nn in 0..n {
                        ch += c_t[nn] * h_in[nn];
                    }
                    let mut y = s_cum[ti].exp() * ch;
                    // intra-chunk: Σ_{s≤t} exp(S_t − S_s)·dt_s·(C_t·B_s)·x_s
                    for s in 0..=ti {
                        let l_ts = (s_cum[ti] - s_cum[s]).exp();
                        let b_s = &b_seq[(t0 + s) * bd + grp * n..(t0 + s) * bd + grp * n + n];
                        let mut cb = 0.0f32;
                        for nn in 0..n {
                            cb += c_t[nn] * b_s[nn];
                        }
                        y += l_ts * dt[s] * cb * x_seq[(t0 + s) * xd + head * dh + pp];
                    }
                    let x_t = x_seq[(t0 + ti) * xd + head * dh + pp];
                    y_seq[(t0 + ti) * xd + head * dh + pp] = y + p.d[head] * x_t;
                }

                // carry state to the next chunk: h_out = exp(S_end)·h_in + intra.
                let srow = &mut state[base..base + n];
                let decay_end = s_end.exp();
                for nn in 0..n {
                    srow[nn] = decay_end * h_in[nn];
                }
                for s in 0..chunk {
                    let l = (s_end - s_cum[s]).exp();
                    let b_s = &b_seq[(t0 + s) * bd + grp * n..(t0 + s) * bd + grp * n + n];
                    let coef = l * dt[s] * x_seq[(t0 + s) * xd + head * dh + pp];
                    for nn in 0..n {
                        srow[nn] += coef * b_s[nn];
                    }
                }
            }
        }
        t0 += chunk;
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

    // ── chunked-prefill (N6) equivalence: ssd_chunked == ssd_sequence ────────

    /// Deterministic pseudo-random fill in [-0.5, 0.5).
    fn fill(seed: &mut u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (*seed >> 8) as f32 / 16_777_216.0 - 0.5
            })
            .collect()
    }

    fn rand_params(seed: &mut u32, h: usize, dh: usize, n: usize, g: usize) -> SsdParams {
        // A_log small so A = -exp(A_log) is a sane decay; dt bounds match nemotron.
        SsdParams {
            num_heads: h,
            head_dim: dh,
            state_size: n,
            n_groups: g,
            dt_min: 0.0,
            dt_max: f32::INFINITY,
            a_log: fill(seed, h).iter().map(|v| v * 0.5).collect(),
            d: fill(seed, h),
            dt_bias: fill(seed, h),
        }
    }

    /// Run both forms on the same random sequence; assert max-abs agreement.
    fn assert_chunked_matches(h: usize, dh: usize, n: usize, g: usize, seq: usize, chunk: usize) {
        let mut seed = 0x00C0FFEEu32 ^ (seq as u32).wrapping_mul(2654435761);
        let p = rand_params(&mut seed, h, dh, n, g);
        let xd = h * dh;
        let bd = g * n;
        let x = fill(&mut seed, seq * xd);
        let b = fill(&mut seed, seq * bd);
        let c = fill(&mut seed, seq * bd);
        let dt = fill(&mut seed, seq * h);

        let mut st_seq = vec![0.0f32; h * dh * n];
        let mut y_seq = vec![0.0f32; seq * xd];
        ssd_sequence(&p, &mut st_seq, &x, &b, &c, &dt, &mut y_seq);

        let mut st_chunk = vec![0.0f32; h * dh * n];
        let mut y_chunk = vec![0.0f32; seq * xd];
        ssd_chunked(&p, &mut st_chunk, &x, &b, &c, &dt, &mut y_chunk, chunk);

        let max_y = y_seq
            .iter()
            .zip(&y_chunk)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_s = st_seq
            .iter()
            .zip(&st_chunk)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_y < 1e-3,
            "y mismatch (seq={seq},chunk={chunk}): max|Δ|={max_y}"
        );
        assert!(
            max_s < 1e-3,
            "final-state mismatch (seq={seq},chunk={chunk}): max|Δ|={max_s}"
        );
    }

    #[test]
    fn chunked_single_chunk_matches_sequential() {
        // seq <= chunk → one chunk, no state passing.
        assert_chunked_matches(4, 8, 16, 2, 12, 256);
    }

    #[test]
    fn chunked_multi_chunk_matches_sequential() {
        // seq spans several chunks → exercises inter-chunk state carry.
        assert_chunked_matches(4, 8, 16, 2, 200, 64);
        assert_chunked_matches(6, 5, 8, 3, 257, 256); // crosses the 256 boundary by 1
    }

    #[test]
    fn chunked_chunk_size_one_matches_sequential() {
        // chunk_size=1 degenerates to the per-token scan — the tightest check
        // that the decomposition's intra/inter split is consistent.
        assert_chunked_matches(3, 4, 8, 1, 40, 1);
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
