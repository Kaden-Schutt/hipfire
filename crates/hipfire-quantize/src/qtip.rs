// SPDX-License-Identifier: Apache-2.0
// hipfire — QTIP (Quantization with Trellises and Incoherence Processing)
//
// Phase C1 (NEXT-STEPS.md): the offline QTIP encoder core. Scalar/Lloyd 2-bit
// is quality-collapse (MQ2-Lloyd 0.8B ppl≈19,651); trellis-coded quantization
// is the only route to *usable* 2 bpw because it approaches the Gaussian
// rate-distortion bound instead of quantizing each weight independently.
//
// This module is the CPU encoder/decoder core, intentionally decoupled from
// any GPU kernel:
//   * Incoherence processing (the Hadamard/FWHT rotation that Gaussianizes a
//     weight group) is ALREADY applied upstream by `cpu_fwht_256` — QTIP
//     consumes the rotated group, same as `quantize_mq2g256`.
//   * **Bitshift trellis** (QTIP §3): the per-weight codebook is *computed*,
//     not stored. The trellis "state" is a sliding window of the last
//     `STATE_BITS` code bits; the decoded value is a hash of that window into
//     an (approximately) unit-variance Gaussian. Because the state is a sliding
//     window, **decode is embarrassingly parallel** (each position hashes its
//     own bit-window) — the GPU kernel (C2) exploits exactly this. The Viterbi
//     search is OFFLINE encode only.
//
// Quality is validated here against uniform 2-bit on synthetic Gaussian data
// (the post-rotation weight distribution). Full-model wiring + `astrea`
// KLD/PPL gating vs MQ4/MQ3 is the next increment, BEFORE the decode kernel.

/// Bits per weight (2-bit target).
pub const BITS_PER_WEIGHT: u32 = 2;

/// Trellis state width in bits. State = last `STATE_BITS` code bits =
/// a sliding window of `STATE_BITS / BITS_PER_WEIGHT` symbols. Larger =
/// richer computed codebook (2^STATE_BITS distinct Gaussian values) and
/// better rate-distortion, at O(2^STATE_BITS) Viterbi cost per group.
// 12 is the cost/quality sweet spot. STATE_BITS=16 was measured: real-weights
// QTIP-2/uniform-3 improved only 1.41→1.34 for ~16× the Viterbi cost (297s vs
// 19s for 256 groups) — infeasible at model scale, diminishing returns. Closing
// the residual gap to uniform-3 parity needs a *better codebook* (the QTIP paper
// 1MAD/3INST hashes / structured trellis), not brute-force state width.
pub const STATE_BITS: u32 = 12;

const NUM_STATES: usize = 1 << STATE_BITS;
const STATE_MASK: u32 = (NUM_STATES as u32) - 1;
const NUM_SYMBOLS: usize = 1 << BITS_PER_WEIGHT;

/// QTIP "1MAD" computed-codebook hash: maps a trellis state to an
/// approximately-N(0,1) value with one multiply-add + a byte sum (central-limit
/// over 4 bytes → Gaussian). Clean-room reimplementation of the published QTIP
/// method (the integer constants ARE the method; cf. QTIP §3 / `decode_1mad`).
/// GPU-cheap and stateless — exactly what the C2 decode kernel will compute, so
/// the offline codebook here matches the on-device decode bit-for-bit.
#[inline]
fn decode_1mad(state: u32) -> f32 {
    let x = (state as u64) & 0xFFFF_FFFF;
    let x = x.wrapping_mul(34_038_481).wrapping_add(76_625_530) & 0xFFFF_FFFF;
    let byte_sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF);
    // Center (E[sum]=510 for 4 uniform bytes) and scale to unit-ish variance.
    (byte_sum as f32 - 510.0) / 147.800_537_109_375
}

/// Build the computed codebook: `codebook[state] = decode_1mad(state)`,
/// renormalized to exact zero mean / unit variance so a single per-group
/// `scale` reconstructs the rotated (≈Gaussian, zero-mean) weights as
/// `scale * codebook[state]`. The renorm is offline-only; the kernel applies
/// the same affine via baked constants.
pub fn build_codebook() -> Vec<f32> {
    let mut cb: Vec<f64> = (0..NUM_STATES as u32)
        .map(|s| decode_1mad(s) as f64)
        .collect();
    let mean = cb.iter().sum::<f64>() / cb.len() as f64;
    for v in cb.iter_mut() {
        *v -= mean;
    }
    let var = cb.iter().map(|v| v * v).sum::<f64>() / cb.len() as f64;
    let inv_std = if var > 0.0 { 1.0 / var.sqrt() } else { 1.0 };
    cb.iter().map(|v| (v * inv_std) as f32).collect()
}

/// Decode a packed symbol stream back to weights: walk the bitshift trellis,
/// hashing each sliding-window state through the codebook. This is the exact
/// computation the GPU decode kernel (C2) will perform per lane — sequential
/// here only for the reference; each output depends solely on its own bit
/// window, so it is parallelizable.
pub fn decode_group(symbols: &[u8], scale: f32, codebook: &[f32]) -> Vec<f32> {
    decode_group_bits(symbols, scale, codebook, BITS_PER_WEIGHT)
}

// ─────────────────────────────────────────────────────────────────────────
// Bit-rate-parametric V=1 trellis (Phase C 3-bit fallback). The codebook is
// a pure state→value map (`build_codebook`, depends only on STATE_BITS), so it
// is shared across bit-rates; only the per-step symbol count (2^bits) and the
// shift width change. `STATE_BITS=12` stays fixed, so the sliding window is
// `12/bits` symbols (4 syms @ 2-bit, 4 syms @ 3-bit). The 2-bit public
// functions delegate here with `bits=2`; 3-bit (qtip3-sim) calls them directly.
// ─────────────────────────────────────────────────────────────────────────

/// `bits`-parametric trellis decode. See `decode_group`.
pub fn decode_group_bits(symbols: &[u8], scale: f32, codebook: &[f32], bits: u32) -> Vec<f32> {
    let sym_mask = (1u32 << bits) - 1;
    let mut state: u32 = 0;
    let mut out = Vec::with_capacity(symbols.len());
    for &sym in symbols {
        state = ((state << bits) | (sym as u32 & sym_mask)) & STATE_MASK;
        out.push(scale * codebook[state as usize]);
    }
    out
}

/// Viterbi encode one group of (FWHT-rotated) weights into a bitshift-trellis
/// symbol stream that minimizes Σ(w_i − scale·codebook[state_i])². OFFLINE.
///
/// Cost: `weights.len() × NUM_STATES × NUM_SYMBOLS`. With STATE_BITS=12 and a
/// 256-wide group that is ~4M float ops/group — fine offline; a beam-search
/// variant will replace it for full-model throughput in a later increment.
pub fn encode_group(weights: &[f32], scale: f32, codebook: &[f32]) -> Vec<u8> {
    let n = weights.len();
    let inf = f64::INFINITY;
    // dp[state] = min cost of any path reaching `state` after the current step.
    let mut dp = vec![inf; NUM_STATES];
    // Virtual start: state 0 with leading-zero history (decoder uses the same).
    dp[0] = 0.0;
    // backptr[step][state] = predecessor state.
    let mut back: Vec<Vec<u32>> = vec![vec![0u32; NUM_STATES]; n];

    let mut next = vec![inf; NUM_STATES];
    for (step, &w) in weights.iter().enumerate() {
        for v in next.iter_mut() {
            *v = inf;
        }
        let w = w as f64;
        for s_prev in 0..NUM_STATES {
            let c_prev = dp[s_prev];
            if c_prev == inf {
                continue;
            }
            let base = ((s_prev as u32) << BITS_PER_WEIGHT) & STATE_MASK;
            for sym in 0..NUM_SYMBOLS as u32 {
                let s_new = (base | sym) as usize;
                let diff = w - scale as f64 * codebook[s_new] as f64;
                let c = c_prev + diff * diff;
                if c < next[s_new] {
                    next[s_new] = c;
                    back[step][s_new] = s_prev as u32;
                }
            }
        }
        std::mem::swap(&mut dp, &mut next);
    }

    // Best final state, then backtrack to recover symbols.
    let mut best_state = 0usize;
    let mut best_cost = inf;
    for (s, &c) in dp.iter().enumerate() {
        if c < best_cost {
            best_cost = c;
            best_state = s;
        }
    }
    let mut symbols = vec![0u8; n];
    let mut state = best_state as u32;
    for step in (0..n).rev() {
        symbols[step] = (state & (NUM_SYMBOLS as u32 - 1)) as u8;
        state = back[step][state as usize];
    }
    symbols
}

/// Beam-search trellis encoder: keeps the top `beam_width` states per step
/// instead of all 2^STATE_BITS, cutting per-group cost from
/// `n × 2^STATE_BITS × NUM_SYMBOLS` to `n × beam_width × NUM_SYMBOLS`
/// (~60× at beam=64, STATE_BITS=12) — what makes full-model encoding feasible.
/// Near-Viterbi-optimal for moderate beams. Same bitshift-trellis semantics as
/// `encode_group`, so `decode_group` reconstructs identically.
pub fn beam_encode_group(
    weights: &[f32],
    scale: f32,
    codebook: &[f32],
    beam_width: usize,
) -> Vec<u8> {
    beam_encode_group_bits(weights, scale, codebook, beam_width, BITS_PER_WEIGHT)
}

/// `bits`-parametric beam-search trellis encode. See `beam_encode_group`.
pub fn beam_encode_group_bits(
    weights: &[f32],
    scale: f32,
    codebook: &[f32],
    beam_width: usize,
    bits: u32,
) -> Vec<u8> {
    let num_symbols = 1usize << bits;
    let n = weights.len();
    // Active beam: (state, cumulative_cost). Start at state 0 (leading zeros).
    let mut beam: Vec<(u32, f64)> = vec![(0u32, 0.0)];
    // Per-step records for backtrack: (state, prev_beam_idx, symbol).
    let mut steps: Vec<Vec<(u32, u32, u8)>> = Vec::with_capacity(n);
    // Reused candidate scratch — avoids per-step allocation (the old HashMap
    // path allocated per step, ~17× slower; sort-based dedup is allocation-free
    // after warmup).
    let mut cand: Vec<(u32, f64, u32, u8)> = Vec::with_capacity(beam_width * num_symbols);

    for &w in weights {
        let w = w as f64;
        cand.clear();
        for (bi, &(s_prev, c_prev)) in beam.iter().enumerate() {
            let base = (s_prev << bits) & STATE_MASK;
            for sym in 0..num_symbols as u32 {
                let s_new = base | sym;
                let diff = w - scale as f64 * codebook[s_new as usize] as f64;
                cand.push((s_new, c_prev + diff * diff, bi as u32, sym as u8));
            }
        }
        // Dedup by state keeping min cost: sort by (state, cost) then keep the
        // first occurrence per state (= the min-cost predecessor for it).
        cand.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        cand.dedup_by_key(|c| c.0);
        // Keep the top-`beam_width` by cost.
        if cand.len() > beam_width {
            cand.select_nth_unstable_by(beam_width, |a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(beam_width);
        }
        let mut rec = Vec::with_capacity(cand.len());
        let mut next_beam = Vec::with_capacity(cand.len());
        for &(st, c, pi, sy) in cand.iter() {
            rec.push((st, pi, sy));
            next_beam.push((st, c));
        }
        steps.push(rec);
        beam = next_beam;
    }

    // Best final beam slot, then backtrack.
    let mut best_idx = 0usize;
    let mut best_cost = f64::INFINITY;
    for (i, &(_, c)) in beam.iter().enumerate() {
        if c < best_cost {
            best_cost = c;
            best_idx = i;
        }
    }
    let mut symbols = vec![0u8; n];
    let mut idx = best_idx;
    for step in (0..n).rev() {
        let (_, prev_idx, sym) = steps[step][idx];
        symbols[step] = sym;
        idx = prev_idx as usize;
    }
    symbols
}

// ─────────────────────────────────────────────────────────────────────────
// V=2 vector trellis (QTIP paper config: V=2 weights/step, K=4 bits/step → 2
// bpw, L=16 state bits). Each step emits a 2-vector; the codebook is computed
// (two 1MAD hashes per state), so still no stored per-group table. This is the
// main rate-distortion lever beyond the V=1 scalar trellis. Kept separate from
// the V=1 path (which stays validated/in-use).
// ─────────────────────────────────────────────────────────────────────────

/// Vector dim, bits/step, state bits for the V=2 trellis.
pub const V2_V: usize = 2;
pub const V2_K: u32 = 4; // bits per step = bits_per_weight (2) × V (2)
pub const V2_STATE_BITS: u32 = 16;
const V2_NUM_STATES: usize = 1 << V2_STATE_BITS;
const V2_STATE_MASK: u32 = (V2_NUM_STATES as u32) - 1;
const V2_NUM_SYMBOLS: usize = 1 << V2_K; // 16
const V2_STEPS: usize = 256 / V2_V; // 128 steps per 256-group

/// Two decorrelated 1MAD draws per state → the V=2 codebook entry.
#[inline]
fn decode_1mad_pair(state: u32) -> (f32, f32) {
    (decode_1mad(state), decode_1mad(state ^ 0x9E37_79B9))
}

/// Number of distinct 2-vectors (tlut entries) the V=2 codebook resolves to.
/// 2^12 optimal 2D centroids — the bitshift state hashes onto these, so the
/// effective codebook is `V2_TLUT` *designed* points, not 2^16 random ones.
const V2_TLUT: usize = 4096;

/// Lloyd (k-means) fit of `t` centroids to N(0,I)₂ — a *designed* 2D codebook
/// that realizes the vector-quantization gain a random codebook can't.
/// Deterministic (seeded), offline, cheap (no model/GPU/torch).
fn lloyd_2d(t: usize, n_samples: usize, iters: usize) -> Vec<[f64; 2]> {
    // Deterministic 2D Gaussian samples (Box–Muller over an LCG).
    let mut lcg: u64 = 0xD1B5_4A32_D192_ED03;
    let mut u01 = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0)
    };
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let (u1, u2) = (u01(), u01());
        let r = (-2.0 * u1.ln()).sqrt();
        samples.push([r * (std::f64::consts::TAU * u2).cos(), r * (std::f64::consts::TAU * u2).sin()]);
    }
    // Init centroids from the first `t` samples.
    let mut cent: Vec<[f64; 2]> = samples[..t].to_vec();
    for _ in 0..iters {
        let mut sum = vec![[0.0f64; 2]; t];
        let mut cnt = vec![0u32; t];
        for s in &samples {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for (i, c) in cent.iter().enumerate() {
                let d = (s[0] - c[0]).powi(2) + (s[1] - c[1]).powi(2);
                if d < bd {
                    bd = d;
                    best = i;
                }
            }
            sum[best][0] += s[0];
            sum[best][1] += s[1];
            cnt[best] += 1;
        }
        for i in 0..t {
            if cnt[i] > 0 {
                cent[i] = [sum[i][0] / cnt[i] as f64, sum[i][1] / cnt[i] as f64];
            }
        }
    }
    cent
}

/// V=2 codebook (flat `[v0,v1]` per state). Each of the 2^L states maps via a
/// splitmix hash onto one of `V2_TLUT` Lloyd-fitted 2D centroids, so the
/// reachable codebook is the *designed* centroid set. Renormalized to exact
/// zero-mean / unit-variance.
pub fn build_codebook_v2() -> Vec<f32> {
    let tlut = lloyd_2d(V2_TLUT, 200_000, 12);
    let mut cb: Vec<f64> = Vec::with_capacity(2 * V2_NUM_STATES);
    for s in 0..V2_NUM_STATES as u32 {
        // splitmix32-ish hash → tlut index.
        let mut z = s.wrapping_mul(0x9E37_79B1);
        z ^= z >> 15;
        z = z.wrapping_mul(0x85EB_CA77);
        z ^= z >> 13;
        let idx = (z as usize) % V2_TLUT;
        cb.push(tlut[idx][0]);
        cb.push(tlut[idx][1]);
    }
    let mean = cb.iter().sum::<f64>() / cb.len() as f64;
    for v in cb.iter_mut() {
        *v -= mean;
    }
    let var = cb.iter().map(|v| v * v).sum::<f64>() / cb.len() as f64;
    let inv = if var > 0.0 { 1.0 / var.sqrt() } else { 1.0 };
    cb.iter().map(|v| (v * inv) as f32).collect()
}

/// Decode a V=2 symbol stream (128 × 4-bit) → 256 weights.
pub fn decode_group_v2(symbols: &[u8], scale: f32, codebook: &[f32]) -> Vec<f32> {
    let mut state: u32 = 0;
    let mut out = Vec::with_capacity(symbols.len() * V2_V);
    for &sym in symbols {
        state = ((state << V2_K) | (sym as u32 & (V2_NUM_SYMBOLS as u32 - 1))) & V2_STATE_MASK;
        out.push(scale * codebook[2 * state as usize]);
        out.push(scale * codebook[2 * state as usize + 1]);
    }
    out
}

/// Closed-form optimal scale for a fixed V=2 symbol stream.
pub fn optimal_scale_v2(weights: &[f32], symbols: &[u8], codebook: &[f32]) -> f32 {
    let mut state: u32 = 0;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &sym) in symbols.iter().enumerate() {
        state = ((state << V2_K) | (sym as u32 & (V2_NUM_SYMBOLS as u32 - 1))) & V2_STATE_MASK;
        for v in 0..V2_V {
            let c = codebook[2 * state as usize + v] as f64;
            num += weights[i * V2_V + v] as f64 * c;
            den += c * c;
        }
    }
    if den > 0.0 {
        (num / den) as f32
    } else {
        group_scale(weights)
    }
}

/// Beam-search V=2 trellis encode of a 256-weight group → 128 × 4-bit symbols.
pub fn beam_encode_group_v2(
    weights: &[f32],
    scale: f32,
    codebook: &[f32],
    beam_width: usize,
) -> Vec<u8> {
    assert_eq!(weights.len(), 256);
    let mut beam: Vec<(u32, f64)> = vec![(0u32, 0.0)];
    let mut steps: Vec<Vec<(u32, u32, u8)>> = Vec::with_capacity(V2_STEPS);
    let mut cand: Vec<(u32, f64, u32, u8)> = Vec::with_capacity(beam_width * V2_NUM_SYMBOLS);

    for step in 0..V2_STEPS {
        let w0 = weights[step * V2_V] as f64;
        let w1 = weights[step * V2_V + 1] as f64;
        cand.clear();
        for (bi, &(s_prev, c_prev)) in beam.iter().enumerate() {
            let base = (s_prev << V2_K) & V2_STATE_MASK;
            for sym in 0..V2_NUM_SYMBOLS as u32 {
                let s_new = base | sym;
                let c0 = scale as f64 * codebook[2 * s_new as usize] as f64;
                let c1 = scale as f64 * codebook[2 * s_new as usize + 1] as f64;
                let d0 = w0 - c0;
                let d1 = w1 - c1;
                cand.push((s_new, c_prev + d0 * d0 + d1 * d1, bi as u32, sym as u8));
            }
        }
        cand.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        cand.dedup_by_key(|c| c.0);
        if cand.len() > beam_width {
            cand.select_nth_unstable_by(beam_width, |a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(beam_width);
        }
        let mut rec = Vec::with_capacity(cand.len());
        let mut next_beam = Vec::with_capacity(cand.len());
        for &(st, c, pi, sy) in cand.iter() {
            rec.push((st, pi, sy));
            next_beam.push((st, c));
        }
        steps.push(rec);
        beam = next_beam;
    }

    let mut best_idx = 0usize;
    let mut best_cost = f64::INFINITY;
    for (i, &(_, c)) in beam.iter().enumerate() {
        if c < best_cost {
            best_cost = c;
            best_idx = i;
        }
    }
    let mut symbols = vec![0u8; V2_STEPS];
    let mut idx = best_idx;
    for step in (0..V2_STEPS).rev() {
        let (_, prev_idx, sym) = steps[step][idx];
        symbols[step] = sym;
        idx = prev_idx as usize;
    }
    symbols
}

// ─────────────────────────────────────────────────────────────────────────
// Real packed QTIP-3 group format (Phase C2 prerequisite). The qtip3-sim path
// stores bf16 (for a kernel-free PPL verdict); the *real* bandwidth win needs
// the packed 3-bit symbol stream on disk, which the C2 fused decode-GEMV kernel
// reads. Layout per 256-weight group: [f32 scale][96 B packed 3-bit symbols] =
// 100 B/group (0.39 B/weight vs MQ4's 0.53). NO zero-point — the QTIP codebook
// is zero-mean by construction (`build_codebook` renorm), unlike MQ3's
// [scale][zero][96 B]=104 B. The 8-symbols×3-bits→3-bytes bit-packing matches
// MQ3's `quantize_hfq3g256` exactly, so the kernel's bit-window unpack is the
// same; only the per-symbol *meaning* (trellis code → computed codebook) and
// the absence of a zero differ.
// ─────────────────────────────────────────────────────────────────────────

/// Bytes per packed QTIP-3 group: 4 (f32 scale) + 96 (256×3-bit symbols).
pub const QTIP3_BLOCK_BYTES: usize = 100;
/// QTIP group size in weights.
pub const QTIP3_GROUP: usize = 256;

/// Pack 8 three-bit symbols into 3 bytes (little-endian bitstream), matching
/// the MQ3 GEMV kernel's unpack window. Shared by pack/unpack below.
#[inline]
fn pack8_3bit(q: &[u8; 8]) -> [u8; 3] {
    let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
    let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
    let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);
    [b0, b1, b2]
}

/// Inverse of `pack8_3bit`.
#[inline]
fn unpack8_3bit(b: &[u8; 3]) -> [u8; 8] {
    [
        b[0] & 7,
        (b[0] >> 3) & 7,
        ((b[0] >> 6) | (b[1] << 2)) & 7,
        (b[1] >> 1) & 7,
        (b[1] >> 4) & 7,
        ((b[1] >> 7) | (b[2] << 1)) & 7,
        (b[2] >> 2) & 7,
        (b[2] >> 5) & 7,
    ]
}

/// Pack one 256-symbol QTIP-3 group (3-bit trellis symbols + scale) into the
/// 100-byte on-disk record the C2 kernel reads.
pub fn pack_qtip3_group(symbols: &[u8], scale: f32) -> Vec<u8> {
    assert_eq!(symbols.len(), QTIP3_GROUP, "QTIP-3 group is 256 symbols");
    let mut out = vec![0u8; QTIP3_BLOCK_BYTES];
    out[0..4].copy_from_slice(&scale.to_le_bytes());
    for chunk in 0..32 {
        let mut q = [0u8; 8];
        for (j, qj) in q.iter_mut().enumerate() {
            *qj = symbols[chunk * 8 + j] & 7;
        }
        let packed = pack8_3bit(&q);
        let bo = 4 + chunk * 3;
        out[bo..bo + 3].copy_from_slice(&packed);
    }
    out
}

/// Unpack a 100-byte QTIP-3 record → (256 symbols, scale).
pub fn unpack_qtip3_group(bytes: &[u8]) -> (Vec<u8>, f32) {
    assert_eq!(bytes.len(), QTIP3_BLOCK_BYTES);
    let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let mut symbols = vec![0u8; QTIP3_GROUP];
    for chunk in 0..32 {
        let bo = 4 + chunk * 3;
        let b = [bytes[bo], bytes[bo + 1], bytes[bo + 2]];
        let q = unpack8_3bit(&b);
        symbols[chunk * 8..chunk * 8 + 8].copy_from_slice(&q);
    }
    (symbols, scale)
}

/// Per-group scale: RMS of the rotated weights (codebook is unit-variance,
/// zero-mean, so RMS ≈ the optimal single scale for ≈Gaussian input). Used to
/// drive the Viterbi search; refine with `optimal_scale` after encoding.
pub fn group_scale(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 1.0;
    }
    let ss: f64 = weights.iter().map(|&w| (w as f64) * (w as f64)).sum();
    (ss / weights.len() as f64).sqrt() as f32
}

/// Closed-form least-squares scale for a fixed symbol stream: minimizes
/// Σ(w_i − s·c_i)² over s ⇒ s* = ⟨w,c⟩ / ⟨c,c⟩, where c_i = codebook[state_i].
/// Strictly ≤ the MSE of any other scale (including the RMS seed), near-free.
/// This is the scale that should be STORED per group.
pub fn optimal_scale(weights: &[f32], symbols: &[u8], codebook: &[f32]) -> f32 {
    optimal_scale_bits(weights, symbols, codebook, BITS_PER_WEIGHT)
}

/// `bits`-parametric closed-form optimal scale. See `optimal_scale`.
pub fn optimal_scale_bits(weights: &[f32], symbols: &[u8], codebook: &[f32], bits: u32) -> f32 {
    let sym_mask = (1u32 << bits) - 1;
    let mut state: u32 = 0;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &sym) in symbols.iter().enumerate() {
        state = ((state << bits) | (sym as u32 & sym_mask)) & STATE_MASK;
        let c = codebook[state as usize] as f64;
        num += weights[i] as f64 * c;
        den += c * c;
    }
    if den > 0.0 {
        (num / den) as f32
    } else {
        group_scale(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic LCG → standard-normal via Box–Muller. Avoids a dev-dep on
    // `rand`; we only need a reproducible Gaussian sample for the quality test.
    struct Lcg(u64);
    impl Lcg {
        fn next_u01(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn next_normal(&mut self) -> f32 {
            let u1 = self.next_u01().max(1e-12);
            let u2 = self.next_u01();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        }
    }

    fn mse(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| ((x - y) as f64) * ((x - y) as f64))
            .sum::<f64>()
            / a.len() as f64
    }

    /// Full QTIP roundtrip MSE: Viterbi-encode at the RMS seed scale, then
    /// store the closed-form optimal scale and decode with it.
    fn qtip_mse(group: &[f32], cb: &[f32]) -> f64 {
        let sym = encode_group(group, group_scale(group), cb);
        let s = optimal_scale(group, &sym, cb);
        mse(group, &decode_group(&sym, s, cb))
    }

    /// Uniform n-bit (`2^bits`-level) per-group quantize+dequant, matching the
    /// MQ packing math — the baseline QTIP must beat. `bits=2` is the iso-bpw
    /// rival; `bits=3` is the quality target (QTIP-2 ≈ uniform-3 = the win).
    fn uniform_nbit_roundtrip(group: &[f32], bits: u32) -> Vec<f32> {
        let levels = (1u32 << bits) as f32;
        let min_v = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_v - min_v;
        let scale = if range > 0.0 {
            range / (levels - 1.0)
        } else {
            1.0
        };
        let inv = if range > 0.0 { 1.0 / scale } else { 0.0 };
        group
            .iter()
            .map(|&w| {
                let q = (((w - min_v) * inv + 0.5) as i32).clamp(0, (levels as i32) - 1) as f32;
                min_v + q * scale
            })
            .collect()
    }

    fn uniform2_roundtrip(group: &[f32]) -> Vec<f32> {
        uniform_nbit_roundtrip(group, 2)
    }

    #[test]
    fn codebook_is_zero_mean_unit_variance() {
        let cb = build_codebook();
        let mean = cb.iter().map(|&v| v as f64).sum::<f64>() / cb.len() as f64;
        let var = cb.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / cb.len() as f64;
        assert!(mean.abs() < 1e-5, "codebook mean {mean}");
        assert!((var - 1.0).abs() < 1e-3, "codebook var {var}");
    }

    /// QTIP-2's correct yardstick is the **2-bit Gaussian rate-distortion
    /// bound** D(R=2) = σ²·2^(-2R) = σ²/16 = 0.0625 (unit variance) — NOT
    /// uniform-3. Uniform-3 (~0.047) spends a whole extra bit and sits *below*
    /// the 2-bit bound, so QTIP-2 reaching uniform-3 MSE is information-
    /// theoretically impossible. The real questions: (1) how close to the
    /// 0.0625 floor is QTIP (vs uniform-2 ≈ 0.26, ~4× the floor), and (2) is
    /// near-optimal-2-bit *usable* — a full-model PPL question (C1d), not a
    /// reconstruction one. This test asserts QTIP-2 beats uniform-2 and reports
    /// the gap to the bound so codebook tuning toward 0.0625 is visible.
    #[test]
    fn qtip2_beats_uniform2_reports_bound_gap() {
        let cb = build_codebook();
        let mut rng = Lcg(0x1234_5678);
        let group: Vec<f32> = (0..256).map(|_| rng.next_normal()).collect();
        let var = group.iter().map(|&w| (w as f64) * (w as f64)).sum::<f64>() / group.len() as f64;
        let bound = var / 16.0; // 2-bit Gaussian rate-distortion bound
        let qtip = qtip_mse(&group, &cb);
        let u2 = mse(&group, &uniform_nbit_roundtrip(&group, 2));
        eprintln!(
            "QTIP-2 MSE={qtip:.5}  uniform-2={u2:.5}  2-bit RD bound={bound:.5}  \
             (QTIP/u2={:.3}, QTIP/bound={:.3})",
            qtip / u2,
            qtip / bound,
        );
        assert!(qtip < u2, "QTIP-2 must beat uniform-2 (iso-bpw)");
    }

    /// C1b real-weights quality gate. Env-gated (needs a local safetensors
    /// file); skips cleanly in CI. Reads the first large 2D BF16 weight,
    /// FWHT-rotates each 256-group via the engine's sign tables, and reports
    /// aggregate reconstruction MSE for QTIP-2 vs uniform 2/3-bit.
    ///   HIPFIRE_QTIP_EVAL_ST=<model.safetensors> cargo test -p hipfire-quantize \
    ///       real_weights_quality_gate -- --nocapture
    #[test]
    fn real_weights_quality_gate() {
        let path = match std::env::var("HIPFIRE_QTIP_EVAL_ST") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip real_weights_quality_gate (set HIPFIRE_QTIP_EVAL_ST)");
                return;
            }
        };
        let file = std::fs::File::open(&path).expect("open safetensors");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
        let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&mmap[8..8 + hlen]).expect("parse header");
        let data_base = 8 + hlen;
        let obj = header.as_object().unwrap();
        let mut chosen: Option<(String, Vec<usize>)> = None;
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dt = meta["dtype"].as_str().unwrap_or("");
            let shape = meta["shape"].as_array().map(|a| a.len()).unwrap_or(0);
            if shape == 2 && dt == "BF16" {
                let off: Vec<usize> = meta["data_offsets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();
                if off[1] - off[0] >= 262144 * 2 {
                    chosen = Some((name.clone(), off));
                    break;
                }
            }
        }
        let (name, off) = chosen.expect("no large 2D BF16 weight found");
        let bytes = &mmap[data_base + off[0]..data_base + off[1]];
        // BF16 → f32; cap at 256 groups (65536 weights) for a fast gate.
        let weights: Vec<f32> = bytes
            .chunks_exact(2)
            .take(256 * 256)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();

        let cb = build_codebook();
        let signs1 = crate::gen_fwht_signs(42, 256);
        let signs2 = crate::gen_fwht_signs(1042, 256);
        let (mut q_acc, mut u2_acc, mut bound_acc, mut groups) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for chunk in weights.chunks(256) {
            if chunk.len() < 256 {
                break;
            }
            let mut g = [0.0f32; 256];
            g.copy_from_slice(chunk);
            crate::cpu_fwht_256(&mut g, &signs1, &signs2);
            let var = g.iter().map(|&w| (w as f64) * (w as f64)).sum::<f64>() / 256.0;
            q_acc += qtip_mse(&g, &cb);
            u2_acc += mse(&g, &uniform_nbit_roundtrip(&g, 2));
            bound_acc += var / 16.0; // 2-bit Gaussian rate-distortion bound
            groups += 1;
        }
        let (q, u2, bound) = (
            q_acc / groups as f64,
            u2_acc / groups as f64,
            bound_acc / groups as f64,
        );
        eprintln!(
            "QTIP real-weights gate: tensor={name} groups={groups}\n  \
             QTIP-2 MSE={q:.6}  uniform-2 MSE={u2:.6}  2-bit RD bound={bound:.6}\n  \
             QTIP-2/uniform-2={:.3}  QTIP-2/bound={:.3} (1.0 = optimal 2-bit)",
            q / u2,
            q / bound,
        );
        assert!(q < u2, "QTIP-2 must beat uniform-2 on real weights");
    }

    /// C1d model-wide reconstruction gate (env-gated). Iterates EVERY large 2D
    /// BF16 weight in the safetensors, FWHT-rotates, beam-encodes a sampled set
    /// of groups per tensor, and reports aggregate QTIP-2 vs the 2-bit RD bound
    /// and vs uniform-2 across the whole model — the strongest quality signal
    /// reachable without the GPU forward (full PPL still needs C2).
    ///   HIPFIRE_QTIP_EVAL_ST=<model.safetensors> cargo test -p hipfire-quantize \
    ///       whole_model_quality_gate -- --nocapture
    #[test]
    fn whole_model_quality_gate() {
        let path = match std::env::var("HIPFIRE_QTIP_EVAL_ST") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip whole_model_quality_gate (set HIPFIRE_QTIP_EVAL_ST)");
                return;
            }
        };
        let file = std::fs::File::open(&path).expect("open safetensors");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
        let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&mmap[8..8 + hlen]).expect("parse header");
        let data_base = 8 + hlen;
        let cb = build_codebook();
        let signs1 = crate::gen_fwht_signs(42, 256);
        let signs2 = crate::gen_fwht_signs(1042, 256);
        const GROUPS_PER_TENSOR: usize = 16; // sampled, evenly spaced

        let (mut q_acc, mut u2_acc, mut bound_acc, mut groups, mut tensors) =
            (0.0f64, 0.0f64, 0.0f64, 0usize, 0usize);
        for (name, meta) in header.as_object().unwrap() {
            if name == "__metadata__" || meta["dtype"].as_str() != Some("BF16") {
                continue;
            }
            let shape = match meta["shape"].as_array() {
                Some(s) if s.len() == 2 => s,
                _ => continue,
            };
            let _ = shape;
            let off: Vec<usize> = meta["data_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let n_weights = (off[1] - off[0]) / 2;
            if n_weights < 256 {
                continue;
            }
            let n_groups = n_weights / 256;
            let bytes = &mmap[data_base + off[0]..data_base + off[1]];
            let stride = (n_groups / GROUPS_PER_TENSOR).max(1);
            tensors += 1;
            for gi in (0..n_groups).step_by(stride).take(GROUPS_PER_TENSOR) {
                let base = gi * 256 * 2;
                let mut g = [0.0f32; 256];
                for (j, slot) in g.iter_mut().enumerate() {
                    let o = base + j * 2;
                    *slot =
                        f32::from_bits((u16::from_le_bytes([bytes[o], bytes[o + 1]]) as u32) << 16);
                }
                crate::cpu_fwht_256(&mut g, &signs1, &signs2);
                let var = g.iter().map(|&w| (w as f64) * (w as f64)).sum::<f64>() / 256.0;
                let sym = beam_encode_group(&g, group_scale(&g), &cb, 128);
                let s = optimal_scale(&g, &sym, &cb);
                q_acc += mse(&g, &decode_group(&sym, s, &cb));
                u2_acc += mse(&g, &uniform_nbit_roundtrip(&g, 2));
                bound_acc += var / 16.0;
                groups += 1;
            }
        }
        let (q, u2, bound) = (
            q_acc / groups as f64,
            u2_acc / groups as f64,
            bound_acc / groups as f64,
        );
        eprintln!(
            "QTIP whole-model gate: {tensors} tensors, {groups} groups (beam=128)\n  \
             QTIP-2 MSE={q:.6}  uniform-2={u2:.6}  2-bit RD bound={bound:.6}\n  \
             QTIP-2/uniform-2={:.3}  QTIP-2/bound={:.3} (1.0 = optimal 2-bit)",
            q / u2,
            q / bound,
        );
        assert!(q < u2, "QTIP-2 must beat uniform-2 model-wide");
    }

    /// FINDING: a *random* (computed-hash) V=2 codebook does NOT beat V=1 —
    /// it's coarser per-dimension (≈√(2^L) pts/dim vs 2^L for scalar), so it
    /// loses the vector-quantization gain. Realizing V=2's advantage needs a
    /// *designed* codebook (E8 lattice / trained tlut), not two 1MAD hashes.
    /// This test validates the V=2 path round-trips and records v2/v1 so a
    /// future designed-codebook change can be measured against it.
    #[test]
    fn v2_vector_trellis_roundtrips_reports_vs_v1() {
        let cb1 = build_codebook();
        let cb2 = build_codebook_v2();
        let mut rng = Lcg(0x2222);
        let group: Vec<f32> = (0..256).map(|_| rng.next_normal()).collect();
        let var = group.iter().map(|&w| (w as f64) * (w as f64)).sum::<f64>() / 256.0;
        let bound = var / 16.0;

        let v1 = qtip_mse(&group, &cb1);
        let scale0 = group_scale(&group);
        let sym2 = beam_encode_group_v2(&group, scale0, &cb2, 128);
        let s2 = optimal_scale_v2(&group, &sym2, &cb2);
        let v2 = mse(&group, &decode_group_v2(&sym2, s2, &cb2));

        eprintln!(
            "V=1 MSE={v1:.5} V=2 MSE={v2:.5} bound={bound:.5} (V2/V1={:.3}, V2/bound={:.3}) \
             [Lloyd-designed V=2 codebook: narrowed V2/V1 from 1.09 (random) to ~1.01 but \
              still not < 1 — vector gain needs the trellis-structured quantlut state→tlut \
              map (cf. QTIP), not a hash; finetune (the bigger lever) impractical on this box]",
            v2 / v1,
            v2 / bound
        );
        assert_eq!(sym2.len(), 128);
        // Round-trip sanity: V=2 must at least clearly beat uniform-2.
        let u2 = mse(&group, &uniform_nbit_roundtrip(&group, 2));
        assert!(v2 < u2 * 0.6, "V=2 should beat uniform-2: {v2} vs {u2}");
    }

    /// Phase C 3-bit fallback: the bits-parametric path must reconstruct
    /// strictly better at 3 bits than at 2 (one extra bit/weight → finer
    /// per-step symbol resolution against the shared codebook). Validates that
    /// `*_bits` plumbing (shift width + symbol count) is wired correctly and
    /// that 3-bit is the bandwidth/quality fallback it's meant to be.
    #[test]
    fn qtip3_beats_qtip2_reconstruction() {
        let cb = build_codebook();
        let mut rng = Lcg(0x3B17_3B17);
        let group: Vec<f32> = (0..256).map(|_| rng.next_normal()).collect();
        let scale0 = group_scale(&group);

        let s2 = beam_encode_group_bits(&group, scale0, &cb, 128, 2);
        let mse2 = mse(&group, &decode_group_bits(&s2, optimal_scale_bits(&group, &s2, &cb, 2), &cb, 2));
        let s3 = beam_encode_group_bits(&group, scale0, &cb, 128, 3);
        let mse3 = mse(&group, &decode_group_bits(&s3, optimal_scale_bits(&group, &s3, &cb, 3), &cb, 3));

        eprintln!("QTIP-2 MSE={mse2:.6}  QTIP-3 MSE={mse3:.6}  (3/2={:.3})", mse3 / mse2);
        assert_eq!(s3.len(), 256, "3-bit emits one symbol per weight (V=1)");
        assert!(mse3 < mse2, "3-bit must reconstruct better than 2-bit: {mse3} vs {mse2}");
    }

    /// C2 prerequisite: the real packed QTIP-3 record round-trips bit-exactly,
    /// and decoding from the unpacked symbols reproduces the direct
    /// `decode_group_bits` output — i.e. the on-disk format the kernel reads is
    /// faithful to the sim that was PPL-validated (15.20). Guards the byte
    /// layout the GEMV kernel will depend on.
    #[test]
    fn qtip3_pack_roundtrip_matches_decode() {
        let cb = build_codebook();
        let mut rng = Lcg(0x9051_3B17);
        let group: Vec<f32> = (0..256).map(|_| rng.next_normal()).collect();
        let scale0 = group_scale(&group);

        let sym = beam_encode_group_bits(&group, scale0, &cb, 128, 3);
        let scale = optimal_scale_bits(&group, &sym, &cb, 3);

        // Pack → unpack must recover symbols + scale exactly.
        let rec = pack_qtip3_group(&sym, scale);
        assert_eq!(rec.len(), QTIP3_BLOCK_BYTES);
        let (sym2, scale2) = unpack_qtip3_group(&rec);
        assert_eq!(sym, sym2, "packed symbols must round-trip bit-exactly");
        assert_eq!(scale.to_bits(), scale2.to_bits(), "scale must round-trip");

        // Decode from the unpacked record == direct decode (kernel faithfulness).
        let direct = decode_group_bits(&sym, scale, &cb, 3);
        let from_pack = decode_group_bits(&sym2, scale2, &cb, 3);
        assert_eq!(direct, from_pack, "decode from packed record must match direct");

        // Format is 0.39 B/weight: 100 B / 256 weights.
        let bpw = QTIP3_BLOCK_BYTES as f64 / QTIP3_GROUP as f64;
        eprintln!("QTIP-3 packed: {QTIP3_BLOCK_BYTES} B/group = {bpw:.3} B/weight (MQ4=0.53)");
        assert!(bpw < 0.40, "QTIP-3 must be < 0.40 B/weight");
    }

    /// C2 kernel correctness: the GPU kernel decodes each weight independently
    /// from a 4-symbol sliding window (`state_i = last 4 symbols ending at i`),
    /// NOT by walking the trellis sequentially. This test replicates the
    /// kernel's exact per-lane math (the `ST(a,b,c,d)` window in
    /// gemv_qtip3g256.hip) and asserts it reproduces the sequential
    /// `decode_group_bits` output bit-for-bit — i.e. the parallel decode the
    /// kernel relies on is equivalent to the reference. Guards against window
    /// off-by-one / leading-zero-padding bugs in the kernel.
    #[test]
    fn kernel_window_decode_matches_sequential_trellis() {
        let cb = build_codebook();
        let mut rng = Lcg(0xC2_F00D);
        let group: Vec<f32> = (0..256).map(|_| rng.next_normal()).collect();
        let scale0 = group_scale(&group);
        let sym = beam_encode_group_bits(&group, scale0, &cb, 128, 3);
        let scale = optimal_scale_bits(&group, &sym, &cb, 3);

        // Sequential reference (what the sim/PPL used).
        let seq = decode_group_bits(&sym, scale, &cb, 3);

        // Parallel 4-symbol-window decode (the kernel's method): state at i is
        // the last 4 symbols ending at i, leading-zero-padded at the start.
        let s = |k: i32| -> u32 {
            if k < 0 {
                0
            } else {
                sym[k as usize] as u32 & 7
            }
        };
        let mut par = vec![0.0f32; 256];
        for i in 0..256i32 {
            let state = ((s(i - 3) << 9) | (s(i - 2) << 6) | (s(i - 1) << 3) | s(i)) & 0xFFF;
            par[i as usize] = scale * cb[state as usize];
        }
        assert_eq!(seq, par, "kernel window decode must match sequential trellis");
    }

    #[test]
    fn beam_encode_is_near_viterbi() {
        let cb = build_codebook();
        let mut rng = Lcg(0xBEEF_F00D);
        let group: Vec<f32> = (0..256).map(|_| rng.next_normal()).collect();
        let scale = group_scale(&group);
        let vit = mse(
            &group,
            &decode_group(&encode_group(&group, scale, &cb), scale, &cb),
        );
        let beam = mse(
            &group,
            &decode_group(&beam_encode_group(&group, scale, &cb, 128), scale, &cb),
        );
        eprintln!(
            "viterbi MSE={vit:.6}  beam128 MSE={beam:.6}  (beam/vit={:.4})",
            beam / vit
        );
        // Beam (128) within ~6% of full Viterbi — the price for ~30× speed that
        // makes full-model encoding feasible. (beam=64 measured ~5.3%.)
        assert!(
            beam <= vit * 1.06,
            "beam128 MSE {beam:.6} not within 6% of Viterbi {vit:.6}"
        );
    }

    #[test]
    fn encode_decode_roundtrips_and_beats_uniform_2bit() {
        let cb = build_codebook();
        let mut rng = Lcg(0xDEADBEEF);
        let group: Vec<f32> = (0..256).map(|_| rng.next_normal()).collect();
        let scale = group_scale(&group);

        let symbols = encode_group(&group, scale, &cb);
        assert_eq!(symbols.len(), 256);
        let recon = decode_group(&symbols, scale, &cb);
        let qtip_mse = mse(&group, &recon);

        let uni = uniform2_roundtrip(&group);
        let uni_mse = mse(&group, &uni);

        // Trellis-coded 2-bit must beat uniform 2-bit on Gaussian input — the
        // whole premise of Phase C. (Empirically a large margin; require a
        // clear win, not just ≤.)
        assert!(
            qtip_mse < uni_mse * 0.85,
            "QTIP MSE {qtip_mse:.5} not < 0.85 × uniform2 MSE {uni_mse:.5}"
        );
    }
}
