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

/// splitmix64 finalizer → u53 → [0,1). Deterministic state→uniform map; the
/// avalanche decorrelates adjacent states so the computed codebook covers the
/// Gaussian well rather than being monotone in the state integer.
#[inline]
fn hash_u01(state: u32) -> f64 {
    let mut z = (state as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Acklam's rational approximation of the inverse standard-normal CDF.
/// |error| < 1.15e-9. Maps u∈(0,1) → N(0,1) quantile.
fn inv_norm_cdf(p: f64) -> f64 {
    // Coefficients.
    const A: [f64; 6] = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Build the computed Gaussian codebook: `codebook[state]` ≈ a unit-variance
/// normal sample deterministically derived from `state`. Renormalized to exact
/// zero mean / unit variance so a single per-group `scale` reconstructs the
/// rotated (≈Gaussian, zero-mean) weights as `scale * codebook[state]`.
pub fn build_codebook() -> Vec<f32> {
    let mut cb: Vec<f64> = (0..NUM_STATES as u32)
        .map(|s| inv_norm_cdf(hash_u01(s)))
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
    let mut state: u32 = 0;
    let mut out = Vec::with_capacity(symbols.len());
    for &sym in symbols {
        state = ((state << BITS_PER_WEIGHT) | (sym as u32 & (NUM_SYMBOLS as u32 - 1))) & STATE_MASK;
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
    use std::collections::HashMap;
    let n = weights.len();
    // Active beam: (state, cumulative_cost). Start at state 0 (leading zeros).
    let mut beam: Vec<(u32, f64)> = vec![(0u32, 0.0)];
    // Per-step records for backtrack: (state, prev_beam_idx, symbol).
    let mut steps: Vec<Vec<(u32, u32, u8)>> = Vec::with_capacity(n);

    for &w in weights {
        let w = w as f64;
        // For each reachable next state keep the single cheapest predecessor.
        let mut best: HashMap<u32, (f64, u32, u8)> = HashMap::new();
        for (bi, &(s_prev, c_prev)) in beam.iter().enumerate() {
            let base = (s_prev << BITS_PER_WEIGHT) & STATE_MASK;
            for sym in 0..NUM_SYMBOLS as u32 {
                let s_new = base | sym;
                let diff = w - scale as f64 * codebook[s_new as usize] as f64;
                let c = c_prev + diff * diff;
                match best.get(&s_new) {
                    Some(&(bc, _, _)) if bc <= c => {}
                    _ => {
                        best.insert(s_new, (c, bi as u32, sym as u8));
                    }
                }
            }
        }
        let mut cand: Vec<(u32, f64, u32, u8)> =
            best.into_iter().map(|(st, (c, pi, sy))| (st, c, pi, sy)).collect();
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cand.truncate(beam_width);
        let mut rec = Vec::with_capacity(cand.len());
        let mut next_beam = Vec::with_capacity(cand.len());
        for (st, c, pi, sy) in cand {
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
    let mut state: u32 = 0;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &sym) in symbols.iter().enumerate() {
        state = ((state << BITS_PER_WEIGHT) | (sym as u32 & (NUM_SYMBOLS as u32 - 1))) & STATE_MASK;
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
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        let scale = if range > 0.0 { range / (levels - 1.0) } else { 1.0 };
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
                    *slot = f32::from_bits((u16::from_le_bytes([bytes[o], bytes[o + 1]]) as u32) << 16);
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

    #[test]
    fn beam_encode_is_near_viterbi() {
        let cb = build_codebook();
        let mut rng = Lcg(0xBEEF_F00D);
        let group: Vec<f32> = (0..256).map(|_| rng.next_normal()).collect();
        let scale = group_scale(&group);
        let vit = mse(&group, &decode_group(&encode_group(&group, scale, &cb), scale, &cb));
        let beam = mse(
            &group,
            &decode_group(&beam_encode_group(&group, scale, &cb, 128), scale, &cb),
        );
        eprintln!("viterbi MSE={vit:.6}  beam128 MSE={beam:.6}  (beam/vit={:.4})", beam / vit);
        // Beam (128) within ~6% of full Viterbi — the price for ~30× speed that
        // makes full-model encoding feasible. (beam=64 measured ~5.3%.)
        assert!(beam <= vit * 1.06, "beam128 MSE {beam:.6} not within 6% of Viterbi {vit:.6}");
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
