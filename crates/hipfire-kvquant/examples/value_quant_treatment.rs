// SPDX-License-Identifier: Apache-2.0
// hipfire — how much treatment does 4-bit V (Value) actually need?
//
// K and V enter attention differently:
//   - K: score = Q·K → softmax → SELECTION. K error is amplified by the
//     nonlinear softmax and dominated by outlier CHANNELS (why K needs var-norm).
//   - V: out = Σ_t softmax_t · V_t → a WEIGHTED AVERAGE. V error enters linearly
//     and per-token errors partially CANCEL in the sum. V has no outlier-channel
//     pathology (it is the "smooth" operand).
//
// So the question for V4 is: is a plain per-group 4-bit affine quant enough, or
// does V also need per-token / per-channel scaling or the full Sinkhorn var-norm?
// Metric that matters = the attention OUTPUT error ‖out_exact − out_quant‖/‖out‖
// under a realistic peaked softmax, NOT raw reconstruction. Also report V-tensor
// reconstruction SNR for reference.

use hipfire_kvquant::kvarn::{dequantize_tile, quantize_tile_qmax};

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn normal(&mut self) -> f32 {
        let u1 = ((self.f() + 1.0) * 0.5).clamp(1e-6, 1.0);
        let u2 = (self.f() + 1.0) * 0.5;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

// Synthetic V [T tokens × D channels]: moderate per-token magnitude, MILD
// per-channel structure (a couple ~3x channels, NOT K's ~20x spikes), Gaussian.
fn make_v(rng: &mut Lcg, t: usize, d: usize) -> Vec<f32> {
    let mut chan = vec![1.0f32; d];
    for (i, cs) in chan.iter_mut().enumerate() {
        *cs = (0.35 * rng.normal()).exp(); // ~log-normal, gentle
        if i % 41 == 7 {
            *cs *= 3.0; // a few mild outlier channels
        }
    }
    let mut v = vec![0.0f32; t * d];
    for ti in 0..t {
        let tok = (0.4 * rng.normal()).exp();
        for c in 0..d {
            v[ti * d + c] = tok * chan[c] * rng.normal();
        }
    }
    v
}

// Peaked attention weights over T keys: a handful dominate, long tail — the
// regime where V actually contributes (uniform weights would trivially cancel).
fn make_weights(rng: &mut Lcg, t: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; t];
    for si in s.iter_mut() {
        *si = 1.5 * rng.normal();
    }
    for _ in 0..4 {
        s[(rng.f().abs() * t as f32) as usize % t] += 4.0; // a few winners
    }
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f32;
    for si in s.iter_mut() {
        *si = (*si - m).exp();
        z += *si;
    }
    for si in s.iter_mut() {
        *si /= z;
    }
    s
}

fn weighted_out(w: &[f32], v: &[f32], t: usize, d: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; d];
    for ti in 0..t {
        let wt = w[ti];
        for c in 0..d {
            o[c] += wt * v[ti * d + c];
        }
    }
    o
}

fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += (*x as f64 - *y as f64).powi(2);
        den += (*x as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn snr_db(orig: &[f32], recon: &[f32]) -> f64 {
    let (mut s, mut e) = (0.0f64, 0.0f64);
    for (o, r) in orig.iter().zip(recon) {
        s += (*o as f64).powi(2);
        e += (*o as f64 - *r as f64).powi(2);
    }
    10.0 * (s / e.max(1e-30)).log10()
}

// ── V quant treatments (all reconstruct V[T×D]) ───────────────────────────
// Per-group affine int(bits): each token's D channels split into groups of
// `g`, each group its own asymmetric min/max scale (the Q8_0 idiom at `bits`).
fn q_pergroup(v: &[f32], t: usize, d: usize, bits: usize, g: usize) -> Vec<f32> {
    let qmax = ((1u32 << bits) - 1) as f32;
    let mut out = vec![0.0f32; t * d];
    for ti in 0..t {
        for gb in 0..(d / g) {
            let base = ti * d + gb * g;
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for k in 0..g {
                lo = lo.min(v[base + k]);
                hi = hi.max(v[base + k]);
            }
            let sc = ((hi - lo) / qmax).max(1e-8);
            for k in 0..g {
                let q = ((v[base + k] - lo) / sc).round().clamp(0.0, qmax);
                out[base + k] = q * sc + lo;
            }
        }
    }
    out
}

// Per-channel symmetric int(bits): one absmax scale per channel across tokens.
fn q_perchannel(v: &[f32], t: usize, d: usize, bits: usize) -> Vec<f32> {
    let qmax = ((1u32 << (bits - 1)) - 1) as f32;
    let mut out = vec![0.0f32; t * d];
    for c in 0..d {
        let mut amax = 0.0f32;
        for ti in 0..t {
            amax = amax.max(v[ti * d + c].abs());
        }
        let sc = (amax / qmax).max(1e-8);
        for ti in 0..t {
            out[ti * d + c] = (v[ti * d + c] / sc).round().clamp(-qmax, qmax) * sc;
        }
    }
    out
}

// Full Sinkhorn var-norm int(bits) over [T×D] quantized PER-TOKEN (rows=tokens).
fn q_varnorm_pertoken(v: &[f32], t: usize, d: usize, bits: usize) -> Vec<f32> {
    let qmax = ((1u32 << bits) - 1) as f32;
    dequantize_tile(&quantize_tile_qmax(v, t, d, qmax)) // r=T rows (per-token)
}

// EXACTLY what the CASK cold tier does today: var-norm over the TRANSPOSED tile
// [D×T] quantized PER-CHANNEL (rows=channels), reusing the K codec on V.
fn q_varnorm_perchannel(v: &[f32], t: usize, d: usize, bits: usize) -> Vec<f32> {
    let qmax = ((1u32 << bits) - 1) as f32;
    let mut vt = vec![0.0f32; t * d];
    for ti in 0..t {
        for c in 0..d {
            vt[c * t + ti] = v[ti * d + c]; // [D×T]
        }
    }
    let deq = dequantize_tile(&quantize_tile_qmax(&vt, d, t, qmax)); // r=D rows (per-channel)
    let mut out = vec![0.0f32; t * d];
    for c in 0..d {
        for ti in 0..t {
            out[ti * d + c] = deq[c * t + ti];
        }
    }
    out
}

fn main() {
    let (t, d) = (512usize, 128usize);
    let trials = 48usize;
    println!(
        "V [T={t} tokens x D={d} ch], {trials} trials — attention-output rel-err (lower=better)"
    );
    println!("  treatment            bits   out-rel-err   V-recon-SNR(dB)");
    let treatments: &[(&str, usize, fn(&[f32], usize, usize, usize) -> Vec<f32>)] = &[
        ("Q8 per-group(32)", 8, |v, t, d, _| {
            q_pergroup(v, t, d, 8, 32)
        }),
        ("int4 per-group(32)", 4, |v, t, d, _| {
            q_pergroup(v, t, d, 4, 32)
        }),
        ("int4 per-channel(absmax)", 4, |v, t, d, _| {
            q_perchannel(v, t, d, 4)
        }),
        ("int4 varnorm per-CHANNEL[CASK now]", 4, |v, t, d, _| {
            q_varnorm_perchannel(v, t, d, 4)
        }),
        ("int4 varnorm per-TOKEN [fix]", 4, |v, t, d, _| {
            q_varnorm_pertoken(v, t, d, 4)
        }),
        ("int2 per-group(32)", 2, |v, t, d, _| {
            q_pergroup(v, t, d, 2, 32)
        }),
        ("int2 varnorm per-CHANNEL[CASK now]", 2, |v, t, d, _| {
            q_varnorm_perchannel(v, t, d, 2)
        }),
        ("int2 varnorm per-TOKEN [fix]", 2, |v, t, d, _| {
            q_varnorm_pertoken(v, t, d, 2)
        }),
    ];
    for &(name, bits, f) in treatments {
        let (mut oerr, mut snr) = (0.0f64, 0.0f64);
        for tr in 0..trials {
            let mut rng = Lcg(0xA5A5 ^ (tr as u64).wrapping_mul(0x9E3779B1));
            let v = make_v(&mut rng, t, d);
            let w = make_weights(&mut rng, t);
            let deq = f(&v, t, d, bits);
            oerr += rel_err(&weighted_out(&w, &v, t, d), &weighted_out(&w, &deq, t, d));
            snr += snr_db(&v, &deq);
        }
        println!(
            "  {name:<22} {bits:>2}   {:>11.5}   {:>10.2}",
            oerr / trials as f64,
            snr / trials as f64
        );
    }
}
