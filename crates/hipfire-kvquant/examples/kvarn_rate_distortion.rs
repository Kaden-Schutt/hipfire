// SPDX-License-Identifier: Apache-2.0
// hipfire — KVarN codec rate-distortion probe.
//
// Question: is the KVarN-4 K reconstruction loss BIT-limited (more code bits
// would help → KVarN-{5,6} worth a new packing) or VAR-NORM/rotation-limited
// (the Sinkhorn balance + per-channel min/max is the error source → more bits
// are wasted and Q8-K is the only real step up)?
//
// Method: synthetic K tiles [head_dim × GROUP] with heavy-tailed per-CHANNEL
// scale (the outlier-channel structure var-norm targets) + mild per-token
// scale. Sweep the var-norm codec at qmax = 2^bits-1 for bits ∈ {2,4,5,6,8};
// compare against a plain per-token absmax int8 K (no var-norm) ≈ "plain Q8 K".
// Report reconstruction SNR (dB) averaged over many random tiles. SNR is a
// codec-level proxy for the attention-score / logit-KLD impact; it isolates the
// rate-distortion curve, which is exactly the build-or-not decision input.

use hipfire_kvquant::kvarn::{dequantize_tile, quantize_tile_qmax};

fn snr_db(orig: &[f32], recon: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (o, r) in orig.iter().zip(recon) {
        sig += (*o as f64) * (*o as f64);
        let e = *o as f64 - *r as f64;
        err += e * e;
    }
    10.0 * (sig / err.max(1e-30)).log10()
}

// Plain Q8 K baseline: per-token (per-column) symmetric absmax int8, no var-norm.
// This is the best-case "plain Q8 K" — the whole point of comparison.
fn plain_q8_pertoken(tile: &[f32], r_dim: usize, c_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; r_dim * c_dim];
    for c in 0..c_dim {
        let mut amax = 0.0f32;
        for r in 0..r_dim {
            amax = amax.max(tile[r * c_dim + c].abs());
        }
        let scale = (amax / 127.0).max(1e-8);
        for r in 0..r_dim {
            let q = (tile[r * c_dim + c] / scale).round().clamp(-127.0, 127.0);
            out[r * c_dim + c] = q * scale;
        }
    }
    out
}

// Deterministic LCG so the sweep is reproducible without an rng dep.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0 // ~U(-1,1)
    }
    // Box-Muller-ish standard normal from two uniforms.
    fn next_normal(&mut self) -> f32 {
        let u1 = ((self.next_f32() + 1.0) * 0.5).clamp(1e-6, 1.0);
        let u2 = (self.next_f32() + 1.0) * 0.5;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

// Synthetic K tile: per-channel (per-row) log-normal scale (heavy tail; a few
// outlier channels dominate), mild per-token scale, Gaussian core.
fn make_k_tile(rng: &mut Lcg, r_dim: usize, c_dim: usize, outlier_frac: f32) -> Vec<f32> {
    let mut chan_scale = vec![0.0f32; r_dim];
    for cs in chan_scale.iter_mut() {
        // log-normal(0, 0.9): median 1, occasional ~5-10x channels.
        *cs = (0.9 * rng.next_normal()).exp();
    }
    // A handful of extreme outlier channels (~20x) — the classic K spikes.
    let n_out = ((r_dim as f32) * outlier_frac).ceil() as usize;
    for i in 0..n_out {
        chan_scale[(i * 7 + 3) % r_dim] *= 20.0;
    }
    let mut tok_scale = vec![0.0f32; c_dim];
    for ts in tok_scale.iter_mut() {
        *ts = (0.3 * rng.next_normal()).exp();
    }
    let mut t = vec![0.0f32; r_dim * c_dim];
    for r in 0..r_dim {
        for c in 0..c_dim {
            t[r * c_dim + c] = chan_scale[r] * tok_scale[c] * rng.next_normal();
        }
    }
    t
}

// ── Attention-selection sensitivity ──────────────────────────────────────
// K quant only matters insofar as it changes WHICH keys attention selects.
// For a query Q over L keys, compare softmax(Q·K/√d) with exact vs quantized K:
//   - attn KL (nats): how much the attention distribution shifts,
//   - top-1 agreement: does the argmax key survive,
//   - top-8 overlap: does the selected set survive.
// This is the "binary selection" view — the metric that actually gates logits.
fn attn_softmax(q: &[f32], k: &[f32], d: usize, l: usize) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut s = vec![0.0f32; l];
    for key in 0..l {
        let mut dot = 0.0f32;
        for i in 0..d {
            dot += q[i] * k[i * l + key]; // k is [d x l] (channel-major, l keys)
        }
        s[key] = dot * scale;
    }
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f32;
    for v in s.iter_mut() {
        *v = (*v - m).exp();
        z += *v;
    }
    for v in s.iter_mut() {
        *v /= z;
    }
    s
}

fn kl_nats(p: &[f32], q: &[f32]) -> f64 {
    let mut kl = 0.0f64;
    for (pi, qi) in p.iter().zip(q) {
        if *pi > 1e-12 {
            kl += *pi as f64 * ((*pi as f64) / (*qi as f64).max(1e-12)).ln();
        }
    }
    kl
}

fn topk_set(p: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
    idx.truncate(k);
    idx
}

fn attention_sensitivity(head_dim: usize, group: usize) {
    let l_keys = 512usize; // 4 GROUP tiles of context
    let n_tiles = l_keys / group;
    let queries = 96usize;
    println!("\n--- attention selection sensitivity [head_dim={head_dim}, L={l_keys} keys] ---");
    println!("  bits   attn-KL(nats)   top1-agree%   top8-overlap%");
    for &(bits, qmax) in &[(2usize, 3.0f32), (4, 15.0), (8, 255.0), (255usize, -1.0f32)] {
        // bits==255 sentinel = plain per-token Q8 K (no var-norm).
        let mut kl_acc = 0.0f64;
        let mut t1 = 0.0f64;
        let mut t8 = 0.0f64;
        for tr in 0..queries {
            let mut rng = Lcg(0xDEADBEEF ^ (tr as u64).wrapping_mul(0x9E3779B1));
            // Build L keys as n_tiles concatenated [head_dim x group] tiles.
            let mut k_exact = vec![0.0f32; head_dim * l_keys];
            let mut k_quant = vec![0.0f32; head_dim * l_keys];
            for t in 0..n_tiles {
                let tile = make_k_tile(&mut rng, head_dim, group, 0.03);
                let deq = if qmax < 0.0 {
                    plain_q8_pertoken(&tile, head_dim, group)
                } else {
                    dequantize_tile(&quantize_tile_qmax(&tile, head_dim, group, qmax))
                };
                for r in 0..head_dim {
                    for c in 0..group {
                        k_exact[r * l_keys + (t * group + c)] = tile[r * group + c];
                        k_quant[r * l_keys + (t * group + c)] = deq[r * group + c];
                    }
                }
            }
            // A query correlated with the key channels (so attention is peaked,
            // not uniform — the regime where selection actually matters).
            let mut q = vec![0.0f32; head_dim];
            for qi in q.iter_mut() {
                *qi = rng.next_normal();
            }
            let a_exact = attn_softmax(&q, &k_exact, head_dim, l_keys);
            let a_quant = attn_softmax(&q, &k_quant, head_dim, l_keys);
            kl_acc += kl_nats(&a_exact, &a_quant);
            let e1 = topk_set(&a_exact, 1);
            let q1 = topk_set(&a_quant, 1);
            if e1[0] == q1[0] {
                t1 += 1.0;
            }
            let e8: std::collections::HashSet<usize> = topk_set(&a_exact, 8).into_iter().collect();
            let q8: std::collections::HashSet<usize> = topk_set(&a_quant, 8).into_iter().collect();
            t8 += e8.intersection(&q8).count() as f64 / 8.0;
        }
        let label = if bits == 255 {
            "Q8*".to_string()
        } else {
            bits.to_string()
        };
        println!(
            "  {label:>4}   {:>11.5}   {:>10.1}   {:>12.1}",
            kl_acc / queries as f64,
            100.0 * t1 / queries as f64,
            100.0 * t8 / queries as f64
        );
    }
}

fn main() {
    let group = 128usize;
    let trials = 64usize;
    for &head_dim in &[128usize, 256] {
        println!("\n=== K tile [head_dim={head_dim} x GROUP={group}], {trials} trials ===");
        println!("  bits  qmax   KVarN-SNR(dB)   bytes/tok(K)");
        let bit_rungs = [
            (2usize, 3.0f32),
            (4, 15.0),
            (5, 31.0),
            (6, 63.0),
            (8, 255.0),
        ];
        for &(bits, qmax) in &bit_rungs {
            let mut acc = 0.0f64;
            for tr in 0..trials {
                let mut rng = Lcg(0x9E3779B97F4A7C15 ^ (tr as u64).wrapping_mul(0x1234567));
                let tile = make_k_tile(&mut rng, head_dim, group, 0.03);
                let qt = quantize_tile_qmax(&tile, head_dim, group, qmax);
                let deq = dequantize_tile(&qt);
                acc += snr_db(&tile, &deq);
            }
            // KVarN K bytes/token: (head_dim*bits/8 codes + per-channel scale/zp
            // fp16 amortized over GROUP + per-token s_col fp16).
            let k_bytes_tok = head_dim as f64 * bits as f64 / 8.0
                + (head_dim as f64 * 2.0 * 2.0) / group as f64
                + 2.0;
            println!(
                "  {bits:>3}  {qmax:>5.0}   {:>10.2}      {:>6.2}",
                acc / trials as f64,
                k_bytes_tok
            );
        }
        // Plain Q8 K baseline (per-token absmax int8, no var-norm).
        let mut accq = 0.0f64;
        for tr in 0..trials {
            let mut rng = Lcg(0x9E3779B97F4A7C15 ^ (tr as u64).wrapping_mul(0x1234567));
            let tile = make_k_tile(&mut rng, head_dim, group, 0.03);
            let recon = plain_q8_pertoken(&tile, head_dim, group);
            accq += snr_db(&tile, &recon);
        }
        // Plain Q8 K bytes/token: head_dim int8 + per-token fp16 scale (per 32-blk
        // in the real kernel; approx one scale/token here).
        let q8_bytes_tok = head_dim as f64 + 2.0;
        println!(
            "  Q8   (255)   {:>10.2}      {:>6.2}   <- plain per-token int8, NO var-norm",
            accq / trials as f64,
            q8_bytes_tok
        );
        attention_sensitivity(head_dim, group);
    }
}
