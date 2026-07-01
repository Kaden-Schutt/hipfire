// SPDX-License-Identifier: Apache-2.0
//! H-Neuron identification core (arXiv 2512.01797).
//!
//! A hallucination-associated neuron is an FFN neuron whose contribution to the
//! block output — quantified by **CETT** — predicts whether the model is
//! hallucinating. This crate is the model-independent host core:
//!
//! * [`cett`] — per-token CETT for one layer, matching the reference
//!   (`third_party/H-Neurons/scripts/extract_activations.py`):
//!   `CETT(j) = |act_j| · ‖W_down[:,j]‖ / ‖down_proj_output‖`.
//! * [`CettFeatures`] — accumulate per-token CETT over a response region into one
//!   `[layers · neurons]` feature vector per example (the `--method mean` aggregation).
//! * [`L1Logreg`] — a sparse L1 logistic-regression probe (proximal gradient); its
//!   positive-weight neurons are the H-Neurons (their `classifier.py`, `penalty=l1`).
//!
//! The `ffn_hidden` forward tap, the daemon capture op, and the collect/label
//! tooling live outside this crate (they need the GPU forward / the daemon).

use serde::{Deserialize, Serialize};

/// Per-neuron CETT for one layer at one token.
///
/// `act` is the down_proj INPUT (the `[intermediate]` FFN activations), `col_norm`
/// is the precomputed `‖W_down[:,j]‖` per neuron (column norms of the down_proj
/// weight), and `out_norm` is `‖down_proj_output‖` (the MLP output norm) at this
/// token. Uses `|act|` (`use_abs`) × the weight magnitude (`use_mag`), matching the
/// reference defaults.
pub fn cett(act: &[f32], col_norm: &[f32], out_norm: f32) -> Vec<f32> {
    debug_assert_eq!(act.len(), col_norm.len());
    let denom = out_norm + 1e-8;
    act.iter()
        .zip(col_norm.iter())
        .map(|(&a, &w)| (a.abs() * w) / denom)
        .collect()
}

/// Accumulates per-token CETT over a response region into a per-layer mean feature
/// for ONE example. Call [`CettFeatures::add_token`] once per (layer, token) in the
/// region, then [`CettFeatures::finish`] to get the `[layers][neurons]` mean.
#[derive(Clone, Debug)]
pub struct CettFeatures {
    /// `sums[layer][neuron]` — running sum of CETT over the region's tokens.
    sums: Vec<Vec<f64>>,
    /// Tokens folded per layer (each layer sees the same token set, but track
    /// independently so a partial forward can't silently skew the mean).
    counts: Vec<u64>,
    neurons: usize,
}

impl CettFeatures {
    pub fn new(num_layers: usize, neurons: usize) -> Self {
        Self {
            sums: vec![vec![0.0; neurons]; num_layers],
            counts: vec![0; num_layers],
            neurons,
        }
    }

    /// Fold one token's per-neuron CETT into `layer`'s running sum.
    pub fn add_token(&mut self, layer: usize, cett: &[f32]) {
        debug_assert_eq!(cett.len(), self.neurons);
        let sum = &mut self.sums[layer];
        for (s, &c) in sum.iter_mut().zip(cett.iter()) {
            *s += c as f64;
        }
        self.counts[layer] += 1;
    }

    /// Per-layer mean CETT over the folded tokens (zeros for empty layers).
    pub fn finish(&self) -> Vec<Vec<f32>> {
        self.sums
            .iter()
            .zip(self.counts.iter())
            .map(|(sum, &n)| {
                let d = n.max(1) as f64;
                sum.iter().map(|&s| (s / d) as f32).collect()
            })
            .collect()
    }

    /// Flatten [`finish`] to a single `[layers · neurons]` feature row (the probe
    /// input; matches `np.load(act).flatten()` in the reference classifier).
    pub fn feature_row(&self) -> Vec<f32> {
        self.finish().into_iter().flatten().collect()
    }
}

/// A trained sparse L1 logistic-regression probe. Positive-weight features are the
/// H-Neurons.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct L1Logreg {
    pub weights: Vec<f32>,
    pub bias: f32,
    /// Feature standardization (fit on train): `(x - mean) / std`.
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// Hyperparameters for [`L1Logreg::fit`].
#[derive(Clone, Copy, Debug)]
pub struct FitConfig {
    /// L1 strength (larger ⇒ sparser). Analogous to `1/C` in sklearn.
    pub l1: f32,
    pub lr: f32,
    pub iters: usize,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            l1: 1e-3,
            lr: 0.5,
            iters: 300,
        }
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn soft_threshold(v: f32, t: f32) -> f32 {
    if v > t {
        v - t
    } else if v < -t {
        v + t
    } else {
        0.0
    }
}

impl L1Logreg {
    /// Fit L1-regularized logistic regression on standardized features via
    /// proximal-gradient (ISTA): a gradient step on the logistic loss followed by
    /// the L1 soft-threshold prox. `x` is `[n_samples][n_features]`, `y` is 0/1.
    pub fn fit(x: &[Vec<f32>], y: &[u8], cfg: FitConfig) -> Self {
        let n = x.len();
        let d = x.first().map_or(0, |r| r.len());
        assert_eq!(n, y.len());

        // Standardize features (column-wise) — L1 penalties assume comparable scales.
        let mut mean = vec![0.0f32; d];
        for row in x {
            for (m, &v) in mean.iter_mut().zip(row.iter()) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= n.max(1) as f32;
        }
        let mut std = vec![0.0f32; d];
        for row in x {
            for (s, (&v, &m)) in std.iter_mut().zip(row.iter().zip(mean.iter())) {
                let dv = v - m;
                *s += dv * dv;
            }
        }
        for s in &mut std {
            *s = (*s / n.max(1) as f32).sqrt().max(1e-6);
        }

        let z: Vec<Vec<f32>> = x
            .iter()
            .map(|row| {
                row.iter()
                    .zip(mean.iter().zip(std.iter()))
                    .map(|(&v, (&m, &s))| (v - m) / s)
                    .collect()
            })
            .collect();

        let mut w = vec![0.0f32; d];
        let mut b = 0.0f32;
        let inv_n = 1.0 / n.max(1) as f32;
        for _ in 0..cfg.iters {
            // Residuals r_i = σ(z_i·w + b) − y_i.
            let mut r = vec![0.0f32; n];
            for (i, zi) in z.iter().enumerate() {
                let dot: f32 = zi.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
                r[i] = sigmoid(dot + b) - y[i] as f32;
            }
            // Gradient of the logistic loss.
            let mut gw = vec![0.0f32; d];
            for (i, zi) in z.iter().enumerate() {
                let ri = r[i] * inv_n;
                for (g, &zij) in gw.iter_mut().zip(zi.iter()) {
                    *g += ri * zij;
                }
            }
            let gb: f32 = r.iter().sum::<f32>() * inv_n;
            // Gradient step + L1 prox on the weights (bias unpenalized).
            for (wi, &gi) in w.iter_mut().zip(gw.iter()) {
                *wi = soft_threshold(*wi - cfg.lr * gi, cfg.lr * cfg.l1);
            }
            b -= cfg.lr * gb;
        }

        Self {
            weights: w,
            bias: b,
            mean,
            std,
        }
    }

    /// Predicted P(y=1) for a raw (unstandardized) feature row.
    pub fn predict_proba(&self, x: &[f32]) -> f32 {
        let z: f32 = x
            .iter()
            .zip(
                self.weights
                    .iter()
                    .zip(self.mean.iter().zip(self.std.iter())),
            )
            .map(|(&v, (&w, (&m, &s)))| w * (v - m) / s)
            .sum();
        sigmoid(z + self.bias)
    }

    /// Classification accuracy on `(x, y)` at threshold 0.5.
    pub fn accuracy(&self, x: &[Vec<f32>], y: &[u8]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        let correct = x
            .iter()
            .zip(y.iter())
            .filter(|(row, &yi)| (self.predict_proba(row) >= 0.5) as u8 == yi)
            .count();
        correct as f32 / x.len() as f32
    }

    /// Indices of the H-Neurons: features with a positive weight (their sign
    /// convention — positive weight ⇒ drives the hallucination class).
    pub fn h_neurons(&self) -> Vec<usize> {
        self.weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Count of non-zero weights (the probe's sparsity).
    pub fn nonzero(&self) -> usize {
        self.weights.iter().filter(|&&w| w != 0.0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cett_matches_reference_formula() {
        // |act|·col_norm / (out_norm+eps).
        let act = [2.0, -3.0, 0.5];
        let col = [1.0, 2.0, 4.0];
        let c = cett(&act, &col, 10.0);
        // ~ [2*1, 3*2, 0.5*4] / 10 = [0.2, 0.6, 0.2]
        assert!((c[0] - 0.2).abs() < 1e-6);
        assert!((c[1] - 0.6).abs() < 1e-6);
        assert!((c[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn cett_features_mean_over_tokens() {
        let mut f = CettFeatures::new(2, 3);
        f.add_token(0, &[2.0, 4.0, 6.0]);
        f.add_token(0, &[4.0, 8.0, 12.0]);
        f.add_token(1, &[1.0, 1.0, 1.0]);
        let feat = f.finish();
        assert_eq!(feat[0], vec![3.0, 6.0, 9.0]); // layer 0 mean of the two tokens
        assert_eq!(feat[1], vec![1.0, 1.0, 1.0]); // layer 1 single token
        assert_eq!(f.feature_row().len(), 6); // 2 layers × 3 neurons flattened
    }

    #[test]
    fn l1_probe_recovers_sparse_signal_and_classifies() {
        // 40 samples, 20 features; only features 3 and 7 carry the label.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..40 {
            let label = (i % 2) as u8;
            let mut row = vec![0.0f32; 20];
            // deterministic pseudo-noise per (i, j)
            for (j, v) in row.iter_mut().enumerate() {
                *v = (((i * 31 + j * 17) % 13) as f32 / 13.0) - 0.5;
            }
            // predictive features: pushed by the label.
            row[3] += if label == 1 { 1.5 } else { -1.5 };
            row[7] += if label == 1 { 1.2 } else { -1.2 };
            x.push(row);
            y.push(label);
        }
        let model = L1Logreg::fit(
            &x,
            &y,
            FitConfig {
                l1: 2e-3,
                lr: 0.5,
                iters: 500,
            },
        );
        // Separates the (linearly separable) training set.
        assert!(
            model.accuracy(&x, &y) > 0.9,
            "acc {}",
            model.accuracy(&x, &y)
        );
        // The predictive features carry the largest weight magnitude.
        let mag: Vec<f32> = model.weights.iter().map(|w| w.abs()).collect();
        let top = (0..20).max_by(|&a, &b| mag[a].total_cmp(&mag[b])).unwrap();
        assert!(top == 3 || top == 7, "top feature was {top}");
        // L1 actually zeroed most features.
        assert!(model.nonzero() < 20, "nonzero {}", model.nonzero());
    }

    #[test]
    fn probe_round_trips_through_json() {
        let x = vec![vec![0.1, 0.2], vec![0.3, 0.9], vec![-0.2, 0.4]];
        let y = [0u8, 1, 0];
        let m = L1Logreg::fit(&x, &y, FitConfig::default());
        let j = serde_json::to_string(&m).unwrap();
        let back: L1Logreg = serde_json::from_str(&j).unwrap();
        assert_eq!(m.weights, back.weights);
        assert_eq!(m.bias, back.bias);
    }
}
