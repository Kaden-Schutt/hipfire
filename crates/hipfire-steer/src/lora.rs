// SPDX-License-Identifier: Apache-2.0
//! Rank-1 residual LoRA: materialize abliteration directions as a stackable,
//! intensity-adjustable adapter.
//!
//! The abliteration apply is linear, so the per-block directions `{v_L}` factor
//! into a low-rank delta. In the residual-stream convention (exact to the
//! block-boundary hook), each block's delta is rank-1 with `A = vᵀ`, `B = -v`, so
//! applying it is `Δx = scale·B(A·x) = -scale·(vᵀx)·v` — identical to
//! [`crate::apply_direction`] in `Ablate` mode. The per-adapter `scale` is the
//! steer `strength`, now adjustable at runtime ("intensity"), and a stack of
//! adapters sums (additively, order-independent).
//!
//! This module is the host core (types + export + a reference stack apply); the
//! GPU stack apply, daemon ops, and `.lora.hfq` container are the next increments.
//! See `docs/plans/2026-06-30-abliteration-lora.md`.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::SteerMode;

/// Where a low-rank delta is applied. This increment supports residual-stream
/// targets only (exact to the steer hook); projection targets (`o_proj`/
/// `down_proj`) are the portable follow-up (see the plan).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LoraTarget {
    /// Block-boundary residual after the given layer.
    Residual { layer: usize },
}

/// One low-rank delta, applied as `y += scale · B (A·y_in)`. Stored row-major:
/// `a[r]` is row r of `A` (length `d_in`) and `b[r]` is column r of `B` (length
/// `d_out`). For abliteration this is rank-1 with `a = [v]` (`= vᵀ`) and `b = [-v]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraDelta {
    pub target: LoraTarget,
    /// `a[r]`: row r of `A`, length `d_in`.
    pub a: Vec<Vec<f32>>,
    /// `b[r]`: column r of `B`, length `d_out`.
    pub b: Vec<Vec<f32>>,
}

impl LoraDelta {
    /// Low-rank rank `r` (rows of `A`). Abliteration deltas are rank 1.
    pub fn rank(&self) -> usize {
        self.a.len()
    }
}

/// Provenance for a materialized adapter.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LoraMeta {
    /// `"ablate"` (the only rank-1 form today).
    pub mode: String,
    /// The derive-time strength the adapter's nominal `scale` was seeded from.
    pub strength: f32,
    pub layer_start: usize,
    pub layer_end: usize,
    pub hidden: usize,
    /// Base model the directions were derived against — loads must compat-gate it.
    #[serde(default)]
    pub base_model_sha256: Option<String>,
}

/// A loadable adapter: per-target deltas plus a runtime `scale` (intensity). The
/// applied contribution is `scale · B(A·x)` summed over `deltas`. `scale` is the
/// live dial — `0` disables, `1` is nominal, `>1` amplifies, `<0` inverts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoraAdapter {
    pub id: String,
    pub scale: f32,
    pub deltas: Vec<LoraDelta>,
    pub meta: LoraMeta,
}

/// Materialize per-block abliteration `directions` as a rank-1 residual adapter.
///
/// One `Residual(L)` delta per block in `layer_range`, with `A = v_L` and
/// `B = -v_L`; the adapter `scale` is seeded to `strength` (the live intensity).
/// `directions[L]` must be the unit direction for block `L`.
///
/// Ablate-only: `Steer` (`x += s·v`) is an additive bias, not a `B(A·x)` delta, so
/// it cannot be a rank-1 LoRA (it needs a bias adapter — future work).
pub fn abliteration_adapter(
    id: impl Into<String>,
    directions: &[Vec<f32>],
    mode: SteerMode,
    strength: f32,
    layer_range: Range<usize>,
) -> Result<LoraAdapter, String> {
    if mode != SteerMode::Ablate {
        return Err(
            "lora: rank-1 export supports ablate (projective) only — steer \
                    is an additive bias, not a B(A·x) delta"
                .to_string(),
        );
    }
    // Width is set by the directions actually in range (the full `directions`
    // slice may carry placeholder entries for out-of-range layers).
    let hidden = directions.get(layer_range.start).map_or(0, |d| d.len());
    let mut deltas = Vec::with_capacity(layer_range.len());
    for layer in layer_range.clone() {
        let v = directions.get(layer).ok_or_else(|| {
            format!(
                "lora: layer {layer} out of range ({} directions)",
                directions.len()
            )
        })?;
        if v.len() != hidden {
            return Err(format!(
                "lora: direction {layer} has width {} != {hidden}",
                v.len()
            ));
        }
        deltas.push(LoraDelta {
            target: LoraTarget::Residual { layer },
            a: vec![v.clone()],
            b: vec![v.iter().map(|&x| -x).collect()],
        });
    }
    Ok(LoraAdapter {
        id: id.into(),
        scale: strength,
        deltas,
        meta: LoraMeta {
            mode: "ablate".to_string(),
            strength,
            layer_start: layer_range.start,
            layer_end: layer_range.end,
            hidden,
            base_model_sha256: None,
        },
    })
}

/// Apply a stack of adapters to the residual `x` at block-boundary `layer`.
///
/// Every adapter's `Residual(layer)` deltas are evaluated against the ORIGINAL `x`
/// and summed — so stacking is the additive, order-independent LoRA sum
/// (`x += Σ_k scale_k·B_k(A_k·x)`), not a sequential composition. (Projective
/// ablations only compose linearly when their directions are orthogonal; stack
/// non-orthogonal ablations at your own risk — orthogonalize at export.)
///
/// Host reference for the GPU stack apply; equals [`crate::apply_direction`] in
/// `Ablate` mode for a single rank-1 ablate adapter.
pub fn apply_residual_stack(stack: &[LoraAdapter], layer: usize, x: &mut [f32]) {
    let mut acc = vec![0.0f32; x.len()];
    for ad in stack {
        if ad.scale == 0.0 {
            continue;
        }
        for d in &ad.deltas {
            if d.target == (LoraTarget::Residual { layer }) {
                accumulate_delta(d, ad.scale, x, &mut acc);
            }
        }
    }
    for (xi, &a) in x.iter_mut().zip(acc.iter()) {
        *xi += a;
    }
}

/// Add `scale · B(A·x_orig)` into `acc` (reads `x_orig`, never the running sum).
fn accumulate_delta(d: &LoraDelta, scale: f32, x_orig: &[f32], acc: &mut [f32]) {
    for (arow, bcol) in d.a.iter().zip(d.b.iter()) {
        let ax: f32 = arow.iter().zip(x_orig.iter()).map(|(&a, &xi)| a * xi).sum();
        let coef = scale * ax;
        for (ai, &bi) in acc.iter_mut().zip(bcol.iter()) {
            *ai += coef * bi;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apply_direction;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn single_ablate_adapter_equals_apply_direction() {
        let v = unit(&[0.3, -0.4, 0.5, 0.2]);
        let dirs = vec![v.clone()];
        let ad = abliteration_adapter("ablit", &dirs, SteerMode::Ablate, 0.8, 0..1).unwrap();

        let x0 = vec![1.0, 2.0, -1.5, 0.7];
        let mut x_lora = x0.clone();
        apply_residual_stack(std::slice::from_ref(&ad), 0, &mut x_lora);

        let mut x_ref = x0.clone();
        apply_direction(&mut x_ref, &v, SteerMode::Ablate, 0.8);

        for (a, b) in x_lora.iter().zip(x_ref.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn orthogonal_stack_sum_equals_sequential() {
        // v1 ⊥ v2 ⇒ summing the two ablate deltas == applying them in sequence.
        let v1 = unit(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = unit(&[0.0, 1.0, 0.0, 0.0]);
        let a1 = abliteration_adapter("a1", &[v1.clone()], SteerMode::Ablate, 0.6, 0..1).unwrap();
        let a2 = abliteration_adapter("a2", &[v2.clone()], SteerMode::Ablate, 0.9, 0..1).unwrap();

        let x0 = vec![2.0, -3.0, 1.0, 0.5];
        let mut x_stack = x0.clone();
        apply_residual_stack(&[a1, a2], 0, &mut x_stack);

        let mut x_seq = x0.clone();
        apply_direction(&mut x_seq, &v1, SteerMode::Ablate, 0.6);
        apply_direction(&mut x_seq, &v2, SteerMode::Ablate, 0.9);

        for (a, b) in x_stack.iter().zip(x_seq.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn zero_scale_is_a_noop() {
        let v = unit(&[0.5, 0.5, 0.5, 0.5]);
        let mut ad = abliteration_adapter("z", &[v], SteerMode::Ablate, 1.0, 0..1).unwrap();
        ad.scale = 0.0;
        let x0 = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = x0.clone();
        apply_residual_stack(std::slice::from_ref(&ad), 0, &mut x);
        assert_eq!(x, x0);
    }

    #[test]
    fn scale_dials_intensity() {
        // The adapter scale linearly dials the ablation strength.
        let v = unit(&[0.2, 0.9, -0.1, 0.3]);
        let dirs = vec![v.clone()];
        let mut ad = abliteration_adapter("s", &dirs, SteerMode::Ablate, 1.0, 0..1).unwrap();
        ad.scale = 0.5;
        let x0 = vec![1.0, -1.0, 2.0, 0.0];
        let mut x_half = x0.clone();
        apply_residual_stack(std::slice::from_ref(&ad), 0, &mut x_half);

        let mut x_ref = x0.clone();
        apply_direction(&mut x_ref, &v, SteerMode::Ablate, 0.5);
        for (a, b) in x_half.iter().zip(x_ref.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn only_applies_to_matching_layer() {
        let v = unit(&[1.0, 1.0, 0.0, 0.0]);
        let ad = abliteration_adapter("L", &[Vec::new(), v], SteerMode::Ablate, 1.0, 1..2).unwrap();
        let x0 = vec![1.0, 2.0, 3.0, 4.0];
        // Delta targets Residual(1); applying at layer 0 must be a no-op.
        let mut x = x0.clone();
        apply_residual_stack(std::slice::from_ref(&ad), 0, &mut x);
        assert_eq!(x, x0);
        // At layer 1 it applies.
        apply_residual_stack(std::slice::from_ref(&ad), 1, &mut x);
        assert_ne!(x, x0);
    }

    #[test]
    fn steer_mode_is_rejected() {
        let v = unit(&[1.0, 0.0]);
        let err = abliteration_adapter("x", &[v], SteerMode::Steer, 1.0, 0..1).unwrap_err();
        assert!(err.contains("ablate"));
    }

    #[test]
    fn adapter_round_trips_through_json() {
        let v = unit(&[0.1, 0.2, 0.3, 0.9]);
        let ad = abliteration_adapter("rt", &[v], SteerMode::Ablate, 0.7, 0..1).unwrap();
        let json = serde_json::to_string(&ad).unwrap();
        let back: LoraAdapter = serde_json::from_str(&json).unwrap();
        assert_eq!(ad, back);
        assert_eq!(back.deltas[0].rank(), 1);
        assert_eq!(back.deltas[0].target, LoraTarget::Residual { layer: 0 });
    }
}
