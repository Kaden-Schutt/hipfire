// SPDX-License-Identifier: Apache-2.0
// hipfire-steer — refusal-direction steering / abliteration.
//
// See docs/plans/2026-06-29-refusal-direction-steering.md.
//
// The whole technique reduces to one runtime op on the *residual stream* at the
// **block boundary** (after a transformer block's residual has settled), where
// the residual is uniformly an addressable f32 buffer across every hipfire arch
// — so MoE/attention kernel fusion is irrelevant (we read/inject *after* the
// block, never inside a fused kernel).
//
// Two phases share the same block-boundary hook:
//   * CAPTURE  — read the residual to accumulate per-block means for a +set and
//                a -set, from which a contrastive direction is derived.
//   * APPLY    — mutate the residual with that direction:
//                  Steer (additive):   x += alpha * v
//                  Ablate (projective): x -= lambda * (v . x) * v   (v unit-norm)
//
// Algebraic note: projective ablation of the *activation* equals directional
// ablation of the *weight* (`W·a - λ v (vᵀW·a) = o - λ v (vᵀo)`), so we get
// Heretic-style abliteration with NO weight edit and NO re-quantization.
//
// Phase-1 STUB BOUNDARY:
//   * Session model, control API, capture accumulation, direction derivation,
//     and the pure-Rust apply math are complete and unit-tested.
//   * APPLY currently uses a host round-trip (download → compute → upload) as a
//     correct-but-slow reference path. Replacing it with on-GPU ops
//     (`upload_f32` the direction once + `scaled_add_inplace_gpu_scalar_f32` /
//     a fused projective-subtract kernel) is the first Phase-1 follow-up.
//   * Granularity is block-boundary only (uniform, MoE-agnostic). Per-component
//     (attn-out vs mlp-out) is deferred — see the plan.

use std::cell::RefCell;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

pub mod driver;

/// How a direction is applied to the residual stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteerMode {
    /// `x += strength * v` — push the residual along the direction (steering).
    Steer,
    /// `x -= strength * (v·x) * v` — remove the component along the direction
    /// (abliteration). Assumes `v` is unit-norm (the derivation guarantees it).
    Ablate,
}

/// A fully-derived steering configuration ready to apply.
///
/// `directions[layer_idx]` is the unit-norm direction for that block. Blocks
/// outside `layer_range` are left untouched.
#[derive(Clone, Debug)]
pub struct SteerSpec {
    pub directions: Vec<Vec<f32>>,
    pub mode: SteerMode,
    pub strength: f32,
    pub layer_range: Range<usize>,
}

/// Per-block running sum of residuals (host f64 for accumulation precision),
/// used during a CAPTURE session.
struct CaptureAcc {
    /// `sums[layer_idx]` has length `hidden`.
    sums: Vec<Vec<f64>>,
    /// Number of prompts folded in (shared across layers).
    count: u64,
    hidden: usize,
}

impl CaptureAcc {
    fn new(num_layers: usize, hidden: usize) -> Self {
        Self {
            sums: vec![vec![0.0; hidden]; num_layers],
            count: 0,
            hidden,
        }
    }

    /// Fold one residual vector (`hidden` elements) at `layer_idx`.
    fn add(&mut self, layer_idx: usize, x: &[f32]) {
        debug_assert_eq!(x.len(), self.hidden);
        let row = &mut self.sums[layer_idx];
        for (s, &v) in row.iter_mut().zip(x.iter()) {
            *s += v as f64;
        }
    }

    /// `count` is bumped once per prompt, i.e. once after the LAST block. We key
    /// it off layer 0 so a single prompt's pass over all blocks counts once.
    fn note_prompt(&mut self, layer_idx: usize) {
        if layer_idx == 0 {
            self.count += 1;
        }
    }

    /// Per-block means as f32. Panics-free: empty capture yields zeros.
    fn means(&self) -> CaptureMeans {
        let n = self.count.max(1) as f64;
        CaptureMeans(
            self.sums
                .iter()
                .map(|row| row.iter().map(|&s| (s / n) as f32).collect())
                .collect(),
        )
    }
}

/// Per-block mean residual for one prompt set. `0[layer_idx]` has length `hidden`.
#[derive(Clone, Debug)]
pub struct CaptureMeans(pub Vec<Vec<f32>>);

enum Session {
    Inactive,
    Capturing(CaptureAcc),
    Applying(SteerSpec),
}

static SESSION: OnceLock<RwLock<Session>> = OnceLock::new();
/// Fast-path gate so the hot forward path pays only one relaxed atomic load when
/// steering is inactive (the common case during normal serving).
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// Bumped on every session change so the per-thread GPU apply cache
/// (`APPLY_CACHE`, which can't live in the `Sync` static because `GpuTensor` is
/// `!Sync`) knows when to refresh its uploaded directions.
static EPOCH: AtomicU64 = AtomicU64::new(0);

fn session() -> &'static RwLock<Session> {
    SESSION.get_or_init(|| RwLock::new(Session::Inactive))
}

fn set_session(s: Session) {
    let active = !matches!(s, Session::Inactive);
    *session().write().unwrap() = s;
    EPOCH.fetch_add(1, Ordering::Release);
    ACTIVE.store(active, Ordering::Release);
}

// ── Control API ─────────────────────────────────────────────────────────────

/// Begin a CAPTURE session: subsequent forwards accumulate per-block residual
/// means. Run the +set, call [`finish_capture`], then the -set similarly.
pub fn begin_capture(num_layers: usize, hidden: usize) {
    set_session(Session::Capturing(CaptureAcc::new(num_layers, hidden)));
}

/// End a CAPTURE session and return the accumulated per-block means (`None` if
/// no capture was active).
pub fn finish_capture() -> Option<CaptureMeans> {
    let means = match &*session().read().unwrap() {
        Session::Capturing(acc) => Some(acc.means()),
        _ => None,
    };
    if means.is_some() {
        set_session(Session::Inactive);
    }
    means
}

/// Begin an APPLY session: subsequent forwards steer/ablate at each in-range
/// block boundary.
pub fn begin_apply(spec: SteerSpec) {
    set_session(Session::Applying(spec));
}

/// Tear down any active session.
pub fn clear() {
    set_session(Session::Inactive);
}

/// Whether a capture or apply session is currently active.
pub fn is_active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

// ── Direction derivation ────────────────────────────────────────────────────

/// Derive per-block unit-norm contrastive directions:
/// `dir_L = normalize(mean_bad_L - mean_good_L)`. When `orthogonalize` is set,
/// the component along the "good" direction is projected out first (projected
/// abliteration), reducing collateral damage to benign behaviour.
pub fn derive_directions(
    good: &CaptureMeans,
    bad: &CaptureMeans,
    orthogonalize: bool,
) -> Vec<Vec<f32>> {
    good.0
        .iter()
        .zip(bad.0.iter())
        .map(|(g, b)| {
            let mut dir: Vec<f32> = b.iter().zip(g.iter()).map(|(&bi, &gi)| bi - gi).collect();
            normalize(&mut dir);
            if orthogonalize {
                let mut good_dir = g.clone();
                normalize(&mut good_dir);
                let proj = dot(&dir, &good_dir);
                for (d, &gd) in dir.iter_mut().zip(good_dir.iter()) {
                    *d -= proj * gd;
                }
                normalize(&mut dir);
            }
            dir
        })
        .collect()
}

// ── Hook entry points (called from arch forwards at the block boundary) ──────

/// Single-vector block-boundary hook for the decode/AR path. `x` is the
/// `[hidden]` residual after block `layer_idx`.
pub fn maybe_steer_block(gpu: &mut Gpu, x: &GpuTensor, layer_idx: usize) -> HipResult<()> {
    if !is_active() {
        return Ok(());
    }
    let epoch = EPOCH.load(Ordering::Acquire);
    match &mut *session().write().unwrap() {
        Session::Inactive => {}
        Session::Capturing(acc) => {
            let host = gpu.download_f32(x)?;
            acc.add(layer_idx, &host);
            acc.note_prompt(layer_idx);
        }
        Session::Applying(spec) => {
            if spec.layer_range.contains(&layer_idx) {
                apply_on_gpu(gpu, x, layer_idx, spec, epoch)?;
            }
        }
    }
    Ok(())
}

/// Batched block-boundary hook for the prefill path. `x_batch` is the
/// `[num_positions * hidden]` residual after block `layer_idx`.
///
/// Convention (matches Heretic / the plan's open question): CAPTURE folds in the
/// LAST position only (the next-token residual); APPLY mutates ALL positions.
pub fn maybe_steer_block_batched(
    gpu: &mut Gpu,
    x_batch: &GpuTensor,
    layer_idx: usize,
    num_positions: usize,
    hidden: usize,
) -> HipResult<()> {
    if !is_active() {
        return Ok(());
    }
    match &mut *session().write().unwrap() {
        Session::Inactive => {}
        Session::Capturing(acc) => {
            let host = gpu.download_f32(x_batch)?;
            let last = (num_positions - 1) * hidden;
            acc.add(layer_idx, &host[last..last + hidden]);
            acc.note_prompt(layer_idx);
        }
        Session::Applying(spec) => {
            if spec.layer_range.contains(&layer_idx) {
                // Prefill is one-shot per request, and the search loop scores via
                // single-token decode forwards, so this host round-trip is amortized
                // — the per-token decode path is the one moved on-GPU. A batched
                // on-GPU apply would need a broadcast axpy (steer) and a no-sigmoid
                // per-row scaled-add (ablate), neither of which exists yet. See plan.
                let mut host = gpu.download_f32(x_batch)?;
                let dir = &spec.directions[layer_idx];
                for p in 0..num_positions {
                    let off = p * hidden;
                    apply_direction(&mut host[off..off + hidden], dir, spec.mode, spec.strength);
                }
                write_back(gpu, x_batch, &host)?;
            }
        }
    }
    Ok(())
}

// ── On-GPU apply (decode/AR path) ───────────────────────────────────────────

thread_local! {
    /// Per-thread GPU resources for the apply path. Lives here rather than in the
    /// `Sync` SESSION static because `GpuTensor` is `!Sync`. Refreshed when EPOCH
    /// moves; buffers are reused across epochs when dims match (no per-trial leak).
    static APPLY_CACHE: RefCell<Option<ApplyCache>> = const { RefCell::new(None) };
}

struct ApplyCache {
    epoch: u64,
    num_layers: usize,
    hidden: usize,
    /// One `[1, hidden]` unit direction per block (2-D so `gemv_f32` reads m=1, k=hidden).
    dirs: Vec<GpuTensor>,
    /// `[1]` device scalar holding `strength` (the additive steer coefficient).
    strength_buf: GpuTensor,
    /// `[1]` scratch for the ablate dot product `v·x`.
    proj_buf: GpuTensor,
    /// `[1]` scratch for the data-dependent ablate coefficient `-strength·(v·x)`.
    coef_buf: GpuTensor,
}

/// Steer/ablate one `[hidden]` residual fully on-GPU — no full-vector host bounce.
/// Reuses register-tiled `gemv_f32` (dot) + `scaled_add_inplace_gpu_scalar_f32`
/// (axpy); for ablate only a 4-byte scalar round-trips for the coefficient.
fn apply_on_gpu(
    gpu: &mut Gpu,
    x: &GpuTensor,
    layer_idx: usize,
    spec: &SteerSpec,
    epoch: u64,
) -> HipResult<()> {
    APPLY_CACHE.with(|cell| -> HipResult<()> {
        let mut slot = cell.borrow_mut();
        ensure_apply_cache(&mut slot, gpu, spec, epoch)?;
        let cache = slot.as_ref().unwrap();
        let dir = &cache.dirs[layer_idx];
        match spec.mode {
            SteerMode::Steer => {
                // x += strength * v
                gpu.scaled_add_inplace_gpu_scalar_f32(x, dir, &cache.strength_buf)?;
            }
            SteerMode::Ablate => {
                // proj = v·x  (dir is [1, hidden] → gemv yields a [1] scalar)
                gpu.gemv_f32(dir, x, &cache.proj_buf)?;
                let proj = gpu.download_f32(&cache.proj_buf)?[0];
                let coef = -spec.strength * proj;
                gpu.memcpy_htod_auto(&cache.coef_buf.buf, &coef.to_le_bytes())?;
                // x += (-strength · proj) * v
                gpu.scaled_add_inplace_gpu_scalar_f32(x, dir, &cache.coef_buf)?;
            }
        }
        Ok(())
    })
}

/// Build (first use / dim change) or refresh (epoch change) the per-thread GPU
/// apply cache. Buffers are reused across epochs when dims match, so a search
/// loop calling `begin_apply` repeatedly neither reallocates nor leaks.
fn ensure_apply_cache(
    slot: &mut Option<ApplyCache>,
    gpu: &mut Gpu,
    spec: &SteerSpec,
    epoch: u64,
) -> HipResult<()> {
    let num_layers = spec.directions.len();
    let hidden = spec.directions.first().map_or(0, |d| d.len());

    let dims_match = slot
        .as_ref()
        .is_some_and(|c| c.num_layers == num_layers && c.hidden == hidden);

    if !dims_match {
        let mut dirs = Vec::with_capacity(num_layers);
        for d in &spec.directions {
            dirs.push(gpu.upload_f32(d, &[1, hidden])?);
        }
        *slot = Some(ApplyCache {
            epoch,
            num_layers,
            hidden,
            strength_buf: gpu.full_f32(&[1], spec.strength)?,
            proj_buf: gpu.alloc_tensor(&[1], DType::F32)?,
            coef_buf: gpu.alloc_tensor(&[1], DType::F32)?,
            dirs,
        });
        return Ok(());
    }

    let cache = slot.as_mut().unwrap();
    if cache.epoch != epoch {
        for (buf, d) in cache.dirs.iter().zip(spec.directions.iter()) {
            gpu.memcpy_htod_auto(&buf.buf, &f32_bytes(d))?;
        }
        gpu.memcpy_htod_auto(&cache.strength_buf.buf, &spec.strength.to_le_bytes())?;
        cache.epoch = epoch;
    }
    Ok(())
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ── Pure math (unit-tested; no GPU) ─────────────────────────────────────────

/// Apply a direction to one residual vector in place.
pub fn apply_direction(x: &mut [f32], v: &[f32], mode: SteerMode, strength: f32) {
    debug_assert_eq!(x.len(), v.len());
    match mode {
        SteerMode::Steer => {
            for (xi, &vi) in x.iter_mut().zip(v.iter()) {
                *xi += strength * vi;
            }
        }
        SteerMode::Ablate => {
            let proj = dot(x, v) * strength;
            for (xi, &vi) in x.iter_mut().zip(v.iter()) {
                *xi -= proj * vi;
            }
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Host → device writeback for the reference apply path.
fn write_back(gpu: &mut Gpu, x: &GpuTensor, host: &[f32]) -> HipResult<()> {
    let bytes: Vec<u8> = host.iter().flat_map(|f| f.to_le_bytes()).collect();
    gpu.memcpy_htod_auto(&x.buf, &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steer_adds_scaled_direction() {
        let mut x = vec![1.0, 2.0, 3.0];
        apply_direction(&mut x, &[1.0, 0.0, 0.0], SteerMode::Steer, 2.0);
        assert_eq!(x, vec![3.0, 2.0, 3.0]);
    }

    #[test]
    fn ablate_removes_component_along_unit_direction() {
        // v is unit-norm along axis 0; full ablation (strength 1) zeros that axis.
        let mut x = vec![5.0, 2.0, 3.0];
        apply_direction(&mut x, &[1.0, 0.0, 0.0], SteerMode::Ablate, 1.0);
        assert!((x[0]).abs() < 1e-6);
        assert_eq!(&x[1..], &[2.0, 3.0]);
    }

    #[test]
    fn derive_is_unit_norm_and_points_bad_minus_good() {
        let good = CaptureMeans(vec![vec![0.0, 0.0]]);
        let bad = CaptureMeans(vec![vec![3.0, 4.0]]);
        let dirs = derive_directions(&good, &bad, false);
        let n = dot(&dirs[0], &dirs[0]).sqrt();
        assert!((n - 1.0).abs() < 1e-6);
        assert!((dirs[0][0] - 0.6).abs() < 1e-6);
        assert!((dirs[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn orthogonalize_removes_good_component() {
        // good points along axis 0; raw refusal dir has a component along it that
        // must be projected out, leaving a pure axis-1 direction.
        let good = CaptureMeans(vec![vec![1.0, 0.0]]);
        let bad = CaptureMeans(vec![vec![1.0, 1.0]]);
        let dirs = derive_directions(&good, &bad, true);
        assert!(dirs[0][0].abs() < 1e-6);
        assert!((dirs[0][1].abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn capture_means_average_over_prompts() {
        let mut acc = CaptureAcc::new(1, 2);
        acc.add(0, &[2.0, 4.0]);
        acc.note_prompt(0);
        acc.add(0, &[4.0, 8.0]);
        acc.note_prompt(0);
        let m = acc.means();
        assert_eq!(m.0[0], vec![3.0, 6.0]);
    }
}
