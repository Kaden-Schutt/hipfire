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
use hipfire_rdna::{DType, Gpu, GpuTensor};

pub mod driver;
pub mod lora;

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
    /// Last residual seen this prompt, per block — overwritten each forward,
    /// folded into `sums` by `commit`.
    current: Vec<Vec<f32>>,
    hidden: usize,
}

impl CaptureAcc {
    fn new(num_layers: usize, hidden: usize) -> Self {
        Self {
            sums: vec![vec![0.0; hidden]; num_layers],
            count: 0,
            current: vec![vec![0.0; hidden]; num_layers],
            hidden,
        }
    }

    /// Record the latest residual at `layer_idx`, overwriting any prior value
    /// this prompt. After a prompt's full prefill, `current[layer]` holds the
    /// LAST prompt-token residual — Heretic's capture position.
    fn observe(&mut self, layer_idx: usize, x: &[f32]) {
        debug_assert_eq!(x.len(), self.hidden);
        self.current[layer_idx].copy_from_slice(x);
    }

    /// Fold the current (last-token) residuals into the running means and count
    /// one prompt. Called by the harness after each prompt's forward.
    fn commit(&mut self) {
        for (sum, cur) in self.sums.iter_mut().zip(self.current.iter()) {
            for (s, &v) in sum.iter_mut().zip(cur.iter()) {
                *s += v as f64;
            }
        }
        self.count += 1;
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

/// One loaded adapter resident in an APPLY session. `directions` are per-block
/// unit vectors; `scale` is the live intensity (the steer `strength`, adjustable
/// via [`set_adapter_scale`]). A `SteerSpec` is the single-adapter case.
#[derive(Clone, Debug)]
struct ResidentAdapter {
    id: String,
    directions: Vec<Vec<f32>>,
    mode: SteerMode,
    scale: f32,
    layer_range: Range<usize>,
}

impl ResidentAdapter {
    /// Whether this adapter mutates block `layer_idx` (in range and not disabled).
    fn touches(&self, layer_idx: usize) -> bool {
        self.scale != 0.0 && self.layer_range.contains(&layer_idx)
    }
}

/// An ordered set of adapters applied at each in-range block boundary. Their
/// deltas SUM (each read from the same pre-apply residual), so the stack is
/// order-independent — see [`apply_stack_host`] / [`apply_on_gpu`].
struct LoraStack {
    adapters: Vec<ResidentAdapter>,
}

enum Session {
    Inactive,
    Capturing(CaptureAcc),
    Applying(LoraStack),
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

/// Serializes tests that mutate the process-global session (the apply control API
/// + the driver orchestration test), which would otherwise race under the default
/// parallel test runner.
#[cfg(test)]
pub(crate) static SESSION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn set_session(s: Session) {
    let active = !matches!(s, Session::Inactive);
    *session().write().unwrap() = s;
    mark_session_changed(active);
}

/// Bump the epoch (so the per-thread GPU apply cache refreshes) and update the
/// fast-path `ACTIVE` gate. For in-place mutations of the resident stack that
/// don't replace the whole `Session` (load/scale/unload).
fn mark_session_changed(active: bool) {
    EPOCH.fetch_add(1, Ordering::Release);
    ACTIVE.store(active, Ordering::Release);
}

// ── Control API ─────────────────────────────────────────────────────────────

/// Begin a CAPTURE session: subsequent forwards accumulate per-block residual
/// means. Run the +set, call [`finish_capture`], then the -set similarly.
pub fn begin_capture(num_layers: usize, hidden: usize) {
    set_session(Session::Capturing(CaptureAcc::new(num_layers, hidden)));
}

/// Fold the current prompt's last-token residuals into the capture means and
/// count it. Call once after each prompt's forward during a CAPTURE session.
pub fn commit_capture() {
    if let Session::Capturing(acc) = &mut *session().write().unwrap() {
        acc.commit();
    }
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

/// Begin an APPLY session from a single spec: replaces any current stack with one
/// adapter (`id = "default"`, `scale = spec.strength`). The back-compat entry the
/// search loop uses; [`load_adapter`] composes a multi-adapter stack instead.
pub fn begin_apply(spec: SteerSpec) {
    let adapter = ResidentAdapter {
        id: "default".to_string(),
        directions: spec.directions,
        mode: spec.mode,
        scale: spec.strength,
        layer_range: spec.layer_range,
    };
    set_session(Session::Applying(LoraStack {
        adapters: vec![adapter],
    }));
}

/// Push (or replace, by `id`) an adapter onto the APPLY stack, starting a session
/// if none is active. `directions` are per-block unit vectors; `scale` is the live
/// intensity. Stacked adapters' deltas sum at each block boundary.
pub fn load_adapter(
    id: impl Into<String>,
    directions: Vec<Vec<f32>>,
    mode: SteerMode,
    scale: f32,
    layer_range: Range<usize>,
) {
    let adapter = ResidentAdapter {
        id: id.into(),
        directions,
        mode,
        scale,
        layer_range,
    };
    {
        let mut guard = session().write().unwrap();
        match &mut *guard {
            Session::Applying(stack) => {
                stack.adapters.retain(|a| a.id != adapter.id);
                stack.adapters.push(adapter);
            }
            _ => {
                *guard = Session::Applying(LoraStack {
                    adapters: vec![adapter],
                });
            }
        }
    }
    mark_session_changed(true);
}

/// Materialize and load a rank-1 residual [`lora::LoraAdapter`] (ablate-only) onto
/// the APPLY stack — the bridge from the serialized artifact to the runtime.
pub fn load_lora_adapter(adapter: &lora::LoraAdapter) -> Result<(), String> {
    use lora::LoraTarget;
    if adapter.deltas.is_empty() {
        return Err("load_lora_adapter: adapter has no deltas".to_string());
    }
    let layer = |t: &LoraTarget| match t {
        LoraTarget::Residual { layer } => *layer,
    };
    let max_layer = adapter
        .deltas
        .iter()
        .map(|d| layer(&d.target))
        .max()
        .unwrap();
    let min_layer = adapter
        .deltas
        .iter()
        .map(|d| layer(&d.target))
        .min()
        .unwrap();
    let hidden = adapter.meta.hidden;
    let mut directions = vec![vec![0.0f32; hidden]; max_layer + 1];
    for d in &adapter.deltas {
        if d.rank() != 1 {
            return Err(format!(
                "load_lora_adapter: layer {} delta is rank {}, only rank-1 supported",
                layer(&d.target),
                d.rank()
            ));
        }
        // Residual form: A row = vᵀ = the unit direction.
        directions[layer(&d.target)] = d.a[0].clone();
    }
    load_adapter(
        adapter.id.clone(),
        directions,
        SteerMode::Ablate,
        adapter.scale,
        min_layer..max_layer + 1,
    );
    Ok(())
}

/// Set a loaded adapter's live `scale` (intensity). Returns `false` if no adapter
/// with `id` is loaded. Cheap — bumps the epoch so the GPU cache refreshes.
pub fn set_adapter_scale(id: &str, scale: f32) -> bool {
    let found = {
        let mut guard = session().write().unwrap();
        match &mut *guard {
            Session::Applying(stack) => stack
                .adapters
                .iter_mut()
                .find(|a| a.id == id)
                .map(|a| a.scale = scale)
                .is_some(),
            _ => false,
        }
    };
    if found {
        mark_session_changed(true);
    }
    found
}

/// Remove an adapter by `id`. Returns `false` if absent. The session goes
/// `Inactive` when the last adapter is unloaded.
pub fn unload_adapter(id: &str) -> bool {
    let (found, empty) = {
        let mut guard = session().write().unwrap();
        match &mut *guard {
            Session::Applying(stack) => {
                let before = stack.adapters.len();
                stack.adapters.retain(|a| a.id != id);
                let found = stack.adapters.len() < before;
                let empty = stack.adapters.is_empty();
                if empty {
                    *guard = Session::Inactive;
                }
                (found, empty)
            }
            _ => (false, false),
        }
    };
    if found {
        mark_session_changed(!empty);
    }
    found
}

/// `(id, scale)` for each loaded adapter (apply session only). Drives a
/// `lora_list`-style introspection.
pub fn loaded_adapters() -> Vec<(String, f32)> {
    match &*session().read().unwrap() {
        Session::Applying(stack) => stack
            .adapters
            .iter()
            .map(|a| (a.id.clone(), a.scale))
            .collect(),
        _ => Vec::new(),
    }
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
            acc.observe(layer_idx, &host);
        }
        Session::Applying(stack) => {
            if stack.adapters.iter().any(|a| a.touches(layer_idx)) {
                apply_on_gpu(gpu, x, layer_idx, stack, epoch)?;
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
            acc.observe(layer_idx, &host[last..last + hidden]);
        }
        Session::Applying(stack) => {
            if stack.adapters.iter().any(|a| a.touches(layer_idx)) {
                // Prefill is one-shot per request, and the search loop scores via
                // single-token decode forwards, so this host round-trip is amortized
                // — the per-token decode path is the one moved on-GPU. The whole
                // stack is summed per position (read from the pre-apply residual).
                let mut host = gpu.download_f32(x_batch)?;
                for p in 0..num_positions {
                    let off = p * hidden;
                    apply_stack_host(&stack.adapters, layer_idx, &mut host[off..off + hidden]);
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
    hidden: usize,
    /// GPU-resident mirror of the stack (per-adapter uploaded directions + scale).
    adapters: Vec<CachedAdapter>,
    /// `[1]` scratch for the ablate dot product `v·x`.
    proj_buf: GpuTensor,
    /// `[1]` scratch for the data-dependent per-adapter coefficient.
    coef_buf: GpuTensor,
}

struct CachedAdapter {
    mode: SteerMode,
    scale: f32,
    layer_range: Range<usize>,
    /// One `[1, hidden]` unit direction per block (2-D so `gemv_f32` reads m=1, k=hidden).
    dirs: Vec<GpuTensor>,
}

/// Apply the whole stack to one `[hidden]` residual fully on-GPU — no full-vector
/// host bounce. Reuses register-tiled `gemv_f32` (dot) +
/// `scaled_add_inplace_gpu_scalar_f32` (axpy). Stacking is the additive sum: every
/// adapter's coefficient is read from the PRE-apply residual (all `gemv` reads run
/// before any `scaled_add` write), so the result is order-independent. For ablate
/// only a 4-byte scalar round-trips per adapter.
fn apply_on_gpu(
    gpu: &mut Gpu,
    x: &GpuTensor,
    layer_idx: usize,
    stack: &LoraStack,
    epoch: u64,
) -> HipResult<()> {
    APPLY_CACHE.with(|cell| -> HipResult<()> {
        let mut slot = cell.borrow_mut();
        ensure_apply_cache(&mut slot, gpu, stack, epoch)?;
        let cache = slot.as_ref().unwrap();

        // Phase 1 (reads): per-adapter coefficient from the pre-apply residual.
        // `gemv_f32` reads `x` without writing, so collecting all coefficients
        // before any write makes the stack a sum from the original `x`.
        let mut writes: Vec<(usize, f32)> = Vec::new();
        for (ai, a) in cache.adapters.iter().enumerate() {
            if a.scale == 0.0 || !a.layer_range.contains(&layer_idx) {
                continue;
            }
            let coef = match a.mode {
                SteerMode::Steer => a.scale,
                SteerMode::Ablate => {
                    gpu.gemv_f32(&a.dirs[layer_idx], x, &cache.proj_buf)?;
                    let proj = gpu.download_f32(&cache.proj_buf)?[0];
                    -a.scale * proj
                }
            };
            writes.push((ai, coef));
        }

        // Phase 2 (writes): x += Σ coef · v.
        for (ai, coef) in writes {
            let dir = &cache.adapters[ai].dirs[layer_idx];
            gpu.memcpy_htod_auto(&cache.coef_buf.buf, &coef.to_le_bytes())?;
            gpu.scaled_add_inplace_gpu_scalar_f32(x, dir, &cache.coef_buf)?;
        }
        Ok(())
    })
}

/// Build (first use / shape change) or refresh (epoch change) the per-thread GPU
/// apply cache to mirror the resident stack. Direction buffers are reused across
/// epochs when the stack shape is unchanged, so the search loop neither
/// reallocates nor leaks; scales/modes/ranges are cheap host fields refreshed
/// every epoch (a `set_adapter_scale` re-uploads dirs too — fine, small, and
/// avoids tracking a separate scale epoch).
fn ensure_apply_cache(
    slot: &mut Option<ApplyCache>,
    gpu: &mut Gpu,
    stack: &LoraStack,
    epoch: u64,
) -> HipResult<()> {
    let hidden = stack
        .adapters
        .first()
        .and_then(|a| a.directions.first())
        .map_or(0, |d| d.len());

    let shape_matches = slot.as_ref().is_some_and(|c| {
        c.hidden == hidden
            && c.adapters.len() == stack.adapters.len()
            && c.adapters
                .iter()
                .zip(stack.adapters.iter())
                .all(|(ca, a)| ca.dirs.len() == a.directions.len())
    });

    if !shape_matches {
        let mut adapters = Vec::with_capacity(stack.adapters.len());
        for a in &stack.adapters {
            let mut dirs = Vec::with_capacity(a.directions.len());
            for d in &a.directions {
                dirs.push(gpu.upload_f32(d, &[1, hidden])?);
            }
            adapters.push(CachedAdapter {
                mode: a.mode,
                scale: a.scale,
                layer_range: a.layer_range.clone(),
                dirs,
            });
        }
        *slot = Some(ApplyCache {
            epoch,
            hidden,
            adapters,
            proj_buf: gpu.alloc_tensor(&[1], DType::F32)?,
            coef_buf: gpu.alloc_tensor(&[1], DType::F32)?,
        });
        return Ok(());
    }

    let cache = slot.as_mut().unwrap();
    if cache.epoch != epoch {
        for (ca, a) in cache.adapters.iter_mut().zip(stack.adapters.iter()) {
            for (buf, d) in ca.dirs.iter().zip(a.directions.iter()) {
                gpu.memcpy_htod_auto(&buf.buf, &f32_bytes(d))?;
            }
            ca.mode = a.mode;
            ca.scale = a.scale;
            ca.layer_range = a.layer_range.clone();
        }
        cache.epoch = epoch;
    }
    Ok(())
}

/// Host reference for [`apply_on_gpu`]: sum every in-range adapter's delta into
/// `x`, each read from the pre-apply residual (so the stack is order-independent).
/// Drives the batched prefill apply and is the unit-tested correctness anchor.
fn apply_stack_host(adapters: &[ResidentAdapter], layer_idx: usize, x: &mut [f32]) {
    let mut acc = vec![0.0f32; x.len()];
    for a in adapters {
        if !a.touches(layer_idx) {
            continue;
        }
        let v = &a.directions[layer_idx];
        match a.mode {
            SteerMode::Steer => {
                for (ai, &vi) in acc.iter_mut().zip(v.iter()) {
                    *ai += a.scale * vi;
                }
            }
            SteerMode::Ablate => {
                let proj = dot(x, v) * a.scale;
                for (ai, &vi) in acc.iter_mut().zip(v.iter()) {
                    *ai -= proj * vi;
                }
            }
        }
    }
    for (xi, &ai) in x.iter_mut().zip(acc.iter()) {
        *xi += ai;
    }
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
mod stack_tests {
    use super::*;
    use crate::lora;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn adapter(id: &str, dir: Vec<f32>, mode: SteerMode, scale: f32) -> ResidentAdapter {
        ResidentAdapter {
            id: id.to_string(),
            directions: vec![dir],
            mode,
            scale,
            layer_range: 0..1,
        }
    }

    #[test]
    fn stack_host_single_ablate_equals_apply_direction() {
        let v = unit(&[0.3, -0.4, 0.5, 0.2]);
        let stack = vec![adapter("a", v.clone(), SteerMode::Ablate, 0.8)];
        let x0 = vec![1.0, 2.0, -1.5, 0.7];
        let mut xs = x0.clone();
        apply_stack_host(&stack, 0, &mut xs);
        let mut xr = x0.clone();
        apply_direction(&mut xr, &v, SteerMode::Ablate, 0.8);
        for (a, b) in xs.iter().zip(&xr) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn stack_host_orthogonal_sum_equals_sequential() {
        let v1 = unit(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = unit(&[0.0, 1.0, 0.0, 0.0]);
        let stack = vec![
            adapter("a", v1.clone(), SteerMode::Ablate, 0.6),
            adapter("b", v2.clone(), SteerMode::Ablate, 0.9),
        ];
        let x0 = vec![2.0, -3.0, 1.0, 0.5];
        let mut xs = x0.clone();
        apply_stack_host(&stack, 0, &mut xs);
        let mut xseq = x0.clone();
        apply_direction(&mut xseq, &v1, SteerMode::Ablate, 0.6);
        apply_direction(&mut xseq, &v2, SteerMode::Ablate, 0.9);
        for (a, b) in xs.iter().zip(&xseq) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn stack_host_skips_zero_scale_and_out_of_range() {
        let v = unit(&[1.0, 1.0, 0.0, 0.0]);
        let mut a = adapter("z", v, SteerMode::Ablate, 0.0);
        let x0 = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = x0.clone();
        apply_stack_host(std::slice::from_ref(&a), 0, &mut x);
        assert_eq!(x, x0); // scale 0 → no-op
        a.scale = 1.0;
        a.layer_range = 1..2;
        apply_stack_host(std::slice::from_ref(&a), 0, &mut x);
        assert_eq!(x, x0); // out of range at layer 0 → no-op
    }

    #[test]
    fn stack_host_matches_lora_module_reference() {
        // The resident-stack apply == the increment-1 LoraAdapter host apply.
        let v = unit(&[0.2, 0.9, -0.1, 0.3]);
        let lad =
            lora::abliteration_adapter("x", &[v.clone()], SteerMode::Ablate, 0.7, 0..1).unwrap();
        let resident = vec![adapter("x", v, SteerMode::Ablate, 0.7)];
        let x0 = vec![1.0, -1.0, 2.0, 0.5];
        let mut xa = x0.clone();
        apply_stack_host(&resident, 0, &mut xa);
        let mut xb = x0.clone();
        lora::apply_residual_stack(std::slice::from_ref(&lad), 0, &mut xb);
        for (a, b) in xa.iter().zip(&xb) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn control_api_load_scale_unload() {
        let _g = crate::SESSION_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        clear();
        assert!(!is_active());
        let v = unit(&[1.0, 0.0, 0.0, 0.0]);
        // begin_apply seeds a single "default" adapter.
        begin_apply(SteerSpec {
            directions: vec![v.clone()],
            mode: SteerMode::Ablate,
            strength: 0.5,
            layer_range: 0..1,
        });
        assert!(is_active());
        assert_eq!(loaded_adapters(), vec![("default".to_string(), 0.5)]);
        // load_adapter stacks a second.
        load_adapter("ablit", vec![v], SteerMode::Ablate, 1.0, 0..1);
        assert_eq!(loaded_adapters().len(), 2);
        // set_adapter_scale dials intensity; unknown id is a no-op.
        assert!(set_adapter_scale("ablit", 0.25));
        assert!(!set_adapter_scale("missing", 1.0));
        let ablit_scale = loaded_adapters()
            .into_iter()
            .find(|(id, _)| id == "ablit")
            .unwrap()
            .1;
        assert_eq!(ablit_scale, 0.25);
        // Unloading one keeps the session active; the last unload clears it.
        assert!(unload_adapter("default"));
        assert!(is_active());
        assert!(unload_adapter("ablit"));
        assert!(!is_active());
        assert!(loaded_adapters().is_empty());
        clear();
    }

    #[test]
    fn load_lora_adapter_populates_stack() {
        let _g = crate::SESSION_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        clear();
        let v = unit(&[0.1, 0.2, 0.3, 0.9]);
        let lad = lora::abliteration_adapter("med", &[v], SteerMode::Ablate, 0.6, 0..1).unwrap();
        load_lora_adapter(&lad).unwrap();
        assert_eq!(loaded_adapters(), vec![("med".to_string(), 0.6)]);
        clear();
    }
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
        acc.observe(0, &[2.0, 4.0]);
        acc.commit();
        acc.observe(0, &[4.0, 8.0]);
        acc.commit();
        let m = acc.means();
        assert_eq!(m.0[0], vec![3.0, 6.0]);
    }
}
