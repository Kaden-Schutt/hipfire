//! TriAttention: KV-cache compression via trigonometric series scoring.
//!
//! Reference: Mao et al. 2026 "TriAttention: Efficient Long Reasoning with
//! Trigonometric KV Compression" (arXiv:2604.04921).
//!
//! ## Core idea
//!
//! Pre-RoPE Q and K vectors are highly concentrated around non-zero centers
//! (Mean Resultant Length R ≈ 1 on most heads — a model-intrinsic property).
//! When Q/K are approximately constant across tokens, the RoPE attention
//! logit at Q-K distance Δ becomes a trigonometric series in Δ alone:
//!
//!   logit(Δ) ≈ Σ_f ||E[q_f]|| · ||k_f|| · cos(ω_f · Δ + φ_f)
//!
//! where `ω_f = θ^{-2f/d}` is the RoPE frequency for band f, and
//! `φ_f = arg(E[q_f]) - arg(k_f)`.
//!
//! Scoring a cached key at position `p_k` for a query at position `p_q` lets
//! us predict attention WITHOUT recomputing it — useful for eviction.
//!
//! ## Hipfire mapping
//!
//! Our K cache stores POST-RoPE K. Paper assumes pre-RoPE K in the formula.
//! Using `k_pre_f = k_post_f · e^{-iω_f·p_k}` (RoPE is unitary):
//!   - `||k_pre_f|| = ||k_post_f||` (norm invariant)
//!   - `arg(k_pre_f) = arg(k_post_f) - ω_f · p_k`
//!
//! Substituting into S_trig collapses the position-dependent phase:
//!
//!   cos(ω_f·Δ + arg(E[q_f]) - arg(k_pre_f))
//!     = cos(ω_f·(p_q - p_k) + arg(E[q_f]) - arg(k_post_f) + ω_f·p_k)
//!     = cos(ω_f·p_q + arg(E[q_f]) - arg(k_post_f))
//!
//! So we score directly on the cached post-RoPE K with a query-position-
//! dependent phase, no de-rotation kernel needed. (asym3's Givens stage
//! does require de-Givens at scoring time — not implemented yet.)

use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Q-side centers for one (layer, head, band) triple.
///
/// Stored fields use pre-RoPE semantics: `E[q_f]` is the complex mean of
/// pre-RoPE queries in band f, `E[abs_q_f]` is the mean of their scalar
/// magnitude. The ratio `||E[q_f]|| / E[abs_q_f]` recovers the Mean
/// Resultant Length R_f used in the concentration-based weighting.
#[derive(Debug, Clone, Copy, Default)]
pub struct BandCenter {
    /// Re(E[q_f])
    pub eq_re: f32,
    /// Im(E[q_f])
    pub eq_im: f32,
    /// E[||q_f||]  (scalar, ≥ 0)
    pub e_abs_q: f32,
}

impl BandCenter {
    pub fn magnitude(&self) -> f32 {
        (self.eq_re * self.eq_re + self.eq_im * self.eq_im).sqrt()
    }
    pub fn phase(&self) -> f32 {
        self.eq_im.atan2(self.eq_re)
    }
    /// Mean Resultant Length R_f ∈ [0, 1] — how concentrated band f is.
    pub fn mrl(&self) -> f32 {
        if self.e_abs_q > 1e-20 {
            (self.magnitude() / self.e_abs_q).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Full Q-side calibration: per (layer, head, band) centers.
///
/// Layout: `centers[layer * n_heads * n_bands + head * n_bands + band]`.
/// `n_bands = head_dim / 2`.
#[derive(Debug, Clone)]
pub struct TriAttnCenters {
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub centers: Vec<BandCenter>,
}

impl TriAttnCenters {
    pub fn new(n_layers: usize, n_heads: usize, head_dim: usize, rope_theta: f32, partial_rotary_factor: f32) -> Self {
        let n_bands = head_dim / 2;
        Self {
            n_layers, n_heads, head_dim, rope_theta, partial_rotary_factor,
            centers: vec![BandCenter::default(); n_layers * n_heads * n_bands],
        }
    }

    #[inline]
    pub fn n_bands(&self) -> usize { self.head_dim / 2 }

    #[inline]
    pub fn get(&self, layer: usize, head: usize, band: usize) -> BandCenter {
        let n_bands = self.n_bands();
        self.centers[layer * self.n_heads * n_bands + head * n_bands + band]
    }

    #[inline]
    pub fn set(&mut self, layer: usize, head: usize, band: usize, c: BandCenter) {
        let n_bands = self.n_bands();
        self.centers[layer * self.n_heads * n_bands + head * n_bands + band] = c;
    }

    /// RoPE frequency ω_f = θ^{-2f/d_rot}  where d_rot = partial_rotary_factor × head_dim.
    /// Bands beyond d_rot/2 are unrotated (ω=0 → phase contribution 0).
    pub fn omega(&self, band: usize) -> f32 {
        let d_rot = (self.head_dim as f32 * self.partial_rotary_factor) as usize;
        if band * 2 >= d_rot { return 0.0; }
        let exponent = -2.0f32 * band as f32 / d_rot as f32;
        self.rope_theta.powf(exponent)
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        // Header: magic=TRIA, version=1, then geometry.
        f.write_all(b"TRIA")?;
        f.write_all(&1u32.to_le_bytes())?;
        f.write_all(&(self.n_layers as u32).to_le_bytes())?;
        f.write_all(&(self.n_heads as u32).to_le_bytes())?;
        f.write_all(&(self.head_dim as u32).to_le_bytes())?;
        f.write_all(&self.rope_theta.to_le_bytes())?;
        f.write_all(&self.partial_rotary_factor.to_le_bytes())?;
        for c in &self.centers {
            f.write_all(&c.eq_re.to_le_bytes())?;
            f.write_all(&c.eq_im.to_le_bytes())?;
            f.write_all(&c.e_abs_q.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"TRIA" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a TRIA sidecar"));
        }
        let mut u32buf = [0u8; 4];
        f.read_exact(&mut u32buf)?; let _ver = u32::from_le_bytes(u32buf);
        f.read_exact(&mut u32buf)?; let n_layers = u32::from_le_bytes(u32buf) as usize;
        f.read_exact(&mut u32buf)?; let n_heads = u32::from_le_bytes(u32buf) as usize;
        f.read_exact(&mut u32buf)?; let head_dim = u32::from_le_bytes(u32buf) as usize;
        f.read_exact(&mut u32buf)?; let rope_theta = f32::from_le_bytes(u32buf);
        f.read_exact(&mut u32buf)?; let partial_rotary_factor = f32::from_le_bytes(u32buf);
        let n_bands = head_dim / 2;
        let n = n_layers * n_heads * n_bands;
        let mut centers = Vec::with_capacity(n);
        for _ in 0..n {
            f.read_exact(&mut u32buf)?; let eq_re = f32::from_le_bytes(u32buf);
            f.read_exact(&mut u32buf)?; let eq_im = f32::from_le_bytes(u32buf);
            f.read_exact(&mut u32buf)?; let e_abs_q = f32::from_le_bytes(u32buf);
            centers.push(BandCenter { eq_re, eq_im, e_abs_q });
        }
        Ok(Self { n_layers, n_heads, head_dim, rope_theta, partial_rotary_factor, centers })
    }
}

/// CPU-side accumulator for a single (layer, head, band) triple.
///
/// Sums complex q_f and scalar |q_f| across calibration samples. `finalize`
/// turns sums into means. Single-pass online statistics; no second pass
/// needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct BandAccumulator {
    pub sum_re: f64,
    pub sum_im: f64,
    pub sum_abs: f64,
    pub count: u64,
}

impl BandAccumulator {
    pub fn add(&mut self, re: f32, im: f32) {
        let re64 = re as f64;
        let im64 = im as f64;
        self.sum_re += re64;
        self.sum_im += im64;
        self.sum_abs += (re64 * re64 + im64 * im64).sqrt();
        self.count += 1;
    }

    pub fn finalize(&self) -> BandCenter {
        if self.count == 0 {
            return BandCenter::default();
        }
        let n = self.count as f64;
        BandCenter {
            eq_re: (self.sum_re / n) as f32,
            eq_im: (self.sum_im / n) as f32,
            e_abs_q: (self.sum_abs / n) as f32,
        }
    }
}

/// Full-model calibration accumulator. One bank of BandAccumulators per
/// (layer, head, band).
pub struct TriAttnCalibState {
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub accs: Vec<BandAccumulator>,
}

impl TriAttnCalibState {
    pub fn new(n_layers: usize, n_heads: usize, head_dim: usize, rope_theta: f32, partial_rotary_factor: f32) -> Self {
        let n_bands = head_dim / 2;
        Self {
            n_layers, n_heads, head_dim, rope_theta, partial_rotary_factor,
            accs: vec![BandAccumulator::default(); n_layers * n_heads * n_bands],
        }
    }

    /// Feed one pre-RoPE Q sample: `q` is [n_heads × head_dim] interleaved
    /// (band f = complex pair (q[2f], q[2f+1]) per head).
    ///
    /// Heads are independent — the accumulator slice for (layer, head) is
    /// written by exactly one thread at a time. We parallelize across heads
    /// via rayon::par_chunks_mut so the inner band loop runs in parallel.
    /// Measured on MI300X EPYC host: 99%+ of sidecar cal wall time was CPU
    /// accumulation in the serial version. Per-head parallelism scales with
    /// core count up to n_heads (typically 16 for Qwen3.5).
    pub fn add_sample(&mut self, layer: usize, q: &[f32]) {
        use rayon::prelude::*;
        let n_bands = self.head_dim / 2;
        let head_dim = self.head_dim;
        let n_heads = self.n_heads;
        assert_eq!(q.len(), n_heads * head_dim,
            "sample length {} != n_heads * head_dim = {}", q.len(), n_heads * head_dim);
        let base_idx = layer * n_heads * n_bands;
        self.accs[base_idx..base_idx + n_heads * n_bands]
            .par_chunks_mut(n_bands)
            .enumerate()
            .for_each(|(h, head_accs)| {
                let q_base = h * head_dim;
                for f in 0..n_bands {
                    let re = q[q_base + 2 * f];
                    let im = q[q_base + 2 * f + 1];
                    head_accs[f].add(re, im);
                }
            });
    }

    /// Feed a batch of samples at once. `q_batch` is [batch × n_heads × head_dim].
    pub fn add_batch(&mut self, layer: usize, q_batch: &[f32]) {
        let stride = self.n_heads * self.head_dim;
        assert!(q_batch.len() % stride == 0, "batch stride mismatch");
        let batch = q_batch.len() / stride;
        for b in 0..batch {
            self.add_sample(layer, &q_batch[b * stride..(b + 1) * stride]);
        }
    }

    pub fn finalize(self) -> TriAttnCenters {
        let n_bands = self.head_dim / 2;
        let centers: Vec<BandCenter> = self.accs.iter().map(|a| a.finalize()).collect();
        assert_eq!(centers.len(), self.n_layers * self.n_heads * n_bands);
        TriAttnCenters {
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            rope_theta: self.rope_theta,
            partial_rotary_factor: self.partial_rotary_factor,
            centers,
        }
    }
}

// ─── Global calibration tap ───────────────────────────────────────────────
//
// Hot-path design: forward_scratch_layers calls `record_prerope_q` at each
// FA layer. A single relaxed atomic read decides whether to take the slow
// path (download Q + accumulate). Non-calibration inference pays only that
// atomic check.

/// Per-token raw capture of pre-RoPE Q and K across all FA layers.
/// Populated by the forward-pass tap when `install_capture` is active.
/// Used by the reconstruction-correlation harness (§3.3 of the paper).
///
/// Layout: `q_samples[token_idx][fa_layer_idx]` is `[n_heads × head_dim]`
/// pre-RoPE Q for that token at that FA layer. Tokens are appended in
/// call order; FA layers appear in model-definition order.
#[derive(Debug, Default)]
pub struct TriAttnCapture {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Entries in layer order for the current in-flight token. Flushed
    /// into `q_samples` / `k_samples` by `finish_token`.
    pub pending_q: Vec<Vec<f32>>,
    pub pending_k: Vec<Vec<f32>>,
    pub pending_layer_ids: Vec<usize>,
    pub q_samples: Vec<Vec<Vec<f32>>>,
    pub k_samples: Vec<Vec<Vec<f32>>>,
    pub layer_ids_per_token: Vec<Vec<usize>>,
}

impl TriAttnCapture {
    pub fn new(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self { n_heads, n_kv_heads, head_dim, ..Default::default() }
    }

    /// Call after each full forward pass (one token) to commit the
    /// captured rows to the per-token history.
    pub fn finish_token(&mut self) {
        let q = std::mem::take(&mut self.pending_q);
        let k = std::mem::take(&mut self.pending_k);
        let ids = std::mem::take(&mut self.pending_layer_ids);
        self.q_samples.push(q);
        self.k_samples.push(k);
        self.layer_ids_per_token.push(ids);
    }
}

enum TapState {
    Calibrate(TriAttnCalibState),
    CalibrateGpu(TriAttnCalibStateGpu),
    Capture(TriAttnCapture),
}

/// GPU-side calibration accumulator. Holds device-resident f64/u64 buffers
/// that persist across calibration chunks. The HIP kernel
/// (`triattn_accumulate_f32` in `kernels/src/triattn_accumulate.hip`)
/// ADDS into these buffers, eliminating the Q-tensor PCIe transfer + CPU
/// sqrt loop that dominated the CPU calibration path (99% of wall time on
/// MI300X). Finalized to `TriAttnCenters` via `take_tap_gpu`.
pub struct TriAttnCalibStateGpu {
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub accs_sum_re: hip_bridge::DeviceBuffer,   // n_layers*n_heads*n_bands × f64
    pub accs_sum_im: hip_bridge::DeviceBuffer,
    pub accs_sum_abs: hip_bridge::DeviceBuffer,
    pub accs_count: hip_bridge::DeviceBuffer,    // u64
}

impl TriAttnCalibStateGpu {
    pub fn new(
        gpu: &mut rdna_compute::Gpu,
        n_layers: usize, n_heads: usize, head_dim: usize,
        rope_theta: f32, partial_rotary_factor: f32,
    ) -> hip_bridge::HipResult<Self> {
        let n_bands = head_dim / 2;
        let n_accs = n_layers * n_heads * n_bands;
        let bytes = n_accs * 8; // f64 / u64 are both 8 bytes
        let accs_sum_re = gpu.hip.malloc(bytes)?;
        let accs_sum_im = gpu.hip.malloc(bytes)?;
        let accs_sum_abs = gpu.hip.malloc(bytes)?;
        let accs_count = gpu.hip.malloc(bytes)?;
        // Zero the buffers so the kernel's ADDs start from a clean slate.
        gpu.hip.memset(&accs_sum_re, 0, bytes)?;
        gpu.hip.memset(&accs_sum_im, 0, bytes)?;
        gpu.hip.memset(&accs_sum_abs, 0, bytes)?;
        gpu.hip.memset(&accs_count, 0, bytes)?;
        Ok(Self {
            n_layers, n_heads, head_dim, rope_theta, partial_rotary_factor,
            accs_sum_re, accs_sum_im, accs_sum_abs, accs_count,
        })
    }

    /// Download + convert to the same TriAttnCenters format the CPU path
    /// produces. Uses the exact same finalize math as `BandAccumulator::finalize`
    /// to ensure identical output.
    pub fn finalize(self, gpu: &mut rdna_compute::Gpu) -> hip_bridge::HipResult<TriAttnCenters> {
        let n_bands = self.head_dim / 2;
        let n_accs = self.n_layers * self.n_heads * n_bands;

        // Download the 4 accumulator arrays to host.
        let mut sum_re = vec![0.0f64; n_accs];
        let mut sum_im = vec![0.0f64; n_accs];
        let mut sum_abs = vec![0.0f64; n_accs];
        let mut count = vec![0u64; n_accs];
        gpu.hip.memcpy_dtoh(
            unsafe { std::slice::from_raw_parts_mut(
                sum_re.as_mut_ptr() as *mut u8, n_accs * 8) },
            &self.accs_sum_re,
        )?;
        gpu.hip.memcpy_dtoh(
            unsafe { std::slice::from_raw_parts_mut(
                sum_im.as_mut_ptr() as *mut u8, n_accs * 8) },
            &self.accs_sum_im,
        )?;
        gpu.hip.memcpy_dtoh(
            unsafe { std::slice::from_raw_parts_mut(
                sum_abs.as_mut_ptr() as *mut u8, n_accs * 8) },
            &self.accs_sum_abs,
        )?;
        gpu.hip.memcpy_dtoh(
            unsafe { std::slice::from_raw_parts_mut(
                count.as_mut_ptr() as *mut u8, n_accs * 8) },
            &self.accs_count,
        )?;

        // Same math as BandAccumulator::finalize: mean(re), mean(im), mean(|q|).
        let centers: Vec<BandCenter> = (0..n_accs).map(|i| {
            let c = count[i];
            if c == 0 {
                BandCenter::default()
            } else {
                let n = c as f64;
                BandCenter {
                    eq_re: (sum_re[i] / n) as f32,
                    eq_im: (sum_im[i] / n) as f32,
                    e_abs_q: (sum_abs[i] / n) as f32,
                }
            }
        }).collect();

        Ok(TriAttnCenters {
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            rope_theta: self.rope_theta,
            partial_rotary_factor: self.partial_rotary_factor,
            centers,
        })
    }
}

static TAP_ENABLED: AtomicBool = AtomicBool::new(false);
static TAP_STATE: Mutex<Option<TapState>> = Mutex::new(None);

/// Install a fresh calibration state (online accumulation, no raw retention).
pub fn install_tap(state: TriAttnCalibState) {
    *TAP_STATE.lock().unwrap() = Some(TapState::Calibrate(state));
    TAP_ENABLED.store(true, Ordering::SeqCst);
}

/// Install a GPU-side calibration state. Q tensors stay resident in GPU
/// memory; the kernel writes partial sums directly into device buffers.
/// ~5-8× faster than the CPU tap on MI300X (measured).
pub fn install_tap_gpu(state: TriAttnCalibStateGpu) {
    *TAP_STATE.lock().unwrap() = Some(TapState::CalibrateGpu(state));
    TAP_ENABLED.store(true, Ordering::SeqCst);
}

/// Remove and return the GPU calibration tap so the caller can finalize.
pub fn take_tap_gpu() -> Option<TriAttnCalibStateGpu> {
    TAP_ENABLED.store(false, Ordering::SeqCst);
    match TAP_STATE.lock().unwrap().take() {
        Some(TapState::CalibrateGpu(s)) => Some(s),
        other => {
            *TAP_STATE.lock().unwrap() = other;
            None
        }
    }
}

/// Dispatch the GPU accumulate kernel for one chunk's worth of Q. Called
/// from the forward-pass tap point in qwen35.rs BEFORE it downloads Q to
/// host. Returns `Ok(true)` if the GPU tap handled the chunk (caller can
/// skip the Q/K downloads), `Ok(false)` if no GPU tap is installed (caller
/// falls back to CPU path). Never errors the whole run; GPU dispatch errors
/// propagate up as HipResult.
pub fn record_prerope_q_batch_gpu_if_applicable(
    gpu: &mut rdna_compute::Gpu,
    layer_idx: usize,
    q_batch: &hip_bridge::DeviceBuffer,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) -> hip_bridge::HipResult<bool> {
    if !TAP_ENABLED.load(Ordering::Relaxed) {
        return Ok(false);
    }
    // Hold the mutex across the kernel-launch call. The kernel launch is
    // async-enqueue (sub-ms), so the lock hold is short and contention-free
    // (qwen35 forward is single-threaded per Gpu). gpu.triattn_accumulate
    // does not touch TAP_STATE itself.
    let guard = TAP_STATE.lock().unwrap();
    let s = match guard.as_ref() {
        Some(TapState::CalibrateGpu(s)) => s,
        _ => return Ok(false),
    };
    gpu.triattn_accumulate(
        q_batch,
        &s.accs_sum_re, &s.accs_sum_im, &s.accs_sum_abs, &s.accs_count,
        n_tokens, n_heads, head_dim, layer_idx,
    )?;
    Ok(true)
}

/// Install a full-capture buffer (per-token raw Q/K retention). Costlier
/// but lets the reconstruction harness compute ground-truth logits on the
/// host.
pub fn install_capture(cap: TriAttnCapture) {
    *TAP_STATE.lock().unwrap() = Some(TapState::Capture(cap));
    TAP_ENABLED.store(true, Ordering::SeqCst);
}

/// Remove and return the calibration tap, disabling the global hook.
pub fn take_tap() -> Option<TriAttnCalibState> {
    TAP_ENABLED.store(false, Ordering::SeqCst);
    match TAP_STATE.lock().unwrap().take() {
        Some(TapState::Calibrate(s)) => Some(s),
        other => {
            // Restore non-calibrate state so a mis-matched take doesn't lose data.
            *TAP_STATE.lock().unwrap() = other;
            None
        }
    }
}

/// Remove and return the full-capture buffer.
pub fn take_capture() -> Option<TriAttnCapture> {
    TAP_ENABLED.store(false, Ordering::SeqCst);
    match TAP_STATE.lock().unwrap().take() {
        Some(TapState::Capture(c)) => Some(c),
        other => {
            *TAP_STATE.lock().unwrap() = other;
            None
        }
    }
}

/// Signal the end of one token's forward pass. Used by the capture tap to
/// flush pending per-layer rows into the per-token history. No-op for
/// calibration or when the tap is disabled.
pub fn capture_finish_token() {
    if !TAP_ENABLED.load(Ordering::Relaxed) { return; }
    if let Some(TapState::Capture(c)) = TAP_STATE.lock().unwrap().as_mut() {
        c.finish_token();
    }
}

/// True if any tap is active.
pub fn tap_enabled() -> bool {
    TAP_ENABLED.load(Ordering::Relaxed)
}

/// True only when the active tap actually consumes K (`Capture`). The
/// `Calibrate` and `CalibrateGpu` taps record only Q, so callers can skip
/// the K download in the calibrate path. ~33 GB PCIe/1M-token 27B run on
/// the CPU fallback path; small but real win on the GPU path too if the
/// caller can avoid building K at all.
pub fn tap_needs_k() -> bool {
    if !TAP_ENABLED.load(Ordering::Relaxed) { return false; }
    matches!(
        TAP_STATE.lock().unwrap().as_ref(),
        Some(TapState::Capture(_)),
    )
}

/// Called from the FA layer pre-RoPE point. `q` is `[n_heads × head_dim]`
/// pre-RoPE Q; `k_opt` is `[n_kv_heads × head_dim]` pre-RoPE K (only used
/// by the capture tap; calibrate ignores it).
pub fn record_prerope_q(layer_idx: usize, q: &[f32]) {
    record_prerope_qk(layer_idx, q, None);
}

pub fn record_prerope_qk(layer_idx: usize, q: &[f32], k_opt: Option<&[f32]>) {
    if !TAP_ENABLED.load(Ordering::Relaxed) { return; }
    let mut guard = TAP_STATE.lock().unwrap();
    match guard.as_mut() {
        Some(TapState::Calibrate(state)) => {
            state.add_sample(layer_idx, q);
        }
        Some(TapState::CalibrateGpu(_)) => {
            // Reached only when a forward path took the host-side download
            // fallback while a GPU-resident calibration tap was installed.
            // Pre-Phase-2 this was a silent no-op, which would have
            // produced an empty sidecar without warning. With gpu_calib
            // = true now the default, fail loudly so any new code path
            // that records Q through host (e.g. single-token decode,
            // unbatched validation) is caught immediately rather than
            // silently dropping samples.
            panic!(
                "triattn: record_prerope_qk hit while CalibrateGpu tap is installed. \
                The forward path should call record_prerope_q_batch_gpu_if_applicable \
                first; a return of Ok(false) means either the GPU tap was lost or a \
                non-batch path is feeding this hook (would produce empty sidecar). \
                layer_idx={layer_idx} q.len={}",
                q.len(),
            );
        }
        Some(TapState::Capture(cap)) => {
            cap.pending_layer_ids.push(layer_idx);
            cap.pending_q.push(q.to_vec());
            cap.pending_k.push(k_opt.map(|k| k.to_vec()).unwrap_or_default());
        }
        None => {}
    }
}

// ─── Scoring (CPU, validation harness) ────────────────────────────────────

/// Compute per-band norms and phases of a post-RoPE K vector.
///
/// `k_post` is `[head_dim]` (one head, one position). Returns `(||k_f||,
/// arg(k_f))` per band f=0..head_dim/2.
pub fn kpost_per_band(k_post: &[f32]) -> Vec<(f32, f32)> {
    let n_bands = k_post.len() / 2;
    let mut out = Vec::with_capacity(n_bands);
    for f in 0..n_bands {
        let re = k_post[2 * f];
        let im = k_post[2 * f + 1];
        let mag = (re * re + im * im).sqrt();
        let ph = im.atan2(re);
        out.push((mag, ph));
    }
    out
}

/// Trigonometric series score for one key at position `p_k` vs query at
/// position `p_q`. Uses post-RoPE K directly (see module docstring).
///
/// `k_post_bands` — precomputed `(||k_post_f||, arg(k_post_f))` per band.
/// The caller passes the layer+head's center slice.
pub fn s_trig(
    centers: &[BandCenter],
    k_post_bands: &[(f32, f32)],
    p_q: f32,
    omega: impl Fn(usize) -> f32,
) -> f32 {
    assert_eq!(centers.len(), k_post_bands.len());
    let mut sum = 0.0f32;
    for (f, (c, (k_mag, k_phase))) in centers.iter().zip(k_post_bands.iter()).enumerate() {
        let c_mag = c.magnitude();
        let c_phase = c.phase();
        let w = omega(f);
        // Identity: replace arg(k_pre) = arg(k_post) - ω·p_k, the p_k term
        // cancels with ω·Δ; what remains is ω·p_q + arg(E[q_f]) - arg(k_post_f).
        let angle = w * p_q + c_phase - k_phase;
        sum += c_mag * k_mag * angle.cos();
    }
    sum
}

/// Norm-based score S_norm (eq. 9 in the paper), weighted by (1 - R_f) so
/// low-concentration bands contribute more.
pub fn s_norm(centers: &[BandCenter], k_post_bands: &[(f32, f32)]) -> f32 {
    assert_eq!(centers.len(), k_post_bands.len());
    let mut sum = 0.0f32;
    for (c, (k_mag, _)) in centers.iter().zip(k_post_bands.iter()) {
        let r = c.mrl();
        sum += (1.0 - r) * c.e_abs_q * k_mag;
    }
    sum
}

/// Combined per-key score: S_trig + S_norm.
pub fn s_total(
    centers: &[BandCenter],
    k_post_bands: &[(f32, f32)],
    p_q: f32,
    omega: impl Fn(usize) -> f32,
) -> f32 {
    s_trig(centers, k_post_bands, p_q, &omega) + s_norm(centers, k_post_bands)
}

// ─── Top-B selection (paper §4.3, eq. 12-13) ─────────────────────────────

/// Given a score matrix `[n_heads × seq_len]`, pick the top `budget`
/// positions using the paper's GQA aggregation:
///
///   1. Per query head: z-score normalize scores across positions.
///   2. Per position: take the max over heads (ensemble "any head
///      thinks this position matters").
///   3. Pick the top `budget` positions by aggregated score.
///
/// Returns the retained positions in **ascending source order** so a
/// subsequent compaction preserves causal ordering — the bake-in of
/// RoPE phases into each cached K makes physical order irrelevant for
/// attention math, but monotone ordering keeps later incremental
/// reasoning about positions straightforward.
pub fn compute_retain_indices(
    scores: &[f32],
    n_heads: usize,
    seq_len: usize,
    budget: usize,
) -> Vec<u32> {
    assert_eq!(scores.len(), n_heads * seq_len, "scores shape mismatch");
    if seq_len == 0 { return Vec::new(); }
    let b = budget.min(seq_len);

    // 1. Per-head z-score
    let mut agg = vec![f32::NEG_INFINITY; seq_len];
    for h in 0..n_heads {
        let row = &scores[h * seq_len..(h + 1) * seq_len];
        let mean: f32 = row.iter().sum::<f32>() / seq_len as f32;
        let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>()
            / seq_len as f32;
        let std = var.sqrt().max(1e-6);
        // 2. Max across heads
        for p in 0..seq_len {
            let z = (row[p] - mean) / std;
            if z > agg[p] { agg[p] = z; }
        }
    }

    // 3. Top-b by score, then sort ascending by position
    let mut indexed: Vec<(f32, usize)> = agg.into_iter().enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(b);
    indexed.sort_by_key(|&(_, i)| i);
    indexed.into_iter().map(|(_, i)| i as u32).collect()
}

// ─── Forward-loop eviction trigger ────────────────────────────────────────

/// Outcome of a successful eviction pass. `retain_mask` is the source-position
/// retain selection from the **last** FA layer processed — callers that need
/// to mirror the eviction into a non-KV auxiliary buffer (DFlash's
/// `draft_scratch.target_hidden`) use it as a single representative mask, since
/// retain decisions across FA layers are strongly correlated in practice.
pub struct EvictionResult {
    pub new_physical: usize,
    pub retain_mask: Vec<u32>,
}

/// Pre-allocated scratch + policy for periodic TriAttention eviction during
/// autoregressive decode. Instantiate once per inference session; call
/// `maybe_evict` after every new-token write. When the physical cache has
/// grown to `budget + beta`, the method scores + compacts every FA layer
/// and returns the new physical count (== `budget`). The caller is
/// responsible for tracking the next physical write slot.
///
/// The paper's trigger cadence is "every β=128 decoded tokens"; our
/// variant is "whenever cache growth since last compaction ≥ β". These
/// two match when β divides the cache size at first trigger, and the
/// physical-threshold variant is robust to variable-length prefills.
pub struct EvictionCtx {
    /// Flattened centers on device: `[fa_count × n_heads × n_bands × 3]`.
    pub centers_dev: GpuTensor,
    pub fa_layer_ids: Vec<usize>,
    pub centers_per_layer: usize,
    pub budget: usize,
    pub beta: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_theta: f32,
    pub max_seq: usize,
    /// Reusable scratch: sized to handle any state up to `max_seq`.
    pub scores_buf: GpuTensor,
    pub k_compact: GpuTensor,
    pub v_compact: GpuTensor,
    pub retain_dev: GpuTensor,
    /// Running count of evictions fired (useful for bench harnesses).
    pub eviction_count: std::cell::Cell<usize>,
}

impl EvictionCtx {
    /// Upload centers and pre-size scratch. `fa_layer_ids` must match the
    /// layer indices the caller will pass through the forward pass (i.e.
    /// those where `layer_types[i] == FullAttention`). `max_seq` must be
    /// at least `budget + beta` for any eviction to fire.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &mut Gpu,
        centers: &TriAttnCenters,
        fa_layer_ids: Vec<usize>,
        budget: usize,
        beta: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        rope_theta: f32,
        max_seq: usize,
    ) -> HipResult<Self> {
        // The eviction trigger is `current_physical >= budget + beta`, where
        // `current_physical` is bounded by `KvCache::physical_cap`. Historically
        // physical_cap == max_seq so the two were interchangeable; now that
        // eviction-aware allocators decouple them, the meaningful invariant is
        // the physical cap (checked at maybe_evict time against the supplied kv).
        assert!(max_seq >= budget + beta, "max_seq < budget+beta; eviction can never fire");
        let n_bands = head_dim / 2;
        let centers_per_layer = n_heads * n_bands * 3;

        let mut centers_flat = Vec::with_capacity(fa_layer_ids.len() * centers_per_layer);
        for &layer_idx in &fa_layer_ids {
            for h in 0..n_heads {
                for f in 0..n_bands {
                    let c = centers.get(layer_idx, h, f);
                    centers_flat.push(c.eq_re);
                    centers_flat.push(c.eq_im);
                    centers_flat.push(c.e_abs_q);
                }
            }
        }
        let centers_dev = gpu.upload_f32(&centers_flat, &[centers_flat.len()])?;

        // Reusable scratch. For compaction temp we allocate enough bytes to
        // hold `budget` rows in the widest layout (Q8_0 V, since K in asym3
        // is narrower). Passing different bytes_per_pos to the gather
        // kernel is safe as long as `budget × bytes_per_pos` fits.
        let q8_bpp = n_kv_heads * (head_dim / 32) * 34;
        let asym3_k_bpp = n_kv_heads * (4 + (head_dim * 3) / 8);
        let asym4_k_bpp = n_kv_heads * (4 + head_dim / 2);
        let asym2_k_bpp = n_kv_heads * (4 + head_dim / 4);
        let widest_bpp = q8_bpp.max(asym3_k_bpp).max(asym4_k_bpp).max(asym2_k_bpp);

        let scores_buf = gpu.alloc_tensor(&[n_heads * max_seq], DType::F32)?;
        let k_compact = gpu.zeros(&[(budget * widest_bpp + 3) / 4], DType::F32)?;
        let v_compact = gpu.zeros(&[(budget * q8_bpp + 3) / 4], DType::F32)?;
        let retain_dev = gpu.alloc_tensor(&[budget], DType::F32)?;

        Ok(Self {
            centers_dev, fa_layer_ids, centers_per_layer,
            budget, beta, n_heads, n_kv_heads, head_dim, n_rot, rope_theta, max_seq,
            scores_buf, k_compact, v_compact, retain_dev,
            eviction_count: std::cell::Cell::new(0),
        })
    }

    /// If the physical cache has grown to `budget + beta` (or beyond), run
    /// score → top-B → compact on every FA layer, update `kv.compact_offset`,
    /// and return `Some(budget)` as the new physical count. Otherwise
    /// return `None` — the caller keeps the current physical position.
    pub fn maybe_evict(
        &self,
        gpu: &mut Gpu,
        kv: &mut crate::llama::KvCache,
        current_physical: usize,
    ) -> HipResult<Option<EvictionResult>> {
        if current_physical < self.budget + self.beta {
            return Ok(None);
        }
        let absolute_pos = current_physical + kv.compact_offset;
        let p_q = absolute_pos as f32;

        enum Mode { Q8, Asym2, Asym3, Asym4 }
        let (mode, k_bytes_per_pos) = if kv.quant_asym3 {
            (Mode::Asym3, self.n_kv_heads * (4 + (self.head_dim * 3) / 8))
        } else if kv.quant_asym4 {
            (Mode::Asym4, self.n_kv_heads * (4 + self.head_dim / 2))
        } else if kv.quant_asym2 {
            (Mode::Asym2, self.n_kv_heads * (4 + self.head_dim / 4))
        } else if kv.quant_q8 {
            (Mode::Q8, self.n_kv_heads * (self.head_dim / 32) * 34)
        } else {
            panic!("TriAttention eviction only supports Q8, asym2, asym3, asym4 KV modes for now");
        };
        let v_bytes_per_pos = self.n_kv_heads * (self.head_dim / 32) * 34;

        let mut last_retain: Vec<u32> = Vec::new();
        for (fa_i, &layer_idx) in self.fa_layer_ids.iter().enumerate() {
            let offset = fa_i * self.centers_per_layer;
            let centers_layer = self.centers_dev.sub_offset(offset, self.centers_per_layer);
            match mode {
                Mode::Asym3 => gpu.triattn_score_asym3(
                    &kv.k_gpu[layer_idx], &centers_layer,
                    kv.givens_cos.as_ref().expect("asym3 KV must have cos table"),
                    kv.givens_sin.as_ref().expect("asym3 KV must have sin table"),
                    &self.scores_buf,
                    self.n_heads, self.n_kv_heads, self.head_dim,
                    self.n_rot, self.rope_theta, p_q, current_physical,
                )?,
                Mode::Asym4 => gpu.triattn_score_asym4(
                    &kv.k_gpu[layer_idx], &centers_layer,
                    kv.givens_cos.as_ref().expect("asym4 KV must have cos table"),
                    kv.givens_sin.as_ref().expect("asym4 KV must have sin table"),
                    &self.scores_buf,
                    self.n_heads, self.n_kv_heads, self.head_dim,
                    self.n_rot, self.rope_theta, p_q, current_physical,
                )?,
                Mode::Asym2 => gpu.triattn_score_asym2(
                    &kv.k_gpu[layer_idx], &centers_layer,
                    kv.givens_cos.as_ref().expect("asym2 KV must have cos table"),
                    kv.givens_sin.as_ref().expect("asym2 KV must have sin table"),
                    &self.scores_buf,
                    self.n_heads, self.n_kv_heads, self.head_dim,
                    self.n_rot, self.rope_theta, p_q, current_physical,
                )?,
                Mode::Q8 => gpu.triattn_score_q8(
                    &kv.k_gpu[layer_idx], &centers_layer, &self.scores_buf,
                    self.n_heads, self.n_kv_heads, self.head_dim,
                    self.n_rot, self.rope_theta, p_q, current_physical,
                )?,
            }
            gpu.hip.device_synchronize()?;

            let scores = gpu.download_f32(&self.scores_buf)?;
            let retain = compute_retain_indices(
                &scores[..self.n_heads * current_physical],
                self.n_heads, current_physical, self.budget,
            );
            let retain_bytes: Vec<u8> = retain.iter().flat_map(|&x| (x as i32).to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&self.retain_dev.buf, &retain_bytes)?;

            gpu.kv_compact_gather(&kv.k_gpu[layer_idx], &self.k_compact, &self.retain_dev, k_bytes_per_pos, self.budget)?;
            gpu.kv_compact_gather(&kv.v_gpu[layer_idx], &self.v_compact, &self.retain_dev, v_bytes_per_pos, self.budget)?;
            gpu.hip.device_synchronize()?;

            gpu.hip.memcpy_dtod_at(&kv.k_gpu[layer_idx].buf, 0, &self.k_compact.buf, 0, self.budget * k_bytes_per_pos)?;
            gpu.hip.memcpy_dtod_at(&kv.v_gpu[layer_idx].buf, 0, &self.v_compact.buf, 0, self.budget * v_bytes_per_pos)?;
            last_retain = retain;
        }

        kv.compact_offset += current_physical - self.budget;
        self.eviction_count.set(self.eviction_count.get() + 1);
        Ok(Some(EvictionResult { new_physical: self.budget, retain_mask: last_retain }))
    }

    /// Release all GPU buffers held by the context. Consumed by value;
    /// the daemon calls this on unload to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.centers_dev);
        let _ = gpu.free_tensor(self.scores_buf);
        let _ = gpu.free_tensor(self.k_compact);
        let _ = gpu.free_tensor(self.v_compact);
        let _ = gpu.free_tensor(self.retain_dev);
    }
}

// ─── Pearson correlation helper ──────────────────────────────────────────

pub fn pearson(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f32;
    if n < 2.0 { return 0.0; }
    let mx: f32 = x.iter().sum::<f32>() / n;
    let my: f32 = y.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut vx = 0.0f32;
    let mut vy = 0.0f32;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom > 1e-20 { cov / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_means_one_sample() {
        let mut a = BandAccumulator::default();
        a.add(0.5, -0.3);
        let c = a.finalize();
        assert!((c.eq_re - 0.5).abs() < 1e-6);
        assert!((c.eq_im - (-0.3)).abs() < 1e-6);
        let expected_abs = (0.25f32 + 0.09).sqrt();
        assert!((c.e_abs_q - expected_abs).abs() < 1e-6);
    }

    #[test]
    fn mrl_constant_vectors_is_one() {
        // If every sample is identical, R_f should be ≈ 1.
        let mut a = BandAccumulator::default();
        for _ in 0..100 {
            a.add(3.0, 4.0);
        }
        let c = a.finalize();
        assert!((c.mrl() - 1.0).abs() < 1e-5, "got mrl={}", c.mrl());
    }

    #[test]
    fn mrl_uniform_directions_is_zero() {
        // Vectors uniformly distributed around the origin → R ≈ 0.
        let mut a = BandAccumulator::default();
        let n = 8;
        for k in 0..n {
            let theta = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
            a.add(theta.cos(), theta.sin());
        }
        let c = a.finalize();
        assert!(c.mrl() < 1e-4, "got mrl={} for uniformly-dispersed samples", c.mrl());
    }

    #[test]
    fn rope_frequency_monotonic_decreasing() {
        let c = TriAttnCenters::new(1, 1, 256, 10000.0, 1.0);
        // ω_0 = 1 (no rotation), ω_{d/2-1} = very small.
        assert!((c.omega(0) - 1.0).abs() < 1e-5);
        for f in 1..128 {
            assert!(c.omega(f) < c.omega(f - 1), "omega should decrease with band");
        }
    }

    #[test]
    fn roundtrip_save_load() {
        let tmp = std::env::temp_dir().join("triattn_roundtrip_test.bin");
        let mut c = TriAttnCenters::new(2, 3, 4, 1_000_000.0, 1.0);
        for l in 0..c.n_layers {
            for h in 0..c.n_heads {
                for f in 0..c.n_bands() {
                    c.set(l, h, f, BandCenter {
                        eq_re: (l * 100 + h * 10 + f) as f32,
                        eq_im: -((l * 100 + h * 10 + f) as f32),
                        e_abs_q: (l + h + f) as f32 + 1.0,
                    });
                }
            }
        }
        c.save(&tmp).unwrap();
        let d = TriAttnCenters::load(&tmp).unwrap();
        assert_eq!(d.n_layers, c.n_layers);
        assert_eq!(d.n_heads, c.n_heads);
        assert_eq!(d.head_dim, c.head_dim);
        for i in 0..c.centers.len() {
            assert!((d.centers[i].eq_re - c.centers[i].eq_re).abs() < 1e-6);
            assert!((d.centers[i].eq_im - c.centers[i].eq_im).abs() < 1e-6);
            assert!((d.centers[i].e_abs_q - c.centers[i].e_abs_q).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn retain_indices_picks_top_by_zscore_max() {
        // n_heads=2, seq_len=6. Build a scores grid where head 0 likes
        // positions {1, 3, 5} and head 1 likes {0, 3}. Budget 3 should
        // retain the three positions whose *max z-score across heads*
        // is highest: {0, 3, 5} (3 wins both, 5 is head 0's peak, 0 is
        // head 1's peak).
        let seq = 6;
        let scores = vec![
            // head 0
            0.0, 2.0, 0.0, 3.0, 0.0, 2.5,
            // head 1
            3.0, 0.0, 0.0, 2.5, 0.0, 0.0,
        ];
        let kept = super::compute_retain_indices(&scores, 2, seq, 3);
        assert_eq!(kept, vec![0, 3, 5]);
    }

    #[test]
    fn retain_indices_sorted_ascending() {
        let scores: Vec<f32> = (0..10).rev().map(|i| i as f32).collect();
        let kept = super::compute_retain_indices(&scores, 1, 10, 4);
        // Highest scores are at positions 0,1,2,3. Must come back sorted ascending.
        assert_eq!(kept, vec![0, 1, 2, 3]);
    }

    #[test]
    fn pearson_identical_streams_is_one() {
        let x: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert!((pearson(&x, &x) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn s_trig_collapses_to_dot_when_constant_q() {
        // When Q/K are exactly at their centers (perfect concentration) and
        // there's only one band, S_trig reduces to ||q||·||k||·cos(angle),
        // which is the real-part of q·conj(k) — i.e. the standard dot
        // product of a 2D Q and K (pre-RoPE at pos 0, so RoPE is identity).
        let c = TriAttnCenters::new(1, 1, 2, 10000.0, 1.0);
        // Band 0, Q center = (1, 0). K post-RoPE at p_k=0 = K pre-RoPE = (cos θ, sin θ).
        let theta = 0.7f32;
        let k_mag = 1.0f32;
        let k_phase = theta;
        let centers_slice = &[BandCenter { eq_re: 1.0, eq_im: 0.0, e_abs_q: 1.0 }];
        let k_bands = &[(k_mag, k_phase)];
        let s = s_trig(centers_slice, k_bands, 0.0, |f| c.omega(f));
        let expected = theta.cos();
        assert!((s - expected).abs() < 1e-5, "s_trig={}, expected={}", s, expected);
    }
}
