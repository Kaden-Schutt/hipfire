// SPDX-License-Identifier: Apache-2.0
// hipfire — Tier-1 calibration collector (lib-ified core).
//
//! The reusable, model-agnostic calibration collector: an [`ActivationCapture`]
//! that accumulates a per-tensor GPTQ Hessian (`Σ x·xᵀ`) and imatrix diagonal
//! (`Σ x²`) on-GPU via the `calib_*_reduce_f32` kernels, and drains to HFQ
//! tensors (`<name>.hessian` [K,K] + `<name>.imatrix` [K], F32 = quant_type 2)
//! plus an internal-consistency metric (`diag(Σxxᵀ)` must equal `Σx²`).
//!
//! This is generic (rdna-compute + the HFQ writer only) so it sits in
//! hipfire-runtime without a cycle on the arch crates. Callers (the
//! `collect_artifacts` CLI, the daemon `Collect` op) own the forward loop +
//! the model-specific taps (MoE router histogram, KLDREF) and arm this via
//! `gpu.active_capture = Some(Arc::new(CalibCollector::default()))`.

use crate::hfq::HfqMemTensor;
use rdna_compute::{ActivationCapture, DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::sync::Mutex;

/// Rows buffered per tensor before flushing the outer-product. A single
/// `calib_hessian_outer_f32` over `[FLUSH_BATCH, K]` is ~FLUSH_BATCH× more
/// efficient than per-token (N=1) launches (the tiled GEMM is built for N≥16),
/// so this is the dominant calibration-throughput lever.
const FLUSH_BATCH: usize = 256;

/// Per-tensor on-GPU accumulators + a small activation row buffer.
struct Acc {
    diag: GpuTensor,      // [K]   Σx²  (imatrix)
    h: Option<GpuTensor>, // [K,K] Σxxᵀ (Hessian); `None` = imatrix-only tensor
    /// Host f64 reference accumulator (`Some` only under `HIPFIRE_CALIB_F64_AUDIT`).
    /// The GPU outer-product accumulates `Σxxᵀ` in f32; RDNA has no f64 matrix
    /// units and only ~1:16 scalar f64, so a faithful f64 reference is computed
    /// CPU-side from the same staged rows. `drain` then reports the max relative
    /// f32-vs-f64 divergence — measure-first before deciding whether f32
    /// accumulation needs replacing for large token counts.
    h_f64: Option<Vec<f64>>,
    buf: GpuTensor,  // [FLUSH_BATCH, K] staged activation rows
    buf_rows: usize, // rows currently staged in `buf`
    k: usize,
    n_tokens: u64,
}

impl Acc {
    /// Reduce the staged rows into the accumulators (one batched launch each),
    /// then reset the buffer. No-op when empty. Imatrix-only tensors (`h` is
    /// `None`) skip the [K,K] outer-product — this is how MoE routed experts
    /// are captured: a full per-expert Hessian (256 experts × ~48 layers ×
    /// [K,K]) is ~196 GB and does not fit, but the imatrix (Σx², a K-vector)
    /// is ~100 MB and is the importance signal AWQ-style quant needs.
    fn flush(&mut self, gpu: &mut Gpu) {
        if self.buf_rows == 0 {
            return;
        }
        gpu.calib_sumsq_reduce_f32(&self.buf, &self.diag, self.buf_rows, self.k)
            .unwrap();
        if let Some(h) = &self.h {
            gpu.calib_hessian_outer_f32(&self.buf, h, self.buf_rows, self.k)
                .unwrap();
        }
        // Audit: accumulate the same rows in f64 on the CPU (no GPU f64 path).
        if let Some(h_f64) = &mut self.h_f64 {
            let k = self.k;
            let rows = gpu
                .download_f32(&self.buf)
                .expect("download buf (f64 audit)");
            for r in 0..self.buf_rows {
                let x = &rows[r * k..r * k + k];
                for i in 0..k {
                    let xi = x[i] as f64;
                    let hrow = &mut h_f64[i * k..i * k + k];
                    for j in 0..k {
                        hrow[j] += xi * x[j] as f64;
                    }
                }
            }
        }
        self.buf_rows = 0;
    }
}

/// Unified Hessian + imatrix collector. Arm via `gpu.active_capture`.
///
/// By default every captured tensor accumulates a full [K,K] Hessian. Tensors
/// whose canonical name contains any of `imatrix_only_substr` accumulate only
/// the imatrix (Σx²); used for MoE routed experts whose full Hessians do not
/// fit in memory (see [`Acc::flush`]).
#[derive(Default)]
pub struct CalibCollector {
    accs: Mutex<HashMap<String, Acc>>,
    imatrix_only_substr: Vec<String>,
    /// When set (`HIPFIRE_CALIB_F64_AUDIT=1`), also accumulate each Hessian in
    /// f64 on the CPU and report the f32-vs-f64 divergence in `drain`. Opt-in,
    /// slow (CPU outer-products) — a measurement tool, not the default path.
    f64_audit: bool,
}

/// `HIPFIRE_CALIB_F64_AUDIT=1` → run the CPU f64 reference accumulation.
fn f64_audit_enabled() -> bool {
    std::env::var("HIPFIRE_CALIB_F64_AUDIT").ok().as_deref() == Some("1")
}

impl CalibCollector {
    pub fn new() -> Self {
        Self {
            accs: Mutex::new(HashMap::new()),
            imatrix_only_substr: Vec::new(),
            f64_audit: f64_audit_enabled(),
        }
    }

    /// Collector that stores imatrix-only (no [K,K] Hessian) for any tensor
    /// whose name contains one of `substr` (e.g. `".experts."` for MoE).
    pub fn with_imatrix_only(substr: Vec<String>) -> Self {
        Self {
            accs: Mutex::new(HashMap::new()),
            imatrix_only_substr: substr,
            f64_audit: f64_audit_enabled(),
        }
    }

    fn wants_hessian(&self, name: &str) -> bool {
        !self.imatrix_only_substr.iter().any(|s| name.contains(s))
    }

    /// Number of distinct tensors captured so far.
    pub fn len(&self) -> usize {
        self.accs.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Per-tensor descriptors (no GPU work): `name`, whether it has a full
    /// Hessian, `k`, and `n_tokens`. The caller uses these to compute counts +
    /// `name -> n_tokens` provenance for the metadata BEFORE the streaming write
    /// (the HFQM index/metadata must be written ahead of the payloads).
    pub fn tensor_descriptors(&self) -> Vec<CalibTensorDesc> {
        let accs = self.accs.lock().unwrap();
        let mut names: Vec<&String> = accs.keys().collect();
        names.sort();
        names
            .iter()
            .map(|name| {
                let acc = &accs[*name];
                CalibTensorDesc {
                    name: (*name).clone(),
                    has_hessian: acc.h.is_some(),
                    k: acc.k,
                    n_tokens: acc.n_tokens,
                }
            })
            .collect()
    }

    /// Stream the accumulated tensors into an HFQM `.calib.hfq` at `path`,
    /// **one tensor at a time** (download → normalize `/ n_tokens` → write →
    /// drop), so peak host memory is a single Hessian rather than all of them
    /// (a 9B is ~32 GB if materialized at once). `extra` holds any small
    /// already-in-RAM tensors (e.g. KLDREF) the caller wants in the same
    /// package. The metadata + index are written first (payload sizes are
    /// deterministic from `k`), then the payloads stream. Returns the max
    /// relative `diag(H)`-vs-`Σx²` consistency error. Also runs the optional
    /// f64 audit (`HIPFIRE_CALIB_F64_AUDIT`) during the per-Hessian download.
    pub fn write_streaming(
        &self,
        gpu: &mut Gpu,
        path: &std::path::Path,
        arch_id: u32,
        metadata_json: &str,
        extra: &[HfqMemTensor],
    ) -> std::io::Result<f32> {
        use crate::hfq::{write_hfqm_package_streaming, HfqStreamEntry};
        use std::cell::{Cell, RefCell};

        let mut accs = self.accs.lock().unwrap();
        // Fold any staged activation rows before reading the accumulators.
        for acc in accs.values_mut() {
            acc.flush(gpu);
        }
        let mut names: Vec<String> = accs.keys().cloned().collect();
        names.sort();

        // Build the index entries (payload sizes from `k`) + a parallel plan of
        // how to produce each payload, in the SAME order.
        enum Plan {
            Hessian(String),
            Imatrix(String),
            Extra(usize),
        }
        let mut entries: Vec<HfqStreamEntry> = Vec::new();
        let mut plan: Vec<Plan> = Vec::new();
        for name in &names {
            let acc = &accs[name];
            if acc.h.is_some() {
                entries.push(HfqStreamEntry {
                    name: format!("{name}.hessian"),
                    quant_type: 2,
                    shape: vec![acc.k as u32, acc.k as u32],
                    group_size: 0,
                    data_len: (acc.k * acc.k * 4) as u64,
                });
                plan.push(Plan::Hessian(name.clone()));
            }
            entries.push(HfqStreamEntry {
                name: format!("{name}.imatrix"),
                quant_type: 2,
                shape: vec![acc.k as u32],
                group_size: 0,
                data_len: (acc.k * 4) as u64,
            });
            plan.push(Plan::Imatrix(name.clone()));
        }
        for (j, t) in extra.iter().enumerate() {
            entries.push(HfqStreamEntry {
                name: t.name.clone(),
                quant_type: t.quant_type,
                shape: t.shape.clone(),
                group_size: t.group_size,
                data_len: t.data.len() as u64,
            });
            plan.push(Plan::Extra(j));
        }

        let max_consistency = Cell::new(0.0f32);
        let audit_max = Cell::new(0.0f64);
        let audit_n = Cell::new(0usize);
        let audit_worst = RefCell::new(String::new());
        let io_err = |e: rdna_compute::HipError| std::io::Error::other(e.to_string());

        write_hfqm_package_streaming(path, arch_id, metadata_json, &entries, |i, w| {
            match &plan[i] {
                Plan::Hessian(name) => {
                    let acc = &accs[name];
                    let inv = 1.0 / acc.n_tokens.max(1) as f32;
                    let h = gpu.download_f32(acc.h.as_ref().unwrap()).map_err(io_err)?;
                    let diag = gpu.download_f32(&acc.diag).map_err(io_err)?;
                    let mut mc = max_consistency.get();
                    for c in 0..acc.k {
                        let rel = (h[c * acc.k + c] - diag[c]).abs() / diag[c].abs().max(1.0);
                        mc = mc.max(rel);
                    }
                    max_consistency.set(mc);
                    if let Some(h_ref) = &acc.h_f64 {
                        let mut tmax = 0.0f64;
                        for idx in 0..acc.k * acc.k {
                            let r = h_ref[idx];
                            tmax = tmax.max((h[idx] as f64 - r).abs() / r.abs().max(1.0));
                        }
                        audit_n.set(audit_n.get() + 1);
                        if tmax > audit_max.get() {
                            audit_max.set(tmax);
                            *audit_worst.borrow_mut() = name.clone();
                        }
                    }
                    write_f32_scaled(w, &h, inv)
                }
                Plan::Imatrix(name) => {
                    let acc = &accs[name];
                    let inv = 1.0 / acc.n_tokens.max(1) as f32;
                    let diag = gpu.download_f32(&acc.diag).map_err(io_err)?;
                    write_f32_scaled(w, &diag, inv)
                }
                Plan::Extra(j) => w.write_all(&extra[*j].data),
            }
        })?;

        if audit_n.get() > 0 {
            eprintln!(
                "F64 AUDIT: max f32-vs-f64 Σxxᵀ rel-diff = {:.3e} over {} Hessians (worst: {})",
                audit_max.get(),
                audit_n.get(),
                audit_worst.borrow()
            );
        }
        Ok(max_consistency.get())
    }
}

/// Per-tensor descriptor from [`CalibCollector::tensor_descriptors`].
pub struct CalibTensorDesc {
    pub name: String,
    pub has_hessian: bool,
    pub k: usize,
    pub n_tokens: u64,
}

/// Stream `v * scale` as little-endian f32 to `w` in bounded chunks (so a
/// multi-hundred-MB Hessian never materializes a second full byte buffer).
fn write_f32_scaled(w: &mut dyn std::io::Write, v: &[f32], scale: f32) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(16384);
    for &x in v {
        buf.extend_from_slice(&(x * scale).to_le_bytes());
        if buf.len() >= 16384 {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    if !buf.is_empty() {
        w.write_all(&buf)?;
    }
    Ok(())
}

impl ActivationCapture for CalibCollector {
    fn capture(&self, gpu: &mut Gpu, tensor_name: &str, input: &GpuTensor, n: usize, k: usize) {
        // n/k come from the gemm — `input` is a shared scratch buffer whose shape
        // (max(dim,hidden)) does NOT reflect the linear's input width.
        let mut accs = self.accs.lock().unwrap();
        if !accs.contains_key(tensor_name) {
            let diag = gpu.zeros(&[k], DType::F32).unwrap();
            let h = if self.wants_hessian(tensor_name) {
                Some(gpu.zeros(&[k, k], DType::F32).unwrap())
            } else {
                None
            };
            let buf = gpu.zeros(&[FLUSH_BATCH, k], DType::F32).unwrap();
            let h_f64 = if self.f64_audit && h.is_some() {
                Some(vec![0.0f64; k * k])
            } else {
                None
            };
            accs.insert(
                tensor_name.to_string(),
                Acc {
                    diag,
                    h,
                    h_f64,
                    buf,
                    buf_rows: 0,
                    k,
                    n_tokens: 0,
                },
            );
        }
        let acc = accs.get_mut(tensor_name).unwrap();
        // Stage each activation row into the flush buffer; the actual reductions
        // run a single batched launch per FLUSH_BATCH rows (Acc::flush). `input`
        // is a shared scratch buffer of width `row_stride` ≥ k, so copy the first
        // k columns of each of the n rows.
        let row_stride = input.numel() / n.max(1);
        for r in 0..n {
            if acc.buf_rows == FLUSH_BATCH {
                acc.flush(gpu);
            }
            gpu.memcpy_dtod_at_auto(
                &acc.buf.buf,
                acc.buf_rows * k * 4,
                &input.buf,
                r * row_stride * 4,
                k * 4,
            )
            .unwrap();
            acc.buf_rows += 1;
        }
        acc.n_tokens += n as u64;
    }
}

/// log(Σ exp(logits)) — numerically stable. For the KLDREF reference (callers
/// that tap lm-head logits).
pub fn logsumexp(logits: &[f32]) -> f32 {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    m + logits.iter().map(|&x| (x - m).exp()).sum::<f32>().ln()
}

/// Top-`k` (index, logit) descending — for the KLDREF reference.
pub fn topk_logits(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    idx.truncate(k);
    idx.into_iter().map(|i| (i, logits[i as usize])).collect()
}
