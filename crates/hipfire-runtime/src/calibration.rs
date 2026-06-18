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
    buf: GpuTensor,       // [FLUSH_BATCH, K] staged activation rows
    buf_rows: usize,      // rows currently staged in `buf`
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
}

impl CalibCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Collector that stores imatrix-only (no [K,K] Hessian) for any tensor
    /// whose name contains one of `substr` (e.g. `".experts."` for MoE).
    pub fn with_imatrix_only(substr: Vec<String>) -> Self {
        Self {
            accs: Mutex::new(HashMap::new()),
            imatrix_only_substr: substr,
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

    /// Drain to HFQ tensors (`<name>.hessian` + `<name>.imatrix`, both finalized
    /// `/ n_tokens`), plus the max relative `diag(H)`-vs-`Σx²` error (should be
    /// ~0; the two reduction kernels must agree on the same activations) and a
    /// `name -> n_tokens` map for provenance.
    pub fn drain(&self, gpu: &mut Gpu) -> (Vec<HfqMemTensor>, f32, HashMap<String, u64>) {
        let mut accs = self.accs.lock().unwrap();
        // Flush any staged activation rows before reading the accumulators.
        for acc in accs.values_mut() {
            acc.flush(gpu);
        }
        let mut names: Vec<&String> = accs.keys().collect();
        names.sort();
        let mut tensors = Vec::with_capacity(names.len() * 2);
        let mut max_consistency = 0.0f32;
        let mut token_counts = HashMap::new();
        let f32_bytes = |v: &[f32]| -> Vec<u8> {
            let mut b = Vec::with_capacity(v.len() * 4);
            for &x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        };
        for name in &names {
            let acc = &accs[*name];
            let diag = gpu.download_f32(&acc.diag).expect("download imatrix");
            let inv = 1.0 / acc.n_tokens.max(1) as f32;
            // Full Hessian (dense projections). Imatrix-only tensors (MoE
            // routed experts) skip this and emit only `.imatrix`.
            if let Some(h_t) = &acc.h {
                let h = gpu.download_f32(h_t).expect("download hessian");
                for c in 0..acc.k {
                    let rel = (h[c * acc.k + c] - diag[c]).abs() / diag[c].abs().max(1.0);
                    max_consistency = max_consistency.max(rel);
                }
                let hessian: Vec<f32> = h.iter().map(|v| v * inv).collect();
                tensors.push(HfqMemTensor {
                    name: format!("{name}.hessian"),
                    quant_type: 2,
                    shape: vec![acc.k as u32, acc.k as u32],
                    group_size: 0,
                    data: f32_bytes(&hessian),
                });
            }
            let imatrix: Vec<f32> = diag.iter().map(|v| v * inv).collect();
            tensors.push(HfqMemTensor {
                name: format!("{name}.imatrix"),
                quant_type: 2,
                shape: vec![acc.k as u32],
                group_size: 0,
                data: f32_bytes(&imatrix),
            });
            token_counts.insert((*name).clone(), acc.n_tokens);
        }
        (tensors, max_consistency, token_counts)
    }
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
            accs.insert(
                tensor_name.to_string(),
                Acc {
                    diag,
                    h,
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
