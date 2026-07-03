// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Activation calibration for diffusion weight quantization.
//!
//! During a CPU-reference generation run we accumulate, per weight matrix, the
//! per-input-channel activation power (`imatrix`, `Σ x_j²`) and — for linear
//! layers whose input dimension fits a budget — the full activation Hessian
//! (`Σ x xᵀ`). These feed the activation-calibrated oq4++/oq8 packers
//! (`hipfire_quantize::ldlq::oq4_ldlq_pack`, which take a `[K,K]` row-major
//! Hessian) and AWQ-style salience scaling.
//!
//! Mechanism (no forward-signature changes): a thread-local registry maps each
//! weight tensor's `data.as_ptr()` to its name (filled in [`CpuTensor::from_hfq`]
//! while calibration is armed; the `Vec<f32>` backing a `CpuTensor` is stable for
//! the model's lifetime). The CPU linear/matmul entry points look the weight
//! pointer up and fold the layer's input activations into that tensor's
//! accumulators. Everything runs on the generation thread (the denoise loop is
//! sequential; rayon parallelism inside conv/linear happens *after* we observe
//! the input), so a plain thread-local is sufficient.
//!
//! Output: a `.calib.hfq` HFQM package (`<base>.hessian` F32 `[K,K]`,
//! `<base>.imatrix` F32 `[K]`) that `hipfire_quantize::hessian_io::HessianSidecar`
//! reads unchanged. `base` is the weight name with the trailing `.weight`
//! stripped — the key the quantizer queries.

use std::cell::RefCell;
use std::collections::HashMap;

use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

/// HFQM quant_type for dense F32 tensors (matches `hessian_io::QUANT_TYPE_F32`).
const QT_F32: u8 = 2;

struct TensorAccum {
    name: String,
    k: usize,
    /// `Σ x_j²` over all observed input vectors (length `k`).
    imatrix: Vec<f64>,
    /// `Σ x xᵀ` row-major `[k,k]` for linears under the Hessian K budget; `None`
    /// otherwise (e.g. K too large, or `k % 256 != 0` so LDLQ can't use it).
    hessian: Option<Vec<f64>>,
}

struct CalibState {
    name_by_ptr: HashMap<usize, String>,
    accum: HashMap<usize, TensorAccum>,
    hessian_max_k: usize,
}

thread_local! {
    static CALIB: RefCell<Option<CalibState>> = const { RefCell::new(None) };
}

/// True while a calibration run is armed on this thread.
pub fn calib_active() -> bool {
    CALIB.with(|c| c.borrow().is_some())
}

/// Arm calibration. Linears with `k <= hessian_max_k` and `k % 256 == 0` get a
/// full Hessian; all observed weights get an imatrix.
pub fn calib_begin(hessian_max_k: usize) {
    CALIB.with(|c| {
        *c.borrow_mut() = Some(CalibState {
            name_by_ptr: HashMap::new(),
            accum: HashMap::new(),
            hessian_max_k,
        });
    });
}

/// Register a freshly loaded weight tensor (called from `CpuTensor::from_hfq`).
pub(crate) fn calib_register(ptr: usize, name: &str) {
    CALIB.with(|c| {
        if let Some(s) = c.borrow_mut().as_mut() {
            s.name_by_ptr.entry(ptr).or_insert_with(|| name.to_string());
        }
    });
}

/// Fold a linear/matmul layer's input activations into its accumulators.
/// `input` is row-major `[rows, k]`; `weight_ptr` identifies the weight tensor.
pub(crate) fn calib_observe_matrix(weight_ptr: usize, input: &[f32], rows: usize, k: usize) {
    if k == 0 || rows == 0 {
        return;
    }
    CALIB.with(|c| {
        let mut guard = c.borrow_mut();
        let Some(state) = guard.as_mut() else {
            return;
        };
        let Some(name) = state.name_by_ptr.get(&weight_ptr).cloned() else {
            return; // not a registered weight (e.g. an activation×activation matmul)
        };
        let want_hessian = k <= state.hessian_max_k && k % 256 == 0;
        let entry = state
            .accum
            .entry(weight_ptr)
            .or_insert_with(|| TensorAccum {
                name,
                k,
                imatrix: vec![0.0; k],
                hessian: want_hessian.then(|| vec![0.0; k * k]),
            });
        if entry.k != k {
            return; // shape changed unexpectedly; skip rather than corrupt
        }
        for r in 0..rows {
            let row = &input[r * k..r * k + k];
            for (acc, &x) in entry.imatrix.iter_mut().zip(row) {
                *acc += (x as f64) * (x as f64);
            }
        }
        if let Some(h) = entry.hessian.as_mut() {
            accumulate_hessian(h, input, rows, k);
        }
    });
}

/// `H += Xᵀ X` for `X = [rows, k]` (row-major), parallel over output rows of `H`.
fn accumulate_hessian(h: &mut [f64], input: &[f32], rows: usize, k: usize) {
    use rayon::prelude::*;
    h.par_chunks_mut(k).enumerate().for_each(|(i, h_row)| {
        for r in 0..rows {
            let xi = input[r * k + i] as f64;
            if xi == 0.0 {
                continue;
            }
            let x_r = &input[r * k..r * k + k];
            for (hij, &xj) in h_row.iter_mut().zip(x_r) {
                *hij += xi * (xj as f64);
            }
        }
    });
}

/// Number of distinct weight tensors that received observations.
pub fn calib_observed_count() -> usize {
    CALIB.with(|c| c.borrow().as_ref().map(|s| s.accum.len()).unwrap_or(0))
}

/// Finish calibration, write the `.calib.hfq` package, and disarm. Returns
/// `(n_hessians, n_imatrices)`.
pub fn calib_finish_and_write(path: &std::path::Path) -> std::io::Result<(usize, usize)> {
    let state = CALIB.with(|c| c.borrow_mut().take());
    let Some(state) = state else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "calibration not armed",
        ));
    };
    let mut tensors: Vec<HfqMemTensor> = Vec::new();
    let (mut n_h, mut n_i) = (0usize, 0usize);
    for entry in state.accum.values() {
        let base = entry
            .name
            .strip_suffix(".weight")
            .unwrap_or(&entry.name)
            .to_string();
        let k = entry.k as u32;
        if let Some(h) = &entry.hessian {
            let data: Vec<u8> = h.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect();
            tensors.push(HfqMemTensor {
                name: format!("{base}.hessian"),
                quant_type: QT_F32,
                shape: vec![k, k],
                group_size: 0,
                data,
            });
            n_h += 1;
        }
        let data: Vec<u8> = entry
            .imatrix
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect();
        tensors.push(HfqMemTensor {
            name: format!("{base}.imatrix"),
            quant_type: QT_F32,
            shape: vec![k],
            group_size: 0,
            data,
        });
        n_i += 1;
    }
    let metadata = r#"{"artifact_kind":"calibration","producer":"hipfire-diffusion"}"#;
    write_hfqm_package_mem(path, 0, metadata, &tensors)?;
    Ok((n_h, n_i))
}

/// Summary of a calibration run.
#[derive(Debug, Clone)]
pub struct CalibrateSummary {
    pub observed_tensors: usize,
    pub hessians: usize,
    pub imatrices: usize,
}

/// Run a CPU-reference calibration pass over `prompts` and write the resulting
/// `.calib.hfq` to `output`. Arms calibration BEFORE loading the model so every
/// weight registers its pointer, runs `steps` denoise steps per prompt at
/// `width`×`height`, then serializes the accumulated Hessians/imatrices.
pub fn calibrate_diffusion_hfq(
    model: &std::path::Path,
    output: &std::path::Path,
    prompts: &[String],
    steps: u32,
    width: u32,
    height: u32,
    cfg_scale: f32,
    hessian_max_k: usize,
) -> anyhow::Result<CalibrateSummary> {
    calib_begin(hessian_max_k);
    // From here on, CpuTensor::from_hfq registers weight pointers.
    let pipeline = crate::DiffusionPipeline::open_hfq(model)?;
    let request = crate::DiffusionBatchRequest {
        prompts: prompts
            .iter()
            .enumerate()
            .map(|(i, p)| crate::DiffusionPrompt {
                prompt: p.clone(),
                negative_prompt: String::new(),
                seed: 1 + i as i64,
                subseed: None,
            })
            .collect(),
        conditioning: None,
        width,
        height,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps,
        cfg_scale,
        distilled_guidance_scale: None,
        scheduler: "Euler".to_string(),
        subseed_strength: 0.0,
        send_images: false,
        save_images: false,
    };
    pipeline.generate_batch_with_runtime_options(
        request,
        crate::DiffusionGenerationRuntimeOptions::cpu_reference(),
    )?;
    let observed = calib_observed_count();
    let (hessians, imatrices) = calib_finish_and_write(output)?;
    Ok(CalibrateSummary {
        observed_tensors: observed,
        hessians,
        imatrices,
    })
}
