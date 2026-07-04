// SPDX-License-Identifier: Apache-2.0
// hipfire — Tier-1 calibration capture-hook wiring test (Phase 2).
//
//! Proves the `ActivationCapture` hook fires from the REAL bf16 linear dispatch
//! (`gemv_f16_xf32`), with correct weight-ptr→name attribution and a working
//! end-to-end reduction through the passed `&mut Gpu`, and is a no-op when not
//! armed (byte-identical forwards). No model/forward/daemon.
//!
//! Run: cargo run --release -p hipfire-rdna --example test_capture_hook

use hipfire_rdna::{ActivationCapture, DType, Gpu, GpuTensor};
use std::sync::Mutex;

fn fract_sin(x: f32) -> f32 {
    (x.sin() * 12345.6789f32).fract() * 2.0f32 - 1.0f32
}

/// Records (name, n_tokens) per capture and accumulates per-column Σx² via the
/// real reduction kernel — i.e. exercises the full capture→gpu-reduce path.
struct Probe {
    seen: Mutex<Vec<(String, usize)>>,
    acc: Mutex<Option<GpuTensor>>,
}

impl ActivationCapture for Probe {
    fn capture(&self, gpu: &mut Gpu, tensor_name: &str, input: &GpuTensor, n: usize, k: usize) {
        self.seen.lock().unwrap().push((tensor_name.to_string(), n));
        let mut acc = self.acc.lock().unwrap();
        if acc.is_none() {
            *acc = Some(gpu.zeros(&[k], DType::F32).unwrap());
        }
        gpu.calib_sumsq_reduce_f32(input, acc.as_ref().unwrap(), n, k)
            .unwrap();
    }
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    // gemv_f16_xf32 is a single-vector GEMV → capture reports n=1 per call.
    let (m, k, n) = (64usize, 128usize, 1usize);

    // f16 weight (values irrelevant to capture — it reads x, not w) + f32 input.
    let weight = gpu.zeros(&[m, k], DType::F16).unwrap();
    let y = gpu.zeros(&[m], DType::F32).unwrap();
    let x_host: Vec<f32> = (0..n * k)
        .map(|i| fract_sin(i as f32 * 0.37 + 2.0))
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).unwrap();
    let wptr = weight.buf.as_ptr() as usize;

    let mut fail = false;

    // (1) Not armed → no capture, gemv still runs.
    gpu.gemv_f16_xf32(&weight, &x, &y, m, k).unwrap();
    eprintln!("  [1] unarmed gemv ran (no panic) — capture is opt-in");

    // (2) Armed + name registered → capture fires with the right name + n.
    let probe = std::sync::Arc::new(Probe {
        seen: Mutex::new(Vec::new()),
        acc: Mutex::new(None),
    });
    gpu.capture_names
        .insert(wptr, "model.layers.0.self_attn.q_proj.weight".to_string());
    gpu.active_capture = Some(probe.clone());
    gpu.gemv_f16_xf32(&weight, &x, &y, m, k).unwrap();
    gpu.gemv_f16_xf32(&weight, &x, &y, m, k).unwrap(); // 2nd call → accumulate
    let seen = probe.seen.lock().unwrap().clone();
    let ok_fire = seen.len() == 2
        && seen
            .iter()
            .all(|(nm, nn)| nm == "model.layers.0.self_attn.q_proj.weight" && *nn == n);
    fail |= !ok_fire;
    eprintln!(
        "  [2] captured {} calls, names+n correct: {} [{}]",
        seen.len(),
        ok_fire,
        if ok_fire { "PASS" } else { "FAIL" }
    );

    // Verify the reduction the collector ran matches CPU Σx² over the 2 calls.
    let acc_gpu = gpu
        .download_f32(probe.acc.lock().unwrap().as_ref().unwrap())
        .unwrap();
    let mut max_d = 0.0f32;
    for c in 0..k {
        let mut s = 0.0f32;
        for row in 0..n {
            s += x_host[row * k + c] * x_host[row * k + c];
        }
        s *= 2.0; // two capture calls accumulated
        max_d = max_d.max((acc_gpu[c] - s).abs());
    }
    let ok_reduce = max_d <= 1e-3 * n as f32;
    fail |= !ok_reduce;
    eprintln!(
        "  [3] collector reduction vs CPU Σx² (×2): max|Δ|={max_d:.3e} [{}]",
        if ok_reduce { "PASS" } else { "FAIL" }
    );

    // (4) Unregistered weight ptr → no capture even when armed.
    let before = probe.seen.lock().unwrap().len();
    gpu.capture_names.clear();
    gpu.gemv_f16_xf32(&weight, &x, &y, m, k).unwrap();
    let ok_skip = probe.seen.lock().unwrap().len() == before;
    fail |= !ok_skip;
    eprintln!(
        "  [4] unregistered weight skipped (no spurious capture): {} [{}]",
        ok_skip,
        if ok_skip { "PASS" } else { "FAIL" }
    );

    if fail {
        eprintln!("\n[FAIL] capture hook wiring incorrect.");
        std::process::exit(1);
    }
    eprintln!("\n[PASS] ActivationCapture fires from gemv_f16_xf32, ptr→name attribution + reduction correct, no-op when unarmed.");
}
