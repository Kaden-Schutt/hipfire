// SPDX-License-Identifier: Apache-2.0
//! Layer-1 gate proof: a GEMV against a dtype with no kernel route must return a
//! classifiable `HipError::is_unsupported()` error — NOT panic / assert deep in a
//! kernel, and NOT surface as a generic code-0 error mistaken for an infra crash.
//!
//! This reproduces the failure class from the rq-protect lm_head report: a quant
//! variant the gemv path can't dispatch. We stand in for it with `DType::Raw`
//! (no GEMV key in any family), which exercises the same resolve→Unsupported→
//! dispatch_err_to_hip path that any genuinely-unsupported dtype hits.
//!
//!   cargo run --release -p hipfire-runtime --example gate_unsupported_gemv
//!
//! Exit 0 = gate works (clean unsupported error). Exit 1 = regression.

use hipfire_runtime::weights::{weight_gemv, WeightTensor};
use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            // No GPU here is not a pass and not a failure of the gate itself.
            println!("SKIP gate_unsupported_gemv: GPU init failed: {e}");
            return;
        }
    };

    let (m, k) = (64usize, 64usize);
    // A buffer with no meaningful element interpretation; Raw has no GEMV route.
    let bytes = vec![0u8; m * k];
    let mut buf = gpu.upload_raw(&bytes, &[bytes.len()]).unwrap();
    buf.dtype = DType::Raw;
    let w = WeightTensor {
        buf,
        gpu_dtype: DType::Raw,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    };

    let x = gpu.upload_f32(&vec![0.0f32; k], &[1, k]).unwrap();
    let y = gpu.alloc_tensor(&[m], DType::F32).unwrap();

    match weight_gemv(&mut gpu, &w, &x, &y) {
        Ok(()) => {
            println!("FAIL: weight_gemv returned Ok for an unsupported dtype (DType::Raw)");
            std::process::exit(1);
        }
        Err(e) if e.is_unsupported() => {
            println!("PASS: unsupported dtype refused cleanly -> {e}");
        }
        Err(e) => {
            println!("FAIL: unsupported dtype produced a non-unsupported error: {e}");
            std::process::exit(1);
        }
    }
}
