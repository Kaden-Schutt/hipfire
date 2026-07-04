#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::print_literal,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::useless_vec,
    clippy::unnecessary_cast
)]

//! QA mirror for the GEMV example.

use std::process::ExitCode;

const SKIP_EXIT: u8 = 10;

fn main() -> ExitCode {
    match run() {
        Ok(msg) => {
            println!("GEMV QA: PASS - {msg}");
            ExitCode::SUCCESS
        }
        Err(Outcome::Skip(msg)) => {
            eprintln!("GEMV QA: SKIP - {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Err(Outcome::Fail(msg)) => {
            eprintln!("GEMV QA: FAIL - {msg}");
            ExitCode::from(1)
        }
    }
}

enum Outcome {
    Skip(String),
    Fail(String),
}

fn run() -> Result<String, Outcome> {
    let mut gpu = hipfire_rdna::Gpu::init()
        .map_err(|e| Outcome::Skip(format!("GPU init unavailable: {e}")))?;

    let m = 256usize;
    let k = 512usize;

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();

    let mut y_ref = vec![0.0f32; m];
    for row in 0..m {
        for col in 0..k {
            y_ref[row] += a[row * k + col] * x[col];
        }
    }

    let d_a = gpu
        .upload_f32(&a, &[m, k])
        .map_err(|e| Outcome::Fail(format!("upload A failed: {e}")))?;
    let d_x = gpu
        .upload_f32(&x, &[k])
        .map_err(|e| Outcome::Fail(format!("upload x failed: {e}")))?;
    let d_y = gpu
        .zeros(&[m], hipfire_rdna::DType::F32)
        .map_err(|e| Outcome::Fail(format!("alloc y failed: {e}")))?;

    gpu.gemv_f32(&d_a, &d_x, &d_y)
        .map_err(|e| Outcome::Fail(format!("gemv_f32 failed: {e}")))?;
    let y_gpu = gpu
        .download_f32(&d_y)
        .map_err(|e| Outcome::Fail(format!("download failed: {e}")))?;

    let mut max_err = 0.0f32;
    let mut errors = 0usize;
    for i in 0..m {
        let err = (y_gpu[i] - y_ref[i]).abs();
        max_err = max_err.max(err);
        if err > 0.01 {
            errors += 1;
        }
    }

    gpu.free_tensor(d_a)
        .map_err(|e| Outcome::Fail(format!("free A failed: {e}")))?;
    gpu.free_tensor(d_x)
        .map_err(|e| Outcome::Fail(format!("free x failed: {e}")))?;
    gpu.free_tensor(d_y)
        .map_err(|e| Outcome::Fail(format!("free y failed: {e}")))?;

    if errors > 0 {
        Err(Outcome::Fail(format!(
            "{errors}/{m} rows exceeded tolerance, max_err={max_err:.6}"
        )))
    } else {
        Ok(format!("{}x{} max_err={max_err:.6}", m, k))
    }
}
