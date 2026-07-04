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

//! Test GEMV kernel on the GPU against CPU reference.

fn main() {
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init failed");
    println!("GPU initialized");

    let m = 1024;
    let k = 2048;

    // Create test data
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();

    // CPU reference
    let mut y_ref = vec![0.0f32; m];
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += a[i * k + j] * x[j];
        }
        y_ref[i] = sum;
    }

    // GPU compute
    let d_a = gpu.upload_f32(&a, &[m, k]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], hipfire_rdna::DType::F32).unwrap();

    println!("Compiling and launching GEMV kernel ({m}x{k})...");
    gpu.gemv_f32(&d_a, &d_x, &d_y).unwrap();

    let y_gpu = gpu.download_f32(&d_y).unwrap();

    // Verify
    let mut max_err: f32 = 0.0;
    let mut errors = 0;
    for i in 0..m {
        let err = (y_gpu[i] - y_ref[i]).abs();
        max_err = max_err.max(err);
        if err > 0.01 {
            errors += 1;
            if errors <= 5 {
                eprintln!(
                    "  row {i}: gpu={:.6} ref={:.6} err={:.6}",
                    y_gpu[i], y_ref[i], err
                );
            }
        }
    }

    gpu.free_tensor(d_a).unwrap();
    gpu.free_tensor(d_x).unwrap();
    gpu.free_tensor(d_y).unwrap();

    println!("Max error: {max_err:.8}");
    if errors == 0 {
        println!("GEMV test: PASS ({m}x{k}, max_err={max_err:.2e})");
    } else {
        eprintln!("GEMV test: FAIL ({errors}/{m} rows wrong)");
        std::process::exit(1);
    }
}
