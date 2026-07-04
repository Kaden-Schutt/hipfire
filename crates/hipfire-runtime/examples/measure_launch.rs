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

//! Measure raw kernel launch overhead by launching a trivial kernel many times.
fn main() {
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");

    // Use add_inplace as a near-zero-work kernel (tiny 1-element tensor)
    let a = gpu.upload_f32(&[1.0], &[1]).unwrap();
    let b = gpu.upload_f32(&[0.0], &[1]).unwrap();

    let n_warmup = 100;
    let n_iter = 10000;

    for _ in 0..n_warmup {
        gpu.add_inplace_f32(&a, &b).unwrap();
    }

    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter {
        gpu.add_inplace_f32(&a, &b).unwrap();
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();

    let us_per_launch = ms * 1000.0 / n_iter as f32;
    eprintln!("Trivial kernel launch overhead: {us_per_launch:.2} us/launch");
    eprintln!(
        "For 286 launches/token: {:.1} us total = {:.2} ms",
        us_per_launch * 286.0,
        us_per_launch * 286.0 / 1000.0
    );
    eprintln!(
        "At 9.2ms/token, that's {:.1}% of forward time",
        us_per_launch * 286.0 / 9200.0 * 100.0
    );
}
