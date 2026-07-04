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

//! Correctness check for LFM2 `conv1d_gated_seq_f32`.
//!
//! Compares the one-launch sequence scan against both a CPU reference and
//! repeated `conv1d_gated_decode_f32` calls. This is the batched-conv primitive
//! needed before LFM2 prompt prefill can stop replaying decode token by token.

use hipfire_rdna::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    for &(seq, channels, k) in &[(1usize, 257usize, 3usize), (7, 257, 3), (17, 513, 5)] {
        eprintln!("\n=== seq={seq} channels={channels} K={k} ===");
        let bcx: Vec<f32> = (0..seq * 3 * channels)
            .map(|i| (((i * 104_729 + 17) % 251) as f32 - 125.0) * 0.004)
            .collect();
        let weight: Vec<f32> = (0..channels * k)
            .map(|i| (((i * 65_537 + 31) % 197) as f32 - 98.0) * 0.006)
            .collect();
        let state0: Vec<f32> = (0..channels * (k - 1))
            .map(|i| (((i * 8191 + 7) % 149) as f32 - 74.0) * 0.005)
            .collect();

        let (cpu_y, cpu_state) = cpu_ref(&bcx, &weight, &state0, seq, channels, k);

        let d_bcx = gpu.upload_f32(&bcx, &[seq, 3 * channels]).unwrap();
        let d_weight = gpu.upload_f32(&weight, &[channels, k]).unwrap();
        let d_seq_state = gpu.upload_f32(&state0, &[channels, k - 1]).unwrap();
        let d_dec_state = gpu.upload_f32(&state0, &[channels, k - 1]).unwrap();
        let d_seq_y = gpu.zeros(&[seq, channels], DType::F32).unwrap();
        let d_dec_y = gpu.zeros(&[seq, channels], DType::F32).unwrap();
        let d_bcx_one = gpu.zeros(&[3 * channels], DType::F32).unwrap();
        let d_y_one = gpu.zeros(&[channels], DType::F32).unwrap();

        gpu.conv1d_gated_seq_f32(&d_bcx, &d_seq_state, &d_weight, &d_seq_y, seq, channels, k)
            .unwrap();

        for t in 0..seq {
            gpu.hip
                .memcpy_dtod_at(
                    &d_bcx_one.buf,
                    0,
                    &d_bcx.buf,
                    t * 3 * channels * 4,
                    3 * channels * 4,
                )
                .unwrap();
            gpu.conv1d_gated_decode_f32(
                &d_bcx_one,
                &d_dec_state,
                &d_weight,
                &d_y_one,
                1,
                channels,
                k,
            )
            .unwrap();
            gpu.hip
                .memcpy_dtod_at(
                    &d_dec_y.buf,
                    t * channels * 4,
                    &d_y_one.buf,
                    0,
                    channels * 4,
                )
                .unwrap();
        }

        let seq_y = gpu.download_f32(&d_seq_y).unwrap();
        let seq_state = gpu.download_f32(&d_seq_state).unwrap();
        let dec_y = gpu.download_f32(&d_dec_y).unwrap();
        let dec_state = gpu.download_f32(&d_dec_state).unwrap();

        check_close("seq_vs_cpu_y", &seq_y, &cpu_y, 1.0e-5);
        check_close("seq_vs_cpu_state", &seq_state, &cpu_state, 0.0);
        check_close("seq_vs_decode_y", &seq_y, &dec_y, 1.0e-5);
        check_close("seq_vs_decode_state", &seq_state, &dec_state, 0.0);

        for t in [
            d_bcx,
            d_weight,
            d_seq_state,
            d_dec_state,
            d_seq_y,
            d_dec_y,
            d_bcx_one,
            d_y_one,
        ] {
            gpu.free_tensor(t).unwrap();
        }
    }

    println!("PASS: conv1d_gated_seq_f32 matches CPU and repeated decode");
}

fn cpu_ref(
    bcx: &[f32],
    weight: &[f32],
    state0: &[f32],
    seq: usize,
    channels: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>) {
    let hist = k - 1;
    let mut state = state0.to_vec();
    let mut y = vec![0.0f32; seq * channels];
    for s in 0..seq {
        let row = &bcx[s * 3 * channels..(s + 1) * 3 * channels];
        for c in 0..channels {
            let b_gate = row[c];
            let c_gate = row[channels + c];
            let x_in = row[2 * channels + c];
            let bx = b_gate * x_in;
            let sbase = c * hist;
            let mut acc = 0.0f32;
            for t in 0..hist {
                acc += state[sbase + t] * weight[c * k + t];
            }
            acc += bx * weight[c * k + hist];
            y[s * channels + c] = c_gate * acc;

            for t in 0..hist.saturating_sub(1) {
                state[sbase + t] = state[sbase + t + 1];
            }
            if hist > 0 {
                state[sbase + hist - 1] = bx;
            }
        }
    }
    (y, state)
}

fn check_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{label} length mismatch");
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    eprintln!("{label}: max|delta|={worst:.3e} at {worst_i}");
    if worst > tol {
        eprintln!(
            "FAIL {label}: got={} want={} delta={}",
            got[worst_i], want[worst_i], worst
        );
        std::process::exit(1);
    }
}
