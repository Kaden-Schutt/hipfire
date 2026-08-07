// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1201 product-vocabulary greedy-selection screen.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const VOCAB: usize = 129_280;
const BLOCKS: usize = 9;
const REPEATS: usize = 64;

fn cpu_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &value)| {
            if value.is_finite() && value > best.1 {
                (i, value)
            } else {
                best
            }
        })
        .0 as u32
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");

    let mut host = (0..VOCAB)
        .map(|index| {
            let mixed = index.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) & 0x00ff_ffff;
            mixed as f32 * (1.0 / 16_777_216.0) - 0.5
        })
        .collect::<Vec<_>>();
    const EXPECTED: u32 = 91_337;
    host[EXPECTED as usize] = 8.0;
    let logits = gpu
        .upload_f32(&host, &[VOCAB])
        .expect("upload product logits");
    let persistent = gpu
        .zeros(&[1], DType::F32)
        .expect("persistent argmax result");

    let full_once = gpu.download_f32(&logits).expect("warm full download");
    assert_eq!(cpu_argmax(&full_once), EXPECTED);
    assert_eq!(
        gpu.argmax_f32(&logits, VOCAB).expect("warm allocating"),
        EXPECTED
    );
    gpu.argmax_f32_batched(&logits, &persistent, VOCAB, 1)
        .expect("warm persistent");
    let mut persistent_id = [0_i32; 1];
    let persistent_bytes =
        unsafe { std::slice::from_raw_parts_mut(persistent_id.as_mut_ptr().cast::<u8>(), 4) };
    gpu.hip
        .memcpy_dtoh(persistent_bytes, &persistent.buf)
        .expect("warm persistent download");
    assert_eq!(persistent_id[0] as u32, EXPECTED);

    let mut full_blocks = Vec::with_capacity(BLOCKS);
    let mut allocating_blocks = Vec::with_capacity(BLOCKS);
    let mut persistent_blocks = Vec::with_capacity(BLOCKS);

    for block in 0..BLOCKS {
        let order = if block & 1 == 0 { [0, 1, 2] } else { [2, 1, 0] };
        for path in order {
            let started = Instant::now();
            match path {
                0 => {
                    for _ in 0..REPEATS {
                        let values = gpu.download_f32(&logits).expect("full download");
                        assert_eq!(cpu_argmax(&values), EXPECTED);
                    }
                    full_blocks.push(started.elapsed().as_secs_f64() * 1.0e6 / REPEATS as f64);
                }
                1 => {
                    for _ in 0..REPEATS {
                        assert_eq!(
                            gpu.argmax_f32(&logits, VOCAB).expect("allocating argmax"),
                            EXPECTED
                        );
                    }
                    allocating_blocks
                        .push(started.elapsed().as_secs_f64() * 1.0e6 / REPEATS as f64);
                }
                2 => {
                    for _ in 0..REPEATS {
                        gpu.argmax_f32_batched(&logits, &persistent, VOCAB, 1)
                            .expect("persistent argmax");
                        gpu.hip
                            .memcpy_dtoh(persistent_bytes, &persistent.buf)
                            .expect("persistent result download");
                        assert_eq!(persistent_id[0] as u32, EXPECTED);
                    }
                    persistent_blocks
                        .push(started.elapsed().as_secs_f64() * 1.0e6 / REPEATS as f64);
                }
                _ => unreachable!(),
            }
        }
    }

    let full_us = median(&mut full_blocks);
    let allocating_us = median(&mut allocating_blocks);
    let persistent_us = median(&mut persistent_blocks);
    println!("gfx1201 vocab={VOCAB} blocks={BLOCKS} repeats={REPEATS} expected={EXPECTED}");
    println!("full_download_cpu_argmax_us={full_us:.3}");
    println!(
        "allocating_gpu_argmax_us={allocating_us:.3} saved_us={:.3}",
        full_us - allocating_us
    );
    println!(
        "persistent_gpu_argmax_us={persistent_us:.3} saved_us={:.3}",
        full_us - persistent_us
    );
}
