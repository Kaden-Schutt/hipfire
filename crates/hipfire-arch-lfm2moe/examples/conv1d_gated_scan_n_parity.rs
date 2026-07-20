// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Bit-exact oracle for the gfx1201 LFM2 batched gated-convolution scan.

use hipfire_arch_lfm2moe::kernels::conv1d_gated_scan_n;
use rdna_compute::{DType, Gpu};

const CHANNELS: usize = 1024;
const LENGTHS: [usize; 8] = [1, 2, 3, 127, 128, 255, 256, 257];

fn sample(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let centered = ((*seed >> 8) as i32 & 0xffff) - 0x8000;
    centered as f32 / 32768.0
}

fn first_bit_mismatch(lhs: &[f32], rhs: &[f32]) -> Option<(usize, u32, u32)> {
    lhs.iter()
        .zip(rhs)
        .enumerate()
        .find_map(|(index, (&a, &b))| {
            (a.to_bits() != b.to_bits()).then_some((index, a.to_bits(), b.to_bits()))
        })
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "conv1d_gated_scan_n oracle requires gfx1201, got {}",
        gpu.arch
    );

    for n in LENGTHS {
        let mut seed = 0x51a7_0000u32 ^ n as u32;
        let bcx: Vec<f32> = (0..n * 3 * CHANNELS).map(|_| sample(&mut seed)).collect();
        let initial_state: Vec<f32> = (0..CHANNELS * 2).map(|_| sample(&mut seed)).collect();
        let weight: Vec<f32> = (0..CHANNELS * 3).map(|_| sample(&mut seed)).collect();

        let bcx_gpu = gpu.upload_f32(&bcx, &[bcx.len()]).expect("upload bcx");
        let weight_gpu = gpu
            .upload_f32(&weight, &[weight.len()])
            .expect("upload weight");
        let scan_state = gpu
            .upload_f32(&initial_state, &[initial_state.len()])
            .expect("upload scan state");
        let eager_state = gpu
            .upload_f32(&initial_state, &[initial_state.len()])
            .expect("upload eager state");
        let scan_out = gpu
            .alloc_tensor(&[n * CHANNELS], DType::F32)
            .expect("alloc scan out");
        let eager_out = gpu
            .alloc_tensor(&[n * CHANNELS], DType::F32)
            .expect("alloc eager out");

        if n == 257 {
            let bcx_first = bcx_gpu.sub_offset(0, 256 * 3 * CHANNELS);
            let out_first = scan_out.sub_offset(0, 256 * CHANNELS);
            conv1d_gated_scan_n(
                &mut gpu,
                &bcx_first,
                &scan_state,
                &weight_gpu,
                &out_first,
                256,
                CHANNELS,
            )
            .expect("scan first 256 rows");
            let bcx_last = bcx_gpu.sub_offset(256 * 3 * CHANNELS, 3 * CHANNELS);
            let out_last = scan_out.sub_offset(256 * CHANNELS, CHANNELS);
            conv1d_gated_scan_n(
                &mut gpu,
                &bcx_last,
                &scan_state,
                &weight_gpu,
                &out_last,
                1,
                CHANNELS,
            )
            .expect("scan final row");
        } else {
            conv1d_gated_scan_n(
                &mut gpu,
                &bcx_gpu,
                &scan_state,
                &weight_gpu,
                &scan_out,
                n,
                CHANNELS,
            )
            .expect("batched scan");
        }

        for row in 0..n {
            let bcx_row = bcx_gpu.sub_offset(row * 3 * CHANNELS, 3 * CHANNELS);
            let out_row = eager_out.sub_offset(row * CHANNELS, CHANNELS);
            gpu.conv1d_gated_decode_f32(
                &bcx_row,
                &eager_state,
                &weight_gpu,
                &out_row,
                1,
                CHANNELS,
                3,
            )
            .expect("sequential gated decode");
        }

        let scan_y = gpu.download_f32(&scan_out).expect("download scan out");
        let eager_y = gpu.download_f32(&eager_out).expect("download eager out");
        let scan_tail = gpu.download_f32(&scan_state).expect("download scan state");
        let eager_tail = gpu
            .download_f32(&eager_state)
            .expect("download eager state");

        if let Some((index, scan_bits, eager_bits)) = first_bit_mismatch(&scan_y, &eager_y) {
            panic!(
                "N={n} output mismatch at {index}: scan=0x{scan_bits:08x} eager=0x{eager_bits:08x}"
            );
        }
        if let Some((index, scan_bits, eager_bits)) = first_bit_mismatch(&scan_tail, &eager_tail) {
            panic!(
                "N={n} state mismatch at {index}: scan=0x{scan_bits:08x} eager=0x{eager_bits:08x}"
            );
        }
        println!("N={n} out_y_bit_identical=true final_state_bit_identical=true");

        for tensor in [
            bcx_gpu,
            weight_gpu,
            scan_state,
            eager_state,
            scan_out,
            eager_out,
        ] {
            gpu.free_tensor(tensor).expect("free oracle tensor");
        }
    }

    println!("CONV1D_GATED_SCAN_N_PARITY_PASS");
}
