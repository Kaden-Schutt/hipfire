// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Direct HIP versus retained AQL oracle for the LFM2 conv+MQ-rotate fusion.

use rdna_compute::replay::{ReplayBackendRequest, ReplayController};
use rdna_compute::{DType, Gpu};

const CHANNELS: usize = 1024;
const KERNEL_SIZE: usize = 3;
const HIST: usize = KERNEL_SIZE - 1;

fn sample(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let centered = ((*seed >> 8) as i32 & 0xffff) - 0x8000;
    centered as f32 / 32768.0
}

fn bytes_of(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no invalid bit patterns and the slice lifetime is preserved.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn assert_bits_eq(label: &str, expected: &[f32], actual: &[f32]) {
    if let Some((index, (left, right))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (left, right))| left.to_bits() != right.to_bits())
    {
        panic!(
            "{label} mismatch at {index}: {:#010x} != {:#010x}",
            left.to_bits(),
            right.to_bits()
        );
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let mut seed = 0xc0_17_f05eu32;
    let bcx: Vec<f32> = (0..3 * CHANNELS).map(|_| sample(&mut seed)).collect();
    let initial_state: Vec<f32> = (0..CHANNELS * HIST).map(|_| sample(&mut seed)).collect();
    let weight: Vec<f32> = (0..CHANNELS * KERNEL_SIZE)
        .map(|_| sample(&mut seed))
        .collect();

    let bcx_gpu = gpu.upload_f32(&bcx, &[bcx.len()]).expect("upload bcx");
    let weight_gpu = gpu
        .upload_f32(&weight, &[weight.len()])
        .expect("upload weight");
    let direct_state = gpu
        .upload_f32(&initial_state, &[initial_state.len()])
        .expect("upload direct state");
    let replay_state = gpu
        .upload_f32(&initial_state, &[initial_state.len()])
        .expect("upload replay state");
    let direct_out = gpu
        .alloc_tensor(&[CHANNELS], DType::F32)
        .expect("alloc direct out");
    let replay_out = gpu
        .alloc_tensor(&[CHANNELS], DType::F32)
        .expect("alloc replay out");
    let direct_sink = gpu
        .alloc_tensor(&[CHANNELS], DType::F32)
        .expect("alloc direct sink");
    let replay_sink = gpu
        .alloc_tensor(&[CHANNELS], DType::F32)
        .expect("alloc replay sink");

    gpu.replay = ReplayController::new(ReplayBackendRequest::Hip);
    gpu.conv1d_gated_decode_mq_rotate_f32(
        &bcx_gpu,
        &direct_state,
        &weight_gpu,
        &direct_out,
        1,
        CHANNELS,
        KERNEL_SIZE,
    )
    .expect("direct fused conv");
    gpu.rotate_x_mq(&direct_out, &direct_sink, CHANNELS)
        .expect("direct output consumer");

    gpu.replay = ReplayController::new_armed(ReplayBackendRequest::Shadow);
    gpu.replay.begin_capture().expect("begin capture");
    gpu.conv1d_gated_decode_mq_rotate_f32(
        &bcx_gpu,
        &replay_state,
        &weight_gpu,
        &replay_out,
        1,
        CHANNELS,
        KERNEL_SIZE,
    )
    .expect("capture fused conv");
    gpu.rotate_x_mq(&replay_out, &replay_sink, CHANNELS)
        .expect("capture output consumer");
    let capture = gpu.replay.finish_capture().expect("finish capture");
    assert_eq!(capture.launch_count, 2);
    let launch = &gpu.replay.recorded_launches()[0];
    assert_eq!(launch.kernel, "conv1d_gated_decode_mq_rotate_f32");
    assert_eq!(launch.kernarg.len(), 64);
    assert_eq!(launch.grid, [4, 1, 1]);
    assert_eq!(launch.block, [32, 1, 1]);

    gpu.hip
        .memcpy_htod(&replay_state.buf, bytes_of(&initial_state))
        .expect("reset replay state");
    gpu.hip
        .memcpy_htod(&replay_out.buf, bytes_of(&vec![0.0; CHANNELS]))
        .expect("reset replay output");
    gpu.replay
        .prepare_linear_aql(0)
        .expect("prepare retained AQL");
    // SAFETY: every allocation referenced by the captured tape remains live.
    unsafe { gpu.replay.replay_linear_aql(0) }.expect("replay retained AQL");

    let expected_out = gpu.download_f32(&direct_out).expect("download direct out");
    let actual_out = gpu.download_f32(&replay_out).expect("download replay out");
    let expected_sink = gpu
        .download_f32(&direct_sink)
        .expect("download direct sink");
    let actual_sink = gpu
        .download_f32(&replay_sink)
        .expect("download replay sink");
    let expected_state = gpu
        .download_f32(&direct_state)
        .expect("download direct state");
    let actual_state = gpu
        .download_f32(&replay_state)
        .expect("download replay state");
    assert_bits_eq("out_rot", &expected_out, &actual_out);
    assert_bits_eq("consumer", &expected_sink, &actual_sink);
    assert_bits_eq("state", &expected_state, &actual_state);

    println!("kernel=conv1d_gated_decode_mq_rotate_f32");
    println!("dispatches=2 out_rot_bit_exact=true state_bit_exact=true");
    println!("CONV1D_GATED_DECODE_MQ_ROTATE_AQL_ORACLE_PASS");
}
