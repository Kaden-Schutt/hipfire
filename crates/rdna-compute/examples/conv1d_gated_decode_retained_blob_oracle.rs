// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Direct-vs-blob oracle for LFM2 `conv1d_gated_decode_f32`.
//!
//! Proves the recorder-aware launch path is bit-identical to the default
//! `kernelParams` path for both `out_y` and the in-place rolling conv state,
//! and that the retained blob records the exact 48-byte padded ABI with fixed
//! geometry `[4,1,1]` / `[256,1,1]` / shared 0.
//!
//! Build/run:
//! ```text
//! flock /tmp/hipfire-gpu.lock cargo run -p rdna-compute --release \
//!   --example conv1d_gated_decode_retained_blob_oracle
//! ```

use hip_bridge::KernargBlob;
use rdna_compute::replay::{ReplayBackendRequest, ReplayController};
use rdna_compute::{DType, Gpu};

const BATCH: usize = 1;
const CHANNELS: usize = 1024;
const KERNEL_SIZE: usize = 3;
const HIST: usize = KERNEL_SIZE - 1;

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
    let mut gpu = Gpu::init().expect("Gpu::init");

    let mut seed = 0xc0_17_dec0u32;
    let bcx: Vec<f32> = (0..BATCH * 3 * CHANNELS)
        .map(|_| sample(&mut seed))
        .collect();
    let initial_state: Vec<f32> = (0..BATCH * CHANNELS * HIST)
        .map(|_| sample(&mut seed))
        .collect();
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
    let blob_state = gpu
        .upload_f32(&initial_state, &[initial_state.len()])
        .expect("upload blob state");
    let direct_out = gpu
        .alloc_tensor(&[BATCH * CHANNELS], DType::F32)
        .expect("alloc direct out");
    let blob_out = gpu
        .alloc_tensor(&[BATCH * CHANNELS], DType::F32)
        .expect("alloc blob out");

    // Default / raw kernelParams path (no recorder).
    gpu.replay = ReplayController::new(ReplayBackendRequest::Hip);
    gpu.conv1d_gated_decode_f32(
        &bcx_gpu,
        &direct_state,
        &weight_gpu,
        &direct_out,
        BATCH,
        CHANNELS,
        KERNEL_SIZE,
    )
    .expect("direct conv1d_gated_decode_f32");

    // Exact retained-blob path: arm recorder, launch once, inspect tape, and
    // execute the same padded blob through launch_kernel_blob.
    gpu.replay = ReplayController::new_armed(ReplayBackendRequest::Shadow);
    gpu.replay.begin_capture().expect("begin_capture");
    gpu.conv1d_gated_decode_f32(
        &bcx_gpu,
        &blob_state,
        &weight_gpu,
        &blob_out,
        BATCH,
        CHANNELS,
        KERNEL_SIZE,
    )
    .expect("recorded conv1d_gated_decode_f32");
    let summary = gpu.replay.finish_capture().expect("finish_capture");
    assert_eq!(
        summary.launch_count, 1,
        "expected exactly one recorded launch"
    );

    let (
        recorded_kernarg,
        recorded_bcx_ptr,
        recorded_state_ptr,
        recorded_weight_ptr,
        recorded_out_ptr,
    ) = {
        let launches = gpu.replay.recorded_launches();
        assert_eq!(launches.len(), 1);
        let launch = &launches[0];
        assert_eq!(launch.kernel, "conv1d_gated_decode_f32");
        assert_eq!(
            launch.kernarg.len(),
            48,
            "padded kernarg must be exactly 48 bytes"
        );
        assert_eq!(launch.grid, [4, 1, 1], "fixed grid for batch*channels=1024");
        assert_eq!(launch.block, [256, 1, 1]);
        assert_eq!(launch.shared_mem, 0);
        assert!(launch.grid_binding.is_none());
        (
            launch.kernarg.clone(),
            bcx_gpu.buf.as_ptr() as usize as u64,
            blob_state.buf.as_ptr() as usize as u64,
            weight_gpu.buf.as_ptr() as usize as u64,
            blob_out.buf.as_ptr() as usize as u64,
        )
    };

    // Re-run the exact recorded blob on fresh tensors and compare again.
    let replay_state = gpu
        .upload_f32(&initial_state, &[initial_state.len()])
        .expect("upload replay state");
    let replay_out = gpu
        .alloc_tensor(&[BATCH * CHANNELS], DType::F32)
        .expect("alloc replay out");
    // Rebuild a known-good blob with the same ABI and pointer values used by
    // the live tensors so the direct launch_kernel_blob path is exercised
    // independently of the already-executed recorded launch above.
    let mut blob = KernargBlob::new();
    blob.push_ptr(bcx_gpu.buf.as_ptr());
    blob.push_ptr(replay_state.buf.as_ptr());
    blob.push_ptr(weight_gpu.buf.as_ptr());
    blob.push_ptr(replay_out.buf.as_ptr());
    blob.push_i32(BATCH as i32);
    blob.push_i32(CHANNELS as i32);
    blob.push_i32(KERNEL_SIZE as i32);
    blob.pad_to(16);
    assert_eq!(blob.len(), 48, "hand-built blob must match padded ABI");
    let mut blob_bytes = blob.into_vec();
    gpu.launch_kernel_blob(
        "conv1d_gated_decode_f32",
        [4, 1, 1],
        [256, 1, 1],
        0,
        blob_bytes.as_mut_slice(),
    )
    .expect("launch_kernel_blob conv1d_gated_decode_f32");

    let direct_y = gpu.download_f32(&direct_out).expect("download direct out");
    let recorded_y = gpu.download_f32(&blob_out).expect("download recorded out");
    let replay_y = gpu.download_f32(&replay_out).expect("download replay out");
    let direct_tail = gpu
        .download_f32(&direct_state)
        .expect("download direct state");
    let recorded_tail = gpu
        .download_f32(&blob_state)
        .expect("download recorded state");
    let replay_tail = gpu
        .download_f32(&replay_state)
        .expect("download replay state");

    if let Some((index, a, b)) = first_bit_mismatch(&direct_y, &recorded_y) {
        panic!("direct vs recorded out_y mismatch at {index}: {a:#x} vs {b:#x}");
    }
    if let Some((index, a, b)) = first_bit_mismatch(&direct_tail, &recorded_tail) {
        panic!("direct vs recorded state mismatch at {index}: {a:#x} vs {b:#x}");
    }
    if let Some((index, a, b)) = first_bit_mismatch(&direct_y, &replay_y) {
        panic!("direct vs blob-path out_y mismatch at {index}: {a:#x} vs {b:#x}");
    }
    if let Some((index, a, b)) = first_bit_mismatch(&direct_tail, &replay_tail) {
        panic!("direct vs blob-path state mismatch at {index}: {a:#x} vs {b:#x}");
    }

    // Sanity: recorded kernarg pointer slots match the live allocation bases
    // used by the recording launch (bcx/state/weight/out at 0/8/16/24).
    let read_ptr = |offset: usize| -> u64 {
        let bytes: [u8; 8] = recorded_kernarg[offset..offset + 8]
            .try_into()
            .expect("pointer slot");
        u64::from_ne_bytes(bytes)
    };
    assert_eq!(read_ptr(0), recorded_bcx_ptr);
    assert_eq!(read_ptr(8), recorded_state_ptr);
    assert_eq!(read_ptr(16), recorded_weight_ptr);
    assert_eq!(read_ptr(24), recorded_out_ptr);
    let read_i32 = |offset: usize| -> i32 {
        let bytes: [u8; 4] = recorded_kernarg[offset..offset + 4]
            .try_into()
            .expect("i32 slot");
        i32::from_ne_bytes(bytes)
    };
    assert_eq!(read_i32(32), BATCH as i32);
    assert_eq!(read_i32(36), CHANNELS as i32);
    assert_eq!(read_i32(40), KERNEL_SIZE as i32);
    assert!(
        recorded_kernarg[44..48].iter().all(|&byte| byte == 0),
        "tail padding must be zero"
    );

    for tensor in [
        bcx_gpu,
        weight_gpu,
        direct_state,
        blob_state,
        direct_out,
        blob_out,
        replay_state,
        replay_out,
    ] {
        gpu.free_tensor(tensor).expect("free oracle tensor");
    }

    println!("kernel=conv1d_gated_decode_f32");
    println!("kernarg_bytes=48 grid=[4,1,1] block=[256,1,1] shared=0");
    println!("out_y_bit_identical=true final_state_bit_identical=true");
    println!("CONV1D_GATED_DECODE_RETAINED_BLOB_ORACLE_PASS");
}
