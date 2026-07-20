// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU argmax tie-break correctness: equal maxima must select the lowest index.
//!
//! The cross-lane tie at indices 1 and 256 catches reductions that use only
//! strict value comparison: lane 0 carries index 256 while lane 1 carries index
//! 1, so a value-only tree incorrectly keeps 256.

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 512;
    let mut gpu = Gpu::init().expect("Gpu::init failed");

    let mut row = vec![-10.0f32; N];
    row[1] = 7.0;
    row[256] = 7.0;
    let data = gpu.upload_f32(&row, &[N])?;
    let single = gpu.argmax_f32(&data, N)?;
    assert_eq!(
        single, 1,
        "single-row GPU argmax must choose lowest tied index"
    );

    let mut rows = row.clone();
    rows.extend_from_slice(&row);
    let batched_data = gpu.upload_f32(&rows, &[2, N])?;
    let batched_result = gpu.zeros(&[2], DType::F32)?;
    gpu.argmax_f32_batched(&batched_data, &batched_result, N, 2)?;
    let mut result = [0i32; 2];
    let result_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            result.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(&result),
        )
    };
    gpu.hip.memcpy_dtoh(result_bytes, &batched_result.buf)?;
    assert_eq!(
        result,
        [1, 1],
        "batched GPU argmax must choose lowest tied index"
    );

    gpu.free_tensor(data)?;
    gpu.free_tensor(batched_data)?;
    gpu.free_tensor(batched_result)?;
    println!("PASS: GPU argmax chooses the lowest vocabulary index on ties");
    Ok(())
}
