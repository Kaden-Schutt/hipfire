// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use rdna_compute::{DType, Gpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    let logits = gpu.upload_f32(&[10.0, -10.0, 0.0, 4.0, 5.0], &[5])?;
    let unique_tokens = gpu.alloc_tensor(&[4], DType::F32)?;
    let token_ids = [0_u32, 1, 2, 99];
    let token_bytes = unsafe {
        std::slice::from_raw_parts(
            token_ids.as_ptr().cast::<u8>(),
            token_ids.len() * std::mem::size_of::<u32>(),
        )
    };
    gpu.hip.memcpy_htod(&unique_tokens.buf, token_bytes)?;

    gpu.apply_hf_repetition_penalty_f32(&logits, &unique_tokens, token_ids.len(), 5, 1.25)?;
    let actual = gpu.download_f32(&logits)?;
    let expected = [8.0, -12.5, 0.0, 4.0, 5.0];
    assert_eq!(actual, expected);

    gpu.free_tensor(logits)?;
    gpu.free_tensor(unique_tokens)?;
    println!("PASS: GPU HF repetition penalty matches once-per-unique-token host semantics");
    Ok(())
}
