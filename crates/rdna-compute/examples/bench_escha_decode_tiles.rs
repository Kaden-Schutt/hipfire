// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Throughput benchmark for `escha_decode_tiles`, kernel-only (no host
//! round trip): upload the K=2 golden code once, then launch the
//! device-resident decode in a loop, syncing once at the end.
//!
//! Temporary tool for the Finding-3 before/after measurement in the Task 7
//! review (dynamic `words[]` array indexing vs. two direct global loads).
//! Not wired into any gate.
use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let (ic, oc, k) = (2048usize, 1024usize, 2usize);
    let path = format!(
        "{}/../hipfire-quantize/tests/data/escha/packed_gu_e0_k2.i16",
        env!("CARGO_MANIFEST_DIR")
    );
    let code_bytes = std::fs::read(&path).expect("run fetch-goldens.sh first");
    assert_eq!(code_bytes.len(), (ic / 16) * (oc / 16) * 16 * k * 2);

    let mut gpu = Gpu::init().expect("gpu");
    let d_code = gpu
        .upload_raw(&code_bytes, &[code_bytes.len() / 2])
        .expect("upload code");
    let d_bare = gpu
        .alloc_tensor(&[ic * oc], DType::F16)
        .expect("alloc bare");

    // Warm up: first call JIT-compiles the kernel (or loads it from the
    // on-disk cache), which must not be counted.
    for _ in 0..8 {
        gpu.escha_decode_tiles(&d_code, &d_bare, ic as u32, oc as u32, k as u32)
            .expect("warmup decode");
    }
    gpu.hip.device_synchronize().expect("sync after warmup");

    const REPS: u32 = 2000;
    let start = Instant::now();
    for _ in 0..REPS {
        gpu.escha_decode_tiles(&d_code, &d_bare, ic as u32, oc as u32, k as u32)
            .expect("decode");
    }
    gpu.hip.device_synchronize().expect("sync after loop");
    let elapsed = start.elapsed();

    let elems = (ic * oc) as f64;
    let per_launch = elapsed.as_secs_f64() / REPS as f64;
    // Bytes moved per launch: code read once (ic*oc/16 elements packed at
    // k*2 bytes per 16-element tile -> code bytes) + bare fp16 written once.
    let code_read_bytes = code_bytes.len() as f64;
    let bare_write_bytes = elems * 2.0;
    let bytes_per_launch = code_read_bytes + bare_write_bytes;

    println!("escha_decode_tiles K=2 {ic}x{oc}: {REPS} reps in {elapsed:?}");
    println!(
        "  {:.3} us/launch, {:.3} Gelem/s, {:.3} GB/s (code-read + bare-write)",
        per_launch * 1e6,
        elems / per_launch / 1e9,
        bytes_per_launch / per_launch / 1e9
    );
}
