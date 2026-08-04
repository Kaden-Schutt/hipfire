// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end check for the pager's ASYNC prefetch path.
//!
//! The failure mode this guards against is silent: `hipMemcpyAsync` returns
//! immediately, so if the event wait is wrong (or the staging is reused while a
//! copy is still reading it) the destination holds torn or stale bytes and
//! nothing errors. That would surface as degraded model output, not a crash —
//! exactly the class of bug expert paging must not introduce.
//!
//! So this writes a known pattern to a file, prefetches disjoint ranges into a
//! device buffer through the pinned/async path, waits on the event, and asserts
//! byte-exact equality. It also re-issues into the same staging to exercise the
//! drain-before-reuse path.

use std::io::Write;

use hipfire_runtime::weight_pager::{FetchReq, PreadH2DTransport};
use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = match Gpu::init_with_device(0) {
        Ok(g) => g,
        Err(e) => {
            println!("async_prefetch_smoke: no GPU ({e:?}) — SKIP");
            return;
        }
    };

    // Known pattern on disk.
    const N: usize = 4 << 20; // 4 MiB, ~one expert role-blob
    let dir = std::env::temp_dir().join("hipfire_async_prefetch_smoke");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("pattern.bin");
    let src: Vec<u8> = (0..N * 3).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    {
        let mut f = std::fs::File::create(&path).expect("create pattern");
        f.write_all(&src).expect("write pattern");
        f.sync_all().ok();
    }

    let mut t = PreadH2DTransport::open(&path).expect("open transport");
    let dst = gpu
        .alloc_tensor(&[N * 3], DType::Raw)
        .expect("alloc device buffer");

    // Three disjoint ranges, deliberately out of file order so a naive
    // sequential implementation would land them in the wrong slots.
    let reqs = vec![
        FetchReq { hfq_offset: 2 * N, len: N, dst_byte_offset: 0 },
        FetchReq { hfq_offset: 0, len: N, dst_byte_offset: N },
        FetchReq { hfq_offset: N, len: N, dst_byte_offset: 2 * N },
    ];

    t.prefetch_batch_into(&reqs, &dst, &mut gpu)
        .expect("prefetch issue");
    println!("async_prefetch_smoke: issued {} ranges ({} MiB)", reqs.len(), N * 3 >> 20);
    t.wait_prefetch(&gpu).expect("prefetch wait");

    let mut got = vec![0u8; N * 3];
    gpu.hip
        .memcpy_dtoh(&mut got, &dst.buf)
        .expect("readback");

    let mut expect = vec![0u8; N * 3];
    expect[0..N].copy_from_slice(&src[2 * N..3 * N]);
    expect[N..2 * N].copy_from_slice(&src[0..N]);
    expect[2 * N..3 * N].copy_from_slice(&src[N..2 * N]);
    assert_eq!(got, expect, "async prefetch landed wrong bytes");
    println!("async_prefetch_smoke: {} MiB byte-exact after event wait", N * 3 >> 20);

    // Re-issue: exercises drain-before-reuse of the pinned staging. If the
    // drain is missing, this second copy races the first and corrupts.
    let reqs2 = vec![FetchReq { hfq_offset: 0, len: N, dst_byte_offset: 0 }];
    t.prefetch_batch_into(&reqs2, &dst, &mut gpu).expect("reissue");
    t.wait_prefetch(&gpu).expect("reissue wait");
    let mut got2 = vec![0u8; N];
    gpu.hip
        .memcpy_dtoh(&mut got2, &dst.buf)
        .expect("readback 2");
    assert_eq!(got2, src[0..N], "staging reuse corrupted the transfer");
    println!("async_prefetch_smoke: staging reuse clean");

    println!("  prefetch blocked for {:.2} ms total", t.prefetch_wait_ns() as f64 / 1e6);
    let _ = std::fs::remove_file(&path);
    println!("\nasync_prefetch_smoke: PASS");
}
