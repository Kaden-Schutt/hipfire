// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone microbench + bit-exactness gate for the Q8_0 batched GEMM.
//!
//! `gemm_q8_0_batched` was measured at **24.6 GiB/s** and **68.5% of DeepSeek-V4
//! kernel time** on gfx942, against 549.9 GiB/s for the routed-expert GEMVs in
//! the same capture. This harness exists so CDNA3 replacements can be iterated
//! in SECONDS instead of 80-second model loads.
//!
//! Two things are measured, and the second is the one that can veto a rewrite:
//!   1. effective GiB/s over the Q8 weight bytes actually streamed
//!   2. **bit-exactness** between the two entry points
//!
//! Two arms are timed, and the second is the one that matters cross-arch:
//!   * `scalar`  — `gemm_q8_0_batched` directly. Always the wave32-shaped
//!     scalar kernel, on every architecture. A fixed reference point.
//!   * `prod`    — `gemm_q8_0_batched_chunked`, the entry point production
//!     actually calls (dspark_core.rs:496). This is ARCH-ROUTED: on any
//!     `has_wmma()` part with K%32==0 it returns `gemm_q8_0_wmma`
//!     (gemm.rs:19753); on CDNA3 `has_wmma()` is false so it falls through to
//!     the scalar kernel. Running this arm on gfx942 and gfx1151 and comparing
//!     the B-scaling curves is what tells us whether MI300X is a faithful
//!     prototyping rig for gfx1151 DSpark block-size tuning.
//!
//! NOTE the two arms are NOT expected to be bit-identical on a WMMA part:
//! `gemm_q8_0_wmma` is f16 WMMA and the tree's own parity test
//! (test_gemm_q8_0_wmma_parity.rs:23-25) passes at mean_rel<2e-3 /
//! max_rel<3.5e-2, explicitly not bitwise. On CDNA3 both arms are the same
//! kernel so they MUST be bit-identical — that is the harness self-check.
//!
//! Bit-exactness is not a nicety here. `gemm_q8_0_batched.hip:25-27` states its
//! FP reduction order deliberately matches `gemv_q8_0` to preserve greedy parity
//! for speculative decode. This kernel feeds the lm_head, so a changed float
//! reduction order shifts logits enough to flip argmax at near-ties — which
//! surfaces as altered greedy output and changed spec-decode acceptance. A
//! candidate that is fast but not bit-identical must be rejected or explicitly
//! justified.
//!
//! Usage:
//!   bench_q8_0_batched                 # sweep the DeepSeek-V4 shape set
//!   bench_q8_0_batched M K B [runs]    # one shape
//!
//! Shapes swept by default (derived from DeepSeek-V4 0731: hidden=4096,
//! vocab=129280; AR decode is B=1, DSpark draft/verify windows are B<=6):
//!   (4096,  4096, 1)   dense projection, AR decode
//!   (4096,  4096, 6)   dense projection, DSpark window
//!   (129280,4096, 1)   lm_head, AR decode
//!   (129280,4096, 5)   lm_head, DSpark block=5
//!   (2048,  4096, 1)   narrow projection

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const WARMUP: usize = 3;

fn wrap_buf(
    raw_ptr: *mut std::ffi::c_void,
    bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(raw_ptr, bytes) },
        shape,
        dtype,
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (b & 0x7f_ffff) >> 13;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        exp = 0x1f;
        return sign | ((exp as u16) << 10);
    }
    sign | ((exp as u16) << 10) | mant as u16
}

/// Q8_0: per 32-element K-group, [2 B f16 scale | 32 B int8]. 34 B/group.
/// Deterministic pseudo-random payload so runs are comparable across processes.
fn quantize_q8_0(k: usize, rows: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0, "Q8_0 requires K % 32 == 0");
    let groups_per_row = k / 32;
    let mut out = Vec::with_capacity(rows * groups_per_row * 34);
    let mut rng = seed;
    for _ in 0..rows {
        for _ in 0..groups_per_row {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Scale in a realistic range; exactly representable in f16.
            let scale = 0.00390625f32 * (1 + ((rng >> 33) & 7)) as f32;
            out.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
            for _ in 0..32 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                out.push(((rng >> 40) & 0xff) as u8);
            }
        }
    }
    out
}

struct Cell {
    m: usize,
    k: usize,
    b: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (cells, runs) = if args.len() >= 4 {
        (
            vec![Cell {
                m: args[1].parse().expect("M"),
                k: args[2].parse().expect("K"),
                b: args[3].parse().expect("B"),
            }],
            args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10usize),
        )
    } else {
        (
            vec![
                Cell { m: 4096, k: 4096, b: 1 },
                Cell { m: 4096, k: 4096, b: 6 },
                Cell { m: 129280, k: 4096, b: 1 },
                Cell { m: 129280, k: 4096, b: 5 },
                Cell { m: 2048, k: 4096, b: 1 },
            ],
            10usize,
        )
    };

    let mut gpu = Gpu::init().expect("gpu init");
    println!("GPU: {}", gpu.arch);
    println!(
        "is_gfx942={}  has_wmma={}  -> `prod` (gemm_q8_0_batched_chunked) routes to {}\n",
        gpu.arch_caps.is_gfx942(),
        gpu.arch_caps.has_wmma(),
        if gpu.arch_caps.has_wmma() { "gemm_q8_0_wmma (f16 WMMA)" } else { "the scalar kernel" }
    );

    println!(
        "{:<9} {:>7} {:>4} {:>11} {:>10} {:>11} {:>10} {:>8} {:>10}",
        "kernel", "M", "B", "w_MiB", "ms", "GiB/s", "speedup", "exact", "max_absdiff"
    );
    println!("{:-<100}", "");

    for c in &cells {
        let groups = c.k / 32;
        let w_bytes = quantize_q8_0(c.k, c.m, 0xD5_4E_1B_A7);
        let x_f32: Vec<f32> = (0..c.b * c.k)
            .map(|i| ((i as f32 * 0.017).sin()) * 0.5)
            .collect();
        let x_bytes: Vec<u8> = x_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

        let w_gpu = gpu.hip.malloc(w_bytes.len()).expect("malloc W");
        let x_gpu = gpu.hip.malloc(x_bytes.len()).expect("malloc X");
        let y_ref_gpu = gpu.hip.malloc(c.b * c.m * 4).expect("malloc Yref");
        let y_new_gpu = gpu.hip.malloc(c.b * c.m * 4).expect("malloc Ynew");
        gpu.hip.memcpy_htod(&w_gpu, &w_bytes).expect("htod W");
        gpu.hip.memcpy_htod(&x_gpu, &x_bytes).expect("htod X");

        let w_t = wrap_buf(w_gpu.as_ptr(), w_bytes.len(), vec![c.m, groups * 34], DType::Q8_0);
        let x_t = wrap_buf(x_gpu.as_ptr(), x_bytes.len(), vec![c.b, c.k], DType::F32);
        let y_ref_t = wrap_buf(y_ref_gpu.as_ptr(), c.b * c.m * 4, vec![c.b, c.m], DType::F32);
        let y_new_t = wrap_buf(y_new_gpu.as_ptr(), c.b * c.m * 4, vec![c.b, c.m], DType::F32);

        // Weight bytes are the streamed quantity; X is tiny and L2-resident.
        let stream_bytes = (c.m * groups * 34) as f64;
        let gib = |ms: f64| stream_bytes / (1024.0f64.powi(3)) / (ms / 1000.0);

        let mut time_kernel = |label: &str, newk: bool, gpu: &mut Gpu| -> Option<f64> {
            let call = |g: &mut Gpu, y: &GpuTensor| -> Result<(), String> {
                if newk {
                    g.gemm_q8_0_batched_chunked(&w_t, &x_t, y, c.m, c.k, c.b)
                        .map_err(|e| format!("{e:?}"))
                } else {
                    g.gemm_q8_0_batched(&w_t, &x_t, y, c.m, c.k, c.b)
                        .map_err(|e| format!("{e:?}"))
                }
            };
            let y = if newk { &y_new_t } else { &y_ref_t };
            for _ in 0..WARMUP {
                if let Err(e) = call(gpu, y) {
                    println!("{label:<9} {:>7} {:>4}   SKIPPED: {e}", c.m, c.b);
                    return None;
                }
            }
            let _ = gpu.hip.device_synchronize();
            let t = Instant::now();
            for _ in 0..runs {
                call(gpu, y).expect("launch");
            }
            let _ = gpu.hip.device_synchronize();
            Some(t.elapsed().as_secs_f64() * 1000.0 / runs as f64)
        };

        let base_ms = time_kernel("scalar", false, &mut gpu).expect("scalar must run");
        println!(
            "{:<9} {:>7} {:>4} {:>11.1} {:>10.3} {:>11.1} {:>10} {:>8} {:>10}",
            "scalar",
            c.m,
            c.b,
            stream_bytes / (1024.0 * 1024.0),
            base_ms,
            gib(base_ms),
            "1.00x",
            "-",
            "-"
        );

        if let Some(new_ms) = time_kernel("prod", true, &mut gpu) {
            // Bit-exactness gate.
            let mut ref_h = vec![0u8; c.b * c.m * 4];
            let mut new_h = vec![0u8; c.b * c.m * 4];
            gpu.hip.memcpy_dtoh(&mut ref_h, &y_ref_gpu).expect("dtoh ref");
            gpu.hip.memcpy_dtoh(&mut new_h, &y_new_gpu).expect("dtoh new");
            let exact = ref_h == new_h;
            let mut max_diff = 0.0f32;
            for i in (0..ref_h.len()).step_by(4) {
                let a = f32::from_le_bytes([ref_h[i], ref_h[i + 1], ref_h[i + 2], ref_h[i + 3]]);
                let b = f32::from_le_bytes([new_h[i], new_h[i + 1], new_h[i + 2], new_h[i + 3]]);
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            println!(
                "{:<9} {:>7} {:>4} {:>11.1} {:>10.3} {:>11.1} {:>9.2}x {:>8} {:>10.3e}",
                "prod",
                c.m,
                c.b,
                stream_bytes / (1024.0 * 1024.0),
                new_ms,
                gib(new_ms),
                base_ms / new_ms,
                if exact { "YES" } else { "NO" },
                max_diff
            );
            if !exact && !gpu.arch_caps.has_wmma() {
                println!(
                    "           ^^ NOT BIT-EXACT on a non-WMMA part, where both arms are the \
                     SAME kernel — harness bug or nondeterminism."
                );
            }
        }
        println!();
    }
}
