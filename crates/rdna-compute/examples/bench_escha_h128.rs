// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Throughput benchmark for `escha_h128_in` / `escha_h128_out`, kernel-only
//! (no host round trip): upload fixed input buffers once, then launch the
//! device-resident `Gpu::escha_h128` in a loop, syncing once at the end.
//!
//! These kernels run on every token, on both sides of every escha matmul —
//! unlike `escha_decode_tiles` (once per expert at load time), this is the
//! hot decode path. Used for the Task 8 naive-vs-parallel-butterfly
//! before/after measurement. Not wired into any gate.
//!
//! Task 8 review fix (finding 3): a single n=2048 shape does not establish
//! whether the parallel-butterfly speedup survives at production widths, and
//! a raw us/launch number conflates real kernel work with fixed launch
//! overhead. This version:
//!   - launches a genuinely empty kernel (`escha_h128_noop`, same grid/block
//!     config as the real launch, zero kernarg bytes) at each shape to
//!     measure the launch-overhead floor, and reports both the raw and the
//!     overhead-subtracted per-launch time;
//!   - sweeps n = 2048, 6144, 17408 (16, 48, 136 blocks of 128) — production
//!     escha hidden/intermediate widths, not just the smallest one.
use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;
use std::time::Instant;

/// A completely empty kernel launched with the same grid/block config as
/// `escha_h128_in`/`escha_h128_out`, used only to measure the fixed
/// per-launch overhead (queue submission, dispatch packet, sync) that has
/// nothing to do with the H128 butterfly itself. Takes no arguments, so its
/// kernarg blob is zero bytes — it does not touch any memory.
const NOOP_SRC: &str = r#"
extern "C" __global__ void escha_h128_noop() {}
"#;

/// The pre-parallelisation kernel (commit `2a73edd46`, before
/// `61e3ab8bb`'s parallel butterfly), reproduced verbatim here under
/// renamed entry points (`_naive` suffix) so this benchmark can measure the
/// naive-vs-parallel speedup at production widths without touching the
/// shipped `escha_h128.hip` (which now only carries the parallel version —
/// there is no live "before" to compare against otherwise). This is a
/// benchmark-only artifact, not part of the shipped kernel; it must never be
/// registered in `kernels.rs`/`dispatch.rs`.
const NAIVE_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define ESCHA_RS 0.0883883476f

__device__ __forceinline__ __half f2h_rne_naive(float v) {
    if (v == 0.0f) {
        unsigned int bits = __float_as_uint(v);
        return __ushort_as_half((unsigned short)(bits >> 16));
    }
    return __float2half(v);
}

__device__ __forceinline__ void h128_block_naive(float* v) {
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
}

extern "C" __global__ void escha_h128_in_naive(
    const float* __restrict__ x, const float* __restrict__ rin,
    __half* __restrict__ xh, int n) {
    __shared__ float s[128];
    int g = blockIdx.x, t = threadIdx.x;
    int idx = g * 128 + t;
    if (idx >= n) return;
    s[t] = x[idx] * rin[idx];
    __syncthreads();
    if (t == 0) h128_block_naive(s);
    __syncthreads();
    xh[idx] = f2h_rne_naive(s[t] * ESCHA_RS);
}

extern "C" __global__ void escha_h128_out_naive(
    const float* __restrict__ mid, const float* __restrict__ rout,
    __half* __restrict__ y, int n) {
    __shared__ float s[128];
    int g = blockIdx.x, t = threadIdx.x;
    int idx = g * 128 + t;
    if (idx >= n) return;
    s[t] = mid[idx];
    __syncthreads();
    if (t == 0) h128_block_naive(s);
    __syncthreads();
    y[idx] = f2h_rne_naive(s[t] * ESCHA_RS * rout[idx]);
}
"#;

const REPS: u32 = 20_000;

/// Time `REPS` back-to-back launches of `f`, returning us/launch. `f` must
/// not itself synchronize; the caller syncs once after the loop.
fn time_launches(gpu: &mut Gpu, mut f: impl FnMut(&mut Gpu)) -> f64 {
    // Warm up: first call JIT-compiles (or loads from the on-disk cache),
    // which must not be counted.
    for _ in 0..8 {
        f(gpu);
    }
    gpu.hip.device_synchronize().expect("sync after warmup");

    let start = Instant::now();
    for _ in 0..REPS {
        f(gpu);
    }
    gpu.hip.device_synchronize().expect("sync after loop");
    start.elapsed().as_secs_f64() / REPS as f64
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    gpu.ensure_kernel_public("escha_h128_noop", NOOP_SRC, "escha_h128_noop")
        .expect("jit noop");
    gpu.ensure_kernel_public("escha_h128_naive", NAIVE_SRC, "escha_h128_in_naive")
        .expect("jit naive in");
    gpu.ensure_kernel_public("escha_h128_naive", NAIVE_SRC, "escha_h128_out_naive")
        .expect("jit naive out");

    println!(
        "{:<18} {:>10} {:>16} {:>22} {:>10}",
        "shape", "kernel", "us/launch", "overhead-sub us", "GB/s"
    );

    // Production escha hidden/intermediate widths (all multiples of 128):
    // 2048 matches the packed_gu_e0_k2 golden fixture's `ic`; 6144 and 17408
    // are the wider matmul dimensions in the real model, included so the
    // occupancy trend (16 -> 48 -> 136 blocks) is visible, not just the
    // smallest case.
    for n in [2048usize, 6144, 17408] {
        let blocks = (n / 128) as u32;

        let x: Vec<f32> = (0..n).map(|i| ((i * 37) as f32 * 0.017).sin()).collect();
        let rin: Vec<f32> = (0..n)
            .map(|i| if i % 3 == 0 { -0.0023 } else { 0.0023 })
            .collect();
        let mut rout: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
        // Pruned channels, same convention as the G3 gate fixture. Both
        // indices are < 2048, the smallest shape swept, so they exist at
        // every n benchmarked here.
        rout[7] = 0.0;
        rout[1000] = 0.0;

        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rin_bytes: Vec<u8> = rin.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rout_bytes: Vec<u8> = rout.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_x = gpu.upload_raw(&x_bytes, &[n]).expect("upload x");
        let d_rin = gpu.upload_raw(&rin_bytes, &[n]).expect("upload rin");
        let d_rout = gpu.upload_raw(&rout_bytes, &[n]).expect("upload rout");
        let d_out_in = gpu.alloc_tensor(&[n], DType::F16).expect("alloc out_in");
        let d_out_out = gpu.alloc_tensor(&[n], DType::F16).expect("alloc out_out");

        // Launch-overhead floor for this grid/block config: an empty kernel,
        // same [blocks, 1, 1] x [128, 1, 1] launch shape, zero kernarg bytes.
        let overhead_us = time_launches(&mut gpu, |gpu| {
            let mut kb = KernargBlob::new();
            gpu.launch_kernel_blob(
                "escha_h128_noop",
                [blocks, 1, 1],
                [128, 1, 1],
                0,
                kb.as_mut_slice(),
            )
            .expect("noop launch");
        }) * 1e6;

        let per_launch_in = time_launches(&mut gpu, |gpu| {
            gpu.escha_h128("escha_h128_in", &d_x, &d_rin, &d_out_in)
                .expect("h128 in");
        }) * 1e6;

        let per_launch_out = time_launches(&mut gpu, |gpu| {
            gpu.escha_h128("escha_h128_out", &d_x, &d_rout, &d_out_out)
                .expect("h128 out");
        }) * 1e6;

        // Naive (pre-parallelisation) kernel at the same shape, launched
        // directly via the raw kernarg-blob path since `Gpu::escha_h128`
        // always compiles against the shipped `ESCHA_H128_SRC`.
        let per_launch_in_naive = time_launches(&mut gpu, |gpu| {
            let mut kb = KernargBlob::new();
            kb.push_ptr(d_x.buf.as_ptr() as *const c_void);
            kb.push_ptr(d_rin.buf.as_ptr() as *const c_void);
            kb.push_ptr(d_out_in.buf.as_ptr() as *const c_void);
            kb.push_i32(n as i32);
            gpu.launch_kernel_blob(
                "escha_h128_in_naive",
                [blocks, 1, 1],
                [128, 1, 1],
                0,
                kb.as_mut_slice(),
            )
            .expect("naive in launch");
        }) * 1e6;

        let per_launch_out_naive = time_launches(&mut gpu, |gpu| {
            let mut kb = KernargBlob::new();
            kb.push_ptr(d_x.buf.as_ptr() as *const c_void);
            kb.push_ptr(d_rout.buf.as_ptr() as *const c_void);
            kb.push_ptr(d_out_out.buf.as_ptr() as *const c_void);
            kb.push_i32(n as i32);
            gpu.launch_kernel_blob(
                "escha_h128_out_naive",
                [blocks, 1, 1],
                [128, 1, 1],
                0,
                kb.as_mut_slice(),
            )
            .expect("naive out launch");
        }) * 1e6;

        // Bytes moved per launch: two f32 reads (a, vec_in) + one f16 write.
        let bytes_per_launch = n as f64 * (4.0 + 4.0 + 2.0);
        let shape = format!("n={n} ({blocks} blk)");

        for (name, raw_us) in [
            ("escha_h128_in", per_launch_in),
            ("escha_h128_out", per_launch_out),
        ] {
            let sub_us = (raw_us - overhead_us).max(0.0);
            // GB/s computed against the overhead-subtracted time: the launch
            // overhead moves no bytes, so folding it into the denominator
            // would understate the kernel's own bandwidth at small n, where
            // overhead is a proportionally larger share of the raw time.
            let gbs = if sub_us > 0.0 {
                bytes_per_launch / (sub_us * 1e-6) / 1e9
            } else {
                f64::NAN
            };
            println!("{shape:<18} {name:>10} {raw_us:>16.3} {sub_us:>22.3} {gbs:>10.3}");
        }
        println!(
            "{shape:<18} {:>10} {overhead_us:>16.3} {:>22} {:>10}",
            "noop", "-", "-"
        );

        for (name, raw_us) in [
            ("in_naive", per_launch_in_naive),
            ("out_naive", per_launch_out_naive),
        ] {
            let sub_us = (raw_us - overhead_us).max(0.0);
            let gbs = if sub_us > 0.0 {
                bytes_per_launch / (sub_us * 1e-6) / 1e9
            } else {
                f64::NAN
            };
            println!("{shape:<18} {name:>10} {raw_us:>16.3} {sub_us:>22.3} {gbs:>10.3}");
        }

        let speedup_in_raw = per_launch_in_naive / per_launch_in;
        let speedup_out_raw = per_launch_out_naive / per_launch_out;
        let speedup_in_sub =
            (per_launch_in_naive - overhead_us).max(0.0) / (per_launch_in - overhead_us).max(1e-9);
        let speedup_out_sub = (per_launch_out_naive - overhead_us).max(0.0)
            / (per_launch_out - overhead_us).max(1e-9);
        println!(
            "{shape:<18} speedup(in)  raw={speedup_in_raw:.3}x  overhead-sub={speedup_in_sub:.3}x"
        );
        println!(
            "{shape:<18} speedup(out) raw={speedup_out_raw:.3}x  overhead-sub={speedup_out_sub:.3}x"
        );
    }
}
