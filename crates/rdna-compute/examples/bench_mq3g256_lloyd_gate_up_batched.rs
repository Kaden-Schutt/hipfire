// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! MQ3-Lloyd batched MoE gate_up GEMV — correctness + batch-scaling triage.
//!
//! Covers `gemv_mq3g256_lloyd_moe_gate_up_indexed_batched_k4.hip`
//! (entry `gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched_k4`) against
//! `gemv_mq3g256_lloyd_moe_gate_up_indexed.hip`
//! (entry `gemv_mq3g256_lloyd_moe_gate_up_k8_indexed`) on the a3b routed-expert
//! gate_up decode shape M=1024, K=2048, k_top=8, over N in {1, 4, 16, 64} tokens.
//!
//! Naming traps this bench deliberately pins down:
//!   * `_k4` is a 4-accumulator ILP unroll over 4 consecutive 256-groups. It is
//!     NOT k_top=4 — K_TOP is a RUNTIME kernel argument (grid.y). Section B runs
//!     K_TOP in {1, 3, 8} through the same compiled kernel to prove it.
//!   * The trailing `k8` in both entry names is legacy naming from the original
//!     k_top=8 shape, not a compile-time constant.
//!
//! Both kernels are JIT'd from source and launched via the kernarg-blob path
//! (`ensure_kernel_public` + `launch_kernel_blob`), so this bench touches NO
//! dispatch plumbing, no kernel source, and no shared file.
//!
//! ── REGIME CAVEAT (read before quoting any number from this file) ──────────
//! This is a HIP-dispatch microbenchmark. It is a TRIAGE FILTER for gross
//! defects — wrong numbers, pathological slowness — NOT a verdict on a kernel.
//! Host launch latency masks device-level effects, and here it masks them
//! ASYMMETRICALLY: the sequential arm pays N host launches per call and the
//! batched arm pays one, so part of any measured win is launch bookkeeping
//! rather than kernel quality. The same kernel can measure differently once
//! lowered to retained PM4 replay. A prior HIP microbench measured the MQ2GL
//! kernel at -3.1% and that was nearly used to kill the format. Final
//! acceptance is the golden bundle (`registry/redline-golden-v1.json`) with
//! HIP and PM4 arms.
//!
//! ── Correctness is the point ──────────────────────────────────────────────
//! The CPU reference is derived from the MQ3-Lloyd (qt=20) FORMAT SPEC, not
//! from the kernel's arithmetic structure:
//!   112 B per group-of-256 = 8 x fp16 codebook (16 B) + 96 B of 3-bit indices.
//! The reference extracts index `w` (0..256) by reading the 3 bits at LSB-first
//! bit offset `3*w` inside the 96-byte region, ONE BIT AT A TIME. The kernel
//! instead does a per-thread 3-byte little-endian load and shifts by `3*i`.
//! Those two only agree if the packing really is contiguous LSB-first 3-bit —
//! a shared misunderstanding of the chunking cannot cancel out.
//!
//! X convention: these kernels expect x FWHT-256 pre-rotated by the caller.
//! This microbench feeds arbitrary synthetic x and uses the SAME bytes for the
//! GPU and the CPU reference, so the rotation is a no-op for correctness here.
//!
//! Tolerance: 1e-4 relative. The GPU accumulates in f32 with a 4-accumulator
//! group-interleave plus a wave32 shuffle reduction; the reference accumulates
//! sequentially in f64. Relative error uses the denominator
//! `max(|want|, 1e-3 * peak|want|)` so an entry that happens to cancel to near
//! zero is judged against the peak output magnitude instead of exploding.
//!
//! Run: cargo run --release -p rdna-compute --example bench_mq3g256_lloyd_gate_up_batched

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor, LLOYD_MQ3_GROUP_BYTES};
use std::time::Instant;

const BATCHED_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_gate_up_indexed_batched_k4.hip");
const SINGLE_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_gate_up_indexed.hip");

const BATCHED_MOD: &str = "bench_mq3l_gate_up_batched_k4";
const SINGLE_MOD: &str = "bench_mq3l_gate_up_single";
const BATCHED_FN: &str = "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched_k4";
const SINGLE_FN: &str = "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed";

// ── MQ3-Lloyd (qt=20) group geometry, straight from the format spec ──
const GROUP_WEIGHTS: usize = 256;
const CB_ENTRIES: usize = 8;
const CB_BYTES: usize = CB_ENTRIES * 2; // 8 x fp16 = 16 B
const IDX_BITS: usize = 3;
const IDX_BYTES: usize = GROUP_WEIGHTS * IDX_BITS / 8; // 96 B
const GROUP_BYTES: usize = CB_BYTES + IDX_BYTES; // 112 B

/// Textbook 8-level Lloyd–Max levels for a unit Gaussian, sorted ascending
/// (the format stores per-block codebooks sorted ascending).
const CB8: [f32; CB_ENTRIES] = [
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
];

const TOL_REL: f64 = 1e-4;

// ───────────────────────── scalar helpers ─────────────────────────

fn mix(x: u64) -> u64 {
    let h = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

fn half_bits(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        e = 30;
    }
    sign | ((e as u16) << 10) | ((mant >> 13) as u16)
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
}

/// Spec-derived bit reader: pull `n` bits starting at LSB-first bit offset
/// `bit` from a packed region. Deliberately per-bit so it shares no structure
/// with the kernel's 3-byte-load + shift decode.
fn read_bits_lsb(region: &[u8], bit: usize, n: usize) -> usize {
    let mut v = 0usize;
    for i in 0..n {
        let b = bit + i;
        let one = (region[b >> 3] >> (b & 7)) & 1;
        v |= (one as usize) << i;
    }
    v
}

// ───────────────────────── synthetic weights ─────────────────────────

/// Build one expert's MQ3-Lloyd weight blob in on-disk byte layout: row-major
/// `[M][K/256]` groups of 112 B = `[0..16) 8 x fp16 codebook`
/// + `[16..112) 96 B of packed 3-bit indices`.
fn build_mq3l(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / GROUP_WEIGHTS;
    let mut out = vec![0u8; m * gpr * GROUP_BYTES];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * GROUP_BYTES;
            // per-group scale ~0.004..0.008 (same spirit as the MQ2 bench)
            let s =
                0.004f32 + ((mix(seed ^ ((row as u64) << 20) ^ g as u64) % 4000) as f32) * 1e-6;
            for (e, &c) in CB8.iter().enumerate() {
                let h = half_bits(c * s);
                out[off + 2 * e] = (h & 0xff) as u8;
                out[off + 2 * e + 1] = (h >> 8) as u8;
            }
            for b in 0..IDX_BYTES {
                out[off + CB_BYTES + b] =
                    (mix(seed ^ ((row as u64) << 32) ^ ((g as u64) << 12) ^ b as u64) & 0xff) as u8;
            }
        }
    }
    out
}

// ───────────────────────── the case harness ─────────────────────────

fn dev_off(t: &GpuTensor, byte_off: usize) -> *const std::ffi::c_void {
    (t.buf.as_ptr() as usize + byte_off) as *const std::ffi::c_void
}

#[allow(dead_code)]
struct Case {
    m: usize,
    k: usize,
    k_top: usize,
    n_tok: usize,
    gpr: usize,
    mi: usize,
    experts_host: Vec<Vec<u8>>,
    x_host: Vec<f32>,
    topk_host: Vec<i32>,
    // kept alive for the lifetime of the case (device memory)
    expert_t: Vec<GpuTensor>,
    ptr_t: GpuTensor,
    topk_t: GpuTensor,
    x_t: GpuTensor,
    y_g: GpuTensor,
    y_u: GpuTensor,
    blob_batched: Vec<u8>,
    blobs_seq: Vec<Vec<u8>>,
}

impl Case {
    fn new(
        gpu: &mut Gpu,
        m: usize,
        k: usize,
        k_top: usize,
        n_tok: usize,
        n_exp: usize,
        seed: u64,
    ) -> Case {
        assert_eq!(k % GROUP_WEIGHTS, 0, "K must be a multiple of 256");
        assert_eq!(m % 2, 0, "M must be even (gate|up split)");
        let gpr = k / GROUP_WEIGHTS;
        let mi = m / 2;
        let expert_bytes = m * gpr * GROUP_BYTES;

        let experts_host: Vec<Vec<u8>> = (0..n_exp)
            .map(|e| build_mq3l(m, k, seed ^ ((e as u64).wrapping_mul(0x9e37))))
            .collect();

        let expert_t: Vec<GpuTensor> = experts_host
            .iter()
            .map(|w| gpu.upload_raw(w, &[expert_bytes]).expect("upload expert"))
            .collect();
        let ptrs: Vec<u64> = expert_t.iter().map(|t| t.buf.as_ptr() as u64).collect();
        let ptr_t = gpu
            .upload_raw(
                &ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>(),
                &[n_exp],
            )
            .expect("upload ptr table");

        // Per-token routing: every token gets a different expert set, so a
        // kernel that ignored `bid` when indexing topk would be caught.
        let topk_host: Vec<i32> = (0..n_tok * k_top)
            .map(|i| {
                let n = i / k_top;
                let r = i % k_top;
                (mix(0xbeef ^ ((n as u64) << 16) ^ r as u64) % n_exp as u64) as i32
            })
            .collect();
        let topk_t = gpu
            .upload_raw(
                &topk_host
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
                &[n_tok * k_top],
            )
            .expect("upload topk");

        // Per-token x that differs SUBSTANTIALLY between tokens — this is what
        // makes the per-token-vs-per-krank X indexing probe sharp.
        let x_host: Vec<f32> = (0..n_tok * k)
            .map(|i| {
                let n = i / k;
                let c = i % k;
                let r = (mix(0xf00d ^ ((n as u64) << 24) ^ c as u64) % 20001) as f32;
                (r * 1e-4 - 1.0) * 0.05
            })
            .collect();
        let x_t = gpu.upload_f32(&x_host, &[n_tok * k]).expect("upload x");

        let y_g = gpu
            .alloc_tensor(&[n_tok * k_top * mi], DType::F32)
            .expect("alloc y_gate");
        let y_u = gpu
            .alloc_tensor(&[n_tok * k_top * mi], DType::F32)
            .expect("alloc y_up");

        // batched blob: (ptrs, topk, x, y_gate, y_up, M, K, K_TOP)
        let mut bb = KernargBlob::new();
        bb.push_ptr(ptr_t.buf.as_ptr() as *const _);
        bb.push_ptr(topk_t.buf.as_ptr() as *const _);
        bb.push_ptr(x_t.buf.as_ptr() as *const _);
        bb.push_ptr(y_g.buf.as_ptr() as *const _);
        bb.push_ptr(y_u.buf.as_ptr() as *const _);
        bb.push_i32(m as i32);
        bb.push_i32(k as i32);
        bb.push_i32(k_top as i32);
        let blob_batched = bb.into_vec();

        // N single-token blobs: same kernel, pointers pre-offset per token so
        // the N launches land in the SAME [N x K_TOP x MI] output buffer.
        let stride = k_top * mi;
        let blobs_seq: Vec<Vec<u8>> = (0..n_tok)
            .map(|b| {
                let mut s = KernargBlob::new();
                s.push_ptr(ptr_t.buf.as_ptr() as *const _);
                s.push_ptr(dev_off(&topk_t, b * k_top * 4));
                s.push_ptr(dev_off(&x_t, b * k * 4));
                s.push_ptr(dev_off(&y_g, b * stride * 4));
                s.push_ptr(dev_off(&y_u, b * stride * 4));
                s.push_i32(m as i32);
                s.push_i32(k as i32);
                s.into_vec()
            })
            .collect();

        Case {
            m,
            k,
            k_top,
            n_tok,
            gpr,
            mi,
            experts_host,
            x_host,
            topk_host,
            expert_t,
            ptr_t,
            topk_t,
            x_t,
            y_g,
            y_u,
            blob_batched,
            blobs_seq,
        }
    }

    fn expert_bytes(&self) -> usize {
        self.m * self.gpr * GROUP_BYTES
    }

    fn launch_batched(&mut self, gpu: &Gpu) {
        gpu.launch_kernel_blob(
            BATCHED_FN,
            [self.m as u32, self.k_top as u32, self.n_tok as u32],
            [32, 1, 1],
            0,
            &mut self.blob_batched,
        )
        .expect("launch batched");
    }

    fn launch_seq(&mut self, gpu: &Gpu) {
        let grid = [self.m as u32, self.k_top as u32, 1];
        for b in self.blobs_seq.iter_mut() {
            gpu.launch_kernel_blob(SINGLE_FN, grid, [32, 1, 1], 0, b)
                .expect("launch single");
        }
    }

    fn zero_out(&self, gpu: &mut Gpu) {
        gpu.fill_f32(&self.y_g, 0.0).expect("zero y_gate");
        gpu.fill_f32(&self.y_u, 0.0).expect("zero y_up");
    }

    /// Independent CPU reference for output row `row` of token `n` at expert
    /// rank `r`, using token `x_tok`'s x slice. Derived from the qt=20 spec.
    fn cpu_dot(&self, n: usize, r: usize, row: usize, x_tok: usize) -> f64 {
        let e = self.topk_host[n * self.k_top + r] as usize;
        let raw = &self.experts_host[e];
        let x = &self.x_host[x_tok * self.k..(x_tok + 1) * self.k];
        let row_bytes = self.gpr * GROUP_BYTES;
        let mut acc = 0.0f64;
        for g in 0..self.gpr {
            let base = row * row_bytes + g * GROUP_BYTES;
            let mut cb = [0.0f32; CB_ENTRIES];
            for (ei, c) in cb.iter_mut().enumerate() {
                *c = half_to_f32(u16::from_le_bytes([raw[base + 2 * ei], raw[base + 2 * ei + 1]]));
            }
            let idx = &raw[base + CB_BYTES..base + GROUP_BYTES];
            for w in 0..GROUP_WEIGHTS {
                let q = read_bits_lsb(idx, IDX_BITS * w, IDX_BITS);
                acc += (cb[q] as f64) * (x[g * GROUP_WEIGHTS + w] as f64);
            }
        }
        acc
    }
}

// ───────────────────────── error accounting ─────────────────────────

struct Report {
    n_pts: usize,
    max_abs: f64,
    max_rel: f64,
    worst: (usize, usize, usize),
    worst_got: f64,
    worst_want: f64,
    peak: f64,
    exact_zeros: usize,
    bit_equal: usize,
}

impl Report {
    fn pass(&self) -> bool {
        self.max_rel <= TOL_REL && self.exact_zeros == 0
    }
    fn line(&self, label: &str) -> String {
        format!(
            "{:<32} pts={:<7} max_abs={:.3e} max_rel={:.3e} worst(n={},r={},row={}) got={:+.6e} want={:+.6e} peak={:.3e} zeros={}",
            label,
            self.n_pts,
            self.max_abs,
            self.max_rel,
            self.worst.0,
            self.worst.1,
            self.worst.2,
            self.worst_got,
            self.worst_want,
            self.peak,
            self.exact_zeros
        )
    }
}

/// Compare GPU output against the CPU reference over `pts`.
/// `x_tok` selects which token's x the reference uses (per-token vs per-krank
/// hypothesis). `idx` maps (n, r, sub) -> flat output index (layout hypothesis).
fn compare<FX, FI>(
    case: &Case,
    yg: &[f32],
    yu: &[f32],
    pts: &[(usize, usize, usize)],
    x_tok: FX,
    idx: FI,
) -> Report
where
    FX: Fn(usize, usize) -> usize,
    FI: Fn(usize, usize, usize) -> usize,
{
    let mut pairs: Vec<(usize, usize, usize, f64, f64, bool)> = Vec::with_capacity(pts.len());
    let mut exact_zeros = 0usize;
    for &(n, r, row) in pts {
        let want = case.cpu_dot(n, r, row, x_tok(n, r));
        let (buf, sub) = if row < case.mi {
            (yg, row)
        } else {
            (yu, row - case.mi)
        };
        let flat = idx(n, r, sub);
        let got = buf[flat] as f64;
        if buf[flat] == 0.0 {
            exact_zeros += 1;
        }
        let bit_eq = (want as f32).to_bits() == buf[flat].to_bits();
        pairs.push((n, r, row, got, want, bit_eq));
    }
    let peak = pairs.iter().fold(0.0f64, |a, p| a.max(p.4.abs()));
    let floor = 1e-3 * peak.max(1e-30);
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst = (0usize, 0usize, 0usize);
    let mut worst_got = 0.0f64;
    let mut worst_want = 0.0f64;
    let mut bit_equal = 0usize;
    for &(n, r, row, got, want, bit_eq) in &pairs {
        if bit_eq {
            bit_equal += 1;
        }
        let d = (got - want).abs();
        if d > max_abs {
            max_abs = d;
        }
        let rel = d / want.abs().max(floor);
        if rel > max_rel {
            max_rel = rel;
            worst = (n, r, row);
            worst_got = got;
            worst_want = want;
        }
    }
    Report {
        n_pts: pairs.len(),
        max_abs,
        max_rel,
        worst,
        worst_got,
        worst_want,
        peak,
        exact_zeros,
        bit_equal,
    }
}

fn all_points(case: &Case) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::with_capacity(case.n_tok * case.k_top * case.m);
    for n in 0..case.n_tok {
        for r in 0..case.k_top {
            for row in 0..case.m {
                v.push((n, r, row));
            }
        }
    }
    v
}

fn sampled_points(case: &Case, count: usize, seed: u64) -> Vec<(usize, usize, usize)> {
    (0..count)
        .map(|i| {
            let a = mix(seed ^ i as u64);
            let n = (a % case.n_tok as u64) as usize;
            let r = ((a >> 20) % case.k_top as u64) as usize;
            let row = ((a >> 33) % case.m as u64) as usize;
            (n, r, row)
        })
        .collect()
}

/// Documented output layout: y_gate / y_up are `[N x K_TOP x MI]`.
fn layout_std(k_top: usize, mi: usize) -> impl Fn(usize, usize, usize) -> usize {
    let stride = k_top * mi;
    move |n, r, sub| n * stride + r * mi + sub
}

fn ktop_seed(kt: usize) -> u64 {
    0x9911 ^ (kt as u64)
}

// ───────────────────────── main ─────────────────────────

fn main() {
    assert_eq!(
        GROUP_BYTES, LLOYD_MQ3_GROUP_BYTES,
        "spec-derived MQ3 group size disagrees with the crate constant"
    );

    let mut gpu = Gpu::init().expect("GPU init");
    println!("arch={}", gpu.arch);
    println!(
        "MQ3-Lloyd (qt=20): {GROUP_BYTES} B / {GROUP_WEIGHTS} weights = {:.4} B/elem = {:.4} bpw \
         ({CB_BYTES} B codebook + {IDX_BYTES} B of {IDX_BITS}-bit indices)",
        GROUP_BYTES as f64 / GROUP_WEIGHTS as f64,
        GROUP_BYTES as f64 * 8.0 / GROUP_WEIGHTS as f64
    );
    println!("NOTE: `_k4` = 4-accumulator ILP unroll, NOT k_top=4. K_TOP is a runtime arg (grid.y).");
    println!("NOTE: a first-pass number is JIT-contaminated; every timing below is post-warmup.\n");

    gpu.ensure_kernel_public(BATCHED_MOD, BATCHED_SRC, BATCHED_FN)
        .expect("JIT batched_k4");
    gpu.ensure_kernel_public(SINGLE_MOD, SINGLE_SRC, SINGLE_FN)
        .expect("JIT single-token");

    // a3b routed-expert gate_up decode shape.
    let m = 1024usize;
    let k = 2048usize;
    let k_top = 8usize;
    let n_exp = 32usize;
    let mut any_fail = false;

    println!("=== A. correctness vs spec-derived CPU reference (M={m} K={k} k_top={k_top}) ===");
    println!("    tolerance {TOL_REL:.0e} relative, denominator max(|want|, 1e-3*peak|want|)\n");

    for &n_tok in &[1usize, 4, 16, 64] {
        let mut case = Case::new(&mut gpu, m, k, k_top, n_tok, n_exp, 0x51e3);
        let lay = layout_std(k_top, m / 2);

        // --- batched arm ---
        case.zero_out(&mut gpu);
        case.launch_batched(&gpu);
        gpu.hip.device_synchronize().expect("sync");
        let bg = gpu.download_f32(&case.y_g).expect("dl y_gate");
        let bu = gpu.download_f32(&case.y_u).expect("dl y_up");

        // Full reference for N=1; wide sample for the bigger batches — the
        // batched-vs-sequential compare below covers every element anyway.
        let pts = if n_tok == 1 {
            all_points(&case)
        } else {
            sampled_points(&case, 4096, 0x1234 + n_tok as u64)
        };
        let rep = compare(&case, &bg, &bu, &pts, |n, _r| n, &lay);
        println!("  {}", rep.line(&format!("N={n_tok} batched_k4 vs CPU")));
        any_fail |= !rep.pass();

        // --- sequential arm (N launches of the single-token kernel) ---
        case.zero_out(&mut gpu);
        case.launch_seq(&gpu);
        gpu.hip.device_synchronize().expect("sync");
        let sg = gpu.download_f32(&case.y_g).expect("dl y_gate seq");
        let su = gpu.download_f32(&case.y_u).expect("dl y_up seq");
        let rep_s = compare(&case, &sg, &su, &pts, |n, _r| n, &lay);
        println!("  {}", rep_s.line(&format!("N={n_tok} Nxsingle   vs CPU")));
        any_fail |= !rep_s.pass();

        // --- batched vs sequential, EVERY element ---
        let (mut d_abs, mut d_rel, mut d_at) = (0.0f64, 0.0f64, 0usize);
        let mut bit_ident = 0usize;
        let peak = bg
            .iter()
            .chain(bu.iter())
            .fold(0.0f64, |a, v| a.max(v.abs() as f64));
        let floor = 1e-3 * peak.max(1e-30);
        for (i, (a, b)) in bg
            .iter()
            .zip(sg.iter())
            .chain(bu.iter().zip(su.iter()))
            .enumerate()
        {
            if a.to_bits() == b.to_bits() {
                bit_ident += 1;
            }
            let d = (*a as f64 - *b as f64).abs();
            if d > d_abs {
                d_abs = d;
            }
            let rel = d / (*b as f64).abs().max(floor);
            if rel > d_rel {
                d_rel = rel;
                d_at = i;
            }
        }
        let total = bg.len() + bu.len();
        println!(
            "  {:<32} elems={:<7} max_abs={:.3e} max_rel={:.3e} @flat={} bit-identical={}/{} ({:.1}%)",
            format!("N={n_tok} batched vs Nxsingle"),
            total,
            d_abs,
            d_rel,
            d_at,
            bit_ident,
            total,
            100.0 * bit_ident as f64 / total as f64
        );
        let cross_ok = d_rel <= TOL_REL;
        any_fail |= !cross_ok;
        println!(
            "  {:<32} {}",
            format!("N={n_tok} VERDICT"),
            if rep.pass() && rep_s.pass() && cross_ok {
                "PASS"
            } else {
                "*** FAIL ***"
            }
        );

        // --- probes on the N=16 case (n_tok > k_top, so the WRONG hypotheses
        //     stay in bounds and fail on VALUES rather than on a segfault) ---
        if n_tok == 16 {
            let mi = case.mi;

            // X indexing: per-token (what the kernel doc claims) vs per-krank.
            let per_tok = compare(&case, &bg, &bu, &pts, |n, _r| n, &lay);
            let per_krank = compare(&case, &bg, &bu, &pts, |_n, r| r, &lay);
            let x_ok = per_tok.max_rel <= TOL_REL && per_krank.max_rel > 1e-2;
            println!(
                "  probe X-INDEXING: per-token max_rel={:.3e}  per-krank max_rel={:.3e}  -> {}",
                per_tok.max_rel,
                per_krank.max_rel,
                if x_ok {
                    "x is PER-TOKEN (per-krank hypothesis rejected)"
                } else {
                    "*** INCONCLUSIVE - investigate ***"
                }
            );
            any_fail |= !x_ok;

            // Output layout: [N x K_TOP x MI] vs transposed [K_TOP x N x MI].
            let stride_b = case.n_tok * mi;
            let alt = move |n: usize, r: usize, sub: usize| r * stride_b + n * mi + sub;
            let lay_a = compare(&case, &bg, &bu, &pts, |n, _r| n, &lay);
            let lay_b = compare(&case, &bg, &bu, &pts, |n, _r| n, alt);
            let l_ok = lay_a.max_rel <= TOL_REL && lay_b.max_rel > 1e-2;
            println!(
                "  probe OUT-LAYOUT: [NxK_TOPxMI] max_rel={:.3e}  [K_TOPxNxMI] max_rel={:.3e}  -> {}",
                lay_a.max_rel,
                lay_b.max_rel,
                if l_ok {
                    "layout is [N x K_TOP x MI] (transpose rejected)"
                } else {
                    "*** INCONCLUSIVE - investigate ***"
                }
            );
            any_fail |= !l_ok;

            let gate_pts = pts.iter().filter(|p| p.2 < mi).count();
            println!(
                "  probe GATE/UP SPLIT: rows [0,{mi}) -> y_gate ({gate_pts} pts), rows [{mi},{m}) \
                 -> y_up ({} pts); both halves checked. f32-exact matches {}/{}",
                pts.len() - gate_pts,
                lay_a.bit_equal,
                lay_a.n_pts
            );
        }
        println!();
    }

    // ── B. K_TOP is a RUNTIME argument, not the `k4` in the kernel name ──
    println!("=== B. K_TOP runtime-arg probe (same compiled `_k4` kernel, N=4) ===");
    for &kt in &[1usize, 3, 8] {
        let mut case = Case::new(&mut gpu, m, k, kt, 4, n_exp, 0x7ab1 + kt as u64);
        let lay = layout_std(kt, m / 2);
        case.zero_out(&mut gpu);
        case.launch_batched(&gpu);
        gpu.hip.device_synchronize().expect("sync");
        let yg = gpu.download_f32(&case.y_g).expect("dl");
        let yu = gpu.download_f32(&case.y_u).expect("dl");
        let pts = sampled_points(&case, 2048, ktop_seed(kt));
        let rep = compare(&case, &yg, &yu, &pts, |n, _r| n, &lay);
        println!(
            "  {}  -> {}",
            rep.line(&format!("K_TOP={kt}")),
            if rep.pass() { "PASS" } else { "*** FAIL ***" }
        );
        any_fail |= !rep.pass();
    }
    println!();

    // ── C. tail-group coverage. K=2048 gives gpr=8 => quads=2, tail=0, so the
    //    kernel's three tail expansions are NEVER exercised on the a3b shape. ──
    println!("=== C. tail-group coverage (M=256, k_top=2, N=3) - the a3b K=2048 shape has tail=0 ===");
    for &kk in &[1280usize, 1536, 1792] {
        let gpr = kk / GROUP_WEIGHTS;
        let mut case = Case::new(&mut gpu, 256, kk, 2, 3, 8, 0x3311 + kk as u64);
        let lay = layout_std(2, 128);
        case.zero_out(&mut gpu);
        case.launch_batched(&gpu);
        gpu.hip.device_synchronize().expect("sync");
        let bg = gpu.download_f32(&case.y_g).expect("dl");
        let bu = gpu.download_f32(&case.y_u).expect("dl");
        let pts = all_points(&case);
        let rep = compare(&case, &bg, &bu, &pts, |n, _r| n, &lay);
        println!(
            "  {}  -> {}",
            rep.line(&format!("K={kk} gpr={gpr} tail={} batched", gpr & 3)),
            if rep.pass() { "PASS" } else { "*** FAIL ***" }
        );
        any_fail |= !rep.pass();

        case.zero_out(&mut gpu);
        case.launch_seq(&gpu);
        gpu.hip.device_synchronize().expect("sync");
        let sg = gpu.download_f32(&case.y_g).expect("dl");
        let su = gpu.download_f32(&case.y_u).expect("dl");
        let rep_s = compare(&case, &sg, &su, &pts, |n, _r| n, &lay);
        println!(
            "  {}  -> {}",
            rep_s.line(&format!("K={kk} gpr={gpr} tail={} single ", gpr & 3)),
            if rep_s.pass() { "PASS" } else { "*** FAIL ***" }
        );
        any_fail |= !rep_s.pass();
    }
    println!();

    println!(
        "*** CORRECTNESS: {} ***  (tolerance {TOL_REL:.0e} relative)\n",
        if any_fail { "FAIL" } else { "PASS" }
    );

    // ── D. timing ──
    println!("=== D. timing - batched_k4 (1 launch) vs N x single-token launches ===");
    println!("    warmup pass, then 7 timed iterations; MIN and MEDIAN reported.");
    println!("    HIP-dispatch microbenchmark: TRIAGE ONLY, not a kernel verdict (see file header).");
    println!(
        "    weight bytes/call = N x k_top x {} KiB (algorithmic; L2 reuse across repeated experts \
         is NOT subtracted)",
        (m * (k / GROUP_WEIGHTS) * GROUP_BYTES) / 1024
    );
    println!(
        "\n  {:<5} {:>11} {:>11} {:>11} {:>11} {:>9} {:>12} {:>11} {:>11}",
        "N",
        "bat_min_us",
        "bat_med_us",
        "seq_min_us",
        "seq_med_us",
        "speedup",
        "bat_wt_GB/s",
        "bat_us/tok",
        "bat_tok/s"
    );

    for &n_tok in &[1usize, 4, 16, 64] {
        let mut case = Case::new(&mut gpu, m, k, k_top, n_tok, n_exp, 0x51e3);
        let reps = match n_tok {
            1 => 20,
            4 => 10,
            16 => 5,
            _ => 3,
        };

        // warmup: JIT + kernel cache + DPM ramp for BOTH arms at THIS shape.
        for _ in 0..(reps * 2) {
            case.launch_batched(&gpu);
        }
        gpu.hip.device_synchronize().expect("sync");
        for _ in 0..reps {
            case.launch_seq(&gpu);
        }
        gpu.hip.device_synchronize().expect("sync");

        let iters = 7usize;
        let mut tb = Vec::with_capacity(iters);
        let mut ts = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            for _ in 0..reps {
                case.launch_batched(&gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            tb.push(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);

            let t1 = Instant::now();
            for _ in 0..reps {
                case.launch_seq(&gpu);
            }
            gpu.hip.device_synchronize().expect("sync");
            ts.push(t1.elapsed().as_secs_f64() * 1e6 / reps as f64);
        }
        tb.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (bmin, bmed) = (tb[0], tb[iters / 2]);
        let (smin, smed) = (ts[0], ts[iters / 2]);

        let wt_bytes = (n_tok * k_top * case.expert_bytes()) as f64;
        println!(
            "  {:<5} {:>11.2} {:>11.2} {:>11.2} {:>11.2} {:>8.2}x {:>12.1} {:>11.2} {:>11.0}",
            n_tok,
            bmin,
            bmed,
            smin,
            smed,
            smed / bmed,
            wt_bytes / (bmed * 1e-6) / 1e9,
            bmed / n_tok as f64,
            n_tok as f64 / (bmed * 1e-6)
        );
    }

    let gpr = k / GROUP_WEIGHTS;
    let ebytes = m * gpr * GROUP_BYTES;
    println!(
        "\n  per-expert gate_up weights: {} KiB  ({:.4} B/elem, {:.4} bpw)",
        ebytes / 1024,
        GROUP_BYTES as f64 / GROUP_WEIGHTS as f64,
        GROUP_BYTES as f64 * 8.0 / GROUP_WEIGHTS as f64
    );
    println!(
        "  x traffic/call: N x k_top x K x 4 B (x is re-read per krank); \
         out traffic/call: N x k_top x M x 4 B"
    );
    println!(
        "  GB/s is decimal (1e9), weight-only, and ignores L2 reuse when two tokens route to the \
         same expert - it is an UPPER bound on real DRAM traffic."
    );
    println!(
        "\n  Reminder: the sequential arm pays N host launches per call, the batched arm pays 1.\n  \
         Part of any speedup here is launch bookkeeping, not kernel quality. Confirm on the golden\n  \
         bundle (registry/redline-golden-v1.json) with HIP and PM4 arms before claiming a win."
    );

    if any_fail {
        std::process::exit(1);
    }
}
