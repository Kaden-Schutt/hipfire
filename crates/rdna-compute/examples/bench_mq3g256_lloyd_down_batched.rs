// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Correctness + triage timing for the **batched** MQ3-Lloyd MoE `down` GEMV:
//! `kernels/src/gemv_mq3g256_lloyd_moe_down_indexed_batched_k4.hip`
//! (`gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4`).
//!
//! Shape: a3b routed-expert `down` — M=2048, K=512, k_top=8, n_experts=256,
//! swept over N (tokens per launch) ∈ {1, 4, 16, 64}.
//!
//! The kernel is JIT'd from source and launched through the kernarg-blob path
//! (`ensure_kernel_public` + `launch_kernel_blob`), so this bench touches NO
//! dispatch plumbing, no kernel, and no shared file.
//!
//! ## REGIME CAVEAT — read before quoting any number from this file
//!
//! This is a **HIP-dispatch microbenchmark**. It is a TRIAGE FILTER for gross
//! defects — wrong numbers, per-token leakage, pathological slowness — and NOT
//! a verdict on the kernel. Host launch latency masks device-level effects, and
//! the same kernel can measure differently once lowered to retained PM4 replay.
//! A prior HIP microbench measured the MQ2GL kernel at -3.1% and that was nearly
//! used to kill the format. Final acceptance is the golden bundle
//! (`registry/redline-golden-v1.json`) with HIP and PM4 arms. Treat the timing
//! section here as "is it in the right order of magnitude", nothing more.
//! Pass 0 of the timing loop is JIT- and cold-cache-contaminated by construction;
//! it is printed, labelled, and discarded.
//!
//! ## Combine semantics — DETERMINED BY READING, NOT ASSUMED
//!
//! The batched `down` path **self-combines via `atomicAdd` into the residual**.
//! It does NOT write an expanded `[N x K_TOP x M]` buffer. Verbatim epilogue,
//! `gemv_mq3g256_lloyd_moe_down_indexed_batched_k4.hip:187-190`:
//!
//! ```text
//!     if (tid == 0) {
//!         const float scale = topk_weights[routing_base + krank];
//!         atomicAdd(&x_residual[(long long)bid * M + row], scale * acc);
//!     }
//! ```
//!
//! The MQ2-Lloyd sibling `gemv_mq2g256_lloyd_moe_down_indexed_batched_k4.hip:102-105`
//! carries the identical epilogue. So `x_residual` is an **in/out** buffer of
//! shape `[N x M]`: the kernel ADDS `sum_krank w * (W_e . x)` onto whatever was
//! already there. The CPU reference models exactly that — it seeds itself from
//! the same residual pattern the device buffer was preloaded with — and the
//! preload is deliberately non-zero so an overwrite-instead-of-accumulate bug
//! cannot pass.
//!
//! Consequence for determinism: the `K_TOP` grid-y blocks contend on the same
//! `x_residual` cell, so FP add order across experts is non-deterministic. That
//! rules out bit-exact comparison; tolerances below are set accordingly.
//!
//! ## Format spec used by the CPU reference (MQ3-Lloyd, qt=20)
//!
//! 112 B per group of 256 weights, AoS, row-major over rows
//! (`row_ptr = A + row * (K/256) * 112`):
//!   * bytes `[0..16)`   - 8 x fp16 codebook entries, ascending (little-endian).
//!   * bytes `[16..112)` - 96 B of 3-bit indices for the group's 256 weights.
//!
//! That is 112 B / 256 weights = **0.4375 B/weight = 3.500 bits/weight**.
//! (Not to be confused with MQ3**GL** qt=39 at 3.0625 bpw — that is the SoA
//! global-codebook format and a different kernel family. This kernel is qt=20.)
//!
//! ## Independence of the CPU reference
//!
//! Three independent expressions of the same bit layout are used, so a shared
//! misunderstanding cannot cancel out:
//!   1. **Writer** (`build_expert`) uses the AUTHORITATIVE encoder packing lifted
//!      from `hipfire-quantize/src/main.rs::quantize_mq3g256_lloyd` — the
//!      chunk-of-8-weights -> 3-bytes formula that actually produces `.hfq` bytes.
//!   2. **Reference reader** (`code3`) is derived from the FORMAT SPEC as a plain
//!      contiguous LSB-first bitstream: weight `j` occupies bits `[3j, 3j+3)` of
//!      the 96 B region. It shares no structure with the writer.
//!   3. **Kernel** slices a per-thread little-endian uint24 (`boff = tid*3`,
//!      8 codes at `(pk >> 3s) & 7`).
//! A host-side round-trip self-check asserts (1) and (2) agree before any GPU
//! work happens; the GPU comparison then puts (3) against them.
//!
//! `x` is NOT actually FWHT-256 rotated here — for a microbench it is an
//! arbitrary deterministic pseudo-random activation, used byte-identically by
//! the GPU and the CPU reference. Rotation is a caller-side transform, outside
//! this kernel's contract.
//!
//! ## What is checked
//!
//!  * batched launch `grid = (M, K_TOP, N)` vs an f64 CPU reference — max abs
//!    err, max rel err, scale-normalised rel err, worst offending (token, row).
//!  * batched vs **N sequential single-token launches** (`grid = (M, K_TOP, 1)`
//!    with per-token offset pointers) — catches batch-index arithmetic bugs.
//!  * **per-token separation**: a poison run where every token's activation is
//!    zero except one; every other token's residual must come back BIT-EXACT
//!    equal to its preload, i.e. token i cannot leak into token j.
//!  * tail-path coverage: `groups_per_row = K/256` selects the kernel's
//!    quad/tail split. The primary K=512 shape has gpr=2 -> quads=0, tail=2,
//!    which exercises ONLY the `TAIL_LOAD_AND_DOT` macro path. Extra
//!    correctness-only shapes cover gpr in {3, 5, 8} -> (quads, tail) =
//!    (0,3), (1,1), (2,0), so the main quad loop and every tail arm are hit.
//!
//! Run: cargo run --release -p rdna-compute --example bench_mq3g256_lloyd_down_batched
use hip_bridge::KernargBlob;
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_down_indexed_batched_k4.hip");
const MODULE: &str = "gemv_mq3g256_lloyd_moe_down_indexed_batched_k4";
const FUNC: &str = "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4";

const GROUP: usize = 256;
const GROUP_BYTES: usize = 112;
const CB_BYTES: usize = 16;
const IDX_BYTES: usize = 96;

/// PASS threshold on the scale-normalised relative error (see `compare`).
const TOL: f64 = 1e-4;

/// Textbook Lloyd-Max levels for a unit Gaussian, 3 bit / 8 levels, ascending —
/// the shape a real per-block MQ3-Lloyd codebook converges to. Ascending order
/// matters: the encoder sorts centroids ascending and remaps indices.
const CB8: [f32; 8] = [
    -1.7479, -1.0500, -0.5606, -0.1806, 0.1806, 0.5606, 1.0500, 1.7479,
];

// ── deterministic host RNG ─────────────────────────────────────────────────

fn mix(x: u64) -> u64 {
    let h = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

/// Uniform in [-1, 1), deterministic in `seed`.
fn unit(seed: u64) -> f32 {
    ((mix(seed) >> 40) & 0xffff) as f32 / 32768.0 - 1.0
}

// ── fp16 <-> f32 on the host ───────────────────────────────────────────────
// Round-to-zero mantissa; every value fed through here sits well inside the
// fp16 normal range, so the truncation is not load-bearing. The CPU reference
// reads the codebook back OUT of the packed blob, so fp16 rounding cancels
// between the GPU and the reference by construction.

fn half_bits(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if e <= 0 {
        return sign; // flush subnormal / zero to signed zero
    }
    if e >= 31 {
        return sign | (30 << 10) | 0x3ff; // clamp to max normal
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

// ── synthetic expert blob in the exact on-disk MQ3-Lloyd byte layout ───────

/// The 8 codebook indices for one 8-weight chunk. Deterministic. Used ONLY by
/// the writer and by the packing self-check — the CPU reference re-reads the
/// packed bytes instead of calling this, so a bug here cannot cancel out.
fn chunk_codes(seed: u64, row: usize, g: usize, chunk: usize) -> [u8; 8] {
    let h = mix(seed ^ 0x5eed ^ ((row as u64) << 32) ^ ((g as u64) << 12) ^ chunk as u64);
    std::array::from_fn(|s| ((h >> (3 * s)) & 7) as u8)
}

/// Build one expert's `down` weight blob: `[M rows][K/256 groups][112 B]`.
///
/// The 3-bit packing below is the AUTHORITATIVE encoder formula, lifted from
/// `hipfire-quantize/src/main.rs::quantize_mq3g256_lloyd`:
/// ```text
///     b0 = q0 | (q1 << 3) | ((q2 & 3) << 6);
///     b1 = (q2 >> 2) | (q3 << 1) | (q4 << 4) | ((q5 & 1) << 7);
///     b2 = (q5 >> 1) | (q6 << 2) | (q7 << 5);
/// ```
fn build_expert(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / GROUP;
    let mut out = vec![0u8; m * gpr * GROUP_BYTES];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * GROUP_BYTES;
            // per-group scale in a realistic band for a3b expert weights
            let s = 0.004f32 + (mix(seed ^ ((row as u64) << 20) ^ g as u64) % 4000) as f32 * 1e-6;
            for (e, &c) in CB8.iter().enumerate() {
                let h = half_bits(c * s);
                out[off + 2 * e] = (h & 0xff) as u8;
                out[off + 2 * e + 1] = (h >> 8) as u8;
            }
            for chunk in 0..(GROUP / 8) {
                let q = chunk_codes(seed, row, g, chunk);
                let b0 = q[0] | (q[1] << 3) | ((q[2] & 3) << 6);
                let b1 = (q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | ((q[5] & 1) << 7);
                let b2 = (q[5] >> 1) | (q[6] << 2) | (q[7] << 5);
                let bo = off + CB_BYTES + chunk * 3;
                out[bo] = b0;
                out[bo + 1] = b1;
                out[bo + 2] = b2;
            }
        }
    }
    out
}

// ── independent (format-spec) index reader ─────────────────────────────────

/// Weight `j` of a 256-group occupies bits `[3j, 3j+3)` of the 96 B index
/// region, LSB-first within each byte, bytes in increasing address order.
/// A 3-bit field never spans more than two bytes.
fn code3(region: &[u8], j: usize) -> usize {
    let bit = 3 * j;
    let byte = bit >> 3;
    let sh = bit & 7;
    let lo = region[byte] as u32;
    let hi = if byte + 1 < region.len() {
        region[byte + 1] as u32
    } else {
        0
    };
    (((lo | (hi << 8)) >> sh) & 7) as usize
}

/// Assert writer (encoder formula) and reader (bitstream spec) agree, on host,
/// before any GPU work. If this trips, everything downstream is meaningless.
fn self_check_packing() {
    let m = 3usize;
    let k = 512usize;
    let gpr = k / GROUP;
    let blob_bytes = build_expert(m, k, 0xabc);
    let mut checked = 0usize;
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * GROUP_BYTES;
            let idx = &blob_bytes[off + CB_BYTES..off + CB_BYTES + IDX_BYTES];
            for chunk in 0..(GROUP / 8) {
                let want = chunk_codes(0xabc, row, g, chunk);
                for (s, &w) in want.iter().enumerate() {
                    let got = code3(idx, chunk * 8 + s);
                    assert_eq!(
                        got, w as usize,
                        "MQ3-Lloyd 3-bit packing self-check FAILED at row={row} g={g} \
                         chunk={chunk} sub={s}: spec bitstream reader says {got}, \
                         encoder wrote {w}"
                    );
                    checked += 1;
                }
            }
        }
    }
    println!("packing self-check: {checked} codes round-tripped encoder-write -> spec-read   OK");
}

// ── CPU reference ──────────────────────────────────────────────────────────

/// Independent f64 reference for the batched down GEMV INCLUDING the atomicAdd
/// residual accumulate:
///
/// ```text
///   y[b][row] = resid_init[b][row]
///             + sum_r  w[b][r] * sum_j  cb_e[row][j/256][ code3(j) ] * x[b][r][j]
///   with e = topk[b][r]
/// ```
///
/// Rows with `row % row_stride != 0` are left as NaN and skipped by `compare`
/// (host-cost control at the largest N; the stride is reported in the output).
#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    experts: &[Vec<u8>],
    topk: &[i32],
    tw: &[f32],
    rot: &[f32],
    resid_init: &[f32],
    n: usize,
    m: usize,
    k: usize,
    k_top: usize,
    row_stride: usize,
) -> Vec<f64> {
    let gpr = k / GROUP;
    let row_bytes = gpr * GROUP_BYTES;
    let mut out = vec![f64::NAN; n * m];
    for b in 0..n {
        for row in (0..m).step_by(row_stride) {
            out[b * m + row] = resid_init[b * m + row] as f64;
        }
        for r in 0..k_top {
            let e = topk[b * k_top + r] as usize;
            let w = tw[b * k_top + r] as f64;
            let a = &experts[e];
            let xs = &rot[(b * k_top + r) * k..(b * k_top + r + 1) * k];
            for row in (0..m).step_by(row_stride) {
                let rp = &a[row * row_bytes..(row + 1) * row_bytes];
                let mut acc = 0.0f64;
                for g in 0..gpr {
                    let gp = &rp[g * GROUP_BYTES..(g + 1) * GROUP_BYTES];
                    let mut cb = [0.0f64; 8];
                    for (ci, c) in cb.iter_mut().enumerate() {
                        *c = half_to_f32(u16::from_le_bytes([gp[2 * ci], gp[2 * ci + 1]])) as f64;
                    }
                    let idx = &gp[CB_BYTES..CB_BYTES + IDX_BYTES];
                    let xg = &xs[g * GROUP..(g + 1) * GROUP];
                    for (j, &xv) in xg.iter().enumerate() {
                        acc += cb[code3(idx, j)] * xv as f64;
                    }
                }
                out[b * m + row] += w * acc;
            }
        }
    }
    out
}

// ── error reporting ────────────────────────────────────────────────────────

struct Err2 {
    max_abs: f64,
    max_rel: f64,
    max_rel_norm: f64,
    worst_b: usize,
    worst_row: usize,
    worst_got: f64,
    worst_want: f64,
    n_checked: usize,
}

/// Compare a GPU result against a reference (NaN reference entries are skipped).
///
/// Two relative metrics are reported because this output is a *signed* sum of
/// `k_top * K` terms with heavy cancellation — the sum of |terms| is ~20x the
/// result — so individual rows land arbitrarily close to zero and a pure
/// relative error there is meaningless (it diverges on fp32 noise alone):
///   * `rel`      = |err| / max(|want|, 1e-30)         - pure, informational
///   * `rel_norm` = |err| / max(|want|, rms(want))     - cancellation-safe
///
/// PASS is gated on `rel_norm`; the worst offender is tracked by `rel_norm`.
/// Sensitivity: rms(want) is ~0.08 on this shape and the expected fp32
/// accumulation noise is ~1e-7 absolute (shallow per-thread chains: 8 terms
/// per accumulator, then a 5-level wave reduction, then k_top atomicAdds), so
/// the floor sits ~3 decades above the noise. Any real defect — wrong stride,
/// wrong codebook slot, wrong batch index, token leakage — moves an entry by
/// order rms and lands at rel_norm ~1, i.e. it fails by four decades. This
/// gate is a gross-defect filter, not an ULP audit.
fn compare(got: &[f32], want: &[f64], m: usize) -> Err2 {
    let mut sum2 = 0.0f64;
    let mut cnt = 0usize;
    for &w in want.iter() {
        if w.is_finite() {
            sum2 += w * w;
            cnt += 1;
        }
    }
    let rms = if cnt > 0 {
        (sum2 / cnt as f64).sqrt()
    } else {
        0.0
    };
    let floor = rms.max(1e-30);
    let mut e = Err2 {
        max_abs: 0.0,
        max_rel: 0.0,
        max_rel_norm: 0.0,
        worst_b: 0,
        worst_row: 0,
        worst_got: 0.0,
        worst_want: 0.0,
        n_checked: cnt,
    };
    for (i, &w) in want.iter().enumerate() {
        if !w.is_finite() {
            continue;
        }
        let g = got[i] as f64;
        let d = (g - w).abs();
        if d > e.max_abs {
            e.max_abs = d;
        }
        let pure = d / w.abs().max(1e-30);
        if pure > e.max_rel {
            e.max_rel = pure;
        }
        let rn = d / w.abs().max(floor);
        if rn > e.max_rel_norm {
            e.max_rel_norm = rn;
            e.worst_b = i / m;
            e.worst_row = i % m;
            e.worst_got = g;
            e.worst_want = w;
        }
    }
    e
}

fn report(label: &str, e: &Err2) -> bool {
    let pass = e.max_rel_norm <= TOL && e.max_abs.is_finite();
    println!(
        "    {:<30} checked={:<8} max|err|={:.3e}  max_rel={:.3e}  max_rel_norm={:.3e}",
        label, e.n_checked, e.max_abs, e.max_rel, e.max_rel_norm
    );
    println!(
        "    {:<30} worst @ token {:<4} row {:<6}  gpu={:+.8e}  ref={:+.8e}",
        "", e.worst_b, e.worst_row, e.worst_got, e.worst_want
    );
    println!(
        "    >>> {} <<<  {}   (tol {:.0e} on rel_norm)",
        if pass { "PASS" } else { "FAIL" },
        label,
        TOL
    );
    pass
}

// ── kernarg blob (5 pointers + M, K, K_TOP) ────────────────────────────────

fn blob(ptrs: [u64; 5], m: i32, k: i32, k_top: i32) -> Vec<u8> {
    let mut b = KernargBlob::new();
    for p in ptrs {
        b.push_u64(p);
    }
    b.push_i32(m);
    b.push_i32(k);
    b.push_i32(k_top);
    b.into_vec()
}

// ── one (M, K, n_exp) case: expert weights on host and device ──────────────

struct Case {
    m: usize,
    k: usize,
    k_top: usize,
    host_experts: Vec<Vec<u8>>,
    _dev_experts: Vec<GpuTensor>,
    ptr_t: GpuTensor,
    expert_bytes: usize,
}

impl Case {
    fn new(gpu: &mut Gpu, m: usize, k: usize, k_top: usize, n_exp: usize) -> Self {
        assert_eq!(k % GROUP, 0, "K must be a multiple of 256");
        let expert_bytes = m * (k / GROUP) * GROUP_BYTES;
        let mut host_experts = Vec::with_capacity(n_exp);
        let mut dev = Vec::with_capacity(n_exp);
        for e in 0..n_exp {
            let h = build_expert(m, k, 0x1000 + e as u64);
            dev.push(gpu.upload_raw(&h, &[expert_bytes]).unwrap());
            host_experts.push(h);
        }
        let ptr_bytes: Vec<u8> = dev
            .iter()
            .flat_map(|t| (t.buf.as_ptr() as u64).to_le_bytes())
            .collect();
        let ptr_t = gpu.upload_raw(&ptr_bytes, &[n_exp]).unwrap();
        Self {
            m,
            k,
            k_top,
            host_experts,
            _dev_experts: dev,
            ptr_t,
            expert_bytes,
        }
    }

    fn n_exp(&self) -> usize {
        self.host_experts.len()
    }

    /// Routing: distinct experts within a token, varied across tokens, weights
    /// positive and summing to 1 per token (softmax-shaped).
    fn routing(&self, n: usize) -> (Vec<i32>, Vec<f32>) {
        let ne = self.n_exp();
        let kt = self.k_top;
        let mut idx = vec![0i32; n * kt];
        let mut w = vec![0f32; n * kt];
        for b in 0..n {
            let mut acc = 0.0f32;
            for r in 0..kt {
                idx[b * kt + r] = ((b * 29 + r * 7 + 3) % ne) as i32;
                let v = 0.05 + 0.95 * (unit(0x7017_0000 ^ ((b as u64) << 8) ^ r as u64) * 0.5 + 0.5);
                w[b * kt + r] = v;
                acc += v;
            }
            for r in 0..kt {
                w[b * kt + r] /= acc;
            }
        }
        (idx, w)
    }

    /// `[N x K_TOP x K]` activations — `down` has a per-(token, krank) slice.
    fn activations(&self, n: usize) -> Vec<f32> {
        (0..n * self.k_top * self.k)
            .map(|i| unit(0xac71_0000 ^ i as u64))
            .collect()
    }

    /// Non-zero, strictly positive residual preload — the
    /// accumulate-vs-overwrite discriminator.
    ///
    /// Magnitude matters. It is deliberately set to the same order as the GEMV
    /// contribution itself (~0.08 rms on the primary shape) so that:
    ///   * an overwrite-instead-of-accumulate bug moves every cell by ~rms and
    ///     fails loudly, AND
    ///   * the preload does not DOMINATE the output and mask an error in the
    ///     GEMV part behind a bit-exactly-preserved constant.
    /// Strictly positive also keeps the leak assay's bit-exactness argument
    /// clean: no cell can be `-0.0`, so `atomicAdd(cell, +0.0)` is a no-op on
    /// the bit pattern.
    fn residual_init(&self, n: usize) -> Vec<f32> {
        (0..n * self.m)
            .map(|i| 0.05 + 0.03 * unit(0x5e51_0000 ^ i as u64))
            .collect()
    }
}

// ── device helpers ─────────────────────────────────────────────────────────

fn upload_i32(gpu: &Gpu, v: &[i32]) -> GpuTensor {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.upload_raw(&bytes, &[v.len()]).unwrap()
}

fn preload(gpu: &Gpu, t: &GpuTensor, data: &[f32]) {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
}

// ── correctness on one (case, N) ───────────────────────────────────────────

/// Returns `true` if every check for this (case, N) passed.
fn correctness(gpu: &mut Gpu, c: &Case, n: usize, row_stride: usize, label: &str) -> bool {
    let (m, k, kt) = (c.m, c.k, c.k_top);
    let gpr = k / GROUP;
    let quads = gpr >> 2;
    let tail = gpr & 3;
    println!(
        "\n── {label}: M={m} K={k} k_top={kt} n_exp={} N={n}  (gpr={gpr} -> quads={quads}, tail={tail}; \
         ref row_stride={row_stride})",
        c.n_exp()
    );

    let (topk, tw) = c.routing(n);
    let rot = c.activations(n);
    let resid = c.residual_init(n);

    let topk_t = upload_i32(gpu, &topk);
    let tw_t = gpu.upload_f32(&tw, &[n * kt]).unwrap();
    let rot_t = gpu.upload_f32(&rot, &[n * kt * k]).unwrap();
    let y_t = gpu.upload_f32(&resid, &[n * m]).unwrap();

    let pp = c.ptr_t.buf.as_ptr() as u64;
    let ip = topk_t.buf.as_ptr() as u64;
    let wp = tw_t.buf.as_ptr() as u64;
    let rp = rot_t.buf.as_ptr() as u64;
    let yp = y_t.buf.as_ptr() as u64;

    // ---- arm 1: one batched launch, grid (M, K_TOP, N) ----
    let mut b_batched = blob([pp, ip, wp, rp, yp], m as i32, k as i32, kt as i32);
    gpu.launch_kernel_blob(
        FUNC,
        [m as u32, kt as u32, n as u32],
        [32, 1, 1],
        0,
        &mut b_batched,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let got_batched = gpu.download_f32(&y_t).unwrap();

    // ---- CPU reference ----
    let t0 = Instant::now();
    let want = cpu_reference(
        &c.host_experts,
        &topk,
        &tw,
        &rot,
        &resid,
        n,
        m,
        k,
        kt,
        row_stride,
    );
    let ref_secs = t0.elapsed().as_secs_f64();

    // Sensitivity diagnostic: how much of the reference output is the
    // bit-exactly-preserved preload vs the GEMV contribution the kernel
    // actually computes. If the preload dominated, an error in the GEMV part
    // would be masked — so this ratio is part of the evidence, not decoration.
    let (mut s_pre, mut s_con, mut cnt) = (0.0f64, 0.0f64, 0usize);
    for (i, &w) in want.iter().enumerate() {
        if w.is_finite() {
            let p = resid[i] as f64;
            s_pre += p * p;
            s_con += (w - p) * (w - p);
            cnt += 1;
        }
    }
    let (rms_pre, rms_con) = (
        (s_pre / cnt as f64).sqrt(),
        (s_con / cnt as f64).sqrt(),
    );
    println!(
        "    signal check: rms(residual preload)={rms_pre:.4e}  rms(GEMV contribution)={rms_con:.4e}  \
         contribution/preload={:.2}x",
        rms_con / rms_pre.max(1e-30)
    );

    let e = compare(&got_batched, &want, m);
    let mut ok = report("batched vs CPU reference", &e);
    println!("    (f64 CPU reference took {ref_secs:.2} s)");

    // ---- arm 2: N sequential single-token launches with offset pointers ----
    // bid is always 0, so token i is addressed purely by shifting the four
    // per-token pointers. Any batch-index arithmetic bug in the kernel shows up
    // here as a mismatch against arm 1.
    preload(gpu, &y_t, &resid);
    for i in 0..n {
        let mut bi = blob(
            [
                pp,
                ip + (i * kt * 4) as u64,
                wp + (i * kt * 4) as u64,
                rp + (i * kt * k * 4) as u64,
                yp + (i * m * 4) as u64,
            ],
            m as i32,
            k as i32,
            kt as i32,
        );
        gpu.launch_kernel_blob(FUNC, [m as u32, kt as u32, 1], [32, 1, 1], 0, &mut bi)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let got_seq = gpu.download_f32(&y_t).unwrap();

    // This arm costs no host compute, so it covers EVERY row regardless of the
    // CPU reference's row_stride — it is the full-coverage batch-indexing check.
    let seq_full: Vec<f64> = got_seq.iter().map(|&v| v as f64).collect();
    let e2 = compare(&got_batched, &seq_full, m);
    ok &= report("batched vs N seq (ALL rows)", &e2);

    let e3 = compare(&got_seq, &want, m);
    ok &= report("sequential vs CPU reference", &e3);

    // ---- arm 3: per-token separation (leak assay) ----
    // Zero every token's activation except one. Every other token's residual
    // must come back BIT-EXACT equal to its preload: cb[q] * 0.0 == 0.0, the
    // wave reduction of zeros is 0.0, scale * 0.0 == 0.0, and atomicAdd(+0.0)
    // on a strictly positive preload leaves the bits untouched. Any leakage
    // from the live token into a dead one is therefore visible exactly.
    if n >= 2 {
        let live = n / 2;
        let mut rot_poison = vec![0.0f32; n * kt * k];
        let lo = live * kt * k;
        rot_poison[lo..lo + kt * k].copy_from_slice(&rot[lo..lo + kt * k]);
        let rotp_t = gpu.upload_f32(&rot_poison, &[n * kt * k]).unwrap();
        preload(gpu, &y_t, &resid);
        let mut bp = blob(
            [pp, ip, wp, rotp_t.buf.as_ptr() as u64, yp],
            m as i32,
            k as i32,
            kt as i32,
        );
        gpu.launch_kernel_blob(
            FUNC,
            [m as u32, kt as u32, n as u32],
            [32, 1, 1],
            0,
            &mut bp,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();
        let poisoned = gpu.download_f32(&y_t).unwrap();

        let mut leaked = 0usize;
        let mut worst = (0usize, 0usize, 0.0f64);
        for b in 0..n {
            if b == live {
                continue;
            }
            for row in 0..m {
                let i = b * m + row;
                if poisoned[i].to_bits() != resid[i].to_bits() {
                    leaked += 1;
                    let d = (poisoned[i] as f64 - resid[i] as f64).abs();
                    if d > worst.2 {
                        worst = (b, row, d);
                    }
                }
            }
        }
        // sanity: the live token MUST have moved, else the assay is vacuous
        let moved = (0..m)
            .filter(|&row| {
                poisoned[live * m + row].to_bits() != resid[live * m + row].to_bits()
            })
            .count();
        let leak_ok = leaked == 0 && moved * 20 >= m * 19;
        println!(
            "    {:<30} live token {live}: {moved}/{m} rows moved   dead tokens: {leaked} leaked cells",
            "per-token separation"
        );
        if leaked > 0 {
            println!(
                "    {:<30} worst leak @ token {} row {}  delta={:.3e}",
                "", worst.0, worst.1, worst.2
            );
        }
        println!(
            "    >>> {} <<<  per-token separation (bit-exact; token i must not leak into token j)",
            if leak_ok { "PASS" } else { "FAIL" }
        );
        ok &= leak_ok;
    } else {
        println!("    (per-token separation skipped: N=1)");
    }

    ok
}

// ── timing (triage only — see the REGIME CAVEAT in the module header) ──────

fn time_case(gpu: &mut Gpu, c: &Case, n: usize) -> (f64, f64) {
    let (m, k, kt) = (c.m, c.k, c.k_top);
    let (topk, tw) = c.routing(n);
    let rot = c.activations(n);
    let resid = c.residual_init(n);

    let topk_t = upload_i32(gpu, &topk);
    let tw_t = gpu.upload_f32(&tw, &[n * kt]).unwrap();
    let rot_t = gpu.upload_f32(&rot, &[n * kt * k]).unwrap();
    let y_t = gpu.upload_f32(&resid, &[n * m]).unwrap();

    let mut b = blob(
        [
            c.ptr_t.buf.as_ptr() as u64,
            topk_t.buf.as_ptr() as u64,
            tw_t.buf.as_ptr() as u64,
            rot_t.buf.as_ptr() as u64,
            y_t.buf.as_ptr() as u64,
        ],
        m as i32,
        k as i32,
        kt as i32,
    );
    let grid = [m as u32, kt as u32, n as u32];
    let block = [32u32, 1, 1];

    // launches per timed sample: keep each sample a few ms regardless of N
    let inner = (160 / n).clamp(3, 40);

    // warmup — absorbs JIT of this (kernel, shape) cell and warms the caches
    for _ in 0..8 {
        gpu.launch_kernel_blob(FUNC, grid, block, 0, &mut b).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let mut samples = Vec::new();
    for _ in 0..7 {
        let t0 = Instant::now();
        for _ in 0..inner {
            gpu.launch_kernel_blob(FUNC, grid, block, 0, &mut b).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        samples.push(t0.elapsed().as_secs_f64() * 1e6 / inner as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (samples[0], samples[samples.len() / 2])
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("================================================================");
    println!("bench_mq3g256_lloyd_down_batched");
    println!("  kernel : {FUNC}");
    println!("  source : kernels/src/{MODULE}.hip");
    println!("  NOTE   : HIP-dispatch MICROBENCHMARK. Triage filter for gross");
    println!("           defects only, NOT a verdict on the kernel. Host launch");
    println!("           latency masks device effects; PM4 replay can rank");
    println!("           differently. Final acceptance = golden bundle");
    println!("           (registry/redline-golden-v1.json), HIP + PM4 arms.");
    println!("  COMBINE: this kernel SELF-COMBINES via atomicAdd into x_residual");
    println!("           ([N x M], in/out). It does NOT write an expanded buffer.");
    println!("           Epilogue quoted verbatim in the module doc comment.");
    println!("  FORMAT : MQ3-Lloyd qt=20 — 112 B/group-of-256 = 16 B (8 x fp16");
    println!("           codebook) + 96 B of 3-bit indices.");
    println!(
        "           = {:.4} B/weight = {:.3} bits/weight",
        GROUP_BYTES as f64 / GROUP as f64,
        GROUP_BYTES as f64 * 8.0 / GROUP as f64
    );
    println!("================================================================\n");

    self_check_packing();

    let mut gpu = Gpu::init().expect("GPU init");
    println!("arch = {}", gpu.arch);
    gpu.ensure_kernel_public(MODULE, SRC, FUNC)
        .expect("JIT gemv_mq3g256_lloyd_moe_down_indexed_batched_k4");

    let k_top = 8usize;
    let mut all_ok = true;

    // ---- primary a3b `down` shape: M=2048, K=512, k_top=8, n_experts=256 ----
    println!("\n### primary shape — building 256 synthetic experts (this is host-side)");
    let t0 = Instant::now();
    let primary = Case::new(&mut gpu, 2048, 512, k_top, 256);
    println!(
        "    built + uploaded 256 x {} KiB = {:.1} MiB of expert weights in {:.2} s",
        primary.expert_bytes / 1024,
        (256 * primary.expert_bytes) as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_secs_f64()
    );

    println!("\n########## CORRECTNESS — primary shape, N sweep ##########");
    for &(n, stride) in &[(1usize, 1usize), (4, 1), (16, 2), (64, 8)] {
        all_ok &= correctness(&mut gpu, &primary, n, stride, "primary");
    }

    // ---- path coverage: gpr selects the kernel's quad/tail split ----
    // primary K=512 -> gpr=2 -> quads=0, tail=2 (tail arms 0 and 1 only).
    // These cover the main quad loop and the third tail arm.
    println!("\n########## CORRECTNESS — quad/tail path coverage ##########");
    for &(m, k, tagname) in &[
        (256usize, 768usize, "gpr=3 quads=0 tail=3"),
        (256, 1280, "gpr=5 quads=1 tail=1"),
        (256, 2048, "gpr=8 quads=2 tail=0"),
    ] {
        let c = Case::new(&mut gpu, m, k, k_top, 16);
        all_ok &= correctness(&mut gpu, &c, 4, 1, tagname);
    }

    // ---- timing (triage) ----
    println!("\n########## TIMING (TRIAGE ONLY) — primary shape ##########");
    println!(
        "  M={} K={} k_top={} n_exp={}   block=(32,1,1)  grid=(M, K_TOP, N)",
        primary.m,
        primary.k,
        primary.k_top,
        primary.n_exp()
    );
    println!(
        "  weight bytes issued per launch = N * k_top * M * (K/256) * 112 = N * {:.2} MiB",
        (k_top * primary.expert_bytes) as f64 / (1024.0 * 1024.0)
    );
    println!("  (that is issued traffic, not unique bytes — with k_top=8 over 256 experts");
    println!("   there is real cache reuse across tokens at the larger N.)\n");

    println!(
        "  {:>4}  {:>10}  {:>10}  {:>11}  {:>11}  {:>12}  {:>10}",
        "N", "min us", "med us", "us/token", "tokens/s", "wt GB/s(med)", "pass"
    );
    for pass in 0..2 {
        for &n in &[1usize, 4, 16, 64] {
            let (mn, md) = time_case(&mut gpu, &primary, n);
            let wt_bytes = (n * k_top * primary.expert_bytes) as f64;
            let gbs = wt_bytes / (md * 1e-6) / 1e9;
            println!(
                "  {:>4}  {:>10.2}  {:>10.2}  {:>11.3}  {:>11.0}  {:>12.1}  {:>10}",
                n,
                mn,
                md,
                md / n as f64,
                n as f64 / (md * 1e-6),
                gbs,
                if pass == 0 { "0 (JIT*)" } else { "1" }
            );
        }
        if pass == 0 {
            println!("  * pass 0 is JIT- and cold-cache-contaminated BY CONSTRUCTION — discard it.");
            println!("    JIT is per-(kernel, shape) cell, so each N warms separately.\n");
        }
    }

    println!("\n================================================================");
    println!(
        "OVERALL: >>> {} <<<",
        if all_ok {
            "PASS — all correctness arms within tolerance"
        } else {
            "FAIL — at least one correctness arm exceeded tolerance"
        }
    );
    println!("================================================================");
    if !all_ok {
        std::process::exit(1);
    }
}
