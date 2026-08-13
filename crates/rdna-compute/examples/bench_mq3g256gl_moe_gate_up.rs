// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! MQ3-GL (qt=39) fused MoE gate_up GEMV — correctness triage + speed against
//! the per-block-codebook MQ3-Lloyd (qt=20) sibling, on the a3b routed-expert
//! gate_up decode shape (M=1024, K=2048, k_top=8).
//!
//! Kernels under test (JIT'd from source, launched through the kernarg-blob
//! path — this bench touches NO dispatch plumbing):
//!   * kernels/src/gemv_mq3g256gl_moe_gate_up_indexed.hip
//!       → gemv_mq3g256gl_moe_gate_up_k8_indexed
//!   * kernels/src/gemv_mq3g256_lloyd_moe_gate_up_indexed.hip   (speed baseline)
//!       → gemv_mq3g256_lloyd_moe_gate_up_k8_indexed
//!
//! ── REGIME CAVEAT — READ BEFORE QUOTING ANY NUMBER FROM THIS FILE ──────────
//! This is a HIP-dispatch microbenchmark. It is a TRIAGE FILTER for gross
//! defects — wrong numbers, pathological slowness — and NOT a verdict on a
//! kernel. Host launch latency masks device-level effects; the same kernel can
//! measure differently once lowered to retained PM4 replay. A prior HIP
//! microbench measured the MQ2GL kernel at -3.1% and that was nearly used to
//! kill the format. Final acceptance is the golden bundle
//! (registry/redline-golden-v1.json) with HIP and PM4 arms. Treat the timing
//! block below as "is it in the right order of magnitude", nothing more.
//! The CORRECTNESS block, by contrast, is load-bearing: it is an independent
//! decode of the on-disk byte layout.
//!
//! ── What the CPU reference independently derives ──────────────────────────
//! The high-risk area in MQ3 is the 3-bit unpack with its cross-byte straddle
//! (codes 2 and 5 span a byte boundary). To make sure a shared misunderstanding
//! cannot cancel out, three INDEPENDENT formulations of the same bit layout are
//! cross-checked:
//!   (a) GOLDEN — hard-coded 3-byte patterns for the 8 one-hot codes, written
//!       out by hand from the encoder spec `acc |= code_j << (3*j)`, then
//!       little-endian 3 bytes (STRADDLE_GOLDEN below).
//!   (b) PACKER — the synthetic-weight writer builds bytes with explicit
//!       byte-level masks taken from the straddle TABLE (b0/b1/b2 formulas),
//!       never from the 24-bit-word idiom the kernel uses.
//!   (c) READER — the CPU reference decodes the 96-byte index region as a flat
//!       LSB-first BITSTREAM, pulling each code one bit at a time from
//!       bit offset 3*w. It never builds a 24-bit word.
//! The kernel is a fourth formulation (build a 24-bit LE word, shift by 3*j).
//! (a) pins (b) and (c); (c) then judges the kernel. All 8 in-span positions
//! are additionally probed ON THE GPU, one per top-k rank, so a permuted or
//! straddle-broken slice shows up per position rather than as one blurred sum.
//!
//! ── Format spec used (authoritative, from the encoder) ─────────────────────
//! MQ3GL (qt=39), SoA, per expert blob, gpr = K/256:
//!   [0 .. M*gpr*96)              packed 3-bit indices, 96 B/group
//!   [M*gpr*96 .. +M*gpr*2)       fp16 per-block scales, 2 B/group
//!   both regions row-major in (row, group); codebook = 8 SCALAR kernel args.
//!   w = scale_block * cb[code]. 3.0625 bpw.
//! MQ3-Lloyd (qt=20), AoS, 112 B/group:
//!   [0..16)   8 × fp16 codebook (ascending)
//!   [16..112) 96 B packed 3-bit indices (same bitstream convention)
//!   w = cb_block[code]. 3.5 bpw.
//!
//! x is FWHT-256 PRE-ROTATED by the caller for both formats. Neither kernel
//! rotates. This bench therefore feeds an ARBITRARY deterministic x and uses
//! the same bytes for the GPU launch and the CPU reference — the rotation is
//! out of scope here and is not exercised.
//!
//! Run: cargo run --release -p rdna-compute --example bench_mq3g256gl_moe_gate_up
use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const MQ3GL_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256gl_moe_gate_up_indexed.hip");
const MQ3L_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_gate_up_indexed.hip");

const MQ3GL_FN: &str = "gemv_mq3g256gl_moe_gate_up_k8_indexed";
const MQ3L_FN: &str = "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed";

/// Global 3-bit Lloyd–Max codebook for a unit Gaussian (GL_CB3 in
/// crates/hipfire-quantize/src/main.rs). Passed to the kernel as 8 scalar args.
const CB3: [f32; 8] = [
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
];

/// Sentinel written into both output buffers before every correctness launch.
/// Any slot still holding it after the launch was never written by the kernel.
const SENTINEL: f32 = -123456789.0;

/// Relative tolerance for the PASS/FAIL gate.
const TOL: f64 = 1e-4;

// ── bit-layout ground truth ───────────────────────────────────────────────
/// The 3 bytes produced by a chunk whose only non-zero code is `code_j = 7`,
/// for j = 0..8. Hand-derived from the encoder spec:
///     acc = 7 << (3*j);  bytes = [acc & 0xFF, acc >> 8, acc >> 16]
/// j = 2 and j = 5 straddle a byte boundary — those two rows are the whole
/// reason this table is written out literally instead of being computed.
const STRADDLE_GOLDEN: [[u8; 3]; 8] = [
    [0x07, 0x00, 0x00], // j=0 : 0x000007
    [0x38, 0x00, 0x00], // j=1 : 0x000038
    [0xC0, 0x01, 0x00], // j=2 : 0x0001C0  ← straddles b0/b1
    [0x00, 0x0E, 0x00], // j=3 : 0x000E00
    [0x00, 0x70, 0x00], // j=4 : 0x007000
    [0x00, 0x80, 0x03], // j=5 : 0x038000  ← straddles b1/b2
    [0x00, 0x00, 0x1C], // j=6 : 0x1C0000
    [0x00, 0x00, 0xE0], // j=7 : 0xE00000
];

/// Formulation (b): pack 8 three-bit codes into 3 bytes using the byte-level
/// straddle TABLE from the kernel header, not the 24-bit accumulator idiom.
///     q0 = b0 & 7                      q4 = (b1 >> 4) & 7
///     q1 = (b0 >> 3) & 7               q5 = ((b1 >> 7) | (b2 << 1)) & 7
///     q2 = ((b0 >> 6) | (b1 << 2)) & 7 q6 = (b2 >> 2) & 7
///     q3 = (b1 >> 1) & 7               q7 = (b2 >> 5) & 7
#[inline]
fn pack3(q: &[u8; 8]) -> [u8; 3] {
    let b0 = q[0] | (q[1] << 3) | ((q[2] & 0x3) << 6);
    let b1 = (q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | ((q[5] & 0x1) << 7);
    let b2 = (q[5] >> 1) | (q[6] << 2) | (q[7] << 5);
    [b0, b1, b2]
}

/// Formulation (c): read code `w` of a 96-byte index region as bits
/// [3*w .. 3*w+3) of a flat LSB-first bitstream, one bit at a time.
/// Deliberately does NOT build a 24-bit word.
#[inline]
fn bits3(region: &[u8], w: usize) -> usize {
    let bit = 3 * w;
    let mut v = 0usize;
    for b in 0..3 {
        let p = bit + b;
        v |= (((region[p >> 3] >> (p & 7)) & 1) as usize) << b;
    }
    v
}

// ── small utilities ───────────────────────────────────────────────────────
fn mix(x: u64) -> u64 {
    let h = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

/// f32 → fp16 bits, truncating mantissa. Only used to MANUFACTURE synthetic
/// blob bytes; every CPU reference reads the stored u16 back with
/// `half_to_f32`, so this converter's rounding never enters the comparison.
/// All values fed here are in [1e-3, 1e-1] — normal fp16 range, no subnormals.
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

#[inline]
fn put_u16(dst: &mut [u8], off: usize, v: u16) {
    dst[off] = (v & 0xff) as u8;
    dst[off + 1] = (v >> 8) as u8;
}

/// Deterministic per-(row, group) fp16 scale, in a comfortable normal range.
/// Depends on BOTH row and group so a wrong scale-region stride shows up.
fn block_scale(seed: u64, row: usize, g: usize) -> u16 {
    let r = mix(seed ^ ((row as u64) << 20) ^ (g as u64 + 0x9e37)) % 5000;
    half_bits(0.003f32 + (r as f32) * 1e-6)
}

// ── synthetic blob builders ───────────────────────────────────────────────
/// Pseudo-random 8 codes for chunk (row, g, c). Same stream feeds both formats
/// so MQ3GL and MQ3L encode comparable weights (they still differ: different
/// codebook source).
#[inline]
fn codes_for(seed: u64, row: usize, g: usize, c: usize) -> [u8; 8] {
    let h = mix(seed ^ ((row as u64) << 32) ^ ((g as u64) << 12) ^ (c as u64));
    let mut q = [0u8; 8];
    for (j, qj) in q.iter_mut().enumerate() {
        *qj = ((h >> (3 * j)) & 7) as u8;
    }
    q
}

/// MQ3GL (qt=39) SoA blob: [M*gpr*96 indices][M*gpr*2 fp16 scales].
fn build_mq3gl(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let idx_bytes = m * gpr * 96;
    let mut out = vec![0u8; idx_bytes + m * gpr * 2];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * 96;
            for c in 0..32 {
                let bs = pack3(&codes_for(seed, row, g, c));
                out[off + 3 * c..off + 3 * c + 3].copy_from_slice(&bs);
            }
            put_u16(&mut out, idx_bytes + (row * gpr + g) * 2, block_scale(seed, row, g));
        }
    }
    out
}

/// MQ3GL probe blob: in EVERY chunk, position `pos` carries code 7 and every
/// other position carries code 0. Combined with a position-dependent x this
/// isolates one of the 8 in-span slots per launch.
fn build_mq3gl_probe(m: usize, k: usize, pos: usize) -> Vec<u8> {
    let gpr = k / 256;
    let idx_bytes = m * gpr * 96;
    let mut out = vec![0u8; idx_bytes + m * gpr * 2];
    let mut q = [0u8; 8];
    q[pos] = 7;
    let bs = pack3(&q);
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * 96;
            for c in 0..32 {
                out[off + 3 * c..off + 3 * c + 3].copy_from_slice(&bs);
            }
            put_u16(&mut out, idx_bytes + (row * gpr + g) * 2, block_scale(0xB0BA, row, g));
        }
    }
    out
}

/// MQ3-Lloyd (qt=20) AoS blob: per group [8×fp16 cb][96 B indices] = 112 B.
/// Codebook = CB3 scaled by the same per-block scale MQ3GL stores separately,
/// so the two formats represent near-identical weights.
fn build_mq3l(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let mut out = vec![0u8; m * gpr * 112];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * 112;
            let s = half_to_f32(block_scale(seed, row, g));
            for (e, &c) in CB3.iter().enumerate() {
                put_u16(&mut out, off + 2 * e, half_bits(c * s));
            }
            for c in 0..32 {
                let bs = pack3(&codes_for(seed, row, g, c));
                out[off + 16 + 3 * c..off + 16 + 3 * c + 3].copy_from_slice(&bs);
            }
        }
    }
    out
}

// ── CPU references (formulation (c): flat LSB-first bitstream) ────────────
/// Returns (dot, Σ|term|). Σ|term| is the conditioning of the dot product and
/// is the honest denominator for a relative-error gate under cancellation.
fn ref_mq3gl_row(raw: &[u8], m: usize, k: usize, row: usize, x: &[f32]) -> (f64, f64) {
    let gpr = k / 256;
    let idx_bytes = m * gpr * 96;
    let mut dot = 0.0f64;
    let mut sab = 0.0f64;
    for g in 0..gpr {
        let so = idx_bytes + (row * gpr + g) * 2;
        let scale = half_to_f32(u16::from_le_bytes([raw[so], raw[so + 1]])) as f64;
        let base = (row * gpr + g) * 96;
        let region = &raw[base..base + 96];
        for w in 0..256 {
            let code = bits3(region, w);
            let t = scale * CB3[code] as f64 * x[g * 256 + w] as f64;
            dot += t;
            sab += t.abs();
        }
    }
    (dot, sab)
}

fn ref_mq3l_row(raw: &[u8], k: usize, row: usize, x: &[f32]) -> (f64, f64) {
    let gpr = k / 256;
    let mut dot = 0.0f64;
    let mut sab = 0.0f64;
    for g in 0..gpr {
        let off = (row * gpr + g) * 112;
        let mut cb = [0.0f64; 8];
        for (e, ce) in cb.iter_mut().enumerate() {
            *ce = half_to_f32(u16::from_le_bytes([raw[off + 2 * e], raw[off + 2 * e + 1]])) as f64;
        }
        let region = &raw[off + 16..off + 112];
        for w in 0..256 {
            let code = bits3(region, w);
            let t = cb[code] * x[g * 256 + w] as f64;
            dot += t;
            sab += t.abs();
        }
    }
    (dot, sab)
}

// ── phase A: host-side bit-layout self test ───────────────────────────────
fn straddle_self_test() -> bool {
    let mut ok = true;
    println!("── phase A ── host bit-layout self test (no GPU)");
    println!("   one-hot code_j = 7, all 8 positions, packer vs hand-written golden vs bit reader");
    for j in 0..8 {
        let mut q = [0u8; 8];
        q[j] = 7;
        let got = pack3(&q);
        let want = STRADDLE_GOLDEN[j];
        let bytes_ok = got == want;
        // the reader must recover 7 at j and 0 everywhere else in the chunk
        let mut read_ok = true;
        for (w, item) in q.iter().enumerate() {
            if bits3(&want, w) != *item as usize {
                read_ok = false;
            }
        }
        let straddle = j == 2 || j == 5;
        println!(
            "   j={j}{} pack={:02X?} golden={:02X?} bytes={} reader={}",
            if straddle { " (straddles byte boundary)" } else { "                          " },
            got,
            want,
            if bytes_ok { "ok " } else { "BAD" },
            if read_ok { "ok " } else { "BAD" },
        );
        ok &= bytes_ok && read_ok;
    }
    // exhaustive round-trip over every 3-byte value: packer ∘ reader == identity
    let mut rt_ok = true;
    for v in 0u32..(1 << 24) {
        let bytes = [(v & 0xff) as u8, ((v >> 8) & 0xff) as u8, ((v >> 16) & 0xff) as u8];
        let mut q = [0u8; 8];
        for (j, qj) in q.iter_mut().enumerate() {
            *qj = bits3(&bytes, j) as u8;
        }
        if pack3(&q) != bytes {
            rt_ok = false;
            println!("   ROUND-TRIP BAD at 0x{v:06X} -> {:02X?}", pack3(&q));
            break;
        }
    }
    println!(
        "   exhaustive 2^24 round-trip (reader then packer == identity): {}",
        if rt_ok { "ok" } else { "BAD" }
    );
    ok &= rt_ok;
    println!(
        "   phase A: {}\n",
        if ok { "PASS" } else { "FAIL — bit layout disagreement on the HOST; GPU results below are meaningless" }
    );
    ok
}

// ── error bookkeeping ─────────────────────────────────────────────────────
struct Sample {
    row: usize,
    krank: usize,
    side: &'static str,
    got: f64,
    want: f64,
    sab: f64,
}

struct Stats {
    max_abs: f64,
    max_rel: f64,
    max_rel_cond: f64,
    worst_rel: String,
    worst_cond: String,
    n_well: usize,
    n_total: usize,
    n_nonfinite: usize,
    sentinel_left: usize,
    min_gate_up_gap: f64,
    pass: bool,
}

fn analyze(samples: &[Sample], sentinel_left: usize, min_gate_up_gap: f64) -> Stats {
    let rms = (samples.iter().map(|s| s.want * s.want).sum::<f64>() / samples.len() as f64).sqrt();
    let floor = 0.1 * rms; // "well conditioned" cut for the plain relative metric
    let mut st = Stats {
        max_abs: 0.0,
        max_rel: 0.0,
        max_rel_cond: 0.0,
        worst_rel: "-".into(),
        worst_cond: "-".into(),
        n_well: 0,
        n_total: samples.len(),
        n_nonfinite: 0,
        sentinel_left,
        min_gate_up_gap,
        pass: false,
    };
    for s in samples {
        // NaN/Inf never wins a `>` comparison, so count it explicitly or a
        // kernel emitting NaN would silently report max_abs = 0.
        if !s.got.is_finite() {
            st.n_nonfinite += 1;
        }
        let e = (s.got - s.want).abs();
        if e > st.max_abs {
            st.max_abs = e;
        }
        let rc = e / s.sab.max(1e-30);
        if rc > st.max_rel_cond {
            st.max_rel_cond = rc;
            st.worst_cond = format!(
                "row={} krank={} {} got={:.6e} want={:.6e} Σ|term|={:.4e}",
                s.row, s.krank, s.side, s.got, s.want, s.sab
            );
        }
        if s.want.abs() >= floor {
            st.n_well += 1;
            let r = e / s.want.abs();
            if r > st.max_rel {
                st.max_rel = r;
                st.worst_rel = format!(
                    "row={} krank={} {} got={:.6e} want={:.6e} abs={:.3e}",
                    s.row, s.krank, s.side, s.got, s.want, e
                );
            }
        }
    }
    st.pass = st.max_rel <= TOL
        && st.max_rel_cond <= TOL
        && sentinel_left == 0
        && st.n_nonfinite == 0
        && min_gate_up_gap > 0.0
        && st.max_abs.is_finite();
    st
}

fn report(label: &str, st: &Stats) {
    println!("   {label}");
    println!("      max abs err                    = {:.6e}", st.max_abs);
    println!(
        "      max rel err (|err|/|want|)     = {:.3e}   over {}/{} well-conditioned outputs",
        st.max_rel, st.n_well, st.n_total
    );
    println!("        worst: {}", st.worst_rel);
    println!(
        "      max rel err (|err|/Σ|term|)    = {:.3e}   cancellation-aware, all {} outputs",
        st.max_rel_cond, st.n_total
    );
    println!("        worst: {}", st.worst_cond);
    println!(
        "      unwritten output slots         = {}   (0 = kernel covered every gate+up slot)",
        st.sentinel_left
    );
    println!(
        "      non-finite outputs             = {}   (NaN/Inf never wins a > compare, counted separately)",
        st.n_nonfinite
    );
    println!(
        "      min |y_gate[i] - y_up[i]|      = {:.6e}   (>0 = the two halves are NOT aliased)",
        st.min_gate_up_gap
    );
    println!(
        "      >>> {} <<<  tol {:.0e} relative",
        if st.pass { "PASS" } else { "FAIL" },
        TOL
    );
}

// ── GPU plumbing ──────────────────────────────────────────────────────────
struct Uploaded {
    _experts: Vec<GpuTensor>,
    _ptr_tab: GpuTensor,
    _topk: GpuTensor,
    _x: GpuTensor,
    y_g: GpuTensor,
    y_u: GpuTensor,
    args: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn upload(
    gpu: &mut Gpu,
    blobs: &[Vec<u8>],
    topk: &[i32],
    x: &[f32],
    m: usize,
    k: usize,
    with_codebook: bool,
) -> Uploaded {
    let k_top = topk.len();
    let mi = m / 2;
    let experts: Vec<GpuTensor> = blobs
        .iter()
        .map(|b| gpu.upload_raw(b, &[b.len()]).unwrap())
        .collect();
    let ptrs: Vec<u8> = experts
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    let ptr_tab = gpu.upload_raw(&ptrs, &[blobs.len()]).unwrap();
    let topk_bytes: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_bytes, &[k_top]).unwrap();
    let x_t = gpu.upload_f32(x, &[k]).unwrap();
    let y_g = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();

    let mut blob = KernargBlob::new();
    blob.push_ptr(ptr_tab.buf.as_ptr() as *const _);
    blob.push_ptr(topk_t.buf.as_ptr() as *const _);
    blob.push_ptr(x_t.buf.as_ptr() as *const _);
    blob.push_ptr(y_g.buf.as_ptr() as *const _);
    blob.push_ptr(y_u.buf.as_ptr() as *const _);
    if with_codebook {
        for c in CB3 {
            blob.push_f32(c);
        }
    }
    blob.push_i32(m as i32);
    blob.push_i32(k as i32);

    Uploaded {
        _experts: experts,
        _ptr_tab: ptr_tab,
        _topk: topk_t,
        _x: x_t,
        y_g,
        y_u,
        args: blob.into_vec(),
    }
}

/// Launch once with both outputs sentinel-filled, then check every element of
/// both halves against the CPU reference. Explicitly verifies the gate/up
/// split: row < M/2 → y_gate[krank*mi + row], row >= M/2 → y_up[krank*mi + row-mi].
#[allow(clippy::too_many_arguments)]
fn check(
    gpu: &mut Gpu,
    func: &str,
    up: &mut Uploaded,
    blobs: &[Vec<u8>],
    topk: &[i32],
    x: &[f32],
    m: usize,
    k: usize,
    gl: bool,
) -> Stats {
    let k_top = topk.len();
    let mi = m / 2;
    gpu.fill_f32(&up.y_g, SENTINEL).unwrap();
    gpu.fill_f32(&up.y_u, SENTINEL).unwrap();
    let grid = [m as u32, k_top as u32, 1];
    let block = [32u32, 1, 1];
    gpu.launch_kernel_blob(func, grid, block, 0, &mut up.args)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let gg = gpu.download_f32(&up.y_g).unwrap();
    let gu = gpu.download_f32(&up.y_u).unwrap();

    let sentinel_left = gg.iter().chain(gu.iter()).filter(|v| **v == SENTINEL).count();
    let min_gap = gg
        .iter()
        .zip(gu.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(f64::INFINITY, f64::min);

    let mut samples = Vec::with_capacity(k_top * m);
    for (krank, &e) in topk.iter().enumerate() {
        let raw = &blobs[e as usize];
        for row in 0..m {
            let (want, sab) = if gl {
                ref_mq3gl_row(raw, m, k, row, x)
            } else {
                ref_mq3l_row(raw, k, row, x)
            };
            let (got, side) = if row < mi {
                (gg[krank * mi + row] as f64, "y_gate")
            } else {
                (gu[krank * mi + (row - mi)] as f64, "y_up")
            };
            samples.push(Sample { row, krank, side, got, want, sab });
        }
    }
    analyze(&samples, sentinel_left, min_gap)
}

/// Split-halves breakdown so the gate/up routing is reported, not just implied.
fn split_report(
    blobs: &[Vec<u8>],
    topk: &[i32],
    x: &[f32],
    m: usize,
    k: usize,
    gl: bool,
    gg: &[f32],
    gu: &[f32],
) {
    let mi = m / 2;
    let mut worst_g = 0.0f64;
    let mut worst_u = 0.0f64;
    for (krank, &e) in topk.iter().enumerate() {
        let raw = &blobs[e as usize];
        for row in 0..m {
            let (want, sab) = if gl {
                ref_mq3gl_row(raw, m, k, row, x)
            } else {
                ref_mq3l_row(raw, k, row, x)
            };
            let (got, dst) = if row < mi {
                (gg[krank * mi + row] as f64, &mut worst_g)
            } else {
                (gu[krank * mi + (row - mi)] as f64, &mut worst_u)
            };
            let r = (got - want).abs() / sab.max(1e-30);
            if r > *dst {
                *dst = r;
            }
        }
    }
    println!(
        "      split: rows [0,{mi}) -> y_gate  max rel(cond) {worst_g:.3e}   |   rows [{mi},{m}) -> y_up  max rel(cond) {worst_u:.3e}"
    );
}

// ── timing ────────────────────────────────────────────────────────────────
fn time_us(gpu: &mut Gpu, func: &str, up: &mut Uploaded, grid: [u32; 3], iters: usize) -> f64 {
    let block = [32u32, 1, 1];
    let t0 = Instant::now();
    for _ in 0..iters {
        gpu.launch_kernel_blob(func, grid, block, 0, &mut up.args)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn main() {
    let host_ok = straddle_self_test();

    let mut gpu = Gpu::init().expect("GPU init");
    println!("arch={}\n", gpu.arch);

    gpu.ensure_kernel_public("gemv_mq3g256gl_moe_gate_up_indexed", MQ3GL_SRC, MQ3GL_FN)
        .expect("JIT mq3gl");
    gpu.ensure_kernel_public("gemv_mq3g256_lloyd_moe_gate_up_indexed", MQ3L_SRC, MQ3L_FN)
        .expect("JIT mq3l");

    // a3b routed-expert gate_up decode shape.
    let m = 1024usize; // 2 * moe_intermediate (gate | up split)
    let k = 2048usize; // hidden
    let k_top = 8usize;
    let n_exp = 32usize; // real a3b has 256; 32 is enough to exercise indexing
    let gpr = k / 256;
    let mi = m / 2;

    let gl_bytes = m * gpr * 98; // 96 idx + 2 scale per group
    let l_bytes = m * gpr * 112;
    println!("shape: M={m} (gate rows [0,{mi}) | up rows [{mi},{m})) K={k} k_top={k_top} n_exp={n_exp} gpr={gpr}");
    println!(
        "per-expert weight bytes: MQ3GL {} KiB ({:.4} B/weight = {:.4} bpw)   MQ3L {} KiB ({:.4} B/weight = {:.4} bpw)   MQ3GL is {:.1}% smaller\n",
        gl_bytes / 1024,
        98.0 / 256.0,
        98.0 * 8.0 / 256.0,
        l_bytes / 1024,
        112.0 / 256.0,
        112.0 * 8.0 / 256.0,
        100.0 * (1.0 - gl_bytes as f64 / l_bytes as f64)
    );

    // ── phase B: on-GPU straddle probe, one code position per top-k rank ──
    println!("── phase B ── GPU straddle probe: expert p has code 7 only at in-span position p");
    println!("   x[col] = 0.125*((col%8)+1) so every one of the 8 positions carries a distinct weight;");
    println!("   a permuted / straddle-broken slice changes the answer for that position only.");
    let probe_blobs: Vec<Vec<u8>> = (0..8).map(|p| build_mq3gl_probe(m, k, p)).collect();
    let probe_topk: Vec<i32> = (0..8).collect();
    let probe_x: Vec<f32> = (0..k).map(|i| 0.125 * ((i % 8) + 1) as f32).collect();
    let mut up_probe = upload(&mut gpu, &probe_blobs, &probe_topk, &probe_x, m, k, true);
    let st_probe = check(
        &mut gpu,
        MQ3GL_FN,
        &mut up_probe,
        &probe_blobs,
        &probe_topk,
        &probe_x,
        m,
        k,
        true,
    );
    let pg = gpu.download_f32(&up_probe.y_g).unwrap();
    let pu = gpu.download_f32(&up_probe.y_u).unwrap();
    for p in 0..8 {
        let raw = &probe_blobs[p];
        let (want, sab) = ref_mq3gl_row(raw, m, k, 0, &probe_x);
        let got = pg[p * mi] as f64;
        let rel = (got - want).abs() / sab.max(1e-30);
        println!(
            "   pos {p}{} row0 gpu={got:>13.6} cpu={want:>13.6} rel(cond)={rel:.2e} {}",
            if p == 2 || p == 5 { " STRADDLE" } else { "         " },
            if rel <= TOL { "ok" } else { "BAD" }
        );
    }
    // the probe only has power if the 8 positions actually produce distinct sums
    let mut distinct = true;
    for a in 0..8 {
        for b in (a + 1)..8 {
            if (pg[a * mi] - pg[b * mi]).abs() < 1e-6 {
                distinct = false;
            }
        }
    }
    println!(
        "   positions produce distinct sums: {}  (if not, the probe has no discriminating power)",
        if distinct { "yes" } else { "NO — probe is vacuous" }
    );
    split_report(&probe_blobs, &probe_topk, &probe_x, m, k, true, &pg, &pu);
    report("MQ3GL straddle probe", &st_probe);
    println!();

    // ── phase C: full random correctness on the a3b gate_up shape ─────────
    println!("── phase C ── MQ3GL vs independent CPU reference, random codes, full M×k_top sweep");
    let x: Vec<f32> = (0..k)
        .map(|i| ((mix(0x5EED ^ i as u64) % 20001) as f32 - 10000.0) * 1e-4)
        .collect();
    let gl_blobs: Vec<Vec<u8>> = (0..n_exp)
        .map(|e| build_mq3gl(m, k, 0x1000 + e as u64))
        .collect();
    let l_blobs: Vec<Vec<u8>> = (0..n_exp)
        .map(|e| build_mq3l(m, k, 0x1000 + e as u64))
        .collect();
    let topk: Vec<i32> = (0..k_top as i32).map(|i| (i * 3) % n_exp as i32).collect();
    println!("   topk_indices = {topk:?}");

    let mut up_gl = upload(&mut gpu, &gl_blobs, &topk, &x, m, k, true);
    let st_gl = check(&mut gpu, MQ3GL_FN, &mut up_gl, &gl_blobs, &topk, &x, m, k, true);
    let gg = gpu.download_f32(&up_gl.y_g).unwrap();
    let gu = gpu.download_f32(&up_gl.y_u).unwrap();
    split_report(&gl_blobs, &topk, &x, m, k, true, &gg, &gu);
    report("MQ3GL  M=1024 K=2048 k_top=8", &st_gl);
    println!();

    // ── phase C2: tail-path coverage (gpr % 4 != 0) ──────────────────────
    println!("── phase C2 ── MQ3GL tail path (the K4 loop leaves 1..3 groups to the TAIL macro)");
    let mut tail_pass = true;
    for &kt in &[768usize, 1280usize] {
        let mt = 64usize;
        let ktop = 2usize;
        let nexp = 4usize;
        let g = kt / 256;
        let xt: Vec<f32> = (0..kt)
            .map(|i| ((mix(0xC0FFEE ^ i as u64) % 20001) as f32 - 10000.0) * 1e-4)
            .collect();
        let blobs: Vec<Vec<u8>> = (0..nexp).map(|e| build_mq3gl(mt, kt, 0x77 + e as u64)).collect();
        let tk: Vec<i32> = (0..ktop as i32).map(|i| i * 2).collect();
        let mut u = upload(&mut gpu, &blobs, &tk, &xt, mt, kt, true);
        let s = check(&mut gpu, MQ3GL_FN, &mut u, &blobs, &tk, &xt, mt, kt, true);
        println!(
            "   K={kt} (gpr={g}: quads={} tail={})  max rel {:.3e}  rel(cond) {:.3e}  unwritten {}  -> {}",
            g / 4,
            g % 4,
            s.max_rel,
            s.max_rel_cond,
            s.sentinel_left,
            if s.pass { "PASS" } else { "FAIL" }
        );
        tail_pass &= s.pass;
    }
    println!();

    // ── phase D: MQ3-Lloyd baseline correctness ──────────────────────────
    println!("── phase D ── MQ3-Lloyd (speed baseline) vs its own independent CPU reference");
    let mut up_l = upload(&mut gpu, &l_blobs, &topk, &x, m, k, false);
    let st_l = check(&mut gpu, MQ3L_FN, &mut up_l, &l_blobs, &topk, &x, m, k, false);
    let lg = gpu.download_f32(&up_l.y_g).unwrap();
    let lu = gpu.download_f32(&up_l.y_u).unwrap();
    split_report(&l_blobs, &topk, &x, m, k, false, &lg, &lu);
    report("MQ3L   M=1024 K=2048 k_top=8", &st_l);
    println!();

    // ── phase E: timing ──────────────────────────────────────────────────
    println!("── phase E ── timing (see REGIME CAVEAT in the file header — triage only)");
    println!("   the FIRST pass of any kernel shape is JIT-contaminated; the warmup pass below");
    println!("   is thrown away for exactly that reason. Numbers are host-launch bound.");
    let grid = [m as u32, k_top as u32, 1];
    let iters = 200usize;
    // throwaway warmup for BOTH cells (JIT is per-(config × kernel-shape))
    let w_gl = time_us(&mut gpu, MQ3GL_FN, &mut up_gl, grid, 50);
    let w_l = time_us(&mut gpu, MQ3L_FN, &mut up_l, grid, 50);
    println!("   warmup (DISCARDED, JIT-contaminated): MQ3GL {w_gl:.2} us  MQ3L {w_l:.2} us");

    let reps = 7usize;
    let mut t_gl = Vec::with_capacity(reps);
    let mut t_l = Vec::with_capacity(reps);
    for _ in 0..reps {
        t_gl.push(time_us(&mut gpu, MQ3GL_FN, &mut up_gl, grid, iters));
        t_l.push(time_us(&mut gpu, MQ3L_FN, &mut up_l, grid, iters));
    }
    t_gl.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_l.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (min_gl, med_gl) = (t_gl[0], t_gl[reps / 2]);
    let (min_l, med_l) = (t_l[0], t_l[reps / 2]);

    let bytes_gl = (k_top * gl_bytes) as f64;
    let bytes_l = (k_top * l_bytes) as f64;
    let gbs = |b: f64, us: f64| b / (us * 1e-6) / 1e9;
    println!(
        "\n   {:<7} {:>10} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "variant", "min us", "med us", "B/weight", "wt B/call", "GB/s (med)", "1/med kHz"
    );
    println!(
        "   {:<7} {:>10.2} {:>10.2} {:>12.4} {:>12.0} {:>12.1} {:>10.1}",
        "MQ3GL",
        min_gl,
        med_gl,
        98.0 / 256.0,
        bytes_gl,
        gbs(bytes_gl, med_gl),
        1e3 / med_gl
    );
    println!(
        "   {:<7} {:>10.2} {:>10.2} {:>12.4} {:>12.0} {:>12.1} {:>10.1}",
        "MQ3L",
        min_l,
        med_l,
        112.0 / 256.0,
        bytes_l,
        gbs(bytes_l, med_l),
        1e3 / med_l
    );
    println!(
        "   MQ3GL vs MQ3L: {:+.2}% on median time, {:+.2}% on min time, {:.1}% fewer weight bytes",
        100.0 * (med_gl / med_l - 1.0),
        100.0 * (min_gl / min_l - 1.0),
        100.0 * (1.0 - bytes_gl / bytes_l)
    );
    println!(
        "   tok-shaped: at {med_gl:.2} us/call this one gate_up call caps a decode step at {:.0} steps/s",
        1e6 / med_gl
    );
    println!("   raw reps (us): MQ3GL {t_gl:.2?}  MQ3L {t_l:.2?}");

    // ── verdict ──────────────────────────────────────────────────────────
    let all = host_ok && st_probe.pass && st_gl.pass && tail_pass && st_l.pass;
    println!(
        "\n>>> OVERALL {} <<<   phaseA(bits)={} phaseB(straddle)={} phaseC(mq3gl)={} phaseC2(tail)={} phaseD(mq3l)={}",
        if all { "PASS" } else { "FAIL" },
        host_ok,
        st_probe.pass,
        st_gl.pass,
        tail_pass,
        st_l.pass
    );
    println!("(triage filter only — final acceptance is the golden bundle registry/redline-golden-v1.json, HIP + PM4 arms)");
    if !all {
        std::process::exit(1);
    }
}
