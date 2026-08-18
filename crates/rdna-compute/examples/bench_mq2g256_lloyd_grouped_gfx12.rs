// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness + triage timing for `gemm_mq2g256_lloyd_moe_grouped_wmma_gfx12`
//! (`kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma.gfx12.hip`) — the gfx12
//! (RDNA4) grouped-WMMA MQ2-Lloyd MoE prefill GEMM.
//!
//! **gfx1200 / gfx1201 ONLY.** The kernel calls
//! `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, which does not exist on
//! gfx10/gfx11 — so this bench detects the arch at runtime and *skips cleanly*
//! (exit 0, loud message) rather than failing a JIT on the wrong box. The gfx11
//! sibling is `gemm_mq2g256_lloyd_moe_grouped_wmma_k2`.
//!
//! ## REGIME CAVEAT — read before quoting a number from this file
//!
//! This is a **HIP-dispatch microbenchmark**. It is a **TRIAGE FILTER for gross
//! defects** — wrong numbers, group-boundary bugs, pathological slowness — and
//! **NOT a verdict on the kernel**. Host launch latency masks device-level
//! effects, and the same kernel can measure differently once lowered to
//! retained PM4 replay. A prior HIP microbench measured the MQ2GL kernel at
//! -3.1% and that number was nearly used to kill the format. **Final acceptance
//! is the golden bundle (`registry/redline-golden-v1.json`) with HIP and PM4
//! arms.** Nothing here is acceptance evidence; a green PASS here only means
//! "not obviously broken, worth measuring properly".
//!
//! Also: the first dispatch of any shape is JIT/cold-cache contaminated. The
//! reported MIN/MEDIAN come from post-warmup reps only, and the output says so.
//!
//! ## What is checked
//!
//! Four phases per shape, all against host-derived references.
//!
//! * **Phase A — expert-isolation probe (FULL coverage, exact).** Every expert
//!   `e` gets a degenerate weight blob whose 4 codebook entries are all
//!   `(e+1)/256`, so `W_e[r][c] == (e+1)/256` for *every* row and column
//!   regardless of the 2-bit index bits (the kernel's bilinear coefficients all
//!   collapse to exactly 0 in that case). X is all-positive. Then
//!   `Y[slot][row] == (e+1)/256 * sum_c X[x_row][c]`, and the *recovered*
//!   expert id `round(Y/sum_x*256) - 1` must equal `expert_tile_ids[slot/16]`
//!   for **every one of the `m_total x M` outputs**. This is the classic-bug
//!   detector: a token routed to expert `e` that gets multiplied by expert
//!   `e'`'s weights shows up as an integer-off expert id, not as a 1e-3 numeric
//!   wobble. It also verifies (a) intra-tile padding slots
//!   (`sorted_slot_index[s] < 0`) write exactly `0.0`, and (b) sentinel tiles
//!   (`expert_tile_ids[t] < 0`) are NOT written at all (poison survives).
//!
//! * **Phase B1 — numeric parity vs an independently-derived CPU reference
//!   (GATING).** The CPU reference decodes straight from the FORMAT SPEC —
//!   "72 B/group = 4 x fp16 codebook then 64 B of 2-bit indices, 4 per byte,
//!   LSB-first", i.e. element `c` of a 256-group lives at bit `2*c` — and does a
//!   flat `sum_c cb[q_c] * x_c` in f64. It deliberately does **not** mirror the
//!   kernel's structure (the kernel reads one 4 B dword per 16-wide K-tile,
//!   shifts by `k_grp*16` to take its own half, and reconstructs via the 2D
//!   bilinear expansion `cb0 + qlo*d01 + qhi*d02 + qlo*qhi*d_xor`). A shared
//!   misunderstanding of the bit order therefore cannot cancel out.
//!
//! * **Phase B2 — realistic Lloyd-Max codebook (INFORMATIONAL).** Same
//!   reference, but with a continuously-scaled textbook Lloyd-Max codebook.
//!   See "why two codebooks" below.
//!
//! * **Phase C — triage timing.** Warmup, then >= 5 timed reps; MIN and MEDIAN
//!   with us/call, GFLOP/s, streamed GB/s and bytes/element.
//!
//! ## Why two codebooks (this is the subtle part)
//!
//! The kernel does not evaluate `cb[q]`; it evaluates the algebraically
//! equivalent bilinear form `cb0 + qlo*(cb1-cb0) + qhi*(cb2-cb0) +
//! qlo*qhi*(cb3-cb2-cb1+cb0)` **in fp16** (that form is what makes the compiler
//! vectorize into WMMA at all). For an arbitrary fp16 codebook those
//! differences and partial sums are not exactly representable, so the
//! reconstructed weight can sit a few fp16 ulps off `cb[q]` — a known,
//! documented property of the whole MQ2-Lloyd kernel family (gfx11 `_k2` has
//! it too), **not** a defect, and worth ~1e-4..1e-3 of normalized output error
//! all by itself.
//!
//! To keep the gating check honest at a 1e-4 tolerance, **Phase B1 uses a
//! codebook that is fp16-exact under that decomposition**: every entry is
//! `u * n` for a power-of-two group scale `u` and small integers
//! `n in {-14, -4, 4, 18}`. Then `d01 = 10u`, `d02 = 18u`,
//! `d_xor = 4u` (deliberately non-zero, so the `qlo*qhi` term is exercised —
//! an affine codebook would zero it and hide a bug there), and every partial
//! sum is a small integer multiple of `u`, hence exact in fp16 whatever the
//! compiler's association or FMA choices. Any four fp16 values are a legal
//! MQ2-Lloyd codebook, so this is a spec-legal input, not a special case; the
//! chosen ratios `[-1.51, -0.43, 0.43, 1.94]` are close to real Lloyd-Max
//! levels. All four quantization levels are still exercised.
//!
//! Phase B2 then re-runs with an actual Lloyd-Max codebook to *quantify* that
//! inherent reconstruction deviation, gated only loosely (5e-3) so it still
//! catches gross defects without failing the build over known fp16 behaviour.
//!
//! ## Tolerance and the error metrics
//!
//! A zero-mean codebook makes `sum_c w_c x_c` a random walk, so an individual
//! output can cancel to near zero; pointwise relative error against such an
//! output is dominated by ordinary f32 accumulation order, not by kernel
//! correctness. The bench therefore gates on two metrics, both at 1e-4:
//!
//!   * `norm_err  = max|gpu-cpu| / rms(cpu)`  — error against the output scale
//!   * `max_rel   = max|gpu-cpu|/|cpu|` restricted to `|cpu| >= rms(cpu)`
//!     (the well-conditioned outputs, where relative error is meaningful)
//!
//! and additionally *reports* the unfloored pointwise max relative error over
//! all samples plus how many samples are ill-conditioned. Detection power is
//! unaffected by the floor: a wrong expert, wrong A-row, wrong bit order or an
//! off-by-one group moves an output by O(rms) -> ~1.0, four orders above the
//! gate; even a **single** wrong 2-bit index moves it by ~rms/sqrt(K)
//! (~1.6e-2 at K=4096), two orders above the gate.
//!
//! ## X and the FWHT rotation
//!
//! In production these formats see an FWHT-256 **pre-rotated** X. The rotation
//! is the caller's job and is irrelevant to this kernel's arithmetic, so the
//! bench feeds arbitrary deterministic pseudo-random fp16 X and uses the
//! *identical bit patterns* on both the GPU and CPU sides. Stated explicitly:
//! **X here is NOT rotated**, and that is fine because both arms consume the
//! same bytes.
//!
//! Run: `cargo run --release -p rdna-compute --example bench_mq2g256_lloyd_grouped_gfx12`

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const KERNEL_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma.gfx12.hip");
const KERNEL_NAME: &str = "gemm_mq2g256_lloyd_moe_grouped_wmma_gfx12";

/// Bytes per 256-element K-group for MQ2-Lloyd (qt=19): 8 B codebook + 64 B of
/// 2-bit indices. 72*8/256 = 2.25 bpw.
const GROUP_BYTES: usize = 72;

/// fp16-exact codebook integers (see "why two codebooks" in the module doc).
/// Entries are `u * CB_EXACT_INT[j]` for a power-of-two group scale `u`.
/// d01 = 10u, d02 = 18u, d_xor = 18-4+4-14 = 4u (non-zero on purpose).
const CB_EXACT_INT: [i32; 4] = [-14, -4, 4, 18];

/// Textbook Lloyd-Max levels for a unit Gaussian at 2 bit. A globally fitted
/// codebook over real a3b expert weights lands within 3 decimals of these.
const CB_LLOYD: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];

/// Written into Y before every checked launch. Sentinel tiles must preserve it.
const POISON: f32 = -98765.0;

/// Gating tolerance for the exact-codebook arm.
const RTOL: f64 = 1e-4;
/// Loose gate for the realistic-codebook arm (fp16 bilinear reconstruction
/// deviation lives here; only gross defects should trip this).
const RTOL_LLOYD: f64 = 5e-3;

// ── fp16 <-> f32 ────────────────────────────────────────────────────────────
// Truncating converter. Bit-exactness of the converter is not required: every
// value that reaches the GPU is read back through `f16_to_f32` on the host, so
// both sides always agree on the *stored* bit pattern.

fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        return sign | 0x7c00;
    }
    sign | ((e as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        return f32::from_bits(sign);
    }
    if e == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
}

// ── deterministic PRNG ──────────────────────────────────────────────────────

fn mix(x: u64) -> u64 {
    let h = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

/// Uniform in [0,1).
fn u01(x: u64) -> f32 {
    (mix(x) % 1_000_003) as f32 / 1_000_003.0
}

// ── weight builders (on-disk MQ2-Lloyd byte layout, qt=19) ──────────────────

#[derive(Clone, Copy, PartialEq)]
enum CbKind {
    /// `u * {-14,-4,4,18}`, `u` a power of two — fp16-exact under the kernel's
    /// bilinear reconstruction.
    Exact,
    /// Textbook Lloyd-Max levels x a continuous per-group scale.
    Lloyd,
}

/// One expert's weight blob: `M` rows x `K/256` groups x 72 B.
///
/// Layout per group, straight from the format spec:
///   `[0..8)`  4 x fp16 codebook entries (ascending)
///   `[8..72)` 64 B of 2-bit indices, LSB-first, 4 per byte
fn build_expert(m: usize, k: usize, seed: u64, kind: CbKind) -> Vec<u8> {
    let gpr = k / 256;
    let mut out = vec![0u8; m * gpr * GROUP_BYTES];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * GROUP_BYTES;
            let sk = seed ^ ((row as u64) << 20) ^ ((g as u64) << 3) ^ 0xA1;
            match kind {
                CbKind::Exact => {
                    // Power-of-two group scale in {2^-6 .. 2^-9}.
                    let shift = 6 + (mix(sk) % 4) as i32;
                    let u = (2.0f32).powi(-shift);
                    for (j, &n) in CB_EXACT_INT.iter().enumerate() {
                        let h = f32_to_f16_bits(u * n as f32);
                        out[off + 2 * j] = (h & 0xff) as u8;
                        out[off + 2 * j + 1] = (h >> 8) as u8;
                    }
                }
                CbKind::Lloyd => {
                    // Continuous per-group scale in [0.02, 0.06).
                    let s = 0.02 + 0.04 * u01(sk);
                    for (j, &c) in CB_LLOYD.iter().enumerate() {
                        let h = f32_to_f16_bits(c * s);
                        out[off + 2 * j] = (h & 0xff) as u8;
                        out[off + 2 * j + 1] = (h >> 8) as u8;
                    }
                }
            }
            for b in 0..64 {
                let r = mix(seed ^ ((row as u64) << 34) ^ ((g as u64) << 12) ^ (b as u64) ^ 0xBEEF);
                out[off + 8 + b] = (r & 0xff) as u8;
            }
        }
    }
    out
}

/// Degenerate "signature" expert: all four codebook entries equal
/// `(expert_id+1)/256`, so every decoded weight is that constant regardless of
/// the index bits (`d01 == d02 == d_xor == 0` exactly). Index bytes are still
/// varied so a decode that read the wrong region would not accidentally pass.
fn build_expert_probe(m: usize, k: usize, expert_id: usize) -> Vec<u8> {
    let gpr = k / 256;
    let h = f32_to_f16_bits((expert_id as f32 + 1.0) / 256.0);
    let mut out = vec![0u8; m * gpr * GROUP_BYTES];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * GROUP_BYTES;
            for j in 0..4 {
                out[off + 2 * j] = (h & 0xff) as u8;
                out[off + 2 * j + 1] = (h >> 8) as u8;
            }
            for b in 0..64 {
                let r = mix(((expert_id as u64) << 40)
                    ^ ((row as u64) << 20)
                    ^ ((g as u64) << 8)
                    ^ b as u64);
                out[off + 8 + b] = (r & 0xff) as u8;
            }
        }
    }
    out
}

/// The exact fp16 constant expert `e` decodes to (host mirror of the probe).
fn probe_weight_value(expert_id: usize) -> f32 {
    f16_to_f32(f32_to_f16_bits((expert_id as f32 + 1.0) / 256.0))
}

// ── X builders ──────────────────────────────────────────────────────────────

/// Returns (fp16 bytes for the GPU, decoded f32 values for the host reference).
/// NOT FWHT-rotated — see the module doc; both arms consume the same bytes.
fn build_x(rows: usize, k: usize, seed: u64, lo: f32, hi: f32) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(rows * k * 2);
    let mut vals = Vec::with_capacity(rows * k);
    for r in 0..rows {
        for c in 0..k {
            let t = u01(seed ^ ((r as u64) << 24) ^ (c as u64));
            let h = f32_to_f16_bits(lo + (hi - lo) * t);
            bytes.extend_from_slice(&h.to_le_bytes());
            vals.push(f16_to_f32(h));
        }
    }
    (bytes, vals)
}

// ── routing / grouped layout ────────────────────────────────────────────────

/// Faithful model of what `moe_scatter_fused_k8` hands the grouped GEMM: slots
/// sorted by expert, each expert's run padded up to a multiple of 16 with `-1`,
/// one `expert_tile_ids` entry per 16-slot tile, then a tail of `-1` sentinel
/// tiles (the dispatcher over-launches to skip the `m_total` dtoh sync, so
/// `m_total` passed to the kernel is the upper bound and the sentinel tiles are
/// killed by the `expert_id < 0` early return, not by the `slot_start >=
/// m_total` one).
struct Routing {
    /// `expert_tile_ids`, length `m_total_max/16`. Tail entries are -1.
    tile_expert: Vec<i32>,
    /// `sorted_slot_index`, length `m_total_max`. -1 = padding slot.
    sorted_slot: Vec<i32>,
    /// Padded slot count actually backed by tiles with a real expert.
    m_total: usize,
    /// What we pass as the kernel's `m_total` arg (includes sentinel tiles).
    m_total_max: usize,
    /// Number of tiles with a real expert id.
    used_tiles: usize,
    /// How many distinct experts got at least one slot.
    live_experts: usize,
    /// Slots that are intra-tile padding (`sorted_slot_index < 0`).
    pad_slots: usize,
}

fn build_routing(n_tokens: usize, k_top: usize, n_experts: usize, seed: u64) -> Routing {
    let mut per_expert: Vec<Vec<i32>> = vec![Vec::new(); n_experts];
    for t in 0..n_tokens {
        let mut chosen: Vec<usize> = Vec::with_capacity(k_top);
        let mut r = mix(seed ^ t as u64);
        while chosen.len() < k_top {
            r = mix(r);
            let e = (r % n_experts as u64) as usize;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        for (j, &e) in chosen.iter().enumerate() {
            // flat slot id; gate_up recovers the token via flat / k_top.
            per_expert[e].push((t * k_top + j) as i32);
        }
    }

    let mut sorted_slot: Vec<i32> = Vec::new();
    let mut tile_expert: Vec<i32> = Vec::new();
    let mut live_experts = 0usize;
    let mut pad_slots = 0usize;
    for (e, slots) in per_expert.iter().enumerate() {
        if slots.is_empty() {
            continue;
        }
        live_experts += 1;
        let cnt = slots.len();
        let tiles = cnt.div_ceil(16);
        sorted_slot.extend_from_slice(slots);
        for _ in cnt..tiles * 16 {
            sorted_slot.push(-1);
            pad_slots += 1;
        }
        for _ in 0..tiles {
            tile_expert.push(e as i32);
        }
    }

    let m_total = sorted_slot.len();
    let used_tiles = tile_expert.len();
    assert_eq!(m_total, used_tiles * 16);

    // Two sentinel tiles on the tail.
    let m_total_max = m_total + 32;
    sorted_slot.resize(m_total_max, -1);
    tile_expert.resize(m_total_max / 16, -1);

    Routing {
        tile_expert,
        sorted_slot,
        m_total,
        m_total_max,
        used_tiles,
        live_experts,
        pad_slots,
    }
}

// ── independent CPU reference ───────────────────────────────────────────────

/// `sum_c W[row][c] * x[c]` for one output element, decoded from the FORMAT
/// SPEC (not from the kernel's tiling): a 256-group is 8 B of fp16 codebook
/// followed by 64 B in which element `c` occupies bits `[2c, 2c+2)` counting
/// LSB-first within byte `c/4`.
fn cpu_ref(wbytes: &[u8], k: usize, row: usize, xrow: &[f32]) -> f64 {
    let gpr = k / 256;
    let mut acc = 0f64;
    for g in 0..gpr {
        let off = (row * gpr + g) * GROUP_BYTES;
        let mut cb = [0f32; 4];
        for (j, slot) in cb.iter_mut().enumerate() {
            *slot = f16_to_f32(u16::from_le_bytes([wbytes[off + 2 * j], wbytes[off + 2 * j + 1]]));
        }
        for c in 0..256 {
            let byte = wbytes[off + 8 + c / 4];
            let q = ((byte >> (2 * (c % 4))) & 0x3) as usize;
            acc += cb[q] as f64 * xrow[g * 256 + c] as f64;
        }
    }
    acc
}

// ── error statistics ────────────────────────────────────────────────────────

/// (slot, row, cpu_want, gpu_got)
type Sample = (usize, usize, f64, f64);

struct ErrStats {
    rms: f64,
    max_abs: f64,
    /// max|d| / rms — error against the output scale.
    norm_err: f64,
    /// max|d|/|want| over the well-conditioned subset (|want| >= rms).
    max_rel_wc: f64,
    n_wc: usize,
    /// Unfloored pointwise max relative error over ALL samples (informational).
    max_rel_raw: f64,
    /// Count of samples with |want| < 0.1 * rms (cancellation-dominated).
    ill: usize,
    /// argmax of |d|/rms.
    worst: Option<Sample>,
}

fn error_stats(samples: &[Sample]) -> ErrStats {
    let n = samples.len().max(1);
    let rms = (samples.iter().map(|s| s.2 * s.2).sum::<f64>() / n as f64).sqrt();
    let denom = if rms > 0.0 { rms } else { 1.0 };
    let mut st = ErrStats {
        rms,
        max_abs: 0.0,
        norm_err: 0.0,
        max_rel_wc: 0.0,
        n_wc: 0,
        max_rel_raw: 0.0,
        ill: 0,
        worst: None,
    };
    for &(slot, row, want, got) in samples {
        let d = (got - want).abs();
        if d > st.max_abs {
            st.max_abs = d;
        }
        let nrm = d / denom;
        if nrm > st.norm_err {
            st.norm_err = nrm;
            st.worst = Some((slot, row, want, got));
        }
        let raw = d / want.abs().max(f64::MIN_POSITIVE);
        if raw > st.max_rel_raw {
            st.max_rel_raw = raw;
        }
        if want.abs() >= denom {
            st.n_wc += 1;
            let rel = d / want.abs();
            if rel > st.max_rel_wc {
                st.max_rel_wc = rel;
            }
        }
        if want.abs() < 0.1 * denom {
            st.ill += 1;
        }
    }
    st
}

// ── shape descriptor ────────────────────────────────────────────────────────

struct Shape {
    label: &'static str,
    m: usize,
    k: usize,
    n_tokens: usize,
    k_top: usize,
    n_experts: usize,
    /// 1 for the `down` contract (X is slot-major, `x_row = flat`), `k_top` for
    /// `gate_up` (X is token-major, `x_row = flat / k_top`).
    x_row_div: usize,
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: GPU init failed ({e:?}) — nothing to bench.");
            return;
        }
    };
    println!("arch = {}", gpu.arch);

    if !gpu.arch.starts_with("gfx12") {
        println!();
        println!("================================================================");
        println!("SKIP — {KERNEL_NAME} is gfx12 (RDNA4) ONLY.");
        println!("  This device reports arch = {}.", gpu.arch);
        println!("  The kernel calls __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12,");
        println!("  which does not exist on gfx10/gfx11 — the JIT would fail, not mis-measure.");
        println!("  Run this on gfx1200/gfx1201 (hiptrx R9700, or a 9070 XT).");
        println!("  For gfx11 use the sibling: gemm_mq2g256_lloyd_moe_grouped_wmma_k2.");
        println!("================================================================");
        return;
    }

    if let Err(e) = gpu.ensure_kernel_public(KERNEL_NAME, KERNEL_SRC, KERNEL_NAME) {
        println!("FAIL: JIT of {KERNEL_NAME} failed: {e:?}");
        std::process::exit(1);
    }
    println!("JIT ok: {KERNEL_NAME}");
    println!();
    println!("REGIME: HIP-dispatch microbenchmark = TRIAGE FILTER for gross defects,");
    println!("        NOT a kernel verdict. Host launch latency masks device-level effects;");
    println!("        the same kernel measures differently under retained PM4 replay.");
    println!("        Final acceptance = golden bundle (registry/redline-golden-v1.json),");
    println!("        HIP and PM4 arms. First dispatch of any shape is JIT-contaminated.");
    println!("        X is NOT FWHT-rotated here; both arms consume identical fp16 bytes.");

    // a3b: hidden=2048, moe_intermediate=512 (gate|up split -> M=1024), k_top=8,
    // n_experts=256. The third shape is the dense o_proj/out_proj residual shape
    // driven through the grouped path as a large-K stress case — labelled as
    // such because it is not a real MoE call site.
    let shapes = [
        Shape {
            label: "routed expert gate_up (a3b)",
            m: 1024,
            k: 2048,
            n_tokens: 512,
            k_top: 8,
            n_experts: 256,
            x_row_div: 8,
        },
        Shape {
            label: "routed expert down (a3b)",
            m: 2048,
            k: 512,
            n_tokens: 512,
            k_top: 8,
            n_experts: 256,
            x_row_div: 1,
        },
        Shape {
            label: "dense o_proj/out_proj residual shape (large-K stress, NOT a MoE call site)",
            m: 2048,
            k: 4096,
            n_tokens: 64,
            k_top: 8,
            n_experts: 16,
            x_row_div: 1,
        },
    ];

    let mut all_pass = true;
    for sh in shapes.iter() {
        all_pass &= run_shape(&mut gpu, sh);
    }

    println!();
    if all_pass {
        println!("################  OVERALL: PASS  ################");
        println!("(triage only — NOT acceptance evidence; see the REGIME note above)");
    } else {
        println!("################  OVERALL: FAIL  ################");
        std::process::exit(1);
    }
}

fn run_shape(gpu: &mut Gpu, sh: &Shape) -> bool {
    let Shape {
        label,
        m,
        k,
        n_tokens,
        k_top,
        n_experts,
        x_row_div,
    } = *sh;
    let gpr = k / 256;
    assert_eq!(k % 256, 0, "K must be a multiple of 256");
    assert_eq!(m % 16, 0, "M must be a multiple of 16 for the 16-row tile");

    let r = build_routing(n_tokens, k_top, n_experts, 0xD15EA5E ^ (m as u64) ^ ((k as u64) << 8));
    let row_tiles = m.div_ceil(16);
    let slot_tiles = r.m_total_max / 16;
    let per_expert_bytes = m * gpr * GROUP_BYTES;
    let grid = [row_tiles as u32, slot_tiles as u32, 1];
    let block = [32u32, 1, 1];

    println!();
    println!("================================================================");
    println!("=== {label}");
    println!(
        "    M={m} K={k} groups/row={gpr}  tokens={n_tokens} k_top={k_top} experts={n_experts} (live {})",
        r.live_experts
    );
    println!(
        "    m_total={} (+{} sentinel slots) used_tiles={} pad_slots={}  grid=[{row_tiles},{slot_tiles}] block=[32]",
        r.m_total,
        r.m_total_max - r.m_total,
        r.used_tiles,
        r.pad_slots
    );
    println!(
        "    weight/expert = {} KiB   x_row_div={x_row_div}   format = MQ2-Lloyd 72 B/group = 2.25 bpw",
        per_expert_bytes / 1024
    );

    // ── device buffers shared by every phase ────────────────────────────────
    let x_rows = if x_row_div > 1 { n_tokens } else { n_tokens * k_top };

    let tile_t = gpu
        .upload_raw(
            &r.tile_expert
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[r.tile_expert.len()],
        )
        .expect("upload expert_tile_ids");
    let slot_t = gpu
        .upload_raw(
            &r.sorted_slot
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[r.sorted_slot.len()],
        )
        .expect("upload sorted_slot_index");

    // Expert weight blobs. Allocated once; re-uploaded between phases so peak
    // VRAM stays at one set.
    let mut w_tensors = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        w_tensors.push(
            gpu.upload_raw(&vec![0u8; per_expert_bytes], &[per_expert_bytes])
                .expect("alloc expert weights"),
        );
    }
    let wptrs: Vec<u64> = w_tensors.iter().map(|t| t.buf.as_ptr() as u64).collect();
    let wptr_t = gpu
        .upload_raw(
            &wptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>(),
            &[n_experts],
        )
        .expect("upload expert_weight_ptrs");

    let y_t = gpu
        .alloc_tensor(&[r.m_total_max, m], DType::F32)
        .expect("alloc Y_grouped");

    let mut pass = true;

    // ── Phase A: expert-isolation probe (full coverage) ─────────────────────
    let (xa_bytes, xa_vals) = build_x(x_rows, k, 0x5EED_A, 0.25, 0.65);
    let xa_t = gpu
        .upload_raw(&xa_bytes, &[x_rows, k])
        .expect("upload X (probe)");

    for (e, wt) in w_tensors.iter().enumerate() {
        let probe = build_expert_probe(m, k, e);
        gpu.hip
            .memcpy_htod(&wt.buf, &probe)
            .expect("htod probe weights");
    }

    let mut ka = {
        let mut b = KernargBlob::new();
        b.push_ptr(wptr_t.buf.as_ptr() as *const _);
        b.push_ptr(tile_t.buf.as_ptr() as *const _);
        b.push_ptr(slot_t.buf.as_ptr() as *const _);
        b.push_ptr(xa_t.buf.as_ptr() as *const _);
        b.push_ptr(y_t.buf.as_ptr() as *const _);
        b.push_i32(m as i32);
        b.push_i32(k as i32);
        b.push_i32(x_row_div as i32);
        b.push_i32(r.m_total_max as i32);
        b.into_vec()
    };

    gpu.fill_f32(&y_t, POISON).expect("poison Y");
    gpu.launch_kernel_blob(KERNEL_NAME, grid, block, 0, &mut ka)
        .expect("launch (probe)");
    gpu.hip.device_synchronize().expect("sync");
    let y_probe = gpu.download_f32(&y_t).expect("download Y (probe)");

    // Row sums of X, in f64 over the exact stored fp16 values.
    let sum_x: Vec<f64> = (0..x_rows)
        .map(|row| {
            xa_vals[row * k..(row + 1) * k]
                .iter()
                .map(|&v| v as f64)
                .sum()
        })
        .collect();

    {
        let mut wrong_expert = 0usize;
        let mut first_wrong: Option<(usize, usize, i32, i64)> = None;
        let mut bad_pad = 0usize;
        let mut bad_sentinel = 0usize;
        let mut max_rel_probe = 0f64;
        let mut nonfinite = 0usize;

        for slot in 0..r.m_total_max {
            let tile = slot / 16;
            let e = r.tile_expert[tile];
            let base = slot * m;
            if e < 0 {
                // Sentinel tile: kernel must early-return, poison must survive.
                for row in 0..m {
                    if y_probe[base + row] != POISON {
                        bad_sentinel += 1;
                    }
                }
                continue;
            }
            let flat = r.sorted_slot[slot];
            if flat < 0 {
                // Intra-tile padding slot: contributes a zero B row -> exactly 0.
                for row in 0..m {
                    if y_probe[base + row] != 0.0 {
                        bad_pad += 1;
                    }
                }
                continue;
            }
            let x_row = if x_row_div > 1 {
                flat as usize / x_row_div
            } else {
                flat as usize
            };
            let sx = sum_x[x_row];
            let want = probe_weight_value(e as usize) as f64 * sx;
            for row in 0..m {
                let got = y_probe[base + row] as f64;
                if !got.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                // Recover the expert id the kernel ACTUALLY multiplied by.
                let recovered = (got / sx * 256.0).round() as i64 - 1;
                if recovered != e as i64 {
                    wrong_expert += 1;
                    if first_wrong.is_none() {
                        first_wrong = Some((slot, row, e, recovered));
                    }
                }
                let rel = (got - want).abs() / want.abs().max(1e-12);
                if rel > max_rel_probe {
                    max_rel_probe = rel;
                }
            }
        }

        let ok = wrong_expert == 0
            && bad_pad == 0
            && bad_sentinel == 0
            && nonfinite == 0
            && max_rel_probe <= RTOL;
        pass &= ok;
        println!();
        println!(
            "  [Phase A] expert-isolation probe — FULL coverage ({} elements)",
            r.m_total_max * m
        );
        println!(
            "    wrong-expert cells : {wrong_expert}    padding-slot != 0 : {bad_pad}    sentinel-tile written : {bad_sentinel}"
        );
        println!("    non-finite         : {nonfinite}    max_rel = {max_rel_probe:.3e}  (all-positive accumulation, well conditioned)");
        if let Some((slot, row, want_e, got_e)) = first_wrong {
            println!(
                "    !! GROUP-BOUNDARY BUG: slot {slot} (tile {}) row {row}: expert_tile_ids says {want_e}, weights came from expert {got_e}",
                slot / 16
            );
        }
        println!(
            "    ==> {}  (a token routed to expert e must hit expert e's weights ONLY)",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    // ── Phase B: numeric parity vs the independent CPU reference ────────────
    let (xb_bytes, xb_vals) = build_x(x_rows, k, 0x5EED_B, -0.6, 0.6);
    let xb_t = gpu
        .upload_raw(&xb_bytes, &[x_rows, k])
        .expect("upload X (numeric)");

    let mut kb = {
        let mut b = KernargBlob::new();
        b.push_ptr(wptr_t.buf.as_ptr() as *const _);
        b.push_ptr(tile_t.buf.as_ptr() as *const _);
        b.push_ptr(slot_t.buf.as_ptr() as *const _);
        b.push_ptr(xb_t.buf.as_ptr() as *const _);
        b.push_ptr(y_t.buf.as_ptr() as *const _);
        b.push_i32(m as i32);
        b.push_i32(k as i32);
        b.push_i32(x_row_div as i32);
        b.push_i32(r.m_total_max as i32);
        b.into_vec()
    };

    // Every used tile is sampled (so every tile boundary is covered), a few
    // slots and rows each.
    let slot_probes: [usize; 4] = [0, 1, 7, 15];
    let row_probes: Vec<usize> = {
        let mut v = vec![0usize, 1, m / 2, m - 2, m - 1];
        v.dedup();
        v
    };

    for (phase, kind, tol) in [
        ("B1", CbKind::Exact, RTOL),
        ("B2", CbKind::Lloyd, RTOL_LLOYD),
    ] {
        let mut host_w: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
        for (e, wt) in w_tensors.iter().enumerate() {
            let bytes = build_expert(m, k, 0xC0FFEE ^ ((e as u64) << 13), kind);
            gpu.hip.memcpy_htod(&wt.buf, &bytes).expect("htod weights");
            host_w.push(bytes);
        }

        gpu.fill_f32(&y_t, POISON).expect("poison Y");
        gpu.launch_kernel_blob(KERNEL_NAME, grid, block, 0, &mut kb)
            .expect("launch (numeric)");
        gpu.hip.device_synchronize().expect("sync");
        let y_num = gpu.download_f32(&y_t).expect("download Y (numeric)");

        let mut samples: Vec<Sample> = Vec::new();
        let mut nonfinite = 0usize;
        for tile in 0..r.used_tiles {
            let e = r.tile_expert[tile] as usize;
            for &so in slot_probes.iter() {
                let slot = tile * 16 + so;
                let flat = r.sorted_slot[slot];
                if flat < 0 {
                    continue;
                }
                let x_row = if x_row_div > 1 {
                    flat as usize / x_row_div
                } else {
                    flat as usize
                };
                let xrow = &xb_vals[x_row * k..(x_row + 1) * k];
                for &row in row_probes.iter() {
                    let want = cpu_ref(&host_w[e], k, row, xrow);
                    let got = y_num[slot * m + row] as f64;
                    if !got.is_finite() {
                        nonfinite += 1;
                        continue;
                    }
                    samples.push((slot, row, want, got));
                }
            }
        }
        drop(host_w);

        let st = error_stats(&samples);
        let ok = !samples.is_empty()
            && nonfinite == 0
            && st.norm_err <= tol
            && st.max_rel_wc <= tol;
        // B2 is informational: it measures the kernel family's inherent fp16
        // bilinear-reconstruction deviation, gated loosely for gross defects.
        if kind == CbKind::Exact {
            pass &= ok;
        }

        let kind_txt = match kind {
            CbKind::Exact => "fp16-EXACT codebook u*{-14,-4,4,18} (GATING)",
            CbKind::Lloyd => "realistic Lloyd-Max codebook (INFORMATIONAL)",
        };
        println!();
        println!(
            "  [Phase {phase}] numeric parity vs independent CPU reference — {kind_txt}"
        );
        println!(
            "    {} sampled outputs across all {} used tiles   ref rms = {:.6e}",
            samples.len(),
            r.used_tiles,
            st.rms
        );
        println!(
            "    max_abs_err = {:.6e}    norm_err = max|d|/rms = {:.3e}",
            st.max_abs, st.norm_err
        );
        println!(
            "    max_rel_err (|cpu| >= rms, {} samples) = {:.3e}    max_rel_err unfloored (all) = {:.3e}   ill-conditioned (|cpu| < 0.1 rms) = {}",
            st.n_wc, st.max_rel_wc, st.max_rel_raw, st.ill
        );
        if let Some((slot, row, want, got)) = st.worst {
            println!(
                "    worst index: slot={slot} (tile {}, expert {}) row={row}   cpu={want:.9e}  gpu={got:.9e}  |d|={:.3e}",
                slot / 16,
                r.tile_expert[slot / 16],
                (got - want).abs()
            );
        }
        if nonfinite > 0 {
            println!("    !! {nonfinite} non-finite GPU outputs");
        }
        println!(
            "    ==> {}  (tolerance {:.0e}{})",
            if ok { "PASS" } else { "FAIL" },
            tol,
            if kind == CbKind::Exact {
                ""
            } else {
                ", informational — fp16 bilinear reconstruction deviation lives here"
            }
        );
    }

    // ── Phase C: triage timing (runs on the Phase-B2 weights) ───────────────
    {
        const INNER: usize = 10;
        const REPS: usize = 7;

        // Throwaway pass: JIT / cold-cache / DPM-ramp contaminated. Discarded.
        for _ in 0..INNER {
            gpu.launch_kernel_blob(KERNEL_NAME, grid, block, 0, &mut kb)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        // Warmup.
        for _ in 0..(2 * INNER) {
            gpu.launch_kernel_blob(KERNEL_NAME, grid, block, 0, &mut kb)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let mut us: Vec<f64> = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t0 = Instant::now();
            for _ in 0..INNER {
                gpu.launch_kernel_blob(KERNEL_NAME, grid, block, 0, &mut kb)
                    .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            us.push(t0.elapsed().as_secs_f64() * 1e6 / INNER as f64);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_us = us[0];
        let med_us = us[us.len() / 2];
        let max_us = us[us.len() - 1];

        // Streamed traffic per call (no cache-reuse credit):
        //   weights: every row-tile block re-reads its expert's 16 rows, so each
        //            used slot-tile pulls the whole M x K expert blob once.
        //   X      : each row-tile block reads its 16 slots' K fp16.
        //   Y      : one f32 per (slot, row) over slots in used tiles.
        let wt_bytes = r.used_tiles as f64 * per_expert_bytes as f64;
        let x_bytes = row_tiles as f64 * r.m_total as f64 * k as f64 * 2.0;
        let y_bytes = r.m_total as f64 * m as f64 * 4.0;
        let total_bytes = wt_bytes + x_bytes + y_bytes;
        let flops = 2.0 * m as f64 * k as f64 * r.m_total as f64;
        let gbs = |t_us: f64| total_bytes / (t_us * 1e-6) / 1e9;
        let gflops = |t_us: f64| flops / (t_us * 1e-6) / 1e9;

        println!();
        println!("  [Phase C] triage timing — {REPS} reps x {INNER} launches, post-warmup");
        println!("    NOTE: the first pass of any shape is JIT / cold-cache contaminated and is DISCARDED.");
        println!(
            "    MIN    {min_us:>9.2} us/call   {:>8.1} GFLOP/s   {:>7.1} GB/s streamed",
            gflops(min_us),
            gbs(min_us)
        );
        println!(
            "    MEDIAN {med_us:>9.2} us/call   {:>8.1} GFLOP/s   {:>7.1} GB/s streamed",
            gflops(med_us),
            gbs(med_us)
        );
        println!(
            "    spread min..max = {min_us:.2}..{max_us:.2} us ({:+.1}%)",
            100.0 * (max_us / min_us - 1.0)
        );
        println!(
            "    weight bytes/element = {:.5} B ({:.4} bpw)   streamed bytes per output element = {:.2} B",
            GROUP_BYTES as f64 / 256.0,
            GROUP_BYTES as f64 * 8.0 / 256.0,
            total_bytes / (r.m_total as f64 * m as f64)
        );
        println!(
            "    per-call traffic: weights {:.1} MiB | X {:.1} MiB | Y {:.1} MiB | total {:.1} MiB",
            wt_bytes / 1048576.0,
            x_bytes / 1048576.0,
            y_bytes / 1048576.0,
            total_bytes / 1048576.0
        );
        println!(
            "    tok-shaped: {:.2} us per 1k routed slots   {:.0} prefill tok/s for THIS GEMM alone ({n_tokens} tok/call)",
            med_us * 1000.0 / r.m_total as f64,
            n_tokens as f64 / (med_us * 1e-6)
        );
    }

    println!();
    println!("  SHAPE VERDICT: {}", if pass { "PASS" } else { "FAIL" });
    pass
}
