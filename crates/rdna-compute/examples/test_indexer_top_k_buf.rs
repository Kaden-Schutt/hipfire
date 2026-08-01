// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exactness / raw-parity probe for DeepSeek V4's graph-safe indexer top-K.
//!
//! Launches SERIAL (`indexer_top_k_buf`) and PARALLEL (`indexer_top_k_buf_parallel`)
//! against the SAME frozen score buffer and compares the ordered 512-slot i32
//! outputs element-by-element. Finite cases must match the host oracle and each
//! other; non-finite cases must match SERIAL↔PARALLEL (eligibility filter) but
//! are not asserted against the generic total_cmp host oracle.
//!
//! Run on the target GPU (MI300X / gfx942 or gfx1151):
//!   cargo run --release -p rdna-compute --example test_indexer_top_k_buf
//!
//! Compile-check only (no GPU):
//!   cargo check -p rdna-compute --example test_indexer_top_k_buf

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const MAX_N: usize = 2048;
const K: usize = 512;
const N_HEADS: i32 = 1;

const SERIAL_SRC: &str = include_str!("../../../kernels/src/indexer_top_k_buf.hip");
const PARALLEL_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_parallel.gfx1151.hip");
const PARALLEL_GFX942_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_parallel.gfx942.hip");
const SERIAL_MODULE: &str = "indexer_top_k_buf";
const SERIAL_SYMBOL: &str = "indexer_top_k_buf";
const SERIAL_BLOCK: [u32; 3] = [128, 1, 1];
const PARALLEL_BLOCK: [u32; 3] = [256, 1, 1];
const SERIAL_POISON: i32 = -7777;
const PARALLEL_POISON: i32 = -8888;

// ─── minimal SHA-256 (no extra crate dep; example-only) ──────────────────────

fn sha256(data: &[u8]) -> [u8; 32] {
    fn rotr(x: u32, n: u32) -> u32 {
        x.rotate_right(n)
    }
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for i in 0..64 {
            let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn hex32(digest: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn sha256_i32(values: &[i32]) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    hex32(&sha256(bytes))
}

fn sha256_f32(values: &[f32]) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    hex32(&sha256(bytes))
}

// ─── host I/O helpers ────────────────────────────────────────────────────────

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let tensor = gpu
        .alloc_tensor(&[values.len() * 4], DType::Raw)
        .expect("alloc i32 tensor");
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("upload i32 tensor");
    tensor
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Vec<i32> {
    let mut values = vec![0i32; len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("download i32 tensor");
    values
}

/// Stable host oracle: identity when N <= K, else score DESC / index ASC via total_cmp.
fn expected_top_k(scores: &[f32], n: usize) -> Vec<i32> {
    if n <= K {
        return (0..K)
            .map(|index| if index < n { index as i32 } else { -1 })
            .collect();
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    indices
        .into_iter()
        .take(K)
        .map(|index| index as i32)
        .collect()
}

// ─── launch plumbing ─────────────────────────────────────────────────────────

struct KernelSpec {
    label: &'static str,
    module: &'static str,
    source: &'static str,
    symbol: &'static str,
    block: [u32; 3],
    /// Dynamic LDS bytes. SERIAL uses max_n_compressed (taken bitmap); PARALLEL uses 0.
    smem_fn: fn(max_n: i32) -> u32,
}

const SERIAL: KernelSpec = KernelSpec {
    label: "SERIAL",
    module: SERIAL_MODULE,
    source: SERIAL_SRC,
    symbol: SERIAL_SYMBOL,
    block: SERIAL_BLOCK,
    smem_fn: |max_n| max_n as u32,
};

const PARALLEL_GFX1151: KernelSpec = KernelSpec {
    label: "PARALLEL",
    module: "indexer_top_k_buf_parallel",
    source: PARALLEL_GFX1151_SRC,
    symbol: "indexer_top_k_buf_parallel",
    block: PARALLEL_BLOCK,
    smem_fn: |_| 0,
};

const PARALLEL_GFX942: KernelSpec = KernelSpec {
    label: "PARALLEL_GFX942",
    module: "indexer_top_k_buf_parallel_gfx942",
    source: PARALLEL_GFX942_SRC,
    symbol: "indexer_top_k_buf_parallel_gfx942",
    block: PARALLEL_BLOCK,
    smem_fn: |_| 0,
};

fn parallel_spec(arch: &str) -> &'static KernelSpec {
    match arch {
        "gfx942" => &PARALLEL_GFX942,
        "gfx1151" => &PARALLEL_GFX1151,
        _ => unreachable!("architecture checked by main"),
    }
}

fn ensure_both(gpu: &mut Gpu, parallel: &KernelSpec) {
    gpu.ensure_kernel_public(SERIAL.module, SERIAL.source, SERIAL.symbol)
        .expect("compile SERIAL indexer_top_k_buf");
    gpu.ensure_kernel_public(parallel.module, parallel.source, parallel.symbol)
        .expect("compile PARALLEL indexer_top_k_buf_parallel");
}

fn launch_one(
    gpu: &Gpu,
    spec: &KernelSpec,
    scores: &GpuTensor,
    out: &GpuTensor,
    n_buf: &GpuTensor,
    k_buf: &GpuTensor,
    max_n: i32,
    max_k: i32,
) {
    let grid = [N_HEADS as u32, 1, 1];
    let smem = (spec.smem_fn)(max_n);
    eprintln!(
        "  LAUNCH {} symbol={} grid={:?} block={:?} dynamic_lds={}",
        spec.label, spec.symbol, grid, spec.block, smem
    );
    let mut kb = KernargBlob::new();
    kb.push_ptr(scores.buf.as_ptr() as *const c_void);
    kb.push_ptr(out.buf.as_ptr() as *const c_void);
    kb.push_ptr(n_buf.buf.as_ptr() as *const c_void);
    kb.push_ptr(k_buf.buf.as_ptr() as *const c_void);
    kb.push_i32(N_HEADS);
    kb.push_i32(max_k);
    kb.pad_to(16);
    gpu.launch_kernel_blob(spec.symbol, grid, spec.block, smem, kb.as_mut_slice())
        .unwrap_or_else(|e| panic!("launch {} failed: {e:?}", spec.label));
}

// ─── score builders ──────────────────────────────────────────────────────────

/// Exact-tie finite pattern from the divergence analysis.
fn finite_tie_scores() -> Vec<f32> {
    (0..MAX_N).map(|i| ((37 * i + 11) % 64) as f32).collect()
}

enum CaseKind {
    Finite,
    NegInf,
    Nan,
    NegFltMax,
}

struct Case {
    name: &'static str,
    n: usize,
    kind: CaseKind,
    scores: Vec<f32>,
}

fn build_cases() -> Vec<Case> {
    let mut cases = Vec::new();

    for &n in &[512usize, 513, 640, 2048] {
        cases.push(Case {
            name: match n {
                512 => "finite_n512_identity",
                513 => "finite_n513_product_boundary",
                640 => "finite_n640",
                2048 => "finite_n2048",
                _ => unreachable!(),
            },
            n,
            kind: CaseKind::Finite,
            scores: finite_tie_scores(),
        });
    }

    // 200 finite + 313 -inf at N=513 → fewer than 512 selectable under SERIAL.
    {
        let mut scores = finite_tie_scores();
        // Keep first 200 finite; fill remaining of the N window with -inf.
        for i in 200..513 {
            scores[i] = f32::NEG_INFINITY;
        }
        // Pad beyond N is unused by kernels but keep deterministic.
        for i in 513..MAX_N {
            scores[i] = f32::NEG_INFINITY;
        }
        cases.push(Case {
            name: "nonfinite_n513_neginf_pool",
            n: 513,
            kind: CaseKind::NegInf,
            scores,
        });
    }

    // A few NaNs at N=513 — PARALLEL rank-0 collisions expected.
    {
        let mut scores = finite_tie_scores();
        scores[0] = f32::from_bits(0x7fc0_0001); // quiet NaN payload 1
        scores[257] = f32::from_bits(0x7fc0_0002); // quiet NaN payload 2
        scores[400] = f32::NAN;
        cases.push(Case {
            name: "nonfinite_n513_nan",
            n: 513,
            kind: CaseKind::Nan,
            scores,
        });
    }

    // Several exact -FLT_MAX entries (SERIAL best=-FLT_MAX cannot select via strict >).
    {
        let mut scores = finite_tie_scores();
        for &i in &[10usize, 100, 250, 400, 512] {
            scores[i] = -f32::MAX; // == -FLT_MAX
        }
        cases.push(Case {
            name: "nonfinite_n513_neg_flt_max",
            n: 513,
            kind: CaseKind::NegFltMax,
            scores,
        });
    }

    cases
}

// ─── comparison / reporting ──────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct SlotStats {
    poison: usize,
    pad_neg1: usize,
    real: usize,
    out_of_range: usize,
    duplicates: usize,
}

fn slot_stats(out: &[i32], n: usize, poison: i32) -> SlotStats {
    let mut seen = vec![0u8; n.max(1)];
    let mut poison_c = 0usize;
    let mut pad = 0usize;
    let mut real = 0usize;
    let mut oor = 0usize;
    let mut dups = 0usize;
    for &v in out {
        if v == poison {
            poison_c += 1;
        } else if v == -1 {
            pad += 1;
        } else if v < 0 || (v as usize) >= n {
            oor += 1;
        } else {
            real += 1;
            let idx = v as usize;
            if seen[idx] != 0 {
                dups += 1;
            } else {
                seen[idx] = 1;
            }
        }
    }
    SlotStats {
        poison: poison_c,
        pad_neg1: pad,
        real,
        out_of_range: oor,
        duplicates: dups,
    }
}

fn first_diff(a: &[i32], b: &[i32]) -> Option<(usize, i32, i32)> {
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            return Some((i, a[i], b[i]));
        }
    }
    if a.len() != b.len() {
        return Some((a.len().min(b.len()), -1, -1));
    }
    None
}

fn same_set(a: &[i32], b: &[i32]) -> bool {
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_unstable();
    sb.sort_unstable();
    sa == sb
}

fn ascending_set_hash(values: &[i32]) -> String {
    let mut s = values.to_vec();
    s.sort_unstable();
    sha256_i32(&s)
}

struct CaseResult {
    name: String,
    n: usize,
    finite: bool,
    serial_vs_parallel: &'static str,
    first_diff_rank: String,
    same_set: bool,
    serial_unwritten: usize,
    parallel_unwritten: usize,
    serial_vs_oracle: &'static str,
    parallel_vs_oracle: &'static str,
    accept_fail: bool,
}

fn score_census(scores: &[f32], n: usize) {
    let mut finite = 0usize;
    let mut nan = 0usize;
    let mut neginf = 0usize;
    let mut posinf = 0usize;
    let mut neg_flt_max = 0usize;
    let mut min_finite = f32::INFINITY;
    for &s in &scores[..n] {
        if s.is_nan() {
            nan += 1;
        } else if s == f32::NEG_INFINITY {
            neginf += 1;
        } else if s == f32::INFINITY {
            posinf += 1;
        } else {
            finite += 1;
            if s < min_finite {
                min_finite = s;
            }
            if s == -f32::MAX {
                neg_flt_max += 1;
            }
        }
    }
    eprintln!(
        "  score_census n={n}: finite={finite} nan={nan} -inf={neginf} +inf={posinf} -FLT_MAX={neg_flt_max} min_finite={}",
        if finite > 0 {
            format!("{min_finite}")
        } else {
            "n/a".into()
        }
    );
}

fn run_case(gpu: &mut Gpu, parallel: &KernelSpec, case: &Case) -> CaseResult {
    let n = case.n;
    assert!(n <= MAX_N);
    let finite = matches!(case.kind, CaseKind::Finite);

    eprintln!(
        "\n======== CASE {} N={} finite={} ========",
        case.name, n, finite
    );
    score_census(&case.scores, n);
    let scores_sha = sha256_f32(&case.scores);
    eprintln!("  scores_sha256={scores_sha} (full MAX_N={} buffer)", MAX_N);

    // Build score buffer ONCE, upload ONCE.
    let scores_gpu = gpu
        .upload_f32(&case.scores, &[MAX_N])
        .expect("upload scores");
    let n_gpu = upload_i32(gpu, &[n as i32]);
    let k_gpu = upload_i32(gpu, &[K as i32]);

    // Two independently poisoned output buffers.
    let serial_poison_host = vec![SERIAL_POISON; K];
    let parallel_poison_host = vec![PARALLEL_POISON; K];
    let serial_out = upload_i32(gpu, &serial_poison_host);
    let parallel_out = upload_i32(gpu, &parallel_poison_host);

    // Launch SERIAL → A, PARALLEL → B from the same input.
    launch_one(
        gpu,
        &SERIAL,
        &scores_gpu,
        &serial_out,
        &n_gpu,
        &k_gpu,
        MAX_N as i32,
        K as i32,
    );
    launch_one(
        gpu,
        parallel,
        &scores_gpu,
        &parallel_out,
        &n_gpu,
        &k_gpu,
        MAX_N as i32,
        K as i32,
    );
    gpu.hip
        .device_synchronize()
        .expect("synchronize after dual launch");

    let serial = download_i32(gpu, &serial_out, K);
    let parallel = download_i32(gpu, &parallel_out, K);
    let oracle = expected_top_k(&case.scores, n);

    let ss = slot_stats(&serial, n, SERIAL_POISON);
    let ps = slot_stats(&parallel, n, PARALLEL_POISON);

    let ordered_eq = serial == parallel;
    let set_eq = same_set(&serial, &parallel);
    let diff = first_diff(&serial, &parallel);
    let n_diffs = serial
        .iter()
        .zip(parallel.iter())
        .filter(|(a, b)| a != b)
        .count();

    let serial_vs_oracle = if serial == oracle { "MATCH" } else { "DIFFER" };
    let parallel_vs_oracle = if parallel == oracle {
        "MATCH"
    } else {
        "DIFFER"
    };
    let serial_vs_parallel = if ordered_eq { "MATCH" } else { "DIFFER" };

    let first_diff_rank = match diff {
        Some((i, a, b)) => {
            eprintln!("  first_diff_rank={i} serial={a} parallel={b}");
            format!("{i}")
        }
        None => {
            eprintln!("  first_diff_rank=- (ordered arrays identical)");
            "-".into()
        }
    };

    eprintln!("  serial_vs_parallel={serial_vs_parallel} n_diffs={n_diffs} same_set={set_eq}");
    eprintln!(
        "  serial_slots: poison={} pad_neg1={} real={} oor={} dups={}",
        ss.poison, ss.pad_neg1, ss.real, ss.out_of_range, ss.duplicates
    );
    eprintln!(
        "  parallel_slots: poison={} pad_neg1={} real={} oor={} dups={}",
        ps.poison, ps.pad_neg1, ps.real, ps.out_of_range, ps.duplicates
    );
    eprintln!(
        "  serial_ordered_sha256={} serial_set_sha256={}",
        sha256_i32(&serial),
        ascending_set_hash(&serial)
    );
    eprintln!(
        "  parallel_ordered_sha256={} parallel_set_sha256={}",
        sha256_i32(&parallel),
        ascending_set_hash(&parallel)
    );
    eprintln!(
        "  oracle_ordered_sha256={} serial_vs_oracle={serial_vs_oracle} parallel_vs_oracle={parallel_vs_oracle}",
        sha256_i32(&oracle)
    );

    if !ordered_eq {
        // Print a short window around the first mismatch for operator grepping.
        if let Some((i, _, _)) = diff {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(K);
            eprintln!("  window ranks[{lo}..{hi}):");
            for r in lo..hi {
                eprintln!(
                    "    rank={r} serial={} parallel={} oracle={}",
                    serial[r], parallel[r], oracle[r]
                );
            }
        }
    }

    // Characterise non-finite pads specifically.
    if !finite {
        eprintln!(
            "  nonfinite_detail: serial(-1={}, real={}, poison={}) parallel(-1={}, real={}, poison={})",
            ss.pad_neg1, ss.real, ss.poison, ps.pad_neg1, ps.real, ps.poison
        );
    }

    // Acceptance:
    // - Finite: serial == parallel == oracle, no poison/dups.
    // - Non-finite: serial == parallel (eligibility filter), no poison/dups.
    //   Host oracle stays a generic total_cmp sort and may DIFFER; report only.
    let accept_fail = {
        let mut bad = false;
        if serial != parallel {
            eprintln!("  ACCEPT_FAIL: SERIAL != PARALLEL");
            bad = true;
        }
        if ss.poison != 0 || ps.poison != 0 {
            eprintln!(
                "  ACCEPT_FAIL: poisoned/unwritten slots remain (serial={} parallel={})",
                ss.poison, ps.poison
            );
            bad = true;
        }
        if ss.duplicates != 0 || ps.duplicates != 0 {
            eprintln!(
                "  ACCEPT_FAIL: duplicate indices (serial={} parallel={})",
                ss.duplicates, ps.duplicates
            );
            bad = true;
        }
        if finite {
            if serial != oracle {
                eprintln!("  ACCEPT_FAIL: SERIAL != host oracle");
                bad = true;
            }
            if parallel != oracle {
                eprintln!("  ACCEPT_FAIL: PARALLEL != host oracle");
                bad = true;
            }
        }
        if !bad {
            if finite {
                eprintln!("  ACCEPT: PASS (finite case byte-identical to oracle and peer)");
            } else {
                eprintln!("  ACCEPT: PASS (non-finite SERIAL==PARALLEL; oracle reported only)");
            }
        }
        bad
    };

    CaseResult {
        name: case.name.to_string(),
        n,
        finite,
        serial_vs_parallel,
        first_diff_rank,
        same_set: set_eq,
        serial_unwritten: ss.poison,
        parallel_unwritten: ps.poison,
        serial_vs_oracle,
        parallel_vs_oracle,
        accept_fail,
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let arch = gpu.arch.as_str();
    eprintln!("detected_arch={arch}");
    if arch != "gfx1151" && arch != "gfx942" {
        panic!(
            "unsupported arch '{arch}': this parity probe accepts only gfx1151 or gfx942 \
             (refuse, do not skip)"
        );
    }
    eprintln!("arch_ok=true (accepted gfx1151|gfx942)");

    let parallel = parallel_spec(arch);
    ensure_both(&mut gpu, parallel);
    eprintln!(
        "kernels_loaded: SERIAL symbol={} block={:?} smem=max_n_compressed; \
         PARALLEL symbol={} block={:?} smem=0",
        SERIAL.symbol, SERIAL.block, parallel.symbol, parallel.block
    );

    let cases = build_cases();
    let mut results = Vec::with_capacity(cases.len());
    let mut any_fail = false;
    for case in &cases {
        let r = run_case(&mut gpu, parallel, case);
        if r.accept_fail {
            any_fail = true;
        }
        results.push(r);
    }

    eprintln!("\n======== SUMMARY (machine-greppable) ========");
    for r in &results {
        eprintln!(
            "RESULT case={} N={} finite={} serial_vs_parallel={} first_diff_rank={} same_set={} serial_unwritten={} parallel_unwritten={} serial_vs_oracle={} parallel_vs_oracle={}",
            r.name,
            r.n,
            r.finite,
            r.serial_vs_parallel,
            r.first_diff_rank,
            r.same_set,
            r.serial_unwritten,
            r.parallel_unwritten,
            r.serial_vs_oracle,
            r.parallel_vs_oracle
        );
    }

    if any_fail {
        eprintln!("OVERALL: FAIL (one or more cases mismatched SERIAL↔PARALLEL or finite oracle)");
        std::process::exit(1);
    }
    eprintln!("OVERALL: PASS (all cases SERIAL==PARALLEL; finite matched oracle)");
}
