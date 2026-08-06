// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exactness / raw-parity probe for DeepSeek V4's graph-safe indexer top-K.
//!
//! Arms, all launched against the SAME frozen score buffer and compared
//! slot-for-slot on the ordered 512-slot i32 outputs:
//!
//! - SERIAL (`indexer_top_k_buf`) — the original single-workgroup reference,
//!   with the eligibility discipline (strict `> -FLT_MAX`).
//! - PARALLEL (arch reference): `indexer_top_k_buf_parallel` on gfx1151,
//!   `indexer_top_k_buf_parallel_gfx942` on gfx942 — the O(N^2) rank-count on
//!   one workgroup. NOTE: the gfx1151 reference lacks the eligibility filter
//!   its gfx942 sibling has, so it FAILS the three non-finite cases today
//!   (NaN rank-0 collisions leave slots unwritten). That is a known kernel
//!   bug, attributed per arm below — not a harness defect, and not masked.
//! - BOUNDED, both arches: `indexer_top_k_buf_bounded.gfx942.hip` (symbol
//!   `indexer_top_k_buf_parallel_gfx942_bounded`) on gfx942 and
//!   `indexer_top_k_buf_bounded.gfx1151.hip` on gfx1151 — the bounded
//!   tile-merge bitonic O(N log^2 K) drop-in for the rank-count kernel. The
//!   gfx1151 bounded kernel exports the SAME symbol as the gfx1151 reference
//!   (`indexer_top_k_buf_parallel`) and its header forbids loading both
//!   implementations under that symbol in one process. `Gpu`'s function cache
//!   is keyed by symbol (`launch_kernel_blob` resolves
//!   `functions.get(func_name)`, and `compile_and_load_kernel` early-returns
//!   on a func_name hit), so a distinct module key alone would silently
//!   re-launch the reference kernel. This harness therefore textually renames
//!   the export in its own runtime copy of the source to
//!   `indexer_top_k_buf_parallel_bounded` under module key
//!   `indexer_top_k_buf_bounded_gfx1151_harness` (see
//!   `bounded_gfx1151_spec`); the kernel file itself is untouched, and
//!   production never sees the rename because `attention.rs` selects exactly
//!   one gfx1151 variant per process. Like the gfx1151 reference, the gfx1151
//!   bounded kernel currently lacks the eligibility discipline, so expect it
//!   to FAIL the non-finite cases pending a kernel fix — again attributed
//!   per arm, not masked.
//! - TWOSTAGE, both arches (portable kernel, no arch suffix):
//!   `kernels/src/indexer_topk_twostage.hip`, a parallel merge tree replacing
//!   the single-workgroup serial tile loop. Per iteration it is a SEQUENCE,
//!   not one launch: `indexer_topk_chunk_sort` at grid [n_runs,1,1], then
//!   ceil(log2(n_runs)) `indexer_topk_merge_round` launches at grid
//!   [n_pairs,1,1] with run_step = 1<<r and n_pairs = ceil(n_runs /
//!   (2*run_step)), then `indexer_topk_finalize` at grid [1,1,1]; all block
//!   [256,1,1], dynamic LDS 0. n_runs = ceil(pad_n/512) derives from the case
//!   PAD, never the live n — the host's graph-safety contract (launch count
//!   fixed by max_compressed). Two scratch workspaces of pad_n elements each
//!   (ws_scores F32, ws_indices i32) carry the sorted runs between launches.
//!   TWOSTAGE carries the eligibility discipline by contract.
//!
//! Finite cases must match the host oracle and every other arm that ran;
//! non-finite cases assert per-arm agreement with the eligibility-disciplined
//! reference (SERIAL, which runs on every non-finite case) but are not
//! asserted against the generic total_cmp host oracle. Verdicts are per arm
//! per case (VERDICT lines plus the ARM_TALLY summary), so a known-bad arm
//! cannot mask a genuinely good one; OVERALL stays strict and attributes
//! failures.
//!
//! Large-N cases (8192 / 32768 / 262144 — the last is 1M context at compressor
//! ratio 4, the campaign target) skip SERIAL per-case: its dynamic LDS is a
//! one-byte taken-bitmap per candidate (`smem_fn = max_n` bytes), which exceeds
//! the 64 KiB LDS limit above N=65536 and is wildly over-occupancy well before
//! that. Those cases compare BOUNDED + TWOSTAGE against PARALLEL and the host
//! oracle only, and exist to make the O(N^2) vs O(N log^2 K) gap measurable in
//! seconds here instead of hours on the full 82 GB model campaign.
//!
//! Crossover sweep (N = 1024 / 2048 / 4096, exact-size pads, splitmix64
//! scores): the host selects two-stage only above a max_compressed threshold
//! (default 8192) and that threshold should be measured, not guessed. These
//! cases bracket it so the TWOSTAGE-vs-BOUNDED ratio crosses 1.0x in-view: at
//! small run counts the tree is launch-overhead bound, at large G pass bound
//! — only the per-stage breakdown (chunk-sort / merge-rounds / finalize,
//! printed per case) distinguishes those.
//!
//! Every arm reports warmup + timed-batch wall time (Instant +
//! device_synchronize host timing, the same idiom as the other rdna-compute
//! examples) in ms/launch — for TWOSTAGE that is the wall time of the WHOLE
//! 1+R+1 launch sequence per iteration, which is what the model pays — plus
//! the BOUNDED-vs-reference and TWOSTAGE-vs-BOUNDED speedup ratios.
//!
//! Run on the target GPU (gfx1100, gfx1151, or gfx942):
//!   cargo run --release -p rdna-compute --example test_indexer_top_k_buf
//!
//! Compile-check only (no GPU):
//!   cargo check -p rdna-compute --example test_indexer_top_k_buf

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

/// Absolute harness ceiling: 1M context at compressor ratio 4.
const MAX_N: usize = 262144;
/// Score-buffer pad length for the original small-N cases. Large-N cases use
/// an exact-size buffer (`pad_n == n`); SERIAL cannot run there because its
/// dynamic LDS is one byte per candidate (see `Case::run_serial`).
const SMALL_PAD_N: usize = 2048;
const K: usize = 512;
const N_HEADS: i32 = 1;

const SERIAL_SRC: &str = include_str!("../../../kernels/src/indexer_top_k_buf.hip");
const PARALLEL_GFX1151_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_parallel.gfx1151.hip");
const PARALLEL_GFX942_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_parallel.gfx942.hip");
const PARALLEL_GFX942_BOUNDED_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_bounded.gfx942.hip");
const PARALLEL_GFX1151_BOUNDED_SRC: &str =
    include_str!("../../../kernels/src/indexer_top_k_buf_bounded.gfx1151.hip");
const TWOSTAGE_SRC: &str = include_str!("../../../kernels/src/indexer_topk_twostage.hip");
const SERIAL_MODULE: &str = "indexer_top_k_buf";
const SERIAL_SYMBOL: &str = "indexer_top_k_buf";
const SERIAL_BLOCK: [u32; 3] = [128, 1, 1];
const PARALLEL_BLOCK: [u32; 3] = [256, 1, 1];
/// TWOSTAGE: one portable module (no arch suffix), three symbols launched as
/// a 1+R+1 sequence. Run width MUST match `TOPK_RUN` in the kernel.
const TWOSTAGE_MODULE: &str = "indexer_topk_twostage";
const TWOSTAGE_CHUNK_SORT: &str = "indexer_topk_chunk_sort";
const TWOSTAGE_MERGE_ROUND: &str = "indexer_topk_merge_round";
const TWOSTAGE_FINALIZE: &str = "indexer_topk_finalize";
const TWOSTAGE_LABEL: &str = "TWOSTAGE";
const TOPK_RUN: usize = 512;
const SERIAL_POISON: i32 = -7777;
const PARALLEL_POISON: i32 = -8888;
const BOUNDED_POISON: i32 = -9999;
const TWOSTAGE_POISON: i32 = -6666;

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
    /// Cow because BOUNDED_GFX1151's source is a runtime-renamed copy (see
    /// `bounded_gfx1151_spec`); every other arm is a borrowed include_str!.
    source: std::borrow::Cow<'static, str>,
    symbol: &'static str,
    block: [u32; 3],
    /// Dynamic LDS bytes. SERIAL uses max_n_compressed (taken bitmap);
    /// PARALLEL and BOUNDED use 0 (BOUNDED's 8 KiB LDS is statically declared).
    smem_fn: fn(max_n: i32) -> u32,
}

const SERIAL: KernelSpec = KernelSpec {
    label: "SERIAL",
    module: SERIAL_MODULE,
    source: std::borrow::Cow::Borrowed(SERIAL_SRC),
    symbol: SERIAL_SYMBOL,
    block: SERIAL_BLOCK,
    smem_fn: |max_n| max_n as u32,
};

const PARALLEL_GFX1151: KernelSpec = KernelSpec {
    label: "PARALLEL",
    module: "indexer_top_k_buf_parallel",
    source: std::borrow::Cow::Borrowed(PARALLEL_GFX1151_SRC),
    symbol: "indexer_top_k_buf_parallel",
    block: PARALLEL_BLOCK,
    smem_fn: |_| 0,
};

const PARALLEL_GFX942: KernelSpec = KernelSpec {
    label: "PARALLEL_GFX942",
    module: "indexer_top_k_buf_parallel_gfx942",
    source: std::borrow::Cow::Borrowed(PARALLEL_GFX942_SRC),
    symbol: "indexer_top_k_buf_parallel_gfx942",
    block: PARALLEL_BLOCK,
    smem_fn: |_| 0,
};

const PARALLEL_GFX942_BOUNDED: KernelSpec = KernelSpec {
    label: "BOUNDED_GFX942",
    module: "indexer_top_k_buf_parallel_gfx942_bounded",
    source: std::borrow::Cow::Borrowed(PARALLEL_GFX942_BOUNDED_SRC),
    symbol: "indexer_top_k_buf_parallel_gfx942_bounded",
    block: PARALLEL_BLOCK,
    smem_fn: |_| 0,
};

fn parallel_spec(arch: &str) -> &'static KernelSpec {
    match arch {
        "gfx942" => &PARALLEL_GFX942,
        // Both are wave32 gfx11.  hipcc still emits an exact-device code
        // object; this shares only the source-level parity reference.
        "gfx1100" | "gfx1151" => &PARALLEL_GFX1151,
        _ => unreachable!("architecture checked by main"),
    }
}

/// Bounded comparison arm, now on BOTH arches: the gfx942 twin exports its
/// own symbol; the gfx1151 twin exports `indexer_top_k_buf_parallel`,
/// identical to the gfx1151 reference, and its header forbids loading both
/// implementations under that symbol in one process.
///
/// `compile_and_load_kernel` keys `functions` by func_name and early-returns
/// on a hit, and `launch_kernel_blob` resolves by the same key — so a
/// distinct module name alone is NOT enough: the second ensure would no-op
/// and the "bounded" arm would silently re-launch the reference kernel. The
/// harness therefore renames the export inside its own runtime copy of the
/// source (see `bounded_gfx1151_spec`); the kernel file itself is untouched.
fn bounded_spec(arch: &str) -> Option<KernelSpec> {
    match arch {
        "gfx942" => Some(PARALLEL_GFX942_BOUNDED),
        "gfx1100" | "gfx1151" => Some(bounded_gfx1151_spec()),
        _ => None,
    }
}

/// gfx1151 bounded arm with the export renamed so the reference and bounded
/// implementations coexist in one process under distinct `functions` keys.
/// The rename is a whole-identifier textual substitution in THIS EXAMPLE's
/// copy of the source — the file has exactly two occurrences of the symbol
/// (the header comment and the `extern "C"` definition), both renamed
/// consistently. Production is unaffected: `attention.rs` selects exactly one
/// gfx1151 variant per process, precisely so it never hits this collision.
fn bounded_gfx1151_spec() -> KernelSpec {
    const REF_SYMBOL: &str = "indexer_top_k_buf_parallel";
    const RENAMED_SYMBOL: &str = "indexer_top_k_buf_parallel_bounded";
    assert!(
        PARALLEL_GFX1151_BOUNDED_SRC.contains("void indexer_top_k_buf_parallel("),
        "gfx1151 bounded kernel no longer exports {REF_SYMBOL}; re-audit the harness rename"
    );
    let renamed = PARALLEL_GFX1151_BOUNDED_SRC.replace(REF_SYMBOL, RENAMED_SYMBOL);
    KernelSpec {
        label: "BOUNDED_GFX1151",
        module: "indexer_top_k_buf_bounded_gfx1151_harness",
        source: std::borrow::Cow::Owned(renamed),
        symbol: RENAMED_SYMBOL,
        block: PARALLEL_BLOCK,
        smem_fn: |_| 0,
    }
}

fn ensure_kernels(gpu: &mut Gpu, parallel: &KernelSpec, bounded: Option<&KernelSpec>) {
    gpu.ensure_kernel_public(SERIAL.module, &SERIAL.source, SERIAL.symbol)
        .expect("compile SERIAL indexer_top_k_buf");
    gpu.ensure_kernel_public(parallel.module, &parallel.source, parallel.symbol)
        .expect("compile PARALLEL indexer_top_k_buf_parallel");
    if let Some(bounded) = bounded {
        gpu.ensure_kernel_public(bounded.module, &bounded.source, bounded.symbol)
            .expect("compile BOUNDED indexer top-K");
    }
    // TWOSTAGE: three symbols from one portable module; the first ensure
    // compiles the module, the rest hit the compiler/module caches.
    for symbol in [TWOSTAGE_CHUNK_SORT, TWOSTAGE_MERGE_ROUND, TWOSTAGE_FINALIZE] {
        gpu.ensure_kernel_public(TWOSTAGE_MODULE, TWOSTAGE_SRC, symbol)
            .expect("compile TWOSTAGE indexer_topk_twostage");
    }
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
    log: bool,
) {
    let grid = [N_HEADS as u32, 1, 1];
    let smem = (spec.smem_fn)(max_n);
    if log {
        eprintln!(
            "  LAUNCH {} symbol={} grid={:?} block={:?} dynamic_lds={}",
            spec.label, spec.symbol, grid, spec.block, smem
        );
    }
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

// ─── arms: single-launch kernels and the TWOSTAGE sequence ──────────────────

/// One comparison arm. `Single` is a one-launch kernel described by a
/// KernelSpec; `TwoStage` is the portable 1+R+1 launch merge-tree sequence
/// with its own pad_n-element workspaces (see `launch_twostage`).
#[derive(Clone, Copy)]
enum Arm<'a> {
    Single(&'a KernelSpec),
    TwoStage,
}

impl Arm<'_> {
    fn label(self) -> &'static str {
        match self {
            Arm::Single(spec) => spec.label,
            Arm::TwoStage => TWOSTAGE_LABEL,
        }
    }
}

/// Per-case device state shared by every arm launch. `ws_scores`/`ws_indices`
/// are the TWOSTAGE workspaces: one sorted TOPK_RUN-run per chunk, pad_n
/// elements each, zero-filled for determinism (the kernels overwrite what
/// they use).
struct LaunchCtx<'a> {
    scores: &'a GpuTensor,
    n_buf: &'a GpuTensor,
    k_buf: &'a GpuTensor,
    ws_scores: &'a GpuTensor,
    ws_indices: &'a GpuTensor,
    pad_n: usize,
    max_k: i32,
}

fn launch_arm(gpu: &Gpu, arm: Arm, ctx: &LaunchCtx, out: &GpuTensor, log: bool) {
    match arm {
        Arm::Single(spec) => launch_one(
            gpu,
            spec,
            ctx.scores,
            out,
            ctx.n_buf,
            ctx.k_buf,
            ctx.pad_n as i32,
            ctx.max_k,
            log,
        ),
        Arm::TwoStage => launch_twostage(gpu, ctx, out, log),
    }
}

/// Grid/launch-count plan for the two-stage tree: (n_runs, merge rounds).
/// n_runs derives from the case PAD (the graph-fixed max_compressed), never
/// the live n; rounds is ceil(log2(n_runs)); total launches per iteration is
/// 1 + rounds + 1.
fn twostage_plan(pad_n: usize) -> (u32, u32) {
    let n_runs = pad_n.div_ceil(TOPK_RUN) as u32;
    let mut rounds = 0u32;
    while (1u64 << rounds) < n_runs as u64 {
        rounds += 1;
    }
    (n_runs, rounds)
}

fn twostage_chunk_sort(gpu: &Gpu, ctx: &LaunchCtx, n_runs: u32, log: bool) {
    if log {
        eprintln!(
            "  LAUNCH TWOSTAGE symbol={} grid=[{n_runs},1,1] block={:?} dynamic_lds=0",
            TWOSTAGE_CHUNK_SORT, PARALLEL_BLOCK
        );
    }
    let mut kb = KernargBlob::new();
    kb.push_ptr(ctx.scores.buf.as_ptr() as *const c_void);
    kb.push_ptr(ctx.ws_scores.buf.as_ptr() as *const c_void);
    kb.push_ptr(ctx.ws_indices.buf.as_ptr() as *const c_void);
    kb.push_ptr(ctx.n_buf.buf.as_ptr() as *const c_void);
    kb.push_i32(n_runs as i32);
    kb.pad_to(16);
    gpu.launch_kernel_blob(
        TWOSTAGE_CHUNK_SORT,
        [n_runs, 1, 1],
        PARALLEL_BLOCK,
        0,
        kb.as_mut_slice(),
    )
    .expect("launch TWOSTAGE chunk_sort failed");
}

fn twostage_merge_round(
    gpu: &Gpu,
    ctx: &LaunchCtx,
    n_runs: u32,
    run_step: u32,
    n_pairs: u32,
    log: bool,
) {
    if log {
        eprintln!(
            "  LAUNCH TWOSTAGE symbol={} grid=[{n_pairs},1,1] block={:?} dynamic_lds=0 run_step={run_step}",
            TWOSTAGE_MERGE_ROUND, PARALLEL_BLOCK
        );
    }
    let mut kb = KernargBlob::new();
    kb.push_ptr(ctx.ws_scores.buf.as_ptr() as *const c_void);
    kb.push_ptr(ctx.ws_indices.buf.as_ptr() as *const c_void);
    kb.push_i32(n_runs as i32);
    kb.push_i32(run_step as i32);
    kb.push_i32(n_pairs as i32);
    kb.pad_to(16);
    gpu.launch_kernel_blob(
        TWOSTAGE_MERGE_ROUND,
        [n_pairs, 1, 1],
        PARALLEL_BLOCK,
        0,
        kb.as_mut_slice(),
    )
    .expect("launch TWOSTAGE merge_round failed");
}

fn twostage_finalize(gpu: &Gpu, ctx: &LaunchCtx, out: &GpuTensor, log: bool) {
    if log {
        eprintln!(
            "  LAUNCH TWOSTAGE symbol={} grid=[1,1,1] block={:?} dynamic_lds=0",
            TWOSTAGE_FINALIZE, PARALLEL_BLOCK
        );
    }
    let mut kb = KernargBlob::new();
    kb.push_ptr(ctx.ws_scores.buf.as_ptr() as *const c_void);
    kb.push_ptr(ctx.ws_indices.buf.as_ptr() as *const c_void);
    kb.push_ptr(out.buf.as_ptr() as *const c_void);
    kb.push_ptr(ctx.n_buf.buf.as_ptr() as *const c_void);
    kb.push_ptr(ctx.k_buf.buf.as_ptr() as *const c_void);
    kb.push_i32(ctx.max_k);
    kb.pad_to(16);
    gpu.launch_kernel_blob(
        TWOSTAGE_FINALIZE,
        [1, 1, 1],
        PARALLEL_BLOCK,
        0,
        kb.as_mut_slice(),
    )
    .expect("launch TWOSTAGE finalize failed");
}

/// The full two-stage sequence: chunk-sort, ceil(log2(n_runs)) merge rounds
/// with run_step = 1<<r and n_pairs = ceil(n_runs / (2*run_step)), finalize.
/// Each timed iteration pays all 1+R+1 launches — that is what the model pays.
fn launch_twostage(gpu: &Gpu, ctx: &LaunchCtx, out: &GpuTensor, log: bool) {
    let (n_runs, _) = twostage_plan(ctx.pad_n);
    twostage_chunk_sort(gpu, ctx, n_runs, log);
    let mut run_step = 1u32;
    while run_step < n_runs {
        let n_pairs = n_runs.div_ceil(2 * run_step);
        twostage_merge_round(gpu, ctx, n_runs, run_step, n_pairs, log);
        run_step *= 2;
    }
    twostage_finalize(gpu, ctx, out, log);
}

/// One instrumented sequence with a device_synchronize between stages,
/// returning (chunk_sort, merge_rounds_total, finalize) ms. The sync
/// boundaries serialise launches the pipelined sequence would overlap, so
/// these numbers ATTRIBUTE cost (launch-overhead bound at small G vs pass
/// bound at large G) rather than price the sequence — ms_per_launch is the
/// price.
fn launch_twostage_timed(gpu: &Gpu, ctx: &LaunchCtx, out: &GpuTensor) -> (f64, f64, f64) {
    let (n_runs, _) = twostage_plan(ctx.pad_n);
    let t = std::time::Instant::now();
    twostage_chunk_sort(gpu, ctx, n_runs, false);
    gpu.hip.device_synchronize().expect("sync after chunk_sort");
    let chunk_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = std::time::Instant::now();
    let mut run_step = 1u32;
    while run_step < n_runs {
        let n_pairs = n_runs.div_ceil(2 * run_step);
        twostage_merge_round(gpu, ctx, n_runs, run_step, n_pairs, false);
        run_step *= 2;
    }
    gpu.hip
        .device_synchronize()
        .expect("sync after merge rounds");
    let merge_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = std::time::Instant::now();
    twostage_finalize(gpu, ctx, out, false);
    gpu.hip.device_synchronize().expect("sync after finalize");
    (chunk_ms, merge_ms, t.elapsed().as_secs_f64() * 1e3)
}

// ─── score builders ──────────────────────────────────────────────────────────

/// Exact-tie finite pattern from the divergence analysis.
fn finite_tie_scores(pad_n: usize) -> Vec<f32> {
    (0..pad_n).map(|i| ((37 * i + 11) % 64) as f32).collect()
}

/// Deterministic splitmix64-derived finite scores for the large-N cases.
/// Reproducible run-to-run (the per-case scores_sha256 makes a run
/// self-describing); f32 mantissa granularity still forces exact ties at
/// N=262144, which stresses the index-ASC tiebreak against the host oracle.
fn finite_large_scores(pad_n: usize) -> Vec<f32> {
    (0..pad_n)
        .map(|i| {
            let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
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
    /// Score-buffer length (pad). `n <= pad_n <= MAX_N`; kernels get max_n=pad_n.
    pad_n: usize,
    /// Explicit per-case SERIAL skip. SERIAL's dynamic LDS is `max_n` bytes —
    /// a one-byte taken-bitmap per candidate — which exceeds the 64 KiB LDS
    /// limit above N=65536 and is wildly over-occupancy well before that, so
    /// large-N cases compare BOUNDED/PARALLEL against the host oracle only.
    run_serial: bool,
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
            pad_n: SMALL_PAD_N,
            run_serial: true,
            scores: finite_tie_scores(SMALL_PAD_N),
        });
    }

    // 200 finite + 313 -inf at N=513 → fewer than 512 selectable under SERIAL.
    {
        let mut scores = finite_tie_scores(SMALL_PAD_N);
        // Keep first 200 finite; fill remaining of the N window with -inf.
        for i in 200..513 {
            scores[i] = f32::NEG_INFINITY;
        }
        // Pad beyond N is unused by kernels but keep deterministic.
        for i in 513..SMALL_PAD_N {
            scores[i] = f32::NEG_INFINITY;
        }
        cases.push(Case {
            name: "nonfinite_n513_neginf_pool",
            n: 513,
            kind: CaseKind::NegInf,
            pad_n: SMALL_PAD_N,
            run_serial: true,
            scores,
        });
    }

    // A few NaNs at N=513 — PARALLEL rank-0 collisions expected.
    {
        let mut scores = finite_tie_scores(SMALL_PAD_N);
        scores[0] = f32::from_bits(0x7fc0_0001); // quiet NaN payload 1
        scores[257] = f32::from_bits(0x7fc0_0002); // quiet NaN payload 2
        scores[400] = f32::NAN;
        cases.push(Case {
            name: "nonfinite_n513_nan",
            n: 513,
            kind: CaseKind::Nan,
            pad_n: SMALL_PAD_N,
            run_serial: true,
            scores,
        });
    }

    // Several exact -FLT_MAX entries (SERIAL best=-FLT_MAX cannot select via strict >).
    {
        let mut scores = finite_tie_scores(SMALL_PAD_N);
        for &i in &[10usize, 100, 250, 400, 512] {
            scores[i] = -f32::MAX; // == -FLT_MAX
        }
        cases.push(Case {
            name: "nonfinite_n513_neg_flt_max",
            n: 513,
            kind: CaseKind::NegFltMax,
            pad_n: SMALL_PAD_N,
            run_serial: true,
            scores,
        });
    }

    // Crossover sweep: the host selects two-stage only above a max_compressed
    // threshold (HIPFIRE_DEEPSEEK4_INDEXER_TOPK_TWOSTAGE_MIN, default 8192)
    // and that threshold should be measured, not guessed. These cases bracket
    // it so the TWOSTAGE-vs-BOUNDED ratio crosses 1.0x in-view: at n_runs
    // 2/4/8 the tree's extra launches may not pay for themselves against one
    // bounded launch. pad_n == n mirrors the production graph contract
    // (n_runs from max_compressed, never the live n); scores use the same
    // splitmix64 distribution as the large-N cases so the sweep is uniform.
    for &n in &[1024usize, 2048, 4096] {
        cases.push(Case {
            name: match n {
                1024 => "finite_n1024_sweep",
                2048 => "finite_n2048_sweep",
                4096 => "finite_n4096_sweep",
                _ => unreachable!(),
            },
            n,
            kind: CaseKind::Finite,
            pad_n: n,
            run_serial: true,
            scores: finite_large_scores(n),
        });
    }

    // Large-N finite cases — the bounded kernel's reason to exist. SERIAL is
    // skipped per-case (see Case::run_serial); BOUNDED_GFX942 is compared
    // against PARALLEL_GFX942 and the host oracle, slot-for-slot. N=262144 is
    // 1M context at compressor ratio 4, the campaign target; at N=2048 only 4
    // tiles of the kpad=512 tile loop are exercised, which cannot see the
    // O(N^2) vs O(N log^2 K) divergence these sizes expose.
    for &n in &[8192usize, 32768, 262144] {
        cases.push(Case {
            name: match n {
                8192 => "finite_n8192",
                32768 => "finite_n32768",
                262144 => "finite_n262144_ctx1m_ratio4",
                _ => unreachable!(),
            },
            n,
            kind: CaseKind::Finite,
            pad_n: n,
            run_serial: false,
            scores: finite_large_scores(n),
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

struct ArmSummary {
    label: &'static str,
    vs_ref: &'static str,
    vs_oracle: &'static str,
    /// Per-arm per-case verdict: finite = byte-identical to the host oracle;
    /// non-finite = byte-identical to the eligibility reference (SERIAL).
    /// Either way requires zero unwritten slots and zero duplicates.
    verdict: &'static str,
    unwritten: usize,
    duplicates: usize,
    ms_per_launch: f64,
}

struct CaseResult {
    name: String,
    n: usize,
    finite: bool,
    arms_ran: String,
    first_diff_rank: String,
    same_set: bool,
    arms: Vec<ArmSummary>,
    speedup_bounded_vs_parallel: Option<f64>,
    speedup_twostage_vs_bounded: Option<f64>,
    /// (chunk_sort, merge_rounds_total, finalize) ms from the instrumented
    /// pass; None only if TWOSTAGE somehow did not run.
    twostage_stage_ms: Option<(f64, f64, f64)>,
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

/// Warmup / timed-batch iteration counts per case. Large N collapses the
/// batch on purpose: the O(N^2) rank-count arm costs seconds per launch at
/// N=32768 and minutes at N=262144, so one timed launch already separates it
/// from the O(N log^2 K) bounded arm.
fn timing_plan(n: usize) -> (usize, usize) {
    if n <= 2048 {
        (3, 10)
    } else if n <= 8192 {
        (2, 5)
    } else if n <= 32768 {
        (1, 3)
    } else {
        (1, 1)
    }
}

fn run_case(
    gpu: &mut Gpu,
    parallel: &KernelSpec,
    bounded: Option<&KernelSpec>,
    case: &Case,
) -> CaseResult {
    let n = case.n;
    let pad_n = case.pad_n;
    assert!(n <= pad_n && pad_n <= MAX_N);
    let finite = matches!(case.kind, CaseKind::Finite);

    // Arms for this case: SERIAL (unless skipped per-case), the arch
    // PARALLEL reference, BOUNDED (both arches), and TWOSTAGE (portable;
    // always runs, on every case size including the identity path).
    let mut arms: Vec<(Arm, i32)> = Vec::new();
    if case.run_serial {
        arms.push((Arm::Single(&SERIAL), SERIAL_POISON));
    }
    arms.push((Arm::Single(parallel), PARALLEL_POISON));
    if let Some(b) = bounded {
        arms.push((Arm::Single(b), BOUNDED_POISON));
    }
    arms.push((Arm::TwoStage, TWOSTAGE_POISON));
    let arms_ran = arms
        .iter()
        .map(|(arm, _)| arm.label())
        .collect::<Vec<_>>()
        .join("+");

    eprintln!(
        "\n======== CASE {} N={} finite={} arms={} ========",
        case.name, n, finite, arms_ran
    );
    score_census(&case.scores, n);
    let scores_sha = sha256_f32(&case.scores);
    eprintln!("  scores_sha256={scores_sha} (full pad_n={pad_n} buffer)");

    // Build score buffer ONCE, upload ONCE.
    let scores_gpu = gpu
        .upload_f32(&case.scores, &[pad_n])
        .expect("upload scores");
    let n_gpu = upload_i32(gpu, &[n as i32]);
    let k_gpu = upload_i32(gpu, &[K as i32]);
    // TWOSTAGE workspaces: one sorted TOPK_RUN-run per chunk, pad_n elements
    // each, zero-filled for determinism (the kernels overwrite what they use).
    let ws_scores = gpu
        .upload_f32(&vec![0.0f32; pad_n], &[pad_n])
        .expect("upload ws_scores");
    let ws_indices = upload_i32(gpu, &vec![0i32; pad_n]);
    let ctx = LaunchCtx {
        scores: &scores_gpu,
        n_buf: &n_gpu,
        k_buf: &k_gpu,
        ws_scores: &ws_scores,
        ws_indices: &ws_indices,
        pad_n,
        max_k: K as i32,
    };

    // One independently poisoned output buffer per arm; launch every arm from
    // the same frozen input.
    let mut out_bufs = Vec::with_capacity(arms.len());
    for (arm, poison) in &arms {
        let poison_host = vec![*poison; K];
        let out = upload_i32(gpu, &poison_host);
        launch_arm(gpu, *arm, &ctx, &out, true);
        out_bufs.push(out);
    }
    gpu.hip
        .device_synchronize()
        .expect("synchronize after arm launches");

    let mut outs = Vec::with_capacity(arms.len());
    for out in &out_bufs {
        outs.push(download_i32(gpu, out, K));
    }
    let oracle = expected_top_k(&case.scores, n);

    // Per-arm timing: warmup iterations then a timed batch, Instant +
    // device_synchronize host wall time (the same idiom as the other
    // rdna-compute examples). Reuses each arm's own output buffer; the
    // exactness results above are already downloaded.
    let (warmup, iters) = timing_plan(n);
    let mut ms_per_launch = Vec::with_capacity(arms.len());
    for (i, (arm, _)) in arms.iter().enumerate() {
        for _ in 0..warmup {
            launch_arm(gpu, *arm, &ctx, &out_bufs[i], false);
        }
        gpu.hip.device_synchronize().expect("sync after warmup");
        let t = std::time::Instant::now();
        for _ in 0..iters {
            launch_arm(gpu, *arm, &ctx, &out_bufs[i], false);
        }
        gpu.hip
            .device_synchronize()
            .expect("sync after timed batch");
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        ms_per_launch.push(ms);
        eprintln!(
            "  timing arm={} warmup={} iters={} ms_per_launch={:.3}",
            arm.label(),
            warmup,
            iters,
            ms
        );
    }

    // TWOSTAGE attribution: report the merge-tree shape (launch count derives
    // ONLY from pad_n — the graph-safety contract), then an instrumented pass
    // timing each stage with a device_synchronize boundary. The syncs
    // serialise launches the pipelined sequence would overlap, so per-stage
    // numbers attribute cost (launch-overhead bound at small G vs pass bound
    // at large G) rather than price the sequence; ms_per_launch is the price.
    let (n_runs, rounds) = twostage_plan(pad_n);
    eprintln!(
        "  twostage_plan pad_n={pad_n} n_runs={n_runs} merge_rounds={rounds} launches_per_iter={}",
        2 + rounds
    );
    let twostage_stage_ms = arms
        .iter()
        .position(|(arm, _)| matches!(arm, Arm::TwoStage))
        .map(|i| {
            for _ in 0..warmup {
                launch_twostage(gpu, &ctx, &out_bufs[i], false);
            }
            gpu.hip
                .device_synchronize()
                .expect("sync after twostage stage warmup");
            let mut acc = (0.0f64, 0.0f64, 0.0f64);
            for _ in 0..iters {
                let (c, m, f) = launch_twostage_timed(gpu, &ctx, &out_bufs[i]);
                acc.0 += c;
                acc.1 += m;
                acc.2 += f;
            }
            let d = iters as f64;
            let stage_ms = (acc.0 / d, acc.1 / d, acc.2 / d);
            eprintln!(
                "  twostage_stage_ms chunk_sort={:.3} merge_rounds={:.3} finalize={:.3}",
                stage_ms.0, stage_ms.1, stage_ms.2
            );
            stage_ms
        });

    let stats: Vec<SlotStats> = arms
        .iter()
        .zip(outs.iter())
        .map(|((_, poison), out)| slot_stats(out, n, *poison))
        .collect();

    // Reference arm = first arm (SERIAL when it ran, else PARALLEL). Ordered
    // equality is the bar: the gfx942 kernels each place a selected index at
    // its own rank slot, so their outputs must agree slot-for-slot.
    let reference = &outs[0];
    let ordered_eq = outs.iter().all(|out| out == reference);
    let set_eq = outs.iter().all(|out| same_set(reference, out));

    let mut first_diff_rank = "-".to_string();
    for (i, out) in outs.iter().enumerate().skip(1) {
        if let Some((r, a, b)) = first_diff(reference, out) {
            eprintln!(
                "  first_diff_rank={r} {}={a} {}={b}",
                arms[0].0.label(),
                arms[i].0.label()
            );
            first_diff_rank = format!("{r}");
            // Print a short window around the first mismatch for operator grepping.
            let lo = r.saturating_sub(2);
            let hi = (r + 3).min(K);
            eprintln!("  window ranks[{lo}..{hi}):");
            for rank in lo..hi {
                eprintln!(
                    "    rank={rank} {}={} {}={} oracle={}",
                    arms[0].0.label(),
                    reference[rank],
                    arms[i].0.label(),
                    out[rank],
                    oracle[rank]
                );
            }
            break;
        }
    }
    if first_diff_rank == "-" {
        eprintln!("  first_diff_rank=- (ordered arrays identical)");
    }

    let n_diffs = outs
        .iter()
        .skip(1)
        .map(|out| {
            reference
                .iter()
                .zip(out.iter())
                .filter(|(a, b)| a != b)
                .count()
        })
        .collect::<Vec<_>>();
    eprintln!(
        "  arms_ordered_eq={} n_diffs_vs_ref={:?} same_set={set_eq}",
        if ordered_eq { "MATCH" } else { "DIFFER" },
        n_diffs
    );

    // Per-arm verdicts, so a known-bad arm (today: both gfx1151 incumbent
    // kernels on the non-finite cases — they lack the eligibility filter) is
    // attributed by name and cannot mask an arm that genuinely passes.
    // - Finite: PASS iff byte-identical to the host oracle, no poison, no
    //   dups. (Every arm matching the oracle implies mutual ordered
    //   equality, so this subsumes the old arms-equal check.)
    // - Non-finite: PASS iff byte-identical to the eligibility-disciplined
    //   reference (outs[0] == SERIAL, which runs on every non-finite case),
    //   no poison, no dups. The host oracle stays a generic total_cmp sort
    //   and may DIFFER; reported only.
    let mut arm_summaries = Vec::with_capacity(arms.len());
    for (i, ((arm, _), out)) in arms.iter().zip(outs.iter()).enumerate() {
        let st = stats[i];
        let label = arm.label().to_lowercase();
        let vs_ref = if i == 0 {
            "REF"
        } else if out == reference {
            "MATCH"
        } else {
            "DIFFER"
        };
        let vs_oracle = if *out == oracle { "MATCH" } else { "DIFFER" };
        eprintln!(
            "  {label}_slots: poison={} pad_neg1={} real={} oor={} dups={}",
            st.poison, st.pad_neg1, st.real, st.out_of_range, st.duplicates
        );
        eprintln!(
            "  {label}_ordered_sha256={} {label}_set_sha256={}",
            sha256_i32(out),
            ascending_set_hash(out)
        );
        eprintln!("  {label}_vs_ref={vs_ref} {label}_vs_oracle={vs_oracle}");
        let verdict_ok = st.poison == 0
            && st.duplicates == 0
            && if finite {
                *out == oracle
            } else {
                out == reference
            };
        arm_summaries.push(ArmSummary {
            label: arm.label(),
            vs_ref,
            vs_oracle,
            verdict: if verdict_ok { "PASS" } else { "FAIL" },
            unwritten: st.poison,
            duplicates: st.duplicates,
            ms_per_launch: ms_per_launch[i],
        });
    }
    eprintln!("  oracle_ordered_sha256={}", sha256_i32(&oracle));

    // Characterise non-finite pads specifically.
    if !finite {
        for (i, (arm, _)) in arms.iter().enumerate() {
            let st = stats[i];
            eprintln!(
                "  nonfinite_detail: {}(-1={}, real={}, poison={})",
                arm.label(),
                st.pad_neg1,
                st.real,
                st.poison
            );
        }
    }

    let mut accept_fail = false;
    for a in &arm_summaries {
        eprintln!("  VERDICT case={} arm={} {}", case.name, a.label, a.verdict);
        if a.verdict == "FAIL" {
            accept_fail = true;
            if finite {
                eprintln!(
                    "  ACCEPT_FAIL detail: {} (vs_oracle={} unwritten={} dups={})",
                    a.label, a.vs_oracle, a.unwritten, a.duplicates
                );
            } else {
                eprintln!(
                    "  ACCEPT_FAIL detail: {} disagrees with the eligibility reference or left \
                     slots unwritten (vs_ref={} unwritten={} dups={})",
                    a.label, a.vs_ref, a.unwritten, a.duplicates
                );
            }
        }
    }
    if !accept_fail {
        if finite {
            eprintln!("  ACCEPT: PASS (finite case byte-identical to oracle for every arm)");
        } else {
            eprintln!(
                "  ACCEPT: PASS (all arms agree with the eligibility reference; oracle reported only)"
            );
        }
    }

    // The point of the large-N and sweep cases: the O(N log^2 K) tree vs the
    // bounded single-workgroup loop vs the O(N^2) rank-count, in ms/launch.
    let pos = |label: &str| arms.iter().position(|(a, _)| a.label() == label);
    let ratio = |num: Option<usize>, den: Option<usize>| match (num, den) {
        (Some(num), Some(den)) if ms_per_launch[den] > 0.0 => {
            Some(ms_per_launch[num] / ms_per_launch[den])
        }
        _ => None,
    };
    let speedup_bounded_vs_parallel = bounded.and_then(|b| {
        let r = ratio(pos(parallel.label), pos(b.label));
        if let Some(r) = r {
            eprintln!(
                "  speedup {} vs {}: {r:.2}x ({:.3} ms -> {:.3} ms)",
                b.label,
                parallel.label,
                ms_per_launch[pos(parallel.label).unwrap()],
                ms_per_launch[pos(b.label).unwrap()]
            );
        }
        r
    });
    let speedup_twostage_vs_bounded = bounded.and_then(|b| {
        let r = ratio(pos(b.label), pos(TWOSTAGE_LABEL));
        if let Some(r) = r {
            eprintln!(
                "  speedup TWOSTAGE vs {}: {r:.2}x ({:.3} ms -> {:.3} ms)",
                b.label,
                ms_per_launch[pos(b.label).unwrap()],
                ms_per_launch[pos(TWOSTAGE_LABEL).unwrap()]
            );
        }
        r
    });

    CaseResult {
        name: case.name.to_string(),
        n,
        finite,
        arms_ran,
        first_diff_rank,
        same_set: set_eq,
        arms: arm_summaries,
        speedup_bounded_vs_parallel,
        speedup_twostage_vs_bounded,
        twostage_stage_ms,
        accept_fail,
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let arch = gpu.arch.as_str();
    eprintln!("detected_arch={arch}");
    if arch != "gfx1100" && arch != "gfx1151" && arch != "gfx942" {
        panic!(
            "unsupported arch '{arch}': this parity probe accepts only gfx1100, gfx1151, or gfx942 \
             (refuse, do not skip)"
        );
    }
    eprintln!("arch_ok=true (accepted gfx1100|gfx1151|gfx942)");

    let parallel = parallel_spec(arch);
    let bounded = bounded_spec(arch);
    ensure_kernels(&mut gpu, parallel, bounded.as_ref());
    eprintln!(
        "kernels_loaded: SERIAL symbol={} block={:?} smem=max_n_compressed; \
         PARALLEL symbol={} block={:?} smem=0; BOUNDED {}; \
         TWOSTAGE module={} symbols={},{},{} block={:?} smem=0",
        SERIAL.symbol,
        SERIAL.block,
        parallel.symbol,
        parallel.block,
        match &bounded {
            Some(b) => format!(
                "label={} module={} symbol={} block={:?} smem=0 (static 8 KiB LDS)",
                b.label, b.module, b.symbol, b.block
            ),
            None => "n/a (not compiled on this arch)".to_string(),
        },
        TWOSTAGE_MODULE,
        TWOSTAGE_CHUNK_SORT,
        TWOSTAGE_MERGE_ROUND,
        TWOSTAGE_FINALIZE,
        PARALLEL_BLOCK,
    );

    let cases = build_cases();
    let mut results = Vec::with_capacity(cases.len());
    let mut any_fail = false;
    for case in &cases {
        let r = run_case(&mut gpu, parallel, bounded.as_ref(), case);
        if r.accept_fail {
            any_fail = true;
        }
        results.push(r);
    }

    eprintln!("\n======== SUMMARY (machine-greppable) ========");
    for r in &results {
        let arms_detail = r
            .arms
            .iter()
            .map(|a| {
                format!(
                    "{}(verdict={} vs_ref={} vs_oracle={} unwritten={} dups={} ms_per_launch={:.3})",
                    a.label, a.verdict, a.vs_ref, a.vs_oracle, a.unwritten, a.duplicates, a.ms_per_launch
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let speedup = r
            .speedup_bounded_vs_parallel
            .map(|s| format!("{s:.2}x"))
            .unwrap_or_else(|| "n/a".to_string());
        let speedup_tvb = r
            .speedup_twostage_vs_bounded
            .map(|s| format!("{s:.2}x"))
            .unwrap_or_else(|| "n/a".to_string());
        let stages = r
            .twostage_stage_ms
            .map(|(c, m, f)| format!("chunk={c:.3},merge={m:.3},finalize={f:.3}"))
            .unwrap_or_else(|| "n/a".to_string());
        eprintln!(
            "RESULT case={} N={} finite={} arms={} first_diff_rank={} same_set={} \
             speedup_bounded_vs_parallel={} speedup_twostage_vs_bounded={} twostage_stage_ms={} {}",
            r.name,
            r.n,
            r.finite,
            r.arms_ran,
            r.first_diff_rank,
            r.same_set,
            speedup,
            speedup_tvb,
            stages,
            arms_detail
        );
    }

    // Per-arm tally across cases: a known-bad arm (today both gfx1151
    // incumbent kernels on the non-finite eligibility cases) is attributed
    // here by name without masking arms that pass everything.
    let mut tally: Vec<(&'static str, usize, usize)> = Vec::new();
    for r in &results {
        for a in &r.arms {
            match tally.iter().position(|(label, _, _)| *label == a.label) {
                Some(idx) => {
                    tally[idx].2 += 1;
                    if a.verdict == "PASS" {
                        tally[idx].1 += 1;
                    }
                }
                None => tally.push((a.label, usize::from(a.verdict == "PASS"), 1)),
            }
        }
    }
    for (label, pass, total) in &tally {
        eprintln!("ARM_TALLY arm={label} verdicts_pass={pass}/{total}");
    }

    if any_fail {
        eprintln!(
            "OVERALL: FAIL (one or more arms failed a case — see VERDICT/ARM_TALLY for \
             attribution; on gfx1151 the incumbent PARALLEL and BOUNDED kernels are KNOWN to \
             fail the non-finite eligibility cases pending a kernel fix)"
        );
        std::process::exit(1);
    }
    eprintln!("OVERALL: PASS (every arm passed every case; finite cases matched the oracle)");
}
